import logging
import statistics
from operator import add
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from striatum.bandit.bandit import BaseBandit
import torchbnn as bnn


LOGGER = logging.getLogger(__name__)


# def truncated_normal_(tensor_p, tensor_n, mean=0, std=1): # 從截斷的正態分佈中輸出隨機值
#     size =  tensor_p.shape # (1,50)
#     tmp = tensor_p.new_empty(size+(4,)).normal_() # 常態分佈中輸出隨機值
#     valid = (tmp < torch.max(tensor_p)) & (tmp > torch.min(tensor_n)) # 截斷有效範圍
#     ind = valid.max(-1, keepdim=True)[1] # 有效範圍的index
#     tensor_p.data.copy_(tmp.gather(-1, ind).squeeze(-1)) # gather有效的index再壓扁
#     tensor_p.data.mul_(std).add_(mean)
#     return tensor_p


# class VAE(nn.Module):
#     ''' This the encoder part of VAE
#     Modify from https://github.com/pytorch/examples/blob/master/vae/main.py
#     '''

#     def __init__(self, in_dim, hid_dim, z_dim):
#         '''
#         Args:
#             input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
#             hidden_dim: A integer indicating the size of hidden dimension.
#             z_dim: A integer indicating the latent dimension.
#         '''
#         super().__init__()
#         self.in_dim = in_dim
#         self.fc1 = nn.Linear(in_dim, hid_dim)
#         self.fc21 = nn.Linear(hid_dim, z_dim)
#         self.fc22 = nn.Linear(hid_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, hid_dim)
#         self.fc4 = nn.Linear(hid_dim, in_dim)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def cust_reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         #eps = torch.randn_like(std)
#         #print('mu:',mu)
#         #print('z_p:',mu + std)
#         #print('z_n:',mu - std)
#         return mu + 2*std,mu - 2*std # 固定上下限 2

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.in_dim))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar, z

#     def cust_forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.in_dim))
#         z_p,z_n = self.cust_reparameterize(mu, logvar)  # VAE UCB score的上界和下界
#         return self.decode(z_p),self.decode(z_n), mu, logvar, z_p,z_n  # 我們要的是中間的latten z不是decode的

#     def loss_function(self, recon_x, x, mu, logvar): # 要最小化
#         BCE = F.binary_cross_entropy(
#             recon_x, x.view(-1, self.in_dim), reduction='sum') # 回傳的是BCE的loss (負的值去做最小化)
#         # see Appendix B from VAE paper:  # 原本是 -D_kl + BCE
#         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         # https://arxiv.org/abs/1312.6114
#         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # 因此也要乘上負號
#         return BCE + KLD # min(BCE_loss + D_kl)


class EEUNN(BaseBandit):
    def __init__(self, seed=2022, history_storage, model_storage, action_storage, paramaters, recommendation_cls=None, optimizer="Adam"):
        super(EEUNN, self).__init__(history_storage, model_storage,
                                      action_storage, recommendation_cls)
        torch.manual_seed(seed) 
        self.actionCount = action_storage.count()
        self.action_storage = action_storage
        # PoissonNLLLosstorch.nn.KLDivLoss(reduction = 'batchmean')#size_average=False reduction='sum'
        self.loss_fn = torch.nn.MSELoss() # torch.nn.CrossEntropyLoss()
        self.learning_rate = paramaters['learning_rate']
        self.batch_size = paramaters['batch_size']
        self.gamma = paramaters['gamma']
#         self.vae_h_dim = paramaters['vae_h_dim']
#         self.z_dimension = paramaters['z_dim']
        self.hid_dim = paramaters['hid_dim']
        self.in_dim = paramaters['in_dim']
        self.num_clasess = paramaters['out_dim']
        # gamma is how many times we use in uncertainty
        self.recommendation_index_list = []
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # AutoEncoder
#         self.vae_model = VAE(self.input_dimension, self.vae_h_dim,
#                              self.z_dimension).to(self.device)
#         self.vae_optimizer = torch.optim.Adam(
#             self.vae_model.parameters(), lr=1e-4)
        # NN
#         self.optimizer = []
#         self.net = []
        self.unselect = []
        self.correct = []
        self.all_fail_data = []
        self.all_success_data = []
        self.success_target = torch.from_numpy(
            np.expand_dims([0, 1], axis=0)).float().cuda()
#         self.delay_success_target = torch.from_numpy(
#             np.expand_dims([0, 1], axis=0)).float().cuda()
        self.fail_target = torch.from_numpy(
            np.expand_dims([1, 0], axis=0)).float().cuda()

        self.bnn_model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.in_dim, out_features=self.hid_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.hid_dim, out_features=2), 
            nn.Softmax(dim=1)).to(self.device)

        self.optimizer = optim.Adam(self.bnn_model.parameters(), lr=self.learning_rate)

#         for i in range(self.actionCount):
#             self.net.append(torch.nn.Sequential(
#                 bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
#                                 in_features=self.input_dimension, out_features=self.hid_dim),
#                 torch.nn.Dropout(0.5),
#                 torch.nn.ReLU(),
#                 bnn.BayesLinear(prior_mu=0, prior_sigma=0.05,
#                                 in_features=self.hid_dim, out_features=self.num_clasess),
#                 torch.nn.Softmax(dim=1)).to(self.device))
#             if i != 0:
#                 self.net[i].load_state_dict(self.net[0].state_dict())
#             self.optimizer.append(optim.Adam(
#                 self.net[i].parameters(), lr=self.learning_rate))#Adam
#             self.unselect.append(0)
#             self.correct.append(0)
#             self.all_fail_data.append([])
#             self.all_success_data.append([])
        for i in range(self.num_classes):
            self.unselect.append(0)
            self.correct.append(0)
            self.all_fail_data.append([])
            self.all_success_data.append([])

    def _nnucb_score(self, in_features):
        """disjoint NNUCB algorithm.
        Parameters
        ----------
        in_features : list
            a list  of input feature

        Returns
        -------
        pred : int
            The index of pred action

        Define
        ----------
        0 as dislike 1as like
        """
        blurry_context = in_features.cuda()
        estimated_reward = []
        uncertainty = []
        #rt_val = []
        score=[]
#         _, _, _, _, z_p,z_n = self.vae_model.cust_forward(data)
#         z_truncated = truncated_normal_(z_p, z_n)

        output_torch = self.bnn_model(blurry_context) # 在上限與下限限制的random normal
        output = torch.flatten(output_torch).tolist()
        target = output
        score.append(max(target))
        uncertainty.append(statistics.stdev(target))
        estimated_reward.append(statistics.mean(target))


#         for net_index in range(self.actionCount):
#             output_torch = self.net[net_index](blurry_context) # 在上限與下限限制的random normal
#             output = torch.flatten(output_torch).tolist()
#             target = output
#             score.append(max(target))
#             uncertainty.append(statistics.stdev(target))
#             estimated_reward.append(statistics.mean(target))
            #rt_val.append(target)
        #print('score:',score)
        #score = list(map(add, estimated_reward, uncertainty))
        return score.index(max(score)), score, estimated_reward, uncertainty # uncertainty: upper bound跟mean的差 

    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dict
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        x2add = np.expand_dims(context, axis=0) # 從二維的變成三維 多加一個外框 e.g. shape from (2,3) -> (1,2,3)
        x2add = np.repeat(x2add, self.gamma, axis=0)
        context_torch = torch.from_numpy(x2add).float()
        recommendation_index, score, estimated_reward, uncertainty = self._nnucb_score( 
            context_torch) # 2021/02/20 Rita
        history_id = self._history_storage.add_history(
            context, recommendation_index)
        self.recommendation_index_list.append(recommendation_index) # 還沒排序好
        return history_id, recommendation_index, score, estimated_reward, uncertainty # 2021/02/20 Rita

#     def early_stop(self, loss, action_id_index, signal):
#         """Control model's weight update
        
#         Parameters
#         ----------
#         loss : tensor
#             The model cost(loss) value.
        
#         action_id_index : int
#             Let the model know which answer should be updated(optimized).
        
#         signal : boolean
#             The signal that wheather the early stop happens or not.
#         """
#         if signal == False:   # update model
#             loss.backward(retain_graph=True)
#             self.optimizer[action_id_index].step()
#         elif signal == True:  # 若early_stop就不要再算了
#             pass
#         else:
#             print('Signal exception ouccur!')
    
    def reward(self, context, action_id_index, reward_in, rt_score_list):
        # remove rtscorelist
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        
        data = torch.from_numpy(np.expand_dims(context, axis=0)).float().cuda()
        data = torch.unsqueeze(data, 1)
        target = None
        if reward_in == 1:
            target = self.success_target
            target = torch.unsqueeze(target, 1)
        else:
            target = self.fail_target
            target = torch.unsqueeze(target, 1)

        # AE
#         for _ in range(10):
#             self.vae_optimizer.zero_grad()
#             recon_batch, mu, logvar, _ = self.vae_model(data)
#             vae_loss = self.vae_model.loss_function(recon_batch, data, mu, logvar)
#             vae_loss.backward()
#             self.vae_optimizer.step()
        if reward_in == 1:
            self.all_success_data[action_id_index].append(data)
        else:
            self.all_fail_data[action_id_index].append(data)

        succcess_count = len(self.all_success_data[action_id_index])
        fail_count = len(self.all_fail_data[action_id_index])
        self.unselect[action_id_index] = self.unselect[action_id_index]+1
        self.optimizer[action_id_index].zero_grad()
        loss = None
        tmp_data = None
        tmp_target = None
        count_limit = self.batch_size

        if succcess_count == 0: # 初始(未成功)

            tmp_data = torch.cat(self.all_fail_data[action_id_index][-fail_count:], out=torch.Tensor(fail_count, 1, self.in_dim).cuda())
            tmp_target = torch.cat([target]*fail_count, out=torch.Tensor(fail_count, 1,  2).cuda()).squeeze(1)

        elif fail_count == 0:   # 初始(未失敗)

            tmp_data = torch.cat(
                self.all_success_data[action_id_index][-succcess_count:], out=torch.Tensor(succcess_count, 1, self.in_dim).cuda())
            tmp_target = torch.cat(
                [target]*succcess_count, out=torch.Tensor(succcess_count, 1,  2).cuda()).squeeze(1)

        elif succcess_count < fail_count: # 失敗次數>成功次數
            if succcess_count > count_limit: # 成功次數>batch size

                data_select = self.all_success_data[action_id_index][-count_limit:] + \
                    self.all_fail_data[action_id_index][-count_limit:]
                tmp_data = torch.cat(data_select, out=torch.Tensor(
                    count_limit*2, 1, 200).cuda())
                tmp_target = torch.cat([self.success_target]*count_limit+[
                                       self.fail_target]*count_limit, out=torch.Tensor(count_limit*2, 1,  2).cuda()).squeeze(1)
            else:

                data_select = self.all_success_data[action_id_index][-succcess_count:] + \
                    self.all_fail_data[action_id_index][-succcess_count:]
                tmp_data = torch.cat(data_select, out=torch.Tensor(
                    succcess_count*2, 1, 200).cuda())
                tmp_target = torch.cat([self.success_target]*succcess_count+[
                                       self.fail_target]*succcess_count, out=torch.Tensor(succcess_count*2, 1,  2).cuda()).squeeze(1)

        else: # 成功次數>失敗次數
            if fail_count > count_limit:

                data_select = self.all_success_data[action_id_index][-count_limit:] + \
                    self.all_fail_data[action_id_index][-count_limit:]
                tmp_data = torch.cat(data_select, out=torch.Tensor(count_limit*2, 1, 200).cuda())
                tmp_target = torch.cat([self.success_target]*count_limit+[
                                       self.fail_target]*count_limit, out=torch.Tensor(count_limit*2, 1,  2).cuda()).squeeze(1)
            else:

                data_select = self.all_success_data[action_id_index][-fail_count:] + \
                    self.all_fail_data[action_id_index][-fail_count:]
                tmp_data = torch.cat(data_select, out=torch.Tensor(
                    fail_count*2, 1, 200).cuda())
                tmp_target = torch.cat([self.success_target]*fail_count+[
                                       self.fail_target]*fail_count, out=torch.Tensor(fail_count*2, 1,  2).cuda()).squeeze(1)

#         _, _, _, z = self.vae_model(tmp_data)
#         output_torch = self.net[action_id_index](z)
        output_torch = self.bnn_model(tmp_data)
        loss = self.loss_fn(output_torch, tmp_target)
        return loss, action_id_index

    