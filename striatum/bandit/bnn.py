import numpy as np
import torch 
from torch import nn
import torchbnn as bnn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_split
import itertools
import os
current_path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim=403, h1_dim=256, h2_dim=128, out_dim=2): # 403+768product+23streamer 每一輪丟(一個人 配 一個商品)
        super(NeuralNetwork, self).__init__()
        self.in_dim = in_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.out_dim = out_dim
        
        
        self.flatten = nn.Flatten()
        self.nn_model = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(h2_dim, out_dim), # 要或不要
            nn.Softmax(dim=1) # 互斥(加總為一)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.nn_model(x)
        return logits
    
    
    def run(context, input_size):
        
        learning_rate = 0.001
        batch_size = 1
        num_epochs = 11
        h1_dim=256
        h2_dim=128
        out_dim=2
        nn_loss_record = []
        regret = []
        #### preprocess ####
#             context = 
#             targets = 
        #### preprocess ####
        print(context.shape, targets.shape)

        data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True) # 訓練模型時打亂順序

        nn_model = NeuralNetwork(input_size, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)

        loss_funtion = nn.MSELoss()
        optimizer = torch.optim.Adam(cs_model.parameters(), lr=learning_rate) 

        # Train the model
        preds = []
        trues = []
        n_correct = 0
        n_samples = 0
        n_total_steps = len(data_loader) # 2800 rounds
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = nn_model(inputs)
            loss = loss_funtion(outputs, labels)
            nn_loss_record.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % n_total_steps == 150:
                print (f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1) # 7 channels * m products, recommend the product with the highest score 
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item() # 只要有中就算對
            regret.append(1 - (n_correct / n_samples))
            preds.extend(predicted.detach().cpu().numpy())
            trues.extend(labels.detach().cpu().numpy())

        acc = 100.0 * n_correct / n_samples
        print(classification_report(trues, preds))
        print(f'Testing Accuracy: {acc:.4f} %') 
        trues_bin = metrics.label_binarize(trues, classes=[i for i in range(2)])
        preds_bin = metrics.label_binarize(preds, classes=[i for i in range(2)])

        torch.save(cs_model.state_dict(), current_path+'/customized/model/trained_model/product_recommendation/bnn_'+str(b)+'_batches.pth')

        return nn_loss_record, trues_bin, preds_bin
        
    
class BayesianNet(BaseBandit):
    def __init__(self, history_storage, model_storage, action_storage, recommendation_cls=None, paramaters,
                 h1_dim=256, h2_dim=128, out_dim=2, gamma=30):            
        super(BayesianNet, self).__init__(history_storage, model_storage, action_storage, recommendation_cls)
        self.in_dim = paramaters['in_dim']
        self.learning_rate = paramaters['learning_rate']
        self.batch_size = paramaters['batch_size']
        self.gamma = paramaters['gamma']

        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.out_dim = out_dim
        
        
        self.bnn_model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_dim, out_features=h1_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=h1_dim, out_features=h2_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=h2_dim, out_features=out_dim)
#             nn.Softmax(dim=1)
        )

    def _recommedation_score(self, context):
        """
        Parameters
        ----------
        context : list
            a customer with a paired product-streamer feature (一次進一個人搭配一個商品arm)

        Returns
        -------
        pred : int
            The index of pred action
            The max pred prob score
        """
        highest_score = []
        prediction = self.bnn_model(context)  # no softmax: CrossEntropyLoss() 
        score = torch.flatten(prediction).tolist()
        highest_score.append(max(score))
        recommend_id = highest_score.index(max(scores))
        return recommend_id, highest_score
    
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
        repeat_context = np.expand_dims(context, axis=0) # 從二維的變成三維 多加一個外框 e.g. shape from (2,3) -> (1,2,3)
        repeat_context = np.repeat(repeat_context, self.gamma, axis=0) # 模擬非deterministic BNN結果
        repeat_context = torch.from_numpy(repeat_context).float()
        
        history_id, score = self._recommedation_score(repeat_context)
        self._history_storage.add_history(context, history_id)
        self.recommend_id_list.append(history_id) # 還沒排序好
        return history_id, score

    def reward(self, context, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
#         context = (self._history_storage
#                    .get_unrewarded_history(history_id)
#                    .context)
        
        action_context = torch.from_numpy(np.expand_dims(context, axis=0)).float().cuda()
        action_context = torch.unsqueeze(data, 1)

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1, 1))
            if reward_in == 1:
                inputs = reward * action_context 
            elif reward_in == 0.5:
                inputs = reward * action_context 
            else:
                inputs = reward * action_context 
            
        optimizer.zero_grad()    
        outputs = self.bnn_model(inputs)
        tot_loss = loss_funtion(outputs, labels, bnn_model)
        bnn_loss_record[str(batch_size)+" Batches Training Loss"].append(tot_loss.item())
        
        tot_loss.backward()
        optimizer.step()
#         self._history_storage.add_reward(history_id, rewards)
        
        return loss.item(), 
        
        
    def loss(self, score, reward, model)
        ce_loss = torch.nn.CrossEntropyLoss(score, reward) # applies softmax()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)(model)
        kl_weight = 0.01
        tot_loss = ce_loss + (kl_weight * kl_loss)
        return tot_loss
    
    def early_stop(self, loss, history_id, signal):
        """Control model's weight update
        
        Parameters
        ----------
        loss : tensor
            The model cost(loss) value.
        
        history_id : int
            Let the model know which answer should be updated(optimized).
        
        signal : boolean
            The signal that wheather the early stop happens or not.
        """
        if signal == False:   # update model
            loss.backward(retain_graph=True)
            self.optimizer.step()
        elif signal == True:  # 若early_stop就不要再算了
            pass
        else:
            print('Signal exception ouccur!')
    
#     def run(context, input_size):
        
#         learning_rate = 0.001
#         batch_list = [64, 128]
#         num_epochs = 11
#         h1_dim=256
#         h2_dim=128
#         out_dim=2
#         bnn_loss_record = {}
#         bnn_epoch_loss = {}
        
#         for batch_size in batch_list:
#             print("Batch Size:", batch_size)
#             bnn_loss_record[str(batch_size)+" Batches Training Loss"] = []
#             bnn_loss_record[str(batch_size)+" Batches Validation Loss"] = []
#             bnn_epoch_loss[str(batch_size)+" Batches Training Loss"] = []
#             #### preprocess ####
# #             context = 
# #             targets = 
#             #### preprocess ####
#             print(context.shape, targets.shape)

#             data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                                     batch_size=batch_size, 
#                                                     shuffle=True) # 訓練模型時打亂順序

#             bnn_model = BayesianNet(input_size, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)

#             loss_funtion = bnn_model.loss()
#             optimizer = torch.optim.Adam(bnn_model.parameters(), lr=learning_rate) 

#             # Train the model
#             n_total_steps = len(data_loader) 
#             n_correct = 0
#             n_samples = 0
#             preds = []
#             trues = []
#             for epoch in range(num_epochs):
#                 for i, (inputs, labels) in enumerate(data_loader):
#                     inputs = inputs.reshape(-1, input_size).to(device)
#                     labels = labels.to(device)

#                     outputs = self.bnn_model(inputs)
#                     loss = loss_funtion(outputs, labels, bnn_model)
#                     bnn_loss_record[str(batch_size)+" Batches Training Loss"].append(loss.item())
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                     if (i+1) % n_total_steps == 0:
#                         bnn_epoch_loss[str(batch_size)+" Batches Training Loss"].append(loss.item())
#                         if (epoch+1) % 5 == 0:
#                             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#                     _, predicted = torch.max(outputs.data, 1)
#                     n_samples += labels.size(0)
#                     n_correct += (predicted == labels).sum().item()

#                     preds.extend(predicted.detach().cpu().numpy())
#                     trues.extend(labels.detach().cpu().numpy())

#             acc = 100.0 * n_correct / n_samples
#             print(classification_report(trues, preds))
#             print(f'Accuracy: {acc:.4f} %') 
#             trues_bin = metrics.label_binarize(trues, classes=[i for i in range(2)])
#             preds_bin = metrics.label_binarize(preds, classes=[i for i in range(2)])
#             metrics.get_roc_auc(trues_bin, preds_bin, 2, t)
                            
# #             torch.save(cs_model.state_dict(), current_path+'/customized/model/trained_model/product_recommendation/bnn_'+str(b)+'_batches.pth')

#         return bnn_loss_record, bnn_epoch_loss, trues_bin, preds_bin