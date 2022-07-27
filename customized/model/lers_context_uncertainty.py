import numpy as np
import pandas as pd
import torch 
from torch import nn
import matplotlib.pyplot as plt
import progressbar
from customized.model import vae
from customized.model import rnns
from customized import preprocess
from customized import dimension_reducer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LERS_context(nn.Module):
    def __init__(self, full_context_dim=408, vae_h_dim=256, z_dim=20, weight_eps=2, in_dim=811, h1_dim=256, h2_dim=64, out_dim=1): 
        super(LERS_context, self).__init__()
        torch.manual_seed(69112) # 69112
        self.learning_rate = 1e-3
        # AE/VAE
        self.full_context_dim = full_context_dim
        self.vae_h_dim = vae_h_dim
        self.z_dim = z_dim
        self.weight_eps = weight_eps
        # NN/BNN
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
            nn.Linear(h2_dim, out_dim) # 要或不要
        )
        self.vae_model = vae.VAE(full_context_dim, vae_h_dim, z_dim, weight_eps).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=self.learning_rate)
        self.cs_model = rnns.GRU(input_size=23, hidden_size=128, num_layers=1, output_size=7).to(device) 
        self.cp_model = rnns.GRU(input_size=768, hidden_size=256, num_layers=1, output_size=768).to(device)
        self.cs_loss = nn.CrossEntropyLoss()
        self.cp_loss = nn.MSELoss()
        self.cs_optimizer = torch.optim.Adam(self.cs_model.parameters(), lr=self.learning_rate) 
        self.cp_optimizer = torch.optim.Adam(self.cp_model.parameters(), lr=self.learning_rate) 
    
    def forward(self, x):
        logits = self.nn_model(x)
        return logits
    
    def early_stop(self, loss, vae_loss, cs_loss, cp_loss, signal, optimizer):
        """Control model's weight update
        
        Parameters
        ----------
        loss : tensor
            The model cost(loss) value.
        
        signal : boolean
            The signal that wheather the early stop happens or not.
        """
        if signal == False:   # update model
            loss.backward(retain_graph=True)
            optimizer.step()
            # AE/VAE backward
            vae_loss.backward()
            self.vae_optimizer.step()    
            
            # cs model backward 
            cs_loss.backward()
            self.cs_optimizer.step()

            # cp model backward 
            cp_loss.backward()
            self.cp_optimizer.step()
        elif signal == True:  # 若early_stop就不要再算了
            pass
        else:
            print('Signal exception ouccur!')
    
    
def run(cust_num, prod_num, streamer_product, rewards_df, z_dim=20, weight_eps=2, pos_weight=0):
    '''
    input:
    rewards_df: 包含0, 1的全部答案(可由pivot matrix用melt轉出來)
    '''
    # cs model
    s_seq = np.load("data/s_seq.npy")  # (2602, 23*5)
    s_labels = np.load("data/s_labels.npy") # (2602, )
    cs_dataset = preprocess.StreamerDataset(s_seq, s_labels)
    cs_dataloader = torch.utils.data.DataLoader(dataset=cs_dataset, batch_size=1, shuffle=False) # 一定要這步驟因為gru要吃batch_size
    cs_batch_data = []
    for batch in cs_dataloader:
        cs_batch_data.append(batch)
    # cp model
    basket_seq_emb = np.load("data/basket_seq_emb.npy") # (2602, 768)
    basket_tar_emb = np.load("data/basket_tar_emb.npy") # (2602, )
    cp_dataset = preprocess.BasketDataset(basket_seq_emb, basket_tar_emb)
    cp_dataloader = torch.utils.data.DataLoader(dataset=cp_dataset, batch_size=1, shuffle=False) # 一定要這步驟因為gru要吃batch_size
    cp_batch_data = []
    for batch in cp_dataloader:
        cp_batch_data.append(batch)
    # static context    
    static_context = np.load("data/trimmed_static.npy") # (2602, 24)
    # AE/VAE
    full_context_dim = 408
    vae_h_dim = 256
    z_dim = z_dim
    weight_eps = weight_eps
    # NN/BNN
    h1_dim=256
    h2_dim=128
    out_dim=1
    in_dim=z_dim + streamer_product.shape[1]
    scores_idx = {}
    rewards = {}
    highest_scores = []
    highest_idxs = []
    recommend_cnt = 0
    n_correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    regret = 0
    regrets = []
    tolerance = int(0.2 * prod_num)
    hr = []
    # for early stop
    the_last_loss = 100
    last_loss_avg = [1.0]
    patience = 4
    trigger_times = 0
    signal = False
    early_stop_cnt = 0 # 整個模型停了幾次


    end2end = LERS_context(full_context_dim=full_context_dim, vae_h_dim=vae_h_dim, z_dim=z_dim, weight_eps=weight_eps, in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)
    
    if (pos_weight>0): # 若有給class weight再用
        loss_funtion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device) # 包含sigmoid()層, 用這個才可以放class weight
    else: # 沒給就不用
        loss_funtion = nn.BCEWithLogitsLoss() # 把sigmoid+BCE()換改成這個以後，預測出來的值不再是0or1，所以要修改code
    optimizer = torch.optim.Adam(end2end.nn_model.parameters(), lr=end2end.learning_rate) 
        
    tot_prod_num = len(set(rewards_df.商品id))
    all_rewards_array = rewards_df.reward.to_numpy().reshape(-1, tot_prod_num) # 200000 -> 1000, 200; 2602*1023 -> 2602, 1023
    
    # start training
    for t in progressbar.progressbar(range(cust_num)): # (2602, )        
        # cs model forward pass
        cs_inputs = cs_batch_data[t][0].reshape(-1, 5, 23).to(device) 
        cs_labels = cs_batch_data[t][1].to(device) 
        cs_outputs, cs_hn = end2end.cs_model(cs_inputs) 
        cs_loss = end2end.cs_loss(cs_outputs, cs_labels)
        end2end.cs_optimizer.zero_grad()
        streamer_hn = cs_hn.detach().squeeze().cpu().numpy() 
        # cp model forward pass
        cp_inputs = cp_batch_data[t][0].reshape(-1, 10, 768).to(device) 
        cp_labels = cp_batch_data[t][1].to(device) 
        cp_outputs, cp_hn = end2end.cp_model(cp_inputs) 
        cp_loss = end2end.cp_loss(cp_outputs, cp_labels)
        end2end.cp_optimizer.zero_grad()
        basket_hn = cp_hn.detach().squeeze().cpu().numpy()
        # full context 
        temporal_context = np.concatenate((basket_hn, streamer_hn), axis=0) # flatten
        full_context = np.concatenate((temporal_context, static_context[t]), axis=0) # (408, )
        # latent/blurry context
        if type(full_context[0]) == np.float64:
#             print("change full context Double to Float")
            full_context = torch.tensor(full_context, dtype=torch.float)

        x = full_context.to(device)
        x_reconst, mu, log_var, z = end2end.vae_model(x)
        vae_loss, reconst_loss, kl_div = end2end.vae_model.loss(x_reconst, x, mu, log_var)
        end2end.vae_optimizer.zero_grad()
        blurry_context = z.detach().squeeze().cpu().numpy()  # vae有做x.view()所以變成(1, 20) -> 需壓回(20, )
        
        # product recommendation
        scores_idx[t] = []
        rewards[t] = []
        repeat_context = np.repeat(blurry_context, streamer_product.shape[0], axis=0).reshape(-1, blurry_context.shape[0]) # (1023, 20)
        recommendation_contexts = np.concatenate((repeat_context, streamer_product), axis=1) # cust, stream-prod paired # (1023, 811)
        context_rewards = all_rewards_array[t] # (1023, )
        # to tensor & gpu
        recommendation_contexts = torch.tensor(recommendation_contexts, dtype=torch.float).reshape(-1, in_dim).to(device)
        context_rewards = torch.tensor(context_rewards, dtype=torch.float).reshape(-1, out_dim).to(device)
        
        # predict
        pred_scores = end2end.nn_model(recommendation_contexts) # (1023, 1)
        optimizer.zero_grad()
    
        # loss with early stop
        the_current_loss = loss_funtion(pred_scores, context_rewards) # 每個商品都計算loss
        last_loss_avg.append(the_current_loss.item()) # 為了做平均要記錄所有的loss
        if the_current_loss > np.mean(last_loss_avg[-3:]): # 取最後三次做移動平均
            trigger_times += 1
            signal = False
            if trigger_times >= patience:
#                 print('Early stopping!')
                signal = True
                early_stop_cnt += 1
        else:
            trigger_times = 0
            signal = False        
        end2end.early_stop(the_current_loss, vae_loss, cs_loss, cp_loss, signal, optimizer)
        the_last_loss = the_current_loss

#         if (t+1) % 1 == 0:
#             print (f'Step [{t+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        # record for hit ratio
        sort_indices = torch.argsort(pred_scores, dim=0, descending=True).view(-1) # 排序分數 (1023, )
        scores_idx[t].extend(sort_indices.detach().cpu().numpy()) # (2602, 1023) # 可找前x名的原index
        
        # recommendation
        highest_score = torch.max(pred_scores).item()
        highest_scores.append(highest_score) # 2602
        highest_idx = torch.argmax(pred_scores).item() # max index from 1023
        highest_idxs.append(highest_idx) # (2602,)

        # acc & regret
        y_pred_tag = torch.round(torch.sigmoid(torch.tensor([highest_score]))).item() # 先轉tensor才能做sigmoid轉0~1之間
        if y_pred_tag == 1:
            recommend_cnt += 1
        if (y_pred_tag == context_rewards[highest_idx].item()): # 查看推薦商品的答案是否喜歡
            n_correct += 1
        if (y_pred_tag == 1) & (context_rewards[highest_idx].item()==1): # 推薦的真喜歡
            tp += 1
        elif (y_pred_tag == 0) & (context_rewards[highest_idx].item()==0): # 不推薦不喜歡的
            tn += 1
        elif (y_pred_tag == 1) & (context_rewards[highest_idx].item()==0): # 推薦的不喜歡
            fp += 1
        elif (y_pred_tag == 0) & (context_rewards[highest_idx].item()==1): # 不推薦真喜歡的
            fn += 1
        else: # for debug
            print("Something wrong!")
            print("y_pred:", y_pred_tag)
            print("reward:", context_rewards[highest_idx].item())
        acc = 100.0 * n_correct / (t+1) # tp + tn / (tp + tn + fp + fn)
        try:
            precision = 100.0 * tp / (tp + fp)
            recall = 100.0 * tp / (tp + fn)
        except: # 可能分母為零
            precision = 0
            recall = 0
        regrets.append(100* fp / (t+1))  # 推薦的不喜歡
        
        # HR@20 per round        
        prod_in_tolerance = sort_indices.detach()[:tolerance] # (tolerance, )
        hit_per_round = sum(context_rewards[[prod_in_tolerance]]==1).cpu().numpy().item()
        hr20 = 100*hit_per_round/tolerance
        hr.append(hr20) # t個HR@20
    print(f'Recommend Ratio: {100*(recommend_cnt / (t+1)):.2f} %') 
    print(f'Accuracy: {acc:.2f} %') 
    print(f'tp: {tp}') 
    print(f'tn: {tn}') 
    print(f'fp: {fp}') 
    print(f'fn: {fn}') 
    print(f'Precision: {precision:.2f} %') 
    print(f'Recall: {recall:.2f} %') 
    print(f'Correct: {n_correct}')
    print(f'Regret: {np.round(regrets[-1], 2)} %')
    print(f'Early Stop times: {early_stop_cnt}')
    
    return regrets, scores_idx, highest_idxs, hr
#         y_pred_tag = torch.round(torch.sigmoid(torch.tensor([highest_score]))).item() # 先轉tensor才能做sigmoid轉0~1之間
#         if y_pred_tag == 1:
#             recommend_cnt += 1
#         if (y_pred_tag == context_rewards[highest_idx].item()): # 查看推薦商品的答案是否喜歡
#             n_correct += 1
#         acc = 100.0 * n_correct / (t+1)
#         regrets.append( 1 - (n_correct / (t+1)) ) # 推薦的不喜歡
        
#     print(f'Recommend Ratio: {100*(recommend_cnt / (t+1)):.2f} %') 
#     print(f'Accuracy: {acc:.2f} %') 
#     print(f'Correct: {n_correct}')
#     print(f'Regret: {np.round(regrets[-1]*100, 2)} %')
#     print(f'Early Stop times: {early_stop_cnt}')

#     return regrets, scores_idx, highest_idxs

