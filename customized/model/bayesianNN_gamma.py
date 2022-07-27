import numpy as np
import pandas as pd
import torch 
from torch import nn
import torchbnn as bnn
import matplotlib.pyplot as plt
import progressbar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesianNet(nn.Module):
    def __init__(self, in_dim=811, h1_dim=256, h2_dim=64, out_dim=1): # 20customer+768product+23streamer 每一輪丟(一個人 配 一個商品)
        super(BayesianNet, self).__init__()
        torch.manual_seed(2016221) # 2016221 1016 10 1218
        self.in_dim = in_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.out_dim = out_dim
        
        
        self.flatten = nn.Flatten()
        self.bnn_model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=in_dim, out_features=h1_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=h1_dim, out_features=h2_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=h2_dim, out_features=out_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.bnn_model(x)
        return logits
        
    def early_stop(self, loss, signal, optimizer):
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
        elif signal == True:  # 若early_stop就不要再算了
            pass
        else:
            print('Signal exception ouccur!')
            
            
def run(lers_context, streamer_product, rewards_df, pos_weight=0):
    '''
    input:
    rewards_df: 包含0, 1的全部答案(可由pivot matrix用melt轉出來)
    '''
    n_total_steps = len(lers_context) # 2602 rounds
    learning_rate = 0.001
    h1_dim=256
    h2_dim=128
    out_dim=1
    in_dim=lers_context.shape[1] + streamer_product.shape[1]
    gamma=30
    scores_idx = {}
    rewards = {}
    highest_scores = []
    highest_idxs = []
    recommend_cnt = 0
    n_correct = 0
    regret = 0
    regrets = []
    # for early stop
    the_last_loss = 100
    last_loss_avg = [1.0]
    patience = 4
    trigger_times = 0
    signal = False
    early_stop_cnt = 0 # 整個模型停了幾次
    # for early stop

    bnn_model = BayesianNet(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)
    if (pos_weight>0): # 若有給class weight再用
        ce_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device) # 包含sigmoid()層, 用這個才可以放class weight
    else: # 沒給就不用
        ce_function = nn.BCEWithLogitsLoss() # 把sigmoid+BCE()換改成這個以後，預測出來的值不再是0or1，所以要修改code
    kl_function = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01
    optimizer = torch.optim.Adam(bnn_model.parameters(), lr=learning_rate) 
    
    tot_prod_num = len(set(rewards_df.商品id))
    all_rewards_array = rewards_df.reward.to_numpy().reshape(-1, tot_prod_num) # 200000 -> 1000, 200; 2602*1023 -> 2602, 1023

    
    # start training
    for t in progressbar.progressbar(range(n_total_steps)): # (2602, )
        scores_idx[t] = []
        rewards[t] = []
        repeat_context = np.repeat(lers_context[t], streamer_product.shape[0], axis=0).reshape(-1, lers_context.shape[1]) # (1023, 20)
        recommendation_contexts = np.concatenate((repeat_context, streamer_product), axis=1) # cust, stream-prod paired # (1023, 811)
        context_rewards = all_rewards_array[t] # (1023, )
        
        # for gamma tricks
        # 重複進來gamma次, 選一次最高的 (只需要score、不能更新gamma次、只能更新最高分的那一次)
        recommendation_contexts_gamma = np.repeat(recommendation_contexts, gamma, axis=0) # (1023*gamma, 20)
        # to tensor & gpu
        recommendation_contexts_gamma = torch.tensor(recommendation_contexts_gamma, dtype=torch.float).reshape(-1, in_dim).to(device) 
        # predict
        pred_scores_gamma = bnn_model(recommendation_contexts_gamma) # 回傳 gamma * 選項 長度的 1維列表(預測分數)
        matrix_gamma_scores = pred_scores_gamma.reshape(-1, gamma)   # shape = (選項, gamma)
        matrix_gamma_scores_cpu = matrix_gamma_scores.detach().cpu().numpy() # 為了使用np的function得先將tensor轉到cpu才能計算
        flatten_index = torch.argmax(matrix_gamma_scores).item()    # argmax只會回傳flatten後的index, 我們真正想知道的是第幾次的第幾個選項最高分
        matrix_index = np.unravel_index(flatten_index, matrix_gamma_scores_cpu.shape) # 獲得shape(選項, gamma次)的最高分的2維index
        which_gamma = matrix_index[1] # 找出是第幾次的gamma的維度
        best_gamma_scores = matrix_gamma_scores_cpu[:, which_gamma] # 用該次gamma獲得整組預測分數(同一批次的1023個選項的分數) 
        best_gamma_scores = torch.tensor(best_gamma_scores.astype('float32'), dtype=torch.float).reshape(-1, 1).to(device) # 要轉float因為前面轉numpy時會變成long
                
        
        # to tensor & gpu
        context_rewards = torch.tensor(context_rewards, dtype=torch.float).reshape(-1, out_dim).to(device)
        
        optimizer.zero_grad()

################## 二擇一 ##################        
        # loss with early stop
        # update loss once with the best results from the gamma times
        ce_loss = ce_function(best_gamma_scores, context_rewards) # applies softmax()
        kl_loss = kl_function(bnn_model)
        the_current_loss = ce_loss + (kl_weight * kl_loss)
        last_loss_avg.append(the_current_loss.item()) # 為了做平均要記錄所有的loss
        if the_current_loss > np.mean(last_loss_avg[-3:]): # 取最後三次做移動平均
            trigger_times += 1
            signal = False
            if trigger_times >= patience:
#                 print('Early stopping!')
                signal = True
                early_stop_cnt += 1
        else:
            trigger_times = 0 # 重新計次
            signal = False        
        bnn_model.early_stop(the_current_loss, signal, optimizer)
        the_last_loss = the_current_loss
        
        # update loss once with the best results from the gamma times
#         ce_loss = ce_function(best_gamma_scores, context_rewards) # applies softmax()
#         kl_loss = kl_function(bnn_model)
#         tot_loss = ce_loss + (kl_weight * kl_loss)
#         optimizer.zero_grad()
#         tot_loss.backward()
#         optimizer.step()
#         if (t+1) % 1 == 0:
#             print (f'Step [{t+1}/{n_total_steps}], Loss: {tot_loss.item():.4f}')
################## 二擇一 ##################        

        # record for hit ratio
        sort_indices = torch.argsort(best_gamma_scores, dim=0, descending=True).view(-1) # 排序分數 (1023, )
        scores_idx[t].extend(sort_indices.detach().cpu().numpy()) # (2602, 1023) # 可找前x名的原index
        
        # recommendation
        highest_score = torch.max(best_gamma_scores).item()
        highest_scores.append(highest_score) # 2602
        highest_idx = torch.argmax(best_gamma_scores).item() # max index from 1023
        highest_idxs.append(highest_idx) # (2602,)

        # acc & regret
        y_pred_tag = torch.round(torch.sigmoid(torch.tensor([highest_score]))).item() # 先轉tensor才能做sigmoid轉0~1之間
        if y_pred_tag == 1:
            recommend_cnt += 1
        if (y_pred_tag == context_rewards[highest_idx].item()): # 查看推薦商品的答案是否喜歡
            n_correct += 1
        acc = 100.0 * n_correct / (t+1)
        regrets.append( 1 - (n_correct / (t+1)) ) # 推薦的不喜歡

    print(f'Recommend Ratio: {100*(recommend_cnt / (t+1)):.2f} %') 
    print(f'Accuracy: {acc:.2f} %') 
    print(f'Correct: {n_correct}')
    print(f'Regret: {np.round(regrets[-1], 4)}')
    print(f'Early Stop times: {early_stop_cnt}')
    print(f'Current seed: {torch.seed()}')

    return regrets, scores_idx, highest_idxs
