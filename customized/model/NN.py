import numpy as np
import pandas as pd
import torch 
from torch import nn
import matplotlib.pyplot as plt
import progressbar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim=811, h1_dim=256, h2_dim=64, out_dim=1): # 20customer+768product+23streamer 每一輪丟(一個人 配 一個商品)
        super(NeuralNetwork, self).__init__()
        torch.manual_seed(69112) # 69112
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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.nn_model(x)
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

    nn_model = NeuralNetwork(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)
    
    if (pos_weight>0): # 若有給class weight再用
        loss_funtion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device) # 包含sigmoid()層, 用這個才可以放class weight
    else: # 沒給就不用
        loss_funtion = nn.BCEWithLogitsLoss() # 把sigmoid+BCE()換改成這個以後，預測出來的值不再是0or1，所以要修改code
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate) 

    tot_prod_num = len(set(rewards_df.商品id))
    all_rewards_array = rewards_df.reward.to_numpy().reshape(-1, tot_prod_num) # 200000 -> 1000, 200; 2602*1023 -> 2602, 1023
    
    # start training
    for t in progressbar.progressbar(range(n_total_steps)): # (2602, )
        scores_idx[t] = []
        rewards[t] = []
        repeat_context = np.repeat(lers_context[t], streamer_product.shape[0], axis=0).reshape(-1, lers_context.shape[1]) # (1023, 20)
        recommendation_contexts = np.concatenate((repeat_context, streamer_product), axis=1) # cust, stream-prod paired # (1023, 811)
        context_rewards = all_rewards_array[t] # (1023, )
        # to tensor & gpu
        recommendation_contexts = torch.tensor(recommendation_contexts, dtype=torch.float).reshape(-1, in_dim).to(device)
        context_rewards = torch.tensor(context_rewards, dtype=torch.float).reshape(-1, out_dim).to(device)
        
        # predict
        pred_scores = nn_model(recommendation_contexts) # (1023, 1)
#         print("sigmoid:",torch.sigmoid(pred_scores))

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
        nn_model.early_stop(the_current_loss, signal, optimizer)
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
        acc = 100.0 * n_correct / (t+1)
        regrets.append( 1 - (n_correct / (t+1)) ) # 推薦的不喜歡
        
    print(f'Recommend Ratio: {100*(recommend_cnt / (t+1)):.2f} %') 
    print(f'Accuracy: {acc:.2f} %') 
    print(f'Correct: {n_correct}')
    print(f'Regret: {np.round(regrets[-1], 4)}')
    print(f'Early Stop times: {early_stop_cnt}')
    print(f'Current seed: {torch.seed()}')

    return regrets, scores_idx, highest_idxs

# old ver. (slow!!!)

# def run(lers_context, all_streamer_product, all_reward_pivot, pos_weight=0):
#     n_total_steps = len(lers_context) # 2602 rounds
#     learning_rate = 0.001
#     h1_dim=256
#     h2_dim=128
#     out_dim=1
#     in_dim=811
#     nn_loss_record = {}
#     scores = {}
#     rewards = {}
#     highest_scores = []
#     highest_idxs = []
#     n_correct = 0
#     n_samples = 0
#     regret = 0
#     regrets = []

#     nn_model = NeuralNetwork(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)
#     if (pos_weight>0): # 若有給class weight再用
#         loss_funtion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).to(device) # 包含sigmoid()層, 用這個才可以放class weight
#     else: # 沒給就不用
#         loss_funtion = nn.BCEWithLogitsLoss() # 把sigmoid+BCE()換改成這個以後，預測出來的值不再是0or1，所以要修改code
#     optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate) 
    

#     for t in progressbar.progressbar(range(n_total_steps)): # 2602
#         nn_loss_record[t] = []
#         scores[t] = []
#         rewards[t] = []
#         for prod in range(len(all_streamer_product)): # 1023
            
#             recommendation_contexts = np.concatenate((lers_context[t], all_streamer_product[prod]), axis=0) # pair (cust, stream-prod)
#             reward = all_reward_pivot.to_numpy()[t][prod] # get the target of the pairred (cust, stream-prod)

#             recommendation_contexts = torch.tensor(recommendation_contexts, dtype=torch.float)
#             reward = torch.tensor([reward], dtype=torch.float).view(1,1)
#             recommendation_contexts = recommendation_contexts.reshape(-1, in_dim).to(device)
#             reward = reward.to(device)

#             # predict
#             pred_score = nn_model(recommendation_contexts) # buy or not
            
# #             print("sigmoid:",torch.sigmoid(pred_score))
#             scores[t].extend(pred_score.detach().cpu().numpy()) # 2602, 1023
#             rewards[t].extend(reward.detach().cpu().numpy()) # 2602, 1023

#             # loss
#             loss = loss_funtion(pred_score, reward) # 每個商品都計算loss
# #             print(loss.item())
#             nn_loss_record[t].append(loss.item()) # 2602, 1023
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

            
# #         if (t+1) % 1 == 0:
# #             print (f'Step [{t+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#         # recommend
#         highest_score = max(scores[t]).item()
#         highest_scores.append(highest_score) # 2602
#         highest_idx = scores[t].index(max(scores[t])) # max index from 1023
#         highest_idxs.append(highest_idx) # 2602

#         # regret
#         n_samples += reward.size(0) # same as t
#         y_pred_tag = torch.round(torch.sigmoid(torch.tensor([highest_score]))).item()
#         if (y_pred_tag == rewards[t][highest_idx].item()): # 只要有中就算對
#             n_correct += 1
#         acc = 100.0 * n_correct / n_samples
#         regrets.append( 1 - (n_correct / n_samples) )
        
# #         if rewards[t][highest_idx] == 1: # 若最終推薦的為買
# #             n_correct += 1 # 可能重複
# #         elif rewards[t][highest_idx] == 0: # 若最終推薦的商品不買
# #             regret += 1
# #         regrets.append(round((regret / n_samples), 4))
        
#     print(f'Accuracy: {acc:.4f} %') 
#     print(f'Correct: {n_correct}')
#     print(f'Regret: {regrets[-1]}')

#     return regrets, rewards, highest_idxs