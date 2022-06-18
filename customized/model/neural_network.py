import numpy as np
import pandas as pd
import torch 
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import os
current_path = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim=811, h1_dim=256, h2_dim=64, out_dim=1): # 20customer+768product+23streamer 每一輪丟(一個人 配 一個商品)
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
            nn.Sigmoid()
        )

    def forward(self, x):
#         x = self.flatten(x)
        score = self.nn_model(x) 
#         score = nn.Sigmoid(logits) # 轉換成0~1的分數
        return score
    
    
def run(lers_context, all_streamer_product, all_reward_pivot, prod_num=1023):
    print(f"放入{prod_num}個商品")
    n_total_steps = len(lers_context) # 2602 rounds
    learning_rate = 0.001
    h1_dim=256
    h2_dim=128
    out_dim=1
    in_dim=811
    nn_loss_record = {}
    scores = {}
    rewards = {}
    highest_scores = []
    highest_idxs = []
    n_correct = 0
    n_samples = 0
    regret = []

    nn_model = NeuralNetwork(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim).to(device)
    loss_funtion = nn.BCELoss() 
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate) 
    

    for t in tqdm(range(n_total_steps)): # 2602
        nn_loss_record[t] = []
        scores[t] = []
        rewards[t] = []
        for prod in range(len(all_streamer_product)): # 1023
#         for prod in range(prod_num): # 1023

            recommendation_contexts = np.concatenate((lers_context[t], all_streamer_product[prod]), axis=0) # pair (cust, stream-prod)
            in_dim = recommendation_contexts.shape[0] # 811
            reward = all_reward_pivot.to_numpy()[t][prod] # get the target of the pairred (cust, stream-prod)

            recommendation_contexts = torch.tensor(recommendation_contexts, dtype=torch.float)
            reward = torch.tensor([reward], dtype=torch.float).view(1,1)
            recommendation_contexts = recommendation_contexts.reshape(-1, in_dim).to(device)
            reward = reward.to(device)

            # predict
            pred_score = nn_model(recommendation_contexts) # probility of buy or not
            scores[t].extend(pred_score.detach().cpu().numpy()) # 2602, 1023
            rewards[t].extend(reward.detach().cpu().numpy()) # 2602, 1023

            # loss
            loss = loss_funtion(pred_score, reward) # 每個商品都計算loss
            nn_loss_record[t].append(loss.item()) # 2602, 1023
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #     if (t+1) % 100 == 0:
    #         print (f'Step [{t+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        # recommend
        highest_score = max(scores[t]).item()
        highest_scores.append(highest_score) # 2602
        highest_idx = scores[t].index(max(scores[t])) # max index from 1023
        highest_idxs.append(highest_idx) # 2602

        # regret
        n_samples += reward.size(0)
        n_correct += (highest_score == rewards[t][highest_idx].item()) # 只要有中就算對
        acc = 100.0 * n_correct / n_samples
        regret.append(round(1 - (n_correct / n_samples), 4)) # 最高分的才推薦並計算regret

    print(f'Accuracy: {acc:.4f} %') 
    print(f'Correct: {n_correct}')
    print(f'Regret: {regret[-1]}')

    return regret, rewards, highest_idxs