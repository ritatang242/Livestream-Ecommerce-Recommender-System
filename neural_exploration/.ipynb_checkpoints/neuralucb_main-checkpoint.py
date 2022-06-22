import numpy as np
import pandas as pd
import progressbar
from neural_exploration import *

# 更新成可以跑 EE-UNN dataset 的 movielens 及 netflix
def run(model):
    """
    修改 UCB 的 run，來達到如果推到使用者沒評分(0.5)的電影時，可以跳過，往後推第二名第三名之類的
    Run an episode of bandit.
    """
    sort_scores = []
    
    for t in progressbar.progressbar(range(model.bandit.T)):
        # update confidence of all arms based on observed features at time t
        model.update_confidence_bounds()

        # pick action with the highest boosted estimated reward            
        action = model.sample_action()
        
        def has_reward(action):
            # 若是使用者對該商品有評分（1或0）回傳 True，若為0.5代表沒看過 回傳False
            return model.bandit.rewards[t, action] != 0.5

        # 確認推薦的商品 是不是使用者 喜歡 或 不喜歡的
        if not has_reward(action):
            ucb_idx = 1 # 從第二名開始找
            # 此輪的 upper_confidence_bound 由大到小對應的action
            ucb = np.argsort(model.upper_confidence_bounds[t])[::-1]
            while not has_reward(action):
                # 因為用來推的商品一定有其中一部是user喜歡的，所以不應該遇到index out of bound
                action = ucb[ucb_idx]
                ucb_idx += 1
        
        sort_scores.append(np.argsort(model.upper_confidence_bounds[t]).tolist())

        model.action = action
        model.actions[t] = model.action

        # update approximator
        if t % model.train_every == 0:
            model.train()
        # update exploration indicator A_inv
        model.update_A_inv()
        # compute regret
        model.regrets[t] = model.bandit.best_rewards_oracle[t]-model.bandit.rewards[t, model.action]
        # increment counter
        model.iteration += 1
        
    return sort_scores

                
def myNeuralUCB(user_features, reward_df, p = 0.2, hidden_size = 32, epochs = 100, use_cuda=False, seed=np.random.seed(2022)):
    """
        user_features: context dataframe
        reward_df: answer dataframe
        task = {'active', 'balance', 'repeat', 'trendy'}
    """    
    reward_df_local = reward_df.copy()
    bandit_reward = reward_df_local.values 
    T = bandit_reward.shape[0] # 2602
    n_arms = bandit_reward.shape[1] # 1023
        
    bandit_user_feature = np.repeat(user_features.values[:, np.newaxis, :], n_arms, axis=1)
    n_features = bandit_user_feature.shape[2]
    
    train_every = 1
    
    ### 隨便改一個沒用的 reward function 因為不會用到
    reward_func = lambda x: 0
    bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=0.1, seed=seed)

    # 把內容改成我們的資料
    bandit.features = bandit_user_feature
    bandit.rewards = bandit_reward
    
    model = NeuralUCB(bandit,
                      hidden_size=hidden_size,
                      reg_factor=1.0,
                      delta=0.1,
                      confidence_scaling_factor=0.1,
                      training_window=100,
                      p=p,
                      learning_rate=0.01,
                      epochs=epochs,
                      train_every=train_every,
                      use_cuda=use_cuda
                     )
    
    
    sort_scores = run(model)
    
    count_dict = {
        'like': 0,
        'hate': 0,
        'delay': 0,
    }

    regret_list = [] # 記錄每輪 推中的情況，0為推中，1為沒推中
    
    for t, pred_action in enumerate(model.actions):
        y_true = reward_df_local.iloc[t, pred_action]
        if y_true == 1:        
            count_dict['like'] += 1
            regret_list.append(0)
        elif y_true == 0.5:
            count_dict['delay'] += 1
            regret_list.append(0)
        elif y_true == 0:
            count_dict['hate'] += 1
            regret_list.append(1)
            
    regret = np.cumsum(regret_list) / range(1, len(regret_list)+1)    
    reward_df_local['action'] = model.actions
    
    return regret, reward_df_local, sort_scores

def coverage(reward_df_local):
    hit_action_set = reward_df_local.reset_index().groupby('asid')['action'].apply(set)
    idx2prodid = list(reward_df_local.columns[:-1])
    for uid in hit_action_set.index:
        hit_index = list(hit_action_set[uid])
        hit_index = [idx2prodid[idx] for idx in hit_index]
        hit_action_set[uid] = set(hit_index)
    total_hit_count = 0
    total_miss_count = 0
    total_likes = 0
    total_hates = 0
    for uid in hit_action_set.index:
        user_rating = reward_df_local.loc[uid].iloc[0].drop('action')
        user_likes = set(user_rating[user_rating == 1].index)
        user_hates = set(user_rating[user_rating == 0].index)

        total_likes += len(user_likes)
        total_hates += len(user_hates)

        # hit count
        total_hit_count += len(user_likes & hit_action_set[uid]) # model推中 且 使用者真的喜歡的數量

        # miss count
        total_miss_count += len(user_likes - hit_action_set[uid]) # model沒推中 但 使用者真的喜歡的數量

#     print('total_hit_count:', total_hit_count)
#     print('total_miss_count:', total_miss_count)
#     print('total_likes:', total_likes)
#     print('total_hates:', total_hates)

    user_cover_dict = {}
    user_miss_dict = {}
    total_hit_count = 0
    total_miss_count = 0
    history_cover_ratio = []
    history_miss_ratio = []

    for t in range(len(reward_df_local)):
        uid = reward_df_local.index[t]
        row = reward_df_local.iloc[t]
        action = int(row['action'])
        movie_id = reward_df_local.columns[action]

        if reward_df_local.iloc[t, action] == 1: # 推中使用者喜歡的
            if uid not in user_cover_dict:
                user_cover_dict[uid] = []

            if movie_id not in user_cover_dict[uid]: # 第一次推中喜歡的
                total_hit_count += 1

            user_cover_dict[uid].append(movie_id)

        else: # 推到使用者不喜歡的
            if uid not in user_miss_dict:
                user_miss_dict[uid] = []

            if movie_id not in user_miss_dict[uid]: # 第一次推到不喜歡的
                total_miss_count += 1

            user_miss_dict[uid].append(movie_id)


        history_cover_ratio.append(total_hit_count/total_likes)
        history_miss_ratio.append(total_miss_count/total_hates)

    cover_miss = {
        'cover':history_cover_ratio,
        'miss':history_miss_ratio,    
    }
#     np.save('result/cover_miss.npy', cover_miss)  
    return cover_miss