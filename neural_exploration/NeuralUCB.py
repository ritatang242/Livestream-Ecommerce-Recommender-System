import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from neural_exploration import *


# 更新成可以跑 EE-UNN dataset 的 movielens 及 netflix
def run(model):
    """
    修改 UCB 的 run，來達到如果推到使用者沒評分(0.5)的電影時，可以跳過，往後推第二名第三名之類的
    Run an episode of bandit.
    """
    postfix = {
        'total regret': 0.0,
        '% optimal arm': 0.0,
    }
    train_every = 1

    with tqdm(total=model.bandit.T, postfix=postfix) as pbar:
        for t in range(model.bandit.T):
            # update confidence of all arms based on observed features at time t
            model.update_confidence_bounds()
            
            # pick action with the highest boosted estimated reward            
            action = model.sample_action()
            
            def has_reward(action):
                # 若是使用者對該電影有評分（1或0）回傳 True，若為0.5代表沒看過 回傳False
                return model.bandit.rewards[t, action] != 0.5
            
            # 確認推薦的電影 是不是使用者 喜歡 或 不喜歡的
            if not has_reward(action):
                ucb_idx = 1 # 從第二名開始找
                # 此輪的 upper_confidence_bound 由大到小對應的action
                ucb = np.argsort(model.upper_confidence_bounds[t])[::-1]
                while not has_reward(action):
                    # 因為用來推的10部電影一定有其中一部是user喜歡的，所以不應該遇到index out of bound
                    action = ucb[ucb_idx]
                    ucb_idx += 1
                    
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

            # log
            postfix['total regret'] += model.regrets[t]
            n_optimal_arm = np.sum(
                model.actions[:model.iteration] == model.bandit.best_actions_oracle[:model.iteration]
            )
            postfix['% optimal arm'] = '{:.2%}'.format(n_optimal_arm / model.iteration)

            if t % model.throttle == 0:
                pbar.set_postfix(postfix)
                pbar.update(model.throttle)
                
                
def neuralUCB(user_features, reward_df, p = 0.2, hidden_size = 32, epochs = 100, use_cuda=False, seed=np.random.seed(2022)):
    """
        user_features: context dataframe
        reward_df: answer dataframe
        task = {'active', 'balance', 'repeat', 'trendy'}
    """    
    bandit_reward = reward_df.values 
    T = bandit_reward.shape[0]
    n_arms = bandit_reward.shape[1]
        
    bandit_user_feature = np.repeat(user_features.values[:, np.newaxis, :], n_arms, axis=1)
    n_features = bandit_user_feature.shape[2]
    
    
    ### 隨便改一個沒用的 reward function 因為不會用到
    reward_func = lambda x: 0
    bandit = ContextualBandit(T, n_arms, n_features, reward_func, noise_std=0.1, seed=seed)

    # 把內容改成我們的movielens跟netflix的資料
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
                      train_every=1,
                      use_cuda=use_cuda
                     )
    
    
    run(model)
    
    count_dict = {
        'like': 0,
        'hate': 0,
        'delay': 0,
    }

    regret_list = [] # 記錄每輪 推中的情況，0為推中，1為沒推中
    
    for t, pred_action in enumerate(model.actions):
        y_true = reward_df.iloc[t, pred_action]
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
    reward_df['action'] = model.actions
    return regret, reward_df