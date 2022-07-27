import numpy as np
import pandas as pd
from striatum.storage import history
from striatum.storage import model
from striatum.storage import Action 
from striatum.storage import action 
from striatum.bandit import ucb1
from striatum.bandit import linucb
from striatum.bandit import exp3

def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [100* x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret    

def policy_generation(bandit, actions):#learningRate,hidden_dimensionNum, gamma,z_dimension,vae_h_dim,batch_size
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    actionstorage = action.MemoryActionStorage()
    actionstorage.add(actions)
    if bandit == 'LinUCB':
        policy = linucb.LinUCB(historystorage, modelstorage, actionstorage, None, 20, 0.3) # 20+23+32
    elif bandit == 'UCB1':
        policy = ucb1.UCB1(historystorage, modelstorage, actionstorage) 
    elif bandit == 'Exp3':
        policy = exp3.Exp3(historystorage, modelstorage, actionstorage, gamma=0.2) 
    elif bandit == 'random':
        policy = 0
    return policy

def policy_evaluation(policy, bandit, num, user_features, reward_list, actions, prod_num): # actions: ans id #patiences
    rec_id=[]
    true_rec_id = []
    lossList=[]
    scoreList=[]
    times = num # 2602
    seq_error = np.zeros(shape=(times, 1))
    actions_id = [actions[i].id for i in range(len(actions))] 
    predict_id = []
    
    if bandit in ['LinUCB','UCB1', 'Exp3']:
        correct=0
        cntDelayLike = 0
        cntLike = 0
        cntHate = 0
        cntdict = {"delay":[],"like":[],'hate':[]}
        for t in range(times): # 2602

            feature = np.array(user_features.iloc[t]) # 2602*811  
            full_context = {}
            for action_id in actions_id:
                full_context[action_id] = feature # 對1023個推薦填充que裡的features當作training -> 2604 * 1024 * 811 

            if bandit == 'UCB1':
                history_id, action, score, estimated_reward, uncertainty, total_action_reward, action_times = policy.get_action(full_context, len(actions)) # [2604, 1024], 1024                
                score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
                score_keys = [str(k) for k in score.keys()]
                predict_id.append(score_keys)
                
            if bandit == 'LinUCB':
                history_id, action, estimated_reward, uncertainty, score = policy.get_action(full_context, len(actions)) # [2604, 1024], 1024
                score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
                score_keys = [str(k) for k in score.keys()]
                predict_id.append(score_keys)
                
            if bandit == 'Exp3':
                history_id, action, prob = policy.get_action(full_context, len(actions))
                prob = dict(sorted(prob.items(), key=lambda x: x[1], reverse=True))
                score_keys = [str(k) for k in prob.keys()]
                predict_id.append(score_keys)
            
            watched_list =np.array(reward_list.iloc[t]) 
            rec_id.append( action[0].action.id)

            action_index=0

            if watched_list[actions_id.index(action[action_index].action.id)]==0 :
                cntHate += 1
                if bandit == 'Exp3':
                    cntdict['delay'].append(cntDelayLike)
                    cntdict['like'].append(cntLike)
                    cntdict['hate'].append(cntHate)
                policy.reward(history_id, {action[action_index].action.id: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0                
            elif watched_list[actions_id.index(action[action_index].action.id)]==1 :
                cntLike += 1
                if bandit == 'Exp3':
                    cntdict['delay'].append(cntDelayLike)
                    cntdict['like'].append(cntLike)
                    cntdict['hate'].append(cntHate)
                policy.reward(history_id, {action[action_index].action.id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]  
                    correct=correct+1
            elif watched_list[actions_id.index(action[action_index].action.id)]== 0.5: 
                cntDelayLike += 1
                policy.reward(history_id, {action[action_index].action.id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
                    correct=correct+0.5
            else:
                print('***WARNING*** Not 1, 0, 0.5 !!!')
                print(watched_list[recommendation_index])
            true_rec_id.append(action[action_index].action.id)
#             print(f'cntDelayLike: {cntDelayLike}')
#             print(f'cntLike: {cntLike}')
#             print(f'cntHate: {cntHate}')

        print("Correct",correct)

    elif bandit == 'random':
        cntDelayLike = 0
        cntLike = 0
        cntHate = 0
        correct=0
        cntdict = {"delay":[],"like":[],'hate':[]}
        for t in range(times):
            action = actions_id[np.random.randint(0, len(actions))] # 0~1024隨機抽index
            watched_list =np.array(reward_list.iloc[t])
            rec_id.append(action)
            if watched_list[actions_id.index(action)]==0:
                cntHate += 1
                cntdict['delay'].append(cntDelayLike)
                cntdict['like'].append(cntLike)
                cntdict['hate'].append(cntHate)
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0
            else:
                if watched_list[actions_id.index(action)]==0.5:
                    cntDelayLike += 1
                    cntdict['delay'].append(cntDelayLike)
                    cntdict['like'].append(cntLike)
                    cntdict['hate'].append(cntHate)
                    if t > 0:
                        seq_error[t] = seq_error[t - 1]
                        correct=correct+1  
                else:
                    cntLike += 1
                    cntdict['delay'].append(cntDelayLike)
                    cntdict['like'].append(cntLike)
                    cntdict['hate'].append(cntHate)
                    if t > 0:
                        seq_error[t] = seq_error[t - 1]
                        correct=correct+1  
            true_rec_id.append(action)
#         for i in cntdict['delay']:
#             print("delay:", i)
#         print("======")
#         for i in cntdict['like']:
#             print("like:", i)
#         print("======")
#         for i in cntdict['hate']:
#             print("hate:", i)

        print("Correct",correct) 
        
    return seq_error, true_rec_id, predict_id


def countif_name(df, axis=0, condition=1.0): 
    dic = {}
    if axis == 0: # row # movie
        for i in range(df.shape[1]): 
            dic[df.columns[i]] = (df.iloc[:,i] == condition).sum() # 這部電影有多少人喜歡
        result = pd.DataFrame.from_dict(dic, orient='index').rename(columns={0:'cnt'})
    elif axis == 1: # column # user
        for i in range(df.shape[0]): 
            dic[df.index[i]] = (df.iloc[i,:] == condition).sum() # 這個人喜歡多少部電影
        result = pd.DataFrame.from_dict(dic, orient='index').rename(columns={0:'cnt'})
    return result

def cal_coverage_non_repeat(rewards_df, rewards_pivot, rec_id):
    rec_df = pd.DataFrame({'asid': list(rewards_pivot.index), 'rec_id': rec_id})
    rec_df.rec_id = rec_df.rec_id.apply(str)
    rewards_df = rewards_df.drop_duplicates().reset_index(drop=True)
    rec_record = rewards_df.merge(rec_df, how='left', left_on=['asid','商品id'], right_on=['asid','rec_id']).fillna(0)
    cnt = 0
    cover_ratio = 0
    coverage = []
    ground_truth = sum(rewards_df.reward==1)
    for i in range(len(rec_record)): # 推薦幾輪
        if (rec_record.reward[i] == 1) & (rec_record.rec_id[i]!=0): # 確實喜歡且有推薦到 # 沒推薦的話rec_id會是na並以0表示
            if cover_ratio < 100: # non-repeated hits # 必須是小於, 這樣最後一輪才會加到剛好100%
                cnt += 1
                cover_ratio = 100* cnt / ground_truth
                coverage.append(cover_ratio)
            else: # repeated hits
                coverage.append(100)
        elif rec_record.rec_id[i]!=0: # 做100輪所以會有100筆記錄, 剩下的情況是不喜歡但有推薦到 # 控制在100輪內(repeat dataset共100rounds)
            coverage.append(cover_ratio)
    return coverage


def cal_coverage_repeat(num, bandit, rewards_df, rewards, rec_id):
    i = 0     
    rewards_df = rewards_df.drop_duplicates().reset_index(drop=True)
    ground_truth = sum(rewards_df.reward==1)
    overall_cover_ratio = 0
    overall_miss_ratio = 0
    coverage_list = []
    miss_rate_list = []
    rec_table = pd.DataFrame({'algo': bandit, 'user': rewards.iloc[:,0], 'rec_id': rec_id})  # 我只要知道重複來的user是誰
    list1 = list(rewards.index)
    list2 = rec_table['rec_id']
    user_list = list1[0:10]
    algo_dict = {}
    for u, m in zip(list1, list2):
        algo_dict.setdefault(u, []).append(m)
    correct_movie = pd.DataFrame([])
    wrong_movie = pd.DataFrame([])
    r = 0
    count_user_likes = countif_name(rewards, axis = 1, condition = 1.0)
    count_user_hates = countif_name(rewards, axis = 1, condition = 0.0) 
    for k,v in algo_dict.items():
        user_likes_cnt = count_user_likes.loc[k][0] # 每個user喜歡的電影數 # 5部
        user_hate_cnt = count_user_hates.loc[k][0]
        if user_likes_cnt == 0:
            print(f"[Round {r}] User {k} has no favor movie.")
            # 這些人對coverage沒有影響,長度會少(因為沒有答案)
        for m in v:  # v is a list
            r += 1   # r means the recommend times of movies
            preference = (rewards.loc[[k], [str(m)]]).values[0] # 因為k(user)在外層loop所以會重複10次, 只要取第一個值即可
            if preference == 1.0:     # 確實喜歡
                correct_movie = correct_movie.append(pd.DataFrame({'user': k, 'correct_movie': m}, index=[i]), ignore_index=True)  # algo推薦的movie同時也是user確實喜歡(猜對)的movie清單
            elif preference == 0.0:
                wrong_movie = wrong_movie.append(pd.DataFrame({'user': k, 'wrong_movie': m}, index=[i]), ignore_index=True) 

            uni_correct_set = correct_movie.drop_duplicates() 
            uni_wrong_set = wrong_movie.drop_duplicates() 

            if (uni_correct_set.shape[0] != 0) & (overall_cover_ratio<100):  
                # 總體
                correct_cnt_table = pd.DataFrame({'cnt':uni_correct_set.groupby(["user"]).size()})['cnt'] 
                overall_cum_cnt = sum(correct_cnt_table.values)
#                 overall_cover_ratio = 100* overall_cum_cnt/(user_likes_cnt*len(set(list1))) # 所有user喜歡的總數  
                overall_cover_ratio = 100* overall_cum_cnt/ground_truth
                coverage_list.append(overall_cover_ratio)

                if (r==num):
                    print("=====hit=====")
                    print(f"[Final] cumulative cover ratio is: {100*overall_cum_cnt/ground_truth: .4f}")
                    print(correct_cnt_table)

            else:                   # 推薦的並不喜歡
                coverage_list.append(overall_cover_ratio)

            if wrong_movie.shape[0] != 0:  
                wrong_cnt_table = pd.DataFrame({'cnt':uni_wrong_set.groupby(["user"]).size()})['cnt'] 
                overall_cum_cnt = sum(wrong_cnt_table.values)
                overall_miss_ratio = 100* overall_cum_cnt/ground_truth
#                 overall_miss_ratio = 100* overall_cum_cnt/(user_hate_cnt*len(set(list1))) # 所有user喜歡的電影總數 # 50部 
                miss_rate_list.append(overall_miss_ratio)

                if (r==num):
                    print("=====miss=====")
                    print(f"[Final] cumulative miss ratio is: {100*overall_cum_cnt/ground_truth: .2f} %")
                    print(wrong_cnt_table)
            else:
                miss_rate_list.append(overall_miss_ratio)
        
    i+=1
    
    return coverage_list # miss_rate_list