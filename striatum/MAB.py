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
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret    

def policy_generation(bandit, actions):#learningRate,hidden_dimensionNum, gamma,z_dimension,vae_h_dim,batch_size
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    actionstorage = action.MemoryActionStorage()
    actionstorage.add(actions)
    if bandit == 'LinUCB':
        policy = linucb.LinUCB(historystorage, modelstorage, actionstorage, None, 20, 0.3) 
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