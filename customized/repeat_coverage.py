import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from customized import preprocess
from customized import metrics
from customized.model import NN
from customized.model import bayesianNN


def get_data(experiment, cust_id, reward_cust_id):
    if experiment in ['aenn', 'aebnn']:
        latent_context = np.load('data/latent_vector.npy')
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    elif experiment in ['vaenn', 'vaebnn']:
        latent_context = np.load('data/blurry_context.npy')
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    return context, context_id


def policy_generation(experiment, context, streamer_product, rewards_df, class_weight=0):
    if experiment == 'aenn':
        regrets, scores_idx, highest_idxs = NN.run(context, streamer_product, rewards_df, class_weight)
    elif experiment == 'vaenn':
        regrets, scores_idx, highest_idxs = NN.run(context, streamer_product, rewards_df, class_weight)
    elif experiment == 'aebnn':
        regrets, scores_idx, highest_idxs = bayesianNN.run(context, streamer_product, rewards_df, class_weight)
    elif experiment == 'vaebnn':
        regrets, scores_idx, highest_idxs = bayesianNN.run(context, streamer_product, rewards_df, class_weight)
    return regrets, scores_idx, highest_idxs


def main(experiments, mylabels, fig_name, cust_num, prod_num, class_weight=False):
    # read data
    cust_id = np.load('data/cust_id2.npy')
    sub_txn = pd.read_pickle('data/sub_txn.pkl')
    streamer = pd.read_pickle('data/streamer.pkl')
    repeat_reward_pivot = pd.read_csv('data/reward_pivot_repeat.csv', index_col=0, low_memory=False)
    # 由於repeat的csv一直無法將id讀作字串所以要先轉, 後面melt時才能對應到
    repeat_reward_pivot.index = repeat_reward_pivot.index.map(str) 
    repeat_reward_pivot.columns = repeat_reward_pivot.columns.map(str)
    # trim data
    reward_cust_id = list(repeat_reward_pivot.index)
    reward_prod_id = list(repeat_reward_pivot.columns)
    # streamer-product features
    repeat_streamer_product = preprocess.concate_streamer_product_features(sub_txn, streamer, reward_prod_id)
    # context
    contexts = {}
    for exp in experiments:
        contexts[exp], context_id = get_data(exp, cust_id, reward_cust_id)
    # reward (sorted by context id)
    repeat_rewards_df = pd.melt(repeat_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward')\
                            .loc[context_id[:10]].reset_index() # 需截斷至10(因為是重複的)
    # for coverage calculation
    repeat_reward = pd.melt(repeat_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward').reset_index()\
                        .drop_duplicates().reset_index(drop=True)
    # measurements
    regrets = {}
    scores_idx = {}
    highest_idxs = {}
    rec_results = {}
    coverages = {}
    hr = {}
    topk = {}
    
    for exp in experiments:
        regrets[exp], scores_idx[exp], highest_idxs[exp] = policy_generation(exp, contexts[exp], repeat_streamer_product,\
                                                                             repeat_rewards_df)    
        rec_results[exp] = preprocess.recommendation_results_df(highest_idxs[exp], reward_cust_id, reward_prod_id)        
        coverages[exp] = preprocess.coverage_list(repeat_reward, rec_results[exp])    
        hr[exp], k_list, topk[exp] = metrics.cal_hit_ratio(repeat_rewards_df, scores_idx[exp], min_val=1, max_val=11, step=1) # 因為只有10個商品所以要縮小topK範圍
    
    # regrets
    metrics.plot_regret(regrets, mylabels, fig_name, rounds=cust_num)
    # coverage
    metrics.plot_coverage(coverages, mylabels, fig_name)
    # hit ratio
    metrics.plot_hit_ratio(hr, k_list, mylabels, fig_name)
    
    return regrets, hr, coverages, repeat_reward_pivot, cust_id, reward_cust_id, reward_prod_id, topk

# if __name__ == '__main__':
#     main()