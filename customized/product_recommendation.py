import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from customized import preprocess
from customized import metrics
from customized.model import NN
from customized.model import bayesianNN
from customized.model import bayesianNN_gamma


def get_data(experiment, cust_id, reward_cust_id):
    if experiment == 'static':
        latent_context = np.load('data/static_latent.npy') # static_latent30
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    elif experiment == 'temporal':
        latent_context = np.load('data/temporal_latent.npy') # temporal_latent30
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    elif experiment == 'streamer':
        latent_context = np.load('data/streamer_latent.npy') 
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    elif experiment == 'basket':
        latent_context = np.load('data/basket_latent.npy') 
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    elif experiment in ['full', 'aenn', 'aebnn']:
        latent_context = np.load('data/latent_context.npy') #  # full_latent30
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    elif experiment in ['vaenn', 'vaebnn']:
        latent_context = np.load('data/blurry_context.npy') # blurry_context
        context, context_id, _ = preprocess.trim_cust_for_context_sort(latent_context, cust_id, reward_cust_id)
    return context, context_id

def policy_generation(experiment, context, streamer_product, rewards_df, class_weight):
    if experiment in ['full', 'static', 'temporal', 'aenn', 'vaenn', 'streamer', 'basket']:
        regrets, scores_idx, highest_idxs = NN.run(context, streamer_product, rewards_df, class_weight)    
    elif experiment in ['aebnn', 'vaebnn']:
        regrets, scores_idx, highest_idxs = bayesianNN_gamma.run(context, streamer_product, rewards_df, class_weight)
    return regrets, scores_idx


def main(experiments, mylabels, fig_name, cust_num, prod_num, class_weight=False):
    # read data
    cust_id = np.load('data/cust_id2.npy')
    sub_txn = pd.read_pickle('data/sub_txn.pkl')
    streamer = pd.read_pickle('data/streamer.pkl')
    reward_pivot_sort = pd.read_csv('data/all_reward_pivot_sort.csv', index_col=0, low_memory=False)
    # trim data
    small_reward_pivot = reward_pivot_sort.iloc[:cust_num, :prod_num]
    reward_cust_id = list(small_reward_pivot.index.map(str))
    reward_prod_id = list(small_reward_pivot.columns.map(str))
    # streamer-product features
    scale_streamer = preprocess.standardize(streamer) # 很重要!!! streamer features有粉絲人數(很大)會使sigmoid爆炸轉不回來
    small_streamer_product = preprocess.concate_streamer_product_features(sub_txn, scale_streamer, reward_prod_id)
    # model_init_weights
    if class_weight == True:
        weight_for_1 = metrics.cal_class_weights(small_reward_pivot)
    else:
        weight_for_1 = 0
        
    # context
    contexts = {}
    for exp in experiments:
        contexts[exp], context_id = get_data(exp, cust_id, reward_cust_id)
    # reward (sorted by context id)
    small_rewards_df = pd.melt(small_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward')\
                    .loc[context_id].reset_index()
    # measurements
    regrets = {}
    scores_idx = {}
    topk = {}
    hr = {}
    for exp in experiments:
        regrets[exp], scores_idx[exp] = policy_generation(exp, contexts[exp], small_streamer_product, small_rewards_df, weight_for_1)        
        hr[exp], k_list, topk[exp] = metrics.cal_hit_ratio(small_rewards_df, scores_idx[exp], min_val=10, max_val=55, step=5)
    
    # regrets
    metrics.plot_regret(regrets, mylabels, fig_name, rounds=cust_num)
    # hit ratio
    metrics.plot_hit_ratio(hr, k_list, mylabels, fig_name)
    
    return regrets, hr, small_reward_pivot, cust_id, reward_cust_id, reward_prod_id, topk

# if __name__ == '__main__':
#     main()