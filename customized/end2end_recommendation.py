import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from customized import preprocess
from customized import metrics
from customized.model import lers_without_uncertainty
from customized.model import lers_context_uncertainty
from customized.model import lers_recommendation_uncertainty
from customized.model import lers_both_uncertainties


def policy_generation(experiment, cust_num, prod_num, streamer_product, rewards_df, z_dim, weight_eps, class_weight):
    if experiment in ['full', 'static', 'temporal', 'aenn', 'streamer', 'basket']:
        regrets, scores_idx, highest_idxs, hr_per_round = lers_without_uncertainty.run(cust_num, prod_num, streamer_product, rewards_df, z_dim, class_weight)
    elif experiment == 'vaenn':
        regrets, scores_idx, highest_idxs, hr_per_round = lers_context_uncertainty.run(cust_num, prod_num, streamer_product, rewards_df, z_dim, weight_eps, class_weight)
    elif experiment == 'aebnn':
        regrets, scores_idx, highest_idxs, hr_per_round = lers_recommendation_uncertainty.run(cust_num, prod_num, streamer_product, rewards_df, z_dim, class_weight)
    elif experiment == 'vaebnn':
        regrets, scores_idx, highest_idxs, hr_per_round = lers_both_uncertainties.run(cust_num, prod_num, streamer_product, rewards_df, z_dim, weight_eps, class_weight)
    return regrets, scores_idx, highest_idxs, hr_per_round


def main(experiments, mylabels, fig_name, cust_num, prod_num, z_dim=20, weight_eps=2, class_weight=False):
    # read data
    cust_id = np.load('data/cust_id2.npy')
    sub_txn = pd.read_pickle('data/sub_txn.pkl')
    streamer = pd.read_pickle('data/streamer.pkl')
    reward_pivot_sort = pd.read_csv('data/all_reward_pivot_sort.csv', index_col=0, low_memory=False)
    # trim data
    small_reward_pivot = reward_pivot_sort.iloc[:cust_num, :prod_num]
    reward_cust_id = list(small_reward_pivot.index.map(str)) # 2602人
    reward_prod_id = list(small_reward_pivot.columns.map(str)) 
    # streamer-product features
    scale_streamer = preprocess.standardize(streamer) # 很重要!!! streamer features有粉絲人數(很大)會使sigmoid爆炸轉不回來
    small_streamer_product = preprocess.concate_streamer_product_features(sub_txn, scale_streamer, reward_prod_id)
    # model_init_weights
    if class_weight == True:
        weight_for_1 = metrics.cal_class_weights(small_reward_pivot)
    else:
        weight_for_1 = 0
        
    # reward (sorted by context id)
    small_rewards_df = pd.melt(small_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward')\
                    .loc[reward_cust_id].reset_index()
    # for coverage calculation
    small_reward = pd.melt(small_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward').reset_index()\
                        .drop_duplicates().reset_index(drop=True)
    # measurements
    regrets = {}
    scores_idx = {}
    topk = {}
    highest_idxs = {}
    rec_results = {}
    coverages = {}
    diversity = {}
    hr = {}
    hr_per_round = {}
    
    for exp in experiments:
        regrets[exp], scores_idx[exp], highest_idxs[exp], hr_per_round[exp] = policy_generation(exp, cust_num, prod_num, small_streamer_product, small_rewards_df, z_dim, weight_eps, weight_for_1)        
        rec_results[exp] = preprocess.recommendation_results_df(highest_idxs[exp], reward_cust_id, reward_prod_id)        
        coverages[exp] = preprocess.coverage_list(small_reward, rec_results[exp])   
        diversity[exp] = metrics.cal_diversity(prod_num, rec_results[exp].rec_id)
        hr[exp], k_list, topk[exp] = metrics.cal_hit_ratio(cust_num, small_rewards_df, scores_idx[exp], min_val=10, max_val=110, step=10)
    
    # regrets
    metrics.plot_regret(regrets, mylabels, fig_name, rounds=cust_num)
    # coverage
    metrics.plot_coverage(coverages, mylabels, fig_name, rounds=cust_num)
    # diversity
    metrics.plot_diversity(diversity, mylabels, fig_name, rounds=cust_num)
    # HR@rounds
    metrics.plot_hit_ratio_per_round(hr_per_round, mylabels, fig_name, rounds=cust_num)
    # HR@K
    metrics.plot_hit_ratio(hr, k_list, mylabels, fig_name)
    
    
    return regrets, hr, coverages, diversity, small_reward_pivot, cust_id, reward_cust_id, reward_prod_id, highest_idxs

# if __name__ == '__main__':
#     main()