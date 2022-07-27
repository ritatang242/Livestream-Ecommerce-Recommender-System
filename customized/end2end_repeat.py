import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from customized import preprocess
from customized import metrics
from customized.model import lers_without_uncertainty
from customized.model import lers_context_uncertainty
from customized.model import lers_recommendation_uncertainty
from customized.model import lers_both_uncertainties


def policy_generation(experiment, cust_num, prod_num, streamer_product, rewards_df, z_dim=20, weight_eps=2, class_weight=0):
    if experiment == 'aenn':
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
    repeat_reward_pivot = pd.read_csv('data/reward_pivot_repeat.csv', index_col=0, low_memory=False)
    # 由於repeat的csv一直無法將id讀作字串所以要先轉, 後面melt時才能對應到
    repeat_reward_pivot.index = repeat_reward_pivot.index.map(str) 
    repeat_reward_pivot.columns = repeat_reward_pivot.columns.map(str)
    # trim data
    reward_cust_id = list(repeat_reward_pivot.index)
    reward_prod_id = list(repeat_reward_pivot.columns)
    # streamer-product features
    scale_streamer = preprocess.standardize(streamer)
    repeat_streamer_product = preprocess.concate_streamer_product_features(sub_txn, scale_streamer, reward_prod_id)
    # reward (sorted by context id)
    repeat_rewards_df = pd.melt(repeat_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward')\
                            .loc[reward_cust_id[:10]].reset_index() # 需截斷至10(因為是重複的)
    # for coverage calculation
    repeat_reward = pd.melt(repeat_reward_pivot, ignore_index=False, var_name='商品id', value_name='reward').reset_index()\
                        .drop_duplicates().reset_index(drop=True)
    # measurements
    regrets = {}
    scores_idx = {}
    highest_idxs = {}
    rec_results = {}
    coverages = {}
    diversity = {}
    hr = {}
    hr_per_round = {}
    topk = {}
    
    for exp in experiments:
        regrets[exp], scores_idx[exp], highest_idxs[exp], hr_per_round[exp] = policy_generation(exp, cust_num, prod_num, repeat_streamer_product, repeat_rewards_df, z_dim, weight_eps)    
        rec_results[exp] = preprocess.recommendation_results_df(highest_idxs[exp], reward_cust_id, reward_prod_id)        
        coverages[exp] = preprocess.coverage_list(repeat_reward, rec_results[exp])
        diversity[exp] = metrics.cal_diversity(prod_num, rec_results[exp].rec_id)
        hr[exp], k_list, topk[exp] = metrics.cal_hit_ratio(cust_num, repeat_rewards_df, scores_idx[exp], min_val=10, max_val=110, step=10) # 因為只有10個商品所以要縮小topK範圍
    
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
    
    return regrets, hr, coverages, diversity, repeat_reward_pivot, cust_id, reward_cust_id, reward_prod_id, hr_per_round

# if __name__ == '__main__':
#     main()