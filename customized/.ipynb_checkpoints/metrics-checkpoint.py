import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import plotly.express as px
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from itertools import cycle
from customized import preprocess

# def get_confusion_matrix(trues, preds, num_classes):
#     labels = [i for i in range(num_classes)]
#     conf_matrix = confusion_matrix(trues, preds, labels)
#     return conf_matrix

# def get_p_r_f1_from_conf_matrix(conf_matrix, num_classes):
#     TP,FP,FN,TN = 0,0,0,0
#     labels = [i for i in range(num_classes)]
#     nums = len(labels)
#     for i in labels:
#         TP += conf_matrix[i, i]
#         FP += (conf_matrix[:i, i].sum() + conf_matrix[i+1:, i].sum())
#         FN += (conf_matrix[i, i+1:].sum() + conf_matrix[i, :i].sum())
#     # print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, total: {sum([TP,FP,FN,TN])}')
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1 = 2 * precision * recall / (precision + recall)
#     return precision, recall, f1

def get_roc_auc(trues, preds, num_classes, t):
    labels = [i for i in range(num_classes)]
    nb_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # print(trues, preds)
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(trues.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
#     lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),color='#1f77b4', linestyle='--')
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='#ff7f0e', linestyle='--')
    colors = cycle(['b','g', 'r', 'c','m', 'y', '#8c564b'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("fig/roc_"+str(t)+".pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.show()

def cal_remaining_metrics(true_bin, preds_bin):  
    print(f"AUC (weighted): {roc_auc_score(true_bin, preds_bin, average = 'weighted'):.4f}")
    print("Precision (micro): %.4f" % precision_score(true_bin, preds_bin, average='micro'))
    print("Recall (micro):    %.4f" % recall_score(true_bin, preds_bin, average='micro'))
    print("F1 score (micro):  %.4f" % f1_score(true_bin, preds_bin, average='micro'), end='\n\n')

# def cal_accurracy_detail(answer, prediction):

#     wrong_single_count = 0
#     wrong_whole_count = 0

#     for i in range(answer.shape[0]):      # n個baskets
#         comparison = answer[i] != prediction[i]
#         if (comparison.sum().item() >= 1):  # 只要有一個猜錯就算全錯
#             wrong_whole_count += 1
#         for j in range(answer.shape[1]):  # m products in 1 basket
#             if prediction[i][j] != answer[i][j]: # single product wrong # 跑m次所以每個都會檢查到
#                 wrong_single_count += 1
#                 #print("wrong_single_count:",wrong_single_count)

#     product_accurracy = 1 - ( wrong_single_count / (answer.shape[0] * answer.shape[1]) ) # 全部n*m products
#     basket_accurracy = 1 - ( wrong_whole_count / answer.shape[0] )

#     print("total %d baskets, total %d wrong guessed products" % (answer.shape[0] , wrong_single_count)) # 全部n籃子, 猜錯幾個商品
#     print("product accurracy : %f" % product_accurracy) # 商品正確率(猜錯幾個商品)
#     print("basket accurracy : %f" % basket_accurracy)   # 購物籃正確率(basket內全部正確)
#     return product_accurracy

def plot_gru_loss_wt_val(loss_record, x_max=120, x_label='Batches in 32', model_name='cs'):
    col = ['#1f77b4','#ff7f0e'] * 3
    linestyle = sorted(['-', '--', ':'] * 2)
    c = 0
    label_list = []
    for k,v in loss_record.items():  
        label_list.append(k)
        print(k,":", round(loss_record[k][x_max-1],4)) # print last loss
        plt.plot(range(x_max), loss_record[k][:x_max], c=col[c], ls=linestyle[c], label=k)
        plt.ylabel('Loss')
        plt.xlabel('Number of '+x_label)
        plt.legend(labels=label_list, bbox_to_anchor=(.98, .98), loc=1, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
        c+=1
        plt.savefig(fname='fig/loss_'+model_name+'_model.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()

def plot_loss_wt_val(loss_record, model_name='cs'):
    col = ['#1f77b4','#ff7f0e'] * 3
    linestyle = sorted(['-', '--', ':'] * 2)
    c = 0
    label_list = []
    for k,v in loss_record.items():  
        label_list.append(k)
        print(k,":", round(loss_record[k][-1],4)) # print last loss
        plt.plot(range(len(v)), loss_record[k], c=col[c], ls=linestyle[c], label=k)
        plt.ylabel('Loss')
        plt.xlabel('Number of Epoches')
        plt.legend(labels=label_list, bbox_to_anchor=(.98, .98), loc=1, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
        c+=1
        plt.savefig(fname='fig/loss_'+model_name+'_model.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    
def plot_loss_wo_val(loss_record, x_max=26, model_name='tr_cs'):
    label_list = []
    for k,v in loss_record.items():  
        label_list.append(k[:-9])
        print(k,":", round(loss_record[k][x_max-1],4))
        plt.plot(range(x_max), loss_record[k][:x_max], label=k)
        plt.ylabel('Training Loss')
        plt.xlabel('Number of Epoches')
        plt.legend(labels=label_list, bbox_to_anchor=(.98, .98), loc=1, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    #     plt.xticks(range(0,20,1))
        plt.savefig(fname='./fig/loss_'+model_name+'_model.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()

def df_for_plot(df, txn_n, prod_n, trim_cust_list, cust_has_ans_list):
    df['is_trimmed_id'] = np.where(df.index.isin(trim_cust_list), 1, 0)
    df['has_ans'] = np.where(df.index.isin(cust_has_ans_list), 1, 0) # 在推薦的商品範圍內
    df = df.join(txn_n[txn_n.seq==max(txn_n.seq)].set_index('asid')).drop(columns=['seq','feedback_score'])
    df['color'] = np.where(df.has_ans == 1, df.user_id, 'Notbuy')
    # for hover product id
    df = df.join(prod_n[prod_n.seq==max(prod_n.seq)].groupby('asid').agg({'商品id' : lambda x: ','.join([str(n) for n in x])}))
    # make product embedding from product name
    id2name = pd.read_pickle('data/id2name.pkl')
    prod_n_wt_name = prod_n[prod_n.seq==max(prod_n.seq)].merge(id2name, how='left', left_on='商品id', right_on='id').drop(columns='id')    
    df = df.join(prod_n_wt_name.groupby('asid').agg({'name' : lambda x: '^^'.join([str(n) for n in x])})).reset_index().rename(columns={'name': 'products_in_basket', 'index': 'asid'})
    return df

def plot_2d_scatter(df, figname, hue, hue_order):
    fig = plt.figure(figsize=(8,8))
    sns.scatterplot(data=df, x='Dim1', y='Dim2', hue=hue, hue_order=hue_order)
    plt.savefig(fname='./fig/'+str(figname)+'.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    
def plot_3d_discrete(df):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    c_order = {"color": list(sorted(set(df.user_id)))+["Notbuy"]}
    fig = px.scatter_3d(df, x='Dim1', y='Dim2', z='Dim3', #symbol='has_ans',
                  color='color', hover_data=['商品id'], opacity=0.7, color_discrete_sequence=colors, category_orders=c_order)
    fig.update_traces(marker_size = 2) # 調整點的大小
    fig.update_layout(legend= {'itemsizing': 'constant'}) # 讓legend不隨著marker size變小
    fig.show()

def generateHex(df):
    df2 = df.copy()
    def convert2rgb(col):
        minimum = min(col)
        maximum = max(col)
        return round((col - minimum)*(255/(maximum-minimum))).astype('int')

    for col in df2[['r','g','b']].columns:
        df2[col] = convert2rgb(df2[col])
        
    basket_hex = []
    for i in range(len(df)):
        hex_value = '#%02X%02X%02X' % (df2.r[i], df2.g[i], df2.b[i])
        basket_hex.append(hex_value)
    df2['basket_hex'] = basket_hex
    return df2    
    
def plot_3d_continuous(df, color):
    fig = px.scatter_3d(df, x='Dim1', y='Dim2', z='Dim3',
                  color=color, hover_data=['商品id'], opacity=0.7, color_continuous_scale=px.colors.sequential.Blues)
    fig.update_traces(marker_size = 2) # 調整點的大小
    fig.update_layout(legend= {'itemsizing': 'constant'}) # 讓legend不隨著marker size變小
    fig.show()
    
def plot_regret(regret_dict, mylables, fig_name, bbox_to_anchor=(0.02, 0.02), loc=3, rounds=2602):
    for k, v in regret_dict.items():
        print(f"{k}'s regret: {np.round(regret_dict[k][-1], 4)}")
        plt.plot(range(rounds), regret_dict[k], ls='-', label=k)
        plt.ylabel('Regret')
        plt.xlabel('Number of Rounds')
        plt.legend(labels=mylables, bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.savefig(fname='./fig/regret_'+str(fig_name)+'.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    
def plot_coverage(coverage_dict, mylables, fig_name, bbox_to_anchor=(.98, 0.02), loc=4):
    for k, v in coverage_dict.items():
        print(f"{k}'s coverage: {np.round(coverage_dict[k][-1], 4)}")
        plt.plot(range(100), coverage_dict[k], ls='-')
        plt.ylabel('Coverage')
        plt.xlabel('Number of Rounds')
        plt.legend(labels=mylables, bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
    plt.savefig(fname='./fig/coverage_'+str(fig_name)+'.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    
def cal_class_weights(reward_pivot):
    total = reward_pivot.shape[0]*reward_pivot.shape[1]
    neg = (reward_pivot==0).sum().sum()
    pos = (reward_pivot==1).sum().sum()
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    print(f'total: {total}, pos: {pos}, neg: {neg}')
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    return weight_for_1    
    
def cal_hit_ratio(rewards_df, scores_idx, min_val=10, max_val=50, step=5):
    cust_num = len(set(rewards_df.asid))
    prod_num = len(set(rewards_df.商品id))
    ground_turth = sum(rewards_df.reward==1)
    hr = []
    k_list = [k for k in range(min_val, max_val, step)]
    for k in k_list:
        idx_for_rewards_df_latent = preprocess.idx_list_at_topK(scores_idx, cust_num=cust_num, prod_num=prod_num, k=k)
        hit = sum(rewards_df.iloc[list(np.ravel(idx_for_rewards_df_latent))].reward==1)
        hr.append(100 * hit/ground_turth)
    return hr, k_list, idx_for_rewards_df_latent

def cal_hit_ratio_mab(rewards_df, predict_id, min_val=10, max_val=50, step=5):
    prod_num = len(set(rewards_df.asid))
    cust_num = len(set(rewards_df.商品id))
    ground_turth = sum(rewards_df.reward==1)
    k_list = [k for k in range(min_val, max_val, step)]
    hr = []
    for k in k_list:
        hit_cnt = 0
        for i in range(cust_num):
            topk = predict_id[i][:k] # 每人有200筆
            topk = [str(i) for i in topk]
            if i == 0:
                per_rewards_df = rewards_df.iloc[:prod_num]    
                hit_cnt += per_rewards_df[per_rewards_df.商品id.isin(topk)].reward.sum()    
            else:
                start_idx = prod_num * i
                end_idx = prod_num * (i+1)
                per_rewards_df = rewards_df.iloc[start_idx:end_idx]    
                hit_cnt += per_rewards_df[per_rewards_df.商品id.isin(topk)].reward.sum()
        hr.append(100 * hit_cnt / ground_turth) # 各個k只要最後一次的hr
    return hr, k_list, topk

def cal_hit_ratio_neuralucb(rewards_df, scores_idx, min_val=10, max_val=50, step=5):
    cust_num = len(set(rewards_df.asid))
    prod_num = len(set(rewards_df.商品id))
    ground_turth = sum(rewards_df.reward==1)
    hr = []
    k_list = [k for k in range(min_val, max_val, step)]
    for k in k_list:
        for i in range(cust_num):
            idx_for_rewards_df_latent = scores_idx[i][:k]
        hit = sum(rewards_df.iloc[idx_for_rewards_df_latent].reward==1)
        hr.append(100 * hit/ground_turth)
    return hr, k_list, idx_for_rewards_df_latent

def plot_hit_ratio(hr_dict, k_list, mylables, fig_name, bbox_to_anchor=(.02, 0.98), loc='upper left'):
    for k, v in hr_dict.items():
        print(f"{k}'s hit ratio: {np.round(hr_dict[k][-1], 2)} %")
        plt.plot(k_list, hr_dict[k], ls='-', label=k)
        plt.ylabel('HR@K (%)')
        plt.xlabel('K')
        plt.legend(labels=mylables, bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
    plt.savefig(fname='./fig/hr_'+str(fig_name)+'.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()