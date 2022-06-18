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

def plot_loss_wt_val(loss_record, x_max=120, x_label='Batches in 32', model_name='cs'):
    col = ['#1f77b4','#ff7f0e'] * 3
    linestyle = sorted(['-', '--', ':'] * 2)
    c = 0
    label_list = []
    for k,v in loss_record.items():  
        label_list.append(k[:-5])
        print(k,":", round(loss_record[k][x_max-1],4)) # print last loss
        plt.plot(range(x_max), loss_record[k][:x_max], c=col[c], ls=linestyle[c], label=k)
        plt.ylabel('Loss')
        plt.xlabel('Number of '+x_label)
        plt.legend(labels=label_list, bbox_to_anchor=(.98, .98), loc=1, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        c+=1
        plt.savefig(fname='fig/loss_'+model_name+'_model.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_loss_wo_val(loss_record, x_max=26, model_name='tr_cs'):
    label_list = []
    for k,v in loss_record.items():  
        label_list.append(k[:-15])
        print(k,":", round(loss_record[k][x_max-1],4))
        plt.plot(range(x_max), loss_record[k][:x_max], label=k)
        plt.ylabel('Training Loss')
        plt.xlabel('Number of Epoches')
        plt.legend(labels=label_list, bbox_to_anchor=(.98, .98), loc=1, borderaxespad=0.)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
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
