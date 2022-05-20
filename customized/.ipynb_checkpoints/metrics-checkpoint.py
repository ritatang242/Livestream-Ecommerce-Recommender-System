import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
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

def get_roc_auc(trues, preds, num_classes):
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
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("fig/ROC.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.show()

def cal_accurracy_detail(answer, prediction):

    wrong_single_count = 0
    wrong_whole_count = 0

    for i in range(answer.shape[0]):      # n個baskets
        comparison = answer[i] != prediction[i]
        if (comparison.sum().item() >= 1):  # 只要有一個猜錯就算全錯
            wrong_whole_count += 1
        for j in range(answer.shape[1]):  # m products in 1 basket
            if prediction[i][j] != answer[i][j]: # single product wrong # 跑m次所以每個都會檢查到
                wrong_single_count += 1
                #print("wrong_single_count:",wrong_single_count)

    product_accurracy = 1 - ( wrong_single_count / (answer.shape[0] * answer.shape[1]) ) # 全部n*m products
    basket_accurracy = 1 - ( wrong_whole_count / answer.shape[0] )

    print("total %d baskets, total %d wrong guessed products" % (answer.shape[0] , wrong_single_count)) # 全部n籃子, 猜錯幾個商品
    print("product accurracy : %f" % product_accurracy) # 商品正確率(猜錯幾個商品)
    print("basket accurracy : %f" % basket_accurracy)   # 購物籃正確率(basket內全部正確)
    return product_accurracy