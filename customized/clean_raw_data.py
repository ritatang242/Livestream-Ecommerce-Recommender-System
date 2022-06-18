import pandas as pd
import pickle
import glob
import time
import preprocess

### read data ###
s = time.time()
data = {'prod':'商品資料_202103_202204/', 'txn':'銷售資料_202103_202204/*/', 'cmt':'*/*/'}
end_date = '2022-04-30'
for k,v in data.items():
    globals()[k] = preprocess.read_data(v)
e = time.time()
print(f'spent: {(e-s)/60:.4f} mins')

### clean data ###
# for prod: 統一id
name = ['邦成-自倉（C倉出貨付款全開）', '邦成-自倉(抓單)', '251.TC(貳伍壹潮流店)', '拼鮮水產 足度男直播買賣', '蔥媽媽直播', '工讀生寵物', '279-🛒現貨直播車', '夢工場', 'TR Box寶藏屋：傘的專家、居的職人', '阿清服裝']
code = sorted(list(set(txn.user_id)))
user_dict = dict(zip(name, code))
for k, v in user_dict.items():
    prod['user_id'] = prod['user_id'].str.replace(k, v, regex=False)
    
# 拼鮮水產 足度男
prod.user_id[(prod.user_id == '17614') | (prod.user_id == '27343')] = '27343'
txn.user_id[(txn.user_id == '17614') | (txn.user_id == '27343')] = '27343'
cmt.user_id[(cmt.user_id == '17614') | (cmt.user_id == '27343')] = '27343'
# 公主派對
prod.user_id[(prod.user_id == '10891') | (prod.user_id == '11566')] = '10891'
txn.user_id[(txn.user_id == '10891') | (txn.user_id == '11566')] = '10891'
cmt.user_id[(cmt.user_id == '10891') | (cmt.user_id == '11566') | (cmt.user_id == '11498') | (cmt.user_id == '11600') | (cmt.user_id == '11618') | (cmt.user_id == '11717')] = '10891'
# 刪除不合理的訂單
txn = txn[txn.總金額>0]
txn = txn[txn.asid != 'blank']
# 時間格式
'''
一筆訂單只有一個商品
一個付款單號有多個商品
下單日期(還沒結帳): 首販日
付款單號: 等於一張發票, 湊滿訂單後一起送出
date: 以付款單號上的日期為主(結帳日期)
'''
txn.時間戳記 = pd.to_datetime(txn.時間戳記).dt.time
txn.下單日期 = pd.to_datetime(txn.下單日期)
txn['date'] = txn.付款單號.str[2:8]
txn['date'] = pd.to_datetime(txn.date, format='%y%m%d')
cmt.created_time = pd.to_datetime(cmt.created_time.dt.date)

### for prod: complement product missing data ###
tmp = pd.read_excel('data/商品資料_202103_202204/商品資料_缺失.xlsx', index_col=None, header=0, engine='openpyxl',
                 converters={
                     'id':str,
                     'shipping_fee':bool,
                     'cold_shipping':bool,
                     'ship_alone':bool,
                     'preorder':bool,
                     'cal_shipping_free_excluded':bool
                     }
                )
tmp.iloc[:, [3,4,6]] = tmp.iloc[:, [3,4,6]].astype('int64')
not_in_prod = list(set(txn.商品id).difference(set(prod.id)))
not_in_prod = [x for x in not_in_prod if str(x) != 'nan']
tmp = tmp.merge(txn[txn.商品id.isin(not_in_prod)][['商品id','user_id']].rename(columns={'商品id':'id'}).drop_duplicates('id', keep='first'), how='left', on='id')
prod = pd.concat([prod, tmp]).sort_values(by=['id'])
### save to pickle ###
pd.to_pickle(prod, 'data/prod.pkl')
pd.to_pickle(txn, 'data/txn.pkl')
pd.to_pickle(cmt, 'data/cmt.pkl')