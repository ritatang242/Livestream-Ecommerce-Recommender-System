import numpy as np
import pandas as pd
from functools import reduce
import glob
import re
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


class StreamerDataset: # Build pytorch datasets for streamer dataloader
    def __init__(self, data, targets):
        self.n_samples = data.shape[0]
        self.data = torch.tensor(data, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class BertTokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]    
    
class BasketDataset: # Build pytorch datasets for basket dataloader
    def __init__(self, data, targets):
        self.n_samples = data.shape[0]
        self.data = torch.tensor(data, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def read_data(data):
    path = 'data/'
    files = glob.glob(path+data+"*.xlsx")
    li = []

    for f in files:
        if '商品' in data:
            print(f)
            df = pd.read_excel(f, index_col=None, header=0, engine='openpyxl',
                 converters={
                     'id':str,
                     'shipping_fee':bool,
                     'cold_shipping':bool,
                     'ship_alone':bool,
                     'preorder':bool,
                     'cal_shipping_free_excluded':bool
                     }
                )
            df['user_id'] = re.search('[0-9]\/商品資料_(.+?).xlsx', f).group(1)  # 抓取括號中間文字
            df.iloc[:, [3,4,6]] = df.iloc[:, [3,4,6]].astype('int64')
        elif '銷售' in data:
            print(f)
            df = pd.read_excel(f, index_col=None, header=0, engine='openpyxl',
                 converters={
                     'PSID':str,
                     'ASID':str,
                     'USER_ID':str,
                     '商品ID':str,
                     '下單日期':pd.to_datetime,
                     '時間戳記':pd.to_datetime
                     }
                )
            try:
                df.iloc[:, np.r_[11:15, 16:19]] = df.iloc[:, np.r_[11:15, 16:19]].astype('int64')
            except:
                print('there is a NaN or string')
        elif 'comment' in f:
            print(f)
            df = pd.read_excel(f, index_col=None, header=0, engine='openpyxl',
                 converters={
                     'USER ID':str,
                     'VIDEO ID':str,
                     'FROM_ID':str,
                     'MESSAGE_SEQ':np.int64
                     })
            try:
                df.CREATED_TIME = pd.to_datetime(df.CREATED_TIME,utc=True, errors='coerce')
            except:
                print('wrong data format')
        df = df.drop_duplicates()
        df.columns = df.columns.str.replace(' ','_')
        df.columns= df.columns.str.lower()
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    # df.info()
    return df

def generate_static_user_context(txn_data, end_date):
    txn_data = txn_data[txn_data.asid != 'blank']
    # 計算顧客RFM
    rfm = summary_data_from_transaction_data(txn_data, 'asid', '下單日期', '總金額',observation_period_end=end_date, include_first_transaction=True) 
    tmp = txn_data.groupby(['asid']).agg(
        payment_method = ("付款方式", lambda x: pd.Series.mode(x)[0]),
        shipping_method = ("運送方式", lambda x: pd.Series.mode(x)[0]) # 眾數會回傳多個(如果頻率一樣時, 取第一個)
    )
    static_df = rfm.reset_index().merge(tmp.reset_index(), how='left', on='asid') # rfm人可能比tmp還少(有的人算不出rfml), 所以用left_join
    static_df = pd.get_dummies(static_df, columns=['payment_method','shipping_method'])
    return rfm, static_df

def generate_streamer_features(streamer_static_fname, txn_data, prod_data, cmt_data, rfm_data, end_date):
    # 讀取直播主靜態特徵
    streamer_static = pd.read_csv('data/'+str(streamer_static_fname),
                    converters={
                        'user_id':str,
                        })
    # 抓出頻道的買家清單
    usr_asid = txn_data.groupby(['user_id','asid'])['付款單號'].nunique().reset_index(level=['user_id','asid'])
    # 合併
    df = usr_asid.merge(rfm_data, how='left', on='asid')
    # 計算streamer平均顧客RFML
    rfm_df = df.groupby('user_id').agg(
        avg_recency = ('recency', 'mean'),
        avg_frequency = ('frequency', 'mean'),
        avg_monatory = ('monetary_value', 'mean'),
        avg_length = ('T', 'mean')
    )
    # 計算streamer的活躍/沈睡顧客數
    no_exist = df.groupby(['user_id']).agg({
        'frequency': lambda x: sum(x<=np.mean(x)),
    }).reset_index().rename(columns={'frequency': 'no_exist'})

    no_sleep = df.groupby(['user_id']).agg({
        'frequency': lambda x: sum(x>3*np.mean(x)),
    }).reset_index().rename(columns={'frequency': 'no_sleep'})
    # 計算streamer商品數量(隨時間而變,找出首販日)
    tmp = txn_data.groupby(['商品id','user_id']).agg({
        '下單日期': 'first'
    }).rename(columns={'下單日期':'首販日'}).reset_index() #.sort_values(by=['user_id','首販日'])
    no_prod = tmp[tmp.首販日<=end_date].iloc[:,:2].user_id.value_counts().reset_index().rename(columns={'index': 'user_id', 'user_id': 'no_prod'})
    # 計算streamer訂單數量(隨時間而變)
    no_orders = txn_data[txn_data.下單日期<=end_date].groupby('user_id')['付款單號'].count().reset_index().rename(columns={'付款單號': 'no_orders'})
    # 計算streamer每月平均直播次數(不隨時間而變)
#     cmt['month'] = cmt.created_time.dt.month
#     no_video = cmt.groupby(['user_id','month'])['video_id'].nunique().reset_index().rename(columns={'video_id': 'no_video'}).groupby(['user_id']).mean('no_video')
    # 因為3月和5月的留言資料非常少，取平均會被拉低，這邊直接加總/或取單一完整月份(即4月)資料替代
    no_video = cmt_data.groupby('user_id')['video_id'].nunique().reset_index().rename(columns={'video_id': 'no_video'})
    # 計算streamer每月平均直播參與人數(不隨時間而變)
    no_engage_cust = cmt_data.groupby(['user_id','video_id']).from_id.nunique().reset_index(level=['user_id','video_id'])
    no_engage_cust = no_engage_cust.groupby('user_id').from_id.sum().reset_index().rename(columns={'from_id': 'no_engage_cust'})
    # 合併
    dfs = [rfm_df, no_exist, no_sleep, streamer_static, no_prod, no_orders, no_video, no_engage_cust]
    streamer = reduce(lambda left,right: pd.merge(left, right, how='left', on='user_id'), dfs).fillna(0)
    streamer = streamer.set_index(streamer.columns[0])
    return streamer

def generate_prodcut_features(prod_data):
    product = prod_data.drop(columns=['name', 'quantity','kind','cal_shipping_free_excluded'])
    product = pd.get_dummies(product, columns=['category','supplier','seller','user_id']).set_index('id')
    return product

def standardize(data):
    scale_array = StandardScaler().fit_transform(data)
    scale_df = pd.DataFrame(scale_array, columns=data.columns, index=data.index).reset_index()
    return scale_df

# for streamer
def generate_last_n_txn(txn_data, t, end_date):
    txn_data = txn_data[txn_data.下單日期<=end_date]
    txn_data['feedback_score'] = np.where(txn_data.psid.notna(), 1, 0.5) # 有留言+購買:1分、只有購買:0.5分
    per_cust_order = txn_data.groupby(['asid']).付款單號.nunique().reset_index().rename(columns={'付款單號':'no_orders'})
    nameList = per_cust_order[(per_cust_order.no_orders>=t+1) & (per_cust_order.asid != 'blank') ].asid.to_list()
    txn_t = txn_data[txn_data.asid.isin(nameList)].sort_values(['asid','付款單號','下單日期','時間戳記']).drop_duplicates(['asid','付款單號'],keep='last')
    txn_t = txn_t.groupby(['asid']).tail(t+1).sort_index()
    txn_t['seq'] = txn_t.groupby(['asid']).cumcount()+1
    txn_t = txn_t.sort_values(['asid','seq'])
    print(f"付款單號不等於t筆: {sum(txn_t.groupby(['asid']).付款單號.nunique()!=t+1)}") # 驗證有幾個付款單號不等於t筆
    return txn_t[['asid','user_id','feedback_score','seq']]

# for product (one-hot encoding)
def generate_last_n_prod(txn_data, t, end_date):
    txn_data = txn_data[txn_data.下單日期<=end_date]
    per_cust_order = txn_data[txn_data.商品id.notna()].groupby(['asid']).付款單號.nunique().reset_index().rename(columns={'付款單號':'no_orders'}) # 計算每人有幾個unique付款單號
    nameList = per_cust_order[(per_cust_order.no_orders>=t+1) & (per_cust_order.asid != 'blank')].asid.to_list() # 篩選users
    # prod_t = txn_data[txn_data.asid.isin(nameList) & (txn_data.商品id.notna())].sort_values(['asid','付款單號','下單日期','時間戳記','商品id']).drop_duplicates(['asid','付款單號','商品id'],keep='last')
    if nameList != []:
        prod_t = txn_data[txn_data.asid.isin(nameList)].sort_values(['asid','付款單號','下單日期','時間戳記','商品id']).groupby(['asid','付款單號','商品id']).size().reset_index().rename(columns={0:'cnt'}) # 統計每筆付款單購買商品的量
        prod_t['seq'] = prod_t.groupby(['asid'])['付款單號'].transform(lambda s: s.factorize()[0] + 1) # 製作seq
        prod_t = prod_t.sort_values(['asid','付款單號','商品id','seq'])[['asid','付款單號','商品id','cnt','seq']]
        prod_t = prod_t[prod_t.seq.isin([i for i in range(1,t+2)])] # 挑出最後t+1次的商品訂單
        print(f"付款單號不等於t筆: {sum(prod_t.groupby(['asid']).付款單號.nunique()!=(t+1))}") # 驗證有幾個付款單號不等於t+1筆
        return prod_t
    else:
        print(f"No customer has {t+1} sequences in data")

def generate_streamer_seq(last_n_txn, scale_streamer_feature, t):
    # 根據streamer id去找出對應features
    mix_df = last_n_txn[last_n_txn.seq <= t].merge(scale_streamer_feature, how='left', on='user_id')
    # 將各個features乘上每個user當下對該streamer的feedback_score
    s_seq_df = mix_df.iloc[:,4:].multiply(mix_df["feedback_score"], axis="index")
    # 轉ndarray(每t個捆成一組)
    s_seq = s_seq_df.to_numpy().reshape(len(set(mix_df.asid)),-1)
    return s_seq

def generate_prod_seq_and_target(last_n_txn, t):
    dfs = []
    for i in range(1,t+2): # 最後t+1次的商品訂單
        # convert to multi-hot encoding by adding one-hot encoding
        tmp = last_n_txn[last_n_txn.seq == i].pivot_table(index="asid", columns="商品id", values="cnt").reindex(list(set(last_n_txn.商品id)), axis=1).fillna(0)
        multi_hot_shape = tmp.shape
        dfs.append(tmp)
    prod_multi_hot = pd.concat(dfs, axis=1)
    prod_seq = prod_multi_hot.to_numpy().reshape(multi_hot_shape[0], t+1, multi_hot_shape[1])
    data = prod_seq[:,0:t,:].reshape(multi_hot_shape[0], -1)
    targets = prod_seq[:,t,:].reshape(multi_hot_shape[0], -1)
    return data, targets

def generate_basket_seq(last_n_txn, scale_product_feature, t):
    rmlist = list(set(last_n_txn.商品id).difference(set(scale_product_feature.id)))
    print(f'不在商品主檔的商品共有: {len(rmlist)}個')
    # (~last_n_txn.商品id.isin(rmlist)) &
    b_seq_df = last_n_txn[(last_n_txn.seq <= t)].merge(scale_product_feature, how='left', left_on='商品id', right_on='id').fillna(0).groupby(['asid','seq']).mean().reset_index()
    b_seq = b_seq_df.iloc[:,3:].to_numpy().reshape(len(set(b_seq_df.asid)),-1)
    return b_seq #, list(set(b_seq_df.asid)), b_seq_df

def generate_streamer_targets(last_n_txn, label_dict):
    s_labels = last_n_txn.groupby(['asid']).agg({
        'user_id': 'last' # seq == t+1
    }).user_id.map(label_dict).to_numpy()
    return s_labels

def generate_basket_targets(last_n_txn, t):
    # b_targets_df = last_n_txn[(last_n_txn.asid.isin(keep_asid)) & (last_n_txn.seq==t+1)] # 最後一次t+1的購物籃
    b_targets_df = last_n_txn[last_n_txn.seq==t+1] # 最後一次t+1的購物籃
    b_targets = b_targets_df.pivot_table(index='asid', columns='商品id', values='seq', aggfunc=pd.Series.nunique).fillna(0).to_numpy()
    return b_targets

def generate_full_context(static_context, cust_streamer_prediction, cust_prod_prediction, streamer, cust_id):
    labels_v = dict(enumerate(streamer.index.to_list())) # user_id as value
    for k, v in cust_streamer_prediction.items():
        cust_streamer_prediction[k] = labels_v[v]
    temporal_s = pd.Series(cust_streamer_prediction, name='user_id').to_frame().reset_index()\
                    .merge(streamer, how='left', on='user_id').rename(columns={'index':'asid'})
    static_user = static_context[static_context.asid.isin(cust_id)]
    full_context_wt_user_id = static_user.merge(temporal_s, how='left', on='asid').set_index('asid')
    full_context = full_context_wt_user_id.drop(columns=['user_id'])
    cust_prod_prediction = 0
    return full_context, full_context_wt_user_id
