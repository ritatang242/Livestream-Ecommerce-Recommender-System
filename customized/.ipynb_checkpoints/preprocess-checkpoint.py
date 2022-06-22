import numpy as np
import pandas as pd
from functools import reduce
import glob
import re
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset
import pickle

'''
一筆訂單只有一個商品
一個付款單號有多個商品
下單日期(還沒結帳): 首販日
付款單號: 等於一張發票, 湊滿訂單後一起送出
date: 以付款單號上的日期為主(結帳日期)
'''

#### for cs/cp models ####

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
    rfm = summary_data_from_transaction_data(txn_data, 'asid', 'date', '總金額',observation_period_end=end_date, include_first_transaction=True) 
    # 類別變數
    tmp = txn_data.groupby(['asid']).agg(
        payment_method = ("付款方式", lambda x: pd.Series.mode(x)[0]),
        shipping_method = ("運送方式", lambda x: pd.Series.mode(x)[0]) # 眾數會回傳多個(如果頻率一樣時, 取第一個)
    )
    static_df = rfm.reset_index().merge(tmp.reset_index(), how='left', on='asid') # rfm人可能比tmp還少(有的人算不出rfml), 所以用left_join
    address = pd.read_csv('data/user_address.csv', index_col=0, dtype={'asid': str, 'city': str, 'area': str})
    address = address[address.asid != 'blank']
    static_df = static_df.merge(address, how='left', on='asid')
    static_df = pd.get_dummies(static_df, columns=['payment_method','shipping_method','city','area'])
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
    no_orders = txn_data[txn_data.date<=end_date].groupby('user_id')['付款單號'].count().reset_index().rename(columns={'付款單號': 'no_orders'})
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
    txn_data = txn_data[txn_data.date<=end_date]
    txn_data['feedback_score'] = np.where(txn_data.psid.notna(), 1, 0.5) # 有留言+購買:1分、只有購買:0.5分
    per_cust_order = txn_data[txn_data.商品id.notna()].groupby(['asid']).付款單號.nunique().reset_index().rename(columns={'付款單號':'no_orders'})
    nameList = per_cust_order[(per_cust_order.no_orders>=(t+1)) & (per_cust_order.asid != 'blank') ].asid.to_list()
    txn_t = txn_data[txn_data.asid.isin(nameList)].sort_values(['asid','付款單號','date','時間戳記']).drop_duplicates(['asid','付款單號'],keep='last')
    txn_t = txn_t.groupby(['asid']).tail(t+1).sort_index().sort_values('date')
    txn_t['seq'] = txn_t.groupby(['asid']).cumcount()+1
    txn_t = txn_t.sort_values(['asid','seq'])
    print(f"付款單號不等於t筆: {sum(txn_t.groupby(['asid']).付款單號.nunique()!=t+1)}") # 驗證有幾個付款單號不等於t筆
    return txn_t[['asid','user_id','feedback_score','seq','date']]

# for product (one-hot encoding)
def generate_last_n_prod(txn_data, t, end_date):
    txn_data = txn_data[txn_data.date<=end_date]
    per_cust_order = txn_data[txn_data.商品id.notna()].groupby(['asid']).付款單號.nunique().reset_index().rename(columns={'付款單號':'no_orders'}) # 計算每人有幾個unique付款單號
    nameList = per_cust_order[(per_cust_order.no_orders>=t+1) & (per_cust_order.asid != 'blank')].asid.to_list() # 篩選users
    if nameList != []:
        prod_t = txn_data[txn_data.asid.isin(nameList)].sort_values(['asid','付款單號','date','時間戳記','商品id']).groupby(['asid','付款單號','商品id']).size().reset_index().rename(columns={0:'cnt'}) # 統計每筆付款單購買商品的量
        prod_t['true_seq'] = prod_t.groupby(['asid'])['付款單號'].transform(lambda s: s.factorize()[0] + 1) # 製作seq
        true_seq = (prod_t.groupby(['asid']).max('true_seq')).rename(columns={'true_seq':'max'})
        true_seq['min'] = true_seq['max']-t
        prod_t = prod_t.set_index('asid').join(true_seq[['max','min']])
        prod_t = prod_t[prod_t.true_seq >= prod_t['min']]
        prod_t['seq'] = prod_t.groupby("asid").true_seq.rank("dense")
        print(f"付款單號不等於t筆: {sum(prod_t.groupby(['asid']).付款單號.nunique()!=(t+1))}") # 驗證有幾個付款單號不等於t+1筆
        return prod_t.reset_index()
    else:
        print(f"No customer has {t+1} sequences in data")

def generate_streamer_seq(last_n_txn, scale_streamer_feature, t):
    # 根據streamer id去找出對應features
    mix_df = last_n_txn[last_n_txn.seq <= t].merge(scale_streamer_feature, how='left', on='user_id')
    # 將各個features乘上每個user當下對該streamer的feedback_score
    s_seq_df = mix_df.iloc[:,5:].multiply(mix_df["feedback_score"], axis=0)
    # 轉ndarray(每t個捆成一組)
    s_seq = s_seq_df.to_numpy().reshape(len(set(mix_df.asid)),-1)
    return s_seq

# def generate_prod_seq_and_target(last_n_txn, t):
#     dfs = []
#     for i in range(1,t+2): # 最後t+1次的商品訂單
#         # convert to multi-hot encoding by adding one-hot encoding
#         tmp = last_n_txn[last_n_txn.seq == i].pivot_table(index="asid", columns="商品id", values="cnt").reindex(list(set(last_n_txn.商品id)), axis=1).fillna(0)
#         multi_hot_shape = tmp.shape
#         dfs.append(tmp)
#     prod_multi_hot = pd.concat(dfs, axis=1)
#     prod_seq = prod_multi_hot.to_numpy().reshape(multi_hot_shape[0], t+1, multi_hot_shape[1])
#     data = prod_seq[:,0:t,:].reshape(multi_hot_shape[0], -1)
#     targets = prod_seq[:,t,:].reshape(multi_hot_shape[0], -1)
#     return data, targets

# def generate_basket_seq(last_n_txn, scale_product_feature, t):
#     rmlist = list(set(last_n_txn.商品id).difference(set(scale_product_feature.id)))
#     print(f'不在商品主檔的商品共有: {len(rmlist)}個')
#     # (~last_n_txn.商品id.isin(rmlist)) &
#     b_seq_df = last_n_txn[(last_n_txn.seq <= t)].merge(scale_product_feature, how='left', left_on='商品id', right_on='id').fillna(0).groupby(['asid','seq']).mean().reset_index()
#     b_seq = b_seq_df.iloc[:,3:].to_numpy().reshape(len(set(b_seq_df.asid)),-1)
#     return b_seq #, list(set(b_seq_df.asid)), b_seq_df

def generate_streamer_targets(last_n_txn, label_dict):
    s_labels = last_n_txn.groupby(['asid']).agg({
        'user_id': 'last' # seq == t+1
    }).user_id.map(label_dict).to_numpy()
    return s_labels

# def generate_basket_targets(last_n_txn, t):
#     # b_targets_df = last_n_txn[(last_n_txn.asid.isin(keep_asid)) & (last_n_txn.seq==t+1)] # 最後一次t+1的購物籃
#     b_targets_df = last_n_txn[last_n_txn.seq==t+1] # 最後一次t+1的購物籃
#     b_targets = b_targets_df.pivot_table(index='asid', columns='商品id', values='seq', aggfunc=pd.Series.nunique).fillna(0).to_numpy()
#     return b_targets

def generate_basket_list(prod_n, t):
    id2name = pd.read_pickle('data/id2name.pkl')
    prod_n_wt_name = prod_n.merge(id2name, how='left', left_on='商品id', right_on='id').drop(columns='商品id')    
    basket_prods = prod_n_wt_name.groupby(['asid','seq']).agg({
        'name' : lambda x: '^^'.join([str(n) for n in x]), # concate prod name
    }).reset_index().rename(columns={'name': 'products_in_basket'})  # return df: asid, seq, name_seq
    basket_seq_prods = basket_prods[basket_prods.seq != (t+1)] # remove t+1 answer
    basket_tar_prods = basket_prods[basket_prods.seq == (t+1)] # the t+1 answer
    return basket_seq_prods, basket_tar_prods

def id2embed(prod_list):
    id2name = pd.read_pickle('data/id2name.pkl')
    with open('data/item2vec.pkl', 'rb') as f:
        item2vec = pickle.load(f)
    prod_wt_name = id2name[id2name.id.isin(prod_list)]
    prod_emb = []
    for name in prod_wt_name.name:
        embeddings = item2vec[name]         
        prod_emb.append(embeddings)
    prod_emb = np.array(prod_emb)        
    return prod_emb

def cal_basket_embedding(basket_prods):
    with open('data/item2vec.pkl', 'rb') as f:
        item2vec = pickle.load(f)
        
    basket_emb = []
    for name in basket_prods.products_in_basket:
#         dim = len(list(item2vec.values())[0] # 這個會計算很久, 直接寫死比較快
        prod_emb = np.zeros(shape=(768, )) # (dim, ) # from Bert
        no_prods = 0
        split_name = name.split("^^")

        for i in split_name:
            prod_emb += item2vec[i]          # 根據同一人同個seq加總
            no_prods += 1                    # 一個人,一個seq內有幾個商品
        basket_emb.append(prod_emb/no_prods) # 取平均值來代表一個basket
        
    basket_emb = np.array(basket_emb)
    basket_emb = basket_emb.reshape(len(set(basket_prods.asid)), -1)
    return basket_emb

# def generate_basket_multilabel_binarizer(last_n_txn, t):
#     b_targets_df = last_n_txn[last_n_txn.seq==(t+1)]
#     basket = b_targets_df.groupby(['asid','付款單號']).agg(
#         basket = ('商品id', lambda x: ','.join([str(n) for n in x]))
#     ).basket
#     b_targets = []
#     for b in basket:
#         b = b.split(",")
#         b_targets.append(b)

#     mlb = MultiLabelBinarizer()
#     b_tar_one_hot = mlb.fit_transform(b_targets)
#     return b_targets, b_tar_one_hot, mlb

# def generate_full_context(static_context, cust_streamer_prediction, cust_prod_prediction, streamer, cust_id, cust_id2):
#     labels_v = dict(enumerate(streamer.index.to_list())) # user_id as value
#     for k, v in cust_streamer_prediction.items():
#         cust_streamer_prediction[k] = labels_v[v]
#     temporal_s = pd.Series(cust_streamer_prediction, name='user_id').to_frame().reset_index()\
#                     .merge(streamer, how='left', on='user_id').rename(columns={'index':'asid'})
#     static_user = static_context[static_context.asid.isin(cust_id)]
#     product_seq_df = pd.DataFrame(cust_prod_prediction, index=cust_id2, columns=['pdim_'+str(i+1) for i in range(np.array(cust_prod_prediction).shape[1])]).reset_index().rename(columns={"index":"asid"})
#     dfs = [static_user, temporal_s, product_seq_df]
#     full_context_wt_user_id = reduce(lambda left,right: pd.merge(left, right, how='left', on='asid'), dfs).set_index('asid')
#     full_context = full_context_wt_user_id.drop(columns=['user_id'])
#     return full_context, full_context_wt_user_id


def combine_context(dfs, cust_id):
    full_context_user_id = reduce(lambda left,right: pd.merge(left, right, how='left', on='asid'), dfs)
    full_context_user_id = full_context_user_id[full_context_user_id.asid.isin(cust_id)]
    full_context = full_context_user_id.set_index('asid').to_numpy()
    print(full_context.shape)
    return full_context



#### for cs/cp models ####

#### for product recommendation models ####

def get_rec_prod_list(user_id, txn_data, next_date):
    rec_prod_list_all = txn_data[(txn_data.date==next_date) & (txn_data.商品id.notna())][['user_id','商品id']].drop_duplicates().reset_index(drop = True)
    rec_prod_list = rec_prod_list_all[rec_prod_list_all.user_id==user_id].reset_index(drop = True)
    return rec_prod_list, rec_prod_list_all

def trim_cust_prod_n(t, txn_data, end_date):
    txn_data = txn_data[txn_data.date<=end_date]
    per_cust_order = txn_data[txn_data.商品id.notna()].groupby(['asid']).付款單號.nunique().reset_index().rename(columns={'付款單號':'no_orders'}) # 計算每人有幾個unique付款單號
    trim_cust = txn_data.groupby(['asid']).date.last()
    align_nameList = list(set( per_cust_order[(per_cust_order.no_orders>=t+1) & (per_cust_order.asid != 'blank')].asid )\
                .intersection( set( trim_cust[trim_cust == end_date].index ) )) # 篩選users
    print("訂單大於t+1筆的人數(切齊至t+1筆為同一天end_date):", len(align_nameList))
    prod_t = txn_data[txn_data.asid.isin(align_nameList)].sort_values(['asid','付款單號','date','時間戳記','user_id','商品id']).groupby(['asid','付款單號','date','user_id','商品id']).size().reset_index().rename(columns={0:'cnt'}) # 統計每筆付款單購買商品的量
    prod_t['true_seq'] = prod_t.groupby(['asid'])['付款單號'].transform(lambda s: s.factorize()[0] + 1) # 製作seq
    true_seq = (prod_t.groupby(['asid']).max('true_seq')).rename(columns={'true_seq':'max'})
    true_seq['min'] = true_seq['max']-t
    prod_t = prod_t.set_index('asid').join(true_seq[['max','min']])
    prod_t = prod_t[prod_t.true_seq >= prod_t['min']]
    prod_t['seq'] = prod_t.groupby("asid").true_seq.rank("dense")
    return prod_t.reset_index(), align_nameList

def channel_answer(user_id, rec_prod_channel, txn_data, prod_t, sequence_length, next_date, align_nameList):
    print("=====")
    print(f"頻道 {str(rec_prod_channel.user_id[0])} 在下一場直播中共有 {str(rec_prod_channel.shape[0])} 個商品")

    like = txn_data[txn_data.asid.isin(align_nameList) & txn_data.商品id.isin(rec_prod_channel.商品id)][['date','asid','user_id','商品id']]
    like_prod_id = list(like.商品id.unique())
    like_cust_id = list(like.asid.unique())
    print(f"[一年內有買] 商品數: {len(like_prod_id)}, 人數: {len(like_cust_id)}")

    super_like = prod_t[(prod_t.seq == (sequence_length+1)) & (prod_t.商品id.isin(rec_prod_channel.商品id))]
    super_like_prod_id = list(super_like.商品id.unique())
    super_like_cust_id = list(super_like.asid.unique())
    print(f"[在t+1時有買] 商品數: {len(super_like_prod_id)}, 人數: {len(super_like_cust_id)}")
    
    potential_like = like[like.date >= next_date]
    potential_like_prod_id = list(potential_like.商品id.unique())
    potential_like_cust_id = list(potential_like.asid.unique())
    print(f"[非t+1但半年內有買] 商品數: {len(potential_like_prod_id)}, 人數: {len(potential_like_cust_id)}")
    
    print(f"有 {len(align_nameList)-len(set(super_like.asid).union(set(potential_like.asid)))} 個人沒有答案; {len(set(super_like.asid).union(set(potential_like.asid)))} 人有答案")
    return align_nameList, super_like, super_like_prod_id, super_like_cust_id, like, like_prod_id, like_cust_id, potential_like, potential_like_prod_id, potential_like_cust_id

def generate_reward_df(super_like, potential_like, pr=1):
    super_like['reward'] = 1
    potential_like['reward'] = pr
    reward_df = super_like.merge(potential_like, how='outer', on=['asid', '商品id']).fillna(0)
    reward_df['reward'] = np.where(reward_df.reward_x >= reward_df.reward_y, reward_df.reward_x, reward_df.reward_y)
    reward_df = reward_df.drop_duplicates(['asid','商品id'])[['asid','商品id','reward']]
    return reward_df

def trim_cust_for_context(full_context, cust_id, trim_cust_list):
    df = pd.DataFrame(np.array(full_context), index=cust_id)
    df = df[df.index.isin(trim_cust_list)].sort_index()
    full_cxt_dict = dict(zip(df.index, df.values))
    full_cxt = df.to_numpy()
    full_cxt_id = list(df.index)
    return full_cxt, full_cxt_id, full_cxt_dict

def trim_cust_for_context_sort(full_context, cust_id, trim_cust_list):
    df = pd.DataFrame(np.array(full_context), index=cust_id)
    df = df[df.index.isin(trim_cust_list)]
    df2 = df.reindex(trim_cust_list)
    full_cxt_dict = dict(zip(df2.index, df2.values))
    full_cxt = df2.to_numpy()
    full_cxt_id = list(df2.index)
    return full_cxt, full_cxt_id, full_cxt_dict

#### for product recommendation models ####

#### for measurements ####

def recommendation_results_df(highest_idxs, reward_cust_id, reward_prod_id):
    rec_id = []
    for t in range(100):
        rec_id.append(str(reward_prod_id[highest_idxs[t]]))
    rec_results = pd.DataFrame({'asid': reward_cust_id, 'rec_id': rec_id})
    return rec_results

def coverage_list(repeat_reward, rec_results):
    rec_record = repeat_reward.merge(rec_results, how='left', left_on=['asid','商品id'], right_on=['asid','rec_id']).fillna(0)
    cnt = 0
    coverage = []
    for i in range(rec_record.shape[0]): # 因為是left join可能超過100次
        if (rec_record.reward[i] == 1) & (rec_record.rec_id[i]!=0): # 確實喜歡且有推薦到
            if cnt < 50:
                cnt += 1
            else:
                cnt += 0
            coverage.append( cnt / 50 )
        elif rec_record.rec_id[i]!=0: # 控制在100輪內
            coverage.append( cnt / 50 )
    return coverage

def idx_list_at_topK(scores_idx, cust_num=1000, prod_num=200, k=10):
    idx_for_rewards_df = []
    for i in range(cust_num):
        if i == 0:
            idx = scores_idx[i][:k]
            idx_for_rewards_df.append(idx)
        else:
            start_idx = prod_num * i # 每人有200個商品
            idx = [start_idx+element for element in scores_idx[i][:k]]
            idx_for_rewards_df.append(idx)
    return idx_for_rewards_df


#### for measurements ####


