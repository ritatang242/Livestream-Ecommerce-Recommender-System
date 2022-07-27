import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import itertools
import torch
import torch.nn as nn
import torchbnn as bnn
from customized import preprocess
from customized import metrics
from customized.model import rnns
import os
current_path = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomerStreamer():
    '''
    v1. Multi-class Classification: using Streamer Features to predict the next purchased Streamer
    '''

    def __init__(self, txn_data, end_date, streamer, scale_streamer, static_df):
        self.txn_data = txn_data
        self.end_date = end_date
        self.streamer = streamer
        self.scale_streamer = scale_streamer
        self.static_df = static_df    

    def train(self):
        # Hyper-parameters 
        num_epochs = 51
        learning_rate = 0.001
        hidden_size = 128
        num_layers = 1
        num_classes = self.streamer.shape[0]
        input_size = self.streamer.shape[1]
        batch_list = [32]
        t_list = [5, 10, 15] # [i for i in range(5, 11)]
        cs_loss_record = {}
        cs_epoch_loss = {}

        for b,t in itertools.product(batch_list, t_list):
            print("Sequence legth: ", t, "Batch size:", b)
            cs_loss_record[str(t)+" Sequences Training"] = []
            cs_loss_record[str(t)+" Sequences Validation"] = []
            cs_epoch_loss[str(t)+" Sequences Training"] = []
            ### preprocess streamer data ###
            txn_n = preprocess.generate_last_n_txn(self.txn_data, t, self.end_date)        # 找出最後n次交易
            s_seq = preprocess.generate_streamer_seq(txn_n, self.scale_streamer, t)  # input sequences
            print(s_seq.shape)
            labels_k = dict(zip(self.streamer.index.to_list(), [i for i in range(self.streamer.shape[0])])) # user_id as key
            s_labels = preprocess.generate_streamer_targets(txn_n, labels_k) # label sequences' targets
            ### preprocess streamer data ###

            x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(s_seq, s_labels, test_size=0.33, random_state=2022) # split data
            print(x_s_train.shape, x_s_test.shape, y_s_train.shape, y_s_test.shape)
            batch_size = b
            sequence_length = t

            # Read & transform data
            train_dataset = preprocess.StreamerDataset(x_s_train, y_s_train)
            test_dataset = preprocess.StreamerDataset(x_s_test, y_s_test)

            # Load data
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True) # 訓練模型時打亂順序
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False) # 測試集不需要打亂結果也是一樣(因為沒有訓練，是拿已訓練完畢的模型直接產出結果)

            cs_model = rnns.GRU(input_size, hidden_size, num_layers, num_classes).to(device) # (batch, 26, t), 128, 2, 9

            # Loss and optimizer
            loss_funtion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(cs_model.parameters(), lr=learning_rate) 

            # Train the model
            n_total_steps = len(train_loader) # total sample/N 批次
            for epoch in range(num_epochs):
                for i, (inputs, labels) in enumerate(train_loader): # 共 n_total_steps * N 人 -> 迭代 n_total_steps 次, 一批次 N 人
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs, hn = cs_model(inputs) # pytorch用法，網路上說應避免 model.forward(images) 這種寫法
                    loss = loss_funtion(outputs, labels)
                    cs_loss_record[str(t)+" Sequences Training"].append(loss.item())
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % n_total_steps == 0:
                        cs_epoch_loss[str(t)+" Sequences Training"].append(loss.item())
                        if (epoch+1) % 5 == 0:
                            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#             torch.save(cs_model.state_dict(), current_path+'/customized/model/trained_model/customer_streamer/gru_'+str(b)+'_'+str(t)+'.pth')

            # Test the model
            # In test phase, we don't need to compute gradients (for memory efficiency)
            test_preds = []
            test_trues = []
            n_total_steps = len(test_loader)
            with torch.no_grad():
                for epoch in range(num_epochs):
                    n_correct = 0
                    n_samples = 0
                    for i, (inputs, labels) in enumerate(test_loader):
                        inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                        labels = labels.to(device)
                        outputs, hn = cs_model(inputs) # softmax的outputs會吐出num_clasess個數的機率
                        loss = loss_funtion(outputs, labels)
                        if (i+1) % n_total_steps == 0:
                            print (f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')        
                        cs_loss_record[str(t)+" Sequences Validation"].append(loss.item())
                        # max returns (value ,index)
                        _, predicted = torch.max(outputs.data, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()

                        test_preds.extend(predicted.detach().cpu().numpy())
                        test_trues.extend(labels.detach().cpu().numpy())

                    acc = 100.0 * n_correct / n_samples
                    print(classification_report(test_trues, test_preds))
                    print(f'Testing Accuracy: {acc:.4f} %') 
                    test_trues_bin = metrics.label_binarize(test_trues, classes=[i for i in range(num_classes)])
                    test_preds_bin = metrics.label_binarize(test_preds, classes=[i for i in range(num_classes)])
                    metrics.get_roc_auc(test_trues_bin, test_preds_bin, num_classes, t)

        return cs_loss_record, cs_epoch_loss, test_trues_bin, test_preds_bin

    def test(self, sequence_length=15, batch_size=32):

        hidden_size = 128
        num_layers = 1
        num_classes = self.streamer.shape[0]
        input_size = self.streamer.shape[1]

        streamer_preds = []
        streamer_trues = []
        streamer_hn = []
        model_path = current_path+'/customized/model/trained_model/customer_streamer/gru_'+str(batch_size)+'_'+str(sequence_length)+'.pth'

        # data preprocess
        txn_n = preprocess.generate_last_n_txn(self.txn_data, sequence_length, self.end_date)   
        cust_id = list(txn_n.groupby(['asid']).last().index)  # keep input's cust_id in order
        s_seq = preprocess.generate_streamer_seq(txn_n, self.scale_streamer, sequence_length)  # input sequences       
        labels_k = dict(zip(self.streamer.index.to_list(), [i for i in range(self.streamer.shape[0])])) # user_id as key
        s_labels = preprocess.generate_streamer_targets(txn_n, labels_k) # label sequences' targets
        print(s_seq.shape, s_labels.shape)
        # data preprocess
        dataset = preprocess.StreamerDataset(s_seq, s_labels)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False) # 測試集不需要打亂結果也是一樣(因為沒有訓練，是拿已訓練完畢的模型直接產出結果)
        # model
        streamer_model = rnns.GRU(input_size, hidden_size, num_layers, num_classes)
        streamer_model.load_state_dict(torch.load(model_path))
        streamer_model.to(device)
        streamer_model.eval()

        # predict
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for inputs, labels in data_loader:
                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs, hn = streamer_model(inputs)
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                streamer_preds.extend(predicted.detach().cpu().numpy())
                streamer_trues.extend(labels.detach().cpu().numpy())
                streamer_hn.extend(hn[0].cpu().numpy()) # hn包含所有的h1, h2, ..., hn的內容, hn[0]是最後一個(倒敘放) 
            acc = 100.0 * n_correct / n_samples
            print(f'Finall Accuracy: {acc:.4f} %') 

        # binds prediction with cust_id
        cust_streamer_preds = dict(zip(cust_id, streamer_preds))
        print("matching of # of customers incorrects: ",len(list(set(cust_id).difference(set(self.static_df.asid)))))

        return streamer_trues, streamer_preds, np.array(streamer_hn), cust_streamer_preds, cust_id, txn_n
    
    
class CustomerProduct(CustomerStreamer):
    '''
    v4. Regression: using aggregated pre-trained Bert Embedding by Product Name to predict the next aggregated Embedding
    '''
    
    def __init__(self, txn_data, end_date, static_df):
        self.txn_data = txn_data
        self.end_date = end_date
        self.static_df = static_df

    def train(self):
        
        # Hyper-parameters 
        num_epochs = 11
        learning_rate = 0.001
        hidden_size = 256
        num_layers = 1
        batch_list = [32]
        t_list = [5, 10, 15]
        # threshold = 0.5
        cp_loss_record = {}
        cp_epoch_loss = {}


        for b,t in itertools.product(batch_list, t_list):
            print("Sequence length: ", t, "Batch size: ", b)
            cp_loss_record[str(t)+" Sequences Training"] = []
            cp_loss_record[str(t)+" Sequences Validation"] = []
            cp_epoch_loss[str(t)+" Sequences Training"] = []
            ### preprocess basket data ###
            prod_n = preprocess.generate_last_n_prod(self.txn_data, t, self.end_date)        # 找出最後n次交易
            basket_seq_prods, basket_tar_prods = preprocess.generate_basket_list(prod_n, t)
            basket_seq_emb = preprocess.cal_basket_embedding(basket_seq_prods)
            basket_tar_emb = preprocess.cal_basket_embedding(basket_tar_prods)
            print(basket_seq_emb.shape, basket_tar_emb.shape)
            ### preprocess basket data ###

            x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(basket_seq_emb, basket_tar_emb, test_size=0.33, random_state=2022) 
            print(x_s_train.shape, x_s_test.shape, y_s_train.shape, y_s_test.shape)
            batch_size = b
            sequence_length = t
            input_size = int(basket_seq_emb.shape[1]/t) # 768
            output_dim = basket_tar_emb.shape[1] # 768

            # Read & transform data
            train_dataset = preprocess.BasketDataset(x_s_train, y_s_train)
            test_dataset = preprocess.BasketDataset(x_s_test, y_s_test)

            # Load data
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False)

            cp_model = rnns.GRU(input_size, hidden_size, num_layers, output_dim).to(device) # (5 batch, 25, 5), 128, 2, out_feature_dim

            # Loss and optimizer
            loss_funtion = nn.MSELoss() # Regression
#             loss_funtion = nn.BCEWithLogitsLoss() # BCE + sigmoid
            optimizer = torch.optim.Adam(cp_model.parameters(), lr=learning_rate) 

            # Train the model
            n_total_steps = len(train_loader) # total sample/N 批次

            print("Start training...")
            for epoch in range(num_epochs):
                for i, (inputs, targets) in enumerate(train_loader): # 不是labels而是targets
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    targets = targets.to(device)

                    # Forward pass
                    outputs, hn = cp_model(inputs) # pytorch用法，網路上說應避免 model.forward(images) 這種寫法
                    loss = loss_funtion(outputs, targets)
                    cp_loss_record[str(t)+" Sequences Training"].append(loss.item())

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % n_total_steps == 0:
                        cp_epoch_loss[str(t)+" Sequences Training"].append(loss.item())
                        if (epoch+1) % 5 == 0:
                            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#             torch.save(cp_model.state_dict(), current_path+'/customized/model/trained_model/customer_product/gru_'+str(b)+'_'+str(t)+'.pth')

            # Test the model
            n_total_steps = len(test_loader) # total sample/N 批次
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    targets = targets.to(device)
                    outputs, hn = cp_model(inputs)
                    loss = loss_funtion(outputs, targets)
                    cp_loss_record[str(t)+" Sequences Validation"].append(loss.item())
                    if (i+1) % n_total_steps == 0:
                        print (f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')        
        
        return cp_loss_record, cp_epoch_loss

    
    def test(self, sequence_length=5, batch_size=32):
        
        hidden_size = 256
        num_layers = 1
#         threshold = 0.5

        ### preprocess basket data ###
        prod_n = preprocess.generate_last_n_prod(self.txn_data, sequence_length, self.end_date)        # 找出最後n次交易
        basket_seq_prods, basket_tar_prods = preprocess.generate_basket_list(prod_n, sequence_length)
        cust_id = list(set(basket_seq_prods.asid))
        basket_seq_emb = preprocess.cal_basket_embedding(basket_seq_prods)
        basket_tar_emb = preprocess.cal_basket_embedding(basket_tar_prods)
        print(basket_seq_emb.shape, basket_tar_emb.shape)
        ### preprocess basket data ###

        output_dim = basket_tar_emb.shape[1]
        input_size = int(basket_seq_emb.shape[1]/sequence_length)
        
        basket_preds = []
        basket_trues = []
        basket_hn = []
        loss_record = []
        model_path = current_path+'/customized/model/trained_model/customer_product/gru_'+str(batch_size)+'_'+str(sequence_length)+'.pth'

        dataset = preprocess.BasketDataset(basket_seq_emb, basket_tar_emb)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False) # 測試集不需要打亂結果也是一樣(因為沒有訓練，是拿已訓練完畢的模型直接產出結果)
        # model
        
        basket_model = rnns.GRU(input_size, hidden_size, num_layers, output_dim)
        basket_model.load_state_dict(torch.load(model_path))
        basket_model.to(device)
        basket_model.eval()

        loss_funtion = nn.MSELoss() # Regression

        # predict
        n_total_steps = len(data_loader) # total sample/N 批次
        print(data_loader)
        print(n_total_steps)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                targets = targets.to(device)
                outputs, hn = basket_model(inputs)
                loss = loss_funtion(outputs, targets)
                loss_record.append(loss.item())
                if (i+1) % 100 == 0:
                    print (f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                # print MAE
                basket_preds.extend(outputs.detach().cpu().numpy())
                basket_trues.extend(targets.detach().cpu().numpy())
                basket_hn.extend(hn[0].cpu().numpy())

        # binds prediction with cust_id
        cust_basket_preds = dict(zip(cust_id, basket_preds))
        print("matching of # of customers incorrects: ",len(list(set(cust_id).difference(set(self.static_df.asid)))))
        
        return basket_trues, basket_preds, np.array(basket_hn), cust_basket_preds, cust_id, prod_n
    
    
'''
Old version of Customer-Product model design

v3. Multi-label Classification: using pre-trained Bert Embedding by Product Name (can deal with HUGE dataset, but BAD performance)
v2. Multi-label Classification: using Product Features (suitable for SMALL dataset (1~2 weeks txn data) ONLY)
v1. Multi-label Classification: using Multi-hot Encoding (suitable for SMALL dataset (1~2 weeks txn data) ONLY) 

'''

class CustomerOneProduct(CustomerStreamer):
    '''
    v4-1. Regression: using aggregated pre-trained Bert Embedding by Product Name to predict the next aggregated Embedding
    To verify the performance of CustomerProduct model, we train each product bert (ONE product in a basket)
    '''
    
    def __init__(self, txn_data, end_date, static_df):
        self.txn_data = txn_data
        self.end_date = end_date
        self.static_df = static_df

    def train(self):
        
        # Hyper-parameters 
        num_epochs = 11
        learning_rate = 0.001
        hidden_size = 256
        num_layers = 1
        batch_list = [32]
        t_list = [5, 10, 15]
        # threshold = 0.5
        cp_loss_record = {}
        cp_epoch_loss = {}


        for b,t in itertools.product(batch_list, t_list):
            print("Sequence length: ", t, "Batch size: ", b)
            cp_loss_record[str(t)+" Sequences Training"] = []
            cp_loss_record[str(t)+" Sequences Validation"] = []
            cp_epoch_loss[str(t)+" Sequences Training"] = []
            ### preprocess basket data ###
            prod_n = preprocess.generate_last_n_prod(self.txn_data, t, self.end_date)  # 找出最後n次交易
            single_prod = prod_n.groupby('asid').seq.count().reset_index() # 計算每個藍子有多少商品
            nameList = list(single_prod[single_prod.seq == (t+1)].asid) 
            single_prod = prod_n[prod_n.asid.isin(nameList)] # 找出籃子中只有一個商品的人
            basket_seq_prods, basket_tar_prods = preprocess.generate_basket_list(single_prod, t)
            basket_seq_emb = preprocess.cal_basket_embedding(basket_seq_prods)
            basket_tar_emb = preprocess.cal_basket_embedding(basket_tar_prods)
            print(basket_seq_emb.shape, basket_tar_emb.shape)
            ### preprocess basket data ###

            x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(basket_seq_emb, basket_tar_emb, test_size=0.33, random_state=2022) 
            print(x_s_train.shape, x_s_test.shape, y_s_train.shape, y_s_test.shape)
            batch_size = b
            sequence_length = t
            input_size = int(basket_seq_emb.shape[1]/t) # 768
            output_dim = basket_tar_emb.shape[1] # 768

            # Read & transform data
            train_dataset = preprocess.BasketDataset(x_s_train, y_s_train)
            test_dataset = preprocess.BasketDataset(x_s_test, y_s_test)

            # Load data
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False)

            cp_one_model = rnns.GRU(input_size, hidden_size, num_layers, output_dim).to(device) # (5 batch, 25, 5), 128, 2, out_feature_dim

            # Loss and optimizer
            loss_funtion = nn.MSELoss() # Regression
#             loss_funtion = nn.BCEWithLogitsLoss() # BCE + sigmoid
            optimizer = torch.optim.Adam(cp_one_model.parameters(), lr=learning_rate) 

            # Train the model
            n_total_steps = len(train_loader) # total sample/N 批次

            print("Start training...")
            for epoch in range(num_epochs):
                for i, (inputs, targets) in enumerate(train_loader): # 不是labels而是targets
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    targets = targets.to(device)

                    # Forward pass
                    outputs, hn = cp_one_model(inputs) # pytorch用法，網路上說應避免 model.forward(images) 這種寫法
                    loss = loss_funtion(outputs, targets)
                    cp_loss_record[str(t)+" Sequences Training"].append(loss.item())

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1) % n_total_steps == 0:
                        cp_epoch_loss[str(t)+" Sequences Training"].append(loss.item())
                        if (epoch+1) % 1 == 0:
                            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
#             torch.save(cp_one_model.state_dict(), current_path+'/customized/model/trained_model/customer_product/gru_'+str(b)+'_'+str(t)+'_one.pth')

            # Test the model
            n_total_steps = len(test_loader) # total sample/N 批次
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    targets = targets.to(device)
                    outputs, hn = cp_one_model(inputs)
                    loss = loss_funtion(outputs, targets)
                    cp_loss_record[str(t)+" Sequences Validation"].append(loss.item())
                    if (i+1) % n_total_steps == 0:
                        print (f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')        
        
        return cp_loss_record, cp_epoch_loss

    
    def test(self, sequence_length=5, batch_size=32):
        
        hidden_size = 256
        num_layers = 1
#         threshold = 0.5

        ### preprocess basket data ###
        prod_n = preprocess.generate_last_n_prod(self.txn_data, sequence_length, self.end_date)        # 找出最後n次交易
        single_prod = prod_n.groupby('asid').seq.count().reset_index() # 計算每個藍子有多少商品
        nameList = list(single_prod[single_prod.seq == (sequence_length+1)].asid) 
        single_prod = prod_n[prod_n.asid.isin(nameList)] # 找出籃子中只有一個商品的人
        basket_seq_prods, basket_tar_prods = preprocess.generate_basket_list(single_prod, sequence_length)
        cust_id = list(set(basket_seq_prods.asid))
        basket_seq_emb = preprocess.cal_basket_embedding(basket_seq_prods)
        basket_tar_emb = preprocess.cal_basket_embedding(basket_tar_prods)
        print(basket_seq_emb.shape, basket_tar_emb.shape)
        ### preprocess basket data ###

        output_dim = basket_tar_emb.shape[1]
        input_size = int(basket_seq_emb.shape[1]/sequence_length)
        
        basket_preds = []
        basket_trues = []
        basket_hn = []
        loss_record = []
        model_path = current_path+'/customized/model/trained_model/customer_product/gru_'+str(batch_size)+'_'+str(sequence_length)+'_one.pth'

        dataset = preprocess.BasketDataset(basket_seq_emb, basket_tar_emb)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False) # 測試集不需要打亂結果也是一樣(因為沒有訓練，是拿已訓練完畢的模型直接產出結果)
        # model
        
        product_model = rnns.GRU(input_size, hidden_size, num_layers, output_dim)
        product_model.load_state_dict(torch.load(model_path))
        product_model.to(device)
        product_model.eval()

        loss_funtion = nn.MSELoss() # Regression

        # predict
        n_total_steps = len(data_loader) # total sample/N 批次
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                targets = targets.to(device)
                outputs, hn = product_model(inputs)
                loss = loss_funtion(outputs, targets)
                loss_record.append(loss.item())
                if (i+1) % 10 == 0:
                    print (f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                # print MAE
                basket_preds.extend(outputs.detach().cpu().numpy())
                basket_trues.extend(targets.detach().cpu().numpy())
                basket_hn.extend(hn[0].cpu().numpy())

        # binds prediction with cust_id
        cust_basket_preds = dict(zip(cust_id, basket_preds))
        print("matching of # of customers incorrects: ",len(list(set(cust_id).difference(set(self.static_df.asid)))))
        print("Final loss:", loss_record[-1])
        
        return basket_trues, basket_preds, np.array(basket_hn), cust_basket_preds, cust_id, single_prod


### v3. ###

# num_epochs = 10
# learning_rate = 0.001
# hidden_size = 128
# num_layers = 1
# batch_list = [32, 64]
# t_list = [5, 10, 15]
# threshold = 0.5

# for b,t in itertools.product(batch_list, t_list):
#     print("Sequence length: ", t, "Batch size: ", b)

#     ### preprocess streamer data ###
#     prod_n = preprocess.generate_last_n_prod(sub_txn, t, end_date)        # 找出最後n次交易
#     basket_emb, basket_seq_prods = preprocess.cal_basket_embedding(prod_n, t)
#     b_targets, b_tar_one_hot = preprocess.generate_basket_multilabel_binarizer(prod_n, t)
#     print(basket_emb.shape, b_tar_one_hot.shape)
#     ### preprocess streamer data ###
    
#     x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(basket_emb, b_tar_one_hot, test_size=0.33, random_state=2022) # split data
#     print(x_s_train.shape, x_s_test.shape, y_s_train.shape, y_s_test.shape)
#     batch_size = b
#     sequence_length = t
#     input_size = int(basket_emb.shape[1]/t)
#     num_classes = b_tar_one_hot.shape[1] # 4萬多

#     # Read & transform data
#     train_dataset = preprocess.BasketDataset(x_s_train, y_s_train)
#     test_dataset = preprocess.BasketDataset(x_s_test, y_s_test)

#     # Load data
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                             batch_size=batch_size, 
#                                             shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                             batch_size=batch_size, 
#                                             shuffle=False)

#     cp_model = rnns.GRU(input_size, hidden_size, num_layers, num_classes).to(device) 
#     loss_funtion = nn.BCEWithLogitsLoss() # BCE + sigmoid
#     optimizer = torch.optim.Adam(cp_model.parameters(), lr=learning_rate) 

#     # Train the model
#     n_total_steps = len(train_loader) # total sample/N 批次
#     print("Start training...")
#     for epoch in range(num_epochs):
#         for i, (inputs, labels) in enumerate(train_loader): 
#             inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
#             labels = labels.to(device)
            
#             # Forward pass
#             outputs, hn = cp_model(inputs) # pytorch用法，網路上說應避免 model.forward(images) 這種寫法
#             loss = loss_funtion(outputs, labels)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 500 == 0:
#                 print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
#     torch.save(cp_model.state_dict(), 'trained_model/customer_product/gru_'+str(b)+'_'+str(t)+'.pth')

#     # Test the model
#     test_preds = []
#     test_trues = []
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
#             labels = labels.to(device)
#             outputs, hn = cp_model(inputs)
#             # multi-label prediction
#             # 法一(小數點6後的尾數 因為把[True, True, False, ...]轉float做mean而與 轉int後加總相除 有些微落差)：
# #             print("Accuracy: ", ((outputs > threshold) == labels).float().mean().item()) # Can only calculate the mean of floating types.
#             # 法二(先轉換後，逐個計算)：
#             outputs[outputs >= threshold] = 1
#             outputs[outputs < threshold] = 0

#             test_preds.extend(outputs.detach().cpu().numpy())
#             test_trues.extend(labels.detach().cpu().numpy())
#         acc = metrics.cal_accurracy_detail(answer=np.array(test_trues), prediction=np.array(test_preds)) # 只有算最後一個batch的            
# # check actual & prediction distribution
# unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
# dict(zip(unique, counts)) # 實際上的0,1分佈
# unique, counts = np.unique(outputs.cpu().numpy(), return_counts=True)
# dict(zip(unique, counts)) # 預測的0,1分佈

### v3. ###



### v2. ###

# num_epochs = 10
# learning_rate = 0.001
# input_size = 112
# hidden_size = 128
# num_layers = 1
# batch_list = [32, 64]
# t_list = [5, 10, 15]
# threshold = 0.5

# for b,t in itertools.product(batch_list, t_list):
#     print("Sequence length: ", t)

#     ### preprocess streamer data ###
#     product = generate_prodcut_features(prod)
#     scale_prod = standardize(product)
#     prod_n = generate_last_n_prod(txn, t) # 找出最後n次交易
#     b_seq, rmlist = generate_basket_seq(prod_n, scale_prod, t)
#     b_targets = generate_basket_targets(prod_n, t)
#     print(b_seq.shape, b_targets.shape)
#     ### preprocess streamer data ###
    
#     x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(b_seq, b_targets, test_size=0.33, random_state=2022) # split data

#     batch_size = b
#     sequence_length = t

#     # Read & transform data
#     train_dataset = BasketDataset(x_s_train, y_s_train)
#     test_dataset = BasketDataset(x_s_test, y_s_test)

#     # Load data
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                             batch_size=batch_size, 
#                                             shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                             batch_size=batch_size, 
#                                             shuffle=False)

#     model = Model(input_size, hidden_size, num_layers, b_targets.shape[1]).to(device)
#     loss_funtion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#     # Train the model
#     n_total_steps = len(train_loader) 
#     for epoch in range(num_epochs):
#         for i, (inputs, labels) in enumerate(train_loader): 
#             inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
#             labels = labels.to(device)
            
#             # Forward pass
#             outputs = model(inputs) 
#             loss = loss_funtion(outputs, labels)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 2 == 0:
#                 print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#     # Test the model
#     with torch.no_grad():
#         n_correct = 0
#         n_samples = 0
#         for inputs, labels in test_loader:
#             inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             # multi-label prediction

#             # 法一(小數點6後的尾數 因為把[True, True, False, ...]轉float做mean而與 轉int後加總相除 有些微落差)：
#             # print("Accuracy: ", ((outputs > threshold) == labels).float().mean().item()) # Can only calculate the mean of floating types.
#             # 法二(先轉換後，逐個計算)：
#             outputs[outputs >= threshold] = 1
#             outputs[outputs < threshold] = 0
#             # print("Accuracy: ", (outputs == labels).sum().item()/(labels.shape[0]*labels.shape[1])) # 直接算product_accuracy
        
#         acc = cal_accurracy_detail(answer=labels, prediction=outputs) # 看完整的資訊

### v2. ###




### v1. ###

# t_record2 = []
# batch_record2 = []
# acc_record2 = []
# samples2 = []

# num_epochs = 10
# learning_rate = 0.001
# hidden_size = 128
# num_layers = 1
# batch_list = [32, 64]
# t_list = [5, 10, 15]
# threshold = 0.5

# for b,t in itertools.product(batch_list, t_list):
#     print("Sequence length: ", t)

#     ### preprocess streamer data ###
#     prod_n = preprocess.generate_last_n_prod(sub_txn, t, end_date)        # 找出最後n次交易
#     p_seq, p_targets = preprocess.generate_prod_seq_and_target(prod_n, t)  # input sequences
#     print(p_seq.shape, p_targets.shape)
#     ### preprocess streamer data ###
    
#     x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(p_seq, p_targets, test_size=0.33, random_state=2022) # split data

#     batch_size = b
#     sequence_length = t
#     input_size = p_targets.shape[1] # (p_seq.shape[1]/t).astype('int64') 

#     # Read & transform data
#     train_dataset = preprocess.BasketDataset(x_s_train, y_s_train)
#     test_dataset = preprocess.BasketDataset(x_s_test, y_s_test)

#     # Load data
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                             batch_size=batch_size, 
#                                             shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                             batch_size=batch_size, 
#                                             shuffle=False)

#     model = rnns.GRU(input_size, hidden_size, num_layers, p_targets.shape[1]).to(device) 
#     loss_funtion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#     # Train the model
#     n_total_steps = len(train_loader)
#     for epoch in range(num_epochs):
#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
#             labels = labels.to(device)
            
#             # Forward pass
#             outputs = model(inputs) # pytorch用法，網路上說應避免 model.forward(images) 這種寫法
#             loss = loss_funtion(outputs, labels)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i+1) % 1000 == 0:
#                 print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#     # Test the model
#     with torch.no_grad():
#         n_correct = 0
#         n_samples = 0
#         for inputs, labels in test_loader:
#             inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             # multi-label prediction

#             # 法一(小數點6後的尾數 因為把[True, True, False, ...]轉float做mean而與 轉int後加總相除 有些微落差)：
#             # print("Accuracy: ", ((outputs > threshold) == labels).float().mean().item()) # Can only calculate the mean of floating types.
#             # 法二(先轉換後，逐個計算)：
#             outputs[outputs >= threshold] = 1
#             outputs[outputs < threshold] = 0
#             # print("Accuracy: ", (outputs == labels).sum().item()/(labels.shape[0]*labels.shape[1])) # 直接算product_accuracy
        
#         acc = metrics.cal_accurracy_detail(answer=labels, prediction=outputs) # 看完整的資訊
#         acc_record2.append(acc)
#         batch_record2.append(batch_size)
#         t_record2.append(t)
#         samples2.append(p_seq.shape[0])

# # plot results
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.io as pio
# pio.renderers.default = 'iframe'

# df = pd.DataFrame(dict(
#     t = t_record3,
#     acc = acc_record3,
#     batch_size = batch_record3
# ))

# # Create figure with secondary y-axis
# fig = make_subplots(specs=[[{"secondary_y": True}]])

# # Add traces
# head, tail, mul = 0, len(t_list), 1
# for i in batch_list:
#     fig.add_trace(
#         go.Scatter(x=t_list, y=df.acc[head*tail:tail*mul], name="Batch Size="+str(i), mode='lines+markers'),
#         secondary_y=True,
#     )
#     head += 1
#     mul += 1

# fig.add_trace(
#     go.Bar(x=t_list, y=samples3[:len(t_list)], name="# of Customers"),
#     secondary_y=False,
# )

# # Add figure title
# fig.update_layout(
#     title_text="Customer-Product Model (Features) Performance", 
#     template="plotly",
#     hovermode="x unified"
# )
# # fig.update_traces(mode="markers+lines", hovertemplate=None)

# # Set x-axis title"
# fig.update_xaxes(title_text="t")

# # Set y-axes titles
# fig.update_yaxes(title_text="Number of Customers", secondary_y=False, showgrid=False)
# fig.update_yaxes(title_text="Product Accuracy", secondary_y=True, zeroline=False)

# fig.show()

### v1. ###
