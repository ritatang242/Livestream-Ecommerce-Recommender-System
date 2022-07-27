import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from .. import preprocess
import os
current_path = os.getcwd()

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=408, z_dim=20):
        super(AutoEncoder, self).__init__()
        self.in_dim = input_dim
        self.z_dim = z_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, z_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, input_dim),
            nn.Tanh()
#             nn.Sigmoid()
        )

    def forward(self, inputs):
        codes = self.encoder(inputs) # latent vector
        decoded = self.decoder(codes) # reconstructed input
        return codes, decoded


    def loss(self, decoded, inputs):
        loss = F.mse_loss(decoded, inputs.view(-1, self.in_dim))
#         loss = nn.MSELoss(decoded, inputs.view(-1, self.in_dim))
        return loss
    


    def mytrain(self, input_full_context, model_name, z_dims=[32, 64]):
        z_dims = z_dims
        num_epochs = 30
        batch_list = [128]
        learning_rate = 0.001
        loss_record = {}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_data, test_data = train_test_split(input_full_context, test_size=0.33, random_state=2022) # split data
        input_dim = train_data.shape[1]
        if type(input_full_context[0][0]) == np.float64:
            print("[Train] change Double to Float")
            train_data = torch.tensor(train_data, dtype=torch.float)
            test_data = torch.tensor(test_data, dtype=torch.float)

        for batch_size, z_dim in itertools.product(batch_list, z_dims):
            print("z-dim:", z_dim, "Batch size: ", batch_size)
            loss_record["Training (z="+str(z_dim)+")"] = []
            loss_record["Validation (z="+str(z_dim)+")"] = []
            

            ae_tr_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                      batch_size=batch_size, 
                                                      shuffle=False)
            ae_ts_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=batch_size, 
                                                      shuffle=False)

            ae_model = AutoEncoder(input_dim, z_dim).to(device)
            optimizer = torch.optim.Adam(ae_model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                for i, x in enumerate(ae_tr_loader):
                    x = x.to(device)
                    codes, decoded = ae_model(x)

                    # loss
                    loss = ae_model.loss(decoded, x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if ((epoch+1) % 5 == 0) & ((i+1) % len(ae_tr_loader) == 0):
                        print("Epoch[{}/{}], Step [{}/{}], Reconstructed Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(ae_tr_loader), loss.item()))
                loss_record["Training (z="+str(z_dim)+")"].append(loss.item())
                
            torch.save(ae_model.state_dict(), current_path+'/customized/model/trained_model/ae/ae_z_'+str(z_dim)+'batch_'+str(batch_size)+'_'+str(model_name)+'.pth')

            with torch.no_grad():
                # 重建
                for epoch in range(num_epochs):
                    for i, x in enumerate(ae_ts_loader):
                        x = x.to(device)
                        codes, decoded = ae_model(x)
                        loss = ae_model.loss(decoded, x)
                    if (epoch+1) % num_epochs == 0:
                        print("[Testing] Step [{}/{}], Reconstructed Loss: {:.4f}".format(i+1, len(ae_ts_loader), loss.item()))
                    loss_record["Validation (z="+str(z_dim)+")"].append(loss.item())

        return loss_record


    def mytest(self, input_full_context, model_name, z_dim=20, batch_size=128):

        input_dim = input_full_context.shape[1]
        num_epochs = 10
        learning_rate = 1e-3
        loss_ae = []

        ae_trues = []
        ae_preds = []
        latent_vector = []

        model_path = current_path+'/customized/model/trained_model/ae/ae_z_'+str(z_dim)+'batch_'+str(batch_size)+'_'+str(model_name)+'.pth'
        if type(input_full_context[0][0]) == np.float64:
            print("[Test] change Double to Float")
            input_full_context = torch.tensor(input_full_context, dtype=torch.float)
            
        data_loader = torch.utils.data.DataLoader(dataset=input_full_context, 
                                                batch_size=batch_size, 
                                                shuffle=False) # 測試集不需要打亂結果也是一樣(因為沒有訓練，是拿已訓練完畢的模型直接產出結果)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ae_model = AutoEncoder(input_dim, z_dim).to(device)
        ae_model.load_state_dict(torch.load(model_path))
        ae_model.to(device)
        ae_model.eval()

        with torch.no_grad():
            for i, x in enumerate(data_loader):
                x = x.to(device).view(-1, input_dim)
                codes, decoded = ae_model(x)
                loss = ae_model.loss(decoded, x)
                loss_ae.append(loss.item())
                if (i+1) % len(data_loader) == 0:
                    print ("Step [{}/{}], Reconstructed Loss: {:.4f}" 
                           .format(i+1, len(data_loader), loss.item()))
                ae_trues.extend(x.detach().cpu().numpy())
                ae_preds.extend(decoded.detach().cpu().numpy())
                latent_vector.extend(codes.detach().cpu().numpy())

        return np.array(ae_trues), np.array(ae_preds), np.array(latent_vector)