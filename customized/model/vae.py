import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
import itertools
import os
current_path = os.getcwd()

class VAE(nn.Module):
    def __init__(self, in_dim=784, h_dim=400, z_dim=20, weight_eps=1):
        super(VAE, self).__init__()
        self.in_dim = in_dim
        self.weight_eps = weight_eps
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, in_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std) * self.weight_eps
        return mu + eps * std

    def decode(self, z):
        h = torch.tanh(self.fc4(z)) # original for MNIST: relu
        return torch.tanh(self.fc5(h)) # original for MNIST: sigmoid
    
    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.in_dim))
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var, z
    
    def loss(self, x_reconst, x, mu, log_var):
#         BCE = F.binary_cross_entropy(x_reconst, x.view(-1, self.in_dim), reduction='sum')
        MSE = F.mse_loss(x_reconst, x.view(-1, self.in_dim))
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         loss = BCE + KLD
        total_loss = MSE + KLD
        return total_loss, MSE, KLD
    
    
    def mytrain(self, input_full_context, model_name, z_dims=[32, 64]):

        # Hyper-parameters
        vae_h_dim = 256
        z_dims = z_dims
        num_epochs = 30
        batch_list = [64]
        learning_rate = 1e-3
        weight_eps = 1
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

            vae_tr_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                      batch_size=batch_size, 
                                                      shuffle=False)
            vae_ts_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=batch_size, 
                                                      shuffle=False)

            vae_model = VAE(input_dim, vae_h_dim, z_dim, weight_eps).to(device)
            optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                for i, x in enumerate(vae_tr_loader):
                    x = x.to(device).view(-1, input_dim)
                    x_reconst, mu, log_var, z = vae_model(x)

                    # loss
                    loss, reconst_loss, kl_div = vae_model.loss(x_reconst, x, mu, log_var)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if ((epoch+1) % 5 == 0) & ((i+1) % len(vae_tr_loader) == 0):
                        print ("Epoch[{}/{}], Step [{}/{}], Total Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                               .format(epoch+1, num_epochs, i+1, len(vae_tr_loader), loss.item(), reconst_loss.item(), kl_div.item()))
                loss_record["Training (z="+str(z_dim)+")"].append(loss.item())

            torch.save(vae_model.state_dict(), current_path+'/customized/model/trained_model/vae/vae_z_'+str(z_dim)+'batch_'+str(batch_size)+'_'+str(model_name)+'.pth')

            with torch.no_grad():
                for epoch in range(num_epochs):
                    for i, x in enumerate(vae_ts_loader):
                        x = x.to(device).view(-1, input_dim)
                        x_reconst, mu, log_var, z = vae_model(x)
                        # loss
                        loss, reconst_loss, kl_div = vae_model.loss(x_reconst, x, mu, log_var)
        #                 loss_record["Validation (z="+str(z_dim)+")"].appen(loss.item())
                    if (epoch+1) % num_epochs == 0:
                        print ("[Testing] Step [{}/{}], Total Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                               .format(i+1, len(vae_ts_loader), loss.item(), reconst_loss.item(), kl_div.item()))
                    loss_record["Validation (z="+str(z_dim)+")"].append(loss.item())

        return loss_record


    def mytest(self, input_full_context, model_name, z_dim=32, batch_size=64, weight=2):

        print("save results at weight:", weight)

        weight_eps = [2,3,4,5]
        input_dim = input_full_context.shape[1]
        vae_h_dim = 256
        num_epochs = 30
        learning_rate = 1e-3
        loss_vae = []

        vae_trues = []
        vae_preds = []
        blurry_context = []

        model_path = current_path+'/customized/model/trained_model/vae/vae_z_'+str(z_dim)+'batch_'+str(batch_size)+'_'+str(model_name)+'.pth'
        if type(input_full_context[0][0]) == np.float64:
            print("[Test] change Double to Float")
            input_full_context = torch.tensor(input_full_context, dtype=torch.float)
            
        data_loader = torch.utils.data.DataLoader(dataset=input_full_context, 
                                                batch_size=batch_size, 
                                                shuffle=False) # 測試集不需要打亂結果也是一樣(因為沒有訓練，是拿已訓練完畢的模型直接產出結果)
        # model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for w in weight_eps:
            print("testing weight:", w)
            vae_model = VAE(input_dim, vae_h_dim, z_dim, w).to(device)
            vae_model.load_state_dict(torch.load(model_path))
            vae_model.to(device)
            vae_model.eval()


            with torch.no_grad():
                for i, x in enumerate(data_loader):
                    x = x.to(device).view(-1, input_dim)
                    x_reconst, mu, log_var, z = vae_model(x)
                    # loss
                    loss, reconst_loss, kl_div = vae_model.loss(x_reconst, x, mu, log_var)
                    loss_vae.append(loss.item())
                    if (i+1) % len(data_loader) == 0:
                        print ("Step [{}/{}], Total Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                               .format(i+1, len(data_loader), loss.item(), reconst_loss.item(), kl_div.item()))
                    if w == weight:
                        vae_trues.extend(x.detach().cpu().numpy())
                        vae_preds.extend(x_reconst.detach().cpu().numpy())
                        blurry_context.extend(z.detach().cpu().numpy())

        return np.array(vae_trues), np.array(vae_preds), np.array(blurry_context)