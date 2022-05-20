import torch
import torch.nn as nn
import torchbnn as bnn
# import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### GRU

# Fully connected neural network with one hidden layer
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): # input_size: num_features (items or streamer)
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # -> x needs to be: (batch_size, seq, input_size)

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # fully connected
        self.fc = nn.Linear(hidden_size, num_classes) # many to one

    def forward(self, x): # x會吃images
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) equals to x.shape(0), namely "batch size of the input"
        # Intiate by random weights
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 25, 5), h0: (2, n, 128)

        # Forward propagate GRU
        out, hn = self.gru(x, h0)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 5, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :] # tensor: (batch: all sample, seq: only the last output, hidden_size: all features)
        # out: (n, 128) 此為 nn.Linear(params)(batch, H_in) input 長相

        out = self.fc(out)
        # out: (n, 8)
        return out, hn# return a label predicted from bnn model


class BayesGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): # input_size: num_features (items or streamer)
        super(BayesGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # -> x needs to be: (batch_size, seq, input_size)

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # bnn
        self.bnn = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_size, out_features=num_classes)

    def forward(self, x): # x會吃images
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) equals to x.shape(0), namely "batch size of the input"
        # Intiate by random weights
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 25, 5), h0: (2, n, 128)

        # Forward propagate GRU
        out, hn = self.gru(x, h0)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 5, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :] # tensor: (batch: all sample, seq: only the last output, hidden_size: all features)
        # out: (n, 128) 此為 nn.Linear(params)(batch, H_in) input 長相

        out = self.bnn(out) # input last lstm output to bnn model
        # out: (n, 8)
        return out, hn # return a label predicted from bnn model

### RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): # input_size: num_features (items or streamer)
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        # fully connected
        self.fc = nn.Linear(hidden_size, num_classes) # many to one

    def forward(self, x): # x會吃images
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) equals to x.shape(0), namely "batch size of the input"
        # Intiate by random weights
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 25, 5), h0: (2, n, 128)

        # Forward propagate RNN
        out, hn = self.rnn(x, h0)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 5, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :] # tensor: (batch: all sample, seq: only the last output, hidden_size: all features)
        # out: (n, 128) 此為 nn.Linear(params)(batch, H_in) input 長相

        out = self.fc(out)
        # out: (n, 8)
        return out, hn # return a label predicted from bnn model


class BayesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): # input_size: num_features (items or streamer)
        super(BayesRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # bnn
        self.bnn = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_size, out_features=num_classes)

    def forward(self, x): # x會吃images
        # Set initial hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) equals to x.shape(0), namely "batch size of the input"
        # Intiate by random weights
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 25, 5), h0: (2, n, 128)

        # Forward propagate RNN
        out, hn = self.rnn(x, h0)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 5, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :] # tensor: (batch: all sample, seq: only the last output, hidden_size: all features)
        # out: (n, 128) 此為 nn.Linear(params)(batch, H_in) input 長相

        out = self.bnn(out) # input last lstm output to bnn model
        # out: (n, 8)
        return out, hn # return a label predicted from bnn model


### LSTM

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): # input_size: num_features (items or streamer)
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # -> x needs to be: (batch_size, seq, input_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # fully connected
        self.fc = nn.Linear(hidden_size, num_classes) # many to one

    def forward(self, x): # x會吃images
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) equals to x.shape(0), namely "batch size of the input"
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Intiate by random weights
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 25, 5), h0: (2, n, 128)

        # Forward propagate
        out, (hn,cn) = self.lstm(x, (h0,c0)) # input images to lstm model

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 5, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :] # tensor: (batch: all sample, seq: only the last output, hidden_size: all features)
        # out: (n, 128) 此為 nn.Linear(params)(batch, H_in) input 長相

        out = self.fc(out)
        # out: (n, 8)
        return out, hn # return a label predicted from bnn model


class BayesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes): # input_size: num_features (items or streamer)
        super(BayesLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # -> x needs to be: (batch_size, seq, input_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # bnn
        self.bnn = bnn.BayesLinear(prior_mu=0, prior_sigma=0.05, in_features=hidden_size, out_features=num_classes)

    def forward(self, x): # x會吃images
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # x.size(0) equals to x.shape(0), namely "batch size of the input"
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Intiate by random weights
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 25, 5), h0: (2, n, 128)

        # Forward propagate
        out, (hn,cn) = self.lstm(x, (h0,c0)) # input images to lstm model

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 5, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :] # tensor: (batch: all sample, seq: only the last output, hidden_size: all features)
        # out: (n, 128) 此為 nn.Linear(params)(batch, H_in) input 長相

        out = self.bnn(out) # input last lstm output to bnn model
        # out: (n, 8)
        return out, hn # return a label predicted from bnn model
