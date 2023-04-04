import torch
import torch.nn as nn
import torchvision
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(45, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear4(x)
        return x


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.LSTM=nn.LSTM(input_size=20, hidden_size=32,batch_first=True,num_layers=2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(32*431, 4096)


        self.linear2 = nn.Linear(4096, 10)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout()
    def forward(self, x):
        x=self.LSTM(x)[0]
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x = self.linear2(x)

        return x





