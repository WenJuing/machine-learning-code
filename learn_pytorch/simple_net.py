# 简单神经网络的训练
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden(x)
        y = self.regression(x)
        return y
    
def get_data_loader():
    train_x, train_y = load_diabetes(return_X_y=True)
    # 数据标准化处理
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    
    train_x = torch.as_tensor(train_x).float()
    train_y = torch.as_tensor(train_y).float()
    train_data = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=40, shuffle=True, num_workers=1)
    
    return data_loader

    
if __name__ == '__main__':
    data_loader = get_data_loader()
    net = SimpleNet()
    optimizer = SGD(net.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    loss_list = []
    
    # for epoch in range(10):
    for batch_x, batch_y in data_loader:
        y = net.forward(batch_x)
        print(y)
        break