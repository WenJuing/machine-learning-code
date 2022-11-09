import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, MNIST
from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 分类数据
def get_FashionMNIST_loader(use_train=True):
    # train  (60000,1,28,28)   x:(28,28)
    # test   (10000,1,28,28)   t:0~9
    data = FashionMNIST('./data/FashionMNIST', train=use_train, transform=transforms.ToTensor(), download=True)
    data_loader = Data.DataLoader(dataset=data, batch_size=64, shuffle=False, num_workers=2)
    
    return data_loader

def get_MNIST_loader():
    # train  (60000,1,28,28)   x:(28,28)
    # test   (10000,1,28,28)   t:0~9
    train_data = MNIST('./data/MNIST', train=True, transform=transforms.ToTensor(), download=False)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    test_data = MNIST('./data/MNIST', train=False, transform=transforms.ToTensor(), download=False)
    test_x = test_data.data.float()
    test_x = torch.unsqueeze(test_x, dim=1)
    test_y = test_data.targets
    
    return data_loader, test_x, test_y

def get_spambase(test_size=0.25):
    # 1 垃圾邮件 1813
    # 0 非垃圾邮件 2788
    spam = np.array(pd.read_csv("./data/spambase.data", header=None))
    # 将数据随机切分为训练集和数据集
    x_train, x_test, y_train, y_test = train_test_split(spam[:,:-1], spam[:,-1], test_size=test_size, random_state=123)
    # 使用最大-最小方法对数据进行归一化
    # scale = MinMaxScaler(feature_range=(0,1))   # 缩放尺度，默认0~1
    # x_train = scale.fit_transform(x_train)      # fit本质求min和max，用过一次后后面transform不用再fit
    # y_train = scale.transform(y_train)
    x_train = torch.as_tensor(x_train).float()
    y_train = torch.as_tensor(y_train).long()
    train_data = Data.TensorDataset(x_train, y_train)
    train_data_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=1)
    
    x_test = torch.as_tensor(x_test).float()
    y_test = torch.as_tensor(y_test).long()

    return train_data_loader, x_test, y_test

# 回归数据
def get_diabetes_loader():
    # train_x   (442,10)  float64
    # train_y   (442,)    float64
    train_x, train_y = load_diabetes(return_X_y=True)
    # 数据标准化处理
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    
    train_x = torch.as_tensor(train_x).float()
    train_y = torch.as_tensor(train_y).float()
    train_data = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    return data_loader

def get_boston_loader():
    # train_x   (506,13)  float64
    # train_y   (506,)    float64
    train_x, train_y = load_boston(return_X_y=True)
    # 数据标准化处理
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    
    train_x = torch.as_tensor(train_x).float()
    train_y = torch.as_tensor(train_y).float()
    train_data = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    return data_loader

def show_data(data_loader):
    for batch_x, batch_y in data_loader:
        break
    batch_size = len(batch_x)
    row = int(np.ceil(batch_size/16))
    batch_x = batch_x.squeeze()
    for i in range(batch_size):
        plt.subplot(row, 16, i+1)
        plt.imshow(batch_x[i], cmap=plt.cm.gray)
        plt.title(batch_y[i].item(), size=9)
        plt.axis("off")
        plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.show()
    
if __name__ == '__main__':
    # data_loader = get_boston_loader()
    # data_loader, test_x, test_y = get_MNIST_loader()
    data_loader = get_FashionMNIST_loader()
    show_data(data_loader)
    # data = get_spambase()

    