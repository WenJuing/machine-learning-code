import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, MNIST
from sklearn.datasets import load_diabetes, load_boston, fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import time
import seaborn as sns

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
    spam = np.array(pd.read_csv("./data/spambase.data", header=None))   # 对于没有表头的数据集，header设置为None
    # 将数据随机切分为训练集和数据集
    x_train, x_test, y_train, y_test = train_test_split(spam[:,:-1], spam[:,-1], test_size=test_size, random_state=123)
    # 使用最大-最小方法对数据进行归一化（数据预处理很重要！！）
    scale = MinMaxScaler(feature_range=(0,1))   # 缩放尺度，默认0~1
    x_train = scale.fit_transform(x_train)      # fit本质求min和max，用过一次后后面transform不用再fit
    x_test = scale.transform(x_test)
    
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


def get_california_loader():
    housedata = fetch_california_housing()
    # X_train.shape: (14448, 8)
    X_train, X_test, y_train, y_test = train_test_split(housedata.data, housedata.target, test_size=0.3, random_state=100)
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    # 查看相关系数热力图
    # df = pd.DataFrame(data=X_train, columns=housedata.feature_names)
    # df['target'] = y_train
    # show_corrcoef(df)
    X_train = torch.as_tensor(X_train).float()
    y_train = torch.as_tensor(y_train).float()
    X_test = torch.as_tensor(X_test).float()
    y_test = torch.as_tensor(y_test).float()
    
    train_data = Data.TensorDataset(X_train, y_train)
    data_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=1)
    
    return data_loader, X_test, y_test
    
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

def show_corrcoef(df):
    """绘制相关系数(correlation coefficient)热力图"""
    datacor = np.corrcoef(df.values, rowvar=0)  # rowvar默认为True，即每一行为一个变量（观测值），这里每一列为一个变量
    datacor = pd.DataFrame(data=datacor, columns=df.columns, index=df.columns)
    
    plt.figure(figsize=(8, 6))
    plt.rcParams['axes.unicode_minus']=False    # 正常显示负号
    ax = sns.heatmap(datacor, square=True, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu", 
                     cbar_kws={"fraction":0.05, "pad": 0.05})
    plt.title("相关系数热力图")
    plt.show()
    
def train_model(model, data_loader, train_rate, loss_function, optimizer, epochs):
    train_batch_num = np.round(len(data_loader) * train_rate)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0
    val_acc = 0
    iter = 0
    
    writer = SummaryWriter()
    
    for epoch in range(epochs):
        # 每个epoch分类训练阶段核评估阶段
        for step, (x_batch, y_batch) in enumerate(data_loader):
            start = time.time() 
            
            if step < train_batch_num:
                model.train()   # 训练模式
                output = model(x_batch)
                train_loss = loss_function(output, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                
                pre = torch.argmax(output, 1)
                train_acc = accuracy_score(y_batch, pre)
                iter += step
                writer.add_scalar("Loss/train", train_loss.item(), iter)
                writer.add_scalar("Accuracy/train", train_acc, iter)
            else:
                model.eval()    # 评估模式
                output = model(x_batch)
                val_loss = loss_function(output, y_batch)
                pre = torch.argmax(output, 1)
                val_acc = accuracy_score(y_batch, pre)
                iter += step
                writer.add_scalar("Loss/test", val_loss.item(), iter)
                writer.add_scalar("Accuracy/test", val_acc, iter)
            end = time.time()
            writer.add_scalar("cost time(s)", end-start, iter)
            print("step:",step+1,"loss=",train_loss,"train acc=",train_acc,"val acc=",val_acc,"cost time=",end-start)
            
        print("epoch:",epoch+1,"/",epochs,"loss=",train_loss,"train acc=",train_acc,"val acc=",val_acc)
        # 每经过一个epoch，判断并选择最大准确率时的模型参数（最优参数）
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_model_wts = copy.deepcopy(model.state_dict())
            
    writer.close()
    # model.load_state_dict(best_model_wts)
    
    return model
        
                
if __name__ == '__main__':
    # data_loader = get_boston_loader()
    # data_loader, test_x, test_y = get_MNIST_loader()
    # data_loader = get_FashionMNIST_loader()
    # show_data(data_loader)
    # data = get_spambase()
    get_california_loader()

    