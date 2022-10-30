from matplotlib import test
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, MNIST
from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 分类数据
def get_FashionMNIST_loader(use_train=True):
    # train  (60000,1,28,28)   x:(28,28)
    # test   (10000,1,28,28)   t:0~9
    data = FashionMNIST('./data/FashionMNIST', train=use_train, transform=transforms.ToTensor(), download=False)
    data_loader = Data.DataLoader(dataset=data, batch_size=10, shuffle=True, num_workers=1)
    
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


if __name__ == '__main__':
    # data_loader = get_boston_loader()
    get_MNIST_loader()