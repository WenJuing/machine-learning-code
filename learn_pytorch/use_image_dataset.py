# 使用图片数据
from torchvision.datasets import FashionMNIST, ImageFolder
import torchvision.transforms as transforms
import torch.utils.data as Data


def get_FashionMNIST_loader(train=True):
    # 读取数据，参数(路径,使用训练集或测试集,数据预处理,下载数据)
    # 数据预处理：1)将形状从[H,W,C]转换成[C,H,W] 2)将元素值归一化
    data = FashionMNIST(root="./data/FashionMNIST", train=train, transform=transforms.ToTensor(), download=False)
    data_loader = Data.DataLoader(dataset=data, batch_size=10000, shuffle=True, num_workers=2)
    return data_loader

def get_object_loader():
    # 多个transforms组合使用
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),     # 随机裁剪大小为(224,224)
                                     transforms.RandomHorizontalFlip(),    # 依概率p=0.5水平翻转
                                     transforms.ToTensor()])        # 转换为张量并归一化
    data = ImageFolder("./data/object", transform=data_transforms)
    data_loader = Data.DataLoader(dataset=data, batch_size=21, shuffle=True, num_workers=1)
    return data_loader
    
if __name__ == '__main__':
    data_loader = get_FashionMNIST_loader(train=False)  # 使用测试集
    for batch_x, batch_y in data_loader:
        print(batch_x.shape)    # (10000, 1, 28, 28)
        print(batch_y.shape)    # (10000)
        
    data_loader = get_object_loader()
    for batch_x, batch_y in data_loader:
        print(batch_x.shape)    # (21, 3, 224, 224)
        print(batch_y.shape)    # (21)