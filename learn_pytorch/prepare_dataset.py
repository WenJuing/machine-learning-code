# 准备训练所需的数据集
import torch
import torch.utils.data as Data
from sklearn.datasets import load_diabetes, load_iris


def get_classify_train_loader():
    # 分类数据准备
    # 读取数据  大小    数据类型
    # train_x   (150,4)  float64
    # train_y   (150,)   int32
    train_x, train_y = load_iris(return_X_y=True)
    train_x = torch.as_tensor(train_x).float()  # pytorch浮点数需要float32的数据类型
    train_y = torch.as_tensor(train_y).long()   # 而整数（类别）需要int64的数据类型（与回归数据的唯一区别）

    train_data = Data.TensorDataset(train_x, train_y)  # 将train_x和train_y整合到一起
    train_loader = Data.DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=1)
    
    return train_loader

def get_regress_train_loader():
    # 回归数据准备
    # 读取数据  大小      数据类型
    # train_x   (442,10)  float64
    # train_y   (442,)    float64
    train_x, train_y = load_diabetes(return_X_y=True)
    train_x = torch.as_tensor(train_x).float()
    train_y = torch.as_tensor(train_y).float()
    
    train_data = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=10, shuffle=True, num_workers=1)
    
    return train_loader
    

if __name__ == '__main__':
    train_loader = get_classify_train_loader()
    for i, (batch_x, batch_y) in enumerate(train_loader):
        print(batch_x)
        print(batch_y)
        if i > 0:
            break
