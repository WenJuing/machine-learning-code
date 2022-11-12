import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import numpy as np

# 分类数据
def get_FashionMNIST_loader(use_train=True):
    # train  (60000,1,28,28)   x:(28,28)
    # test   (10000,1,28,28)   t:0~9
    data = FashionMNIST('./data/fashion', train=use_train, transform=transforms.ToTensor(), download=True)
    data_loader = Data.DataLoader(dataset=data, batch_size=64, shuffle=False, num_workers=1)
    
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
    data_loader = get_FashionMNIST_loader()
    show_data(data_loader)
    