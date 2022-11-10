# 一个用于图像分类的卷积神经网络
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, img_size=28):
        super(ConvNet, self).__init__()
        k_size = 3     # 滤波器大小
        p_size = 2     # 池化大小
        s = 1          # 步长
        p = 1          # 填充
        img_size = img_size  # 图像大小
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=k_size, stride=s, padding=p),
            nn.ReLU(),
            nn.AvgPool2d(p_size, p_size),
            nn.Conv2d(16, 32, k_size, s, p),
            nn.ReLU(),
            nn.MaxPool2d(p_size, p_size),
        )
        
        # 计算卷积后的输出大小
        OH1 = ((img_size + 2*p - k_size) / s + 1) / 2
        OH2 = int(((OH1 + 2*p - k_size) / s + 1) / 2)
        
        self.hidden = nn.Sequential(
            nn.Linear(32*OH2*OH2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.regression = nn.Linear(64, 10)  # 10为类别数量
        
    def forward(self, x):
        x = self.conv(x)
        x = self.hidden(x.view(x.shape[0], -1))
        y = self.regression(x)
        
        return y
        