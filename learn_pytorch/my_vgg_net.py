# 基于vgg16改装的CNN网络
import torch.nn as nn
from commom import get_vgg16_feature

class MyVggNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.vgg_feature = get_vgg16_feature()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),  # 第一个Linear的参数通过可打印vgg16的网络结构获得
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),      # dim指定计算方向，0按列计算，1按行计算
        )
    
    def forward(self, x):
        x = self.vgg_feature(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        
        return output