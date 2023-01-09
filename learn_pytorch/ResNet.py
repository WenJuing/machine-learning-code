""" ResNet复现 """
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Block in ResNet18/34."""
    def __init__(self, in_channel, out_channel, downsample=None):
        super().__init__()
        self.downsample = downsample
        stride1 = 2 if self.downsample is not None else 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        x = self.relu(identity + self.conv(x))
        
        return x


class BottleneckBlock(nn.Module):
    """Block in ResNet50/101/102."""
    def __init__(self, in_channel, out_channel, downsample=None):
        super().__init__()
        self.downsample = downsample
        stride2 = downsample.stride if self.downsample is not None else 1
        channel = int(out_channel / 4)
        # print("[BottleneckBlock] in_channel=",in_channel,"out_channel=",out_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=stride2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # print("x:",x.shape)
        # print("downsample:", (self.downsample is not None))
        # identity = self.downsample(x) if self.downsample is not None else x
        # print("identity:",identity.shape)
        # x2 = self.conv(x)
        # print("x2:",x2.shape)
        # x = self.relu(identity + x2)
        identity = self.downsample(x) if self.downsample is not None else x
        x = self.relu(identity + self.conv(x))
        
        return x


class BasicLayer(nn.Module):
    """Conv2/3/4/5_x, contains repeatedly stacked blocks."""
    def __init__(self, Block, in_channel, block_num):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            # 只有除第一层外的其他层的第一个块需要下采样
            if i == 0:
                if in_channel == 64 and isinstance(Block(1, 1), ResidualBlock):
                    downsample = None
                    blk_in_channel, blk_out_channel = in_channel, in_channel
                if in_channel != 64 and isinstance(Block(1, 1), ResidualBlock):
                    ds_in_channel, scale, ds_stride = int(in_channel/2), 1, 2
                    blk_in_channel, blk_out_channel = int(in_channel/2), in_channel
                if in_channel == 64 and isinstance(Block(1, 1), BottleneckBlock):
                    ds_in_channel, scale, ds_stride = in_channel, 4, 1
                    blk_in_channel, blk_out_channel = in_channel, in_channel * 4
                if in_channel != 64 and isinstance(Block(1, 1), BottleneckBlock):
                    ds_in_channel, scale, ds_stride = in_channel*2, 4, 2
                    blk_in_channel, blk_out_channel = in_channel * 2, in_channel * 4
                # print("[BasicLayer] downsample: in_channel=", ds_in_channel,"out_channel=",in_channel*scale)
                
                if in_channel != 64 or in_channel == 64 and isinstance(Block(1, 1), BottleneckBlock):
                    downsample = nn.Sequential(
                        nn.Conv2d(in_channels=ds_in_channel, out_channels=in_channel * scale, kernel_size=1, stride=ds_stride, bias=False),
                        nn.BatchNorm2d(in_channel * scale))
                    downsample.stride = ds_stride
            else:
                downsample = None
                if isinstance(Block(1, 1), BottleneckBlock):
                    blk_in_channel, blk_out_channel = in_channel * 4, in_channel * 4
                else:
                    blk_in_channel, blk_out_channel = in_channel, in_channel
                
            block = Block(in_channel=blk_in_channel, out_channel=blk_out_channel, downsample=downsample)
            self.blocks.append(block)
            
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x
    

class ResNet(nn.Module):
    """Residual Network
    
    Args:
        Block (ResidualBlock | BottleneckBlock): Block type.
        block_num (tuple(int)): Number of blocks in different layers.
        num_classes (int): Number of classes for classification head. Default: 1000
    """
    def __init__(self, in_channel, Block, block_num, num_classes):
        super().__init__()
        self.features = nn.ModuleList()
        # build first layer
        conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for feature in [conv1, bn1, relu, maxpool]:
            self.features.append(feature)
        # build other layers
        for i_layer in range(4):
            layer = BasicLayer(
                Block=Block,
                in_channel=64 * 2 ** i_layer,    # 64, 128, 256, 512
                block_num=block_num[i_layer])
            self.features.append(layer)
            
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features.append(avgpool)
        
        in_feature = 512 if isinstance(Block(1, 1), ResidualBlock) else 2048
        self.fc = nn.Linear(in_features=in_feature, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x


def ResNet18(in_channel=3, num_classes=1000):
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    model = ResNet(in_channel=in_channel,
                   Block=ResidualBlock, 
                   block_num=(2, 2, 2, 2), 
                   num_classes=num_classes)
    return model


def ResNet34(in_channel=3, num_classes=1000):
    # https://download.pytorch.org/models/resnet34-b627a593.pth
    model = ResNet(in_channel=in_channel,
                   Block=ResidualBlock, 
                   block_num=(3, 4, 6, 3), 
                   num_classes=num_classes)
    return model


def ResNet50(in_channel=3, num_classes=1000):
    # https://download.pytorch.org/models/resnet50-0676ba61.pth
    model = ResNet(in_channel=in_channel,
                   Block=BottleneckBlock, 
                   block_num=(3, 4, 6, 3), 
                   num_classes=num_classes)
    return model


def ResNet101(in_channel=3, num_classes=1000):
    # https://download.pytorch.org/models/resnet101-63fe2227.pth
    model = ResNet(in_channel=in_channel,
                   Block=BottleneckBlock, 
                   block_num=(3, 4, 23, 3), 
                   num_classes=num_classes)
    return model


def ResNet152(in_channel=3, num_classes=1000):
    # https://download.pytorch.org/models/resnet152-394f9c45.pth
    model = ResNet(in_channel=in_channel,
                   Block=BottleneckBlock, 
                   block_num=(3, 8, 36, 3), 
                   num_classes=num_classes)
    return model