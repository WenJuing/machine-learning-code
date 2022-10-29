## 使用卷积操作提取图像轮廓
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('./image/touxiang1.png', cv2.IMREAD_GRAYSCALE)       # 读取为灰度图像
image = torch.as_tensor(image).reshape(1, 1, *image.shape).float()      # 转化为大小为(1,1,h,w)的张量
# 生成卷积核
kersize = 5
ker = torch.ones(kersize, kersize).float() * -1 # 提取边缘卷积核
ker[1, 1] = kersize * kersize - 1
ker = ker.reshape(1, 1, kersize, kersize)
ker2 = torch.ones(kersize, kersize).float() / (kersize * kersize)   # 均值模糊卷积核
ker2 = ker2.reshape(1, 1, kersize, kersize)
# 生成卷积层并添加卷积核
conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)    # 输出两个通道，每个通道应给出使用的卷积核。若无，则为随机卷积核
conv2d.weight.data[0] = ker   # 为通道1添加提取边缘卷积核
conv2d.weight.data[1] = ker2  # 为通道2添加均值模糊卷积核
# 对图像进行卷积操作
image_conv_dout = conv2d(image)     # 注意，卷积运算需要为浮点数
dout = image_conv_dout.squeeze().detach()    # 压缩维度并停止自动梯度（卷积层有自动梯度，池化层没有自动梯度）

plt.subplot(131)
plt.imshow(image.squeeze().detach(), cmap=plt.cm.gray)  # plt可直接输出tensor
plt.axis("off")
plt.title("原图")
plt.subplot(132)
plt.imshow(dout[0], cmap=plt.cm.gray)
plt.axis("off")
plt.title("提取边缘卷积")
plt.subplot(133)
plt.imshow(dout[1], cmap=plt.cm.gray)
plt.axis("off")
plt.title("均值模糊卷积")
plt.show()