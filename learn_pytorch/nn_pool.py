## 对图像使用池化操作
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('./image/touxiang1.png', cv2.IMREAD_GRAYSCALE)       # 读取为灰度图像
image = torch.as_tensor(image).reshape(1, 1, *image.shape).float()      # 转化为大小为(1,1,h,w)的张量
# 生成最大池化层和平均池化层
maxpool2d = nn.MaxPool2d(2)
avgpool2d = nn.AvgPool2d(2)
ada_avgpool2d = nn.AdaptiveAvgPool2d(output_size=(100, 100))    # 选定输出大小
# 对图像进行池化操作
image_maxpool_dout = maxpool2d(image)       # 最大值池化
max_dout = image_maxpool_dout.squeeze()
image_avgpool_dout = avgpool2d(image)       # 平均值池化
avg_dout = image_avgpool_dout.squeeze()
image_ada_avgpool_dout = ada_avgpool2d(image)   # 自适应平均值池化
ada_avg_dout = image_ada_avgpool_dout.squeeze()

plt.subplot(221)
plt.imshow(image.squeeze(), cmap=plt.cm.gray)
plt.axis("off")
plt.title("原图(1200×1200)")
plt.subplot(222)
plt.imshow(max_dout, cmap=plt.cm.gray)
plt.axis("off")
plt.title("最大值池化(600×600)")
plt.subplot(223)
plt.imshow(avg_dout, cmap=plt.cm.gray)
plt.axis("off")
plt.title("平均值池化(600×600)")
plt.subplot(224)
plt.imshow(ada_avg_dout, cmap=plt.cm.gray)
plt.axis("off")
plt.title("自适应平均值池化(100×100)")
plt.show()