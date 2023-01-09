# 可视化类激活热力图
from commom import preprocess_image_to_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torch
import cv2
import numpy as np
import pandas as pd


def hook(g):
    global grads
    grads = g


labels = pd.read_json("./data/imagenet1k_labels.json", typ='series')
im = Image.open("./animous2.jpg")     # [333, 500, 3]
im_input = preprocess_image_to_input(im)
vgg16 = models.vgg16(pretrained=True)
# 正向传播
vgg16.eval()
feature_map = vgg16.features(im_input)  # [1, 512, 7, 7]
output = vgg16.classifier(feature_map.view(1, -1))  # [1, 1000]
# 获得结果中最高分
pre_index = torch.argmax(output).item()
pre_score = output[:, pre_index]
# 获得预测结果和置信度
softmax = torch.nn.Softmax(dim=1)
pre_prob = softmax(output)   # [1, 1000]
pre_class = labels[pre_index]
pre_prob = np.around(torch.max(pre_prob).item(), 3)

feature_map.register_hook(hook)
pre_score.backward()
# [1, 512, 7, 7] -> [1, 512, 1, 1]
pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))  
# 每个通道乘以梯度均值
for i in range(512):
    feature_map[:, i, ...] *= pooled_grads[:, i, ...]
# [1, 512, 7, 7] -> mean -> [1, 7, 7] -> squeeze -> [7, 7]
heatmap = torch.mean(feature_map, dim=1).squeeze()   
heatmap = torch.maximum(heatmap, torch.zeros(1))    # 效果与relu相同
heatmap = heatmap / torch.max(heatmap)  # 归一化
heatmap = heatmap.detach().numpy()
# 展示原始热力图
# plt.matshow(heatmap)    # plt.matshow()可显示矩阵
# plt.show()
h, w = np.array(im).shape[0], np.array(im).shape[1]
heatmap = cv2.resize(heatmap, (w, h))   # 注意，这里rezie的尺寸是反的
# plt.matshow(heatmap)
# plt.show()
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
im_heatmap = heatmap * 0.9 + im     # 0.4是热力图强度因子
im_heatmap = im_heatmap / im_heatmap.max()
# 将热力区域从蓝色改为红色
b, g, r = cv2.split(im_heatmap)
im_heatmap = cv2.merge([r, g, b])
plt.imshow(im_heatmap)
plt.title("预测为 " + pre_class + "  置信度 " + str(pre_prob))
plt.show()