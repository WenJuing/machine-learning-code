# 可视化类激活热力图
from commom import preprocess_image_to_input
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import pandas as pd
import torch
import torch.nn as nn

# 导入imagenet的1000个类别
labels = pd.read_json("./data/imagenet1k_labels.json", typ='series')
vgg16 = models.vgg16(pretrained=True)
im = Image.open("./flower.jpg")
im_input = preprocess_image_to_input(im)
im_pre = vgg16(im_input)
softmax = nn.Softmax(dim=1)
im_prob = softmax(im_pre)
# 获得置信度前5的置信度和对应的label
prob, pre_label = torch.topk(im_prob, 5)   # shape = [batch_size, 5]
for i in range(5):
    print("index:", pre_label[0, i].item(), "| label:", labels[pre_label[0, i].item()], "| probility:", prob[0, i].item())