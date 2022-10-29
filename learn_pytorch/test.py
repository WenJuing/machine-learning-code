from imageio import imopen


import torch.nn as nn
import torch

# 同时改变张量的全部值
a = torch.empty(1, 3)
print(a)    # [0,0,0]
a.data.fill_(1)             # 方法一
print(a)    # [1,1,1]
nn.init.constant(a, val=2)  # 方法二
print(a)    # [2,2,2]