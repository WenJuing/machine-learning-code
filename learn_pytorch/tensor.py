# 张量tensor
import torch
import numpy as np


# 创建tensor
x = torch.tensor([1, 2, 3], dtype=torch.float)  # 方法一：指定元素和类型
print(x)
x = torch.zeros(3, 3)   # 方法二：用函数创建
print(x)
x = x.new_ones(3, 3)    # 方法三：基于已存在的tensor创建tensor
print(x)
x = torch.randn_like(x)
print(x)
# 获取张量维度、类型和元素个数
print(x.size(), x.dtype)    # x.shape也可以
print(x.int().dtype)        # 创建后转换类型
print(x.numel())
# 改变张量形状
x = torch.arange(12)
print(x.shape)
print(x.reshape(3, 4).shape)
# 加法操作
x = torch.ones(2, 2)
y = torch.tensor([[1, 2], [3, 4]])
res = torch.empty(2, 2)
print(x)
print(y)
print(x + y)            # 方法一
print(torch.add(x, y, out=res))  # 方法二
print(res)

# 获取元素值
x = torch.arange(12).reshape(3, 4)
print(x)
print(x[:2])    # 切片
print(x[x>5])   # 条件筛选
# 改变大小
x = x.view(12)
print(x)
x = x.view(4, -1)
print(x)
# 使用item()获取value（只限于一个元素）
x = torch.tensor(2)
print(x)
print(x.item())

# tensor和numpy矩阵的互相转换
a = np.random.randn(3, 3)
print(a, a.dtype)
a = torch.as_tensor(a)  # numpy矩阵转tensor
print(a, a.dtype)
a = a.numpy()           # tensor转numpy矩阵
print(a, a.dtype)

# 正态分布生成张量
torch.manual_seed(1)    # 设置种子
a = torch.normal(mean=0.0, std=torch.ones(3, 3))    # 生成个数和形式看mean和std的形状和值
print(a)
a = torch.randn(3, 3)   # 效果同上
print(a)

# 其他生成张量的方法
a = torch.randperm(10)  # 将0~9随机排序后组成tensor
print(a)
a = torch.arange(0, 11, 2)
print(a)
a = torch.linspace(1, 10, 5)
print(a)

# 增加或删除维度
a = torch.arange(12).reshape(2, 6)
print(a.shape)
a = torch.unsqueeze(a, dim=0)   # 添加维度
print(a.shape)
a = torch.squeeze(a)    # 删除所谓维度为1的维度
print(a.shape)
a = torch.arange(3)
print(a)
a = a.repeat(2, 3)  # 复制
print(a)

# 拼接和拆除
a = torch.ones(3, 3)
b = torch.zeros(3, 3)
print(a)
print(b)
print(torch.cat((a, b), dim=0))  # 在0维度（行）上拼接
print(torch.cat((a, b), dim=1))  # 在1维度（列）上拼接
print(torch.chunk(a, 2, dim=0))