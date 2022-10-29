import torch


a = torch.tensor([1,0,2,3,4,5,6])
b = torch.arange(7)
print(torch.eq(a, b))       # 判断元素是否相等
print(torch.equal(a, b))    # 判断是否有相同的形状和元素

print(torch.clamp_max(a, 3))    # 最大值裁剪
print(torch.clamp_min(a, 3))    # 最大值裁剪
print(torch.clamp(a, 1, 5))     # 范围裁剪

a = torch.arange(1, 10).reshape(3, 3)
print(a, a.dtype)
print(torch.t(a))   # 转置
print(torch.matmul(torch.t(a), a))   # 矩阵乘积
print(torch.inverse(a.float()))  # 求逆（只能浮点数）
print(torch.trace(a))    # 求迹

print(torch.max(a))
print(torch.argmax(a))
print(torch.min(a))
print(torch.argmin(a))

a = torch.randperm(10)
print(a)
print(a.sort())     # 排序（默认升序）
a = torch.randint(0, 17, (4, 4))
print(a)
print(a.sort())     # 二维数组排序，对每行进行排序