# 自动微分
import torch

x = torch.ones(1, 2, requires_grad=True)    # 开启自动计算梯度（注意：只有浮点数才能计算梯度）
out = x.pow(2).sum()     # z = x*x + y*y
out.backward()  # 反向传播时，out必须为单个元素
print(out)
print(x.grad)   # gradz(1,1) = {2, 2}

print(x.requires_grad)  # 是否开启了自动求导

with torch.no_grad():   # 在停止自动求导的环境进行计算
    print((x+2).requires_grad)
    
x = x.detach()  # 关闭自动求导
print(x.requires_grad)


