# 使用对数几率回归进行分类
from distutils.command.build_scripts import first_line_re
from turtle import color
from matplotlib import pyplot as plt
import numpy as np



def get_cov(test):  # 每行代表一个样本
    '''计算训练集的协方差矩阵'''
    u = np.mean(test[:,:2],axis=0).reshape(2,1)      # 均值向量
    cov = np.zeros((2, 2))
    for i in range(test.shape[0]):
        x = test[i, :2].reshape(2, 1)
        cov = cov + np.dot(x - u, (x - u).reshape(1, 2))
    return cov


# 给定训练集
N = 80  # 训练集样本个数
test = np.zeros((4, N))  # 训练集test，test第i列为[xi1,xi2,1,yi]，代表一个样本
test[2] = 1
np.random.seed(10)
for i in range(int(N / 2)):  # 类0
    test[0, i] = np.random.randint(4, 8)+np.random.rand()
    test[1, i] = np.random.randint(1, 4)+np.random.rand()
    test[3, i] = 0
np.random.seed(11)
for i in range(int(N / 2), N):  # 类1
    test[0, i] = np.random.randint(2,6)+np.random.rand()
    test[1, i] = np.random.randint(5,8)+np.random.rand()
    test[3, i] = 1
plt.scatter(test[0], test[1], s=10)
test = test.T

# cov0 = np.cov(test[:int(N/2),:2],rowvar=False)    # 类0的协方差矩阵
# cov1 = np.cov(test[int(N/2):N,:2],rowvar=False)   # 类1的协方差矩阵
cov0 = get_cov(test[:int(N/2)])
cov1 = get_cov(test[int(N/2):N])
u0 = np.mean(test[: int(N / 2), :2], axis=0).reshape(2, 1)  # 类0的均值向量
u1 = np.mean(test[int(N / 2):N, :2], axis=0).reshape(2, 1)  # 类1的均值向量
w = np.dot(np.linalg.inv(cov1+cov0), u0 - u1)
print(w)
x1 = 2
x2 = 9
plt.plot([x1,x2],[x1*w[0]+w[1],x2*w[0]+w[1]], color='red')
plt.show()

