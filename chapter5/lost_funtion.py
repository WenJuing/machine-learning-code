# 损失函数，常用的有两种：均方误差、交叉熵误差
import numpy as np


def mean_squared_error(y, t):
    """计算均方误差"""
    e = np.sum((y - t) ** 2) / 2
    return e


def cross_entropy_error(y, t):
    """计算单个输出的交叉熵误差"""
    e = -np.sum(t * np.log(y+1e-5))
    return e


def cross_entropy_error_pro(y, t):
    """计算多个输出的交叉熵误差"""
    if y.ndim == 1:     # y为一维时，因为后续有二维索引，故需转为二维
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    e = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-5)) / batch_size    # 这里t以标记表示
    return e

if __name__ == '__main__':
    # 2的标记向量
    t = np.zeros(10)
    t[2] = 1
    
    y1 = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0])  # 预测为2
    y2 = np.array([0.1, 0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0])  # 预测为7 
    y3 = np.vstack((y1, y2))
    print(mean_squared_error(y1, t))
    print(mean_squared_error(y2, t))
    print(cross_entropy_error(y1, t))
    print(cross_entropy_error(y2, t))
    print(cross_entropy_error_pro(y3, np.array(2)))