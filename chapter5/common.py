# 公共库
import numpy as np


# 函数
def sigmoid(a):
    """隐藏层激活函数"""
    return 1 / (1 + np.exp(-a))


def softmax(a):
    """输出层激活函数"""
    a = a - np.max(a)   # 防止溢出
    return np.exp(a) / np.sum(np.exp(a), axis=1).reshape(a.shape[0], 1)


def mean_squared_error(y, t):
    """均方误差，t为标记向量"""
    return np.sum((y - t)**2) / 2


def cross_entropy_error(y, t):
    """交叉熵误差, t为标记"""
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    # 监督数据是one-hot-vector的情况下，转换为标记向量的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size    


def numerical_diff(f, x):
    """差分近似求导"""
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)


def numerical_gradient(f, x):
    """梯度"""
    if x.ndim == 1:
        x = x.reshape(1, x.size)
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp = x[i, j]
            x[i, j] = tmp + h
            fxh1 = f(x)
            x[i, j] = tmp - h
            fxh2 = f(x)
            grad[i, j] = (fxh1 - fxh2) / (2*h)
            x[i, j] = tmp

    return grad


def loss(x, t, W1):
    """计算损失"""
    a = np.dot(x, W1)
    y = softmax(a)
    loss = mean_squared_error(y, t)
    return loss


# 类
# 更新参数的方法
class SGD:
    """随机梯度下降"""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        """使用梯度更新参数"""
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
        return params
            
            
class Momentum:
    """Momentum方法"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
            
        return params
        
            
class AdaGrad:
    """AdaGrad方法"""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)   # 分母一般加个微小值
            
        return params
            
    