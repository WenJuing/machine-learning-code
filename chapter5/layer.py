# 层的实现
import numpy as np
import common as com


class MulLayer:
    """乘法层"""
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        """正向传播"""
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):   # dout表示导数
        """反向传播"""
        dx = dout * self.y
        dy = dout * self.x
        
        return dx, dy
    
    
class AddLayer:
    """加法层"""
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y
        return out
    
    def backward(self, dout):   # dout表示导数
        dx = dout
        dy = dout
        return dx, dy
    
    
class Relu:
    """ReLU激活函数层"""
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
    
class Sigmoid:
    """Sigmoid激活函数层"""
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = com.sigmoid(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx
    
    
class Affine:
    """神经网络中的矩阵运算层"""
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
    
    
class SoftmaxWithLoss:
    """Softmax层和交叉熵误差Loss层"""
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = com.softmax(x)
        self.loss = com.cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 传递给前面的层的是单个数据的误差
        
        return dx