# 层的实现


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
    """ReLU激活函数"""
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        out = x.copy()
        out[x<=0] = 0
        return out
    
    def backward(self, dout):
        dout[dout<=0] = 0
        dx = dout
        return dx