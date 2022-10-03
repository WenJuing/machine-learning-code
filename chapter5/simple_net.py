# 一个简单的神经网络类，该神经网络只有输入层和输出层，输入层2个参数，输出层3个神经元
import numpy as np
import common as com


class simpleNet:
    """简单神经网络类"""
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        """类别预测"""
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        """计算损失"""
        a = self.predict(x)
        y = com.softmax(a)
        loss = com.cross_entropy_error(y, t)
        
        return loss

    
if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    t = np.array([2])
    f = lambda w: net.loss(x, t)
    dW = com.numerical_gradient(f, net.W)
    print(dW)