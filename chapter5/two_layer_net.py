# 两层神经网络：用于求参数W
import numpy as np
import common as com


class TwoLayerNet:
    """两层神经网络类"""
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1, output_size))
    
    def predict(self, x):
        """预测"""
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = com.sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = com.softmax(a2)
        
        return y
    
    def loss(self, x, t):
        """计算损失"""
        y = self.predict(x)
        e = com.cross_entropy_error(y, t)
        return e
    
    def accuracy(self, x, t):
        """计算正确率"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / x.shape[0]
        
        return accuracy
    
    def numerical_gradient(self, x, t):
        """计算参数梯度"""
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = com.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = com.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = com.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = com.numerical_gradient(loss_W, self.params['b2'])

        return grads