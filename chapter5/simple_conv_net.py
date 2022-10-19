# 简单的CNN网络实现
import pickle
from common import *
from layers import *
import numpy as np
from collections import OrderedDict


class SimpleConvNet:
    """简单CNN网络"""
    def __init__(self, input_dim=(1,28,28), conv_hyperparam={'filter_num': 30, 'filter_size': 5, 'stride': 1, 'pad': 0}, 
                hidden_size=100, output_size=10, weight_init_std=0.01):
        # 默认输入输出数据和滤波器均为方形
        self.C = input_dim[0]
        self.filter_num = conv_hyperparam['filter_num']
        self.filter_H = conv_hyperparam['filter_size']
        self.filter_stride = conv_hyperparam['stride']
        self.filter_pad = conv_hyperparam['pad']
        self.input_H = input_dim[1]
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv_output_H = (self.input_H + 2*self.filter_pad - self.filter_H) / self.filter_stride + 1
        self.pool_output_size = int(self.filter_num * (self.conv_output_H / 2) ** 2)    # 池化层输出的元素个数
        
        # 初始化参数和层
        self.params = {}
        self._init_weight(weight_init_std)
        
        self.layers = OrderedDict()
        self._init_layers()
        self.last_layer = SoftmaxWithLoss()
        
    def _init_weight(self, weight_init_std):
        """初始化权重"""
        self.params['W1'] = weight_init_std * np.random.randn(self.filter_num, self.C, self.filter_H, self.filter_H)
        self.params['b1'] = np.zeros(self.filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(self.pool_output_size, self.hidden_size)
        self.params['b2'] = np.zeros(self.hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(self.hidden_size, self.output_size)
        self.params['b3'] = np.zeros(self.output_size)
        
    def _init_layers(self):
        """初始化层"""
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], self.filter_stride, self.filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        
    def predict(self, x):
        """预测"""
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        """计算损失"""
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        """计算正确率"""
        y = self.predict(x)
        return np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / x.shape[0]
    
    def gradient(self, x, t):
        """计算梯度"""
        self.loss(x, t)
        
        dloss = 1
        dout = self.last_layer.backward(dloss)
        
        layers_list = list(self.layers.values())
        layers_list.reverse()
        for layer in layers_list:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        
        return grads
        
    def save_params(self, file_name="simple_cnn_params_of_mnist.pkl"):
        """保存训练好的参数"""
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="simple_cnn_params_of_mnist.pkl"):
        """读取训练好的参数"""
        with open(file_name, "rb") as f:
            params = pickle.load(f)
            
        for key, val in params.items():
            self.params[key] = val
            
        for i, val in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[val].W = self.params['W'+str(i+1)]
            self.layers[val].b = self.params['b'+str(i+1)]