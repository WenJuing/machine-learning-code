import numpy as np
from common import *
from layer import *
from collections import OrderedDict


class MultiLayerNet:
    """可拓展多层神经网络"""
    def __init__(self, input_size, hidden_size_list, output_size, activation_function='relu', weight_init_std='relu', 
                 use_batchnorm=False, weight_decay_lambda=0, use_dropout=False, dropout_ration=0.5):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.outputsize = output_size
        self.weight_init_std = weight_init_std
        self.use_batchnorm = use_batchnorm
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.params = {}
        
        # 初始化权重
        self._init_weight(weight_init_std)
        
        # 生成层
        self.layers = OrderedDict()
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        for i in range(1, len(hidden_size_list)+1):  # 生成隐藏层
            self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
            
            if self.use_batchnorm:
                self.params['gamma'+str(i)] = 1
                self.params['beta'+str(i)] = 0
                self.layers['BatchNorm'+str(i)] = BatchNormalization(self.params['gamma'+str(i)], self.params['beta'+str(i)])
                
            self.layers['Activation_function'+str(i)] = activation_layer[activation_function]()
            
            if self.use_dropout:
                self.layers['Dropout'+str(i)] = Dropout(dropout_ration)
            
        i += 1
        self.layers['Affine'+str(i)] = Affine(self.params['W'+str(i)], self.params['b'+str(i)])
        self.lastLayer = SoftmaxWithLoss()
        
    def _init_weight(self, weight_init_std):
        """初始化权重"""
        all_szie_list = [self.input_size] + self.hidden_size_list + [self.outputsize]
        for i in range(1, len(all_szie_list)):
            # 根据激活函数选择尺度
            if weight_init_std.lower() in ('relu', 'he'):
                scale = np.sqrt(2 / all_szie_list[i-1])
            else:
                scale = np.sqrt(1 / all_szie_list[i-1])
            
            self.params['W'+str(i)] = scale * np.random.randn(all_szie_list[i-1], all_szie_list[i])
            self.params['b'+str(i)] = np.zeros(all_szie_list[i])
            
    def predict(self, x, train_fig=True):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_fig)
            else:
                x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        # 权值衰减
        weight_decay = 0
        for i in range(1, len(self.hidden_size_list)+2):
            W = self.params['W'+str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.lastLayer.forward(y, t) + weight_decay
    
    def accuracy(self, x, t):
        y = self.predict(x)
        accuracy = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / x.shape[0]
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t)
        dloss = 1
        dout = self.lastLayer.backward(dloss)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        for i in range(1, len(self.hidden_size_list)+2):
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW + self.weight_decay_lambda * self.layers['Affine'+str(i)].W
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db
        
            if self.use_batchnorm and i != len(self.hidden_size_list)+1:
                grads['gamma'+str(i)] = self.layers['BatchNorm'+str(i)].dgamma
                grads['beta'+str(i)] = self.layers['BatchNorm'+str(i)].dbeta
        return grads    
        