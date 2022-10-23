# 性能优秀的深层CNN网络
from layer import *
from common import *
import pickle

class DeepConvNet:
    """深层CNN网络"""
    def __init__(self, input_dim=(1, 28, 28), conv_params=[
               {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
               {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
               {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
               {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
               {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
               {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}],
               hidden_size=20, output_size=10, use_batchnorm=True):
        print("正在启用深层CNN网络...")
        self.pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        self.weight_init_scales = np.sqrt(2 / self.pre_node_nums)
        # self.weight_init_scales = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_batchnorm = use_batchnorm
        self.conv_params = conv_params
        self.conv_num = len(self.conv_params)    # 卷积层数量
        self.pre_channel_num = input_dim[0]

        self.calcu_layer_indexs = []     # Conv和Affine的层数下标序列
        self._calcu_layer_indexs()
        
        self.params = {}
        self._init_weights()
        
        self.layers = []
        self._init_layers()
        self.last_layer = SoftmaxWithLoss()

    def _init_weights(self):
        """初始化权重"""
        for i, conv_param in enumerate(self.conv_params):   # Conv参数
            self.params['W'+str(i+1)] = self.weight_init_scales[i] * np.random.randn(conv_param['filter_num'], self.pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b'+str(i+1)] = np.zeros(conv_param['filter_num'])
            self.pre_channel_num = conv_param['filter_num']
        # Affine、BatchNorm参数
        self.params['W'+str(self.conv_num+1)] = self.weight_init_scales[self.conv_num] * np.random.randn(self.pre_node_nums[self.conv_num], self.hidden_size)
        self.params['b'+str(self.conv_num+1)] = np.zeros(self.hidden_size)
        if self.use_batchnorm:
            self.params['gamma'+str(self.conv_num+1)] = np.ones(self.hidden_size)
            self.params['beta'+str(self.conv_num+1)] = np.zeros(self.hidden_size)
        self.params['W'+str(self.conv_num+2)] = self.weight_init_scales[self.conv_num+1] * np.random.randn(self.hidden_size, self.output_size)
        self.params['b'+str(self.conv_num+2)] = np.zeros(self.output_size)
        if self.use_batchnorm:
            self.params['gamma'+str(self.conv_num+2)] = np.ones(self.output_size)
            self.params['beta'+str(self.conv_num+2)] = np.zeros(self.output_size)
        print("初始化权重完成...")
        
    def _init_layers(self):
        """初始化层"""
        for i in range(0, self.conv_num, 2):
            self.layers.append(Convolution(self.params['W'+str(i+1)], self.params['b'+str(i+1)], self.conv_params[i]['stride'], self.conv_params[i]['pad']))
            self.layers.append(Relu())
            self.layers.append(Convolution(self.params['W'+str(i+2)], self.params['b'+str(i+2)], self.conv_params[i+1]['stride'], self.conv_params[i+1]['pad']))
            self.layers.append(Relu())
            self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W'+str(self.conv_num+1)], self.params['b'+str(self.conv_num+1)]))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma'+str(self.conv_num+1)], self.params['beta'+str(self.conv_num+1)]))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W'+str(self.conv_num+2)], self.params['b'+str(self.conv_num+2)]))
        if self.use_batchnorm:
            self.layers.append(BatchNormalization(self.params['gamma'+str(self.conv_num+2)], self.params['beta'+str(self.conv_num+2)]))
        self.layers.append(Dropout(0.5))
        print("初始化层完成...")
        
    def _calcu_layer_indexs(self):
        for i in range(0, int(5*self.conv_num/2), 5):
            self.calcu_layer_indexs.append(i)
            self.calcu_layer_indexs.append(i+2)
        i += 2
        self.calcu_layer_indexs.append(i+3)
        if self.use_batchnorm:
            self.calcu_layer_indexs.append(i+7)
        else:
            self.calcu_layer_indexs.append(i+6)

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        y = self.predict(x, train_flg=False)
        acc = np.sum(np.argmax(y, axis=1) == t)

        return acc / x.shape[0]
    
    def gradient(self, x, t):
        """计算梯度"""
        self.loss(x, t)
        
        dloss = 1
        dout = self.last_layer.backward(dloss)
        layers = self.layers.copy()
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        for i, layer_i in enumerate(self.calcu_layer_indexs):
            if isinstance(self.layers[layer_i], Affine):
                grads['W'+str(i+1)] = self.layers[layer_i].dW
                grads['b'+str(i+1)] = self.layers[layer_i].db
                if self.use_batchnorm:
                    grads['gamma'+str(i+1)] = self.layers[layer_i+1].dgamma
                    grads['beta'+str(i+1)] = self.layers[layer_i+1].dbeta
            else:
                grads['W'+str(i+1)] = self.layers[layer_i].dW
                grads['b'+str(i+1)] = self.layers[layer_i].db
                
        return grads
    
    def save_params(self, file_name="deep_conv_net_weight.pkl"):
        """保存权重参数"""
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
            
    def load_params(self, file_name="deep_conv_net_weight.pkl"):
        """加载权重参数"""
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
            
        for i, layer_i in enumerate(self.calcu_layer_indexs):
            self.layers[layer_i].W = self.params['W'+str(i+1)]
            self.layers[layer_i].b = self.params['b'+str(i+1)]
            
        
        
        
        
    
    