# 公共库
import numpy as np
from keras.datasets import mnist
import tensorflow as tf


# 神经网络功能相关函数
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


# 其他函数
def shuffle_dataset(x, t):
    """打乱数据集"""
    random_index = np.random.permutation(x.shape[0])
    return x[random_index], t[random_index]

def get_mnist_data(use_cnn=False):
    """获得mnist数据集"""
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    
    if not use_cnn:
        x_train = x_train.reshape(-1, 784).astype("float32") / 255
        x_test = x_test.reshape(-1, 784).astype("float32") / 255
    else:
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
        
    t_train = np.array(tf.one_hot(t_train, 10))
    t_test = np.array(tf.one_hot(t_test, 10))
    
    return x_train, t_train, x_test, t_test

def divide_validation_data(x, t, validation_rate):
    """将训练集划分成验证集和训练集"""
    data_size = x.shape[0]
    validation_num = int(data_size * validation_rate)
    x, t = shuffle_dataset(x, t)
    x_val = x[:validation_num]
    t_val = t[:validation_num]
    x_train = x[validation_num:]
    t_train = t[validation_num:]
    
    return x_val, t_val, x_train, t_train

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
    
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col的逆运算"""
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

#  optimizer_calss: 更新参数的类（方式）
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
            
    
class Adam:
    """融合了Momentum和AdaGrad的方法(http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
        return params