# from re import A
# import matplotlib.pyplot as plt
from pickle import TRUE
from time import time
import numpy as np
# from keras.datasets import mnist
# import tensorflow as tf
# from collections import OrderedDict
# from chapter5.two_layer_net_pro import TwoLayerNet

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
# t_train = np.array(tf.one_hot(y_train, 10))
a = {'1005':[3,5,7], '1003':[6,5,1], '1001':[5,2,8]}
a = sorted(a.items(), key=lambda x:x[0])
print(a)
print(a[0][1])