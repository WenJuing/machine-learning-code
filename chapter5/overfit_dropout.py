# 通过Dropout缓解过拟合现象
from multi_layer_net import MultiLayerNet
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from common import *
from trainer import Trainer


(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255  # 样本一维化和归一化
t_train = np.array(tf.one_hot(t_train, 10))     # 转为标记向量
x_test = x_test.reshape(-1, 784).astype('float32') / 255
t_test = np.array(tf.one_hot(t_test, 10))

train_acc_list = []     # 训练集识别正确率
test_acc_list = []      # 测试集识别正确率
dp_train_acc_list = []
dp_test_acc_list = []

# 超参数
iters_num = 10000         # 更新次数
epoch = 10                 # 训练代
learning_rate = 0.01       # 学习率
batch_size = 100         # mini_batch
train_size = x_train.shape[0]
weight_decay_lambda = 0.1   # 权值衰减
use_dropout = True
dropout_ration = 0.15

optimizer = SGD(lr=learning_rate)

# 只用300个数据进行训练和使用7层神经网络，创造过拟合条件
x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100], output_size=10)
dp_network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100], output_size=10, 
                        use_dropout=use_dropout, dropout_ration=dropout_ration)

trainer = Trainer(network, x_train, t_train, x_test, t_test, epoch=epoch, batch_size=batch_size)
trainer.train()
dp_trainer = Trainer(dp_network, x_train, t_train, x_test, t_test, epoch=epoch, batch_size=batch_size)
dp_trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list
dp_train_acc_list, dp_test_acc_list = dp_trainer.train_acc_list, dp_trainer.test_acc_list

plt.subplot(121)
plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="train")
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label="test")
plt.text(len(train_acc_list)-1, train_acc_list[-1], train_acc_list[-1])
plt.text(len(test_acc_list)-1, test_acc_list[-1], test_acc_list[-1])
plt.xlabel("peoch")
plt.ylabel("accuracy")
plt.title("normal network learning")
plt.ylim([0, 1])
plt.subplot(122)
plt.plot(np.arange(len(dp_train_acc_list)), dp_train_acc_list, label="train")
plt.plot(np.arange(len(dp_test_acc_list)), dp_test_acc_list, label="test")
plt.text(len(dp_train_acc_list)-1, dp_train_acc_list[-1], dp_train_acc_list[-1])
plt.text(len(dp_test_acc_list)-1, dp_test_acc_list[-1], dp_test_acc_list[-1])
plt.xlabel("peoch")
plt.ylabel("accuracy")
plt.title("use dropout network learning")
plt.ylim([0, 1])
plt.legend()
plt.show()