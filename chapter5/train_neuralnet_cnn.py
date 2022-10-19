# 训练简单CNN神经网络
from trainer import Trainer
from simple_conv_net import SimpleConvNet
from common import *
import numpy as np
import matplotlib.pyplot as plt


x_train, t_train, x_test, t_test = get_mnist_data(use_cnn=True)

network = SimpleConvNet(input_dim=(1,28,28),conv_hyperparam={'filter_num': 30, 'filter_size': 5, 'stride': 1, 'pad': 0},
                        hidden_size=30, output_size=10)

trainer = Trainer(network, x_train, t_train, x_test, t_test, epoch=10, batch_size=100, 
                  optimizer='adam', optimizer_param={'lr': 0.001}, rate_of_datanum_to_acc=0.15)
trainer.train()
network.save_params()

plt.subplot(121)
plt.plot(np.arange(len(trainer.loss_list)), trainer.loss_list)
plt.xlabel("Learning iter")
plt.ylabel("Loss")
plt.subplot(122)
plt.plot(np.arange(len(trainer.train_acc_list)), trainer.train_acc_list, label="训练集")
plt.plot(np.arange(len(trainer.test_acc_list)), trainer.test_acc_list, label="测试集")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()