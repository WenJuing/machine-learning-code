# 训练深层CNN神经网络
from trainer import Trainer
from deep_conv_net import DeepConvNet
from common import *
import numpy as np
import matplotlib.pyplot as plt


x_train, t_train, x_test, t_test = get_mnist_data(use_cnn=True)

network = DeepConvNet(hidden_size=50,use_batchnorm=True)

trainer = Trainer(network, x_train, t_train, x_test, t_test, epoch=10, batch_size=100,
                  optimizer='adam', optimizer_param={'lr': 0.001}, datanum_to_acc=1000)
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