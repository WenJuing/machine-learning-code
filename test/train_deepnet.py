# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from trainer import Trainer
from common import *

x_train, t_train, x_test, t_test = get_mnist_data(use_cnn=True)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
