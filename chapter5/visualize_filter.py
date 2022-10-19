# 可视化滤波器，观察训练前后滤波器的形态变化
import matplotlib.pyplot as plt
from simple_conv_net import SimpleConvNet
import numpy as np


def show_filter(filter, title="visualize filter"):
    """显示滤波器"""
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i in range(filter.shape[0]):
        ax = fig.add_subplot(int(np.ceil(filter.shape[0]/5)), 5, i+1, xticks=[], yticks=[])
        ax.imshow(filter[i])
        if i == 2:
            plt.title(title)
    plt.show()

network = SimpleConvNet()
# W1_size=(30,1,5,5)
filter = network.params['W1'].reshape(30, 5, 5)
show_filter(filter, title="Visualize filter (Learning before)")

network.load_params("simple_cnn_params_of_mnist.pkl")
filter = network.params['W1'].reshape(30, 5, 5)
show_filter(filter, title="Visualize filter (Learning after)")
