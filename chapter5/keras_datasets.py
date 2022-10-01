# 使用keras提供的mnist手写数字数据集
import matplotlib.pyplot as plt
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.subplots_adjust(hspace=0.6, wspace=0.4)     # 调整图片上下和左右间距
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap=plt.cm.gray)     # 第二个参数设置灰度图像
    plt.title('no:%d label:%d' % (i, y_train[i]), fontsize=10)
    
plt.show()