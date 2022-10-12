# 寻找最优化的超参数
from cProfile import label
from multi_layer_net import MultiLayerNet
from common import *
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np


x_train, t_train, x_test, t_test = get_mnist_data()
x_train = x_train[:500]
t_train = t_train[:500]
validation_rate = 0.2
x_val, t_val, x_train, t_train = divide_validation_data(x_train, t_train, validation_rate=validation_rate)

optimization_trial = 15    # 尝试次数
result_val = {}
result_train = {}

# 超参数
epoch = 20
mini_batch = 100

for i in range(optimization_trial):
    lr = 10 ** np.random.uniform(-6, -2)
    weight_decay_lambda = 10 ** np.random.uniform(-8, -4)
    
    network = MultiLayerNet(784, [10], 10, weight_decay_lambda=weight_decay_lambda)
    trainer = Trainer(network, x_val, t_val, x_train, t_train, epoch, mini_batch,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    val_acc_list, train_acc_list = trainer.train_acc_list, trainer.test_acc_list
    
    key = "val acc:" + str(val_acc_list[-1]) + " | lr=" + str(lr) + " weight_decay=" + str(weight_decay_lambda)
    result_val[key] = val_acc_list
    result_train[key] = train_acc_list
    print(key)
    
# 按验证集的最后一个正确率进行降序排序
result_val = sorted(result_val.items(), key=lambda x:x[1][-1], reverse=True)
# 绘制结果
i = 0
x = np.arange(len(result_val[0][1]))
print("The hyperparams of top 5 accuracy","-"*50)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
for (key, val_acc_list) in result_val:
    if i < 5:   # 输出正确率前5高的超参数信息
        print(key)
    plt.subplot(3, 5, i+1)
    plt.plot(x, val_acc_list)
    plt.plot(x, result_train[key], "--")
    plt.ylim([0, 1])
    i += 1
plt.show()
