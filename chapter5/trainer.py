# 训练神经网络类
from time import time
import numpy as np
from common import *


class Trainer:
    """训练神经网络类"""
    def __init__(self, network, x_train, t_train, x_test, t_test, epoch=10, batch_size=100, 
                 optimizer='SGD', optimizer_param={'lr': 0.01}, datanum_to_acc=None, verbose=True):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epoch = epoch
        self.batch_size = batch_size
        self.datanum_to_acc = datanum_to_acc
        self.verbose = verbose
        
        self.train_size = x_train.shape[0]
        self.iter_num = self.epoch * self.batch_size
        
        # 更新方式
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'adagrad': AdaGrad, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.current_iter = 0
        self.current_epoch = 0
        
        self.loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
    def train_step(self):
        """训练一次"""
        start = time()
        # 达到一个epoch时，计算并输出一次mini-btach和测试集的正确率
        if self.current_iter % self.batch_size == 0:
            self.current_epoch += 1
            
            # 为了减少计算量，只取一部分进行估算正确率
            if not self.datanum_to_acc is None:
                train_mask = np.random.choice(self.x_train.shape[0], self.datanum_to_acc)
                test_mask = np.random.choice(self.x_test.shape[0], self.datanum_to_acc)
            else:
                train_mask = np.arange(self.x_train.shape[0])
                test_mask = np.arange(self.x_test.shape[0])
                
                
            train_acc = self.network.accuracy(self.x_train[train_mask], self.t_train[train_mask])
            self.train_acc_list.append(train_acc)
            test_acc = self.network.accuracy(self.x_test[test_mask], self.t_test[test_mask])
            self.test_acc_list.append(test_acc)
            
            if self.verbose: print("the", self.current_epoch, "epoch: train acc=", train_acc, "test acc=", test_acc, "="*50)
            
        mask = np.random.choice(self.train_size, self.batch_size)    # 随机生成mini-batch
        x_train_batch = self.x_train[mask]
        t_train_batch = self.t_train[mask]    
        
        grads = self.network.gradient(x_train_batch, t_train_batch) # 计算梯度并更新参数
        self.network.params = self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_train_batch, t_train_batch)  # 计算损失
        self.loss_list.append(loss)
        end = time()
        if self.verbose: print("the", self.current_iter, "time: loss=", loss)
        
        self.current_iter += 1
            
    def train(self):
        """训练神经网络"""
        for i in range(self.iter_num):
            self.train_step()
            
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        if self.verbose: print("The final test acc is", test_acc)
            
    
        
        