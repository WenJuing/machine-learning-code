import torch
from BiMLP import BiMLP
from torch.optim import Adam
import torch.nn as nn
from commom import get_spambase
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


bimlp = BiMLP(input_features=57)
train_data_loader, x_test, y_test = get_spambase(test_size=0.25)
optimizer = Adam(bimlp.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

print_step = 25

writer = SummaryWriter(log_dir="./data/train_BiMLP_log")

if __name__ == '__main__':
    iter = 0
    all_epoch = 5
    for epoch in range(all_epoch):
        for step, (train_x, train_y) in enumerate(train_data_loader):
            y = bimlp(train_x)
            train_loss = loss_func(y, train_y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            iter = epoch * len(train_data_loader) + step + 1
            
            if iter % print_step == 0:
                y = bimlp(x_test)
                _, y = torch.max(y, 1)
                test_accuracy = accuracy_score(y_test, y)  # accuracy_score(真实标记，预测标记)
                print("iter=",iter,"loss=",train_loss,"test_acc=",test_accuracy)
                
                writer.add_scalar("train loss", train_loss.item(), global_step=iter)
                writer.add_scalar("test accuracy", test_accuracy, global_step=iter)
                