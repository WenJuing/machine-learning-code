import torch
from MLP import MLP
from torch.optim import Adam
import torch.nn as nn
from commom import get_spambase
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter


mlp = MLP(input_features=57, out_features=2)
train_data_loader, x_test, y_test = get_spambase(test_size=0.25)
optimizer = Adam(mlp.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

print_step = 25

writer = SummaryWriter(log_dir="./data/train_BiMLP_log") # 提供创建event file的高级接口

if __name__ == '__main__':
    iter = 0
    all_epoch = 10
    for epoch in range(all_epoch):
        for step, (train_x, train_y) in enumerate(train_data_loader):
            y = mlp(train_x)
            train_loss = loss_func(y, train_y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            iter = epoch * len(train_data_loader) + step + 1
            
            if iter % print_step == 0:
                y = mlp(x_test)
                _, y = torch.max(y, 1)
                test_accuracy = accuracy_score(y_test, y)  # accuracy_score(真实标记，预测标记)
                print("iter=",iter,"loss=",train_loss.item(),"test_acc=",test_accuracy)
                
                writer.add_scalar("Loss and Accuracy/train loss", train_loss.item(), iter)    # 记录标量 参数(label, y, x)
                writer.add_scalar("Loss and Accuracy/test accuracy", test_accuracy, iter)
                
    writer.close()
    torch.save(mlp, './data/model_and_params/MLP.pkl')