from my_vgg_net import MyVggNet
from commom import get_flower_loader, train_model
from torch.optim import Adam
import torch.nn as nn
import torch

num_classes = 5
epochs = 10
batch_size = 16
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    my_vgg = MyVggNet(num_classes=num_classes).to(device)
    data_loader = get_flower_loader(batch_size=batch_size)

    optimizer = Adam(my_vgg.parameters(), lr=lr, weight_decay=5E-2)
    loss_func = nn.CrossEntropyLoss()

    my_vgg = train_model(my_vgg, data_loader, loss_func, optimizer, epochs, device, model_name="my_vgg")
    torch.save(my_vgg, "./data/models/myVGG.pkl")