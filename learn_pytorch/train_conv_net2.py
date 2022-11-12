from conv_net2 import ConvNet
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from commom import get_FashionMNIST_loader, train_model

if __name__ == '__main__':
    convnet = ConvNet()
    data_loader = get_FashionMNIST_loader()
    optimizer = Adam(convnet.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()
    train_rate = 0.8
    epochs = 20
    convnet = train_model(convnet, data_loader, 0.8, loss_func, optimizer, epochs=20)