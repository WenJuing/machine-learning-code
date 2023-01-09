from RNN import RNN
from commom import get_flower_loader, train_model, get_MNIST_loader
from torch.optim import RMSprop
import torch.nn as nn
import torch

def main(num_classes, epochs, batch_size, lr, device, model_name, dataset_name=None, img_size=224):
    # create model
    model = RNN(input_dim=img_size, hidden_dim=128, layer_num=1, num_classes=num_classes).to(device)

    data_loader = get_flower_loader(batch_size)
    # data_loader, _, _ = get_MNIST_loader()
    optimizer = RMSprop(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    model = train_model(model, data_loader, loss_func, optimizer, epochs, device, 
                        model_name=model_name, dataset_name=dataset_name, is_rnn=True)
    torch.save(model, "./data/models/" + model_name + ".pkl")
    
if __name__ == '__main__':
    num_classes = 5
    img_size = 224
    epochs = 30
    batch_size = 32
    lr = 0.0003
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "RNN"
    dataset_name = "flower"
    
    main(num_classes, epochs, batch_size, lr, device, model_name, dataset_name, img_size)
