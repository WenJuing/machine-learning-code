from conv_net import ConvNet
from ResNet import ResNet18
from commom import get_flower_loader, train_model, get_MNIST_loader
from torch.optim import Adam
import torch.nn as nn
import torch
import os

def main(num_classes, epochs, batch_size, lr, device, model_name, model_weight_path="", dataset_name=None):
    # create model
    # model = ConvNet(in_channel=3, img_size=224).to(device)
    model = ResNet18(num_classes=num_classes, in_channel=1).to(device)
    # load pre-train weight
    if model_weight_path != "":
        assert os.path.exists(model_weight_path), "weights file: '{}' not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "fc" in k:
                del weights_dict[k]
        # print(model.load_state_dict(weights_dict, strict=False))

    # data_loader = get_flower_loader(batch_size)
    data_loader, _, _ = get_MNIST_loader()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1E-3)
    loss_func = nn.CrossEntropyLoss()

    model = train_model(model, data_loader, loss_func, optimizer, epochs, device, model_name=model_name, dataset_name=dataset_name)
    # torch.save(model, "./data/models/" + model_name + ".pkl")
    
if __name__ == '__main__':
    num_classes = 10
    epochs = 50
    batch_size = 128
    lr = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "ResNet18(2)"
    dataset_name = "MNIST"
    # model_weight_path = "./data/weights/resnet18_pre.pth"
    model_weight_path = ""
    
    main(num_classes, epochs, batch_size, lr, device, model_name, model_weight_path, dataset_name)
