from ResNet import ResNet18, ResNet34, ResNet50
from commom import get_flower_loader, train_model
from torch.optim import Adam
import torch.nn as nn
import torch
import os

def main(num_classes, epochs, batch_size, lr, device, model_name, model_weight_path=""):
    # create model
    model = ResNet34(num_classes=num_classes).to(device)
    # load pre-train weight
    if model_weight_path != "":
        assert os.path.exists(model_weight_path), "weights file: '{}' not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "fc" in k:
                del weights_dict[k]
        # print(model.load_state_dict(weights_dict, strict=False))

    data_loader = get_flower_loader(batch_size=batch_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1E-3)
    loss_func = nn.CrossEntropyLoss()

    model = train_model(model, data_loader, loss_func, optimizer, epochs, device, model_name=model_name)
    torch.save(model, "./data/models/" + model_name + ".pkl")
    
if __name__ == '__main__':
    num_classes = 5
    epochs = 20
    batch_size = 16
    lr = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "ResNet34"
    model_weight_path = "./data/weights/resnet34_pre.pth"
    # model_weight_path = ""
    
    main(num_classes, epochs, batch_size, lr, device, model_name, model_weight_path)
