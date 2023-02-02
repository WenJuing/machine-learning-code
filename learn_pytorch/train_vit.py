from vision_transformer import vit_base_16
from commom import get_flower_loader, train_model
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
import os
import math


def main(num_classes, epochs, batch_size, lr, lrf, device, model_name, model_weight_path="", dataset_name=None):
    # create model
    # model = ConvNet(in_channel=3, img_size=224).to(device)
    model = vit_base_16(num_classes=num_classes).to(device)
    # load pre-train weight
    if model_weight_path != "":
        assert os.path.exists(model_weight_path), "weights file: '{}' not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "fc" in k:
                del weights_dict[k]
        # print(model.load_state_dict(weights_dict, strict=False))

    data_loader = get_flower_loader(batch_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1E-3)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    loss_func = nn.CrossEntropyLoss()

    model = train_model(model, data_loader, loss_func, optimizer, epochs, device, model_name=model_name, dataset_name=dataset_name,
                        scheduler=scheduler)
    # torch.save(model, "./data/models/" + model_name + ".pkl")
    
if __name__ == '__main__':
    num_classes = 5
    epochs = 10
    batch_size = 16
    lr = 0.0002
    lrf = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "myViT-B-16"
    dataset_name = "flower"
    model_weight_path = ""
    
    main(num_classes, epochs, batch_size, lr, lrf, device, model_name, model_weight_path, dataset_name)
