from ResNet import ResNet18
from commom import train_one_epoch, test, get_CIFAR100_loader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
import os
import argparse
import math


def main(opts):
    if os.path.exists("D:/weights/"+opts.model_name) is False:
        os.makedirs("D:/weights/"+opts.model_name)
        
    sw = SummaryWriter(log_dir="./runs/train_" + opts.model_name)
    
    # create model
    model = ResNet18(num_classes=opts.num_classes).to(device)
    
    # load pre-train weight
    if opts.weights != "":
        assert os.path.exists(opts.weights), "weights file: '{}' not exist.".format(opts.weights)
        weights_dict = torch.load(opts.weights, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "fc" or "head" or "classifer" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    train_loader, test_loader = get_CIFAR100_loader()
    
    # pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opts.epochs)) / 2) * (1 - opts.lrf) + opts.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    loss_func = nn.CrossEntropyLoss()

    best_acc = 0.0
    print("model:", opts.model_name, "| dataset:", opts.dataset_name, "| device:", opts.device)
    for epoch in range(opts.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_func, optimizer, epoch, device)
        scheduler.step()
        # test
        test_loss, test_acc = test(model, test_loader, loss_func, epoch, device)
        
        sw.add_scalars(opts.model_name+"/"+opts.dataset_name+" Loss", {'train': train_loss, 'test': test_loss}, epoch)
        sw.add_scalars(opts.model_name+"/"+opts.dataset_name+" Accuracy", {'train': train_acc, 'test': test_acc}, epoch)
        sw.add_scalar(opts.model_name+"/"+opts.dataset_name+" learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "D:/weights/"+opts.model_name+"/best_model.pth")
        torch.save(model.state_dict(), "D:/weights/"+opts.model_name+"/latest_model.pth")

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    
    parse.add_argument("--model_name", type=str, default="ResNet18")
    parse.add_argument("--dataset_name", type=str, default="CIFAR100")
    
    parse.add_argument("--num_classes", type=int, default=100)
    parse.add_argument("--epochs", type=int, default=10)
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--lr", type=float, default=0.0001)
    parse.add_argument("--lrf", type=float, default=0.01)
    parse.add_argument("--weight_decay", type=float, default=1E-2)
    
    parse.add_argument("--data_path", type=str, default="")
    parse.add_argument("--weights", type=str, default="")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--device", type=str, default=device)

    opts = parse.parse_args()
        
    main(opts)
