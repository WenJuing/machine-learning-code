import torch


weight = torch.load("D:/weights/swin_transformer_v2/swinv2_tiny_patch4_window8_256.pth")
print(weight['model'].keys())