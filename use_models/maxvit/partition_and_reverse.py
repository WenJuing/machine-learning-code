import torch
import numpy as np


def window_partition(x, P):
    H, W = x.shape
    # [H, W] -> reshape -> [H // P, P, W // P, P] -> transpose -> [H // P, W // P, P, P] -> reshape -> [H*W // P**2, P*P]
    x = x.reshape(H // P, P, W // P, P).transpose(1, 2).reshape(-1, P*P)
    return x
    
def window_reverse(x, H, W):
    P = int(np.sqrt(x.shape[1]))
    # [H*W // P**2, P*P] -> reshape -> [H // P, W // P, P, P] -> transpose -> [H // P, P, W // P, P] -> reshape -> [H, W]
    x = x.reshape(H // P, W // P, P, P).transpose(1, 2).reshape(H, W)
    return x

def grid_partition(x, G):
    H, W = x.shape
    # [H, W] -> reshape -> [G, H // G, G, W // G] -> permute -> [H // G, W // G, G, G] -> [H*W // G**2, G*G]
    x = x.reshape(G, H // G, G, W // G).permute(1, 3, 0, 2).reshape(-1, G*G)
    return x

def grid_reverse(x, H, W):
    G = int(np.sqrt(x.shape[1]))
    x = x.reshape(H // G, W // G, G, G).permute(2, 0, 3, 1).reshape(H, W)
    # [H*W // G**2, G*G] -> [H // G, W // G, G, G] -> permute -> [G, H // G, G, W // G] -> reshape -> [H, W]
    return x


if __name__ == "__main__":
    H = 8
    W = 8
    P = 4
    G = 4
    # x: [H, W]，简单起见，省略了 B 和 C 维    
    x = torch.arange(1, H*W+1).reshape(H, W)
    print(x)
    x = window_partition(x, P)
    print(x)
    x = window_reverse(x, H, W)
    print(x)
    x = grid_partition(x, G)
    print(x)
    x = grid_reverse(x, H, W)
    print(x)