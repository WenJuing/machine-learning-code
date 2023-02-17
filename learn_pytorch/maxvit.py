"""MaxViT复现

Paper: `MaxViT: Multi-Axis Vision Transformer`
Data: 2023/2/17
Written by Ziwen Tan (wenjuing)
"""
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from timm.models.layers import DropPath
from timm.models.efficientnet_blocks import DepthwiseSeparableConv


class SqueezeExcitation(nn.Module):
    """SE Block.
    paper `Squeeze-and-Excitation Networks`

    Args:
        rd_rate (float): reduce rate of in_channels.
    """
    def __init__(self, in_channels, rd_rate=0.25) -> Tensor:
        super().__init__()
        rd_channels = int(in_channels * rd_rate)
        self.proj1 = nn.Conv2d(in_channels, rd_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.proj2 = nn.Conv2d(rd_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # [B, C, H, W] -> mean -> [B, C, 1, 1]
        gate = x.mean((2, 3), keepdim=True)
        # [B, C, 1, 1] -> proj1 -> [B, rd_C, 1, 1] -> proj2 -> [B, C, 1, 1]
        gate = self.softmax(self.proj2(self.relu(self.proj1(gate))))
        
        return x * gate
        

class MBConv(nn.Module):
    """MBConv structure.
    Args:
        downsample (bool): value is True in the first block of each stage,
                           value is False when in_channels and out_channels is equal.
        """
    def __init__(self, in_channels, out_channels, downsample) -> Tensor:
        super().__init__()
        self.math_branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            DepthwiseSeparableConv(in_channels, out_channels, 3, stride=2 if downsample else 1),
            SqueezeExcitation(out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        self.sub_branch = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, 1),
        ) if downsample else nn.Identity()
        
    def forward(self, x):
        x = self.math_branch(x) + self.sub_branch(x)
        
        return x
    
    
def window_partition(x, P=7):
    """Deviding the tensor into windows.
    
    Args:
        x (tensor): input tensor.
        P (int): window size.
    """
    B, C, H, W = x.shape
    # [B, C, H, W] -> reshape -> [B, C, H/P, P, W/P, P] -> permute -> [B, H/P, W/P, P, P, C]
    x = x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 3, 5, 1)
    # [B, H // P, W // P, P, P, C] -> reshape -> [B*HW/P/P, P*P, C] = [_B, n, C]
    x = x.reshape(-1, P*P, C)
    
    return x


def window_reverse(x, H, W):
    """The reverse operation about window partition.
    
    Args:
        x (tensor): input tensor.
        H (int): original H of x.
        W (int): original W of x.
    """
    _B, n, C = x.shape
    P = int(np.sqrt(n))
    # [_B, n, C] -> reshape -> [B, H/P, W/P, P, P, C] -> permute -> [B, C, H/P, P, W/P, P] -> reshape -> [B, C, H, W]
    x = x.reshape(-1, H // P, W // P, P, P, C).permute(0, 5, 1, 3, 2, 4).reshape(-1, C, H, W)
    
    return x
    

def grid_partition(x, G=7):
    """Deviding the tensor into grids.
    
    Args:
        x (tensor): input tensor.
        G (int): grid size.
    """
    B, C, H, W = x.shape
    # [B, C, H, W] -> reshape -> [B, C, G, H/G, G, W/G] -> permute -> [B, H/G, W/G, G, G, C]
    x = x.reshape(B, C, G, H // G, G, W // G).permute(0, 3, 5, 2, 4, 1)
    # [B, H/G, W/G, G, G, C] -> reshape -> [B*HW/G/G, G*G, C] = [_B, n, C]
    x = x.reshape(-1, G*G, C)
    
    return x


def grid_reverse(x, H, W):
    """The reverse operation about grid partition.
    
    Args:
        x (tensor): input tensor.
        H (int): original H of x.
        W (int): original W of x.
    """
    _B, n, C = x.shape
    G = int(np.sqrt(n))
    # [_B, n, C] -> reshape -> [B, H/G, W/G, G, G, C] -> permute -> [B, C, G, H/G, G, W/G] -> reshape -> [B, C, H, W]
    x = x.reshape(-1, H // G, W // G, G, G, C).permute(0, 5, 3, 1, 4, 2).reshape(-1, C, H, W)
    
    return x


def get_relative_position_bias_index(M):
    # [2, M, M] -> flatten -> [2, MM]
    coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M))).flatten(1)
    # [2, MM, 1] - [2, 1, MM] -> [2, MM, MM]
    relative_coords = coords[:, :, None] - coords[:, None, :]
    #[2, MM, MM] -> permute -> [MM, MM, 2]
    relative_coords = relative_coords.permute(1, 2, 0)
    relative_coords[...] += M - 1
    relative_coords[:, :, 0] += 2 * M - 1
    # [MM, MM, 2] -> sum -> [MM, MM] = [n, n]
    relative_coords_index = relative_coords.sum(2)
    return relative_coords_index


class RelativeAttention(nn.Module):
    """Relative Self-Attention.
    Reference paper `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    RelativeAttention(Q, K, V) = Softmax(QK.T/√d + B)V
    
    Args:
        dim (int): patch dimension.
        num_heads (int): number of heads.
        attn_drop (float): dropout rate after softmax.
        drop (float): dropout rate after Wo.
        M (int): side size of attention aera.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., drop=0., M=7) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hd = int(dim // self.num_heads)
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(dim, 3*dim)
        self.softmax = nn.Softmax(-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.Wo = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*(M-1))**2, num_heads))
        self.relative_position_bias_index = get_relative_position_bias_index(M)
        # save bias into buffer that become a part of model weights
        self.register_buffer("relative_position_bias", self._get_relative_position_bias(M))
    
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def _get_relative_position_bias(self, M):
        # [(2M-1)*(2M-1), h] -> select -> [nn, h]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_bias_index.view(-1)]
        # [nn, h] -> reshape -> [n, n, h] -> permute -> [h, n, n] -> unsqueeze -> [1, h, n, n]
        relative_position_bias = relative_position_bias.reshape(M*M, M*M, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        
        return relative_position_bias
    
    def forward(self, x):
        B, n, d = x.shape
        # [B, n, d] -> qkv -> [B, n, 3d] -> reshape -> [B, n, 3, h, hd] -> permute -> [3, B, h, n, hd]
        qkv = self.qkv(x).reshape(B, n, 3, self.num_heads, self.hd).permute(2, 0, 3, 1, 4)
        # Q, K, V: [B, h, n, hd]
        Q, K, V = torch.unbind(qkv)
        # [B, h, n, hd] @ [B, h, hd, n] -> [B, h, n, n]
        attn = self.softmax(Q @ K.transpose(-1, -2) * self.scale + self.relative_position_bias)
        # [B, h, n, n] @ [B, h, n, hd] -> [B, h, n, hd]
        attn = self.attn_drop(attn) @ V
        # [B, h, n, hd] -> transpose -> [B, n, h, hd] -> flatten -> [B, n, d]
        x = attn.transpose(1, 2).flatten(2)
        x = self.drop(self.Wo(x))
        
        return x


class MLP(nn.Module):
    """MLP in Vision Transformer Block.
    
    Args:
        expansion_rate (float): expansion rate of in_channels.
    """
    def __init__(self, in_channels, expansion_rate=4) -> Tensor:
        super().__init__()
        hidden_channels = int(in_channels * expansion_rate)
        self.proj1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.proj2 = nn.Linear(hidden_channels, in_channels)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.proj1(x)))
        x = self.relu(self.proj2(x))
        
        return x
        

class TransformerBlock(nn.Module):
    """Block Attention or Grid Attention.
    Args:
        dim (int): patch dimension.
        num_heads (int): number of heads.
        attn_drop (float): dropout rate after softmax.
        drop (float): dropout rate after Wo.
        partition_function (def): window_partition or grid_partition.
        reverse_function (def): window_reverse or grid_reverse.
        """
    def __init__(self, dim, num_heads, attn_drop, drop, partition_function, reverse_function) -> Tensor:
        super().__init__()
    
    
if __name__ == "__main__":
    def test_window_partition_and_reverse():
        x = torch.rand(1, 64, 12, 12)
        x = window_partition(x, 2)
        print(x.shape)
        x = window_reverse(x, 12, 12)
        print(x.shape)
        
    def test_grid_partition_and_reverse():
        x = torch.rand(1, 64, 12, 12)
        x = grid_partition(x, 2)
        print(x.shape)
        x = grid_reverse(x, 12, 12)
        print(x.shape)
    
    def test_relative_attention():
        # x: [B, n, d]  n = M * M
        M = 7
        x = torch.rand(128, M*M, 64)
        relative_attention = RelativeAttention(dim=64)
        x = relative_attention(x)
        print(x.shape)
    
    def test_mbconv():
        downsample = True
        x = torch.rand(1, 64, 32, 32)
        mbconv = MBConv(in_channels=64, out_channels=128, downsample=downsample)
        x = mbconv(x)
        print(x.shape)
    
    test_mbconv()