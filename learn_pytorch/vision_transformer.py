import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    "Converting image to patch embedding."
    def __init__(self, in_channel, img_size, dim, patch_size):
        super().__init__() 
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size / patch_size + 1)**2 if img_size % patch_size else (img_size / patch_size)**2
        self.conv = nn.Conv2d(in_channel, dim, patch_size, patch_size)
    
    def forward(self, x):
        # if can not divide image to patches, padding image.
        pad = self.img_size % self.patch_size
        if pad:
            x = F.pad(x, (0, self.patch_size - pad,
                          0, self.patch_size - pad,
                          0, 0))
        # [B, C, H, W] -> [B, dim(d), H/patch_size, W/patch_size]
        x = self.conv(x)
        # [B, d, H/patch_size, W/patch_size] -> flatten -> [B, d, n] -> transpose -> [B, n, d]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class Attention(nn.Module):
    """Self-Attention or Multi-Attention"""
    def __init__(self, num_heads, dim):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.sub_dim = int(dim / self.num_heads)
        self.scale = self.sub_dim ** -0.5
        self.dropout = nn.Dropout()
        self.Wo = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, n, d = x.shape
        # [B, n, d] -> qkv -> [B, n, 3d] -> reshape -> [B, n, 3, h, sd] -> permute -> [3, B, h, n, sd]
        x = self.qkv(x).reshape(B, n, 3, self.num_heads, self.sub_dim).permute(2, 0, 3, 1, 4)
        # q, k, v: [B, h, n, sd]
        q, k, v = x[0], x[1], x[2]
        # [B, h, n, sd] @ [B, h, sd, n] -> [B, h, n, n]
        attn = q @ k.transpose(-1, -2) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        # [B, h, n, n] @ [B, h, n, sd] -> [B, h, n, sd]
        x = attn @ v
        # [B, h, n, sd] -> transpose -> [B, n, h, sd] -> flatten -> [B, n, d]
        x = x.transpose(1, 2).flatten(2)
        x = self.dropout(self.Wo(x))
        
        return x
        

class Mlp(nn.Module):
    def __init__(self, dim, expansion_rate=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim*expansion_rate),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim*expansion_rate, dim))
        
    def forward(self, x):
        # [B, n, d] -> fc -> [B, n, d]
        x = self.fc(x)

        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, num_heads, dim, expansion_rate):
        super().__init__()
        self.subnet1 = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(num_heads, dim),
            nn.Dropout()
        )
        self.subnet2 = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(dim, expansion_rate),
            nn.Dropout()
        )
        
    def forward(self, x):
        x = x + self.subnet1(x)
        x = x + self.subnet2(x)
    
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, in_channel=3, img_size=224, dim=768, patch_size=16, num_heads=12, num_blocks=12, 
                 num_classes=1000, expansion_rate=4):
        super().__init__()
        self.embed = PatchEmbed(in_channel, img_size, dim, patch_size)
        self.dropout = nn.Dropout()
        self.blocks = nn.Sequential(*[EncoderBlock(num_heads, dim, expansion_rate) for i in range(num_blocks)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_token = nn.Parameter(torch.zeros(1, int(self.embed.num_patches)+1, dim))
        self.head = nn.Linear(dim, num_classes)
        
        # init weight
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_token, std=.02)
        self.apply(_init_weights)
        
    def forward(self, x):
        # [B, C, H, W] -> [B, n, d]
        x = self.embed(x)
        # [B, n, d] -> cat -> [B, n+1, d]
        x = self.dropout(torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_token)
        x = self.ln(self.blocks(x))
        # [B, n+1, d] -> head -> [B, 1, num_classes]
        x = self.head(x[:, 0]).softmax(dim=1)
        
        return x
        
def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def vit_base_16(num_classes=1000):
    model = VisionTransformer(
        num_blocks=12,
        dim=768,
        num_heads=12,
        num_classes=num_classes)
    
    return model

def vit_large_16(num_classes=1000):
    model = VisionTransformer(
        num_blocks=24,
        dim=1024,
        num_heads=16,
        num_classes=num_classes)
    
    return model

def vit_huge_16(num_classes=1000):
    model = VisionTransformer(
        num_blocks=32,
        dim=1280,
        num_heads=16,
        num_classes=num_classes)
    
    return model