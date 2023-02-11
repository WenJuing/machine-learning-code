import torch
from thop import profile
from model import max_vit_tiny_224 as create_model

model = create_model()
input = torch.randn(1, 3, 224, 224)  # batch_size不影响计算结果
flops, params = profile(model, (input,), verbose=False)
print("FLOPs: %.1f G" % (flops / 1E9))
print("Params: %.1f M" % (params / 1E6))