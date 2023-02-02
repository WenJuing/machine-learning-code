import torch
from vision_transformer import vit_base_16 as create_model
from thop import profile

model = create_model()
input = torch.randn(1, 3, 224, 224)  # batch_size不影响计算结果
flops, params = profile(model, (input,), verbose=False)
print("FLOPs: %.1f G" % (flops / 1E9))
print("Params: %.1f M" % (params / 1E6))