import torch
from thop import profile
from model import davit_small as create_model
# from model import swinv2_large_patch4_window12_192_22k as create_model

model = create_model().cuda()
input = torch.randn(1, 3, 224, 224).cuda()  # batch_size不影响计算结果
flops, params = profile(model, (input,), verbose=False)
print("FLOPs: %.1f G" % (flops / 1E9))
print("Params: %.1f M" % (params / 1E6))