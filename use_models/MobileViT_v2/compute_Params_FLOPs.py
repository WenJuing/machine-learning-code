import torch
from thop import profile
from model import mobile_vit_v2 as create_model
import argparse


def main(opt):
    model = create_model(opt)
    input = torch.randn(1, 3, 224, 224)  # batch_size不影响计算结果
    flops, params = profile(model, (input,), verbose=False)   # 不知道为啥结果都是0.0
    print("FLOPs: %.1f G" % (flops / 1E9))
    print("Params: %.1f M" % (params / 1E6))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--width_multiplier', type=float, default=2.0)
    opt = parse.parse_args()
    
    main(opt)