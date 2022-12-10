# pyiqa的基本使用
import pyiqa
import torch
from load_data import get_LIVE_dataset

LIVE_PATH = './data/pyiqa/LIVE/'

# 支持的度量方法
all_metrics = pyiqa.list_models()
print(all_metrics, len(all_metrics))

# 使用iqa度量方法（默认设置），默认as_loss=False，若启用则可作为损失函数
metric = pyiqa.create_metric('mad', as_loss=True, device=torch.device('cuda'))
# 检测该指标是越低越好（True），还是越高越好（False）
print(metric.lower_better)

live_meta_info = get_LIVE_dataset()
# 注意，若在无参考度量中输入多个图像参数，则只度量第一个图像的质量；若在全参考度量中只输入一个图像参数，则会报错
# 图像参数支持路径或image_tensor两种形式
score_fr = metric(LIVE_PATH + live_meta_info['ref_name'][0], LIVE_PATH + live_meta_info['dist_name'][0])
print(score_fr)