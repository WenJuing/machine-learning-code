import pandas as pd
import numpy as np


def get_LIVE_dataset():
    "ref_name,dist_name,mos 779"
    meta_info = pd.read_csv('./data/pyiqa/meta_info/meta_info_LIVEIQADataset.csv')
    print(meta_info[meta_info['ref_name'] == "refimgs/rapids.bmp"])
    return meta_info
    

if __name__ == '__main__':
    get_LIVE_dataset()