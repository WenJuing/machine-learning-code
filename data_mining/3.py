# 从“有房者”和“婚姻状况”两个属性中分别使用ID3算法和C4.5算法选择测试属性
import numpy as np
import pandas as pd


def Ent(data):
    df = data
    posi = df[df['cheat'] == 'yes'].count()['cheat']
    nega = df[df['cheat'] == 'no'].count()['cheat']
    ent = -(posi/(posi+nega)*np.log2(posi/(posi+nega)+1e-8) + nega/(posi+nega)*np.log2(nega/(posi+nega)+1e-8))
    
    return ent

def Gain(data, a):
    df = data
    ent = Ent(data)
    if a == 'refund':
        d1 = df[df[a]=='yes']
        ent1 = Ent(d1)
        d2 = df[df[a]=='no']
        ent2 = Ent(d2)
        
        d_size = df.shape[0]
        d1_size = d1.shape[0]
        d2_size = d2.shape[0]
        gain = ent - (d1_size/d_size*ent1 + d2_size/d_size*ent2)
    
    if a == 'marital status':
        d1 = df[df[a]=='single']
        ent1 = Ent(d1)
        d2 = df[df[a]=='married']
        ent2 = Ent(d2)
        d3 = df[df[a]=='divorced']
        ent3 = Ent(d3)
        
        d_size = df.shape[0]
        d1_size = d1.shape[0]
        d2_size = d2.shape[0]
        d3_size = d3.shape[0]
        gain = ent - (d1_size/d_size*ent1 + d2_size/d_size*ent2 + d3_size/d_size*ent3)
    
    return gain

def GainRation(data, a):
    df = data
    gain = Gain(data, a)
    d_size = data.shape[0]
    if a == 'refund':
        d1_size = df[df[a]=='yes'].shape[0]
        d2_size = df[df[a]=='no'].shape[0]
        iv = -(d1_size/d_size*np.log2(d1_size/d_size+1e-8) + d2_size/d_size*np.log2(d2_size/d_size+1e-8))
    if a == 'marital status':
        d1_size = df[df[a]=='single'].shape[0]
        d2_size = df[df[a]=='married'].shape[0]
        d3_size = df[df[a]=='divorced'].shape[0]
        iv = -(d1_size/d_size*np.log2(d1_size/d_size+1e-8) + d2_size/d_size*np.log2(d2_size/d_size+1e-8) + d3_size/d_size*np.log2(d3_size/d_size+1e-8))
    gain_ration = gain / iv
    
    return gain_ration
        
if __name__ == '__main__':
    data = pd.read_csv('3.csv')
    ent = Ent(data)
    print('ent=', ent)
    for a in ('refund', 'marital status'):
        gain = Gain(data, a)
        gain_ration = GainRation(data, a)
        print(a)
        print('gain=', gain)
        print('gain_ration=', gain_ration)
