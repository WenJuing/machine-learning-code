# 选择最优划分属性
import numpy as np
import pandas as pd


# fmt:off
# 属性对照表，为了方便索引，设置从0开始，'ij'表示为第i类属性的第j个样式
table = {"color": 0, "root": 1, "knock": 2, "stripe": 3, "navel": 4, "touch": 5,
    '00': 'green', '01': 'black',        '02': 'white',
    '10': 'curl',  '11': 'little_curl',  '12': 'hard',
    '20': 'sound', '21': 'dull',         '22': 'crisp',
    '30': 'clear', '31': 'little_vague', '32': 'vague',
    '40': 'sag',   '41': 'little_sag',   '42': 'flag',
    '50': 'slide', '51': 'stick',        
}
 # 属性集
A = {   
    'color': ['green', 'black', 'white'],          # 色泽：青绿 乌黑 浅白
    'root': ['curl', 'little_curl', 'hard'],       # 根蒂：蜷缩 稍蜷 硬挺
    'knock': ['sound', 'dull', 'crisp'],           # 敲声：浊响 沉闷 清脆
    'stripe': ['clear', 'little_vague', 'vague'],  # 纹理：清晰 稍糊 模糊
    'navel': ['sag', 'little_sag', 'flag'],        # 脐部：凹陷 稍凹 平坦
    'touch': ['slide', 'stick'],                   # 触感：硬滑 软粘
}
is_goood_melon = ['yes', 'no']    # 是否为好瓜：是 否
# fmt:on


def get_testD():
    """获得训练集"""
    # fmt:off
    testD = [   # 训练集
        [A['color'][0], A['root'][0], A['knock'][0], A['stripe'][0], A['navel'][0], A['touch'][0], is_goood_melon[0]],   # 编号：1
        [A['color'][1], A['root'][0], A['knock'][1], A['stripe'][0], A['navel'][0], A['touch'][0], is_goood_melon[0]],   # 编号：2
        [A['color'][1], A['root'][0], A['knock'][0], A['stripe'][0], A['navel'][0], A['touch'][0], is_goood_melon[0]],   # 编号：3
        [A['color'][0], A['root'][0], A['knock'][1], A['stripe'][0], A['navel'][0], A['touch'][0], is_goood_melon[0]],   # 编号：4
        [A['color'][2], A['root'][0], A['knock'][0], A['stripe'][0], A['navel'][0], A['touch'][0], is_goood_melon[0]],   # 编号：5
        [A['color'][0], A['root'][1], A['knock'][0], A['stripe'][0], A['navel'][1], A['touch'][1], is_goood_melon[0]],   # 编号：6
        [A['color'][1], A['root'][1], A['knock'][0], A['stripe'][1], A['navel'][1], A['touch'][1], is_goood_melon[0]],   # 编号：7
        [A['color'][1], A['root'][1], A['knock'][0], A['stripe'][0], A['navel'][1], A['touch'][0], is_goood_melon[0]],   # 编号：8
        
        [A['color'][1], A['root'][1], A['knock'][1], A['stripe'][1], A['navel'][1], A['touch'][0], is_goood_melon[1]],   # 编号：9
        [A['color'][0], A['root'][2], A['knock'][2], A['stripe'][0], A['navel'][2], A['touch'][1], is_goood_melon[1]],   # 编号：10
        [A['color'][2], A['root'][2], A['knock'][2], A['stripe'][2], A['navel'][2], A['touch'][0], is_goood_melon[1]],   # 编号：11
        [A['color'][2], A['root'][0], A['knock'][0], A['stripe'][2], A['navel'][2], A['touch'][1], is_goood_melon[1]],   # 编号：12
        [A['color'][0], A['root'][1], A['knock'][0], A['stripe'][1], A['navel'][0], A['touch'][0], is_goood_melon[1]],   # 编号：13
        [A['color'][2], A['root'][1], A['knock'][1], A['stripe'][1], A['navel'][0], A['touch'][0], is_goood_melon[1]],   # 编号：14
        [A['color'][1], A['root'][1], A['knock'][0], A['stripe'][0], A['navel'][1], A['touch'][1], is_goood_melon[1]],   # 编号：15
        [A['color'][2], A['root'][0], A['knock'][0], A['stripe'][2], A['navel'][2], A['touch'][0], is_goood_melon[1]],   # 编号：16
        [A['color'][0], A['root'][0], A['knock'][1], A['stripe'][1], A['navel'][1], A['touch'][0], is_goood_melon[1]],   # 编号：17
    ]
    # fmt:on
    testD = np.array(testD)
    return testD


def get_Ent(D):
    """根据计算信息熵"""
    try:  # 防止数据集中缺少相关项造成程序中断
        good_melon = pd.value_counts(D[:, 6])["yes"]
    except KeyError:
        good_melon = 0
    try:
        bad_melon = pd.value_counts(D[:, 6])["no"]
    except KeyError:
        bad_melon = 0

    p0 = good_melon / (good_melon + bad_melon)
    if p0 == 0:  # 防止log2(0)中断程序
        p0 = 1
    p1 = bad_melon / (good_melon + bad_melon)
    Ent = np.round(-(p0 * np.log2(p0) + p1 * np.log2(p1)), 3)
    return Ent


def classify_D(testD, attribute):
    """根据属性对训练集进行分类"""
    attribute = table[attribute]
    df = pd.DataFrame(testD)  # 列表转换为df对象
    # 属性有3个值
    if attribute <= 4:
        attribute_str = str(attribute)
        # fmt:off
        D1 = df.drop(df[(df[attribute] == table[attribute_str+'1']) | (df[attribute] == table[attribute_str+'2'])].index)
        D2 = df.drop(df[(df[attribute] == table[attribute_str+'0']) | (df[attribute] == table[attribute_str+'2'])].index)
        D3 = df.drop(df[(df[attribute] == table[attribute_str+'0']) | (df[attribute] == table[attribute_str+'1'])].index)
        # fmt:on
        D1 = np.array(D1)
        D2 = np.array(D2)
        D3 = np.array(D3)
        return D1, D2, D3
    # 属性有2个值
    if attribute == 5:
        attribute_str = str(attribute)
        D1 = df.drop(df[df[attribute] == table[attribute_str + "1"]].index)
        D2 = df.drop(df[df[attribute] == table[attribute_str + "0"]].index)
        D1 = np.array(D1)
        D2 = np.array(D2)
        return D1, D2


def get_Gain(testD, attribute):
    """计算属性的信息增益"""
    num_D = testD.shape[0]
    Ent_D = get_Ent(testD)  # 根结点的信息熵
    if attribute == "touch":
        D1, D2 = classify_D(testD, attribute)
        Ent_D1 = get_Ent(D1)
        Ent_D2 = get_Ent(D2)
        num_D1 = D1.shape[0]
        num_D2 = D2.shape[0]
        Gain = Ent_D - (num_D1 / num_D * Ent_D1 + num_D2 / num_D * Ent_D2)
    else:
        D1, D2, D3 = classify_D(testD, attribute)
        Ent_D1 = get_Ent(D1)
        Ent_D2 = get_Ent(D2)
        Ent_D3 = get_Ent(D3)
        num_D1 = D1.shape[0]
        num_D2 = D2.shape[0]
        num_D3 = D3.shape[0]
        Gain = Ent_D - (
            num_D1 / num_D * Ent_D1 + num_D2 / num_D * Ent_D2 + num_D3 / num_D * Ent_D3
        )
    Gain = np.round(Gain, 3)

    return Gain


def get_Gain_ratio(D, attribute):
    """计算属性的增益率"""
    num_D = testD.shape[0]
    if attribute == "touch":
        D1, D2 = classify_D(testD, attribute)
        num_D1 = D1.shape[0]
        num_D2 = D2.shape[0]
        iv = -(
            num_D1 / num_D * np.log2(num_D1 / num_D)
            + num_D2 / num_D * np.log2(num_D2 / num_D)
        )
    else:
        D1, D2, D3 = classify_D(testD, attribute)
        num_D1 = D1.shape[0]
        num_D2 = D2.shape[0]
        num_D3 = D3.shape[0]
        iv = -(
            num_D1 / num_D * np.log2(num_D1 / num_D)
            + num_D2 / num_D * np.log2(num_D2 / num_D)
            + num_D3 / num_D * np.log2(num_D3 / num_D)
        )
    Gain_ratio = np.round(get_Gain(D, attribute) / iv, 3)

    return Gain_ratio


def get_best_attribute(D, A):
    """获得最优属性"""
    Gains = []  # A中每个属性的信息增益
    Gain_ratios = []  # A中每个属性的增益率
    indexs = []  # 候选最优属性的下标
    for a in A:
        gain = get_Gain(D, a)
        gain_ratio = get_Gain_ratio(D, a)
        Gains.append(gain)
        Gain_ratios.append(gain_ratio)
    mean_Gain = np.mean(Gains)  # 平均信息增益
    for i in Gains:  # 排除信息增益不高于平均水平的属性
        if i <= mean_Gain:
            Gain_ratios[Gains.index(i)] = 0

    best_attribute = A[Gain_ratios.index(max(Gain_ratios))]
    A.remove(best_attribute)

    return best_attribute, A


if __name__ == "__main__":
    testD = get_testD()
    A = ["color", "root", "knock", "stripe", "navel", "touch"]
    print("原属性集：")
    print(A)
    best_attribute, A = get_best_attribute(testD, A)
    print("最优属性：")
    print(best_attribute)
    print("选择最优属性后的属性集：")
    print(A)
