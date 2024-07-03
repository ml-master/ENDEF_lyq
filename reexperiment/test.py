import json
import random

import numpy as np


def datafipe():
    # 读取原始数据
    with open('./data/gossipcop_v3-1_style_based_fake.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取数据键的列表
    data_keys = list(data.keys())

    # 使用随机数生成器打乱数据键的顺序
    random.shuffle(data_keys)

    # 计算划分的索引
    total_size = len(data_keys)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 划分数据键
    train_keys = data_keys[:train_size]
    val_keys = data_keys[train_size:train_size + val_size]
    test_keys = data_keys[train_size + val_size:]

    # 从原始数据中提取训练集、验证集和测试集的条目
    train_data = {key: data[key] for key in train_keys}
    val_data = {key: data[key] for key in val_keys}
    test_data = {key: data[key] for key in test_keys}

    # 保存划分后的数据集
    with open('my-train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open('./data/my-val_data.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    with open('./data/my-test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


def opennpy():
    # 指定.npy文件的路径
    file_path = "./data/train_emo.npy"

    # 使用NumPy的load方法加载.npy文件中的数据
    data = np.load(file_path)

    # 现在，您可以使用加载的数据进行进一步的处理和分析
    print("Loaded data shape:", data.shape)

    print(data)

if __name__ == '__main__':
    opennpy()