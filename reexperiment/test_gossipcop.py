import json
import random
from pprint import pprint

import numpy as np


def split_json_by_truth(json_data):
    real_data = {}
    fake_data = {}

    for key,value in json_data.items():
        if value['label'] == 'real':
            real_data[key] = value
        else:
            fake_data[key] = value


    return real_data,fake_data


def datafipe(data):
    # 读取原始数据
    # with open('./data/gossipcop_v3_keep_data_in_proper_length.json', 'r', encoding='utf-8') as f:
    #    data = json.load(f)

    # 获取数据键的列表
    data_keys = list(data.keys())

    # 使用随机数生成器打乱数据键的顺序
    random.shuffle(data_keys)

    # 计算划分的索引
    total_size = len(data_keys)
    train_size = 80
    val_size = 80
    test_size = total_size - train_size - val_size

    # 划分数据键
    train_keys = data_keys[:train_size]
    val_keys = data_keys[train_size:train_size + val_size]
    test_keys = data_keys[train_size + val_size:]

    # 从原始数据中提取训练集、验证集和测试集的条目
    train_data = {key: data[key] for key in train_keys}
    val_data = {key: data[key] for key in val_keys}
    test_data = {key: data[key] for key in test_keys}
    return train_data, val_data, test_data

    # 保存划分后的数据集
    # with open('my-train_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=4)
    #
    # with open('./data/my-val_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(val_data, f, ensure_ascii=False, indent=4)
    #
    # with open('./data/my-test_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(test_data, f, ensure_ascii=False, indent=4)


def opennpy():
    # 指定.npy文件的路径
    file_path = "./data/train_emo.npy"

    # 使用NumPy的load方法加载.npy文件中的数据
    data = np.load(file_path)

    # 现在，您可以使用加载的数据进行进一步的处理和分析
    print("Loaded data shape:", data.shape)

    print(data)

if __name__ == '__main__':
    with open('data/gossipcop/raw/gossipcop_v3_keep_data_in_proper_length.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    real_data, fake_data = split_json_by_truth(data)

    train_real,valid_real,test_real = datafipe(real_data)
    train_fake,valid_fake,test_fake = datafipe(fake_data)
    train_real.update(train_fake)
    valid_real.update(valid_fake)
    test_real.update(test_fake)
    with open('data/gossipcop/raw/train_raw.json', 'w', encoding='utf-8') as f:
        json.dump(train_real, f, ensure_ascii=False, indent=4)

    with open('data/gossipcop/raw/valid_raw.json', 'w', encoding='utf-8') as f:
        json.dump(valid_real, f, ensure_ascii=False, indent=4)
    with open('data/gossipcop/raw/test_raw.json', 'w', encoding='utf-8') as f:
        json.dump(test_real, f, ensure_ascii=False, indent=4)
