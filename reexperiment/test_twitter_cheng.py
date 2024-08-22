import json
import os
import random
import shutil
from pprint import pprint
import pandas as pd
import numpy as np


def read_dataAsDf(data_path):
    label_file = open(data_path + '/twitter_label.txt', 'r', encoding='utf-8')
    text_file = open(data_path + '/twitter_lines.txt', 'r', encoding='utf-8')

    label_list = [int(s[0]) for s in label_file.readlines()]

    text_list = [s[:-1] for s in text_file.readlines()]

    data_size = min(len(label_list), len(text_list))
    return pd.DataFrame(
        {
            'id': np.arange(data_size),
            'label': label_list[:data_size],
            'text': text_list[:data_size]
        }
    )

def split_data(data):
    train_size = 80
    valid_size = 80
    return data[:train_size], data[train_size:train_size+valid_size], data[train_size+valid_size:]


def split_by_label(df):
    return df[df['label'] == 0] , df[df['label'] == 1]


if __name__ == '__main__':
    data_dir = './data/twitter_cheng/origin'
    df = read_dataAsDf(data_dir)
    fake_data, real_data = split_by_label(df)
    train_fake, valid_fake, test_fake = split_data(fake_data)
    train_real, valid_real, test_real = split_data(real_data)

    train_data = pd.concat([train_fake, train_real], axis=0)
    valid_data = pd.concat([valid_fake, valid_real], axis=0)
    test_data = pd.concat([test_fake, test_real], axis=0)

    train_data.to_csv('./data/twitter_cheng/train.csv', encoding='utf-8')
    valid_data.to_csv('./data/twitter_cheng/valid.csv', encoding='utf-8')
    test_data.to_csv('./data/twitter_cheng/test.csv', encoding='utf-8')










