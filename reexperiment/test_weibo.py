import json
import os
import random
from pprint import pprint

import numpy as np


def split_data(data):
    train_line = 80 * 3
    valid_line = 80 * 3
    return data[:train_line], data[train_line: train_line + valid_line], data[train_line + valid_line:]


def concat_txt(dir_path):
    """将原数据集中已经拆分完成的数据还原"""
    rumor_file_suffix = '_rumor.txt'
    nonrumor_file_suffix = '_nonrumor.txt'
    files = os.listdir(dir_path)
    file_prefix = dir_path + '/'

    rumor_lines = []
    nonrumor_lines = []
    for file in files:
        if file.endswith(rumor_file_suffix):
            with open(file_prefix + file, 'r') as f:
                rumor_lines += f.readlines()
        if file.endswith(nonrumor_file_suffix):
            with open(file_prefix + file, 'r') as f:
                nonrumor_lines += f.readlines()

    return rumor_lines, nonrumor_lines


def filter_data(data):
    result = []
    n_lines = len(data)
    for idx in range(2, n_lines, 3):
        text = data[idx].strip()
        if text:
            result += data[idx-2:idx+1]
    return result


def read_data(data_path):
    rumor_lines, nonrumor_lines = concat_txt(data_path)
    rumor_lines = filter_data(rumor_lines)
    nonrumor_lines = filter_data(nonrumor_lines)

    train_rumor ,valid_rumor ,test_rumor = split_data(rumor_lines)
    train_nonrumor ,valid_nonrumor ,test_nonrumor = split_data(nonrumor_lines)
    return (train_rumor ,valid_rumor ,test_rumor) ,(train_nonrumor ,valid_nonrumor ,test_nonrumor)

def list2file(data ,file):
    with open(file, 'w') as f:
        for item in data:
            f.write(item)




if __name__ == '__main__':
    data_path = "./data/weibo/tweets"
    fake_data,real_data = read_data(data_path)
    data_type = ['train','valid','test']
    for i in range(len(data_type)):
        list2file(real_data[i],'{}/{}_{}.txt'.format(data_path,data_type[i],'real'))
        list2file(fake_data[i],'{}/{}_{}.txt'.format(data_path,data_type[i],'fake'))






