import json
import os
import random
import shutil
from pprint import pprint
import pandas as pd
import numpy as np


def read_origin_data(data_dir):
    dev_df = pd.read_csv('{}/devset/posts.txt'.format(data_dir),sep='\t',encoding='utf-8')
    dev_df = dev_df.rename(columns={
        'image_id(s)': 'image_id'
    })
    cols = list(dev_df.columns)
    cols[3], cols[4] = cols[4], cols[3]
    dev_df = dev_df[cols]
    test_df = pd.read_csv('{}/testset/posts_groundtruth.txt'.format(data_dir),sep='\t',encoding='utf-8')
    return dev_df,test_df


def move_image_dir(image_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for file in os.listdir(image_dir):
        if not file.endswith('.txt'):
            shutil.copy(image_dir + '/' + file, target_dir + '/' + file)

def has_image(images,image_set):
    return bool(set(images.split(',')) & image_set)
def filter_no_image(df,image_set):
    return df[
        df['image_id'].apply(lambda x: has_image(x,image_set))
    ]

def get_image_set(image_dir):
    image_set = set(filter(lambda x: not x.endswith('.txt'), os.listdir(image_dir)))
    return {file.split('.')[0] for file in image_set}


def split_truth(df):
    return df[df['label'] == 'real'],df[df['label'] == 'fake']

def split_data0(df):
    train_num = 80
    valid_num = 80
    return df.iloc[:train_num],df.iloc[train_num:train_num+valid_num], df.iloc[train_num+valid_num:]
def split_data(df):
    real_data,fake_data = split_truth(df)
    train_real, valid_real,test_real = split_data0(real_data)
    train_fake, valid_fake,test_fake = split_data0(fake_data)
    train_data = pd.concat([train_real,train_fake]).sample(frac=1).reset_index(drop=True)
    valid_data = pd.concat([valid_real,valid_fake]).sample(frac=1).reset_index(drop=True)
    test_data = pd.concat([test_real,test_fake]).sample(frac=1).reset_index(drop=True)
    return train_data,valid_data,test_data






if __name__ == '__main__':
    data_dir = './data/twitter'
    dev_df,test_df = read_origin_data(data_dir)
    move_image_dir('./data/twitter/devset/images', './data/twitter/images')
    move_image_dir('./data/twitter/testset/images', './data/twitter/images')
    image_set = get_image_set('./data/twitter/images')
    legal_data = pd.concat([
        filter_no_image(dev_df, image_set),
        filter_no_image(test_df, image_set),
    ])
    train_data, valid_data, test_data = split_data(legal_data)
    train_data.to_csv('{}/train.csv'.format(data_dir), index=False, sep='\t', encoding='utf-8')
    valid_data.to_csv('{}/valid.csv'.format(data_dir), index=False, sep='\t', encoding='utf-8')
    test_data.to_csv('{}/test.csv'.format(data_dir), index=False, sep='\t', encoding='utf-8')








