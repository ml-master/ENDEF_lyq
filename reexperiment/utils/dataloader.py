import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

# 标签字典，用于将标签映射为数字
label_dict = {
    "real": 0,
    "fake": 1
}

# # 年份类别字典，用于将年份映射为数字类别
# category_dict = {
#     "2000": 0,
#     "2001": 0,
#     "2002": 0,
#     "2003": 0,
#     "2005": 0,
#     "2004": 0,
#     "2006": 0,
#     "2007": 0,
#     "2008": 0,
#     "2009": 0,
#     "2010": 0,
#     "2011": 0,
#     "2012": 0,
#     "2013": 0,
#     "2014": 0,
#     "2015": 0,
#     "2016": 0,
#     "2017": 1,
#     "2018": 2
# }

# 将文本转换为模型输入的函数
def word2input(texts, max_len):
    # 加载BERT tokenizer
    tokenizer_path = '/data/lj_data/bert'  # 实际的BERT模型文件路径
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    token_ids = []
    for i, text in enumerate(texts):
        # 使用tokenizer将文本转换为token id，并进行padding和truncation
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)  # 创建mask张量，用于区分padding部分和实际内容部分
    return token_ids, masks

# 数据增强函数，根据给定的概率对内容和实体进行不同的处理
def data_augment(content, entity_list, aug_prob):
    entity_content = []
    random_num = random.randint(1,100)
    if random_num <= 50:  # 50%的概率进行实体处理
        for item in entity_list:
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):  # 根据aug_prob概率进行实体替换
                content = content.replace(item["entity"], '[MASK]')
            elif random_num <= int(2 * aug_prob * 100):  # 根据2 * aug_prob概率删除实体
                content = content.replace(item["entity"], '')
            else:
                entity_content.append(item["entity"])
        entity_content = ' [SEP] '.join(entity_content)  # 将处理后的实体以[SEP]连接
    else:
        content = list(nltk.word_tokenize(content))  # 使用nltk进行分词处理
        for index in range(len(content) - 1, -1, -1):
            random_num = random.randint(1,100)
            if random_num <= int(aug_prob * 100):  # 根据aug_prob概率删除单词
                del content[index]
            elif random_num <= int(2 * aug_prob * 100):  # 根据2 * aug_prob概率替换单词为[MASK]
                content[index] = '[MASK]'
        content = ' '.join(content)  # 将处理后的内容重新拼接为字符串
        entity_content = get_entity(entity_list)  # 获取实体列表并以[SEP]连接

    return content, entity_content

# 获取实体列表并以[SEP]连接
def get_entity(entity_list):
    entity_content = []
    for item in entity_list:
        entity_content.append(item["entity"])
    entity_content = ' [SEP] '.join(entity_content)
    return entity_content

# 加载数据并构建数据加载器
def get_dataloader(path, max_len, batch_size, shuffle, use_endef, aug_prob):
    data_list = json.load(open(path, 'r',encoding='utf-8'))  # 加载JSON数据文件
    df_data = pd.DataFrame(columns=('content','label'))  # 创建空的DataFrame用于存储数据
    for item in data_list:
        tmp_data = {}
        if shuffle == True and use_endef == True:  # 如果需要进行数据增强
            tmp_data['content'], tmp_data['entity'] = data_augment(item['content'], item['entity_list'], aug_prob)
        else:
            tmp_data['content'] = item['content']
            tmp_data['entity'] = get_entity(item['entity_list'])
        tmp_data['label'] = item['label']
        tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]  # 提取年份信息

        df_data = pd.concat([df_data, pd.DataFrame([tmp_data])], ignore_index=True)  # 将处理后的数据添加到DataFrame中

    content = df_data['content'].to_numpy()
    entity_content = df_data['entity'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())

    # 调用word2input函数将文本转换为模型输入
    content_token_ids, content_masks = word2input(content, max_len)
    entity_token_ids, entity_masks = word2input(entity_content, 50)

    # 构建TensorDataset和DataLoader
    dataset = TensorDataset(content_token_ids,
                            content_masks,
                            entity_token_ids,
                            entity_masks,
                            label)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,  # 多线程读取数据
        pin_memory=True,  # 将数据保存在pin memory中，加速GPU数据读取
        shuffle=shuffle
    )
    return dataloader

#  2个字典 4个方法
#
#  word2input 函数：
#      -使用BERT tokenizer将文本转换为模型输入所需的token id和masks张量
#
#  data_augment 函数：根据随机数生成，有50%的概率进行数据增强
#      -方式1：实体处理-随机替换实体为 [MASK]，或删除实体
#      -方式2：内容处理-随机删除单词或将单词替换为 [MASK]
#
#  get_entity 函数：
#      -从实体列表中获取实体内容并以[SEP]连接
#
#  get_dataloader 函数：
#      -加载JSON格式的数据文件，并根据需要进行数据增强。
#      -构建TensorDataset和DataLoader，将数据批量加载到模型中进行训练或评估。
#
#
#
#
