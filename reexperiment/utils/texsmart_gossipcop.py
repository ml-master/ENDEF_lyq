import os
import json
from json import JSONDecodeError

import requests
from tqdm import tqdm


def te(raw_file,processed_file):

    # 初始化一个空列表，用于存储所有处理后的数据
    processed_data_list = []


    with open(raw_file, "r", encoding="utf-8") as file:  # 以读取模式打开文件
        data = json.load(file)  # 读取文件内容，并将其解析为JSON对象
        for item_id, item_data in tqdm(data.items()):  # 遍历文件中的每个条目
            # 提取文本内容
            text = item_data["text"]  # 获取条目中的文本内容

            # 调用TexSmart工具进行实体识别
            # ... (TexSmart API调用代码，参考之前的示例)
            obj = {
                "str": text,
                "options": {       #”alg”表示对应的功能需要调用什么算法，”enable”可以取”true”或”false”,表示是否要激活对应的功能。
                    "input_spec": {"lang": "en"},  #表示输入的语言种类，它有三个取值，分别是自动识别语言(“auto”)，中文（“chs”）和英文（“en”）
                    "word_seg": {"enable": False},  #分词
                    "pos_tagging": {"enable": False, "alg": "log_linear"},   #词性标注
                    "ner": {"enable": True, "alg": "fine.std"},   #”coarse\fine”表示返回的是粗粒度/细粒度命名实体识别的结果
                    "syntactic_parsing": {"enable": False},    #句法分析
                    "srl": {"enable": False},                   #语义角色标注
                    "text_cat": {"enable": False},              #文本分类工具
                }
            }
            req_str = json.dumps(obj).encode()  # 将 对象转换为JSON字符串，并编码为字节
            # print(item_id)  # 打印条目ID，用于调试
            url = "https://texsmart.qq.com/api"  # 指定TexSmart API的URL
            r = requests.post(url, data=req_str)  # 向API发送POST请求
            r.encoding = "utf-8"  # 设置API响应的编码格式为UTF-8

            # print("API Response:")
            # print(r.text)
            # print("***********")
            try:
                response = json.loads(r.text)     # 将API响应解析为JSON对象
            # 将TexSmart工具的输出结果格式化成目标格式
            # ... (格式化代码，参考之前的示例)
            except JSONDecodeError as e:
                print("error item:"+item_id)
                continue

            entity_list = []     # 用于存储实体识别结果的列表
            for entity in response["entity_list"]:
                related = entity.get("meaning", {}).get("related", [])
                entity_list.append({
                    "entity": entity["str"],
                    "tag": entity["tag"],
                    "tag_i18n": entity.get("tag_i18n", ""),
                    "related": related  # 包含相关性信息
                })
            # 根据原始数据中的处理前的标签定义标签
            processed_label = int(item_data['label']=='real')

            # 组织处理后的数据
            processed_data = {
                "content": text,
                "label": processed_label,
                "time": "",  # 在原始数据中没有时间信息，这里留空
                "entity_list": entity_list  # entity_list是从TexSmart API获取的
            }

            # 将处理后的数据添加到列表中
            processed_data_list.append(processed_data)

    # 将处理后的数据存储到新的JSON文件中
    processed_data_json = json.dumps(processed_data_list, indent=4)
    with open(processed_file, "w", encoding="utf-8") as output_file:
        output_file.write(processed_data_json)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(os.getcwd())
    # 准备要处理的数据所在文件夹路径
    raw_path = "../data/gossipcop/raw/"
    processed_path = "../datanew/"

    te(raw_path+'train_raw.json',processed_path+'train.json')
    te(raw_path+'valid_raw.json',processed_path+'valid.json')
    te(raw_path+'test_raw.json',processed_path+'test.json')

