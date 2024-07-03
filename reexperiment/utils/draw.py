import json

# 读取JSON文件
with open('../datanew/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 计算顶层对象数量
num_data = len(data)

print(f"JSON文件中包含 {num_data} 条数据。")
