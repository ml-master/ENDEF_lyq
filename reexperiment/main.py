import os
import argparse

from grid_search import Run  # 假设从grid_search模块导入了Run类或函数

# 使用argparse定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='mdfend_endef')  # 模型名称，默认为'mdfend_endef'
parser.add_argument('--epoch', type=int, default=20)  # 训练轮数，默认为10
parser.add_argument('--aug_prob', type=float, default=0.1)  # 数据增强的概率，默认为0.1             ********
parser.add_argument('--max_len', type=int, default=170)  # 序列最大长度，默认为170
parser.add_argument('--early_stop', type=int, default=5)  # 提前停止的步数，默认为5                 ********
parser.add_argument('--root_path', default='./datanew/')  # 数据根目录，默认为'./data/'
parser.add_argument('--batchsize', type=int, default=64)  # 批大小，默认为64
parser.add_argument('--seed', type=int, default=2021)  # 随机种子，默认为2021                  ********
parser.add_argument('--gpu', default='0')  # 指定GPU，默认为'0'
parser.add_argument('--emb_dim', type=int, default=768)  # 嵌入维度，默认为768                      ********
parser.add_argument('--lr', type=float, default=0.0001)  # 学习率，默认为0.0001
parser.add_argument('--emb_type', default='bert')  # 嵌入类型，默认为'bert'                      ********
parser.add_argument('--save_log_dir', default='./logs')  # 日志保存目录，默认为'./logs'
parser.add_argument('--save_param_dir', default='./param_model')  # 参数保存目录，默认为'./param_model'
parser.add_argument('--param_log_dir', default='./logs/param')  # 参数日志保存目录，默认为'./logs/param'
parser.add_argument('--model_path', default='bert-base-uncased')  # 使用预处理模型路径,默认使用bert-base-uncased预处理

# 解析命令行参数
args = parser.parse_args()

# 根据--gpu参数设置CUDA可见的设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 导入必要的库
import torch
import numpy as np
import random

# 设置随机种子以保证实验的可重复性
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 打印部分解析后的参数，确认配置正确
print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))

# 定义配置字典，包含解析后的参数及其他预定义设置
config = {
    'use_cuda': True,  # 是否使用CUDA加速，根据--gpu参数决定
    'batchsize': args.batchsize,  # 批大小
    'max_len': args.max_len,  # 序列最大长度
    'early_stop': args.early_stop,  # 提前停止步数
    'root_path': args.root_path,  # 数据根目录
    'aug_prob': args.aug_prob,  # 数据增强概率
    'weight_decay': 1e-3,  # 权重衰减
    'model': {
        'mlp': {'dims': [384], 'dropout': 0.5},# MLP模型参数
        'model_path': args.model_path
    },
    'emb_dim': args.emb_dim,  # 嵌入维度
    'lr': args.lr,  # 学习率
    'epoch': args.epoch,  # 训练轮数
    'model_name': args.model_name,  # 模型名称
    'seed': args.seed,  # 随机种子
    'save_log_dir': args.save_log_dir,  # 日志保存目录
    'save_param_dir': args.save_param_dir,  # 参数保存目录
    'param_log_dir': args.param_log_dir  # 参数日志保存目录
}

if __name__ == '__main__':
    print(os.getcwd() + args.root_path)  # 打印当前工作目录加上'/data'
    # 初始化grid_search模块中的Run类或函数，并传入config字典调用其main()方法
    Run(config=config).main()


# 导入和参数解析
# 参数解析 (argparse.ArgumentParser() 块)
# 设置CUDA可见性
# 设置随机种子  ——使用Python、NumPy和PyTorch的随机种子确保实验可重复性
# 配置字典 (config)  ——构建包含解析后参数和其他预设设置的字典
# 主执行块
#    Run(config=config).main()：初始化grid_search模块中的Run类或函数，并传入config字典调用其main()方法。
#