import logging
import os
import json

# 导入各个模型的Trainer类
from models.bigru import Trainer as BiGRUTrainer
from models.bert import Trainer as BertTrainer
from models.eann import Trainer as EANNTrainer
from models.mdfend import Trainer as MDFENDTrainer
from models.bertemo import Trainer as BertEmoTrainer
from models.bigruendef import Trainer as BiGRU_ENDEFTrainer
from models.bertendef import Trainer as BERT_ENDEFTrainer
from models.bertemoendef import Trainer as BERTEmo_ENDEFTrainer
from models.eannendef import Trainer as EANN_ENDEFTrainer
from models.mdfendendef import Trainer as MDFEND_ENDEFTrainer


# 定义一个生成浮点数范围的函数
def frange(x, y, jump):
    while x < y:
        x = round(x, 8)  # 四舍五入到小数点后8位
        yield x
        x += jump


# 定义Run类，用于管理训练流程
class Run():
    def __init__(self, config):
        self.config = config  # 初始化配置信息

    # 获取文件日志记录器
    def getFileLogger(self, log_file):
        logger = logging.getLogger()  # 获取全局日志记录器
        logger.setLevel(level=logging.INFO)  # 设置日志级别为INFO
        handler = logging.FileHandler(log_file)  # 创建文件处理器，将日志写入文件
        handler.setLevel(logging.INFO)  # 设置处理器的日志级别为INFO
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 定义日志格式
        handler.setFormatter(formatter)  # 将格式应用到处理器
        logger.addHandler(handler)  # 将处理器添加到日志记录器
        return logger  # 返回日志记录器

    # 将配置信息转换为字典形式
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    # 主函数，负责执行训练流程
    def main(self):
        param_log_dir = self.config['param_log_dir']  # 获取参数日志保存目录
        if not os.path.exists(param_log_dir):  # 如果目录不存在
            os.makedirs(param_log_dir)  # 创建目录
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + 'param.txt')  # 参数日志文件路径
        logger = self.getFileLogger(param_log_file)  # 获取文件日志记录器

        train_param = {
            'lr': [self.config['lr']] * 1,  # 学习率列表
        }
        print(train_param)
        param = train_param  # 将train_param赋给param
        best_param = []  # 存储最佳参数
        json_path = './logs/json/' + self.config['model_name'] + str(self.config['aug_prob']) + '.json'  # JSON结果保存路径
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        json_result = []  # 存储JSON结果

        for p, vs in param.items():  # 遍历参数字典
            best_metric = {}  # 存储最佳指标
            best_metric['metric'] = 0  # 初始化最佳指标值为0
            best_v = vs[0]  # 初始最佳参数值
            best_model_path = None  # 初始化最佳模型路径为None

            for i, v in enumerate(vs):  # 遍历参数值列表
                self.config['lr'] = v  # 更新配置中的学习率参数

                # 根据模型名称选择相应的Trainer类进行训练
                if self.config['model_name'] == 'eann':
                    trainer = EANNTrainer(self.config)
                elif self.config['model_name'] == 'bertemo':
                    trainer = BertEmoTrainer(self.config)
                elif self.config['model_name'] == 'bigru':
                    trainer = BiGRUTrainer(self.config)
                elif self.config['model_name'] == 'mdfend':
                    trainer = MDFENDTrainer(self.config)
                elif self.config['model_name'] == 'bert':
                    trainer = BertTrainer(self.config)
                elif self.config['model_name'] == 'bigru_endef':
                    trainer = BiGRU_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'bert_endef':
                    trainer = BERT_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'bertemo_endef':
                    trainer = BERTEmo_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'eann_endef':
                    trainer = EANN_ENDEFTrainer(self.config)
                elif self.config['model_name'] == 'mdfend_endef':
                    trainer = MDFEND_ENDEFTrainer(self.config)

                # 调用Trainer类的train方法进行训练，返回训练指标和模型路径
                metrics, model_path = trainer.train(logger)
                json_result.append(metrics)  # 将训练指标记录到JSON结果中

                # 更新最佳指标和最佳参数值
                if metrics['metric'] > best_metric['metric']:
                    best_metric['metric'] = metrics['metric']
                    best_v = v
                    best_model_path = model_path

            best_param.append({p: best_v})  # 记录最佳参数
            print("best model path:", best_model_path)  # 打印最佳模型路径
            print("best metric:", best_metric)  # 打印最佳指标
            logger.info("best model path:" + best_model_path)  # 记录最佳模型路径到日志
            logger.info("best param " + p + ": " + str(best_v))  # 记录最佳参数到日志
            logger.info("best metric:" + str(best_metric))  # 记录最佳指标到日志
            logger.info('--------------------------------------\n')  # 记录分隔线到日志

        # 将JSON结果写入文件
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)


# 一个生成浮点数范围的方法

# 一个run类：
#       3个方法：getFileLogger 方法（日志相关）
#               config2dict 方法（将配置信息转换为字典形式）
#               main函数（）-创建参数日志目录 (param_log_dir)，如果不存在则创建。
#                          设置参数日志文件路径 (param_log_file) 并获取文件日志记录器 (logger)。
#                          定义训练参数 (train_param)，在本例中只包含学习率 (lr)。
#                          针对每个参数 (train_param)，循环尝试不同的参数值进行训练，记录最佳模型路径、最佳参数和最佳指标。
#                          将训练过程中的指标记录到 JSON 文件 (json_path) 中。
#
#