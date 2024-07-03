from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np


# 记录器类，用于在训练过程中根据评估指标决定是否保存模型参数
class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}  # 最佳指标记录
        self.cur = {'metric': 0}  # 当前指标记录
        self.maxindex = 0  # 最佳指标记录时的迭代次数
        self.curindex = 0  # 当前迭代次数
        self.early_step = early_step  # 提前停止的步数阈值

    # 添加新的指标值并判断是否保存模型参数
    def add(self, x):
        self.cur = x  # 更新当前指标记录
        self.curindex += 1  # 当前迭代次数加一
        print("当前指标", self.cur)
        return self.judge()  # 调用判断函数

    # 判断是否保存模型参数
    def judge(self):
        if self.cur['metric'] > self.max['metric']:  # 如果当前指标优于最佳指标
            self.max = self.cur  # 更新最佳指标记录
            self.maxindex = self.curindex  # 更新最佳指标时的迭代次数
            self.showfinal()  # 打印最佳指标信息
            return 'save'  # 返回保存标记
        self.showfinal()  # 打印当前指标信息
        if self.curindex - self.maxindex >= self.early_step:  # 如果超过提前停止的步数阈值
            return 'esc'  # 返回提前停止标记
        else:
            return 'continue'  # 继续训练

    # 打印最佳指标信息
    def showfinal(self):
        print("最佳指标", self.max)


# 计算各种评估指标的函数
def metrics(y_true, y_pred):
    all_metrics = {}

    # 计算ROC AUC
    all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)

    # 将预测值四舍五入为整数类型
    y_pred = np.around(np.array(y_pred)).astype(int)

    # 计算F1分数
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)

    # 计算召回率
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)

    # 计算精确率
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)

    # 计算准确率
    all_metrics['acc'] = accuracy_score(y_true, y_pred)

    # 计算真新闻被正确分类的概率（真新闻的准确率）
    true_positive = sum(y_true == 1 and y_pred == 1 for y_true, y_pred in zip(y_true, y_pred))
    true_negative = sum(y_true == 0 and y_pred == 0 for y_true, y_pred in zip(y_true, y_pred))


    # 计算假新闻被正确分类的概率（假新闻的准确率）
    false_positive = sum(y_true == 0 and y_pred == 1 for y_true, y_pred in zip(y_true, y_pred))
    false_negative = sum(y_true == 1 and y_pred == 0 for y_true, y_pred in zip(y_true, y_pred))

    all_metrics['accuracy_real'] = true_positive / (true_positive+false_negative)
    all_metrics['accuracy_fake'] = true_negative /(true_negative+false_positive)

    return all_metrics


# 将批次数据移动到GPU的函数
def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'entity': batch[2].cuda(),
            'entity_masks': batch[3].cuda(),
            'label': batch[4].cuda(),
            # 'year': batch[5].cuda(),
            # 'emotion': batch[6].cuda()
        }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'entity': batch[2],
            'entity_masks': batch[3],
            'label': batch[4],
            # 'year': batch[5],
            # 'emotion': batch[6]
        }
    return batch_data


# 平均值计算类
class Averager():

    def __init__(self):
        self.n = 0  # 样本数量
        self.v = 0  # 平均值

    # 添加新的样本值，更新平均值
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    # 返回当前的平均值
    def item(self):
        return self.v



# 两个类Recorder 类和Averager 类 + 两个函数
#      Recorder 类：
#       -   __init__ 方法初始化了记录器，包括最佳指标记录、当前指标记录、迭代次数等信息。
#       -   add 方法用于添加新的指标并判断是否需要保存模型参数。
#       -   judge 方法根据当前指标和最佳指标决定是否提前停止训练或继续训练。
#       -   showfinal 方法用于打印当前和最佳指标信息。
#      Averager 类：
#       -   用于计算样本的平均值，包括 add 方法用于添加新的样本值并更新平均值，item 方法返回当前的平均值
#
#      metrics 函数：
#       -    计算了各种评估指标，包括ROC AUC、F1 分数、召回率、精确率和准确率
#
#      data2gpu 函数：
#       -     将批次数据移动到GPU上
#
#
#
#
#
#
#