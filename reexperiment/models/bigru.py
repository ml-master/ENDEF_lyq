import json
import os
import torch
import tqdm
import torch.nn as nn
import numpy as np

from utils.dataloader import get_dataloader
from .layers import *
from sklearn.metrics import *
from transformers import BertModel, BertConfig

from utils.utils import Recorder,data2gpu, Averager, metrics

class BiGRUModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, num_layers):
        super(BiGRUModel, self).__init__()
        self.fea_size = emb_dim


        # self.bert = BertModel.from_pretrained('bert-base-uncased').requires_grad_(False)

        local_model_path = '/data/lj_data/bert'  # 这里填写您解压模型文件的实际路径
        config = BertConfig.from_pretrained(local_model_path)
        self.bert = BertModel.from_pretrained(local_model_path, config=config).requires_grad_(False)

        self.embedding = self.bert.embeddings
        
        self.rnn = nn.GRU(input_size = emb_dim,
                          hidden_size = self.fea_size,
                          num_layers = num_layers, 
                          batch_first = True, 
                          bidirectional = True)

        input_shape = self.fea_size * 2
        self.attention = MaskAttention(input_shape)
        self.mlp = MLP(input_shape, mlp_dims, dropout)
    
    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        feature = self.embedding(inputs)
        feature, _ = self.rnn(feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(feature)
        return torch.sigmoid(output.squeeze(1))


class Trainer():
    def __init__(self,
                 config
                 ):
        self.config = config
        
        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        

    def train(self, logger = None):
        if(logger):
            logger.info('start training......')
        self.model = BiGRUModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['model']['mlp']['dropout'], num_layers=1)
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'], self.config['batchsize'], shuffle=True, use_endef=False, aug_prob=self.config['aug_prob'])
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            # ------------------------------------------------------------------------------------------
            # 检查root_path是否为./data/，如果是，则添加origin前缀
            if self.config['root_path'] == './data/':
                origin_prefix = 'origin_'
            else:
                origin_prefix = ''

            # 保存训练损失
            with open(os.path.join(self.save_path, origin_prefix + 'train_losses.json'), 'a') as f:
                json.dump({'epoch': epoch, 'loss': avg_loss.item()}, f)
                f.write('\n')

            results = self.test(val_loader)

            # 保存验证结果
            with open(os.path.join(self.save_path, origin_prefix + 'validation_results.json'), 'a') as f:
                json.dump({'epoch': epoch, 'results': results}, f)
                f.write('\n')

            # ------------------------------------------------------------------------

            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_bigru.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bigru.pkl')))

        test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'])
        future_results = self.test(test_future_loader)
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bigru.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)



#  两个类：BiGRUModel 类  和  Trainer 类
#        BiGRUModel 类：（两个方法-初始化、向前传播）
#          实现了一个BiGRU模型，使用预训练的BERT模型作为嵌入层，通过双向GRU层提取特征，然后经过注意力层和MLP层进行分类预测。（一个初始化方法和一个向前传播模型）
#
#        Trainer 类：   （三个方法-初始化，tarin，test）
#          实现了模型的训练和测试流程。
#          -在初始化方法中，根据配置设置保存模型参数的路径。
#          -train 方法负责模型的训练过程，包括损失函数的定义、优化器的选择、训练数据和验证数据的加载及遍历，以及根据验证集结果决定是否保存模型参数。
#          -test 方法用于模型在测试集上的评估，计算并返回评估指标。
#
#
#
