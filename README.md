# ENDEF
论文全称：**Generalizing to the Future: Mitigating Entity Bias in Fake News Detection**

论文地址：https://dl.acm.org/doi/abs/10.1145/3477495.3531816 

官方公开代码：https://github.com/ICTMCG/ENDEF-SIGIR2022 

* 这篇论文介绍了一种称为**实体去偏差框架-ENDEF**的方法，旨在提升假新闻检测模型对未来数据的适应能力。研究发现现有方法忽视了现实世界中**特定实体**在不同
时间段**真实性的变化**，导致模型泛化能力下降。ENDEF框架通过**因果图**建模实体、新闻内容和真实性之间的关系，在训练中**减少了对特定实体的过度依赖**，从而显著提升了检测器的性能，并在实际测试中得到验证。

# Introduction
本工作旨在利用该论文所提出的方法，结合课程提供的**新的新闻数据集**，验证该方法的性能，评估其在假新闻检测领域的性能表现。

# Dataset
#### 这里使用到了两份数据集： 

数据集1是原论文中的英文数据集，来源于FakeNewsNet的GossipCop数据：https://github.com/KaiDMML/FakeNewsNet 

数据集2是课程提供的基于风格的新闻数据集：https://github.com/junyachen/Data-examples/blob/main/README.md#style-based-fake
#### 数据集处理：
数据集1（ENDEF_en/data）中原作者已经将数据进行划分和实体提取，但并未给出处理具体代码。 因此我对数据集2首先进行划分，然后使用texsmart工具提取其中的实体信息，详见代码（reexperiment/test.py和reexperiment/utils/texsmart.py）。处理后的数据集在（reexperiment/datanew）。

# Code
### Requirements
* Python 3.6
* PyTorch > 1.0
* Pandas
* Numpy
* Tqdm
* bert(预先在官网下载到本地，所有使用的模型都使用预训练的BERT模型作为嵌入层)
### File Structure
整个代码结构可以分成两大部分：
* ENDEF_en：这一部分包含了原作者的代码和数据集1，并未进行任何改动（除了为了与实验的其他部分保持一致而调整了一些训练参数）。这部分代码主要用于**复现原作者的实验结果**，确保实验的准确性。
* reexperiment：这部分涉及新数据集2的使用，并对原有代码进行了部分修改，以适应新数据集的结构。reexperiment 部分用于**测试新数据集的性能**，并进行了**消融实验**，即使用修改后的代码对数据集1进行训练，以评估不同组件对模型性能的影响。

下面以reexperiment结构为例介绍代码结构，ENDEF_en结构类似。
```python
├── ENDEF-glj<br>
│   └── readme.md<br>
│   ENDEF_en       (复现原文数据集)<br>
└── reexperiment(新数据集实验及消融实验)<br>
    ├── main.py<br>
    ├── grid_search.py<br>
    ├── test.py<br>
    ├── data       (原文数据集消融版)<br>
    │   ├── train.json<br>
    │   ├── val.json<br>
    │   ├── test.json<br>
    ├── datanew    (新数据集)<br>
    │   ├── train.json<br>
    │   ├── val.json<br>
    │   ├── test.json<br>
    ├── logs       (日志信息)<br>
    ├── models(训练)<br>
    │   ├── bigru.py<br>
    │   ├── bigruendef.py<br>
    │   └── ...<br>
    ├── param_model           (训练结果)<br>
    │	│   ├── origin... .json   (消融)<br>
    │   │   ├── ... .json         (新数据集)<br>
    │	│   └── ...<br>
    └── utils     (数据加载/实体提取/评价指标等等)<br>                
        ├── dataloader.py<br>
        ├── texsmart.py<br>
        ├── draw.py<br>
        └── utils.py<br>
```
# Run
因为此框架与方法无关，因此这里使用论文中相同的五种基础模型进行实验，即BiGRU、EANN、BERT、MDFEND和BERT-Emo，_endef后缀表示该方法与ENDEF框架结合。
```python  
# --model_name可选：bigru, bigru_endef, bert, bert_endef, bertemo, bertemo_endef, eann, eann_endef, mdfend, mdfend_endef
python exmain.py --gpu 0 --lr 0.0001 --model_name bigru 
```
# Results

| acc    | acc_real | acc_fake |
| ------ | -------- | -------- |
| 0.6890 | 0.7480   | 0.6300   |

