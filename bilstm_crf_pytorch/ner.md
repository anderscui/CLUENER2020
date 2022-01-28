# Named Entity Recognition

## 定义

维基百科的定义：**命名实体识别**（Named Entity Recognition，简称 NER），是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等，以及时间、数量、货币、比例数值等文字。指的是可以用专有名词（名称）标识的事物，一个命名实体一般代表唯一一个具体事物个体，包括人名、地名等。 

NER 是典型的**序列标注**任务，它强调"具体实体"，但也可以看作"信息抽取"任务，在有些应用中未必如此"具体"。

## 标注格式

IOB1、IOB2、IOE1、IOE2、IOBES、IO。 其中 IOB2 是最简洁明了且有效的（无歧义）。 

## 

## 主要方法

* 传统方法：规则（人工整理的模式与词典等）；统计方法，如 CRF、MEMM（Maximum Entropy Markov Model）
* 深度学习：BiLSTM 系列以及基于 Transformers 系列

