# Linux跨平台二进制函数识别

## Gemini复现

> 以学院服务器环境和路径为例

### 环境配置

> 官方代码基于Python 2.7 和 Tensorflow 1.4 实现，在此基础上做了微调

- Linux (CPU: 32 cores, GPU: Tesla T4 * 8, Memory: 187G)
- Python 3.6.8
- Tensorflow 2.0.0 (using v1 API)
- CUDA 10.0

### 数据描述

1. 源文件

- train.func.json: 训练集函数记录，220w+
- test.func.update.json: 测试集函数记录，62w+
- train.group.csv: 训练集函数分组标记记录，（组id，函数id，...，函数id），7w
- test.question.csv: 测试集目标/候选函数记录，（目标函数id，候选函数id，...，候选函数id），1w

2. 中间文件

- train.feature.json: 训练集函数特征
- test.feature.json: 测试集函数特征
- test.dict.json: 测试集嵌入特征

3.结果文件

- submission.*.csv：匹配结果

### 复现步骤

> 当前目录为代码根目录
> 部分数据分析与文件操作位于 linux_binary.ipynb（可本地使用）

1. 创建并配置实验环境

```shell
conda create --prefix=/public/ly/gemini python=3.6.8
conda activate /public/ly/gemini
pip -r requirements.txt
```

2. 特征工程，将原始数据集处理为Gemini特征

```shell
cd application
python feature_engineering.py /data/bin_data train.func.json
python feature_engineering.py /data/bin_data test.func.update.json
```

3. 模型构建与训练，基于训练集构建GNN模型

```shell
cd ..
python train.py /data/bin_data train.feature.json
```

4. 在测试集函数特征基础上，创建函数id->ACFG特征+邻接矩阵的字典，为嵌入做准备

```shell
cd application
python pre_sim.py  /data/bin_data test.feature.json
```
5. 基于训练好的模型对测试集函数记录进行相似度匹配，并保存结果

```shell
./batch_run.sh
# python similarity.py /data/bin_data test.dict.json
```

### 问题记录

> 出发点基于对MAP的影响以及能否符合复现要求。复现机器配置（CPU 8核，内存 32G，单显卡 P40），复现完整运行时间不超过72小时，其中训练时间不超过64小时，预测时间不超过8小时。


1. 特征

- 提取ACFG特征时，体现CFG结构的特征——子代数量(No. of springs)需要递归获取，效率较低（存在函数有超过6000个block）。因此当前方案用当前节点的直接子代数量来代替，特征提取部分耗时约2.5个小时。(application/feature_engineering.py)

2. 模型

- 模型训练过程进行了训练集、验证集、测试集8：1：1的划分，但经过GNN初始化和第1个epoch之后，内存占用就已超过130G。因此当前方案使用仅经过1个epoch之后保存的模型，耗时约50分钟。(train.py)

- 对测试集进行相似性匹配的过程中，要对10000个目标函数各自进行10000个候选函数的匹配（CFG构建及向量嵌入），也就是同样的操作要进行10^8次。当前方案试图对已计算的嵌入向量进行保存(train.py: 148)，并通过对测试集函数记录进行分割（test.question.{0-7}.csv）来使用8个GPU并行计算并分别保存结果(submission.{0-7}.*.csv)，仍需耗时约7小时。(application/similarity.py)


### 参考资料

- [Neural Network-based Graph Embedding for Cross-Platform Binary Code Similarity Detection (CCS 2017, CCF A)](https://x-16xb.github.io/2020/01/10/Neural-Network-based-Graph-Embedding-for-Cross-Platform/)
- [Code - DNN Binary Code Similarity Detection](https://github.com/xiaojunxu/dnn-binary-code-similarity)
- [Code - ACFG Extraction and Similarity Calculation](https://github.com/shouguoyang/Gemini)
