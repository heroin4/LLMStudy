# Transformer模型实现

基于论文 **"Attention is All You Need"** (Vaswani et al., 2017) 的完整PyTorch实现。

## 📋 目录

- [模型架构](#模型架构)
- [核心组件](#核心组件)
- [安装](#安装)
- [使用方法](#使用方法)
- [论文参数](#论文参数)
- [文件说明](#文件说明)

## 🏗️ 模型架构

Transformer采用编码器-解码器(Encoder-Decoder)架构，完全基于注意力机制，不使用循环神经网络(RNN)或卷积神经网络(CNN)。

```
输入序列 → Encoder → Decoder → 输出序列
```

### Encoder结构
- **输入嵌入层** + **位置编码**
- **N层Encoder Layer**，每层包含:
  - 多头自注意力机制 (Multi-Head Self-Attention)
  - 位置前馈网络 (Position-wise Feed-Forward Network)
  - 残差连接 (Residual Connection) + Layer Normalization

### Decoder结构
- **输出嵌入层** + **位置编码**
- **N层Decoder Layer**，每层包含:
  - 掩码多头自注意力机制 (Masked Multi-Head Self-Attention)
  - 编码器-解码器注意力 (Encoder-Decoder Attention)
  - 位置前馈网络 (Position-wise Feed-Forward Network)
  - 残差连接 + Layer Normalization

## 🔧 核心组件

### 1. 多头注意力机制 (Multi-Head Attention)

```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

- **缩放点积注意力**: 计算Query和Key的相似度，对Value进行加权求和
- **多头机制**: 将注意力分成多个头，每个头学习不同的表示子空间
- **参数**: `d_model=512`, `num_heads=8`, `d_k=d_v=64`

### 2. 位置编码 (Positional Encoding)

由于Transformer不包含循环或卷积结构，需要注入位置信息:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. 位置前馈网络 (Position-wise Feed-Forward Network)

```python
FFN(x) = max(0, xW1 + b1)W2 + b2
```

- 两层全连接网络，中间使用ReLU激活
- 参数: `d_model=512`, `d_ff=2048`

### 4. 残差连接和Layer Normalization

每个子层使用残差连接，然后进行Layer Normalization:

```python
LayerNorm(x + Sublayer(x))
```

## 📦 安装

### 环境要求
- Python 3.7+
- PyTorch 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装:

```bash
pip install torch numpy
```

## 🚀 使用方法

### 基本使用

```python
from transformer import Transformer
import torch

# 创建模型（使用论文中的参数）
model = Transformer(
    src_vocab_size=10000,      # 源语言词汇表大小
    tgt_vocab_size=10000,      # 目标语言词汇表大小
    d_model=512,               # 模型维度
    num_heads=8,               # 注意力头数
    d_ff=2048,                 # 前馈网络维度
    num_encoder_layers=6,      # Encoder层数
    num_decoder_layers=6,      # Decoder层数
    max_seq_len=100,           # 最大序列长度
    dropout=0.1                # Dropout率
)

# 准备输入数据
batch_size = 2
src_seq_len = 10
tgt_seq_len = 12

src = torch.randint(1, 10000, (batch_size, src_seq_len))  # 源序列
tgt = torch.randint(1, 10000, (batch_size, tgt_seq_len))  # 目标序列

# 前向传播
output = model(src, tgt)  # 输出形状: (batch_size, tgt_seq_len, tgt_vocab_size)
```

### 运行示例

```bash
python example.py
```

示例包含:
1. **基本翻译任务**: 展示模型的输入输出
2. **训练步骤**: 展示如何训练模型
3. **组件测试**: 测试各个核心组件

### 训练示例

```python
import torch.nn as nn
import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 训练循环
model.train()
for epoch in range(num_epochs):
    for src, tgt_input, tgt_output in dataloader:
        # 前向传播
        output = model(src, tgt_input)
        
        # 计算损失
        loss = criterion(output.reshape(-1, tgt_vocab_size), 
                        tgt_output.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 📊 论文参数

论文中使用的标准配置:

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 512 | 模型维度 |
| `num_heads` | 8 | 注意力头数 |
| `d_ff` | 2048 | 前馈网络隐藏层维度 |
| `num_layers` | 6 | Encoder和Decoder的层数 |
| `dropout` | 0.1 | Dropout率 |
| `d_k` | 64 | Key的维度 (d_model / num_heads) |
| `d_v` | 64 | Value的维度 (d_model / num_heads) |

论文中还提到了一个更大的模型配置:
- `d_model=1024`, `d_ff=4096`, `num_heads=16`

## 📁 文件说明

```
transformer-implementation/
├── transformer.py      # Transformer模型完整实现
├── example.py         # 使用示例和测试代码
├── requirements.txt   # 依赖包列表
└── README.md         # 项目说明文档
```

### transformer.py 包含的类:

- `MultiHeadAttention`: 多头注意力机制
- `PositionWiseFeedForward`: 位置前馈网络
- `PositionalEncoding`: 位置编码
- `EncoderLayer`: Encoder层
- `DecoderLayer`: Decoder层
- `Encoder`: 完整的Encoder
- `Decoder`: 完整的Decoder
- `Transformer`: 完整的Transformer模型

## 🔑 关键特性

✅ **完整实现**: 包含论文中的所有核心组件  
✅ **详细注释**: 中文注释，易于理解  
✅ **模块化设计**: 每个组件独立实现，便于复用  
✅ **掩码机制**: 实现了padding掩码和因果掩码  
✅ **位置编码**: 使用正弦和余弦函数的位置编码  
✅ **可扩展**: 易于修改和扩展到其他任务  

## 📚 参考资料

- 论文: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- 作者: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- 发表: NIPS 2017

## 💡 应用场景

Transformer模型广泛应用于:
- 机器翻译 (Machine Translation)
- 文本摘要 (Text Summarization)
- 问答系统 (Question Answering)
- 文本生成 (Text Generation)
- 语言模型 (Language Modeling)
- 以及更多NLP任务...

## 🎯 下一步

可以基于此实现进行:
1. 在真实数据集上训练（如WMT翻译数据集）
2. 实现学习率调度器（Warmup + Decay）
3. 添加Beam Search解码
4. 实现Label Smoothing
5. 添加模型保存和加载功能
6. 可视化注意力权重

## 📝 许可

本项目仅用于学习和研究目的。
