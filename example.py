"""
Transformer模型使用示例
"""

import torch
import torch.nn as nn
from transformer import Transformer


def example_translation():
    """机器翻译任务示例"""
    
    # 模型参数（与论文一致）
    src_vocab_size = 10000  # 源语言词汇表大小
    tgt_vocab_size = 10000  # 目标语言词汇表大小
    d_model = 512           # 模型维度
    num_heads = 8           # 注意力头数
    d_ff = 2048             # 前馈网络维度
    num_encoder_layers = 6  # Encoder层数
    num_decoder_layers = 6  # Decoder层数
    max_seq_len = 100       # 最大序列长度
    dropout = 0.1           # Dropout率
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    
    # 打印模型结构
    print("=" * 80)
    print("Transformer模型结构")
    print("=" * 80)
    print(model)
    print("\n")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("\n")
    
    # 创建示例数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    # 源序列 (batch_size, src_seq_len)
    # 注意: 0用作padding token
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    
    # 目标序列 (batch_size, tgt_seq_len)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print("=" * 80)
    print("示例输入")
    print("=" * 80)
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"\n源序列示例:\n{src[0]}")
    print(f"\n目标序列示例:\n{tgt[0]}")
    print("\n")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    
    print("=" * 80)
    print("模型输出")
    print("=" * 80)
    print(f"输出形状: {output.shape}")
    print(f"期望形状: (batch_size={batch_size}, tgt_seq_len={tgt_seq_len}, tgt_vocab_size={tgt_vocab_size})")
    
    # 获取预测的token
    predictions = torch.argmax(output, dim=-1)
    print(f"\n预测序列形状: {predictions.shape}")
    print(f"预测序列示例:\n{predictions[0]}")
    print("\n")


def example_training_step():
    """训练步骤示例"""
    
    print("=" * 80)
    print("训练步骤示例")
    print("=" * 80)
    
    # 简化的模型参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_encoder_layers = 3
    num_decoder_layers = 3
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 创建训练数据
    batch_size = 4
    src_seq_len = 15
    tgt_seq_len = 20
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt_input = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    tgt_output = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 训练模式
    model.train()
    
    # 前向传播
    output = model(src, tgt_input)
    
    # 计算损失
    # output: (batch_size, tgt_seq_len, tgt_vocab_size)
    # tgt_output: (batch_size, tgt_seq_len)
    loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output.reshape(-1))
    
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("训练步骤完成！")
    print("\n")


def example_attention_visualization():
    """注意力机制可视化示例"""
    
    print("=" * 80)
    print("注意力机制组件测试")
    print("=" * 80)
    
    from transformer import MultiHeadAttention, PositionalEncoding
    
    # 测试多头注意力
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = mha(x, x, x)
    
    print(f"多头注意力输入形状: {x.shape}")
    print(f"多头注意力输出形状: {output.shape}")
    print(f"维度保持不变: {x.shape == output.shape}")
    print("\n")
    
    # 测试位置编码
    pos_encoding = PositionalEncoding(d_model, max_seq_len=100)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_encoding(x)
    
    print(f"位置编码输入形状: {x.shape}")
    print(f"位置编码输出形状: {output.shape}")
    print(f"维度保持不变: {x.shape == output.shape}")
    print("\n")


def main():
    """主函数"""
    
    print("\n")
    print("*" * 80)
    print("Transformer模型实现示例")
    print("基于论文: Attention is All You Need (Vaswani et al., 2017)")
    print("*" * 80)
    print("\n")
    
    # 示例1: 基本使用
    example_translation()
    
    # 示例2: 训练步骤
    example_training_step()
    
    # 示例3: 注意力机制测试
    example_attention_visualization()
    
    print("*" * 80)
    print("所有示例运行完成！")
    print("*" * 80)
    print("\n")


if __name__ == "__main__":
    main()
