"""
Transformer模型实现
基于论文 "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout比率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出的线性变换层
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Args:
            Q: Query矩阵 (batch_size, num_heads, seq_len, d_k)
            K: Key矩阵 (batch_size, num_heads, seq_len, d_k)
            V: Value矩阵 (batch_size, num_heads, seq_len, d_k)
            mask: 掩码矩阵 (batch_size, 1, seq_len, seq_len) 或 (batch_size, 1, 1, seq_len)
        Returns:
            output: 注意力输出
            attention_weights: 注意力权重
        """
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: 掩码
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # 线性变换并分割成多头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最后的线性变换
        output = self.W_o(attn_output)
        
        return output


class PositionWiseFeedForward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度
            dropout: dropout比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Encoder层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(EncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: 掩码
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 多头自注意力 + 残差连接 + Layer Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Decoder层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout比率
        """
        super(DecoderLayer, self).__init__()
        
        # 掩码多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: 解码器输入 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
        Returns:
            output: (batch_size, tgt_seq_len, d_model)
        """
        # 掩码多头自注意力 + 残差连接 + Layer Norm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # 编码器-解码器注意力 + 残差连接 + Layer Norm
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Encoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            num_layers: Encoder层数
            max_seq_len: 最大序列长度
            dropout: dropout比率
        """
        super(Encoder, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 多层Encoder
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入序列 (batch_size, seq_len)
            mask: 掩码
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 通过所有Encoder层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class Decoder(nn.Module):
    """Transformer Decoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            num_layers: Decoder层数
            max_seq_len: 最大序列长度
            dropout: dropout比率
        """
        super(Decoder, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 多层Decoder
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: 目标序列 (batch_size, tgt_seq_len)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            output: (batch_size, tgt_seq_len, d_model)
        """
        # 词嵌入 + 缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 通过所有Decoder层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_encoder_layers=6, num_decoder_layers=6, 
                 max_seq_len=5000, dropout=0.1):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度 (论文中为512)
            num_heads: 注意力头数 (论文中为8)
            d_ff: 前馈网络隐藏层维度 (论文中为2048)
            num_encoder_layers: Encoder层数 (论文中为6)
            num_decoder_layers: Decoder层数 (论文中为6)
            max_seq_len: 最大序列长度
            dropout: dropout比率 (论文中为0.1)
        """
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, 
            num_encoder_layers, max_seq_len, dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_decoder_layers, max_seq_len, dropout
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def make_src_mask(self, src):
        """
        创建源序列掩码（用于padding）
        Args:
            src: (batch_size, src_seq_len)
        Returns:
            src_mask: (batch_size, 1, 1, src_seq_len)
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """
        创建目标序列掩码（padding + 因果掩码）
        Args:
            tgt: (batch_size, tgt_seq_len)
        Returns:
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        batch_size, tgt_len = tgt.shape
        
        # Padding掩码
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
        
        # 因果掩码（下三角矩阵）
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        
        # 组合两个掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return tgt_mask
    
    def forward(self, src, tgt):
        """
        Args:
            src: 源序列 (batch_size, src_seq_len)
            tgt: 目标序列 (batch_size, tgt_seq_len)
        Returns:
            output: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 创建掩码
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.fc_out(decoder_output)
        
        return output
