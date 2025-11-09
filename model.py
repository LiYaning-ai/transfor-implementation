"""
Transformer 模型实现
包含完整的 Encoder-Decoder 架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        query: [batch_size, seq_len_q, d_model]
        key: [batch_size, seq_len_k, d_model]
        value: [batch_size, seq_len_v, d_model]
        mask: [batch_size, seq_len_q, seq_len_k] 或 None
        """
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_q, d_k]
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_k, d_k]
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L_v, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L_q, L_k]
        
        # 应用 mask
        if mask is not None:
            # mask 可能是 [B, 1, 1, L_k] 或 [B, 1, L_q, L_k]，需要广播到 [B, H, L_q, L_k]
            if mask.dim() == 4:
                mask = mask.squeeze(1)  # 移除头维度如果存在
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 添加头维度
            # 扩展到头数
            mask = mask.expand(-1, self.n_heads, -1, -1)  # [B, H, L_q, L_k]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)  # [B, H, L_q, d_k]
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [B, L_q, d_model]
        
        # 输出投影
        output = self.W_o(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderBlock(nn.Module):
    """Encoder 块：Self-Attention + FFN，包含残差连接和 LayerNorm"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # LayerNorm 层（每个子层一个）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, 1, seq_len] 或 None
        
        使用 Post-Norm 架构（LayerNorm 在残差连接之后）:
        x = LayerNorm(x + Dropout(Sublayer(x)))
        """
        # 1. Self-Attention 子层 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)  # 残差连接 + Dropout
        x = self.norm1(x)  # LayerNorm
        
        # 2. Feed-Forward 子层 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # 残差连接 + Dropout
        x = self.norm2(x)  # LayerNorm
        
        return x


class DecoderBlock(nn.Module):
    """Decoder 块：Masked Self-Attention + Cross-Attention + FFN，包含残差连接和 LayerNorm"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # LayerNorm 层（每个子层一个）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        x: [batch_size, tgt_len, d_model] - Decoder 输入
        enc_output: [batch_size, src_len, d_model] - Encoder 输出
        src_mask: [batch_size, 1, 1, src_len] 或 None (用于 cross-attention)
        tgt_mask: [batch_size, 1, tgt_len, tgt_len] 或 None (用于掩码未来位置)
        
        使用 Post-Norm 架构（LayerNorm 在残差连接之后）:
        x = LayerNorm(x + Dropout(Sublayer(x)))
        """
        # 1. Masked Self-Attention 子层 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)  # 残差连接 + Dropout
        x = self.norm1(x)  # LayerNorm
        
        # 2. Cross-Attention 子层 + 残差连接 + LayerNorm
        # src_mask 用于 cross-attention，维度应该是 [B, 1, 1, src_len]
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(cross_output)  # 残差连接 + Dropout
        x = self.norm2(x)  # LayerNorm
        
        # 3. Feed-Forward 子层 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)  # 残差连接 + Dropout
        x = self.norm3(x)  # LayerNorm
        
        return x


class Transformer(nn.Module):
    """完整的 Encoder-Decoder Transformer"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        
        # 词嵌入（源语言和目标语言使用不同的嵌入层）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 输出层（输出目标语言词汇表大小）
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [batch_size, src_len] - 源序列
        tgt: [batch_size, tgt_len] - 目标序列
        src_mask: [batch_size, 1, 1, src_len] 或 None (用于 encoder 和 decoder cross-attention)
        tgt_mask: [batch_size, 1, tgt_len, tgt_len] 或 None (用于 decoder self-attention)
        """
        # Encoder（使用源语言嵌入）
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        enc_output = self.dropout(src_emb)
        
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder（使用目标语言嵌入）
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        dec_output = self.dropout(tgt_emb)
        
        # src_mask 用于 decoder 的 cross-attention
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输出（目标语言词汇表）
        output = self.fc_out(dec_output)
        
        return output
    
    def count_parameters(self):
        """统计模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

