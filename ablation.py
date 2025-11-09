"""
消融实验脚本
测试不同组件对模型性能的影响
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import Transformer
from utils import (TextDataset, generate_dummy_data, load_iwslt_dataset, create_masks,
                   LearningRateScheduler, plot_training_curves)
from config import Config
import os


class AblationTransformer(nn.Module):
    """支持消融实验的 Transformer 变体"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len, 
                 dropout=0.1, use_positional_encoding=True, use_residual=True,
                 use_layernorm=True, use_ffn=True, use_multi_head=True):
        super(AblationTransformer, self).__init__()
        from model import PositionalEncoding
        
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.use_ffn = use_ffn
        self.use_multi_head = use_multi_head
        
        # 词嵌入（支持不同的源语言和目标语言词汇表）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        ####关键代码：控制是否启用位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoding = nn.Identity()####禁用时用恒等映射代替，不添加位置信息
        
        # Encoder
        ###控制是否使用多头注意力
        if use_multi_head:
            self.encoder_layers = nn.ModuleList([
                AblationEncoderBlock(d_model, n_heads, d_ff, dropout,
                                   use_residual, use_layernorm, use_ffn) 
                for _ in range(n_layers)
            ])
        else:
            # 单头注意力
            self.encoder_layers = nn.ModuleList([
                AblationEncoderBlock(d_model, 1, d_ff, dropout,
                                   use_residual, use_layernorm, use_ffn) 
                for _ in range(n_layers)
            ])
        
        # Decoder
        ####控制是否启用多头注意力，与编码器一致
        if use_multi_head:
            self.decoder_layers = nn.ModuleList([
                AblationDecoderBlock(d_model, n_heads, d_ff, dropout,
                                   use_residual, use_layernorm, use_ffn) 
                for _ in range(n_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                AblationDecoderBlock(d_model, 1, d_ff, dropout,
                                   use_residual, use_layernorm, use_ffn) 
                for _ in range(n_layers)
            ])
        
        # 输出层（输出目标语言词汇表大小）
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """前向传播"""
        import math
        
        # Encoder（使用源语言嵌入）
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        if self.use_positional_encoding:
            src_emb = self.pos_encoding(src_emb)
        enc_output = self.dropout(src_emb)
        
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder（使用目标语言嵌入）
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        if self.use_positional_encoding:
            tgt_emb = self.pos_encoding(tgt_emb)
        dec_output = self.dropout(tgt_emb)
        
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输出（目标语言词汇表）
        output = self.fc_out(dec_output)
        return output


class AblationEncoderBlock(nn.Module):
    """支持消融的 Encoder 块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 use_residual=True, use_layernorm=True, use_ffn=True):
        super(AblationEncoderBlock, self).__init__()
        from model import MultiHeadAttention, PositionwiseFeedForward
        
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.use_ffn = use_ffn
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        ##是否启用前馈网络
        if use_ffn:
            self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        else:
            self.ffn = nn.Identity()
        ###是否启用层归一化
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        ####是否移除残差网络
        if self.use_residual:
            x = self.norm1(x + self.dropout1(attn_output))
        else:
            x = self.norm1(self.dropout1(attn_output))
        
        ffn_output = self.ffn(x)
        ##是否启用残差网络（和上一层一致）
        if self.use_residual:
            x = self.norm2(x + self.dropout2(ffn_output))
        else:
            x = self.norm2(self.dropout2(ffn_output))
        
        return x


class AblationDecoderBlock(nn.Module):
    """支持消融的 Decoder 块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 use_residual=True, use_layernorm=True, use_ffn=True):
        super(AblationDecoderBlock, self).__init__()
        from model import MultiHeadAttention, PositionwiseFeedForward
        
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.use_ffn = use_ffn
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        ###是否移除前馈
        if use_ffn:
            self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        else:
            self.ffn = nn.Identity()
        #是否启用层归一化
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        ##是否启用残差
        if self.use_residual:
            x = self.norm1(x + self.dropout1(attn_output))
        else:
            x = self.norm1(self.dropout1(attn_output))
        
        cross_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        ##是否使用残差，同上一层
        if self.use_residual:
            x = self.norm2(x + self.dropout2(cross_output))
        else:
            x = self.norm2(self.dropout2(cross_output))
        
        ffn_output = self.ffn(x)
        ##是否使用残差同上一层
        if self.use_residual:
            x = self.norm3(x + self.dropout3(ffn_output))
        else:
            x = self.norm3(self.dropout3(ffn_output))
        
        return x


def train_ablation(model, train_loader, val_loader, optimizer, criterion, scheduler, device, config, 
                   num_epochs=10, src_pad_idx=0, tgt_pad_idx=0):
    """训练消融实验模型"""
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for src, tgt_input, tgt_output in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            src_mask, tgt_mask, _ = create_masks(src, tgt_input, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / num_batches
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for src, tgt_input, tgt_output in val_loader:
                src = src.to(device)
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)
                
                src_mask, tgt_mask, _ = create_masks(src, tgt_input, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
                src_mask = src_mask.to(device)
                tgt_mask = tgt_mask.to(device)
                
                output = model(src, tgt_input, src_mask, tgt_mask)
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.view(-1)
                
                loss = criterion(output, tgt_output)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_losses.append(val_loss)
        model.train()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    return train_losses, val_losses


def run_ablation_study(config):
    """运行消融实验"""
    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)
    
    # 加载数据
    use_iwslt = config.ablation_use_iwslt_data
    if use_iwslt:
        print("\nLoading IWSLT dataset for ablation study...")
        # 检查训练文件是否存在
        if not os.path.exists(config.train_src_path):
            print(f"Warning: IWSLT training file not found: {config.train_src_path}")
            print("Falling back to dummy data...")
            use_iwslt = False
    
    if use_iwslt:
        train_data, dev_data, _, src_vocab, tgt_vocab = load_iwslt_dataset(
            train_src_path=config.train_src_path,
            train_tgt_path=config.train_tgt_path,
            dev_src_path=config.dev_src_path if os.path.exists(config.dev_src_path) else None,
            dev_tgt_path=config.dev_tgt_path if os.path.exists(config.dev_tgt_path) else None,
            min_freq=config.min_freq,
            max_vocab_size=config.max_vocab_size,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang
        )
        # 使用训练集的一部分作为验证集（如果开发集不存在）
        if not dev_data:
            split_idx = len(train_data) // 10
            dev_data = train_data[:split_idx]
            train_data = train_data[split_idx:]
        
        train_dataset = TextDataset(train_data, src_vocab, tgt_vocab, max_len=config.max_len)
        val_dataset = TextDataset(dev_data, src_vocab, tgt_vocab, max_len=config.max_len)
        src_pad_idx = src_vocab.get('<pad>', 0)
        tgt_pad_idx = tgt_vocab.get('<pad>', 0)
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)
    else:
        print("\nUsing dummy data for ablation study...")
        # 使用虚拟数据
        train_data, vocab = generate_dummy_data(num_samples=3000, vocab_size=1000)
        val_data, _ = generate_dummy_data(num_samples=1000, vocab_size=1000)
        train_dataset = TextDataset(train_data, vocab, vocab, max_len=config.max_len)
        val_dataset = TextDataset(val_data, vocab, vocab, max_len=config.max_len)
        src_vocab_size = len(vocab)
        tgt_vocab_size = len(vocab)
        src_pad_idx = vocab.get('<pad>', 0)
        tgt_pad_idx = vocab.get('<pad>', 0)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 消融实验配置
    ablation_configs = [
        {
            'name': 'Full Model',
            'use_positional_encoding': True,   ## 启用位置编码
            'use_residual': True,              #启用残差连接
            'use_layernorm': True,             #启用曾归一化
            'use_ffn': True,                   #启用前馈
            'use_multi_head': True,            #启用多头
        },
        {
            'name': 'No Positional Encoding',
            'use_positional_encoding': False,
            'use_residual': True,
            'use_layernorm': True,
            'use_ffn': True,
            'use_multi_head': True,
        },
        {
            'name': 'No Residual Connections',
            'use_positional_encoding': True,
            'use_residual': False,
            'use_layernorm': True,
            'use_ffn': True,
            'use_multi_head': True,
        },
        {
            'name': 'No LayerNorm',
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': False,
            'use_ffn': True,
            'use_multi_head': True,
        },
        {
            'name': 'No FFN',
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': True,
            'use_ffn': False,
            'use_multi_head': True,
        },
        {
            'name': 'Single Head Attention',
            'use_positional_encoding': True,
            'use_residual': True,
            'use_layernorm': True,
            'use_ffn': True,
            'use_multi_head': False,
        },
    ]
    
    results = {}
    
    for ablation_config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {ablation_config['name']}")
        print(f"{'='*60}")
        
        # 创建模型
        model = AblationTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_len=config.max_len,
            dropout=config.dropout,
            **{k: v for k, v in ablation_config.items() if k != 'name'}
        ).to(config.device)
        
        # 统计参数
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        
        # 训练
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = LearningRateScheduler(
            optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps
        )
        
        train_losses, val_losses = train_ablation(
            model, train_loader, val_loader, optimizer, criterion, scheduler,
            config.device, config, num_epochs=config.ablation_num_epochs,
            src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx
        )
        
        results[ablation_config['name']] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'num_params': num_params
        }
        
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Val Loss: {val_losses[-1]:.4f}")
        print(f"Best Val Loss: {min(val_losses):.4f}")
    
    # 保存结果
    os.makedirs(config.ablation_results_dir, exist_ok=True)
    
    # 保存 JSON
    results_json = {k: {
        'final_train_loss': v['final_train_loss'],
        'final_val_loss': v['final_val_loss'],
        'best_val_loss': v['best_val_loss'],
        'num_params': v['num_params']
    } for k, v in results.items()}
    
    results_path = os.path.join(config.ablation_results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 训练损失对比
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # 验证损失对比
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # 最终损失对比（条形图）
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    final_val_losses = [results[n]['final_val_loss'] for n in names]
    plt.barh(names, final_val_losses)
    plt.xlabel('Final Validation Loss')
    plt.title('Final Validation Loss Comparison')
    plt.grid(True, axis='x')
    
    # 最佳验证损失对比
    plt.subplot(2, 2, 4)
    best_val_losses = [results[n]['best_val_loss'] for n in names]
    plt.barh(names, best_val_losses)
    plt.xlabel('Best Validation Loss')
    plt.title('Best Validation Loss Comparison')
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    comparison_path = os.path.join(config.ablation_results_dir, 'comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Ablation Study Complete!")
    print(f"{'='*60}")
    print("\nResults Summary:")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Final Val Loss: {result['final_val_loss']:.4f}")
        print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"  Parameters: {result['num_params']:,}")
        print()
    
    print(f"Results saved to {config.ablation_results_dir}/")


if __name__ == '__main__':
    config = Config()
    run_ablation_study(config)

