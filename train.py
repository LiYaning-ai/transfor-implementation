"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from model import Transformer
from utils import (TextDataset, load_iwslt_dataset, create_masks,
                   LearningRateScheduler, plot_training_curves,
                   save_model, load_model)
from config import Config
import json


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, config, src_pad_idx=0, tgt_pad_idx=0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(tqdm(dataloader, desc="Training")):
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        # 创建 masks
        src_mask, tgt_mask, _ = create_masks(src, tgt_input, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算损失（只计算非 padding 位置）
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        
        loss = criterion(output, tgt_output)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # 更新参数
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % config.print_freq == 0:
            print(f'Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {scheduler.get_lr():.6f}')
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, src_pad_idx=0, tgt_pad_idx=0):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in tqdm(dataloader, desc="Validating"):
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # 创建 masks
            src_mask, tgt_mask, _ = create_masks(src, tgt_input, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            
            # 前向传播
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 按模块统计
    print("\nParameters by module:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")


def save_vocab(vocab, path):
    """保存词汇表到文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    """从文件加载词汇表"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    config = Config()
    
    print("=" * 50)
    print("Transformer Training with IWSLT 2017 Dataset")
    print("=" * 50)
    
    # 加载 IWSLT 数据集
    print("\nLoading IWSLT dataset...")
    # 检查文件是否存在
    if not os.path.exists(config.train_src_path):
        raise FileNotFoundError(f"Training source file not found: {config.train_src_path}\n"
                              f"Please download the IWSLT 2017 dataset and place it in the correct directory.\n"
                              f"See IWSLT_USAGE.md for details.")
    if not os.path.exists(config.train_tgt_path):
        raise FileNotFoundError(f"Training target file not found: {config.train_tgt_path}\n"
                              f"Please download the IWSLT 2017 dataset and place it in the correct directory.\n"
                              f"See IWSLT_USAGE.md for details.")
    
    train_data, dev_data, test_data, src_vocab, tgt_vocab = load_iwslt_dataset(
        train_src_path=config.train_src_path,
        train_tgt_path=config.train_tgt_path,
        dev_src_path=config.dev_src_path if os.path.exists(config.dev_src_path) else None,
        dev_tgt_path=config.dev_tgt_path if os.path.exists(config.dev_tgt_path) else None,
        test_src_path=config.test_src_path if os.path.exists(config.test_src_path) else None,
        test_tgt_path=config.test_tgt_path if os.path.exists(config.test_tgt_path) else None,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang
    )
    
    # 保存词汇表
    os.makedirs(config.vocab_dir, exist_ok=True)
    src_vocab_path = os.path.join(config.vocab_dir, f'src_vocab_{config.src_lang}.json')
    tgt_vocab_path = os.path.join(config.vocab_dir, f'tgt_vocab_{config.tgt_lang}.json')
    save_vocab(src_vocab, src_vocab_path)
    save_vocab(tgt_vocab, tgt_vocab_path)
    print(f"\nVocabularies saved to {config.vocab_dir}")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_data, src_vocab, tgt_vocab, max_len=config.max_len)
    val_dataset = TextDataset(dev_data if dev_data else train_data[:1000], src_vocab, tgt_vocab, max_len=config.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 获取 pad_idx
    src_pad_idx = src_vocab.get('<pad>', 0)
    tgt_pad_idx = tgt_vocab.get('<pad>', 0)
    
    # 创建模型
    print("\nCreating model...")
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout
    ).to(config.device)
    
    # 统计参数
    print("\n" + "=" * 50)
    print("Model Parameters:")
    print("=" * 50)
    count_parameters(model)
    
    # 损失函数（忽略 padding）
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    
    # 优化器（AdamW）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = LearningRateScheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )
    
    # 训练循环
    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, 
                                 config.device, config, src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, config.device, 
                           src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
        val_losses.append(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_lr():.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.save_dir, 'best_model.pt')
            # 保存完整的检查点（包括词汇表信息）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': {
                    'src_vocab_size': len(src_vocab),
                    'tgt_vocab_size': len(tgt_vocab),
                    'd_model': config.d_model,
                    'n_heads': config.n_heads,
                    'n_layers': config.n_layers,
                    'd_ff': config.d_ff,
                    'max_len': config.max_len,
                    'dropout': config.dropout
                }
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved to {save_path}")
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': {
                    'src_vocab_size': len(src_vocab),
                    'tgt_vocab_size': len(tgt_vocab),
                    'd_model': config.d_model,
                    'n_heads': config.n_heads,
                    'n_layers': config.n_layers,
                    'd_ff': config.d_ff,
                    'max_len': config.max_len,
                    'dropout': config.dropout
                }
            }
            torch.save(checkpoint, checkpoint_path)
    
    # 绘制训练曲线
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    plot_path = os.path.join(config.log_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")


if __name__ == '__main__':
    main()

