"""
工具函数：数据加载、可视化等
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Tuple, Dict


class TextDataset(Dataset):
    """文本数据集，支持源语言和目标语言不同的词汇表"""
    def __init__(self, data, src_vocab, tgt_vocab, max_len=128):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.src_pad_idx = src_vocab.get('<pad>', 0)
        self.tgt_pad_idx = tgt_vocab.get('<pad>', 0)
        self.sos_idx = tgt_vocab.get('<sos>', 1)
        self.eos_idx = tgt_vocab.get('<eos>', 2)
        self.src_unk_idx = src_vocab.get('<unk>', 3)
        self.tgt_unk_idx = tgt_vocab.get('<unk>', 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        # 转换为索引
        src_ids = [self.src_vocab.get(token, self.src_unk_idx) for token in src]
        tgt_ids = [self.tgt_vocab.get(token, self.tgt_unk_idx) for token in tgt]
        
        # 添加 SOS 和 EOS
        tgt_input = [self.sos_idx] + tgt_ids
        tgt_output = tgt_ids + [self.eos_idx]
        
        # 截断到最大长度
        src_ids = src_ids[:self.max_len-1]  # 留一个位置
        tgt_input = tgt_input[:self.max_len-1]
        tgt_output = tgt_output[:self.max_len-1]
        
        # Padding
        src_len = len(src_ids)
        tgt_len = len(tgt_input)
        
        src_ids = src_ids + [self.src_pad_idx] * (self.max_len - src_len)
        tgt_input = tgt_input + [self.tgt_pad_idx] * (self.max_len - tgt_len)
        tgt_output = tgt_output + [self.tgt_pad_idx] * (self.max_len - len(tgt_output))
        
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_input, dtype=torch.long), \
               torch.tensor(tgt_output, dtype=torch.long)


def generate_dummy_data(num_samples=1000, vocab_size=100, max_len=50):
    """生成虚拟数据用于测试"""
    np.random.seed(42)
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # 创建词汇表
    for i in range(4, vocab_size):
        vocab[f'word_{i}'] = i
    
    data = []
    for _ in range(num_samples):
        src_len = np.random.randint(5, max_len // 2)
        tgt_len = np.random.randint(5, max_len // 2)
        
        src = [f'word_{np.random.randint(4, vocab_size)}' for _ in range(src_len)]
        # 简单的复制任务：目标序列是源序列的逆序或复制
        if np.random.random() > 0.5:
            tgt = src[::-1]  # 逆序
        else:
            tgt = src.copy()  # 复制
        
        data.append((src, tgt))
    
    return data, vocab


def load_iwslt_text_file(file_path: str) -> List[str]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")
    
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和 XML 标签行（如 <doc>, </doc>, <url> 等）
            if not line:
                continue
            # 跳过 XML 标签行，但保留包含文本的标签行
            if line.startswith('<') and line.endswith('>') and not re.search(r'[^<>]', line):
                continue
            # 移除 XML 标签但保留文本内容
            line = re.sub(r'<[^>]+>', '', line)
            if line.strip():
                sentences.append(line.strip())
    return sentences


def load_iwslt_xml_file(file_path: str) -> List[str]:
    """加载 IWSLT XML 文件（开发集和测试集）"""
    if not os.path.exists(file_path):
        print(f"Warning: XML file not found: {file_path}, returning empty list")
        return []
    
    sentences = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # IWSLT XML 格式通常包含 <seg> 标签
        for seg in root.iter('seg'):
            text = seg.text
            if text:
                text = text.strip()
                if text:
                    sentences.append(text)
        
        # 如果没有找到 <seg> 标签，尝试其他常见标签
        if not sentences:
            for doc in root.iter('doc'):
                for seg in doc.iter('seg'):
                    text = seg.text
                    if text:
                        text = text.strip()
                        if text:
                            sentences.append(text)
    except ET.ParseError as e:
        print(f"Warning: Failed to parse XML file {file_path}: {e}")
        print("Trying to read as plain text...")
        # 如果 XML 解析失败，尝试作为纯文本读取
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ('seg' in line or not line.startswith('<')):
                    # 尝试提取文本内容
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        sentences.append(match.group(1).strip())
                    elif not line.startswith('<'):
                        sentences.append(line)
    
    return sentences


def tokenize_sentence(sentence: str, lower: bool = True) -> List[str]:
    """简单的分词函数（空格分词 + 标点符号处理）"""
    if lower:
        sentence = sentence.lower()
    
    # 在标点符号前后添加空格
    sentence = re.sub(r'([,.!?;:])', r' \1 ', sentence)
    # 移除多余空格
    sentence = re.sub(r'\s+', ' ', sentence)
    # 按空格分词
    tokens = sentence.strip().split()
    
    return tokens


def build_vocab(sentences: List[str], min_freq: int = 2, max_vocab_size: int = None) -> Dict[str, int]:
    """构建词汇表"""
    # 统计词频
    word_counts = Counter()
    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        word_counts.update(tokens)
    
    # 构建词汇表
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    # 按频率排序
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 添加高频词
    idx = 4
    for word, count in sorted_words:
        if count >= min_freq:
            if max_vocab_size is None or len(vocab) < max_vocab_size:
                vocab[word] = idx
                idx += 1
            else:
                break
    
    return vocab


def load_iwslt_dataset(
    train_src_path: str,
    train_tgt_path: str,
    dev_src_path: str = None,
    dev_tgt_path: str = None,
    test_src_path: str = None,
    test_tgt_path: str = None,
    min_freq: int = 2,
    max_vocab_size: int = None,
    src_lang: str = 'en',
    tgt_lang: str = 'de'
) -> Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]], Dict[str, int], Dict[str, int]]:
    """
    加载 IWSLT 2017 数据集
    
    Args:
        train_src_path: 训练集源语言文件路径
        train_tgt_path: 训练集目标语言文件路径
        dev_src_path: 开发集源语言文件路径（XML 格式）
        dev_tgt_path: 开发集目标语言文件路径（XML 格式）
        test_src_path: 测试集源语言文件路径（XML 格式）
        test_tgt_path: 测试集目标语言文件路径（XML 格式）
        min_freq: 最小词频
        max_vocab_size: 最大词汇表大小
        src_lang: 源语言代码
        tgt_lang: 目标语言代码
    
    Returns:
        train_data: 训练数据列表，每个元素为 (src_tokens, tgt_tokens)
        dev_data: 开发数据列表
        test_data: 测试数据列表
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
    """
    print("Loading IWSLT dataset...")
    
    # 加载训练集
    print(f"Loading training set from {train_src_path} and {train_tgt_path}")
    train_src_sentences = load_iwslt_text_file(train_src_path)
    train_tgt_sentences = load_iwslt_text_file(train_tgt_path)
    ####截取相同长度
    min_len = 10000
    train_src_sentences = train_src_sentences[:min_len]
    train_tgt_sentences = train_tgt_sentences[:min_len]

    assert len(train_src_sentences) == len(train_tgt_sentences), \
        f"Source and target training files have different lengths: {len(train_src_sentences)} vs {len(train_tgt_sentences)}"
    
    # 加载开发集（如果有）
    dev_data = []
    if dev_src_path and dev_tgt_path:
        print(f"Loading dev set from {dev_src_path} and {dev_tgt_path}")
        dev_src_sentences = load_iwslt_xml_file(dev_src_path)
        dev_tgt_sentences = load_iwslt_xml_file(dev_tgt_path)
        
        assert len(dev_src_sentences) == len(dev_tgt_sentences), \
            f"Source and target dev files have different lengths: {len(dev_src_sentences)} vs {len(dev_tgt_sentences)}"
        
        for src, tgt in zip(dev_src_sentences, dev_tgt_sentences):
            dev_data.append((tokenize_sentence(src), tokenize_sentence(tgt)))
    
    # 加载测试集（如果有）
    test_data = []
    if test_src_path and test_tgt_path:
        print(f"Loading test set from {test_src_path} and {test_tgt_path}")
        test_src_sentences = load_iwslt_xml_file(test_src_path)
        test_tgt_sentences = load_iwslt_xml_file(test_tgt_path)
        
        assert len(test_src_sentences) == len(test_tgt_sentences), \
            f"Source and target test files have different lengths: {len(test_src_sentences)} vs {len(test_tgt_sentences)}"
        
        for src, tgt in zip(test_src_sentences, test_tgt_sentences):
            test_data.append((tokenize_sentence(src), tokenize_sentence(tgt)))
    
    # 构建词汇表（仅使用训练集）
    print("Building vocabulary...")
    src_vocab = build_vocab(train_src_sentences, min_freq=min_freq, max_vocab_size=max_vocab_size)
    tgt_vocab = build_vocab(train_tgt_sentences, min_freq=min_freq, max_vocab_size=max_vocab_size)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # 处理训练数据
    print("Processing training data...")
    train_data = []
    for src, tgt in zip(train_src_sentences, train_tgt_sentences):
        train_data.append((tokenize_sentence(src), tokenize_sentence(tgt)))
    
    print(f"Loaded {len(train_data)} training samples")
    if dev_data:
        print(f"Loaded {len(dev_data)} dev samples")
    if test_data:
        print(f"Loaded {len(test_data)} test samples")
    
    return train_data, dev_data, test_data, src_vocab, tgt_vocab


def create_masks(src, tgt, src_pad_idx=0, tgt_pad_idx=0):
    """创建 mask"""
    # Source mask: [B, 1, 1, src_len]
    src_mask = create_padding_mask(src, src_pad_idx)
    
    # Target mask: 结合 padding mask 和 look-ahead mask
    tgt_padding_mask = create_padding_mask(tgt, tgt_pad_idx)  # [B, 1, 1, tgt_len]
    seq_len = tgt.size(1)
    look_ahead = create_look_ahead_mask(seq_len).to(tgt.device)  # [1, 1, tgt_len, tgt_len]
    
    # 扩展 padding mask 到 [B, 1, tgt_len, tgt_len]
    tgt_padding_mask = tgt_padding_mask.expand(-1, -1, seq_len, -1)
    # 结合两个 mask
    tgt_mask = tgt_padding_mask & (~look_ahead)
    
    # Cross-attention mask: [B, 1, 1, src_len] (用于 decoder 的 cross-attention)
    src_mask_cross = create_padding_mask(src, src_pad_idx)
    
    return src_mask, tgt_mask, src_mask_cross


def create_padding_mask(seq, pad_idx=0):
    """创建 padding mask"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    return mask


def create_look_ahead_mask(size):
    """创建 look-ahead mask"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]


class LearningRateScheduler:
    """学习率调度器（带预热）"""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(
            self.current_step ** (-0.5),
            self.current_step * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def save_model(model, optimizer, epoch, loss, save_path):
    """保存模型"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Model saved to {save_path}")


def load_model(model, optimizer, load_path):
    """加载模型"""
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {load_path}")
    return epoch, loss

