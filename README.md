# Transformer 实现

这是一个手工实现的完整 Transformer 模型，用于序列到序列任务。包含完整的 Encoder-Decoder 架构、训练脚本和消融实验。

## 功能特性

### 核心组件
- ✅ Multi-head Self-Attention（多头自注意力）
- ✅ Position-wise Feed-Forward Network（位置前馈网络）
- ✅ Residual Connections + LayerNorm（残差连接和层归一化）
- ✅ Positional Encoding（位置编码）
- ✅ Encoder Block 和 Decoder Block
- ✅ 完整的 Encoder-Decoder Transformer 架构

### 训练功能
- ✅ 学习率调度（带预热的 Transformer 学习率调度）
- ✅ 梯度裁剪
- ✅ AdamW 优化器
- ✅ 模型参数统计
- ✅ 模型保存/加载
- ✅ 训练曲线可视化

### 消融实验
- ✅ 位置编码消融
- ✅ 残差连接消融
- ✅ LayerNorm 消融
- ✅ FFN 消融
- ✅ 多头注意力消融

## 项目结构

```
transformer/
├── model.py          # Transformer 模型实现
├── train.py          # 训练脚本
├── ablation.py       # 消融实验脚本
├── utils.py          # 工具函数（数据加载、可视化等）
├── config.py         # 配置文件
├── requirements.txt  # 依赖包
└── README.md         # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练脚本会：
- 加载数据集
- 创建 Transformer 模型
- 显示模型参数统计
- 进行训练并保存最佳模型
- 生成训练曲线图

### 2. 运行消融实验

```bash
python ablation.py
```

消融实验会：
- 测试不同组件对模型性能的影响
- 生成对比图表
- 保存实验结果到 JSON 文件

## 配置说明

在 `config.py` 中可以修改以下配置：

```python
# 模型配置
d_model = 512          # 模型维度
n_heads = 8            # 注意力头数
n_layers = 6           # Encoder 和 Decoder 层数
d_ff = 2048            # FFN 隐藏层维度
dropout = 0.1          # Dropout 率

# 训练配置
batch_size = 32
num_epochs = 20
learning_rate = 1e-4
warmup_steps = 4000    # 学习率预热步数
max_grad_norm = 1.0    # 梯度裁剪阈值
weight_decay = 0.01    # AdamW 权重衰减
```

## 模型架构

### Encoder Block
- Multi-head Self-Attention
- Residual Connection + LayerNorm
- Position-wise FFN
- Residual Connection + LayerNorm

### Decoder Block
- Masked Multi-head Self-Attention
- Residual Connection + LayerNorm
- Multi-head Cross-Attention
- Residual Connection + LayerNorm
- Position-wise FFN
- Residual Connection + LayerNorm

## 输出文件

训练完成后会生成：
- `checkpoints/best_model.pt` - 最佳模型检查点
- `checkpoints/checkpoint_epoch_*.pt` - 定期保存的检查点
- `logs/training_curves.png` - 训练曲线图

消融实验完成后会生成：
- `ablation_results/results.json` - 实验结果（JSON 格式）
- `ablation_results/comparison.png` - 对比图表

## 注意事项

1.注意源文件和目标文件长度一致

2.可以根据需要调整 `config.py` 中的超参数

## 参考

本实现参考了 "Attention Is All You Need" (Vaswani et al., 2017) 论文。

