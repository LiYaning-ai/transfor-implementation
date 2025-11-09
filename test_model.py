"""
快速测试脚本：验证模型是否能正常运行
"""
import torch
from model import Transformer
from config import Config

def test_model():
    """测试模型前向传播"""
    config = Config()
    
    print("Testing Transformer Model...")
    print("=" * 50)
    
    # 创建模型
    model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=2,  # 使用较少层数进行快速测试
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout
    ).to(config.device)
    
    print(f"Model created on {config.device}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建测试数据
    batch_size = 2
    src_len = 10
    tgt_len = 10
    
    src = torch.randint(0, config.vocab_size, (batch_size, src_len)).to(config.device)
    tgt = torch.randint(0, config.vocab_size, (batch_size, tgt_len)).to(config.device)
    
    print(f"\nTest input shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")
    
    # 创建 masks
    from utils import create_masks
    src_mask, tgt_mask, _ = create_masks(src, tgt, pad_idx=0)
    src_mask = src_mask.to(config.device)
    tgt_mask = tgt_mask.to(config.device)
    
    print(f"\nMasks created:")
    print(f"  src_mask: {src_mask.shape}")
    print(f"  tgt_mask: {tgt_mask.shape}")
    
    # 前向传播
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [batch_size, tgt_len, vocab_size] = [{batch_size}, {tgt_len}, {config.vocab_size}]")
    
    # 验证输出形状
    assert output.shape == (batch_size, tgt_len, config.vocab_size), \
        f"Output shape mismatch! Expected {(batch_size, tgt_len, config.vocab_size)}, got {output.shape}"
    
    print("\n✅ Model test passed!")
    print("=" * 50)

if __name__ == '__main__':
    test_model()


