"""
配置文件：定义模型和训练的超参数
"""
import torch

class Config:
    # 数据配置
    # IWSLT 数据集路径iwslt2017
    data_dir = 'en-de'  # 数据集根目录
    train_src_path = 'en-de/train.tags.en-de.en'  # 训练集源语言文件
    train_tgt_path = 'en-de/train.tags.en-de.de'  # 训练集目标语言文件
    
    # 开发集路径
    dev_src_path = 'en-de/IWSLT17.TED.dev2010.en-de.en.xml'  # 开发集源语言文件
    dev_tgt_path = 'en-de/IWSLT17.TED.dev2010.en-de.de.xml'  # 开发集目标语言文件
    
    # 测试集路径（2010年）
    test_src_path = 'en-de/IWSLT17.TED.tst2010.en-de.en.xml'  # 测试集源语言文件
    test_tgt_path = 'en-de/IWSLT17.TED.tst2010.en-de.de.xml'  # 测试集目标语言文件
    
    # 其他年份测试集路径（可选）
    test_2011_src_path = 'den-de/IWSLT17.TED.tst2011.en-de.en.xml'
    test_2011_tgt_path = 'en-de/IWSLT17.TED.tst2011.en-de.de.xml'
    
    test_2012_src_path = 'en-de/IWSLT17.TED.tst2012.en-de.en.xml'
    test_2012_tgt_path = 'en-de/IWSLT17.TED.tst2012.en-de.de.xml'
    
    test_2013_src_path = 'en-de/IWSLT17.TED.tst2013.en-de.en.xml'
    test_2013_tgt_path = 'en-de/IWSLT17.TED.tst2013.en-de.de.xml'
    
    test_2014_src_path = 'en-de/IWSLT17.TED.tst2014.en-de.en.xml'
    test_2014_tgt_path = 'en-de/IWSLT17.TED.tst2014.en-de.de.xml'
    
    test_2015_src_path = 'en-de/IWSLT17.TED.tst2015.en-de.en.xml'
    test_2015_tgt_path = 'en-de/IWSLT17.TED.tst2015.en-de.de.xml'
    
    # 获取所有测试集路径的方法
    @staticmethod
    def get_test_sets():
        """返回所有可用的测试集路径"""
        test_sets = {
            '2010': {
                'src': Config.test_src_path,
                'tgt': Config.test_tgt_path
            },
            '2011': {
                'src': Config.test_2011_src_path,
                'tgt': Config.test_2011_tgt_path
            },
            '2012': {
                'src': Config.test_2012_src_path,
                'tgt': Config.test_2012_tgt_path
            },
            '2013': {
                'src': Config.test_2013_src_path,
                'tgt': Config.test_2013_tgt_path
            },
            '2014': {
                'src': Config.test_2014_src_path,
                'tgt': Config.test_2014_tgt_path
            },
            '2015': {
                'src': Config.test_2015_src_path,
                'tgt': Config.test_2015_tgt_path
            }
        }
        return test_sets
    
    # 语言设置
    src_lang = 'en'  # 源语言
    tgt_lang = 'de'  # 目标语言
    
    # 词汇表配置
    min_freq = 2  # 最小词频
    max_vocab_size = None  # 最大词汇表大小（None 表示不限制）
    max_len = 128  # 最大序列长度
    
    # 模型配置
    d_model = 512  # 模型维度
    n_heads = 8  # 注意力头数
    n_layers = 6  # Encoder 和 Decoder 层数
    d_ff = 2048  # FFN 隐藏层维度
    dropout = 0.1  # Dropout 率
    
    # 训练配置
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    warmup_steps = 4000  # 学习率预热步数
    max_grad_norm = 1.0  # 梯度裁剪阈值
    weight_decay = 0.01  # AdamW 权重衰减
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 其他
    save_dir = 'checkpoints'
    log_dir = 'logs'
    vocab_dir = 'vocab'  # 词汇表保存目录
    ablation_results_dir = 'ablation_results'  # 消融实验结果目录
    print_freq = 100  # 打印频率
    
    # 消融实验配置
    ablation_num_epochs = 10  # 消融实验训练轮数
    ablation_use_iwslt_data = True  # 是否使用 IWSLT 数据集（False 则使用虚拟数据）


