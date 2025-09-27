"""
固定位置编码 (Fixed Positional Encoding)
======================================

这个模块实现了固定的位置编码，使用正弦和余弦函数生成位置编码。
这是原始Transformer论文中使用的方法。

特点:
- 位置编码是固定的，不参与训练
- 使用数学函数生成，具有良好的外推性
- 可以处理任意长度的序列（在合理范围内）
- 不同位置具有独特的编码模式
"""

import torch
import torch.nn as nn
import math


class FixedPositionalEncoding(nn.Module):
    """
    固定位置编码 (Sinusoidal Positional Encoding)
    
    使用正弦和余弦函数生成位置编码，这是原始Transformer论文中的方法。
    
    公式:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中，Pos是token在续联中的位置, i是维度索引。

    Args:
        d_model (int): 模型的维度大小
        max_seq_length (int): 支持的最大序列长度，默认5000
        dropout (float): dropout概率，默认0.1
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super(FixedPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        # x shape: [batch_size, seq_length, d_model]
        seq_length = x.size(1)
        
        # 添加位置编码
        x = x + self.pe[:, :seq_length, :]
        
        return self.dropout(x)
    
if __name__ == "__main__":
    d_model = 512
    seq_length = 20
    batch_size = 2
    
    pos_encoding = FixedPositionalEncoding(d_model=d_model, max_seq_length=1000, dropout=0.1)
    input_embeddings = torch.randn(batch_size, seq_length, d_model)
    output = pos_encoding(input_embeddings)
    print(output.shape)