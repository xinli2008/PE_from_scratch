"""
可学习位置编码 (Learnable Positional Encoding)
===========================================

这个模块实现了可学习的位置编码，位置编码的参数会在训练过程中被优化。
这种方法被ViT等模型广泛使用。

特点:
- 位置编码参数可以通过反向传播进行学习
- 需要预先设定最大序列长度
- 对于不同任务可能有更好的适应性
"""

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码 (Learnable Positional Encoding)
    
    位置编码参数会在训练过程中通过反向传播进行优化。
    这种方法被ViT等模型广泛使用。
    
    Args:
        d_model (int): 模型的维度大小
        max_seq_length (int): 支持的最大序列长度，默认5000
        dropout (float): dropout概率，默认0.1
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建可学习的位置编码参数
        self.position_embeddings = nn.Parameter(
            torch.randn(max_seq_length, d_model)
        )
        
        # 初始化位置编码参数
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_length, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        # x shape: [batch_size, seq_length, d_model]
        batch_size, seq_length, _ = x.shape
        
        position_encodings = self.position_embeddings[:seq_length].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + position_encodings
        
        return self.dropout(x)


if __name__ == "__main__":
    d_model = 512
    seq_length = 20
    batch_size = 2
    
    pos_encoding = LearnablePositionalEncoding(d_model=d_model, max_seq_length=1000, dropout=0.1)
    input_embeddings = torch.randn(batch_size, seq_length, d_model)
    output = pos_encoding(input_embeddings)
    print(output.shape)