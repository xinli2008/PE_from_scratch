import torch
import torch.nn as nn
import math

class TrainableAbsoluePE(nn.Module):
    r"""可以训练的绝对位置编码"""
    def __init__(self, seq_length, emb_dim):
        super(TrainableAbsoluePE, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, seq_length, emb_dim))
    
    def forward(self, x):
        # NOTE: x的维度一般是[batch, seq_length, emb_dim]
        # 通过广播机制将位置编码添加到输入上
        return x + self.position_embedding
    
class FixedAbsolutePE(nn.Module):
    r"""固定的绝对位置编码"""
    def __init__(self, seq_len, emb_dim):
        super(FixedAbsolutePE, self).__init__()
        # NOTE: 创建一个seq_len * emb_dim的位置编码矩阵
        positional_encoding = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        # 使用sin和cos函数生成编码
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # 为了方便与输入相加, 添加一个batch维度
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x