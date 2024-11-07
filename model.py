import torch
import torch.nn as nn

class TrainableAbsoluePE(nn.Module):
    r"""可以训练的绝对位置编码"""
    def __init__(self, seq_length, emb_dim):
        super(TrainableAbsoluePE, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, seq_length, emb_dim))
    
    def forward(self, x):
        # NOTE: x的维度一般是[batch, seq_length, emb_dim]
        # 通过广播机制将位置编码添加到输入上
        return x + self.position_embedding