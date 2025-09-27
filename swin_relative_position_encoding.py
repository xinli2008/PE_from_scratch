"""
Swin Transformer风格的相对位置编码 (Swin Relative Positional Encoding)
================================================================

这个模块实现了Swin Transformer中使用的相对位置编码方法。
Swin使用的是基于窗口的相对位置编码，特别适用于视觉任务。

特点:
- 使用可学习的相对位置偏置表
- 它只生成用于注意力机制的偏置矩阵，而不是直接添加到输入特征上
- 生成的偏置会被添加到注意力权重上
"""

import torch
import torch.nn as nn


class SwinRelativePositionalEncoding(nn.Module):
    """
    Swin Transformer风格的相对位置编码
    
    在Swin Transformer中，相对位置编码被用作注意力机制中的偏置项，
    而不是直接添加到输入特征上。
    
    Args:
        window_size (tuple[int]): 窗口大小 (height, width)
        num_heads (int): 注意力头的数量
    """
    
    def __init__(self, window_size, num_heads):
        super(SwinRelativePositionalEncoding, self).__init__()
        
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        
        # 定义相对位置偏置参数表
        # 大小为 (2*Wh-1) * (2*Ww-1) x num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 生成相对位置索引
        self._generate_relative_position_index()
        
        # 初始化参数
        self._init_parameters()
    
    def _generate_relative_position_index(self):
        """生成相对位置索引"""
        # 创建窗口内的坐标网格
        coords_h = torch.arange(self.window_size[0])  # [0, 1, 2, ..., Wh-1]
        coords_w = torch.arange(self.window_size[1])  # [0, 1, 2, ..., Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # 计算相对坐标
        # coords_flatten[:, :, None] - coords_flatten[:, None, :] 产生所有位置对之间的相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        
        # 将相对坐标转换为正数（加上偏移量）
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 高度方向偏移
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 宽度方向偏移
        
        # 将2D相对坐标映射为1D索引
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
        # 注册为缓冲区，不参与梯度计算
        self.register_buffer("relative_position_index", relative_position_index)
    
    def _init_parameters(self):
        """初始化相对位置偏置参数"""
        # 使用截断正态分布初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self):
        """
        获取相对位置偏置
        
        Returns:
            torch.Tensor: 相对位置偏置，形状为 [num_heads, Wh*Ww, Wh*Ww]
        """
        # 根据索引获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], 
            -1
        )  # Wh*Ww, Wh*Ww, num_heads
        
        # 调整维度顺序：num_heads, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        return relative_position_bias
    

if __name__ == "__main__":
    window_size = (7, 7)
    num_heads = 12
    
    swin_rel_pos = SwinRelativePositionalEncoding(window_size, num_heads)
    relative_position_bias = swin_rel_pos()
    print(relative_position_bias.shape)

    # 在注意力机制中使用
    # attn = (q @ k.transpose(-2, -1)) / scale
    # attn = attn + relative_position_bias.unsqueeze(0)  # 添加偏置
    # attn = softmax(attn)