"""
旋转位置编码 (Rotary Position Embedding - RoPE)
=============================================

这个模块实现了旋转位置编码，这是一种通过旋转变换来编码位置信息的方法。
被广泛应用于LLaMA、GPT-NeoX等现代大语言模型中。

特点:  
- 位置编码通过旋转矩阵实现，它不但编码了绝对位置信息, 还在注意力计算中自然地引入了相对位置信息
- 适用于Transformer的自注意力机制，直接应用于Query和Key。
- 具有良好的外推性，能够处理比训练时更长的序列。

"""
import torch
from torch import Tensor
from typing import Tuple
from einops import rearrange


def rope(pos: Tensor, dim: int, theta: float = 10000.0) -> Tensor:
    """
    生成旋转位置编码矩阵
    
    Args:
        pos: 位置张量 [batch_size, seq_len] 或 [seq_len]
        dim: 嵌入维度（必须是偶数）
        theta: 旋转基数，控制频率衰减速度
        
    Returns:
        旋转矩阵 [batch_size, seq_len, dim//2, 2, 2]
    """
    assert dim % 2 == 0, "维度必须是偶数"
    
    # 确保位置张量是浮点型
    pos = pos.float()
    dim_half = dim // 2
    
    # 生成频率：i从0到dim_half-1
    i = torch.arange(0, dim_half, dtype=torch.float32, device=pos.device)
    freqs = 1.0 / (theta ** (i / dim_half))
    
    # 计算角度：pos * freqs
    angles = torch.einsum('...s,d->...sd', pos, freqs)
    
    # 计算cos和sin
    cos = torch.cos(angles)  # [..., seq_len, dim_half]
    sin = torch.sin(angles)  # [..., seq_len, dim_half]
    
    # 构造旋转矩阵 [[cos, -sin], [sin, cos]]
    rot_mat = torch.stack([
        torch.stack([cos, -sin], dim=-1),  # 第一行
        torch.stack([sin, cos], dim=-1)    # 第二行
    ], dim=-2)  # [..., seq_len, dim_half, 2, 2]
    
    return rot_mat


def apply_rope(q: Tensor, k: Tensor, rot_mat: Tensor) -> Tuple[Tensor, Tensor]:
    """
    将旋转位置编码应用到Query和Key上
    
    Args:
        q: Query张量 [batch_size, num_heads, seq_len, head_dim]
        k: Key张量 [batch_size, num_heads, seq_len, head_dim]
        rot_mat: 旋转矩阵 [batch_size, seq_len, head_dim//2, 2, 2]
        
    Returns:
        旋转后的Query和Key
    """
    # 确保旋转矩阵有正确的batch维度
    if rot_mat.dim() == 4:  # [seq_len, head_dim//2, 2, 2]
        rot_mat = rot_mat.unsqueeze(0)  # [1, seq_len, head_dim//2, 2, 2]
    
    # 将Query和Key重塑为二维向量组 [batch_size, num_heads, seq_len, head_dim//2, 2]
    q_2d = q.float().reshape(*q.shape[:-1], -1, 2)
    k_2d = k.float().reshape(*k.shape[:-1], -1, 2)
    
    # 在num_heads维度上扩展旋转矩阵
    rot_mat_expanded = rot_mat.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim//2, 2, 2]
    
    # 应用旋转矩阵：对每个二维向量进行旋转
    q_rotated = torch.einsum('bhsde,bhsdef->bhsdf', q_2d, rot_mat_expanded)
    k_rotated = torch.einsum('bhsde,bhsdef->bhsdf', k_2d, rot_mat_expanded)
    
    # 重塑回原始形状
    q_rotated = q_rotated.reshape(*q.shape).type_as(q)
    k_rotated = k_rotated.reshape(*k.shape).type_as(k)
    
    return q_rotated, k_rotated


def attention(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> Tensor:
    """
    使用RoPE的注意力计算
    
    Args:
        q: Query [batch_size, num_heads, seq_len, head_dim]
        k: Key [batch_size, num_heads, seq_len, head_dim]
        v: Value [batch_size, num_heads, seq_len, head_dim]
        positions: 位置索引 [batch_size, seq_len] 或 [seq_len]
        
    Returns:
        注意力输出 [batch_size, seq_len, num_heads * head_dim]
    """
    # 生成旋转位置编码
    head_dim = q.size(-1)
    rot_mat = rope(positions, head_dim)
    
    q, k = apply_rope(q, k, rot_mat)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")  # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
    
    return x


if __name__ == "__main__":
    """测试RoPE注意力"""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # 位置索引 [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=q.device)
    
    output = attention(q, k, v, positions)
    print("Attention output shape:", output.shape)  
