# PE_from_scratch
pytorch版本实现的position_embedding,用于学习目的, 帮助更好的了解它的工作原理。

## 什么是Position Embedding？

在使用如Transformer这样的模型时, 由于这些模型通常由堆叠的自注意力层组成,它们本质上是对序列中的元素进行处理时不考虑其顺序的。然而,对于大多数任务而言,序列中元素的顺序是至关重要的信息, 所以需要使用位置编码来让模型感知到序列中元素的顺序。

## TODO List
- [x] 可训练的绝对位置编码
- [x] 不可训练的SinCos绝对位置编码，与原始Transformer论文一致。
- [x] Swin_Transformer中的相对位置编码
- [x] 旋转位置编码Rope

## Acknowledgements
- [Swin Transformer之相对位置编码详解](https://www.zhihu.com/tardis/zm/art/577855860?source_id=1005)
- [旋转位置编码 (RoPE), 原理与应用](https://vortezwohl.github.io/nlp/2025/05/22/%E8%AF%A6%E8%A7%A3%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.html#rope-%E7%9A%84%E5%AE%9E%E7%8E%B0-%E5%9F%BA%E4%BA%8E-torch)
- [十分钟读懂旋转编码（RoPE）](https://www.zhihu.com/tardis/bd/art/647109286)
- [Flux中的RoPE](https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py)