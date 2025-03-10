# PE_from_scratch
pytorch版本实现的position_embedding,用于学习目的, 帮助更好的了解它的工作原理。

## 什么是Position Embedding？

在使用如Transformer这样的模型时, 由于这些模型通常由堆叠的自注意力层组成,它们本质上是对序列中的元素进行处理时不考虑其顺序的。然而,对于大多数任务而言,序列中元素的顺序是至关重要的信息, 所以需要使用位置编码来让模型感知到序列中元素的顺序。

## TODO List
- [x] 可训练的绝对位置编码
- [x] 不可训练的SinCos绝对位置编码
- [ ] Swin Transformer中的相对位置编码
- [ ] 旋转位置编码Rope