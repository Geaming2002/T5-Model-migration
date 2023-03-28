# T5-Model-migration

记录T5模型的迁移过程。从基于`pytorch`深度学习框架的`transformers`库(v4.26.1)迁移至基于`MindSpore`深度学习框架的`mindnlp`库中。

> 参考代码地址：https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/t5/modeling_t5.py

> 迁移代码地址：./mindnlp/models/t5/t5.py

# 模型迁移

迁移的具体操作方式可以查看[MindNLP-基于Mindspore2.0的GPT2预训练模型迁移教程](https://zhuanlan.zhihu.com/p/611786486)

如果是第一次迁移模型的话，建议先迁移单个模块然后写出测试代码进行对齐测试，要不然debug会显得很痛苦。

另外，可以查看已完成迁移的代码文件并与源代码文件对比，熟悉一些常见的迁移方法。

[PyTorch与MindSpore API映射表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#pytorch%E4%B8%8Emindspore-api%E6%98%A0%E5%B0%84%E8%A1%A8)

# 模型对齐

对齐主要包括以下两个方面：

- 形状对齐：对于同一输入，同一模块/模型的输出tensor其形状也应一致

- 精度对齐：对于同一输入，同一模块/模型的输出tensor其精度误差不能差异过大。

    - 一般误差取：1e-5/1e-3/5e-3

    - 对于单模块的loss一般取1e-5

    - 对于模型的loss一般取1e-3

    - 5e-3一般只会用于预训练参数过大如3b以上，1e-3确实过不了的情况中。(慎用)

## 模块对齐

## 整网对齐

## 预训练参数加载对齐
