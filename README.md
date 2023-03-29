# T5-Model-migration

记录T5模型的迁移对齐过程。从基于`pytorch`深度学习框架的`transformers`库(v4.26.1)迁移至基于`MindSpore`深度学习框架的`mindnlp`库中。

> 参考代码地址：https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/t5/modeling_t5.py

> 迁移代码地址：[./code/mindnlp/models/t5/t5.py](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/mindnlp/models/t5/t5.py)


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

    - 5e-3一般只会用于预训练参数过多，1e-3确实过不了的情况中。(慎用)

整个过程大致如下：

- 加载模块/模型(pt_model和ms_model)

    - 初始化模块/模型

    - 权重迁移

- 准备数据`input`

    - 使用`numpy`生成

    - 转换成`torch.tensor`和`mindspore.Tensor`

- 传入模块/模型获得输出`output`

- 输出对齐：形状对齐 + 精度对齐

## 模块对齐

T5所有模块的测试代码见: [./code/t5_test.py](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/t5_test.py)

-> 以`T5Attention`模块为例：[模块对齐.ipynb](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/%E6%A8%A1%E5%9D%97%E5%AF%B9%E9%BD%90.ipynb)

- 加载模块

    - init config：先初始化所有需要在模块传入的参数，且保证一致。在这里即为`T5Config`。

    - init model：初始化`T5Attention`模块

        ```python
        # init model
        ms_model = m.T5Attention(ms_config, has_relative_attention_bias=True)
        pt_model = pt.T5Attention(pt_config, has_relative_attention_bias=True)
        ```

        在这里设置`has_relative_attention_bias=True`，是为了便于展示模型参数迁移中可能会遇到的不同模型参数名称不一致的情况。具体设置见模块不同设置。

    - load parameters：因为不同框架的参数初始化方式不一样，故在初始化完成之后需要进行权重迁移。因为对齐的目标是`pt_model`，所以我们需要把`pt_model`中的权重迁移至`ms_model`。

        在这里可能会遇到两个模型参数名称不一致，需要进行相对应的替换。在这里可以先打印出两个模型的参数名称，查看其是否有不一致。`T5Attention`相关参数名称打印如下：

        |  pt_model   | ms_model  |
        |  ----  | ----  |
        | q.weight  | q.weight |
        | k.weight  | k.weight |
        | v.weight  | v.weight |
        | o.weight  | o.weight |
        | relative_attention_bias.weight  | relative_attention_bias.embedding_table |

        可以看到两个模型在`relative_attention_bias`上的参数名称不同，故在迁移权重时，需要进行名称的替换。

        ```python
        # load parameters
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight') # different name in two models
            param.set_data(mindspore.Tensor(pt_params.get(key).detach().numpy()))
        ```

    - set eval mode：为了避免`dropout`等问题导致精度差异过大，在这里需要设置模型为`eval`。

        ```python
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        ```

- 准备数据

    在这里首先使用`numpy`生成数据，然后分别转为`mindspore.Tensor`和`torch.tensor`。需要注意的是，不同模块对于输入的类型可能有具体的要求，比如有些模块要求输入的数值类型为int而不是float。

    ```python
    # prepare data
    x = np.random.randn(4, 64, 512)
    ms_x = mindspore.Tensor(x, dtype=mindspore.float32) # dtype depends on model
    pt_x = torch.tensor(x, dtype=torch.float32)         # sometimes maybe int not float
    ```

- 传入模块

    ```python
    # output
    ms_out = ms_model(ms_x)
    pt_out = pt_model(pt_x)
    ```

- 输出对齐

    形状对齐 + 精度对齐

    在这里模块的精度对齐我们直接使用1e-5即可。输出可能是`tuple`或者`tensor`，对于`tuple`我们需要对里面包含的`tensor`都取出来进行形状和精度的对齐。

    ```python
    # shape & loss
    assert ms_out[0].shape == pt_out[0].shape
    # assert ms_out[1].shape == pt_out[1].shape # NoneType
    assert ms_out[2].shape == pt_out[2].shape
    assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
    # assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5) # NoneType
    assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)
    ```

    有些时候需要先打印出输出的类型才能知道其类型，在这里我提供一个函数`judge`使用递归的方式进行对齐操作，而不需要区分类型。

    ```python
    def judge(o1, o2, loss = 1e-3, prefix = '-'):
        prefix += '-'
        if (isinstance(o1, tuple)):
            for i in range(len(o1)):
                judge(o1[i], o2[i], loss=loss, prefix=prefix)
        elif (isinstance(o1,mindspore.Tensor)):
            print(f"{prefix}{np.allclose(o1.asnumpy(), o2.detach().numpy(), loss, loss)}")
        else:
            print(f"{type(o1)}-{type(o2)}:{o1==o2}")
    ```

    使用`judge`进行对齐，输出如下：

    ```python
    judge(ms_out, pt_out)
    ```

    ```plain text
    ---True
    <class 'NoneType'>-<class 'NoneType'>:True
    ---True
    ```

## 整网对齐

与模块对齐类似，将整网视为一个“大”模块即可。

## 预训练参数加载对齐
