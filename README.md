# T5-Model-migration

记录T5模型的迁移对齐过程。从基于`pytorch`深度学习框架的`transformers`库(v4.26.1)迁移至基于`MindSpore`深度学习框架的`mindnlp`库中。

> 参考代码地址：https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/t5/modeling_t5.py

> 迁移代码地址：[./code/mindnlp/models/t5/t5.py](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/mindnlp/models/t5/t5.py)

<!-- TOC -->

- [T5-Model-migration](#t5-model-migration)
- [模型迁移](#%E6%A8%A1%E5%9E%8B%E8%BF%81%E7%A7%BB)
- [模型对齐](#%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90)
    - [模块对齐](#%E6%A8%A1%E5%9D%97%E5%AF%B9%E9%BD%90)
    - [整网对齐](#%E6%95%B4%E7%BD%91%E5%AF%B9%E9%BD%90)
    - [预训练参数加载对齐](#%E9%A2%84%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E5%8A%A0%E8%BD%BD%E5%AF%B9%E9%BD%90)
        - [预训练参数下载及转换为ckpt](#%E9%A2%84%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E4%B8%8B%E8%BD%BD%E5%8F%8A%E8%BD%AC%E6%8D%A2%E4%B8%BAckpt)
            - [下载](#%E4%B8%8B%E8%BD%BD)
            - [转换](#%E8%BD%AC%E6%8D%A2)
        - [预训练参数加载并对齐](#%E9%A2%84%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E5%8A%A0%E8%BD%BD%E5%B9%B6%E5%AF%B9%E9%BD%90)
            - [_1](#_1)
            - [_2](#_2)
        - [t5-small参数名称对比](#t5-small%E5%8F%82%E6%95%B0%E5%90%8D%E7%A7%B0%E5%AF%B9%E6%AF%94)
- [Tokenizer迁移](#tokenizer%E8%BF%81%E7%A7%BB)
    - [基础迁移](#%E5%9F%BA%E7%A1%80%E8%BF%81%E7%A7%BB)
    - [额外操作](#%E9%A2%9D%E5%A4%96%E6%93%8D%E4%BD%9C)
    - [ut代码](#ut%E4%BB%A3%E7%A0%81)

<!-- /TOC -->

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

-> 以`T5Attention`模块为例：[模块对齐.ipynb](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/T5Attention%E6%A8%A1%E5%9D%97%E5%AF%B9%E9%BD%90.ipynb)

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

与模块对齐过程类似，将整网视为一个“大”模块即可。

## 预训练参数加载对齐

T5预训练模型参数大小分为：

|  T5Model   | pytorch_model.bin  |
|  ----  | ----  |
| [small](https://huggingface.co/t5-small/tree/main)  | 242MB |
| [base](https://huggingface.co/t5-base/tree/main)  | 892MB |
| [large](https://huggingface.co/t5-large/tree/main)  | 2.95GB |
| [3b](https://huggingface.co/t5-3b/tree/main)  | 11.4GB |
| [11b](https://huggingface.co/t5-11b/tree/main)  | 45.2GB |

### 预训练参数下载及转换为ckpt

#### 下载

T5预训练模型参数及文件可以从huggingface官方直接下载，huggingface也提供了`hf_hub_url`能够直接输出文件的下载链接。因为使用的是Ubuntu，所有可以直接用`wget`命令进行下载。

```python
from huggingface_hub import hf_hub_url

path = "/home/data/T5ckpt"

def download_script(size:str):
    """print wget to download files of a pretrained model"""
    print(f"wget {hf_hub_url(repo_id=size, filename='config.json')} -P {path}/{size}")
    print(f"wget {hf_hub_url(repo_id=size, filename='tokenizer.json')} -P {path}/{size}")
    print(f"wget {hf_hub_url(repo_id=size, filename='pytorch_model.bin')} -P {path}/{size}")
```
```python
download_script("t5-small")
```
```plain text
wget https://huggingface.co/t5-small/resolve/main/config.json -P /home/data/T5ckpt/t5-small
wget https://huggingface.co/t5-small/resolve/main/tokenizer.json -P /home/data/T5ckpt/t5-small
wget https://huggingface.co/t5-small/resolve/main/pytorch_model.bin -P /home/data/T5ckpt/t5-small
```

下载完成后，因为需要测试不同大小的预训练模型，故对于一些文件进行改名。最后t5-small的模型文件如下：

- t5-small
    - pytorch_model.bin
    - t5-small_config.json
    - t5-small_tokenizer.json (暂时未涉及)

#### 转换

-> [T5Model预训练参数转换.ipynb](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/T5Model%E9%A2%84%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E8%BD%AC%E6%8D%A2.ipynb)

下载好相关文件后，因为使用的深度学习框架的差异，我们需要将pytorch_model.bin转换为ckpt格式。在这里以T5-small为例。首先我们将下载好的文件加载进原T5Model。然后分别打印出两个T5Model模型的参数名称并进行对比，查看哪些参数名称需要进行替换。文末附有t5-small的参数名称对比表格。

指定相关文件路径

```python
path = r"/home/data/T5ckpt/t5-small"
config_path = f"{path}/t5-small_config.json"
pytorch_model_path = f"{path}/pytorch_model.bin"
```

初始化`transformers`的T5model并加载预训练参数。

```python
# init config
import json
with open(config_path, encoding='utf-8') as config:
    config = json.load(config)

pt_config = pt.T5Config(**config)
pt_dict = torch.load(pytorch_model_path)
pt_model = pt.T5Model(pt_config)
pt_model.load_state_dict(pt_dict, False)
```

在`transformers`的T5Model加载`pytorch_model.bin`时，有以下信息：

`_IncompatibleKeys(missing_keys=['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'], unexpected_keys=['decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight'])`

可以在`transformers`的T5Model源码中查看到下列信息，其实在T5Model中`encoder.embed_tokens.weight`和`decoder.embed_tokens.weight`都是直接使用的`shared.weight`，故无事发生。

```python
_keys_to_ignore_on_load_missing = [
    r"encoder.embed_tokens.weight",
    r"decoder.embed_tokens.weight",
]
_keys_to_ignore_on_load_unexpected = [
    r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
]
```

加载完预训练参数后，打印相关模型参数，同时我们也需要打印`mindnlp`中的T5Model相关模型参数。

```python
# print pt_model parameters' name
pt_params = pt_model.state_dict()
for key in pt_params.keys():
    print(key)
```

!!!需要注意的是，在某些情况下`key`和`param.name`不一定一致，尤其是存在某些参数属于共享参数时，这时以`param.name`为准。因为`mindspore.load_param_into_net`是以`param.name`作为键去寻找参数。

```python
# init config
ms_config = m.T5Config(**config)
ms_model = m.T5Model(ms_config)
# print ms_model parameters' name
for key, param in ms_model.parameters_and_names():
    print(param.name)
```

从文末的表格对比可知，在此模型中，我们只需要替换：

- `shared.weight`->`decoder.embed_tokens.embedding_table`
- `relative_attention_bias.weight`->`relative_attention_bias.embedding_table`

转换函数如下：

```python
import logging
def torch_to_mindspore(pth_file, size:str=None):
    try:
        import torch
    except:
        raise ImportError(f"'import torch' failed, please install torch by "
                          f"`pip install torch` or instructions from 'https://pytorch.org'")

    size = "mindspore" if not size else size # rename ckpt

    from mindspore import Tensor
    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

    for k, v in state_dict.items():
        if 'shared.weight' in k:
            k = k.replace('shared.weight', 'decoder.embed_tokens.embedding_table')
        if 'relative_attention_bias.weight' in k:
            k = k.replace('relative_attention_bias.weight', 'relative_attention_bias.embedding_table')
        ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

    ms_ckpt_path = pth_file.replace('.bin','.ckpt')
    ms_ckpt_path = ms_ckpt_path.replace('pytorch',size)
    try:
        save_checkpoint(ms_ckpt, ms_ckpt_path)
    except:
        raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.')

    return ms_ckpt_path
```

### 预训练参数加载并对齐

#### _1

-> [T5Model预训练参数加载对齐_1.ipynb](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/T5Model%E9%A2%84%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E5%8A%A0%E8%BD%BD%E5%AF%B9%E9%BD%90_1.ipynb)

主要代码如下：

```python
size = 't5-small'

config_path = f"{path[size]}/{size}_config.json"
with open(config_path, encoding='utf-8') as config:
    config = json.load(config)
# init config
pt_config = pt.T5Config(**config)
ms_config = m.T5Config(**config)

# init model
pt_model = pt.T5Model(pt_config)
ms_model = m.T5Model(ms_config)

# load parameters
pt_dict = torch.load(f"{path[size]}/pytorch_model.bin")
pt_model.load_state_dict(pt_dict, False) 

ms_dict = mindspore.load_checkpoint(f"{path[size]}/{size}_model.ckpt")
param_not_load = mindspore.load_param_into_net(ms_model, ms_dict)
print(f"Param_not_load:{param_not_load}")

# set eval mode
pt_model.eval()
ms_model.set_train(False)
```

!!!注意打印`param_not_load`，其会返回两个列表。第一个是模型中还未load的参数，第二个是参数文件中为load的参数。这里提示`decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.embedding_table`未被load，但是经确认模型结构中不存在这一层参数，且在原代码中是被放进`_keys_to_ignore_on_load_unexpected`中的，无事发生again。

#### _2

[T5Model预训练参数加载对齐_2.ipynb](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/T5Model%E9%A2%84%E8%AE%AD%E7%BB%83%E5%8F%82%E6%95%B0%E5%8A%A0%E8%BD%BD%E5%AF%B9%E9%BD%90_2.ipynb)

当模型参数量过大，可能导致使用前一种方法也无法使得精度对齐，调整loss至5e-3也无果，这里提供另一种方式进行精度对齐。

即引入tokenizer对输入进行处理，随机生成`decoder_input_ids`，输入进模型获得输出。

与前一种方法的差异在于prepare data

```python
# tokenizer
tokenizer = pt.T5Tokenizer.from_pretrained(size)

# prepare data
input_ids = "translate English to German: With T5, we propose reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input. Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task."
decoder_input_ids = [[np.random.randint(0,1000)]]

pt_input_ids = tokenizer([input_ids], return_tensors="pt").input_ids
pt_decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
ms_input_ids = mindspore.Tensor(pt_input_ids.detach().numpy()).to(mindspore.int64)
ms_decoder_input_ids = mindspore.Tensor(pt_decoder_input_ids.detach().numpy()).to(mindspore.int64)
```



### t5-small参数名称对比

|transformers|mindnlp|一致|
|---|----|-----|
|shared.weight|decoder.embed_tokens.embedding_table|False
|encoder.embed_tokens.weight||False
|encoder.block.0.layer.0.SelfAttention.q.weight|encoder.block.0.layer.0.SelfAttention.q.weight|True
|encoder.block.0.layer.0.SelfAttention.k.weight|encoder.block.0.layer.0.SelfAttention.k.weight|True
|encoder.block.0.layer.0.SelfAttention.v.weight|encoder.block.0.layer.0.SelfAttention.v.weight|True
|encoder.block.0.layer.0.SelfAttention.o.weight|encoder.block.0.layer.0.SelfAttention.o.weight|True
|encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight|encoder.block.0.layer.0.SelfAttention.relative_attention_bias.embedding_table|False
|encoder.block.0.layer.0.layer_norm.weight|encoder.block.0.layer.0.layer_norm.weight|True
|encoder.block.0.layer.1.DenseReluDense.wi.weight|encoder.block.0.layer.1.DenseReluDense.wi.weight|True
|encoder.block.0.layer.1.DenseReluDense.wo.weight|encoder.block.0.layer.1.DenseReluDense.wo.weight|True
|encoder.block.0.layer.1.layer_norm.weight|encoder.block.0.layer.1.layer_norm.weight|True
|encoder.block.1.layer.0.SelfAttention.q.weight|encoder.block.1.layer.0.SelfAttention.q.weight|True
|encoder.block.1.layer.0.SelfAttention.k.weight|encoder.block.1.layer.0.SelfAttention.k.weight|True
|encoder.block.1.layer.0.SelfAttention.v.weight|encoder.block.1.layer.0.SelfAttention.v.weight|True
|encoder.block.1.layer.0.SelfAttention.o.weight|encoder.block.1.layer.0.SelfAttention.o.weight|True
|encoder.block.1.layer.0.layer_norm.weight|encoder.block.1.layer.0.layer_norm.weight|True
|encoder.block.1.layer.1.DenseReluDense.wi.weight|encoder.block.1.layer.1.DenseReluDense.wi.weight|True
|encoder.block.1.layer.1.DenseReluDense.wo.weight|encoder.block.1.layer.1.DenseReluDense.wo.weight|True
|encoder.block.1.layer.1.layer_norm.weight|encoder.block.1.layer.1.layer_norm.weight|True
|encoder.block.2.layer.0.SelfAttention.q.weight|encoder.block.2.layer.0.SelfAttention.q.weight|True
|encoder.block.2.layer.0.SelfAttention.k.weight|encoder.block.2.layer.0.SelfAttention.k.weight|True
|encoder.block.2.layer.0.SelfAttention.v.weight|encoder.block.2.layer.0.SelfAttention.v.weight|True
|encoder.block.2.layer.0.SelfAttention.o.weight|encoder.block.2.layer.0.SelfAttention.o.weight|True
|encoder.block.2.layer.0.layer_norm.weight|encoder.block.2.layer.0.layer_norm.weight|True
|encoder.block.2.layer.1.DenseReluDense.wi.weight|encoder.block.2.layer.1.DenseReluDense.wi.weight|True
|encoder.block.2.layer.1.DenseReluDense.wo.weight|encoder.block.2.layer.1.DenseReluDense.wo.weight|True
|encoder.block.2.layer.1.layer_norm.weight|encoder.block.2.layer.1.layer_norm.weight|True
|encoder.block.3.layer.0.SelfAttention.q.weight|encoder.block.3.layer.0.SelfAttention.q.weight|True
|encoder.block.3.layer.0.SelfAttention.k.weight|encoder.block.3.layer.0.SelfAttention.k.weight|True
|encoder.block.3.layer.0.SelfAttention.v.weight|encoder.block.3.layer.0.SelfAttention.v.weight|True
|encoder.block.3.layer.0.SelfAttention.o.weight|encoder.block.3.layer.0.SelfAttention.o.weight|True
|encoder.block.3.layer.0.layer_norm.weight|encoder.block.3.layer.0.layer_norm.weight|True
|encoder.block.3.layer.1.DenseReluDense.wi.weight|encoder.block.3.layer.1.DenseReluDense.wi.weight|True
|encoder.block.3.layer.1.DenseReluDense.wo.weight|encoder.block.3.layer.1.DenseReluDense.wo.weight|True
|encoder.block.3.layer.1.layer_norm.weight|encoder.block.3.layer.1.layer_norm.weight|True
|encoder.block.4.layer.0.SelfAttention.q.weight|encoder.block.4.layer.0.SelfAttention.q.weight|True
|encoder.block.4.layer.0.SelfAttention.k.weight|encoder.block.4.layer.0.SelfAttention.k.weight|True
|encoder.block.4.layer.0.SelfAttention.v.weight|encoder.block.4.layer.0.SelfAttention.v.weight|True
|encoder.block.4.layer.0.SelfAttention.o.weight|encoder.block.4.layer.0.SelfAttention.o.weight|True
|encoder.block.4.layer.0.layer_norm.weight|encoder.block.4.layer.0.layer_norm.weight|True
|encoder.block.4.layer.1.DenseReluDense.wi.weight|encoder.block.4.layer.1.DenseReluDense.wi.weight|True
|encoder.block.4.layer.1.DenseReluDense.wo.weight|encoder.block.4.layer.1.DenseReluDense.wo.weight|True
|encoder.block.4.layer.1.layer_norm.weight|encoder.block.4.layer.1.layer_norm.weight|True
|encoder.block.5.layer.0.SelfAttention.q.weight|encoder.block.5.layer.0.SelfAttention.q.weight|True
|encoder.block.5.layer.0.SelfAttention.k.weight|encoder.block.5.layer.0.SelfAttention.k.weight|True
|encoder.block.5.layer.0.SelfAttention.v.weight|encoder.block.5.layer.0.SelfAttention.v.weight|True
|encoder.block.5.layer.0.SelfAttention.o.weight|encoder.block.5.layer.0.SelfAttention.o.weight|True
|encoder.block.5.layer.0.layer_norm.weight|encoder.block.5.layer.0.layer_norm.weight|True
|encoder.block.5.layer.1.DenseReluDense.wi.weight|encoder.block.5.layer.1.DenseReluDense.wi.weight|True
|encoder.block.5.layer.1.DenseReluDense.wo.weight|encoder.block.5.layer.1.DenseReluDense.wo.weight|True
|encoder.block.5.layer.1.layer_norm.weight|encoder.block.5.layer.1.layer_norm.weight|True
|encoder.final_layer_norm.weight|encoder.final_layer_norm.weight|True
|decoder.embed_tokens.weight||False
|decoder.block.0.layer.0.SelfAttention.q.weight|decoder.block.0.layer.0.SelfAttention.q.weight|True
|decoder.block.0.layer.0.SelfAttention.k.weight|decoder.block.0.layer.0.SelfAttention.k.weight|True
|decoder.block.0.layer.0.SelfAttention.v.weight|decoder.block.0.layer.0.SelfAttention.v.weight|True
|decoder.block.0.layer.0.SelfAttention.o.weight|decoder.block.0.layer.0.SelfAttention.o.weight|True
|decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight|decoder.block.0.layer.0.SelfAttention.relative_attention_bias.embedding_table|False
|decoder.block.0.layer.0.layer_norm.weight|decoder.block.0.layer.0.layer_norm.weight|True
|decoder.block.0.layer.1.EncDecAttention.q.weight|decoder.block.0.layer.1.EncDecAttention.q.weight|True
|decoder.block.0.layer.1.EncDecAttention.k.weight|decoder.block.0.layer.1.EncDecAttention.k.weight|True
|decoder.block.0.layer.1.EncDecAttention.v.weight|decoder.block.0.layer.1.EncDecAttention.v.weight|True
|decoder.block.0.layer.1.EncDecAttention.o.weight|decoder.block.0.layer.1.EncDecAttention.o.weight|True
|decoder.block.0.layer.1.layer_norm.weight|decoder.block.0.layer.1.layer_norm.weight|True
|decoder.block.0.layer.2.DenseReluDense.wi.weight|decoder.block.0.layer.2.DenseReluDense.wi.weight|True
|decoder.block.0.layer.2.DenseReluDense.wo.weight|decoder.block.0.layer.2.DenseReluDense.wo.weight|True
|decoder.block.0.layer.2.layer_norm.weight|decoder.block.0.layer.2.layer_norm.weight|True
|decoder.block.1.layer.0.SelfAttention.q.weight|decoder.block.1.layer.0.SelfAttention.q.weight|True
|decoder.block.1.layer.0.SelfAttention.k.weight|decoder.block.1.layer.0.SelfAttention.k.weight|True
|decoder.block.1.layer.0.SelfAttention.v.weight|decoder.block.1.layer.0.SelfAttention.v.weight|True
|decoder.block.1.layer.0.SelfAttention.o.weight|decoder.block.1.layer.0.SelfAttention.o.weight|True
|decoder.block.1.layer.0.layer_norm.weight|decoder.block.1.layer.0.layer_norm.weight|True
|decoder.block.1.layer.1.EncDecAttention.q.weight|decoder.block.1.layer.1.EncDecAttention.q.weight|True
|decoder.block.1.layer.1.EncDecAttention.k.weight|decoder.block.1.layer.1.EncDecAttention.k.weight|True
|decoder.block.1.layer.1.EncDecAttention.v.weight|decoder.block.1.layer.1.EncDecAttention.v.weight|True
|decoder.block.1.layer.1.EncDecAttention.o.weight|decoder.block.1.layer.1.EncDecAttention.o.weight|True
|decoder.block.1.layer.1.layer_norm.weight|decoder.block.1.layer.1.layer_norm.weight|True
|decoder.block.1.layer.2.DenseReluDense.wi.weight|decoder.block.1.layer.2.DenseReluDense.wi.weight|True
|decoder.block.1.layer.2.DenseReluDense.wo.weight|decoder.block.1.layer.2.DenseReluDense.wo.weight|True
|decoder.block.1.layer.2.layer_norm.weight|decoder.block.1.layer.2.layer_norm.weight|True
|decoder.block.2.layer.0.SelfAttention.q.weight|decoder.block.2.layer.0.SelfAttention.q.weight|True
|decoder.block.2.layer.0.SelfAttention.k.weight|decoder.block.2.layer.0.SelfAttention.k.weight|True
|decoder.block.2.layer.0.SelfAttention.v.weight|decoder.block.2.layer.0.SelfAttention.v.weight|True
|decoder.block.2.layer.0.SelfAttention.o.weight|decoder.block.2.layer.0.SelfAttention.o.weight|True
|decoder.block.2.layer.0.layer_norm.weight|decoder.block.2.layer.0.layer_norm.weight|True
|decoder.block.2.layer.1.EncDecAttention.q.weight|decoder.block.2.layer.1.EncDecAttention.q.weight|True
|decoder.block.2.layer.1.EncDecAttention.k.weight|decoder.block.2.layer.1.EncDecAttention.k.weight|True
|decoder.block.2.layer.1.EncDecAttention.v.weight|decoder.block.2.layer.1.EncDecAttention.v.weight|True
|decoder.block.2.layer.1.EncDecAttention.o.weight|decoder.block.2.layer.1.EncDecAttention.o.weight|True
|decoder.block.2.layer.1.layer_norm.weight|decoder.block.2.layer.1.layer_norm.weight|True
|decoder.block.2.layer.2.DenseReluDense.wi.weight|decoder.block.2.layer.2.DenseReluDense.wi.weight|True
|decoder.block.2.layer.2.DenseReluDense.wo.weight|decoder.block.2.layer.2.DenseReluDense.wo.weight|True
|decoder.block.2.layer.2.layer_norm.weight|decoder.block.2.layer.2.layer_norm.weight|True
|decoder.block.3.layer.0.SelfAttention.q.weight|decoder.block.3.layer.0.SelfAttention.q.weight|True
|decoder.block.3.layer.0.SelfAttention.k.weight|decoder.block.3.layer.0.SelfAttention.k.weight|True
|decoder.block.3.layer.0.SelfAttention.v.weight|decoder.block.3.layer.0.SelfAttention.v.weight|True
|decoder.block.3.layer.0.SelfAttention.o.weight|decoder.block.3.layer.0.SelfAttention.o.weight|True
|decoder.block.3.layer.0.layer_norm.weight|decoder.block.3.layer.0.layer_norm.weight|True
|decoder.block.3.layer.1.EncDecAttention.q.weight|decoder.block.3.layer.1.EncDecAttention.q.weight|True
|decoder.block.3.layer.1.EncDecAttention.k.weight|decoder.block.3.layer.1.EncDecAttention.k.weight|True
|decoder.block.3.layer.1.EncDecAttention.v.weight|decoder.block.3.layer.1.EncDecAttention.v.weight|True
|decoder.block.3.layer.1.EncDecAttention.o.weight|decoder.block.3.layer.1.EncDecAttention.o.weight|True
|decoder.block.3.layer.1.layer_norm.weight|decoder.block.3.layer.1.layer_norm.weight|True
|decoder.block.3.layer.2.DenseReluDense.wi.weight|decoder.block.3.layer.2.DenseReluDense.wi.weight|True
|decoder.block.3.layer.2.DenseReluDense.wo.weight|decoder.block.3.layer.2.DenseReluDense.wo.weight|True
|decoder.block.3.layer.2.layer_norm.weight|decoder.block.3.layer.2.layer_norm.weight|True
|decoder.block.4.layer.0.SelfAttention.q.weight|decoder.block.4.layer.0.SelfAttention.q.weight|True
|decoder.block.4.layer.0.SelfAttention.k.weight|decoder.block.4.layer.0.SelfAttention.k.weight|True
|decoder.block.4.layer.0.SelfAttention.v.weight|decoder.block.4.layer.0.SelfAttention.v.weight|True
|decoder.block.4.layer.0.SelfAttention.o.weight|decoder.block.4.layer.0.SelfAttention.o.weight|True
|decoder.block.4.layer.0.layer_norm.weight|decoder.block.4.layer.0.layer_norm.weight|True
|decoder.block.4.layer.1.EncDecAttention.q.weight|decoder.block.4.layer.1.EncDecAttention.q.weight|True
|decoder.block.4.layer.1.EncDecAttention.k.weight|decoder.block.4.layer.1.EncDecAttention.k.weight|True
|decoder.block.4.layer.1.EncDecAttention.v.weight|decoder.block.4.layer.1.EncDecAttention.v.weight|True
|decoder.block.4.layer.1.EncDecAttention.o.weight|decoder.block.4.layer.1.EncDecAttention.o.weight|True
|decoder.block.4.layer.1.layer_norm.weight|decoder.block.4.layer.1.layer_norm.weight|True
|decoder.block.4.layer.2.DenseReluDense.wi.weight|decoder.block.4.layer.2.DenseReluDense.wi.weight|True
|decoder.block.4.layer.2.DenseReluDense.wo.weight|decoder.block.4.layer.2.DenseReluDense.wo.weight|True
|decoder.block.4.layer.2.layer_norm.weight|decoder.block.4.layer.2.layer_norm.weight|True
|decoder.block.5.layer.0.SelfAttention.q.weight|decoder.block.5.layer.0.SelfAttention.q.weight|True
|decoder.block.5.layer.0.SelfAttention.k.weight|decoder.block.5.layer.0.SelfAttention.k.weight|True
|decoder.block.5.layer.0.SelfAttention.v.weight|decoder.block.5.layer.0.SelfAttention.v.weight|True
|decoder.block.5.layer.0.SelfAttention.o.weight|decoder.block.5.layer.0.SelfAttention.o.weight|True
|decoder.block.5.layer.0.layer_norm.weight|decoder.block.5.layer.0.layer_norm.weight|True
|decoder.block.5.layer.1.EncDecAttention.q.weight|decoder.block.5.layer.1.EncDecAttention.q.weight|True
|decoder.block.5.layer.1.EncDecAttention.k.weight|decoder.block.5.layer.1.EncDecAttention.k.weight|True
|decoder.block.5.layer.1.EncDecAttention.v.weight|decoder.block.5.layer.1.EncDecAttention.v.weight|True
|decoder.block.5.layer.1.EncDecAttention.o.weight|decoder.block.5.layer.1.EncDecAttention.o.weight|True
|decoder.block.5.layer.1.layer_norm.weight|decoder.block.5.layer.1.layer_norm.weight|True
|decoder.block.5.layer.2.DenseReluDense.wi.weight|decoder.block.5.layer.2.DenseReluDense.wi.weight|True
|decoder.block.5.layer.2.DenseReluDense.wo.weight|decoder.block.5.layer.2.DenseReluDense.wo.weight|True
|decoder.block.5.layer.2.layer_norm.weight|decoder.block.5.layer.2.layer_norm.weight|True
|decoder.final_layer_norm.weight|decoder.final_layer_norm.weight|True

# Tokenizer迁移

测试代码：

[T5Tokenizer.ipynb](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/T5Tokenizer.ipynb)

最终提交仓库代码文件：

[\_\_init\_\_.py](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/mindnlp/transforms/__init__.py)

[t5_tokenizer.py](https://github.com/Geaming-CHN/T5-Model-migration/blob/main/code/mindnlp/transforms/tokenizers/t5_tokenizer.py)

[test_t5_tokenizer.py](#ut%E4%BB%A3%E7%A0%81)


对于提供的预训练模型，往往会有其对应的tokenizer进行搭配使用。以T5模型为例总共有5种size，其中每种都有对应的tokenizer。通过上文的`download_script`我们可以获得各个模型对应的`tokenizer.json`的链接

|T5Model|tokenizer|
|--|--|
|small|https://huggingface.co/t5-small/resolve/main/tokenizer.json|
|base|https://huggingface.co/t5-base/resolve/main/tokenizer.json|
|large|https://huggingface.co/t5-large/resolve/main/tokenizer.json|
|3b|https://huggingface.co/t5-3b/resolve/main/tokenizer.json|
|11b|https://huggingface.co/t5-11b/resolve/main/tokenizer.json|

## 基础迁移

主要使用`tokenizers`中的`Tokenizer`，`models`来实现基本的`encode`和`decode`功能。代码如下：
```python
import os
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils.download import cache_file
from tokenizers import Tokenizer, models

URL = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/tokenizer.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/tokenizer.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/tokenizer.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/tokenizer.json"
    
}

class T5Tokenizer():
    def __init__(
        self,
        tokenizer_file=None,
    ):
        if tokenizer_file != None:
            self._tokenizer = Tokenizer(models.Unigram()).from_file(tokenizer_file)

    def __call__(self, text_input):
        return super().__call__(text_input)

    @classmethod
    def from_pretrained(cls, size:str):
        cache_dir = os.path.join(DEFAULT_ROOT, "tokenizers", size)
        path, _ = cache_file(None, url=URL[size], cache_dir=cache_dir)
        tokenizer = cls(tokenizer_file=str(path))
        return tokenizer

    def encode(self, text_input):
        tokens = self._tokenizer.encode(text_input)
        return tokens

    def decode(self, ids: list):
        return self.decode(ids)
```

其中通过huggingface的官方文档可以知道，T5Tokenizer是基于Unigram算法进行分词的，故在代码中加载的是`model.Unigram()`，在迁移时请注意原本的tokenizer使用的是什么分词器，如`BPE`之类的还需要加载对应的`spiece.model`等。

这时候我们给出的测试代码如下，主要是将两个tokenizer进行对比：
```python
from mindnlp.transforms import T5Tokenizer
from transformers import T5TokenizerFast

pt_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
ms_tokenizer = T5Tokenizer.from_pretrained('t5-base')

text = "Believing that faith can triumph over everything is in itself the greatest belief"

print(pt_tokenizer.encode(text))
print(ms_tokenizer.encode(text).ids)

print(pt_tokenizer(text).attention_mask)
print(ms_tokenizer.encode(text).attention_mask)

print(pt_tokenizer.decode(pt_tokenizer.encode(text)))
print(ms_tokenizer.decode(ms_tokenizer.encode(text).ids))
```
结果如下：
```plain text
[493, 1896, 3745, 24, 3251, 54, 20020, 147, 762, 19, 16, 1402, 8, 4016, 7750, 1]
[493, 1896, 3745, 24, 3251, 54, 20020, 147, 762, 19, 16, 1402, 8, 4016, 7750, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Believing that faith can triumph over everything is in itself the greatest belief</s>
Believing that faith can triumph over everything is in itself the greatest belief
```
除了最后的结尾符`</s>`，其余保持一致。

## 额外操作

在基础操作之上，我们还需要将tokenizer进一步改写，使其能够符合`mindspore`的`map`等操作。主要操作：

- 继承`PyTensorOperation`类
- 引入`Implementation`类
- 增加`execute_py`，`_execute_py`，`_convert_to_unicode`函数
- 修改`__call__`函数

修改后的代码如下：

```python
import os
import numpy as np
from tokenizers import Tokenizer, models
from mindspore.dataset.transforms.transforms import PyTensorOperation
from mindspore.dataset.text.transforms import Implementation
from mindnlp.utils.download import cache_file
from mindnlp.configs import DEFAULT_ROOT

URL = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/tokenizer.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/tokenizer.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/tokenizer.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/tokenizer.json"
}

class T5Tokenizer(PyTensorOperation):
    """
    Tokenizer used for Bert text process.
    Args:
        tokenizer_file (Str): The path of the tokenizer.json 
    Examples:
        >>> from mindspore.dataset import text
        >>> from mindnlp.transforms import T5Tokenizer
        >>> text = "Believing that faith can triumph over everything is in itself the greatest belief"
        >>> tokenizer = T5Tokenizer.from_pretrained('t5-base')
        >>> tokens = tokenizer.encode(text)
    """
    def __init__(
        self,
        tokenizer_file=None,
    ):
        super().__init__()
        if tokenizer_file is not None:
            self._tokenizer = Tokenizer(models.Unigram()).from_file(tokenizer_file)
        self.implementation = Implementation.PY

    def __call__(self, text_input):
        if isinstance(text_input, str):
            text_input = np.array(text_input)
        elif not isinstance(text_input, np.ndarray):
            raise TypeError(
                f"Input should be a text line in 1-D NumPy format, got {type(text_input)}.")
        return super().__call__(text_input)

    @classmethod
    def from_pretrained(cls, size:str):
        """load T5Tokenizer from pretrained tokenizer.json"""
        cache_dir = os.path.join(DEFAULT_ROOT, "tokenizers", size)
        path, _ = cache_file(None, url=URL[size], cache_dir=cache_dir)
        tokenizer = cls(tokenizer_file=str(path))
        return tokenizer

    def encode(self, text_input):
        """encode function"""
        tokens = self._tokenizer.encode(text_input)
        return tokens

    def decode(self, ids: list):
        """decode function"""
        return self._tokenizer.decode(ids)

    def execute_py(self, text_input):
        """
        Execute method.
        """
        return self._execute_py(text_input)

    def _execute_py(self, text_input):
        """
        Execute method.
        """
        text_input = self._convert_to_unicode(text_input)
        tokens = self._tokenizer.encode(text_input)
        return tokens.ids

    def _convert_to_unicode(self, text_input):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text_input, str):
            return text_input
        if isinstance(text_input, bytes):
            return text_input.decode("utf-8", "ignore")
        if isinstance(text_input, np.ndarray):
            if text_input.dtype.type is np.bytes_:
                text_input = np.char.decode(text_input, "utf-8")
            return str(text_input)
        raise ValueError(f"Unsupported string type: {type(text_input)}, {text_input.dtype}")
```

对应的，我们需要对修改后的tokenizer进行测试，测试代码如下：

```python
from mindnlp.transforms import T5Tokenizer
from mindspore.dataset import GeneratorDataset
from transformers import T5TokenizerFast

pt_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
texts = ['i make a small mistake when i\'m working!']
test_dataset = GeneratorDataset(texts, 'text')
test_dataset = test_dataset.map(operations=tokenizer)
dataset_after = next(test_dataset.create_tuple_iterator())[0]

print(pt_tokenizer.encode(texts[0]))
print(dataset_after)
```
结果如下，运行成功且结果一致：
```plain text
[3, 23, 143, 3, 9, 422, 6202, 116, 3, 23, 31, 51, 464, 55, 1]
[   3   23  143    3    9  422 6202  116    3   23   31   51  464   55
    1]
```

## ut代码

最后，对应于T5Tokenizer的ut测试代码如下：

```python
import unittest
from mindspore.dataset import GeneratorDataset
from mindnlp.transforms import T5Tokenizer
class TestT5Tokenizer(unittest.TestCase):
    r"""
    Test T5Tokenizer
    """
    def test_t5_tokenizer(self):
        """test T5Tokenizer based on t5-base"""
        text = "Believing that faith can triumph over everything is in itself the greatest belief"
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        tokens = tokenizer.encode(text)
        assert len(tokens.ids) == 16
        assert len(tokens.attention_mask) == 16
        assert text == tokenizer.decode(tokens.ids)

    def test_t5_tokenizer_op(self):
        """test T5Tokenizer based on t5-base"""
        texts = ['i make a small mistake when i\'m working!']
        test_dataset = GeneratorDataset(texts, 'text')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        test_dataset = test_dataset.map(operations=tokenizer)
        dataset_after = next(test_dataset.create_tuple_iterator())[0]
        assert len(dataset_after) == 15
```
