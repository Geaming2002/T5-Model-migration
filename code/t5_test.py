import numpy as np
import torch
import mindspore
from mindspore import nn
from mindspore import Tensor

from mindnlp.models.t5 import t5 as m
from transformers.models.t5 import modeling_t5 as pt

dtype_list = [(mindspore.float32, torch.float32)]

class T5_test():
    def __init__(self):
        print("<===========T5_test===========>")

    def test_T5LayerNorm(self, ms_dtype, pt_dtype):
        print(">===========test_T5LayerNorm_begin")
        # init model
        hidden_size = 3
        ms_model = m.T5LayerNorm((3,), eps = 1e-6)
        pt_model = pt.T5LayerNorm((3,), eps = 1e-6)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(hidden_size)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5LayerNorm_end")

    def test_T5DenseActDense(self, ms_dtype, pt_dtype):
        print(">===========test_T5DenseActDense_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5DenseActDense(ms_config)
        pt_model = pt.T5DenseActDense(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(ms_config.d_model)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5DenseActDense_end")

    def test_T5DenseGatedActDense(self, ms_dtype, pt_dtype):
        print(">===========test_T5DenseGatedActDense_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5DenseGatedActDense(ms_config)
        pt_model = pt.T5DenseGatedActDense(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(ms_config.d_model)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)
        
        print(">===========test_T5DenseGatedActDense_end")

    def test_T5LayerFF(self, ms_dtype, pt_dtype):
        print(">===========test_T5LayerFF_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5LayerFF(ms_config)
        pt_model = pt.T5LayerFF(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(ms_config.d_model)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5LayerFF_end")

    def test_T5Attention(self, ms_dtype, pt_dtype):
        print(">===========test_T5Attention_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5Attention(ms_config)
        pt_model = pt.T5Attention(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(4, 64, 512)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        # assert ms_out[1].shape == pt_out[1].shape # NoneType
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        # assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5) # NoneType
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)

        # has_relative_attention_bias=True
        # init model
        ms_model = m.T5Attention(ms_config, True)
        pt_model = pt.T5Attention(pt_config, True)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        # assert ms_out[1].shape == pt_out[1].shape # NoneType
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        # assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5) # NoneType
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5Attention_end")

    def test_T5LayerSelfAttention(self, ms_dtype, pt_dtype):
        print(">===========test_T5LayerSelfAttention_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5LayerSelfAttention(ms_config)
        pt_model = pt.T5LayerSelfAttention(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(4, 64, 512)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        # assert ms_out[1].shape == pt_out[1].shape # NoneType
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)

        # has_relative_attention_bias=True
        # init model
        ms_model = m.T5LayerSelfAttention(ms_config, True)
        pt_model = pt.T5LayerSelfAttention(pt_config, True)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        # assert ms_out[1].shape == pt_out[1].shape # NoneType
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        # assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5LayerSelfAttention_end")

    def test_T5LayerCrossAttention(self, ms_dtype, pt_dtype):
        print(">===========test_T5LayerCrossAttention_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5LayerCrossAttention(ms_config)
        pt_model = pt.T5LayerCrossAttention(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(4, 64, 512)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x, key_value_states=None)
        pt_out = pt_model(pt_x, key_value_states=None)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        # assert ms_out[1].shape == pt_out[1].shape # NoneType
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        # assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)

        # has_relative_attention_bias=True
        # init model
        ms_model = m.T5LayerCrossAttention(ms_config)
        pt_model = pt.T5LayerCrossAttention(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # output
        ms_out = ms_model(ms_x, key_value_states=None)
        pt_out = pt_model(pt_x, key_value_states=None)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        # assert ms_out[1].shape == pt_out[1].shape # NoneType
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        # assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5LayerCrossAttention_end")

    def test_T5Block(self, ms_dtype, pt_dtype):
        print(">===========test_T5Block_begin")
        # init config
        ms_config = m.T5Config()
        pt_config = pt.T5Config()
        # init model
        ms_model = m.T5Block(ms_config)
        pt_model = pt.T5Block(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(4, 64, 512)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert ms_out[1].shape == pt_out[1].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)

        # has_relative_attention_bias=True
        # init model
        ms_model = m.T5Block(ms_config, True)
        pt_model = pt.T5Block(pt_config, True)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randn(4, 64, 512)
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model(ms_x)
        pt_out = pt_model(pt_x)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert ms_out[1].shape == pt_out[1].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5Block_end")

    def test_T5PreTrainedModel(self, ms_dtype, pt_dtype):
        print(">===========test_T5PreTrainedModel_begin")
        # init config
        decoder_start_token_id = 0
        ms_config = m.T5Config(decoder_start_token_id = decoder_start_token_id)
        pt_config = pt.T5Config(decoder_start_token_id = decoder_start_token_id)
        # init model
        ms_model = m.T5PreTrainedModel(ms_config)
        pt_model = pt.T5PreTrainedModel(pt_config)
        # prepare data
        x = [[1, 2, 3, -100, -100, -100], [1, 2, 3, -100, -100, -100]]
        ms_x = Tensor(x, dtype=ms_dtype)
        pt_x = torch.tensor(x, dtype=pt_dtype)
        # output
        ms_out = ms_model._shift_right(ms_x)
        pt_out = pt_model._shift_right(pt_x)
        # shape & loss
        assert ms_out.shape == pt_out.shape
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 0, 0)
        print(">===========test_T5PreTrainedModel_end")

    def test_T5Stack(self, ms_dtype, pt_dtype):
        print(">===========test_T5Stack_begin")
        # init config
        ms_config = m.T5Config(dropout_rate=0, return_dict = False)
        pt_config = pt.T5Config(dropout_rate=0, return_dict = False)
        # init embedding
        ms_embed = nn.Embedding(1024, 512)
        pt_embed = torch.nn.Embedding(1024, 512)
        # init model
        ms_model = m.T5Stack(ms_config, embed_tokens=ms_embed)
        pt_model = pt.T5Stack(pt_config, embed_tokens=pt_embed)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        x = np.random.randint(0, 100, (1, 4))
        ms_x = Tensor(x, dtype=mindspore.int64)
        pt_x = torch.tensor(x, dtype=torch.long)
        # output
        ms_out = ms_model(ms_x, use_cache=False)
        pt_out = pt_model(pt_x, use_cache=False)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5Stack_end")

    def test_T5Model(self, ms_dtype, pt_dtype):
        print(">===========test_T5Model_begin")
        # init config
        ms_config = m.T5Config(decoder_start_token_id = 0, return_dict = False)
        pt_config = pt.T5Config(decoder_start_token_id=0, return_dict = False)
        # init model
        ms_model = m.T5Model(ms_config)
        pt_model = pt.T5Model(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        input_ids = np.random.randint(0,100,(1,10))
        decoder_input_ids = np.random.randint(0,100,(1,20))
        ms_input_ids = Tensor(input_ids, dtype=mindspore.int64)
        ms_decoder_input_ids = Tensor(decoder_input_ids, dtype=mindspore.int64)
        pt_input_ids = torch.tensor(input_ids, dtype=torch.long)
        pt_decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        # output
        ms_out = ms_model(input_ids=ms_input_ids, decoder_input_ids=ms_decoder_input_ids)
        pt_out = pt_model(input_ids=pt_input_ids, decoder_input_ids=pt_decoder_input_ids)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        for i in range(len(ms_out[1])):
            for j in range(len(ms_out[1][i])):
                assert ms_out[1][i][j].shape == pt_out[1][i][j].shape
                assert np.allclose(ms_out[1][i][j].asnumpy(), pt_out[1][i][j].detach().numpy(), 1e-5, 1e-5)
        assert ms_out[2].shape == pt_out[2].shape
        assert np.allclose(ms_out[2].asnumpy(), pt_out[2].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5Stack_end")

    def test_T5ForConditionalGeneration(self, ms_dtype, pt_dtype):
        print(">===========test_T5ForConditionalGeneration_begin")
        # init config
        ms_config = m.T5Config(decoder_start_token_id = 0, return_dict = False)
        pt_config = pt.T5Config(decoder_start_token_id=0, return_dict = False)
        # init model
        ms_model = m.T5ForConditionalGeneration(ms_config)
        pt_model = pt.T5ForConditionalGeneration(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        input_ids = np.random.randint(0,100,(1,10))
        labels = np.random.randint(0,100,(1,20))
        ms_input_ids = Tensor(input_ids, dtype=mindspore.int64)
        ms_labels = Tensor(labels, dtype=mindspore.int32)
        pt_input_ids = torch.tensor(input_ids, dtype=torch.long)
        pt_labels = torch.tensor(labels, dtype=torch.long)
        # output
        ms_out = ms_model(input_ids=ms_input_ids, labels = ms_labels)
        pt_out = pt_model(input_ids=pt_input_ids, labels = pt_labels)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        assert ms_out[1].shape == pt_out[1].shape
        assert np.allclose(ms_out[1].asnumpy(), pt_out[1].detach().numpy(), 1e-5, 1e-5)
        for i in range(len(ms_out[2])):
            for j in range(len(ms_out[2][i])):
                assert ms_out[2][i][j].shape == pt_out[2][i][j].shape
                assert np.allclose(ms_out[2][i][j].asnumpy(), pt_out[2][i][j].detach().numpy(), 1e-5, 1e-5)
        assert ms_out[3].shape == pt_out[3].shape
        assert np.allclose(ms_out[3].asnumpy(), pt_out[3].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5ForConditionalGeneration_end")

    def test_T5EncoderModel(self, ms_dtype, pt_dtype):
        print(">===========test_T5EncoderModel_begin")
        # init config
        ms_config = m.T5Config(decoder_start_token_id = 0, return_dict = False)
        pt_config = pt.T5Config(decoder_start_token_id=0,  return_dict = False)
        # init model
        ms_model = m.T5EncoderModel(ms_config)
        pt_model = pt.T5EncoderModel(pt_config)
        # load parameters
        pt_params = pt_model.state_dict()
        for key, param in ms_model.parameters_and_names():
            if 'embedding_table' in key:
                key = key.replace('embedding_table', 'weight')
            param.set_data(Tensor(pt_params.get(key).detach().numpy()))
        # set eval mode
        ms_model.set_train(False)
        pt_model.eval()
        # prepare data
        input_ids = np.random.randint(0,100,(1,10))
        ms_input_ids = Tensor(input_ids, dtype=mindspore.int64)
        pt_input_ids = torch.tensor(input_ids, dtype=torch.long)
        # output
        ms_out = ms_model(input_ids=ms_input_ids)
        pt_out = pt_model(input_ids=pt_input_ids)
        # shape & loss
        assert ms_out[0].shape == pt_out[0].shape
        assert np.allclose(ms_out[0].asnumpy(), pt_out[0].detach().numpy(), 1e-5, 1e-5)
        print(">===========test_T5EncoderModel_end")

if __name__ == "__main__":
    t = T5_test()
    t.test_T5LayerNorm(*dtype_list[0])
    t.test_T5DenseActDense(*dtype_list[0])
    t.test_T5DenseGatedActDense(*dtype_list[0])
    t.test_T5Attention(*dtype_list[0])
    t.test_T5LayerSelfAttention(*dtype_list[0])
    t.test_T5LayerCrossAttention(*dtype_list[0])
    t.test_T5Block(*dtype_list[0])
    t.test_T5PreTrainedModel(*dtype_list[0])
    t.test_T5Stack(*dtype_list[0])
    t.test_T5Model(*dtype_list[0])
    t.test_T5ForConditionalGeneration(*dtype_list[0])
    t.test_T5EncoderModel(*dtype_list[0])
