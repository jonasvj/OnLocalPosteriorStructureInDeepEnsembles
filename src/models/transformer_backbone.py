from transformers import BertConfig, BertModel
import torch.nn as nn
import torch
from src.utils.utils import unfreeze_pretrained_backbone
torch.use_deterministic_algorithms(True, warn_only=False)

#The default parameters below are taken from https://github.com/google-research/bert. Should be approx 42M params.
class BERTBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int=512,
        n_layers: int=8,
        num_attention_heads: int=8,
        intermediate_size: int=2048,
        hidden_dropout_prob: float=0.1,
        attention_probs_dropout_prob: float=0.1,
    ):
        super(BERTBackbone, self).__init__()

        configuration = BertConfig(hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_size=hidden_size, num_hidden_layers=n_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size)
        # Initializing a model (with random weights) from the configuration specified above.
        self.model = BertModel(configuration)

        # param_size = 0
        # for param in self.model.parameters():
        #     param_size += param.nelement() * param.element_size()
        # buffer_size = 0
        # for buffer in self.model.buffers():
        #     buffer_size += buffer.nelement() * buffer.element_size()
        # size_all_mb = (param_size + buffer_size) / 1024**2
        # print(size_all_mb)

        unfreeze_pretrained_backbone(self.model)


    # x here should be tokenized by BERT tokenizer from 
    def forward(self, x):
        out = self.model(input_ids=x[:,0,:], attention_mask=x[:,1,:])
        # From https://huggingface.co/transformers/v3.0.2/model_doc/bert.html, first element of output is last hidden state.
        return out[1]
