from pytorch_transformers import BertConfig,BertModel
from pytorch_transformers.modeling_utils import SequenceSummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def seq_len_to_mask(seq_len,max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
    
    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")
    
    return mask   
          
class BertC(nn.Module):
    def __init__(self, name='bert-base-uncased',dropout=0.1,num_class=2):
        super(BertC, self).__init__()
        config = BertConfig.from_pretrained(name)     
        self.bert = BertModel(config) 
        self.proj = nn.Linear(config.hidden_size,num_class)
        self.loss_f=nn.CrossEntropyLoss()    
        self.drop=nn.Dropout(p=dropout)

    def forward(self, src, seq_len, gold=None):
        src_mask=seq_len_to_mask(seq_len,src.size(1))
        out = self.bert(src,attention_mask=src_mask)
        embed=out[1]
        #print(embed.size())
        logits=self.proj(self.drop(embed))
        ret={"pred":logits}
        if gold is not None:
            ret["loss"]=self.loss_f(logits,gold)
        return ret