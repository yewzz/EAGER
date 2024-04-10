import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import copy
#from models.modules.multihead_attention import MultiheadAttention

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print(mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 5)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, with_mem=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        if with_mem:
            mem = value
            # mem, self.attn = attention(value, query, query,
            #                            mask=mask.transpose(2, 3), dropout=self.dropout)
            # mem = mem.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            return self.linears[3](x), mem, self.attn  # self.linears[3](mem)
        return self.linears[3](x), self.attn

# TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model,  num_heads, dropout=0.0, pre_ln=False):
        super().__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dropout, pre_ln)])
        self.pre_ln = pre_ln
        if num_layers > 1:
            for _ in range(num_layers - 1):
                self.encoder_layers.append(TransformerEncoderLayer(d_model, num_heads, dropout))
        # self.encoder_layers = nn.ModuleList([
        #     TransformerEncoderLayer(d_model, d_model_2, num_heads, dropout)
        #     for _ in range(num_layers)
        # ])
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, attn_mask=None, mem=None):
        # non_padding_mask = None if mask is None else 1 - mask
        # attn_mask = None if attn_mask is None else 1 - attn_mask
        # x = x.transpose(0, 1)
        encoded_list = []
        for layer in self.encoder_layers:
            if mem is None:
                x = layer(x, mask, attn_mask)
            else:
                x, mem = layer(x, mask, attn_mask, mem)
            encoded_list.append(x)
        if self.pre_ln:
            x = self.final_layer_norm(x)
        # x = x.transpose(0, 1)
        encoded_list = torch.stack(encoded_list, 0).transpose(0, 1)
        return x, encoded_list


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, pre_ln=False):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout

        self.self_attn = MultiHeadedAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 2)
        self.fc2 = nn.Linear(d_model << 2, d_model)
        # self.downsample = nn.Linear(d_model, d_model_2) # make the dimension of the two adjacent res layer same
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.pre_ln = pre_ln

    def forward(self, x, mask, attn_mask=None, mem=None):
        dim = x.size(0)

        # attn_mask = None if self.attn_mask is None else self.attn_mask.cuda(2)[:dim, :dim]
        res = x
        # mask = mask == 0
        if self.pre_ln:
            x = self.self_attn_layer_norm(x)
        if mem is None:
            x, weight = self.self_attn(x, x, x, mask)#, attn_mask=attn_mask)
        else:
            x, weight = self.self_attn(x, mem, mem, mask)
            mem = mem[:, :mem.size(1)]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        if not self.pre_ln:
            x = self.self_attn_layer_norm(x)

        # res = self.downsample(x)
        res = x
        if self.pre_ln:
            x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        if not self.pre_ln:
            x = self.final_layer_norm(x)

        if mem is None:
            return x
        else:
            return x, mem