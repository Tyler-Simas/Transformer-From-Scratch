import torch
from torch import Tensor
import torch.nn as nn
import math

def scaled_dot_product_attention(query, key, value, mask = None, dropout = None) -> tuple[Tensor, Tensor]:
    scores = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scores = scores / math.sqrt(d_k) # scale to prevent a one-hot style situation
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf")) # prevent look-ahead

    attn_weights = torch.softmax(scores, dim=-1) # get prob dist
    if dropout is not None:
        attn_weights = dropout(attn_weights) # set some values to 0 so the model is not overly reliant
    
    output = torch.matmul(attn_weights, value)
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1,2)
    
    def _merge_heads(self, x):
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout = self.dropout
        )
        self.attn_weights = attn_weights.detach()

        merged = self._merge_heads(attn_out)
        return self.w_o(merged)

if __name__ == "__main__":
    mha = MultiHeadAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 10, 128)
    out = mha(x, x, x)
    assert out.shape == x.shape  # self-attention should preserve shape
    print("Shape check passed.")