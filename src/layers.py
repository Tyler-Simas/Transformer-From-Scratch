import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, norm_mode):
        super().__init__()
        assert norm_mode in ("pre", "post"), "Use 'pre' and 'post' for normalization mode."
        self.norm_mode = norm_mode
        self.d_model = d_model

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): # we need to add sublayer instead of precalculating because of pre vs post
        if self.norm_mode == 'pre':
            return x + self.dropout(sublayer(self.layer_norm(x)))
        elif self.norm_mode == 'post':
            return self.layer_norm(x + self.dropout(sublayer(x)))
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1, norm_mode = "pre"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, norm_mode)
        self.sublayer2 = SublayerConnection(d_model, dropout, norm_mode)

    def forward(self, x, src_mask):
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, src_mask))
        x = self.sublayer2(x, self.feed_forward)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, norm_mode="pre"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, norm_mode)
        self.sublayer2 = SublayerConnection(d_model, dropout, norm_mode)
        self.sublayer3 = SublayerConnection(d_model, dropout, norm_mode)

    def forward(self, x, memory, tgt_mask, src_mask):
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, tgt_mask))
        x = self.sublayer2(x, lambda t: self.cross_attn(t, memory, memory, src_mask))
        x = self.sublayer3(x, self.feed_forward)
        return x