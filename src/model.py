import torch.nn as nn
import copy
import math
import torch
from .layers import EncoderLayer, DecoderLayer
from .positional_encoding import PositionalEncoding

def _clone_layers(layer, n):
    '''
    Helper function: produce n independent copies of a layer module each with separate weights (deepcopy).
    '''
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class Encoder(nn.Module):
    def __init__(self, layer, num_layers, d_model, norm_mode):
        super().__init__()
        self.layers = _clone_layers(layer, num_layers)
        self.final_norm = nn.LayerNorm(d_model) if norm_mode == 'pre' else nn.Identity() # nn.Identity is a no-op module
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.final_norm(x) # Will do nothing in Post-LN (identiy), applies extra norm in Pre-LN mode
    
class Decoder(nn.Module):
    def __init__(self, layer, num_layers, d_model, norm_mode):
        super().__init__()
        self.layers = _clone_layers(layer, num_layers)
        self.final_norm = nn.LayerNorm(d_model) if norm_mode == 'pre' else nn.Identity() # nn.Identity is a no-op module
    
    def forward(self, x, memory, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.final_norm(x) # Will do nothing in Post-LN (identiy), applies extra norm in Pre-LN mode

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
    
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=128, 
                 num_heads=4, 
                 num_layers=3, 
                 d_ff=512, 
                 max_len=256, 
                 dropout=0.1, 
                 norm_mode='pre',
                 pad_idx=0):
        super().__init__()

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, norm_mode)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, norm_mode)

        self.src_embed = Embeddings(src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(enc_layer, num_layers, d_model, norm_mode)
        self.decoder = Decoder(dec_layer, num_layers, d_model, norm_mode)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx
        self._init_weights()

    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        batch, tgt_len = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool)
        )
        return pad_mask & causal

    def _init_weights(self):
            for param in self.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src))
        return self.encoder(x, src_mask)
    
    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self.pos_enc(self.tgt_embed(tgt))
        return self.decoder(x, memory, tgt_mask, src_mask)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(out)
    