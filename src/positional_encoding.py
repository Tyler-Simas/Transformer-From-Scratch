import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # table we will fill in

        pos = torch.arange(0, max_len).unsqueeze(1) # represents position
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # formula
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0) # batch dimension

        self.register_buffer('pe', pe) # optimizer should not touch this, but it should still move to CUDA if required
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] # make sure the size is not longer than batch size
        return self.dropout(x)