from src.positional_encoding import PositionalEncoding
import torch

def test_positional_encoding_preserves_shape():

    pe = PositionalEncoding(d_model=128, max_len=50)
    x = torch.zeros(2, 10, 128)
    out = pe(x)
    assert out.shape == x.shape