import torch.nn as nn
import torch
from src.layers import SublayerConnection, EncoderLayer


def test_encoder_layer_preserves_shape_pre():
    layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512, norm_mode="pre")
    x = torch.randn(2, 10, 128)
    mask = torch.ones(2, 1, 1, 10).bool()
    out = layer(x, mask)
    assert out.shape == x.shape

def test_encoder_layer_preserves_shape_post():
    layer = EncoderLayer(d_model=128, num_heads=4, d_ff=512, norm_mode="post")
    x = torch.randn(2, 10, 128)
    mask = torch.ones(2, 1, 1, 10).bool()
    out = layer(x, mask)
    assert out.shape == x.shape