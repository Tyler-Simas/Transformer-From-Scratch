import torch
from src.attention import MultiHeadAttention, scaled_dot_product_attention


def test_multihead_attention_preserves_shape():
    mha = MultiHeadAttention(d_model=128, num_heads=4)
    x = torch.randn(2, 10, 128)
    out = mha(x, x, x)
    assert out.shape == x.shape


def test_attention_weights_sum_to_one():
    q = k = v = torch.randn(1, 1, 5, 8)
    _, attn_weights = scaled_dot_product_attention(q, k, v)
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)