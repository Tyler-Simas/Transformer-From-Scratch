import torch
import pytest
from src.model import Transformer


@pytest.mark.parametrize("norm_mode", ["pre", "post"])
def test_transformer_forward_shape(norm_mode):
    model = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        max_len=20,
        norm_mode=norm_mode,
    )
    src = torch.randint(1, 20, (2, 10))
    tgt = torch.randint(1, 20, (2, 8))
    out = model(src, tgt)
    assert out.shape == (2, 8, 20)

def test_tgt_mask_blocks_future_and_padding():
    model = Transformer(src_vocab_size=20, tgt_vocab_size=20, pad_idx=0, max_len=20)
    tgt = torch.tensor([[5, 8, 3, 0, 0]])
    mask = model.make_tgt_mask(tgt)

    assert mask[0, 0, 0, 1] == False
    assert mask[0, 0, 4, 3] == False
    assert mask[0, 0, 2, 0] == True