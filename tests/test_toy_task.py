import torch
from data.toy_task import ReverseDataset

def test_teacher_forcing_offset():
    ds = ReverseDataset(num_samples=5, seq_len=3, vocab_size=20, seed=0)
    src, tgt_in, tgt_out = ds[0]
    assert torch.equal(tgt_in[1:], tgt_out[:-1])