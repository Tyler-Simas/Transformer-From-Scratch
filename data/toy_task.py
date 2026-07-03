from torch.utils.data import Dataset
import torch

"""
This is a toy-task designed to prove the architecture actually works.
"""


# Module level constraints
PAD = 0 # padding
BOS = 1 # beginning
EOS = 2 # end of source content

class ReverseDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size=20, seed=None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        g = torch.Generator().manual_seed(seed) if seed is not None else None

        # Use 3 since 0, 1, 2 are PAD, BOS, and EOS
        self.data = torch.randint(3, vocab_size, (num_samples, seq_len), generator=g)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src = self.data[idx]
        tgt = torch.flip(src, dims=[0]) # Reverses along axis 0, this is the ground truth the model needs to produce
        src = torch.cat([src, torch.tensor([EOS])]) # Add END token to the end (useful for convention)
        tgt_in = torch.cat([torch.tensor([BOS]), tgt]) # Add BEGINNING token to the beginning
        tgt_out = torch.cat([tgt, torch.tensor([EOS])]) # Add END token to target
        return src, tgt_in, tgt_out
    
def collate_batch(batch):
    """
    DataLoader calls __getitem__ multiple times and ends up with a Python list of batch_size tuples. 
    This is not very usable. This function reshapes into batched tensors, not a list of tuples of tensors.
    """
    srcs, tgt_ins, tgt_outs = zip(*batch)
    return torch.stack(srcs), torch.stack(tgt_ins), torch.stack(tgt_outs)
    