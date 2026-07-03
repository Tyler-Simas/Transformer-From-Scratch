import torch
import torch.nn as nn

class WarmupInverseSqrtSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=400):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer) # Must assign attributes first since __init__ will call self.get_lr()

    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [scale for _ in self.base_lrs]
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = (target != self.pad_idx).unsqueeze(1)
        true_dist = true_dist * mask # handles the case where the correct label is padding
        loss = -(true_dist * log_probs).sum(dim=-1) # smoothed cross-entropy

        num_tokens = mask.sum().clamp(min=1)
        return loss.sum() / num_tokens

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
