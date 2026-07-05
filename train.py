import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--norm-mode", choices=["pre", "post"], default="pre")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--vocab-size", type=int, default=20)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()

from torch.utils.data import DataLoader
from src.model import Transformer
from src.utils import LabelSmoothingLoss, WarmupInverseSqrtSchedule, count_parameters
from data.toy_task import ReverseDataset, collate_batch, PAD
import torch

def run(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ReverseDataset(
        num_samples=args.steps * args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = Transformer(
            src_vocab_size=args.vocab_size,
            tgt_vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            max_len=args.seq_len + 2,
            dropout=args.dropout,
            norm_mode=args.norm_mode,
            pad_idx=PAD,
        ).to(device)

    print(f"[{args.norm_mode}-LN] {count_parameters(model):,} trainable parameters")

    criterion = LabelSmoothingLoss(args.vocab_size, pad_idx=PAD, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupInverseSqrtSchedule(optimizer, args.d_model, args.warmup_steps)

    history = {"step": [], "loss": [], "grad_norm": [], "lr": []}

    model.train()
    for step, (src, tgt_in, tgt_out) in enumerate(loader, start=1):
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["grad_norm"].append(float(grad_norm))
            history["lr"].append(scheduler.get_last_lr()[0])
            print(f"[{args.norm_mode}-LN] step {step:5d} | loss {loss.item():.4f} "
                    f"| grad_norm {grad_norm:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")
    
    return history

if __name__ == "__main__":
    run(parse_args())