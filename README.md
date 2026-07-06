# Transformer From Scratch

A full encoder-decoder Transformer (Vaswani et al. 2017) implemented with primitives from PyTorch (no `nn.Transformer` or `nn.MultiHeadAttention`). Inspired by Andrej Karpathy's "Zero to Hero" series. I wanted to extend what Karpathy built by including the decoder, tests, file-structure, and Pre-LN vs. Post-LN examination; in hopes of understanding the architecture end-to-end.

## What's Implemented
 - **Scaled dot-product and multi-head attention** --`src/attention.py`
- **Sinusoidal positional encoding** -- `src/positional_encoding.py`
- **Position-wise feed-forward network** -- `src/feedforward.py`
- **Encoder/decoder layers with configurable Pre-LN / Post-LN** -- `src/layers.py`
- **Full encoder-decoder stack, masking, greedy decoding** -- `src/model.py`
- **Warmup + inverse-sqrt LR schedule, label smoothing loss** -- `src/utils.py`

Each component maps to a section of the original paper. My goal was to be able to point at any line of code and explain which formula from the paper it implements, not to simply build a trainable model.

## Pre-LN vs. Post-LN

I wanted to explore the differences in training and results between Post-LN and Pre-LN. The original paper employs Post-LN, but many modern LLMs are built off Pre-LN now. I wanted to explore this.

- **Post-LN** (original paper): `LayerNorm(x + Sublayer(x))`
- **Pre-LN** (GPT-2 and most modern LLMs): `x + Sublayer(LayerNorm(x))`


Pre-LN keeps the residual stream itself un-normalized, giving gradients a more direct path backward through the network; Post-LN normalizes after every residual add, which theory suggests should produce larger, noisier gradients — especially early in training.

I tested this on a small toy task (sequence reversal — see below) with both variants trained under identical data, seed, and LR schedule, changing only `norm_mode`:

![Pre-LN vs Post-LN comparison](results/pre_vs_post_ln_comparison.png)

Pre-LN converges faster and holds a tighter, lower gradient-norm band throughout training; Post-LN is noisier and slower to converge, even with gradient clipping enabled. This matches the expected direction from the literature (Xiong et al., 2020). One thing that is worth being clear about is the scope: this is a 3-layer, single-seed toy experiment — not a controlled study — so please treat it as a confirming sanity check on a well-established and researched result, not a novel finding.

## The toy task

Rather than a real translation dataset, the model is trained on **sequence reversal**: given a random token sequence, output it reversed. This trains in minutes on CPU and, more importantly, can't be solved by shortcut — the decoder has to use cross-attention across the full encoded source to produce each output token, so a working model is real evidence the attention/masking implementation is correct, not just shape-compatible.

## Running it

```bash
pip install -r requirements.txt
pytest tests/                                    # verify the implementation
python train.py --norm-mode pre --steps 2 00     # quick test to verify (~1 min)
python compare_ln.py                             # full 3000-step training for both variants, generate the comparison plot
```

## What I'd extend next

- Rerun the comparison at greater depth (6–8 layers) — the Pre-LN/Post-LN gap is expected to widen with depth, and this setup makes that a simple follow-up experiment.
- Rotary or relative positional encodings as an alternative to the fixed sinusoidal version.
- A held-out generalization check: train at `seq_len=12`, evaluate at `seq_len=20`, to test whether sinusoidal encoding's extrapolation property actually holds in practice.