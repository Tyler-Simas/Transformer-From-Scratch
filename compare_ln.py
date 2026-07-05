import argparse
from train import run

if __name__ == "__main__":

    base_config = dict(
        steps=3000,
        batch_size=64,
        seq_len=12,
        vocab_size=20,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=512,
        dropout=0.1,
        warmup_steps=400,
        seed=0,
        log_every=50,
    )

    pre_args = argparse.Namespace(norm_mode="pre", **base_config)
    post_args = argparse.Namespace(norm_mode="post", **base_config)

    print("=" * 60)
    print("Training Pre-LN variant")
    print("=" * 60)
    pre_history = run(pre_args)

    print("=" * 60)
    print("Training Post-LN variant")
    print("=" * 60)
    post_history = run(post_args)


    import json
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)
    with open("results/pre_ln_history.json", "w") as f:
        json.dump(pre_history, f, indent=2)
    with open("results/post_ln_history.json", "w") as f:
        json.dump(post_history, f, indent=2)



    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(pre_history["step"], pre_history["loss"], label="Pre-LN")
    axes[0].plot(post_history["step"], post_history["loss"], label="Post-LN")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(pre_history["step"], pre_history["grad_norm"], label="Pre-LN")
    axes[1].plot(post_history["step"], post_history["grad_norm"], label="Post-LN")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].set_title("Gradient Norm (Training Stability)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/pre_vs_post_ln_comparison.png", dpi=150)
    print("Saved comparison plot to results/pre_vs_post_ln_comparison.png")