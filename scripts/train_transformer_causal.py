"""
Train GPT-style causal transformer on tokenized .bin data.
Uses shared model from model_causal; supports LR warmup+cosine, checkpointing, resume.
Run from repo root: python scripts/train_transformer_causal.py [options]
"""
import argparse
import math
import pathlib
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

# Import shared model from repo root
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from model_causal import GPTMini, get_default_config

# -------------------------
# Paths
# -------------------------
TRAIN_BIN = "data/processed/train.bin"
VALID_BIN = "data/processed/valid.bin"
TOK_PATH = "tokenizer/bpe_tokenizer.json"
OUT_DIR = pathlib.Path("models/transformer")

EVAL_EVERY = 200
EVAL_BATCHES = 30


def load_tokens(path):
    return np.fromfile(path, dtype=np.uint16)


def get_batch(tokens_np, block_size, batch_size):
    ix = np.random.randint(0, len(tokens_np) - block_size - 1, batch_size)
    x = np.stack([tokens_np[i : i + block_size] for i in ix])
    y = np.stack([tokens_np[i + 1 : i + block_size + 1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def lr_cosine_with_warmup(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio * base_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (base_lr - min_lr_ratio * base_lr)


def main():
    ap = argparse.ArgumentParser(description="Train causal transformer (GPTMini)")
    ap.add_argument("--max-steps", type=int, default=25000, help="Training steps (more = lower perplexity)")
    ap.add_argument("--batch-size", type=int, default=16, help="Larger = more stable, lower perplexity")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=192)
    ap.add_argument("--num-heads", type=int, default=6, help="Must divide embed-dim")
    ap.add_argument("--num-layers", type=int, default=4, help="More layers = lower perplexity, slower")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay (reduces overfitting)")
    ap.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps then cosine decay")
    ap.add_argument("--max-train-tokens", type=int, default=None, help="Cap train tokens (default: use all). Use 0 for no limit.")
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR, help="Checkpoint directory")
    ap.add_argument("--resume", action="store_true", help="Resume from out-dir/ckpt.pt if present")
    ap.add_argument("--train-bin", default=TRAIN_BIN)
    ap.add_argument("--valid-bin", default=VALID_BIN)
    ap.add_argument("--tok-path", default=TOK_PATH)
    args = ap.parse_args()
    if args.max_train_tokens == 0:
        args.max_train_tokens = None

    cfg = get_default_config()
    cfg["BATCH_SIZE"] = args.batch_size
    cfg["BLOCK_SIZE"] = args.block_size
    cfg["EMBED_DIM"] = args.embed_dim
    cfg["NUM_HEADS"] = args.num_heads
    cfg["NUM_LAYERS"] = args.num_layers
    cfg["DROPOUT"] = args.dropout

    assert cfg["EMBED_DIM"] % cfg["NUM_HEADS"] == 0, "embed-dim must be divisible by num-heads"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Data
    train_tokens = load_tokens(args.train_bin)
    valid_tokens = load_tokens(args.valid_bin)
    if args.max_train_tokens is not None and len(train_tokens) > args.max_train_tokens:
        train_tokens = train_tokens[: args.max_train_tokens]
        print(f"(Capped to {args.max_train_tokens:,} train tokens)")

    tokenizer = Tokenizer.from_file(args.tok_path)
    cfg["VOCAB_SIZE"] = tokenizer.get_vocab_size()

    print("Train tokens:", len(train_tokens))
    print("Valid tokens:", len(valid_tokens))
    print("Vocab size:", cfg["VOCAB_SIZE"])
    print("Config:", {k: v for k, v in cfg.items()})

    # Model & optimizer
    model = GPTMini.from_config(cfg, dropout_inference=cfg["DROPOUT"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    start_step = 1
    best_val_loss = float("inf")

    if args.resume and (args.out_dir / "ckpt.pt").is_file():
        ckpt = torch.load(args.out_dir / "ckpt.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step}, best_val_loss {best_val_loss:.4f}")

    def estimate_val():
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                xb, yb = get_batch(
                    valid_tokens, cfg["BLOCK_SIZE"], cfg["BATCH_SIZE"]
                )
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
        model.train()
        avg = sum(losses) / len(losses)
        return avg, math.exp(avg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    grad_accum = max(1, args.grad_accum)
    print("\nTraining...")
    for step in range(start_step, args.max_steps + 1):
        lr = lr_cosine_with_warmup(
            step, args.warmup_steps, args.max_steps, args.lr
        )
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        for _ in range(grad_accum):
            xb, yb = get_batch(
                train_tokens, cfg["BLOCK_SIZE"], cfg["BATCH_SIZE"]
            )
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0:
            vloss, vppl = estimate_val()
            dt = time.time() - t0
            print(
                f"step {step:5d} | lr {lr:.2e} | train_loss {loss.item():.4f} | "
                f"val_loss {vloss:.4f} | val_ppl {vppl:.2f} | {dt:.0f}s"
            )
            if vloss < best_val_loss:
                best_val_loss = vloss
                ckpt = {
                    "state_dict": model.state_dict(),
                    "config": dict(cfg),
                    "tokenizer_path": str(args.tok_path),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "optimizer_state": opt.state_dict(),
                }
                torch.save(ckpt, args.out_dir / "ckpt_best.pt")
                print(f"  -> saved best ckpt_best.pt (val_loss {vloss:.4f})")

    # Final checkpoint (always save)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": dict(cfg),
        "tokenizer_path": str(args.tok_path),
        "step": args.max_steps,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, args.out_dir / "ckpt.pt")
    print("\nSaved:", args.out_dir / "ckpt.pt")
    if (args.out_dir / "ckpt_best.pt").is_file():
        print("Best (by val loss):", args.out_dir / "ckpt_best.pt")

    # Sample (use slightly lower temperature for more coherent output)
    print("\n--- Transformer sample ---\n")
    model.eval()
    bos = tokenizer.token_to_id("<bos>")
    start_id = bos if bos is not None else 0
    start = torch.tensor([[start_id]], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=250, seed=42, temperature=0.8, top_k=40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
