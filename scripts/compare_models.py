"""
Compare Bigram, Trigram, and Transformer checkpoints on validation perplexity.

Usage:
  python scripts/compare_models.py

Optional paths:
  --valid-bin data/processed/valid.bin
  --bigram-counts models/bigram_counts.npy
  --bigram-unigram models/bigram_unigram.npy
  --trigram models/trigram.pkl
  --transformer-best models/transformer/ckpt_best.pt
"""
import argparse
import json
import math
import pathlib
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _file_size_mb(path: pathlib.Path) -> Optional[float]:
    if not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def _fmt_float(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "n/a"
    return f"{x:,}"


def _load_valid_tokens(path: pathlib.Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Validation bin not found: {path}")
    return np.fromfile(path, dtype=np.uint16)


def bigram_metrics(
    valid_tokens: np.ndarray,
    counts_path: pathlib.Path,
    alpha: float = 0.1,
) -> Dict[str, Any]:
    if not counts_path.is_file():
        return {"available": False}

    counts = np.load(counts_path)   # shape (V, V) — saved by train_bigram.py
    V = counts.shape[0]
    unigram = counts.sum(axis=1)    # row sums: how often each token appeared as "prev"

    prev = valid_tokens[:-1].astype(np.int64)
    nxt  = valid_tokens[1:].astype(np.int64)
    pair_counts = counts[prev, nxt].astype(np.float64)
    denom = unigram[prev].astype(np.float64) + alpha * V
    probs = (pair_counts + alpha) / denom
    probs = np.clip(probs, 1e-12, 1.0)
    avg_nll = float(-np.log(probs).mean())
    ppl = float(math.exp(avg_nll))

    return {
        "available": True,
        "val_loss": avg_nll,
        "val_ppl": ppl,
        "artifact_mb": _file_size_mb(counts_path) or 0.0,
        "params_or_states": int(counts.size),
        "artifacts": [str(counts_path)],
    }


def ngram_metrics(valid_tokens: np.ndarray, ngram_path: pathlib.Path) -> Dict[str, Any]:
    """Evaluate an interpolated n-gram model saved by train_ngram.py."""
    if not ngram_path.is_file():
        return {"available": False}

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_ngram import compute_perplexity  # reuse exact same eval logic

    with open(ngram_path, "rb") as f:
        model = pickle.load(f)

    n             = model["n"]
    lambdas       = model["lambdas"]
    alpha         = model["alpha"]
    vocab_size    = model["vocab_size"]
    ngram_counts  = model["ngram_counts"]
    ctx_totals    = model["ctx_totals"]

    tokens = valid_tokens.tolist()
    ppl = compute_perplexity(tokens, lambdas, ngram_counts, ctx_totals, vocab_size, alpha, n)
    avg_nll = math.log(ppl)

    total_states = sum(
        sum(len(v) for v in d.values()) if isinstance(next(iter(d.values()), None), dict) else len(d)
        for d in ngram_counts
    )
    return {
        "available": True,
        "n": n,
        "val_loss": float(avg_nll),
        "val_ppl": float(ppl),
        "artifact_mb": _file_size_mb(ngram_path),
        "params_or_states": total_states,
        "artifacts": [str(ngram_path)],
    }


def transformer_metrics(transformer_best_path: pathlib.Path) -> Dict[str, Any]:
    if not transformer_best_path.is_file():
        return {"available": False}

    ckpt = torch.load(transformer_best_path, map_location="cpu")
    val_loss = ckpt.get("best_val_loss")
    val_ppl = math.exp(val_loss) if val_loss is not None else None

    state_dict = ckpt.get("state_dict", {})
    param_count = 0
    for _, tensor in state_dict.items():
        param_count += int(tensor.numel())

    return {
        "available": True,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "val_ppl": float(val_ppl) if val_ppl is not None else None,
        "artifact_mb": _file_size_mb(transformer_best_path),
        "params_or_states": param_count or None,
        "artifacts": [str(transformer_best_path)],
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["Model", "Val loss", "Val ppl", "States/Params", "Artifact size (MB)"]
    print("| " + " | ".join(headers) + " |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            "| "
            + " | ".join(
                [
                    r["model"],
                    _fmt_float(r.get("val_loss"), 4),
                    _fmt_float(r.get("val_ppl"), 2),
                    _fmt_int(r.get("params_or_states")),
                    _fmt_float(r.get("artifact_mb"), 2),
                ]
            )
            + " |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Bigram/Trigram/Transformer metrics")
    ap.add_argument("--valid-bin", type=pathlib.Path, default=pathlib.Path("data/processed/valid.bin"))
    ap.add_argument("--bigram-counts", type=pathlib.Path, default=pathlib.Path("models/bigram_counts.npy"))
    ap.add_argument("--ngram", type=pathlib.Path, default=pathlib.Path("models/ngram_n3.pkl"))
    ap.add_argument("--transformer-best", type=pathlib.Path, default=pathlib.Path("models/transformer/ckpt_best.pt"))
    ap.add_argument("--json-out", type=pathlib.Path, default=pathlib.Path("models/model_comparison.json"))
    args = ap.parse_args()

    valid_tokens = _load_valid_tokens(args.valid_bin)
    rows: List[Dict[str, Any]] = []

    bg = bigram_metrics(valid_tokens, args.bigram_counts)
    if bg.get("available"):
        rows.append({"model": "Bigram", **bg})

    tg = ngram_metrics(valid_tokens, args.ngram)
    if tg.get("available"):
        rows.append({"model": f"{tg.get('n', 3)}-gram", **tg})

    tf = transformer_metrics(args.transformer_best)
    if tf.get("available"):
        rows.append({"model": "Transformer", **tf})

    if not rows:
        raise SystemExit("No model artifacts found. Train models first, then run compare_models.py.")

    rows.sort(key=lambda r: (r.get("val_ppl") is None, r.get("val_ppl", float("inf"))))
    print_table(rows)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved comparison JSON: {args.json_out}")


if __name__ == "__main__":
    main()
