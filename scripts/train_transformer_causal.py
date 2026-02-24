import math
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# -------------------------
# Config (CPU-friendly defaults)
# -------------------------
BATCH_SIZE = 8
BLOCK_SIZE = 256

EMBED_DIM = 192
NUM_HEADS = 6          # must divide EMBED_DIM
NUM_LAYERS = 3
DROPOUT = 0.1

LEARNING_RATE = 3e-4
TRAIN_STEPS = 12000
EVAL_EVERY = 200
EVAL_BATCHES = 30

MAX_TRAIN_TOKENS = 10_000_000  # set None to use full
DEVICE = "cpu"

TRAIN_BIN = "data/processed/train.bin"
VALID_BIN = "data/processed/valid.bin"
TOK_PATH = "tokenizer/bpe_tokenizer.json"
OUT_DIR = pathlib.Path("models/transformer")

# -------------------------
# Data
# -------------------------
def load_tokens(path):
    return np.fromfile(path, dtype=np.uint16)

train_tokens = load_tokens(TRAIN_BIN)
valid_tokens = load_tokens(VALID_BIN)

if MAX_TRAIN_TOKENS is not None and len(train_tokens) > MAX_TRAIN_TOKENS:
    train_tokens = train_tokens[:MAX_TRAIN_TOKENS]

tokenizer = Tokenizer.from_file(TOK_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

print("Train tokens:", len(train_tokens))
print("Valid tokens:", len(valid_tokens))
print("Vocab size:", VOCAB_SIZE)

def get_batch(tokens_np):
    ix = np.random.randint(0, len(tokens_np) - BLOCK_SIZE - 1, BATCH_SIZE)
    x = np.stack([tokens_np[i:i+BLOCK_SIZE] for i in ix])
    y = np.stack([tokens_np[i+1:i+BLOCK_SIZE+1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# -------------------------
# Model (GPT-style)
# -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert EMBED_DIM % NUM_HEADS == 0
        self.nh = NUM_HEADS
        self.hd = EMBED_DIM // NUM_HEADS

        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.drop = nn.Dropout(DROPOUT)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.drop(out)
        return out

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, 4 * EMBED_DIM)
        self.fc2 = nn.Linear(4 * EMBED_DIM, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, seed=42, temperature=0.9, top_k=50):
        rng = np.random.default_rng(seed)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = logits.squeeze(0)

            if top_k and top_k > 0:
                v, ix = torch.topk(logits, k=top_k)
                probs = F.softmax(v, dim=-1).cpu().numpy()
                next_id = int(rng.choice(ix.cpu().numpy(), p=probs))
            else:
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                next_id = int(rng.choice(VOCAB_SIZE, p=probs))

            idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
        return idx

# -------------------------
# Train + Eval
# -------------------------
model = GPTMini().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def estimate_val():
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            xb, yb = get_batch(valid_tokens)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nTraining...")
t0 = time.time()
for step in range(1, TRAIN_STEPS + 1):
    xb, yb = get_batch(train_tokens)
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)

    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % EVAL_EVERY == 0:
        vloss, vppl = estimate_val()
        dt = time.time() - t0
        print(f"step {step:5d} | train_loss {loss.item():.4f} | val_loss {vloss:.4f} | val_ppl {vppl:.2f} | {dt:.0f}s")

# Save checkpoint
ckpt = {
    "state_dict": model.state_dict(),
    "config": {
        "BATCH_SIZE": BATCH_SIZE,
        "BLOCK_SIZE": BLOCK_SIZE,
        "EMBED_DIM": EMBED_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "DROPOUT": DROPOUT,
        "VOCAB_SIZE": VOCAB_SIZE,
    }
}
torch.save(ckpt, OUT_DIR / "ckpt.pt")
print("\nSaved:", OUT_DIR / "ckpt.pt")

print("\n--- Transformer sample ---\n")
bos = tokenizer.token_to_id("<bos>")
start_id = bos if bos is not None else 0
start = torch.tensor([[start_id]], dtype=torch.long)
out = model.generate(start, max_new_tokens=250, seed=42, temperature=0.9, top_k=50)[0].tolist()
print(tokenizer.decode(out))