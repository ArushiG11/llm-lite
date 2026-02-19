import math
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer


# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 32
BLOCK_SIZE = 128      # context length
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
LEARNING_RATE = 3e-4
MAX_TRAIN_TOKENS = 2_000_000
EPOCHS = 3


# -----------------------
# Load data
# -----------------------
def load_tokens(path):
    return np.fromfile(path, dtype=np.uint16)


train_tokens = load_tokens("data/processed/train.bin")
valid_tokens = load_tokens("data/processed/valid.bin")

if len(train_tokens) > MAX_TRAIN_TOKENS:
    train_tokens = train_tokens[:MAX_TRAIN_TOKENS]

tokenizer = Tokenizer.from_file("tokenizer/bpe_tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

print("Train tokens:", len(train_tokens))
print("Valid tokens:", len(valid_tokens))
print("Vocab size:", VOCAB_SIZE)


# -----------------------
# Dataset batching
# -----------------------
def get_batch(tokens):
    ix = np.random.randint(0, len(tokens) - BLOCK_SIZE - 1, BATCH_SIZE)
    x = np.stack([tokens[i:i+BLOCK_SIZE] for i in ix])
    y = np.stack([tokens[i+1:i+BLOCK_SIZE+1] for i in ix])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -----------------------
# Transformer model
# -----------------------
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=4*EMBED_DIM,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


# -----------------------
# Train
# -----------------------
device = "cpu"
model = TransformerLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    for step in range(200):  # 200 steps per epoch
        xb, yb = get_batch(train_tokens)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("\nTraining complete.")


# -----------------------
# Validation perplexity
# -----------------------
def evaluate(tokens):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(tokens) - BLOCK_SIZE, BLOCK_SIZE):
            x = torch.tensor(tokens[i:i+BLOCK_SIZE], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(tokens[i+1:i+BLOCK_SIZE+1], dtype=torch.long).unsqueeze(0)
            _, loss = model(x, y)
            losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


ppl = evaluate(valid_tokens[:100_000])
print("Validation perplexity:", ppl)
