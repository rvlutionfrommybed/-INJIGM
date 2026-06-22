# TinyGPT professor-style Colab notebook
# Run this cell, then change QUICK_RUN / EPOCHS / MAX_STEPS if needed.

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path("input.txt")

# Colab execution settings
QUICK_RUN = True       # first run: True. For submission run, change to False.
EPOCHS = 20            # used when QUICK_RUN = False
MAX_STEPS = 300        # used when QUICK_RUN = False
BATCH_SIZE = 64
BLOCK_SIZE = 64
EMB_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
DROPOUT = 0.1
START_TEXT = "ROMEO:"
MAX_NEW_TOKENS = 500

if QUICK_RUN:
    EPOCHS = 2
    MAX_STEPS = 20
    EMB_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2


def ensure_data(path: Path) -> str:
    if not path.exists():
        print(f"downloading {DATA_URL}")
        path.write_text(urllib.request.urlopen(DATA_URL, timeout=30).read().decode("utf-8"), encoding="utf-8")
    return path.read_text(encoding="utf-8")


class NextTokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class Head(nn.Module):
    def __init__(self, emb_dim: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attention_scores = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        attention_scores = attention_scores.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(
            *[Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time = x.shape
        positions = torch.arange(time, device=x.device)
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)[None, :, :]
        h = token_embeddings + position_embeddings
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.lm_head(h)


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, device, max_steps=None) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count


@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device, start_text="ROMEO:", max_new_tokens=500):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], dtype=torch.long, device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)

    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)


# Data preparation
text = ensure_data(DATA_PATH)
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

dataset = NextTokenDataset(data, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
xb, yb = next(iter(loader))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    emb_dim=EMB_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("text length:", len(text))
print("vocab_size:", vocab_size)
print("xb.shape:", tuple(xb.shape))
print("yb.shape:", tuple(yb.shape))
print("logits.shape:", tuple(model(xb.to(device)).shape))
print("device:", device)

log_lines = []
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=MAX_STEPS)
    line = f"epoch {epoch:02d} | train loss {train_loss:.4f}"
    print(line)
    log_lines.append(line)

generated = sample_gpt(
    model,
    BLOCK_SIZE,
    stoi,
    itos,
    device,
    start_text=START_TEXT,
    max_new_tokens=MAX_NEW_TOKENS,
)

print("\n--- sample ---")
print(generated)

Path("professor_style_train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
Path("professor_style_sample.txt").write_text(generated, encoding="utf-8")
torch.save(model.state_dict(), "professor_style_tiny_gpt.pt")
print("\nsaved: professor_style_train_log.txt, professor_style_sample.txt, professor_style_tiny_gpt.pt")
