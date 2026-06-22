from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int | None = None,
) -> float:
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
def sample_gpt(
    model: TinyGPT,
    block_size: int,
    stoi: dict[str, int],
    itos: dict[int, str],
    device: str,
    start_text: str = "ROMEO:",
    max_new_tokens: int = 500,
) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Professor-style TinyGPT implementation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke training")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.max_steps = 20
        args.emb_dim = 64
        args.num_heads = 4
        args.num_layers = 2

    root = Path(__file__).resolve().parent
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    text = ensure_data(root / "input.txt")

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    dataset = NextTokenDataset(data, args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    xb, yb = next(iter(loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("text length:", len(text))
    print("vocab_size:", vocab_size)
    print("xb.shape:", tuple(xb.shape))
    print("yb.shape:", tuple(yb.shape))
    print("logits.shape:", tuple(model(xb.to(device)).shape))
    print("device:", device)

    log_lines = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=args.max_steps)
        line = f"epoch {epoch:02d} | train loss {train_loss:.4f}"
        print(line)
        log_lines.append(line)

    generated = sample_gpt(
        model,
        args.block_size,
        stoi,
        itos,
        device,
        start_text="ROMEO:",
        max_new_tokens=500,
    )
    print("\n--- sample ---")
    print(generated)

    (logs_dir / "professor_style_train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    (logs_dir / "professor_style_sample.txt").write_text(generated, encoding="utf-8")
    torch.save(model.state_dict(), root / "professor_style_tiny_gpt.pt")


if __name__ == "__main__":
    main()
