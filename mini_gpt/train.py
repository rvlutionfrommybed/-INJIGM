from __future__ import annotations

from pathlib import Path

import torch

from model import GPTLanguageModel


batch_size = 16
block_size = 64
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 20


torch.manual_seed(1337)
root = Path(__file__).resolve().parent
text = (root / "input.txt").read_text(encoding="utf-8")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    log_lines = [f"device={device}, vocab_size={vocab_size}, data_chars={len(text)}"]

    for iteration in range(max_iters + 1):
        if iteration % eval_interval == 0:
            losses = estimate_loss(model)
            line = f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            print(line)
            log_lines.append(line)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=400)[0].tolist())
    print("\n--- generated text ---")
    print(generated)

    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    (logs / "sample_output.txt").write_text(generated, encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "block_size": block_size,
            "vocab_size": vocab_size,
        },
        root / "mini_gpt.pt",
    )


if __name__ == "__main__":
    main()
