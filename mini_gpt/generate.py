from __future__ import annotations

from pathlib import Path

import torch

from model import GPTLanguageModel


def main() -> None:
    root = Path(__file__).resolve().parent
    checkpoint = torch.load(root / "mini_gpt.pt", map_location="cpu")
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]
    block_size = checkpoint["block_size"]

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    decode = lambda ids: "".join([itos[i] for i in ids])
    context = torch.zeros((1, 1), dtype=torch.long)
    output = decode(model.generate(context, max_new_tokens=400)[0].tolist())
    print(output)


if __name__ == "__main__":
    main()
