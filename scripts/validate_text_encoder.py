#!/usr/bin/env python3
"""Golden-reference TextEncoder harness for Ferrocarril Phase 3.

Loads Kokoro-82M's original PyTorch checkpoint, instantiates a faithful
reimplementation of Kokoro's `TextEncoder` (copied verbatim from
`kokoro/kokoro/modules.py` but without the wider kokoro package
dependencies), runs forward on a fixed token input, and dumps the output
as `.npy` under `tests/fixtures/text_encoder/`.

Usage:
    python3 scripts/validate_text_encoder.py

Requirements:
    torch >= 2.0
    numpy
    huggingface_hub
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from huggingface_hub import hf_hub_download


REPO_ID = "hexgrad/Kokoro-82M"
PTH_FILE = "kokoro-v1_0.pth"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "text_encoder"


# ---------------------------------------------------------------------------
# Faithful reimplementation of Kokoro's LayerNorm and TextEncoder.
# Source: kokoro/kokoro/modules.py (in-repo reference checkout).
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        depth,
        n_symbols,
        actv=nn.LeakyReLU(0.2),
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels, channels, kernel_size=kernel_size, padding=padding
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        self.lstm = nn.LSTM(
            channels, channels // 2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, chn]
        lengths = (
            input_lengths
            if input_lengths.device == torch.device("cpu")
            else input_lengths.to("cpu")
        )
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros(
            [x.shape[0], x.shape[1], m.shape[-1]], device=x.device
        )
        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def download_checkpoint() -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, filename=PTH_FILE))


def load_text_encoder_state_dict(pth_path: Path) -> dict[str, torch.Tensor]:
    full = torch.load(pth_path, map_location="cpu", weights_only=False)
    if "text_encoder" not in full:
        raise RuntimeError(
            f"checkpoint has no 'text_encoder' key. Got: {list(full.keys())[:5]}"
        )
    sd = full["text_encoder"]
    stripped = {}
    for k, v in sd.items():
        if k.startswith("module."):
            stripped[k[len("module.") :]] = v
        else:
            stripped[k] = v
    return stripped


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/locating {PTH_FILE}...")
    pth_path = download_checkpoint()
    print(f"  checkpoint: {pth_path}")

    print("Loading text_encoder state dict...")
    te_sd = load_text_encoder_state_dict(pth_path)
    print(f"  {len(te_sd)} tensors in text_encoder sub-state-dict")

    print(
        "Instantiating TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178)..."
    )
    model = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178)
    missing, unexpected = model.load_state_dict(te_sd, strict=False)
    if missing:
        raise RuntimeError(
            f"TextEncoder load_state_dict: {len(missing)} missing keys: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"TextEncoder load_state_dict: {len(unexpected)} unexpected keys: {unexpected}"
        )
    model.eval()

    # Same token input as validate_bert.py.
    input_ids = torch.tensor([[0, 50, 86, 54, 59, 135, 0]], dtype=torch.long)
    input_lengths = torch.tensor([7], dtype=torch.long)
    text_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    print(f"Running forward pass on input_ids {list(input_ids.shape)}...")
    with torch.no_grad():
        out = model(input_ids, input_lengths, text_mask)

    print("Dumping fixture...")
    np.save(OUTPUT_DIR / "input_ids.npy", input_ids.numpy())
    np.save(OUTPUT_DIR / "text_mask.npy", text_mask.numpy())
    np.save(OUTPUT_DIR / "output.npy", out.numpy())

    manifest = {
        "repo_id": REPO_ID,
        "pth_file": PTH_FILE,
        "input_ids": input_ids.numpy().tolist(),
        "output_shape": list(out.shape),
        "fixture_files": sorted(p.name for p in OUTPUT_DIR.glob("*.npy")),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nFixtures written to {OUTPUT_DIR}/")
    data = out.numpy()[0]  # [channels, time]
    print(f"Output shape: {data.shape}")
    print(f"First channel, first 8 time steps: {data[0, :8]}")
    print(f"Last channel, first 8 time steps: {data[-1, :8]}")
    print(f"Mean abs: {np.mean(np.abs(data)):.6f}")
    print(f"Std: {np.std(data):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())