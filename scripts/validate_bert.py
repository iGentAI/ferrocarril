#!/usr/bin/env python3
"""Golden-reference BERT harness for Ferrocarril Phase 3 numerical validation.

Loads Kokoro-82M's original PyTorch checkpoint, instantiates an AlbertModel
matching Kokoro's `plbert` sub-config, runs forward on a fixed token input,
and dumps each layer's hidden state as `.npy` files under
`tests/fixtures/bert/`. A future Rust test can load these fixtures and diff
layer-by-layer against `CustomAlbert::forward` to find numerical drift.

Usage:
    python3 scripts/validate_bert.py

Requirements (`pip install --break-system-packages torch transformers`):
    torch >= 2.0
    transformers >= 4.30
    numpy
    huggingface_hub (for the .pth file)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel


REPO_ID = "hexgrad/Kokoro-82M"
PTH_FILE = "kokoro-v1_0.pth"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "bert"


def download_checkpoint() -> Path:
    """Return a local path to the real Kokoro-82M .pth file."""
    path = hf_hub_download(repo_id=REPO_ID, filename=PTH_FILE)
    return Path(path)


def build_albert_config() -> AlbertConfig:
    """Match Kokoro's `plbert` sub-config from config.json."""
    return AlbertConfig(
        vocab_size=178,
        embedding_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2048,
        max_position_embeddings=512,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        type_vocab_size=2,
        num_hidden_groups=1,
        inner_group_num=1,
    )


def load_bert_state_dict(pth_path: Path) -> dict[str, torch.Tensor]:
    """Extract the `bert` sub-state-dict from the Kokoro checkpoint.

    Kokoro saves each component under its own top-level key. Within `bert`
    the weights are prefixed with `module.` (DataParallel artifact). Strip
    the prefix to match `AlbertModel`'s native naming.
    """
    full = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(full, dict) or "bert" not in full:
        raise RuntimeError(
            f"Unexpected checkpoint format: keys={list(full.keys())[:5]}..."
        )
    bert_sd = full["bert"]
    stripped: dict[str, torch.Tensor] = {}
    for k, v in bert_sd.items():
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

    print("Loading BERT state dict...")
    bert_sd = load_bert_state_dict(pth_path)
    print(f"  {len(bert_sd)} tensors in bert sub-state-dict")

    print("Instantiating AlbertModel...")
    config = build_albert_config()
    model = AlbertModel(config)
    missing, unexpected = model.load_state_dict(bert_sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(
            f"  WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}"
        )
    model.eval()

    # Fixed token input: BOS + phoneme ids corresponding to "h ɛ l oʊ" from
    # the Kokoro vocab (h=50, ɛ=86, l=54, o=59, ʊ=135) + EOS.
    input_ids = torch.tensor([[0, 50, 86, 54, 59, 135, 0]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    print(f"Running forward pass on input_ids {list(input_ids.shape)}...")
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    print("Dumping fixtures...")
    np.save(OUTPUT_DIR / "input_ids.npy", input_ids.numpy())
    np.save(OUTPUT_DIR / "attention_mask.npy", attention_mask.numpy())
    np.save(OUTPUT_DIR / "last_hidden.npy", out.last_hidden_state.numpy())
    for i, h in enumerate(out.hidden_states):
        np.save(OUTPUT_DIR / f"hidden_{i:02d}.npy", h.numpy())

    # Human-readable manifest.
    manifest = {
        "repo_id": REPO_ID,
        "pth_file": PTH_FILE,
        "pth_path": str(pth_path),
        "input_ids": input_ids.numpy().tolist(),
        "attention_mask": attention_mask.numpy().tolist(),
        "num_hidden_states": len(out.hidden_states),
        "hidden_state_shape": list(out.last_hidden_state.shape),
        "fixture_files": sorted(p.name for p in OUTPUT_DIR.glob("*.npy")),
        "config": {
            "vocab_size": config.vocab_size,
            "embedding_size": config.embedding_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "hidden_act": config.hidden_act,
            "layer_norm_eps": config.layer_norm_eps,
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nFixtures written to {OUTPUT_DIR}/:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {p.name}")

    # Print the first few values of the last hidden state so Rust can eyeball.
    last = out.last_hidden_state.numpy()[0]  # [T, 768]
    print(f"\nLast hidden state shape: {last.shape}")
    print(f"First token, first 8 dims: {last[0, :8]}")
    print(f"Last token, first 8 dims: {last[-1, :8]}")
    print(f"Mean abs: {np.mean(np.abs(last)):.6f}")
    print(f"Std: {np.std(last):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())