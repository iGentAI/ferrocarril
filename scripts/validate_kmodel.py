#!/usr/bin/env python3
"""End-to-end Kokoro KModel golden-reference harness for Phase 3.

Runs the official `kokoro.model.KModel.forward_with_tokens` on a fixed
input, hooks every major submodule, and dumps every intermediate output
as `.npy` plus a manifest under `tests/fixtures/kmodel/`. Future Rust
golden tests can load these to validate `ProsodyPredictor`, `Decoder`,
`Generator`, and voice-embedding selection layer-by-layer.

Usage:
    python3 scripts/validate_kmodel.py

Requirements (`pip install --break-system-packages misaki kokoro` already done):
    torch
    numpy
    huggingface_hub
    kokoro (editable install from ./kokoro)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from kokoro.model import KModel


REPO_ID = "hexgrad/Kokoro-82M"
PTH_FILE = "kokoro-v1_0.pth"
VOICE_NAME = "af_heart"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "kmodel"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading config and model...")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    pth_path = hf_hub_download(repo_id=REPO_ID, filename=PTH_FILE)
    voice_path = hf_hub_download(repo_id=REPO_ID, filename=f"voices/{VOICE_NAME}.pt")
    print(f"  config: {config_path}")
    print(f"  model: {pth_path}")
    print(f"  voice: {voice_path}")

    print("Instantiating KModel...")
    model = KModel(repo_id=REPO_ID, config=config_path, model=pth_path)
    model.eval()

    # Load the voice pack. Shape: [510, 1, 256].
    voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    print(f"  voice pack shape: {tuple(voice_pack.shape)}")

    # Same input as the BERT/TextEncoder harnesses: BOS + 5 phonemes + EOS.
    input_ids = torch.tensor([[0, 50, 86, 54, 59, 135, 0]], dtype=torch.long)
    num_phonemes = input_ids.shape[1] - 2  # Strip BOS/EOS for voice indexing.
    voice_row = num_phonemes - 1  # Python KModel indexes by len(ps) - 1.
    ref_s = voice_pack[voice_row].clone()  # [1, 256]
    print(f"  num_phonemes={num_phonemes}, voice_row={voice_row}, ref_s shape={tuple(ref_s.shape)}")

    # Save inputs.
    np.save(OUTPUT_DIR / "input_ids.npy", input_ids.numpy())
    np.save(OUTPUT_DIR / "ref_s.npy", ref_s.numpy())
    np.save(OUTPUT_DIR / "voice_pack.npy", voice_pack.numpy())

    # Set up forward hooks on every interesting submodule.
    captured: dict[str, np.ndarray] = {}

    def hook_fn(name: str):
        def fn(_module, _input, output):
            # Output may be a tensor, a tuple, or something else.
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu().numpy()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                # LSTM-style (output, (h, c)) — store the main tensor.
                captured[name] = output[0].detach().cpu().numpy()
        return fn

    # Register hooks on the components we care about.
    targets: list[tuple[str, torch.nn.Module]] = [
        ("bert", model.bert),
        ("bert_encoder", model.bert_encoder),
        ("predictor.text_encoder", model.predictor.text_encoder),
        ("predictor.lstm", model.predictor.lstm),
        ("predictor.duration_proj", model.predictor.duration_proj),
        ("predictor.shared", model.predictor.shared),
        ("predictor.F0_proj", model.predictor.F0_proj),
        ("predictor.N_proj", model.predictor.N_proj),
        ("text_encoder", model.text_encoder),
        ("decoder.encode", model.decoder.encode),
        ("decoder.F0_conv", model.decoder.F0_conv),
        ("decoder.N_conv", model.decoder.N_conv),
        ("decoder.asr_res", model.decoder.asr_res),
        ("decoder.generator.conv_post", model.decoder.generator.conv_post),
    ]
    for i, blk in enumerate(model.predictor.F0):
        targets.append((f"predictor.F0.{i}", blk))
    for i, blk in enumerate(model.predictor.N):
        targets.append((f"predictor.N.{i}", blk))
    for i, blk in enumerate(model.decoder.decode):
        targets.append((f"decoder.decode.{i}", blk))

    handles = []
    for name, mod in targets:
        h = mod.register_forward_hook(hook_fn(name))
        handles.append(h)

    print("Running forward_with_tokens...")
    with torch.no_grad():
        audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, speed=1.0)

    for h in handles:
        h.remove()

    print(f"  audio shape: {tuple(audio.shape)}")
    print(f"  pred_dur: {pred_dur.tolist() if pred_dur.dim() > 0 else pred_dur.item()}")

    # Dump everything.
    captured["audio"] = audio.detach().cpu().numpy()
    captured["pred_dur"] = pred_dur.detach().cpu().numpy()

    print(f"\nDumping {len(captured)} fixtures to {OUTPUT_DIR}/...")
    for name, arr in captured.items():
        safe_name = name.replace(".", "_")
        np.save(OUTPUT_DIR / f"{safe_name}.npy", arr)
        print(f"  {safe_name}.npy: shape={arr.shape}")

    manifest = {
        "repo_id": REPO_ID,
        "pth_file": PTH_FILE,
        "voice_name": VOICE_NAME,
        "input_ids": input_ids.numpy().tolist(),
        "num_phonemes": num_phonemes,
        "voice_row": voice_row,
        "audio_samples": int(audio.numel()),
        "pred_dur": pred_dur.numpy().tolist() if pred_dur.numel() > 1 else int(pred_dur.item()),
        "captured": {
            name: list(arr.shape) for name, arr in captured.items()
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nFixtures complete. Manifest: {OUTPUT_DIR}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())