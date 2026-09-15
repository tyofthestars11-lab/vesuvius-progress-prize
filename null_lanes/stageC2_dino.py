#!/usr/bin/env python3
"""LANE C Stage 2 — tiled DINO inference on the new patch window.
Same recipe as stageD_dino.py: 2x2 tiles of 64x128x128 (full 64x256x256
forward OOM-kills this 7GB box), EMA weights, sigmoid, z max-project.
Input: wstacks_patchC/w_23040_256.npy  Output: wprobs_dino_patchC/
"""
import glob
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

VILLA = os.path.expanduser("~/workspace/vesuvius-first-letters/villa_src/vesuvius/src")
sys.path.insert(0, VILLA)

from vesuvius.ink_detection.models.checkpoint import (
    load_checkpoint, config_from_checkpoint, load_model_state)
from vesuvius.ink_detection.models.model import make_model

BASE = os.path.expanduser("~/workspace/pherc1667")
CKPT = os.path.join(BASE, "ckpt_dino/ckpt_78k_fullsup.pth")
WDIR = os.path.join(BASE, "path4_ink/wstacks_patchC")
OUT = os.path.join(BASE, "path4_ink/wprobs_dino_patchC")
os.makedirs(OUT, exist_ok=True)

TILE = 128


def infer_tile(model, tile):
    x = torch.from_numpy(tile[None, None])
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        out = out["ink"]
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.sigmoid(out).cpu().numpy()[0, 0]


def main():
    print("[dinoC] loading checkpoint...", flush=True)
    ck = load_checkpoint(CKPT)
    cfg = config_from_checkpoint(ck, source=CKPT)
    model = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dinoC] params: {n_params/1e6:.1f}M", flush=True)
    load_model_state(model, ck["ema_model"] if "ema_model" in ck else ck["model"])
    model.eval()
    print("[dinoC] ema weights loaded, eval mode", flush=True)

    files = sorted(glob.glob(os.path.join(WDIR, "w_*.npy")))
    print(f"[dinoC] {len(files)} windows", flush=True)
    for wf in files:
        t0 = time.time()
        w = np.load(wf).astype(np.float32)
        assert w.shape == (256, 256, 62), w.shape
        vol = np.transpose(w, (2, 0, 1))
        vol = np.pad(vol, ((0, 2), (0, 0), (0, 0)), mode="edge")
        full = np.zeros((64, 256, 256), np.float32)
        for i in range(2):
            for j in range(2):
                tile = vol[:, i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE]
                full[:, i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE] = infer_tile(model, tile)
        pmap = full.max(axis=0)
        name = os.path.splitext(os.path.basename(wf))[0]
        np.save(os.path.join(OUT, f"{name}_dino.npy"), pmap.astype(np.float32))
        print(f"[dinoC] {name}: max={pmap.max():.3f} mean={pmap.mean():.4f} "
              f"frac>0.5={float((pmap>0.5).mean()):.4f} "
              f"frac>0.7={float((pmap>0.7).mean()):.4f} ({time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
