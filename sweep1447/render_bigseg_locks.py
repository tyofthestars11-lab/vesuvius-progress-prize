#!/usr/bin/env python3
"""Render top LOCKED hotspots for the big segment: prediction | surface | overlay.
Lightweight: uint8 TIFF reads, zarr windowed reads only.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr

BASE = os.path.expanduser("~/workspace/vesuvius-first-letters")
SWEEP = os.path.join(BASE, "PHerc1447/sweep")
SEG = "20251105093211"
HALF = 250

rep = json.load(open(os.path.join(SWEEP, f"lucas_lock_{SEG}.json")))
locked = [h for h in rep["hotspots"] if h["verdict"] == "LOCKED"][:6]
print(f"rendering {len(locked)} locked hotspots")

pred = tifffile.imread(os.path.join(SWEEP, f"ink_pred_{SEG}_seed42_forward.tif")).astype(np.float32) / 255.0
g = zarr.open(os.path.join(SWEEP, f"seg_{SEG}.zarr"), mode="r")
try:
    vol = g["0"]
    _ = vol.shape
except Exception:
    vol = g
D, H, W = vol.shape
print("vol", vol.shape, vol.dtype)
z0, z1 = D // 3, 2 * D // 3  # central slab

fig, axes = plt.subplots(len(locked), 3, figsize=(12, 4 * len(locked)))
for i, hspot in enumerate(locked):
    cy, cx = int(hspot["centroid_px"][0]), int(hspot["centroid_px"][1])
    y0, y1 = max(0, cy - HALF), min(H, cy + HALF)
    x0, x1 = max(0, cx - HALF), min(W, cx + HALF)
    p = pred[y0:y1, x0:x1]
    s = np.asarray(vol[z0:z1, y0:y1, x0:x1]).astype(np.float32).max(axis=0)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    ax = axes[i] if len(locked) > 1 else axes
    ax[0].imshow(p, cmap="hot", vmin=0, vmax=1)
    ax[0].set_title(f"hs{hspot['id']} pred min_peak={hspot['min_peak']}")
    ax[1].imshow(s, cmap="gray")
    ax[1].set_title(f"surface max-proj z[{z0}:{z1}]")
    ax[2].imshow(s, cmap="gray")
    ax[2].imshow(p, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax[2].set_title(f"overlay area={hspot['area_px']}px")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
out = os.path.join(SWEEP, f"locks_{SEG}_top6.png")
plt.savefig(out, dpi=80)
print("saved", out)
