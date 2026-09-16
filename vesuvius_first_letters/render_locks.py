"""Render hotspot crops (prediction | surface | overlay) for LUCAS-locked regions.

Usage: venv/bin/python render_locks.py <lucas_json> <zarr> <pred_tif> [--outdir DIR] [--half WIDTH]
Reads hotspot centroids from the lucas_lock JSON, renders LOCKED (and MARGINAL)
regions as side-by-side prediction | surface | overlay crops.
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


def load_pred(p):
    a = tifffile.imread(p)
    return a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)


def main():
    lj, zpath, ptif = sys.argv[1:4]
    outdir = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else os.path.dirname(lj)
    half = int(sys.argv[sys.argv.index("--half") + 1]) if "--half" in sys.argv else 200
    rep = json.load(open(lj))
    pred = load_pred(ptif)
    g = zarr.open(zpath, mode="r")
    try:
        surf = g["0"][15]  # mid layer surface
    except Exception:
        surf = g[15]
    surf = surf.astype(np.float64)
    n = 0
    for hs in rep["hotspots"]:
        if hs["verdict"] not in ("LOCKED", "MARGINAL"):
            continue
        cy, cx = int(hs["centroid_px"][0]), int(hs["centroid_px"][1])
        y0, y1 = max(0, cy - half), min(pred.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(pred.shape[1], cx + half)
        p = pred[y0:y1, x0:x1]
        s = surf[y0:y1, x0:x1]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(p, cmap="hot", vmin=0, vmax=1)
        ax[0].set_title(f"pred {hs['verdict']} peak={hs['min_peak']:.3f}")
        ax[1].imshow(s, cmap="gray")
        ax[1].set_title("surface (mid layer)")
        ax[2].imshow(s, cmap="gray")
        ax[2].imshow(p, cmap="hot", alpha=0.55, vmin=0, vmax=1)
        ax[2].set_title("overlay")
        for a in ax:
            a.axis("off")
        mm = hs["centroid_mm"]
        fig.suptitle(f"{rep['segment']} hs{hs['id']:02d} @ ({mm[0]:.1f}, {mm[1]:.1f}) mm")
        plt.tight_layout()
        out = os.path.join(outdir, f"lock_{rep['segment']}_hs{hs['id']:02d}_{hs['verdict'].lower()}.png")
        plt.savefig(out, dpi=90)
        plt.close()
        print(f"wrote {out}", flush=True)
        n += 1
    print(f"{n} region renders", flush=True)


if __name__ == "__main__":
    main()
