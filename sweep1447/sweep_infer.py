"""Sweep orchestrator: seed42-forward inference over all 14 sweep segments.

Resumable: skips segments whose output TIFF already exists. Logs per-segment
stats (max, mean, frac>0.55, frac>0.6 in valid region) to sweep_stats.jsonl.
Saves a downsampled overview PNG per segment.

Responding segments (max >= 0.55) are flagged for full 4-way LUCAS locking;
quiet segments are flagged QUIET.

Usage: venv/bin/python sweep_infer.py [--only SEG_ID ...]
Runs in foreground; launch with nohup for background.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import tifffile
import zarr
from scipy import ndimage

BASE = os.path.expanduser("~/workspace/vesuvius-first-letters")
VENV_PY = os.path.join(BASE, "venv/bin/python")
VILLA = os.path.join(BASE, "villa_src/vesuvius/src")
CKPT42 = os.path.join(BASE, "checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth")
SWEEP = os.path.join(BASE, "PHerc1447/sweep")
LOG = os.path.join(SWEEP, "sweep_stats.jsonl")

SEGMENTS = [
    "20250502183421", "20251105093211", "20250502182142", "20250502185519",
    "20250702235910", "20250703025628", "20250502205333", "20250502182456",
    "20250502183138", "20250502184845", "20250502180748", "20250502184658",
    "20250502184201", "20250502180708",
]

INFER_ARGS = ["--stride", "64", "--overlap", "0.5", "--blend-mode", "hann",
              "--batch-size", "8", "--num-workers", "2",
              "--direction", "forward", "--no-compile"]


def done_segs():
    out = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.add(json.loads(line)["segment"])
    return out


def main():
    only = set(a for a in sys.argv[1:] if not a.startswith("--"))
    # support --only flag style too
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        only = set(sys.argv[i + 1:])
    finished = done_segs()
    env = dict(os.environ, PYTHONPATH=VILLA)
    for seg in SEGMENTS:
        if only and seg not in only:
            continue
        if seg in finished:
            print(f"[skip] {seg} already in log", flush=True)
            continue
        zpath = os.path.join(SWEEP, f"seg_{seg}.zarr")
        out_tif = os.path.join(SWEEP, f"ink_pred_{seg}_seed42_forward.tif")
        if not os.path.exists(zpath):
            print(f"[skip] {seg} zarr missing", flush=True)
            continue
        if os.path.exists(out_tif):
            print(f"[note] {seg} tif exists, computing stats only", flush=True)
        else:
            print(f"[infer] {seg} seed42 forward ...", flush=True)
            t0 = time.time()
            r = subprocess.run(
                [VENV_PY, "-u", "-W", "ignore", "-m",
                 "vesuvius.ink_detection.inference.infer",
                 zpath, CKPT42, out_tif] + INFER_ARGS,
                env=env, capture_output=True, text=True)
            dt = time.time() - t0
            if r.returncode != 0 or not os.path.exists(out_tif):
                print(f"[FAIL] {seg} rc={r.returncode}\n{r.stderr[-2000:]}", flush=True)
                continue
            print(f"[infer] {seg} done in {dt:.0f}s", flush=True)
        # stats in valid region (TIFF is uint8 prediction*255 -> normalize)
        a = tifffile.imread(out_tif)
        if a.dtype == np.uint8:
            a = a.astype(np.float64) / 255.0
        else:
            a = a.astype(np.float64)
        g = zarr.open(zpath, mode="r")
        try:
            vol = g["0"][:]
        except Exception:
            vol = g[:]
        vmask = vol.max(axis=0) > 0
        vmask = ndimage.binary_erosion(vmask, iterations=32)
        v = a[vmask]
        rec = {
            "segment": seg,
            "shape": list(a.shape),
            "max": round(float(v.max()), 4),
            "mean": round(float(v.mean()), 4),
            "frac_gt_055": round(float((v > 0.55).mean()), 5),
            "frac_gt_060": round(float((v > 0.60).mean()), 5),
            "verdict": "RESPONDS" if v.max() >= 0.55 else "QUIET",
            "output": out_tif,
        }
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{seg}] max={rec['max']} mean={rec['mean']} "
              f"frac>0.55={rec['frac_gt_055']} -> {rec['verdict']}", flush=True)
        # overview render (downsampled)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            small = a[::4, ::4]
            plt.figure(figsize=(10, 8))
            plt.imshow(small, cmap="hot", vmin=0, vmax=1)
            plt.colorbar(shrink=0.7)
            plt.title(f"PHerc1447 {seg} seed42 forward (max={rec['max']:.3f})")
            plt.tight_layout()
            plt.savefig(os.path.join(SWEEP, f"overview_{seg}_seed42_forward.png"), dpi=80)
            plt.close()
        except Exception as e:
            print(f"[render fail] {seg}: {e}", flush=True)


if __name__ == "__main__":
    main()
