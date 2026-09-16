"""LUCAS STABILIZER — cross-seed x cross-direction stability locking.

A hotspot LOCKS as genuine signal only if it holds stable across BOTH model
seeds (42/43) AND BOTH layer directions (forward/reverse). Unstable scattered
responses are REJECTED as model noise. This is the pipeline's documented
false-positive mitigation for the First Letters prize submission.

Locking rule (applied uniformly, no hand-tuning per segment):
  - Detect hotspots as connected components of (avg of 4 maps) > 0.55,
    area >= 25 px, inside the eroded valid-data mask (padding excluded).
  - For each hotspot, measure peak prediction in each of the 4 maps within
    the component bbox dilated by 16 px (tolerates small spatial shifts).
  - LOCKED   : peak >= 0.60 in ALL FOUR maps
  - MARGINAL : peak >= 0.55 in all four, but < 0.60 in at least one
  - REJECTED : peak < 0.55 in any map

Supporting stability numbers per segment:
  - pairwise Pearson r between the 4 maps (global, valid-mask only)
  - strong-pixel (>0.6) IoU: cross-seed (42f vs 43f, 42r vs 43r),
    cross-direction (42f vs 42r, 43f vs 43r)

Usage:
  venv/bin/python lucas_stabilizer.py <seg_id> <zarr> <t42f> <t42r> <t43f> <t43r> [--outdir DIR]
"""
import json
import os
import sys

import numpy as np
import tifffile
import zarr
from scipy import ndimage

THRESH_DETECT = 0.55
THRESH_LOCK = 0.60
MIN_AREA = 25
DILATE_PX = 16
ERODE_PX = 32


def load_pred(path):
    a = tifffile.imread(path)
    # CLI writes uint8 (prediction*255); normalize to [0,1] for thresholds
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    else:
        a = a.astype(np.float64)
        if a.max() > 1.5:  # unexpected scale, normalize defensively
            a = a / 255.0
    return a


def valid_mask(zarr_path):
    g = zarr.open(zarr_path, mode="r")
    # handle multiscale group (level '0' = full res) or bare array
    try:
        arr = g["0"]
        _ = arr.shape
    except Exception:
        arr = g
    vol = arr[:]  # (depth, h, w)
    m = vol.max(axis=0) > 0
    # erode to kill padding-edge artifacts
    m = ndimage.binary_erosion(m, iterations=ERODE_PX)
    return m


def iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def pearson(a, b, mask):
    x = a[mask].ravel()
    y = b[mask].ravel()
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / d) if d else 0.0


def main():
    seg_id, zarr_path, t42f, t42r, t43f, t43r = sys.argv[1:7]
    outdir = sys.argv[8] if len(sys.argv) > 8 and sys.argv[7] == "--outdir" else os.path.dirname(t42f)

    maps = {
        "seed42_forward": load_pred(t42f),
        "seed42_reverse": load_pred(t42r),
        "seed43_forward": load_pred(t43f),
        "seed43_reverse": load_pred(t43r),
    }
    names = list(maps.keys())
    arrs = [maps[n] for n in names]
    h, w = arrs[0].shape
    assert all(a.shape == (h, w) for a in arrs), "map shapes differ"

    print(f"[{seg_id}] building valid mask...", flush=True)
    vmask = valid_mask(zarr_path)
    n_valid = int(vmask.sum())
    print(f"[{seg_id}] valid px: {n_valid} / {h*w} ({100*n_valid/(h*w):.1f}%)", flush=True)

    # global stats per map (valid region only)
    stats = {}
    for n, a in zip(names, arrs):
        v = a[vmask]
        stats[n] = {
            "max": float(v.max()), "mean": float(v.mean()),
            "frac_gt_055": float((v > 0.55).mean()),
            "frac_gt_060": float((v > 0.60).mean()),
        }

    # pairwise Pearson r
    pr = {}
    for i in range(4):
        for j in range(i + 1, 4):
            pr[f"{names[i]}__vs__{names[j]}"] = pearson(arrs[i], arrs[j], vmask)

    # strong-pixel IoU: cross-seed and cross-direction
    s = {n: (a > THRESH_LOCK) & vmask for n, a in zip(names, arrs)}
    ious = {
        "cross_seed_forward_42f_vs_43f": iou(s["seed42_forward"], s["seed43_forward"]),
        "cross_seed_reverse_42r_vs_43r": iou(s["seed42_reverse"], s["seed43_reverse"]),
        "cross_dir_seed42_fwd_vs_rev": iou(s["seed42_forward"], s["seed42_reverse"]),
        "cross_dir_seed43_fwd_vs_rev": iou(s["seed43_forward"], s["seed43_reverse"]),
    }

    # hotspot detection on the 4-map average
    avg = sum(arrs) / 4.0
    det = (avg > THRESH_DETECT) & vmask
    lab, nlab = ndimage.label(det)
    print(f"[{seg_id}] raw components: {nlab}", flush=True)

    hotspots = []
    for i in range(1, nlab + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < MIN_AREA:
            continue
        ys, xs = np.nonzero(comp)
        cy, cx = float(ys.mean()), float(xs.mean())
        # dilated bbox for peak search (tolerate small shifts)
        y0, y1 = max(0, int(ys.min()) - DILATE_PX), min(h, int(ys.max()) + DILATE_PX + 1)
        x0, x1 = max(0, int(xs.min()) - DILATE_PX), min(w, int(xs.max()) + DILATE_PX + 1)
        peaks = {n: float(a[y0:y1, x0:x1].max()) for n, a in zip(names, arrs)}
        means = {n: float(a[y0:y1, x0:x1][comp[y0:y1, x0:x1]].mean()) for n, a in zip(names, arrs)}
        pmin = min(peaks.values())
        if pmin >= THRESH_LOCK:
            verdict = "LOCKED"
        elif pmin >= THRESH_DETECT:
            verdict = "MARGINAL"
        else:
            verdict = "REJECTED"
        hotspots.append({
            "id": len(hotspots),
            "centroid_px": [round(cy, 1), round(cx, 1)],
            "centroid_mm": [round(cy * 0.00864, 2), round(cx * 0.00864, 2)],
            "area_px": area,
            "peaks": {k: round(v, 3) for k, v in peaks.items()},
            "min_peak": round(pmin, 3),
            "verdict": verdict,
        })

    hotspots.sort(key=lambda d: -d["min_peak"])
    n_locked = sum(1 for x in hotspots if x["verdict"] == "LOCKED")
    n_marginal = sum(1 for x in hotspots if x["verdict"] == "MARGINAL")
    n_rejected = sum(1 for x in hotspots if x["verdict"] == "REJECTED")

    report = {
        "segment": seg_id,
        "locking_rule": "LOCKED iff peak>=0.60 in all 4 maps (2 seeds x 2 directions); "
                        "MARGINAL iff peak>=0.55 in all 4; else REJECTED. "
                        "Hotspots = connected components of 4-map avg > 0.55, area>=25px, "
                        "inside valid mask eroded by 32px (padding excluded).",
        "map_stats": stats,
        "pearson_r": {k: round(v, 4) for k, v in pr.items()},
        "strong_pixel_iou": {k: round(v, 4) for k, v in ious.items()},
        "hotspots": hotspots,
        "tally": {"LOCKED": n_locked, "MARGINAL": n_marginal, "REJECTED": n_rejected},
    }
    out = os.path.join(outdir, f"lucas_lock_{seg_id}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[{seg_id}] LOCKED={n_locked} MARGINAL={n_marginal} REJECTED={n_rejected} -> {out}", flush=True)
    for x in hotspots[:12]:
        print(f"  hs{x['id']:02d} {x['verdict']:8s} min_peak={x['min_peak']:.3f} "
              f"mm={x['centroid_mm']} peaks=" +
              " ".join(f"{k.split('_')[0][-2:]}{k.split('_')[1][0]}:{v:.2f}" for k, v in x["peaks"].items()),
              flush=True)


if __name__ == "__main__":
    main()
