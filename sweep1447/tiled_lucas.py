#!/usr/bin/env python3
"""Tiled LUCAS stabilizer — OOM fix for segment 20251105093211 (112M px).

Same locking rule as lucas_stabilizer.py, no science changed:
  - Hotspots = connected components of 4-map avg > 0.55, area >= 25 px, in vmask
  - LOCKED iff peak >= 0.60 in ALL FOUR maps (2 seeds x 2 directions)
  - MARGINAL iff peak >= 0.55 in all four; else REJECTED
  - Peak search in component bbox dilated by 16 px (full-image context via memmap)

Tiling: 1024x1024 tiles, 64 px overlap on every side. A component is owned by
the tile whose non-overlap core contains its centroid -> counted exactly once.
Global stats (Pearson, IoU) accumulated per tile from streaming sums.
Peak memory ~200 MB regardless of segment size.
"""
import json
import os
import sys
import time

import numpy as np
import tifffile
import zarr
from scipy import ndimage

THRESH_DETECT = 0.55
THRESH_LOCK = 0.60
MIN_AREA = 25
DILATE_PX = 16
ERODE_PX = 32
TILE = 1024
OV = 64

BASE = os.path.expanduser("~/workspace/vesuvius-first-letters")
SWEEP = os.path.join(BASE, "PHerc1447/sweep")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_vmask(zarr_path, h, w):
    log("building valid mask (streamed)...")
    g = zarr.open(zarr_path, mode="r")
    try:
        arr = g["0"]
        _ = arr.shape
    except Exception:
        arr = g
    d = arr.shape[0]
    m = np.zeros((h, w), dtype=bool)
    step = max(1, d // 8)
    for z0 in range(0, d, step):
        blk = np.asarray(arr[z0:z0 + step])
        m |= blk.reshape(-1, h, w).max(axis=0) > 0
        log(f"  depth {z0 + step}/{d}")
    m = ndimage.binary_erosion(m, iterations=ERODE_PX)
    log(f"valid px: {m.sum()} / {h * w}")
    return m


def main():
    seg_id = sys.argv[1]
    t42f = os.path.join(SWEEP, f"ink_pred_{seg_id}_seed42_forward.tif")
    t42r = os.path.join(SWEEP, f"ink_pred_{seg_id}_seed42_reverse.tif")
    t43f = os.path.join(SWEEP, f"ink_pred_{seg_id}_seed43_forward.tif")
    t43r = os.path.join(SWEEP, f"ink_pred_{seg_id}_seed43_reverse.tif")
    zarr_path = os.path.join(SWEEP, f"seg_{seg_id}.zarr")
    names = ["seed42_forward", "seed42_reverse", "seed43_forward", "seed43_reverse"]
    paths = [t42f, t42r, t43f, t43r]
    for p in paths:
        assert os.path.exists(p), f"missing {p}"

    log("reading 4 TIFFs as uint8 (112M px = 112MB each, fits in RAM)...")
    mms = []
    for p in paths:
        log(f"  {os.path.basename(p)}")
        a = tifffile.imread(p)
        assert a.dtype == np.uint8, a.dtype
        mms.append(a)
    h, w = mms[0].shape
    assert all(m.shape == (h, w) for m in mms)
    log(f"shape {h}x{w} = {h * w / 1e6:.1f}M px")

    vmask = build_vmask(zarr_path, h, w)

    # streaming accumulators
    n = np.zeros(4, dtype=np.float64)
    sx = np.zeros(4, dtype=np.float64)
    sx2 = np.zeros(4, dtype=np.float64)
    sxy = np.zeros((4, 4), dtype=np.float64)
    iou_inter = np.zeros(4, dtype=np.float64)  # pairs: (0,2),(1,3),(0,1),(2,3)
    iou_union = np.zeros(4, dtype=np.float64)
    hotspots = []
    PAIRS = [(0, 2), (1, 3), (0, 1), (2, 3)]
    PAIR_NAMES = ["cross_seed_forward_42f_vs_43f", "cross_seed_reverse_42r_vs_43r",
                  "cross_dir_seed42_fwd_vs_rev", "cross_dir_seed43_fwd_vs_rev"]

    nty = (h + TILE - 1) // TILE
    ntx = (w + TILE - 1) // TILE
    log(f"{nty}x{ntx} tiles")
    t0 = time.time()
    for ty in range(nty):
        for tx in range(ntx):
            # core (ownership) region
            cy0, cy1 = ty * TILE, min((ty + 1) * TILE, h)
            cx0, cx1 = tx * TILE, min((tx + 1) * TILE, w)
            # extended region (detection context)
            ey0, ey1 = max(0, cy0 - OV), min(h, cy1 + OV)
            ex0, ex1 = max(0, cx0 - OV), min(w, cx1 + OV)
            vm = vmask[ey0:ey1, ex0:ex1]
            if not vm.any():
                continue
            tiles = [mms[k][ey0:ey1, ex0:ex1].astype(np.float32) / 255.0 for k in range(4)]
            tv = vm
            cnt = tv.sum()
            for k in range(4):
                xk = tiles[k][tv]
                n[k] += cnt
                sx[k] += xk.sum()
                sx2[k] += (xk ** 2).sum()
            for k in range(4):
                for j in range(k + 1, 4):
                    sxy[k, j] += (tiles[k][tv] * tiles[j][tv]).sum()
            strong = [(tiles[k] > THRESH_LOCK) & vm for k in range(4)]
            for pi, (a, b) in enumerate(PAIRS):
                iou_inter[pi] += np.logical_and(strong[a], strong[b]).sum()
                iou_union[pi] += np.logical_or(strong[a], strong[b]).sum()
            # hotspot detection on 4-map average
            avg = (tiles[0] + tiles[1] + tiles[2] + tiles[3]) / 4.0
            det = (avg > THRESH_DETECT) & vm
            if not det.any():
                continue
            lab, nl = ndimage.label(det)
            for i in range(1, nl + 1):
                ys, xs = np.nonzero(lab == i)
                area = len(ys)
                if area < MIN_AREA:
                    continue
                gcy, gcx = ys.mean() + ey0, xs.mean() + ex0
                if not (cy0 <= gcy < cy1 and cx0 <= gcx < cx1):
                    continue  # owned by neighboring tile
                # exact peak search in dilated bbox, full-image context
                y0 = max(0, int(ys.min() + ey0) - DILATE_PX)
                y1 = min(h, int(ys.max() + ey0) + DILATE_PX + 1)
                x0 = max(0, int(xs.min() + ex0) - DILATE_PX)
                x1 = min(w, int(xs.max() + ex0) + DILATE_PX + 1)
                peaks = {}
                for k, nm in enumerate(names):
                    peaks[nm] = float(mms[k][y0:y1, x0:x1].astype(np.float32).max() / 255.0)
                pmin = min(peaks.values())
                verdict = "LOCKED" if pmin >= THRESH_LOCK else \
                          "MARGINAL" if pmin >= THRESH_DETECT else "REJECTED"
                hotspots.append({
                    "id": len(hotspots),
                    "centroid_px": [round(float(gcy), 1), round(float(gcx), 1)],
                    "centroid_mm": [round(float(gcy) * 0.00864, 2), round(float(gcx) * 0.00864, 2)],
                    "area_px": int(area),
                    "peaks": {k: round(v, 3) for k, v in peaks.items()},
                    "min_peak": round(pmin, 3),
                    "verdict": verdict,
                })
        log(f"row {ty + 1}/{nty} done ({time.time() - t0:.0f}s)")

    # global stats from accumulators
    stats = {}
    for k, nm in enumerate(names):
        stats[nm] = {
            "max": None,  # computed below (streaming max)
            "mean": float(sx[k] / n[k]),
            "frac_gt_055": None, "frac_gt_060": None,
        }
    pr = {}
    for k in range(4):
        for j in range(k + 1, 4):
            nk = n[k]
            cov = (sxy[k, j] - sx[k] * sx[j] / nk) / nk
            vk = (sx2[k] - sx[k] ** 2 / nk) / nk
            vj = (sx2[j] - sx[j] ** 2 / nk) / nk
            pr[f"{names[k]}__vs__{names[j]}"] = round(float(cov / np.sqrt(vk * vj)) if vk > 0 and vj > 0 else 0.0, 4)
    ious = {pn: round(float(iou_inter[pi] / iou_union[pi]) if iou_union[pi] else 0.0, 4)
            for pi, pn in enumerate(PAIR_NAMES)}

    # streaming max + frac>threshold (second light pass over memmaps)
    log("second pass: max + frac>threshold...")
    mx = np.zeros(4)
    f55 = np.zeros(4)
    f60 = np.zeros(4)
    for ty in range(nty):
        for tx in range(ntx):
            y0, y1 = ty * TILE, min((ty + 1) * TILE, h)
            x0, x1 = tx * TILE, min((tx + 1) * TILE, w)
            vm = vmask[y0:y1, x0:x1]
            if not vm.any():
                continue
            for k in range(4):
                t = mms[k][y0:y1, x0:x1].astype(np.float32)[vm] / 255.0
                if t.size:
                    mx[k] = max(mx[k], float(t.max()))
                    f55[k] += (t > THRESH_DETECT).sum()
                    f60[k] += (t > THRESH_LOCK).sum()
    for k, nm in enumerate(names):
        stats[nm]["max"] = round(float(mx[k]), 4)
        stats[nm]["frac_gt_055"] = round(float(f55[k] / n[k]), 5)
        stats[nm]["frac_gt_060"] = round(float(f60[k] / n[k]), 5)

    hotspots.sort(key=lambda d: -d["min_peak"])
    for i, x in enumerate(hotspots):
        x["id"] = i
    n_locked = sum(1 for x in hotspots if x["verdict"] == "LOCKED")
    n_marginal = sum(1 for x in hotspots if x["verdict"] == "MARGINAL")
    n_rejected = sum(1 for x in hotspots if x["verdict"] == "REJECTED")

    report = {
        "segment": seg_id,
        "tiled": True,
        "locking_rule": "TILED (1024px, 64px overlap, core-centroid ownership). "
                        "Same rule as lucas_stabilizer.py: LOCKED iff peak>=0.60 in all 4 maps "
                        "(2 seeds x 2 directions); MARGINAL iff peak>=0.55 in all 4; else REJECTED. "
                        "Hotspots = connected components of 4-map avg > 0.55, area>=25px, "
                        "inside valid mask eroded by 32px (padding excluded).",
        "map_stats": stats,
        "pearson_r": pr,
        "strong_pixel_iou": ious,
        "hotspots": hotspots,
        "tally": {"LOCKED": n_locked, "MARGINAL": n_marginal, "REJECTED": n_rejected},
    }
    out = os.path.join(SWEEP, f"lucas_lock_{seg_id}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    log(f"LOCKED={n_locked} MARGINAL={n_marginal} REJECTED={n_rejected} -> {out}")
    for x in hotspots[:12]:
        log(f"  hs{x['id']:02d} {x['verdict']:8s} min_peak={x['min_peak']:.3f} mm={x['centroid_mm']}")


if __name__ == "__main__":
    main()
