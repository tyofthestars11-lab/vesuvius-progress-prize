"""Full 4-way LUCAS lock for responding sweep segments.

For each RESPONDING segment (seed42-forward max >= 0.55):
  1. seed42 reverse
  2. seed43 forward
  3. seed43 reverse
  4. lucas_stabilizer.py on all 4 maps
  5. render_locks.py for LOCKED/MARGINAL regions

Processes segments in heat order (frac>0.55 descending). Resumable:
skips segments with a lucas_lock JSON already present.
"""
import json
import os
import subprocess
import sys
import time

BASE = os.path.expanduser("~/workspace/vesuvius-first-letters")
VENV_PY = os.path.join(BASE, "venv/bin/python")
VILLA = os.path.join(BASE, "villa_src/vesuvius/src")
CKPT42 = os.path.join(BASE, "checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth")
CKPT43 = os.path.join(BASE, "checkpoints/ink_9um/hybrid_3d2d-seed43/step-075000.pth")
SWEEP = os.path.join(BASE, "PHerc1447/sweep")
LOG = os.path.join(SWEEP, "lock4way.log")
INFER_ARGS = ["--stride", "64", "--overlap", "0.5", "--blend-mode", "hann",
              "--batch-size", "8", "--num-workers", "2", "--no-compile"]

env = dict(os.environ, PYTHONPATH=VILLA)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_infer(zarr, ckpt, out_tif, direction):
    t0 = time.time()
    r = subprocess.run(
        [VENV_PY, "-u", "-W", "ignore", "-m",
         "vesuvius.ink_detection.inference.infer",
         zarr, ckpt, out_tif] + INFER_ARGS + ["--direction", direction],
        env=env, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0 and os.path.exists(out_tif)
    log(f"  infer {os.path.basename(out_tif)} {direction}: "
        f"{'OK' if ok else 'FAIL'} in {dt:.0f}s")
    if not ok:
        log("  stderr: " + r.stderr[-1500:])
    return ok


def main():
    segs = []
    for line in open(os.path.join(SWEEP, "sweep_stats.jsonl")):
        r = json.loads(line)
        if r["verdict"] == "RESPONDS":
            segs.append((r["frac_gt_055"], r["segment"]))
    segs.sort(reverse=True)  # hottest first
    log(f"{len(segs)} responding segments, hottest first")
    for _, seg in segs:
        lock_json = os.path.join(SWEEP, f"lucas_lock_{seg}.json")
        if os.path.exists(lock_json):
            log(f"[skip] {seg} already locked")
            continue
        log(f"=== {seg} ===")
        zarr = os.path.join(SWEEP, f"seg_{seg}.zarr")
        f42 = os.path.join(SWEEP, f"ink_pred_{seg}_seed42_forward.tif")
        r42 = os.path.join(SWEEP, f"ink_pred_{seg}_seed42_reverse.tif")
        f43 = os.path.join(SWEEP, f"ink_pred_{seg}_seed43_forward.tif")
        r43 = os.path.join(SWEEP, f"ink_pred_{seg}_seed43_reverse.tif")
        if not os.path.exists(f42):
            log(f"  [fail] missing {f42}, skipping segment")
            continue
        ok = True
        if not os.path.exists(r42):
            ok = run_infer(zarr, CKPT42, r42, "reverse") and ok
        if not os.path.exists(f43):
            ok = run_infer(zarr, CKPT43, f43, "forward") and ok
        if not os.path.exists(r43):
            ok = run_infer(zarr, CKPT43, r43, "reverse") and ok
        if not ok:
            log(f"  [fail] {seg} inference incomplete, skipping lock")
            continue
        # LUCAS lock
        r = subprocess.run(
            [VENV_PY, "-u", os.path.join(BASE, "lucas_stabilizer.py"),
             seg, zarr, f42, r42, f43, r43, "--outdir", SWEEP],
            capture_output=True, text=True)
        log(f"  lucas rc={r.returncode}")
        if r.returncode == 0 and os.path.exists(lock_json):
            rep = json.load(open(lock_json))
            log(f"  tally {rep['tally']}")
            # renders
            r2 = subprocess.run(
                [VENV_PY, "-u", os.path.join(BASE, "render_locks.py"),
                 lock_json, zarr, f42, "--outdir", SWEEP, "--half", "200"],
                capture_output=True, text=True)
            log(f"  renders rc={r2.returncode}")
        else:
            log("  LUCAS FAILED: " + r.stderr[-1500:])
    log("4-way locking complete")


if __name__ == "__main__":
    main()
