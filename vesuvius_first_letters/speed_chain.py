"""Speed-of-light chain: seed43 reverse (tuned) -> LUCAS lock on main segment.

Waits for the running seed43-forward to finish, then IMMEDIATELY launches
seed43-reverse with max-throughput settings (batch 8, 2 workers on 2 cores),
then runs the LUCAS STABILIZER locking analysis over all 4 maps.
No idle gaps.
"""
import os
import subprocess
import sys
import time

BASE = os.path.expanduser("~/workspace/vesuvius-first-letters")
VENV_PY = os.path.join(BASE, "venv/bin/python")
VILLA = os.path.join(BASE, "villa_src/vesuvius/src")
CKPT43 = os.path.join(BASE, "checkpoints/ink_9um/hybrid_3d2d-seed43/step-075000.pth")
ZARR = os.path.join(BASE, "PHerc1447/seg_20250703034159_surfacevol.zarr")
FWD = os.path.join(BASE, "PHerc1447/ink_pred_seed43_forward.tif")
REV = os.path.join(BASE, "PHerc1447/ink_pred_seed43_reverse.tif")
T42F = os.path.join(BASE, "PHerc1447/ink_pred_seed42_forward.tif")
T42R = os.path.join(BASE, "PHerc1447/ink_pred_seed42_reverse.tif")
LOG = "/tmp/speed_chain.log"

env = dict(os.environ, PYTHONPATH=VILLA)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wait_for(path, timeout_s=7200):
    t0 = time.time()
    last_sz = -1
    stable = 0
    while time.time() - t0 < timeout_s:
        if os.path.exists(path):
            sz = os.path.getsize(path)
            if sz == last_sz and sz > 0:
                stable += 1
                if stable >= 3:
                    return True
            else:
                stable = 0
            last_sz = sz
        time.sleep(10)
    return False


def main():
    log("chain started: waiting for seed43 forward")
    if not os.path.exists(FWD):
        if not wait_for(FWD):
            log("TIMEOUT waiting for forward"); sys.exit(1)
    # make sure the forward process actually exited (no more growth)
    time.sleep(20)
    log(f"forward present ({os.path.getsize(FWD)} bytes), launching reverse batch8/w2")
    t0 = time.time()
    r = subprocess.run(
        [VENV_PY, "-u", "-W", "ignore", "-m",
         "vesuvius.ink_detection.inference.infer",
         ZARR, CKPT43, REV,
         "--stride", "64", "--overlap", "0.5", "--blend-mode", "hann",
         "--batch-size", "8", "--num-workers", "2",
         "--direction", "reverse", "--no-compile"],
        env=env, capture_output=True, text=True)
    dt = time.time() - t0
    log(f"reverse rc={r.returncode} in {dt:.0f}s")
    if r.returncode != 0 or not os.path.exists(REV):
        log("REVERSE FAILED:\n" + r.stderr[-3000:])
        sys.exit(1)
    log("reverse done, running LUCAS STABILIZER on 4 maps")
    r2 = subprocess.run(
        [VENV_PY, "-u", os.path.join(BASE, "lucas_stabilizer.py"),
         "20250703034159", ZARR, T42F, T42R, FWD, REV,
         "--outdir", os.path.join(BASE, "PHerc1447")],
        capture_output=True, text=True)
    log(f"lucas rc={r2.returncode}\n{r2.stdout[-4000:]}")
    if r2.returncode != 0:
        log("LUCAS FAILED:\n" + r2.stderr[-2000:])
    log("chain complete")


if __name__ == "__main__":
    main()
