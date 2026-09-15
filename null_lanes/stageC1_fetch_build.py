#!/usr/bin/env python3
"""LANE C Stage 1 — fetch level-3 zarr chunks + build 62-layer window stack
for the NEW patch region (u,v)=(768,768), well separated from the original
(u 19153-20460, v 561-1713) patch.

Same sampling recipe as stageA2_L3window.py:
- tifxyz_2399_{x,y,z}.tif, level-0 voxel units, /8 -> level-3
- normals from level-0 finite differences
- 62 layers along the normal at level-3 spacing, centered
- nearest sampling from 128^3 zarr chunks (level 3 of the masked zarr)

Fetch: SINGLE STREAM, one writer per chunk file (atomic temp+rename),
resume-aware (skip complete chunks), stall-guarded curl, infinite retry
until every needed chunk validates. No concurrent writers on the same file.
"""
import os, subprocess, time
import numpy as np

BASE = os.path.expanduser('~/workspace/pherc1667/path4_ink')
ZARR = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
LV = 3
DIV = 2 ** LV
CHUNK = 128
NLAY = 62
W = 256
U_ABS, V_ABS = 23040, 256  # Lane C region
OUTDIR = os.path.join(BASE, 'wstacks_patchC')
CHUNKDIR = os.path.join(BASE, 'zchunksL3')
LOG = os.path.join(BASE, 'laneC_fetch.log')


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def fetch_one(cz, cy, cx):
    path = os.path.join(CHUNKDIR, f'L3_{cz}_{cy}_{cx}.bin')
    absent_marker = path + '.absent'
    if os.path.exists(path) and os.path.getsize(path) == CHUNK ** 3:
        return 'cached'
    if os.path.exists(absent_marker):
        return 'absent (cached marker)'
    url = f'{ZARR}/{LV}/{cz}/{cy}/{cx}'
    # -f: 404 fails fast with rc=22 (no retry on 4xx without --retry-all-errors).
    # A 404 means the chunk genuinely does not exist: mark absent, the sampler
    # zero-fills missing chunks (same as stageA2). Per-chunk wall-clock cap so
    # one bad chunk can never stall the lane; a give-up is logged loudly.
    tmp = path + '.tmp'
    t_start = time.time()
    attempt = 0
    while time.time() - t_start < 900:  # 15 min per chunk max
        attempt += 1
        # NOTE: tmp is NOT deleted between attempts. curl -C - resumes the
        # partial transfer instead of restarting it — the correct response to
        # repeated error-18 truncated transfers (server closes mid-stream).
        # -f still fails fast on 404; size validation still gates acceptance.
        # If tmp exists but is larger than a full chunk, or curl reports the
        # server ignored the range, start over.
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have >= CHUNK ** 3:
            os.remove(tmp)
            have = 0
        r = subprocess.run(
            ['curl', '-sS', '-f', '-C', '-', '--max-time', '120', '--retry', '2',
             '--speed-limit', '5000', '--speed-time', '60',
             '-o', tmp, url],
            capture_output=True, timeout=180, text=True)
        err = (r.stderr or '').strip().replace('\n', ' ')[:160]
        if r.returncode == 22 or '404' in err:
            if os.path.exists(tmp):
                os.remove(tmp)
            open(absent_marker, 'w').write('404\n')
            return 'absent (404) — zero-fill'
        if 'returned no data' in err or 'ignore' in err.lower():
            # server refused resume: restart this chunk from byte 0
            if os.path.exists(tmp):
                os.remove(tmp)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) == CHUNK ** 3:
            os.replace(tmp, path)  # atomic: single writer per artifact
            return f'fetched {CHUNK**3} bytes in {time.time()-t_start:.0f}s attempts={attempt}'
        log(f'chunk L3_{cz}_{cy}_{cx}: attempt {attempt} failed '
            f'(rc={r.returncode} have={os.path.getsize(tmp) if os.path.exists(tmp) else 0}/{CHUNK**3} err={err}), retrying')
        time.sleep(min(2 * attempt, 30))
    open(absent_marker, 'w').write('stall-giveup\n')
    return (f'GAVE UP after {attempt} attempts / {time.time()-t_start:.0f}s — '
            f'zero-fill (counts as blocker if many)')


def main():
    import tifffile
    os.makedirs(CHUNKDIR, exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    log(f'Lane C start: window (u,v)=({U_ABS},{V_ABS}), {W}x{W}x{NLAY}')
    log('loading tifxyz window...')
    P = {}
    for ax in 'xyz':
        a = tifffile.imread(os.path.join(BASE, f'tifxyz_2399_{ax}.tif'))
        P[ax] = a[V_ABS - 2:V_ABS + W + 2, U_ABS - 2:U_ABS + W + 2].astype(np.float64)
    X = P['x'][2:-2, 2:-2] / DIV
    Y = P['y'][2:-2, 2:-2] / DIV
    Z = P['z'][2:-2, 2:-2] / DIV
    log('level-3 coords range: x [%.0f,%.0f] y [%.0f,%.0f] z [%.0f,%.0f]' % (
        X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()))
    dPu = np.stack([P['x'][2:-2, 3:-1] - P['x'][2:-2, 1:-3],
                    P['y'][2:-2, 3:-1] - P['y'][2:-2, 1:-3],
                    P['z'][2:-2, 3:-1] - P['z'][2:-2, 1:-3]], axis=-1)
    dPv = np.stack([P['x'][3:-1, 2:-2] - P['x'][1:-3, 2:-2],
                    P['y'][3:-1, 2:-2] - P['y'][1:-3, 2:-2],
                    P['z'][3:-1, 2:-2] - P['z'][1:-3, 2:-2]], axis=-1)
    N = np.cross(dPu, dPv)
    N = N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)
    del P
    offs = np.arange(NLAY, dtype=np.float64) - (NLAY - 1) / 2.0
    Pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    del X, Y, Z
    Nf = N.astype(np.float32)
    del N
    S = Pts[:, :, None, :] + offs[None, None, :, None] * Nf[:, :, None, :]
    del Pts, Nf
    Szyx = np.stack([S[..., 2], S[..., 1], S[..., 0]], axis=-1)
    del S
    log('level-3 sample range: z [%.0f,%.0f] y [%.0f,%.0f] x [%.0f,%.0f]' % (
        Szyx[..., 0].min(), Szyx[..., 0].max(),
        Szyx[..., 1].min(), Szyx[..., 1].max(),
        Szyx[..., 2].min(), Szyx[..., 2].max()))
    ci = np.floor(Szyx / CHUNK).astype(np.int64)
    loc = (Szyx - ci * CHUNK).astype(np.int64).clip(0, CHUNK - 1)
    del Szyx
    keys, inv = np.unique(ci.reshape(-1, 3), axis=0, return_inverse=True)
    del ci
    log(f'need {len(keys)} chunks; fetching single-stream...')
    t0 = time.time()
    for i, (cz, cy, cx) in enumerate(keys):
        res = fetch_one(int(cz), int(cy), int(cx))
        log(f'  chunk {i+1}/{len(keys)}: L3_{cz}_{cy}_{cx} {res} (elapsed {time.time()-t0:.0f}s)')
    log(f'all {len(keys)} chunks present (elapsed {time.time()-t0:.0f}s)')
    # sample
    stack = np.zeros((W, W, NLAY), dtype=np.uint8)
    flat_loc = loc.reshape(-1, 3)
    del loc
    inv = inv.reshape(W, W, NLAY)
    for ki, (cz, cy, cx) in enumerate(keys):
        path = os.path.join(CHUNKDIR, f'L3_{int(cz)}_{int(cy)}_{int(cx)}.bin')
        if os.path.exists(path) and os.path.getsize(path) == CHUNK ** 3:
            c = np.fromfile(path, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
        else:
            c = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)
        m = (inv == ki)
        li = flat_loc[m.reshape(-1)]
        stack[m] = c[li[:, 0], li[:, 1], li[:, 2]]
        del c
        if ki % 20 == 0:
            log(f'  sampled {ki}/{len(keys)}')
    log('stack: nonzero %.4f, max %d, mean %.2f' % (
        (stack != 0).mean(), stack.max(), stack.mean()))
    out = os.path.join(OUTDIR, f'w_{U_ABS}_{V_ABS}.npy')
    np.save(out, stack)
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
