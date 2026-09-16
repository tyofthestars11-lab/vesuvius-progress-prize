#!/usr/bin/env python3
"""V2 stack build + stencil run for one window.
Replicates build_vstack.py (nearest-neighbor 24-layer stack from 128^3 chunks)
then the pilot's stencil estimator (phi_stencil_depth.py) on the stack.

Usage: build_stencil.py <window_dir>   (contains vertices_xyz.npy, normals.npy,
         chunk_keys.json, window_meta.json; chunks in <window_dir>/chunks/)
Writes: stack.npy, <arm>_<fuel|nofuel> disp .npy, results json, log.
"""
import os, sys, json, time, math, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PHI = (1 + 5 ** 0.5) / 2
K = 4
WS, NLAY, CHUNK = 384, 24, 128
HALF = (NLAY - 1) / 2.0
CHUNK_BYTES = CHUNK ** 3
ZARR = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
CURL = ['curl', '-s', '-C', '-', '--retry', '5', '--retry-delay', '2']

logf = None
def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    if logf:
        logf.write(line + '\n'); logf.flush()

def good(p):
    return os.path.exists(p) and os.path.getsize(p) == CHUNK_BYTES

def robust_fetch(p, url, attempts=14):
    for _ in range(attempts):
        if good(p):
            return True
        try:
            subprocess.run(CURL + ['-o', p, url], capture_output=True, timeout=180)
        except Exception:
            pass
        if good(p):
            return True
        try:
            with open(p, 'rb') as f:
                head = f.read(200)
            if head.startswith(b'<?xml') or (os.path.getsize(p) < 100000 and b'NoSuchKey' in head):
                return False  # 404: not yet addressed
        except Exception:
            pass
    return good(p)

def stencil_offsets(d, k=K):
    t = [0.0]
    for i in range(1, k + 1):
        t += [-d * PHI ** (i - 1), d * PHI ** (i - 1)]
    return np.array(sorted(t), dtype=np.float64)

def estimate(profiles, offsets, fuel):
    med = np.median(profiles, axis=1, keepdims=True)
    w = np.clip(profiles - med, 0.0, None) * (1.0 + fuel[:, None])
    s = w.sum(axis=1)
    flat = s <= 0
    s = np.where(flat, 1.0, s)
    cent = (w * offsets[None, :]).sum(axis=1) / s
    cent[flat] = 0.0
    return cent, flat

def main(wdir):
    global logf
    t_all = time.perf_counter()
    logf = open(os.path.join(wdir, 'build_stencil.log'), 'w')
    meta = json.load(open(os.path.join(wdir, 'window_meta.json')))
    u0, v0 = meta['window_tifxyz'][:2]
    log('window u[%d,%d] v[%d,%d]' % (u0, u0 + WS, v0, v0 + WS))

    V = np.load(os.path.join(wdir, 'vertices_xyz.npy')).astype(np.float64)  # [v,u,3]
    N = np.load(os.path.join(wdir, 'normals.npy')).astype(np.float64)
    assert V.shape == (WS, WS, 3) and N.shape == (WS, WS, 3)

    # ---- fuel mask from measured hits ----
    hd = np.genfromtxt(os.path.expanduser('~/workspace/pherc1667/hits_vertices.csv'),
                       delimiter=',', names=True)
    mask = np.zeros((WS, WS), dtype=np.uint8)
    iu = np.round(hd['u'] - u0).astype(np.int64)
    iv = np.round(hd['v'] - v0).astype(np.int64)
    ok = (iu >= 0) & (iu < WS) & (iv >= 0) & (iv < WS)
    mask[iv[ok], iu[ok]] = 1
    fuel = mask.reshape(-1).astype(np.float64)
    log('measured hits: %d total, %d in window' % (len(hd), int(ok.sum())))

    # ---- chunk fetch (local chunks/ dir, then S3 resume for missing) ----
    cdir = os.path.join(wdir, 'chunks')
    os.makedirs(cdir, exist_ok=True)
    keys = [tuple(k) for k in json.load(open(os.path.join(wdir, 'chunk_keys.json')))]
    todo = []
    for k in keys:
        p = os.path.join(cdir, 'L0_%d_%d_%d.bin' % k)
        if not good(p):
            todo.append((k, p))
    log('chunks needed %d, already local %d, to fetch %d' % (len(keys), len(keys) - len(todo), len(todo)))
    missing = []
    if todo:
        def one(item):
            k, p = item
            cz, cy, cx = k
            if robust_fetch(p, '%s/0/%d/%d/%d' % (ZARR, cz, cy, cx)):
                return None
            return k
        with ThreadPoolExecutor(max_workers=24) as ex:
            res = list(ex.map(one, todo))
        missing = [k for k in res if k is not None]
        log('S3 fetch done: %d missing/404 of %d' % (len(missing), len(todo)))
    cache = {}
    def load_chunk(k):
        if k in cache:
            return cache[k]
        p = os.path.join(cdir, 'L0_%d_%d_%d.bin' % k)
        if good(p):
            a = np.fromfile(p, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
        else:
            a = None
        cache[k] = a
        if len(cache) > 400:
            cache.pop(next(iter(cache)))
        return a

    # ---- 24-layer stack, nearest-neighbor (== build_vstack.py) ----
    t0 = time.perf_counter()
    offs = np.arange(NLAY, dtype=np.float64) - HALF
    X, Y, Z = V[..., 0], V[..., 1], V[..., 2]
    stack = np.zeros((WS, WS, NLAY), dtype=np.uint8)
    nmiss = 0
    BAND = 96
    for b0 in range(0, WS, BAND):
        b1 = min(b0 + BAND, WS)
        Sb = np.stack([X[b0:b1], Y[b0:b1], Z[b0:b1]], -1).astype(np.float32)[:, :, None, :] + \
             offs[None, None, :, None] * N[b0:b1, :, None, :].astype(np.float32)
        Szyx = np.stack([Sb[..., 2], Sb[..., 1], Sb[..., 0]], -1).astype(np.float32)
        ci = np.floor(Szyx / CHUNK).astype(np.int64)
        loc = (Szyx - ci * CHUNK).astype(np.int64).clip(0, CHUNK - 1)
        del Sb, Szyx
        ukeys, inv = np.unique(ci.reshape(-1, 3), axis=0, return_inverse=True)
        del ci
        fl = loc.reshape(-1, 3)
        del loc
        invb = inv.reshape(b1 - b0, WS, NLAY)
        del inv
        for ki, kk in enumerate(ukeys):
            c = load_chunk((int(kk[0]), int(kk[1]), int(kk[2])))
            if c is None:
                nmiss += int((invb == ki).sum())
                continue
            m = (invb == ki)
            li = fl[m.reshape(-1)]
            stack[b0:b1][m] = c[li[:, 0], li[:, 1], li[:, 2]]
        log('  band [%d,%d] %d chunk-keys' % (b0, b1, len(ukeys)))
    np.save(os.path.join(wdir, 'stack.npy'), stack)
    log('stack built in %.1fs; samples on missing chunks: %d (of %d)' %
        (time.perf_counter() - t0, nmiss, WS * WS * NLAY))
    log('stack stats: nonzero %.4f min %d max %d mean %.2f' %
        ((stack != 0).mean(), int(stack.min()), int(stack.max()), float(stack.mean())))
    sf = stack.astype(np.float64)

    # ---- stencil profiles + estimator (== phi_stencil_depth.py) ----
    res = {'phi': PHI, 'K': K, 'window_tifxyz': [u0, v0, WS, WS],
           'fuel_mask': {'n_measured_hits': int(len(hd)), 'n_in_window': int(ok.sum())},
           'missing_chunk_samples': int(nmiss)}
    soffs = {d: stencil_offsets(d) for d in (1, 2)}
    res['offsets_vox'] = {str(d): [float(x) for x in soffs[d]] for d in (1, 2)}
    for d in (1, 2):
        t = soffs[d]
        f = HALF + t
        l0 = np.floor(f).astype(np.int64).clip(0, NLAY - 2)
        w = f - l0
        prof = np.empty((WS * WS, 2 * K + 1), dtype=np.float64)
        for j in range(2 * K + 1):
            prof[:, j] = ((1.0 - w[j]) * sf[:, :, l0[j]] + w[j] * sf[:, :, l0[j] + 1]).reshape(-1)
        np.save(os.path.join(wdir, 'profiles_d%d.npy' % d), prof.astype(np.float32))
        for fname, ff in (('nofuel', np.zeros_like(fuel)), ('fuel', fuel)):
            cent, flat = estimate(prof, t, ff)
            a = np.abs(cent)
            key = 'arm_d%d_%s' % (d, fname)
            res[key] = {
                'n': int(len(cent)),
                'mean': float(cent.mean()), 'median': float(np.median(cent)),
                'std': float(cent.std()), 'min': float(cent.min()), 'max': float(cent.max()),
                'n_abs_gt_1vox': int((a > 1.0).sum()),
                'frac_abs_gt_1vox': float((a > 1.0).mean()),
                'frac_abs_gt_0.5vox': float((a > 0.5).mean()),
                'n_flat_profile': int(flat.sum()),
            }
            np.save(os.path.join(wdir, 'disp_d%d_%s.npy' % (d, fname)), cent.astype(np.float32))
            log('%s: mean=%+.4f median=%+.4f std=%.4f min=%+.4f max=%+.4f |d|>1 %d (%.3f%%) flat %d' % (
                key, cent.mean(), np.median(cent), cent.std(), cent.min(), cent.max(),
                (a > 1).sum(), 100 * (a > 1).mean(), flat.sum()))
    for d in (1, 2):
        a = np.load(os.path.join(wdir, 'disp_d%d_nofuel.npy' % d))
        b = np.load(os.path.join(wdir, 'disp_d%d_fuel.npy' % d))
        same = bool(np.array_equal(a, b))
        ndiff = int((a != b).sum())
        dmax = float(np.abs(a - b).max()) if ndiff else 0.0
        log('fuel-vs-nofuel d=%d: %s (n_diff=%d, max|diff|=%.4f vox)' %
            (d, 'IDENTICAL' if same else 'DIFFER', ndiff, dmax))
        res['arm_d%d' % d] = {'fuel_selfcheck_identical': same, 'n_diff': ndiff,
                              'max_abs_diff_vox': dmax}
    # cross-arm pearson
    a1 = np.load(os.path.join(wdir, 'disp_d1_nofuel.npy')).astype(np.float64)
    a2 = np.load(os.path.join(wdir, 'disp_d2_nofuel.npy')).astype(np.float64)
    res['cross_arm_pearson_nofuel'] = float(np.corrcoef(a1, a2)[0, 1])
    log('cross-arm Pearson r(d1,d2) nofuel = %.4f' % res['cross_arm_pearson_nofuel'])
    res['t_total_s'] = round(time.perf_counter() - t_all, 2)
    json.dump(res, open(os.path.join(wdir, 'results.json'), 'w'), indent=1)
    log('TOTAL %.1fs — wrote results.json + disp/profiles/stack .npy' % res['t_total_s'])
    logf.close()

if __name__ == '__main__':
    main(sys.argv[1])
