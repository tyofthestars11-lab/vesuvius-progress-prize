#!/usr/bin/env python3
"""
PHI-STENCIL DEPTH PILOT — Virtual Philodemus φ-stencil recipe on PHerc.1667.

Recipe (Virtual Philodemus, scrollprize.org, 2026-09-15):
  for every mesh vertex, sample the CT volume on a compact geometric stencil along
  the vertex normal, spaced by multiplicative factor φ (φ=(1+√5)/2, φ²=φ+1):
      t[0] = 0 ;  t[+k] = +d·φ^(k-1) , t[-k] = -d·φ^(k-1)   (k = 1..K)
  convert the 2K+1 samples into ONE depth target per vertex with a robust estimator;
  a binary "fuel" mask M(p)∈{0,1} acts as multiplicative +1 weight boosting labeled
  locations so the estimator favors signal over noise.

THIS PILOT: READOUT ONLY. No mesh update. The mesh update waits for TYREE's word.

Pilot region : the gate's column-15 / rung-15 window —
               tifxyz u[19614,19998] × v[950,1334], 384×384 = 147,456 vertices,
               2.399 µm/voxel (window per vstack_384_meta.json).
Mesh         : pilot-region vertices read from the same tifxyz_2399_{x,y,z}.tif
               source the gate's vstack was built from; per-vertex normals =
               vstack_384_normals.npy (central-difference, |N|=1).
Volume       : vstack_384.npy — uint8, 384×384×24. This IS the CT volume resampled
               onto exactly the stencil's geometry: stack[v,u,l] was sampled (at
               build time, from the same zarr chunks) at P(v,u)+(l−11.5)·N(v,u),
               l=0..23 → ±11.5 voxels along each vertex's own normal.
               Stencil query at offset t  ⟺  layer f = 11.5+t, linear-interp
               between floor/ceil layers. (u,v) are grid nodes, so trilinear ≡
               linear along the normal axis here — stated, not hidden.
               WHY NOT RAW CHUNKS: the stencil's world-space footprint touches
               ~2,900 base chunks (5.8 GB; more with trilinear +1 neighbors) vs
               512 MB /tmp on this box (OOM history) — infeasible. All |t|≤8.47
               sit inside the stack's ±11.5 span, so nothing is extrapolated.
Fidelity chk: 16×16=256 center vertices ALSO sampled true-trilinear from the raw
               CT chunks (extracted to /tmp, bounded) and compared against the
               stack-linear profiles — proves the proxy is faithful.
Arms         : d=1 and d=2 voxels, K=4 → 9 samples/vertex/arm.
Estimator    : intensity-weighted centroid around the profile peak:
               w_j = max(0, s_j − median(s)) · (1 + M(p)) ; target = Σw_j·t_j / Σw_j
               Σw=0 → 0 (flat profile, counted). Reported displacement =
               (target − vertex)·normal = the centroid, in voxels.
Fuel mask    : M(p)=1 at vertices coinciding with MEASURED hits from
               hits_vertices.csv mapped into the window (u−19614, v−950).
               Measured result: 0 of 26,932 hits fall in this window → mask is
               all-zero here; the fuel arm is therefore inert on this pilot, which
               doubles as a machinery self-check (fuel≡nofuel must match bit-exact).
               What would make fuel live: a scroll-wide hit run covering the window.

Memory discipline: vectorized over the full 147,456-vertex grid per arm
(147456×9 float32 ≈ 5 MB/arm); chunk-fidelity phase bounded to 256 vertices.
"""
import os, sys, struct, math, json, time, subprocess, resource
import numpy as np

PHI = (1 + 5 ** 0.5) / 2
OUTDIR = os.path.expanduser('~/workspace/vesuvius-prize-repo/phi_stencil')
CHUNKDIR = '/tmp/phi_stencil_chunks'
ARCH = os.path.expanduser('~/workspace/pherc1667/v_recovery/zchunks_v.tar.gz')
TIFD = os.path.expanduser('~/workspace/pherc1667/path4_ink')
VREC = os.path.expanduser('~/workspace/pherc1667/v_recovery')
WU0, WV0, WS = 19614, 950, 384
TW = 30097
CH = 128
K = 4
NLAY = 24
HALF = (NLAY - 1) / 2.0  # 11.5

logf = None  # opened in main(); stays None on plain import (import-safe)

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if logf is not None:
        logf.write(line + '\n'); logf.flush()

def stencil_offsets(d, k=K):
    t = [0.0]
    for i in range(1, k + 1):
        t += [-d * PHI ** (i - 1), d * PHI ** (i - 1)]
    return np.array(sorted(t), dtype=np.float64)

def tif_window(f, u0, v0, w, h):
    with open(f, 'rb') as fh:
        head = fh.read(8)
        bo = '<' if head[:2] == b'II' else '>'
        off = struct.unpack(bo + 'I', head[4:8])[0]
        fh.seek(off); n = struct.unpack(bo + 'H', fh.read(2))[0]
        strip = None
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(bo + 'HHI4s', fh.read(12))
            if tag == 273:
                strip = struct.unpack(bo + 'I', val)[0]
        out = np.empty((h, w), dtype=np.float32)
        for r in range(h):
            fh.seek(strip + ((v0 + r) * TW + u0) * 4)
            out[r] = np.frombuffer(fh.read(w * 4), dtype='<f4')
    return out

def estimate(profiles, offsets, fuel):
    """profiles (n,9) float64, offsets (9,), fuel (n,) {0,1} → centroid vox, flat, peak idx."""
    med = np.median(profiles, axis=1, keepdims=True)
    w = np.clip(profiles - med, 0.0, None) * (1.0 + fuel[:, None])
    s = w.sum(axis=1)
    flat = s <= 0
    s = np.where(flat, 1.0, s)
    cent = (w * offsets[None, :]).sum(axis=1) / s
    cent[flat] = 0.0
    return cent, flat, np.argmax(profiles, axis=1)

# ---------- fidelity phase: true trilinear from raw chunks ----------
def load_chunk(key, cache, stats):
    if key in cache:
        return cache[key]
    p = os.path.join(CHUNKDIR, 'zchunks_v', 'L0_%d_%d_%d.bin' % key)
    if os.path.exists(p) and os.path.getsize(p) == CH ** 3:
        a = np.fromfile(p, dtype=np.uint8).reshape(CH, CH, CH).astype(np.float64)
    else:
        a = np.zeros((CH, CH, CH), dtype=np.float64)
        stats['missing_chunks'].add(key)
    cache[key] = a
    return a

def sample_trilinear(pts_zyx, cache, stats):
    """pts_zyx: (n,3) float64 world-voxel coords in (z,y,x) order. True trilinear."""
    base = np.floor(pts_zyx).astype(np.int64)
    fr = pts_zyx - base
    out = np.zeros(pts_zyx.shape[0], dtype=np.float64)
    for dz in (0, 1):
        wz = fr[:, 0] if dz else 1.0 - fr[:, 0]
        for dy in (0, 1):
            wy = fr[:, 1] if dy else 1.0 - fr[:, 1]
            for dx in (0, 1):
                wx = fr[:, 2] if dx else 1.0 - fr[:, 2]
                cor = base + np.array([dz, dy, dx], dtype=np.int64)
                ckey = cor // CH
                cloc = cor % CH
                uk, inv = np.unique(ckey, axis=0, return_inverse=True)
                vals = np.empty(pts_zyx.shape[0], dtype=np.float64)
                for gi in range(len(uk)):
                    ukt = (int(uk[gi, 0]), int(uk[gi, 1]), int(uk[gi, 2]))
                    arr = load_chunk(ukt, cache, stats)
                    if ukt in stats['missing_chunks']:
                        stats['missing_samples'] += int((inv == gi).sum())
                    m = inv == gi
                    l = cloc[m]
                    vals[m] = arr[l[:, 0], l[:, 1], l[:, 2]]
                out += (wz * wy * wx) * vals
    return out

def main():
    global logf
    logf = open(os.path.join(OUTDIR, 'phi_stencil_depth.log'), 'w')
    t_all = time.perf_counter()
    res = {'phi': PHI, 'K': K, 'window_tifxyz': [WU0, WV0, WS, WS],
           'voxel_um': 2.399, 'arms': [1, 2]}
    # ---- 1. mesh vertices + normals (normals sanity; P kept for fidelity phase) ----
    t0 = time.perf_counter()
    log('loading pilot-region vertices (tifxyz window u[%d,%d] v[%d,%d])'
        % (WU0, WU0 + WS, WV0, WV0 + WS))
    X = tif_window(os.path.join(TIFD, 'tifxyz_2399_x.tif'), WU0, WV0, WS, WS)
    Y = tif_window(os.path.join(TIFD, 'tifxyz_2399_y.tif'), WU0, WV0, WS, WS)
    Z = tif_window(os.path.join(TIFD, 'tifxyz_2399_z.tif'), WU0, WV0, WS, WS)
    P = np.stack([X, Y, Z], -1).astype(np.float64)   # [v,u,:] world voxels
    del X, Y, Z
    N = np.load(os.path.join(VREC, 'vstack_384_normals.npy')).astype(np.float64)
    assert N.shape == (WS, WS, 3)
    nn = np.linalg.norm(N, axis=-1)
    log('vertices: %d | |N| min/mean/max %.6f/%.6f/%.6f'
        % (WS * WS, nn.min(), nn.mean(), nn.max()))
    res['n_vertices'] = WS * WS
    res['t_mesh_load_s'] = round(time.perf_counter() - t0, 2)

    # ---- 2. volume: the uint8 stack ----
    t0 = time.perf_counter()
    stack = np.load(os.path.join(VREC, 'vstack_384.npy'))  # [v,u,l] uint8
    assert stack.shape == (WS, WS, NLAY)
    log('stack: shape %s dtype %s nonzero %.4f min %d max %d mean %.2f'
        % (stack.shape, stack.dtype, (stack != 0).mean(),
           int(stack.min()), int(stack.max()), float(stack.mean())))
    res['stack_stats'] = {'nonzero_frac': float((stack != 0).mean()),
                          'min': int(stack.min()), 'max': int(stack.max()),
                          'mean': float(stack.mean())}
    res['t_stack_load_s'] = round(time.perf_counter() - t0, 2)
    sf = stack.astype(np.float64)
    VV, UU = np.mgrid[0:WS, 0:WS]  # grid nodes

    # ---- 3. fuel mask from measured hits ----
    hd = np.genfromtxt(os.path.expanduser('~/workspace/pherc1667/hits_vertices.csv'),
                       delimiter=',', names=True)
    mask = np.zeros((WS, WS), dtype=np.uint8)
    iu = np.round(hd['u'] - WU0).astype(np.int64)
    iv = np.round(hd['v'] - WV0).astype(np.int64)
    ok = (iu >= 0) & (iu < WS) & (iv >= 0) & (iv < WS)
    mask[iv[ok], iu[ok]] = 1
    fuel = mask.reshape(-1).astype(np.float64)
    log('measured hits: %d total, %d in pilot window → fuel mask %s'
        % (len(hd), int(ok.sum()), 'ALL-ZERO (inert arm)' if ok.sum() == 0 else 'ACTIVE'))
    res['fuel_mask'] = {'n_measured_hits': int(len(hd)), 'n_in_window': int(ok.sum()),
                        'construction': 'M=1 at (round(u)-19614, round(v)-950) from hits_vertices.csv'}

    # ---- 4. stencil profiles per arm (vectorized over full grid) ----
    offs = {d: stencil_offsets(d) for d in (1, 2)}
    for d in (1, 2):
        log('arm d=%d offsets (vox): %s max|t|=%.6f'
            % (d, np.array2string(offs[d], precision=6), float(np.abs(offs[d]).max())))
    res['offsets_vox'] = {str(d): [float(x) for x in offs[d]] for d in (1, 2)}
    profiles = {}
    for d in (1, 2):
        t0 = time.perf_counter()
        t = offs[d]
        f = HALF + t
        l0 = np.floor(f).astype(np.int64).clip(0, NLAY - 2)
        w = f - l0
        prof = np.empty((WS * WS, 2 * K + 1), dtype=np.float32)
        for j in range(2 * K + 1):
            # (u,v) are grid nodes → trilinear ≡ linear along layer axis
            prof[:, j] = ((1.0 - w[j]) * sf[:, :, l0[j]] + w[j] * sf[:, :, l0[j] + 1]).reshape(-1)
        profiles[d] = prof.astype(np.float64)
        np.save(os.path.join(OUTDIR, 'phi_stencil_profiles_d%d.npy' % d), prof)
        log('arm d=%d profiles built in %.1fs' % (d, time.perf_counter() - t0))
        res['arm_d%d' % d] = {'t_profile_s': round(time.perf_counter() - t0, 2)}

    # ---- 5. estimator: fuel off/on per arm ----
    for d in (1, 2):
        for fname, f in (('nofuel', np.zeros_like(fuel)), ('fuel', fuel)):
            cent, flat, peak = estimate(profiles[d], offs[d], f)
            a = np.abs(cent)
            hist, edges = np.histogram(cent, bins=25)
            key = 'arm_d%d_%s' % (d, fname)
            res[key] = {
                'n': int(len(cent)),
                'displacement_vox': {
                    'mean': float(cent.mean()), 'median': float(np.median(cent)),
                    'std': float(cent.std()), 'min': float(cent.min()),
                    'max': float(cent.max())},
                'n_abs_gt_1vox': int((a > 1.0).sum()),
                'frac_abs_gt_1vox': float((a > 1.0).mean()),
                'frac_abs_gt_0.5vox': float((a > 0.5).mean()),
                'n_flat_profile': int(flat.sum()),
                'peak_offset_mean_vox': float(offs[d][peak].mean()),
                'hist_counts': [int(x) for x in hist],
                'hist_edges': [float(x) for x in edges],
            }
            np.save(os.path.join(OUTDIR, 'phi_stencil_disp_d%d_%s.npy' % (d, fname)),
                    cent.astype(np.float32))
            log('%s: mean=%+.4f median=%+.4f std=%.4f min=%+.4f max=%+.4f |d|>1vox %d (%.3f%%) flat %d' % (
                key, cent.mean(), np.median(cent), cent.std(), cent.min(), cent.max(),
                (a > 1.0).sum(), 100 * (a > 1.0).mean(), flat.sum()))
    for d in (1, 2):
        a = np.load(os.path.join(OUTDIR, 'phi_stencil_disp_d%d_nofuel.npy' % d))
        b = np.load(os.path.join(OUTDIR, 'phi_stencil_disp_d%d_fuel.npy' % d))
        same = bool(np.array_equal(a, b))
        log('self-check d=%d fuel≡nofuel: %s' % (d, 'IDENTICAL' if same else 'MISMATCH!'))
        res['arm_d%d' % d]['fuel_selfcheck_identical'] = same

    # ---- 6. fidelity: 16×16 center block, true trilinear from raw chunks ----
    log('fidelity phase: 16x16 center block vs raw CT chunks')
    t0 = time.perf_counter()
    c0 = WS // 2 - 8
    vv, uu = np.mgrid[c0:c0 + 16, c0:c0 + 16].reshape(2, -1)
    nsub = vv.size
    Pv = P[vv, uu]          # (nsub,3) world vox
    Nv = N[vv, uu]
    stats = {'missing_chunks': set(), 'missing_samples': 0}
    need = set()
    Qv = {}
    for d in (1, 2):
        Q = Pv[:, None, :] + offs[d][None, :, None] * Nv[:, None, :]
        Qv[d] = Q[..., [2, 1, 0]]  # (z,y,x)
        kk = np.unique(np.floor(Qv[d].reshape(-1, 3) / CH).astype(np.int64), axis=0)
        for k in map(tuple, kk.tolist()):
            need.add(k)
            need.add((k[0] + 1, k[1], k[2])); need.add((k[0], k[1] + 1, k[2])); need.add((k[0], k[1], k[2] + 1))
    need = sorted(need)
    log('fidelity chunk census: %d (%.1f MB)' % (len(need), len(need) * CH ** 3 / 1e6))
    members = ['zchunks_v/L0_%d_%d_%d.bin' % k for k in need]
    r = subprocess.run(['tar', '-xzf', ARCH, '-C', CHUNKDIR] + members,
                       capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        log('tar note: ' + r.stderr[:300])
    cache = {}
    fid = {}
    for d in (1, 2):
        prof_chunk = np.empty((nsub, 2 * K + 1), dtype=np.float64)
        for j in range(2 * K + 1):
            prof_chunk[:, j] = sample_trilinear(Qv[d][:, j, :], cache, stats)
        prof_stack = profiles[d].reshape(WS, WS, 2 * K + 1)[vv, uu, :]
        diff = np.abs(prof_chunk - prof_stack)
        # Pearson per offset, then mean
        rs = []
        for j in range(2 * K + 1):
            x, y = prof_chunk[:, j], prof_stack[:, j]
            rs.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan'))
        # estimator-level agreement
        c_c, _, _ = estimate(prof_chunk, offs[d], np.zeros(nsub))
        c_s, _, _ = estimate(prof_stack, offs[d], np.zeros(nsub))
        dd = np.abs(c_c - c_s)
        fid['d%d' % d] = {
            'n_vertices': nsub,
            'profile_mean_abs_diff': float(diff.mean()),
            'profile_max_abs_diff': float(diff.max()),
            'profile_pearson_r_mean': float(np.nanmean(rs)),
            'displacement_mean_abs_diff_vox': float(dd.mean()),
            'displacement_max_abs_diff_vox': float(dd.max()),
            'missing_chunks': len(stats['missing_chunks']),
            'missing_samples': int(stats['missing_samples']),
        }
        log('fidelity d=%d: profile |Δ| mean %.3f max %.3f, Pearson r mean %.5f | '
            'displacement |Δ| mean %.4f max %.4f vox (missing chunks %d)' % (
                d, diff.mean(), diff.max(), float(np.nanmean(rs)),
                dd.mean(), dd.max(), len(stats['missing_chunks'])))
    res['fidelity'] = fid
    res['fidelity_note'] = ('16x16 center block: stack-linear profiles vs true-trilinear '
                            'profiles from raw CT chunks.')
    res['t_fidelity_s'] = round(time.perf_counter() - t0, 2)

    res['t_total_s'] = round(time.perf_counter() - t_all, 2)
    res['peak_rss_MB'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    res['estimator'] = ('w_j = max(0, s_j − median(s)) · (1+M(p)); '
                        'displacement = Σw_j·t_j/Σw_j (voxels along normal); Σw=0 → 0')
    res['notes'] = [
        'READOUT ONLY — no mesh update performed.',
        'Sampling volume = vstack_384 uint8 (CT data re-gridded onto vertex-normal geometry at build time); (u,v) grid nodes → trilinear ≡ linear along normal axis.',
        'Raw-chunk world footprint of the full stencil ≈ 2,900 base chunks (5.8 GB) — infeasible for 512 MB /tmp; fidelity subset validates the proxy.',
        'Fuel mask all-zero on this window: the only measured hit block (32^3 level-5 probe, origin (64,112,288)) does not intersect the column-15 window.',
        'd=1 max|t|=4.236 vox; d=2 max|t|=8.472 vox — both inside the stack ±11.5 vox span; no extrapolation.',
    ]
    json.dump(res, open(os.path.join(OUTDIR, 'phi_stencil_results.json'), 'w'), indent=1)
    log('TOTAL %.1fs peakRSS %.1f MB' % (res['t_total_s'], res['peak_rss_MB']))
    log('wrote phi_stencil_results.json + disp/profiles .npy')
    logf.close()

if __name__ == '__main__':
    main()
