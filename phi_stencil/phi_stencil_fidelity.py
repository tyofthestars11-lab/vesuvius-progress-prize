#!/usr/bin/env python3
"""
PHI-STENCIL FIDELITY PHASE (standalone).

The main pilot (phi_stencil_depth.py) completed its full-grid readout, then died
mid-fidelity when the runtime wiped /tmp (tmpfs) — extracted chunks and the
python process vanished together. Main-phase .npy outputs survived in OUTDIR.

This script re-runs ONLY the fidelity phase, extracting the 51 bounded chunks to
WORKSPACE scratch (OUTDIR/chunks_fidelity/, deleted afterwards), then:
  16x16=256 center vertices: true-trilinear profiles from raw CT chunks
  vs stack-linear profiles → agreement stats.
It also (re)builds the complete phi_stencil_results.json from the saved .npy
displacement/profile files, since the main script never reached its write step.
"""
import os, sys, json, time, shutil, subprocess, resource
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_stencil_depth import (tif_window, stencil_offsets, sample_trilinear,
                               estimate, PHI, K, NLAY, HALF, CH,
                               WU0, WV0, WS, TIFD, VREC, ARCH)

OUTDIR = os.path.expanduser('~/workspace/vesuvius-prize-repo/phi_stencil')
CHUNKDIR = os.path.join(OUTDIR, 'chunks_fidelity')
os.makedirs(CHUNKDIR, exist_ok=True)

logf = open(os.path.join(OUTDIR, 'phi_stencil_depth.log'), 'a')
def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    logf.write(line + '\n'); logf.flush()

def main():
    t_all = time.perf_counter()
    log('fidelity rerun: /tmp was wiped mid-run; extracting to WORKSPACE scratch')
    offs = {d: stencil_offsets(d) for d in (1, 2)}

    # ---- mesh/normals/profiles (fast reloads) ----
    t0 = time.perf_counter()
    X = tif_window(os.path.join(TIFD, 'tifxyz_2399_x.tif'), WU0, WV0, WS, WS)
    Y = tif_window(os.path.join(TIFD, 'tifxyz_2399_y.tif'), WU0, WV0, WS, WS)
    Z = tif_window(os.path.join(TIFD, 'tifxyz_2399_z.tif'), WU0, WV0, WS, WS)
    P = np.stack([X, Y, Z], -1).astype(np.float64)
    del X, Y, Z
    N = np.load(os.path.join(VREC, 'vstack_384_normals.npy')).astype(np.float64)
    profiles = {d: np.load(os.path.join(OUTDIR, 'phi_stencil_profiles_d%d.npy' % d)).astype(np.float64)
                for d in (1, 2)}
    log('reloads done in %.1fs' % (time.perf_counter() - t0))

    # ---- chunk census for 16x16 center block ----
    c0 = WS // 2 - 8
    vv, uu = np.mgrid[c0:c0 + 16, c0:c0 + 16].reshape(2, -1)
    nsub = vv.size
    Pv, Nv = P[vv, uu], N[vv, uu]
    need = set()
    Qv = {}
    for d in (1, 2):
        Q = Pv[:, None, :] + offs[d][None, :, None] * Nv[:, None, :]
        Qv[d] = Q[..., [2, 1, 0]]
        kk = np.unique(np.floor(Qv[d].reshape(-1, 3) / CH).astype(np.int64), axis=0)
        for k in map(tuple, kk.tolist()):
            need.add(k)
            need.add((k[0] + 1, k[1], k[2])); need.add((k[0], k[1] + 1, k[2])); need.add((k[0], k[1], k[2] + 1))
    need = sorted(need)
    log('fidelity chunk census: %d (%.1f MB) → %s' % (len(need), len(need) * CH ** 3 / 1e6, CHUNKDIR))
    members = ['zchunks_v/L0_%d_%d_%d.bin' % k for k in need]
    t0 = time.perf_counter()
    r = subprocess.run(['tar', '-xzf', ARCH, '-C', CHUNKDIR] + members,
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        log('tar note: ' + r.stderr[:300])
    got = sum(1 for k in need
              if os.path.exists(os.path.join(CHUNKDIR, 'zchunks_v', 'L0_%d_%d_%d.bin' % k)))
    log('extracted %d/%d in %.1fs' % (got, len(need), time.perf_counter() - t0))

    # ---- compare: chunk-trilinear vs stack-linear ----
    import phi_stencil_depth as psd
    psd.CHUNKDIR = CHUNKDIR  # redirect load_chunk to workspace scratch
    stats = {'missing_chunks': set(), 'missing_samples': 0}
    cache = {}
    fid = {}
    for d in (1, 2):
        t0 = time.perf_counter()
        prof_chunk = np.empty((nsub, 2 * K + 1), dtype=np.float64)
        for j in range(2 * K + 1):
            prof_chunk[:, j] = sample_trilinear(Qv[d][:, j, :], cache, stats)
        prof_stack = profiles[d].reshape(WS, WS, 2 * K + 1)[vv, uu, :]
        diff = np.abs(prof_chunk - prof_stack)
        rs = []
        for j in range(2 * K + 1):
            x, y = prof_chunk[:, j], prof_stack[:, j]
            rs.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan'))
        c_c, _, _ = estimate(prof_chunk, offs[d], np.zeros(nsub))
        c_s, _, _ = estimate(prof_stack, offs[d], np.zeros(nsub))
        dd = np.abs(c_c - c_s)
        fid['d%d' % d] = {
            'n_vertices': nsub,
            'profile_mean_abs_diff': float(diff.mean()),
            'profile_max_abs_diff': float(diff.max()),
            'profile_pearson_r_mean': float(np.nanmean(rs)),
            'profile_pearson_r_per_offset': [float(x) for x in rs],
            'displacement_mean_abs_diff_vox': float(dd.mean()),
            'displacement_max_abs_diff_vox': float(dd.max()),
            'missing_chunks': len(stats['missing_chunks']),
            'missing_samples': int(stats['missing_samples']),
            'chunks_extracted': got, 'chunks_needed': len(need),
        }
        log('fidelity d=%d (%.1fs): profile |Δ| mean %.3f max %.3f, Pearson r mean %.6f | '
            'disp |Δ| mean %.4f max %.4f vox, missing chunks %d/%d' % (
                d, time.perf_counter() - t0, diff.mean(), diff.max(), float(np.nanmean(rs)),
                dd.mean(), dd.max(), len(stats['missing_chunks']), len(need)))
    json.dump(fid, open(os.path.join(OUTDIR, 'phi_stencil_fidelity.json'), 'w'), indent=1)

    # ---- rebuild complete results JSON from saved .npy files ----
    log('rebuilding phi_stencil_results.json from saved arrays')
    stack = np.load(os.path.join(VREC, 'vstack_384.npy'))
    res = {'phi': PHI, 'K': K, 'window_tifxyz': [WU0, WV0, WS, WS],
           'voxel_um': 2.399, 'arms': [1, 2], 'n_vertices': WS * WS,
           'stack_stats': {'nonzero_frac': float((stack != 0).mean()),
                           'min': int(stack.min()), 'max': int(stack.max()),
                           'mean': float(stack.mean())},
           'offsets_vox': {str(d): [float(x) for x in offs[d]] for d in (1, 2)},
           'fuel_mask': {'n_measured_hits': 26932, 'n_in_window': 0,
                         'construction': 'M=1 at (round(u)-19614, round(v)-950) from hits_vertices.csv — ALL-ZERO on this window'}}
    fuel0 = np.zeros(WS * WS)
    for d in (1, 2):
        for fname in ('nofuel', 'fuel'):
            cent = np.load(os.path.join(OUTDIR, 'phi_stencil_disp_d%d_%s.npy' % (d, fname))).astype(np.float64)
            # flat count + peak-offset mean need the estimator re-run (cheap)
            _, flat, peak = estimate(profiles[d], offs[d], fuel0)
            a = np.abs(cent)
            hist, edges = np.histogram(cent, bins=25)
            res['arm_d%d_%s' % (d, fname)] = {
                'n': int(len(cent)),
                'displacement_vox': {'mean': float(cent.mean()), 'median': float(np.median(cent)),
                                     'std': float(cent.std()), 'min': float(cent.min()),
                                     'max': float(cent.max())},
                'n_abs_gt_1vox': int((a > 1.0).sum()),
                'frac_abs_gt_1vox': float((a > 1.0).mean()),
                'frac_abs_gt_0.5vox': float((a > 0.5).mean()),
                'n_flat_profile': int(flat.sum()),
                'peak_offset_mean_vox': float(offs[d][peak].mean()),
                'hist_counts': [int(x) for x in hist],
                'hist_edges': [float(x) for x in edges]}
    for d in (1, 2):
        a = np.load(os.path.join(OUTDIR, 'phi_stencil_disp_d%d_nofuel.npy' % d))
        b = np.load(os.path.join(OUTDIR, 'phi_stencil_disp_d%d_fuel.npy' % d))
        res['arm_d%d' % d] = {'fuel_selfcheck_identical': bool(np.array_equal(a, b))}
    res['fidelity'] = fid
    res['fidelity_note'] = ('16x16 center block: stack-linear profiles vs true-trilinear '
                            'profiles from raw CT chunks (workspace scratch; /tmp was wiped mid-run).')
    res['estimator'] = ('w_j = max(0, s_j − median(s)) · (1+M(p)); '
                        'displacement = Σw_j·t_j/Σw_j (voxels along normal); Σw=0 → 0')
    res['notes'] = [
        'READOUT ONLY — no mesh update performed.',
        'Sampling volume = vstack_384 uint8 (CT data re-gridded onto vertex-normal geometry at build time); (u,v) grid nodes → trilinear ≡ linear along normal axis.',
        'Raw-chunk world footprint of the full stencil ≈ 2,900 base chunks (5.8 GB) — infeasible for 512 MB /tmp; fidelity subset validates the proxy instead.',
        'Fuel mask all-zero on this window: the only measured hit block (32^3 level-5 probe, origin (64,112,288)) does not intersect the column-15 window.',
        'd=1 max|t|=4.236 vox; d=2 max|t|=8.472 vox — both inside the stack ±11.5 vox span; no extrapolation.',
        'Fidelity rerun 2026-09-16: /tmp (tmpfs) was wiped by the runtime mid-extraction; chunks re-extracted to workspace scratch and removed after.',
    ]
    res['t_total_s'] = round(time.perf_counter() - t_all, 2)
    res['peak_rss_MB'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    json.dump(res, open(os.path.join(OUTDIR, 'phi_stencil_results.json'), 'w'), indent=1)

    # ---- cleanup scratch (fidelity json kept; numbers also folded into results) ----
    shutil.rmtree(CHUNKDIR, ignore_errors=True)
    log('scratch removed; TOTAL %.1fs peakRSS %.1f MB' % (res['t_total_s'], res['peak_rss_MB']))
    log('wrote phi_stencil_results.json (fidelity folded in)')
    logf.close()

if __name__ == '__main__':
    main()
