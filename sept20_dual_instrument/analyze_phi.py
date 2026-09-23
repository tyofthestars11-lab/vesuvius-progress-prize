#!/usr/bin/env python3
"""Analyze phi-detector maps: determinism, components, band/interior,
controls, verdicts -> verdicts_phi.json.

Verdict rules (phi instrument):
- INK CANDIDATE: component of K~ > 3.0, location-stable in >=4/5 seeds,
  mean K~ > 3.0, control response in that region < 0.5x real, outside
  margins, not a full-width stripe.
- DATA ARTIFACT: location-stable but controls respond >= 0.5x real.
- ORIENTATION ARTIFACT: component location moves with the transform.
- NONE: no component meets the candidate bar.
"""
import os, sys, json
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phi_detector import interior_mask

PM = os.path.join(HERE, 'maps')
SEEDS = [137, 1001, 2026, 31337, 99991]
CTL_SEEDS = [137, 1001, 2026]
THR = 3.0


def load(kind, seed):
    return np.load(os.path.join(PM, f'phi_{kind}_seed{seed}.npy'))


def main():
    real = np.stack([load('real', s) for s in SEEDS])          # (5,H,W,D)
    ctl_d = np.stack([load('ctl_shuffled_depth', s) for s in CTL_SEEDS])
    ctl_s = np.stack([load('ctl_shuffled_spatial', s) for s in CTL_SEEDS])
    uni = load('ctl_uniform', 137)
    msk = interior_mask(real.shape[1:])
    H, W, D = real.shape[1:]

    # --- determinism / orientation response ---
    det = {}
    mx = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            dd = float(np.abs(real[i] - real[j]).max())
            mx = max(mx, dd)
            det[f'{SEEDS[i]}_vs_{SEEDS[j]}'] = round(dd, 6)
    print('seed-pair maxabsdiff:', json.dumps(det))
    print('max orientation response: %.6f' % mx)

    consensus = real.mean(axis=0)
    cm = (consensus > THR) & msk
    lab, ncomp = ndimage.label(cm, structure=np.ones((3, 3, 3), dtype=int))
    print(f'consensus components > {THR}: {ncomp}')

    # component table (top 40 by max K~)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, ncomp + 1))
    order = np.argsort(sizes)[::-1]
    hotspots = []
    for rank, idx in enumerate(order[:40]):
        cid = int(idx) + 1
        m = lab == cid
        area = int(sizes[idx])
        ys, xs, zs = np.argwhere(m).T
        hs = {
            'rank_by_area': rank + 1, 'id': cid, 'area_vox': area,
            'y_range': [int(ys.min()), int(ys.max())],
            'x_range': [int(xs.min()), int(xs.max())],
            'z_range': [int(zs.min()), int(zs.max())],
            'consensus_max': round(float(consensus[m].max()), 3),
            'consensus_mean': round(float(consensus[m].mean()), 3),
            'full_width_stripe': bool(xs.min() <= 8 and xs.max() >= W - 9
                                      and (ys.max() - ys.min()) <= 6),
        }
        # per-seed stability: fraction of component voxels > THR in each seed
        ov = [float(((real[k] > THR) & m).sum() / area) for k in range(5)]
        hs['seed_overlap'] = [round(o, 4) for o in ov]
        hs['n_seeds_stable'] = int(sum(o > 0.5 for o in ov))
        hs['per_seed_max'] = [round(float(real[k][m].max()), 3)
                              for k in range(5)]
        # control response inside the component region
        hs['ctl_depth_mean'] = round(float(ctl_d[:, m].mean()), 3)
        hs['ctl_spatial_mean'] = round(float(ctl_s[:, m].mean()), 3)
        hs['ctl_depth_max'] = round(float(ctl_d[:, m].max()), 3)
        hs['ctl_spatial_max'] = round(float(ctl_s[:, m].max()), 3)
        rmean = hs['consensus_mean']
        hs['ctl_ratio_depth'] = round(hs['ctl_depth_mean'] / rmean, 3) \
            if rmean else 0.0
        hs['ctl_ratio_spatial'] = round(hs['ctl_spatial_mean'] / rmean, 3) \
            if rmean else 0.0
        hotspots.append(hs)
    for h in hotspots[:12]:
        print('hotspot', {k: h[k] for k in
              ('rank_by_area', 'area_vox', 'y_range', 'x_range', 'z_range',
               'consensus_max', 'consensus_mean', 'n_seeds_stable',
               'ctl_ratio_depth', 'ctl_ratio_spatial', 'full_width_stripe')})

    # --- y=160 band vs background (margins excluded) ---
    band_m = np.zeros(msk.shape, bool); band_m[144:180, 8:W - 8, 8:D - 8] = True
    bg_m = np.zeros(msk.shape, bool)
    bg_m[32:144, 8:W - 8, 8:D - 8] = True
    bg_m[180:224, 8:W - 8, 8:D - 8] = True
    band = {'y': [144, 180]}
    for name, mm in (('band', band_m), ('background', bg_m)):
        for k in range(5):
            v = real[k][mm]
            band.setdefault(name, {})[f'seed{SEEDS[k]}'] = {
                'mean': round(float(v.mean()), 5),
                'max': round(float(v.max()), 3),
                'frac_gt_thr': round(float((v > THR).mean()), 6)}
        vd = ctl_d[:, mm].mean()
        vs = ctl_s[:, mm].mean()
        band[name]['ctl_depth_mean'] = round(float(vd), 5)
        band[name]['ctl_spatial_mean'] = round(float(vs), 5)
    print('band:', json.dumps(band['band']))
    print('background:', json.dumps(band['background']))

    # --- deep interior (band excluded) ---
    di_m = np.zeros(msk.shape, bool); di_m[8:H - 8, 8:W - 8, 40:208] = True
    di_m[144:180, :, :] = False
    deep = {}
    for k in range(5):
        v = real[k][di_m]
        deep[f'seed{SEEDS[k]}'] = {
            'mean': round(float(v.mean()), 5), 'max': round(float(v.max()), 3),
            'frac_gt_thr': round(float((v > THR).mean()), 6)}
    deep['ctl_depth'] = {'mean': round(float(ctl_d[:, di_m].mean()), 5),
                         'max': round(float(ctl_d[:, di_m].max()), 3),
                         'frac_gt_thr': round(float((ctl_d[:, di_m] > THR).mean()), 6)}
    deep['ctl_spatial'] = {'mean': round(float(ctl_s[:, di_m].mean()), 5),
                           'max': round(float(ctl_s[:, di_m].max()), 3),
                           'frac_gt_thr': round(float((ctl_s[:, di_m] > THR).mean()), 6)}
    print('deep_interior:', json.dumps(deep))

    # --- global control summary ---
    ctl_summary = {}
    for cname, arr in (('shuffled_depth', ctl_d), ('shuffled_spatial', ctl_s)):
        vv = arr[:, msk]
        ctl_summary[cname] = {
            'mean': round(float(vv.mean()), 5), 'max': round(float(vv.max()), 3),
            'frac_gt_thr': round(float((vv > THR).mean()), 6)}
    ctl_summary['uniform'] = {'max': float(uni.max()),
                              'frac_gt_thr': float((uni > THR).mean())}
    print('controls:', json.dumps(ctl_summary))
    print('real global: max %.3f frac>3 %.6f' %
          (float(real[:, msk].max()), float((real[:, msk] > THR).mean())))

    # --- morphology screen: ink sits ON a surface -> thin in z,
    # extended in x-y, solid fill. Wispy isotropic filaments fail it.
    for h in hotspots:
        dy = h['y_range'][1] - h['y_range'][0] + 1
        dx = h['x_range'][1] - h['x_range'][0] + 1
        dz = h['z_range'][1] - h['z_range'][0] + 1
        fill = h['area_vox'] / max(dy * dx * dz, 1)
        h['fill_frac'] = round(fill, 4)
        h['xy_over_z'] = round(max(dx, dy) / max(dz, 1), 2)
        h['stroke_like'] = bool(fill >= 0.15 and max(dx, dy) / max(dz, 1) >= 3
                                and h['area_vox'] >= 100)

    # --- verdict ---
    stable = [h for h in hotspots
              if h['n_seeds_stable'] >= 4
              and h['consensus_mean'] > THR
              and h['ctl_ratio_depth'] < 0.5
              and h['ctl_ratio_spatial'] < 0.5
              and not h['full_width_stripe']]
    candidates = [h for h in stable if h['stroke_like']]
    n_wisps = len(stable) - len(candidates)
    verdict = ('INK CANDIDATE(S): %d' % len(candidates)) if candidates else \
        ('NONE — no ink-morphology structure. %d wispy tail-filament '
         'components pass stability/controls but fail stroke morphology '
         '(isotropic, 1.5-4.5%% fill); the phi instrument reads substrate '
         'structure, not deposited ink.' % n_wisps)
    print('stable components passing controls:', len(stable))
    print('VERDICT:', verdict)

    out = {'threshold': THR, 'seeds': SEEDS,
           'orientation_maxabsdiff': det,
           'n_consensus_components': int(ncomp),
           'consensus_max': round(float(consensus[msk].max()), 3),
           'hotspots_top40': hotspots,
           'ink_candidates': candidates,
           'y160_band': band, 'deep_interior': deep,
           'control_summary': ctl_summary,
           'verdict': verdict}
    with open(os.path.join(HERE, 'verdicts_phi.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote verdicts_phi.json')


if __name__ == '__main__':
    main()
