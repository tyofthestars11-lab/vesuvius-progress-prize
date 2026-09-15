#!/usr/bin/env python3
"""V_RECOVERY: F_C_new with REAL V + full character pipeline.

F_C_new = V * PHI**(s/(pi*PHI2)) * Psi_Omega * P3 * exp(i*3*PHI*th_g)
Component definitions are the ACTUAL ones from phi2_decoder/phi2_step1_rungs.py
(not invented forms):
  PHI=(1+sqrt5)/2, PHI2=PHI+1, GA=2*pi/PHI2, TA=acos(-1/3), P3=11/27
  th_g=s*GA, th_t=s*TA, th_r=s*2*pi
  Psi_Omega=(exp(i*PHI*th_g)+exp(i*PHI*th_t)+exp(i*PHI*th_r))/sqrt(3)
  amp=PHI**(s/(pi*PHI2)), rung=ln|F_C|/ln(PHI), phase=frac(rung)*360

s: absolute volume indices. The original decode-pass used level-5 indices;
here sample positions are level-0 voxels, so s=(x+y+z)/32 (level-5-equivalent).
Documented, not invented: keeps amp=PHI**(s/(pi*PHI2)) in float64 range
(level-0 s~37000 would overflow; level-5-equivalent s~1160 gives amp~1e29).

Signal for the character test: S = |F_C_new|/amp = V*|Psi|*P3 (amp detrended by
exact algebra — the exponential amp gradient would otherwise dominate thresholds).
Secondary: phase_new scatter at fixed s (does real V break s-determinism of phase?).

Pipeline: thresholds -> 26-connectivity 3D components -> PCA->2D 64x64
-> cosine vs 24 DejaVu Sans Greek templates -> shuffle stabilizer (20, seed 42)
-> published column-15 Greek comparison -> passage.
"""
import os, json, struct, math
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.expanduser('~/workspace/pherc1667/v_recovery')
BASE4 = os.path.expanduser('~/workspace/pherc1667/path4_ink')
PHI = (1 + 5 ** 0.5) / 2
PHI2 = PHI + 1
GA = 2 * math.pi / PHI2
TA = math.acos(-1 / 3)
P3 = 11 / 27
LN_PHI = math.log(PHI)
TW, TH = 30097, 2061
WS = 384
NLAY = 24

GREEK = list('αβγδεζηθικλμνξοπρστυφχψω')
NAMES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
         'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
         'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']

def render_letter(ch, size=64):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 52)
    img = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(img)
    bb = d.textbbox((0, 0), ch, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), ch, font=font, fill=255)
    return (np.array(img) > 128).astype(np.float64)

def tif_window(f, u0, v0, w, h):
    with open(f, 'rb') as fh:
        head = fh.read(8)
        bo = '<' if head[:2] == b'II' else '>'
        off = struct.unpack(bo + 'I', head[4:8])[0]
        fh.seek(off); n = struct.unpack(bo + 'H', fh.read(2))[0]
        strip = None
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(bo + 'HHI4s', fh.read(12))
            if tag == 273: strip = struct.unpack(bo + 'I', val)[0]
        out = np.empty((h, w), dtype=np.float32)
        for r in range(h):
            fh.seek(strip + ((v0 + r) * TW + u0) * 4)
            out[r] = np.frombuffer(fh.read(w * 4), dtype='<f4')
    return out

def project_component(coords):
    c = coords - coords.mean(axis=0)
    cov = np.cov(c.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    p2 = c @ vecs[:, order[:2]]
    lo = p2.min(axis=0); hi = p2.max(axis=0)
    span = (hi - lo).max()
    if span < 1e-9: return np.zeros((64, 64))
    scale = 60.0 / span
    q = (p2 - lo) * scale + 2.0
    img = np.zeros((64, 64))
    qi = np.clip(np.rint(q).astype(int), 0, 63)
    img[qi[:, 1], qi[:, 0]] = 1.0
    return img

def main():
    meta = json.load(open(os.path.join(BASE, 'vstack_384_meta.json')))
    WU0, WV0 = meta['window_tifxyz'][:2]
    stack = np.load(os.path.join(BASE, 'vstack_384.npy')).astype(np.float64)
    normals = np.load(os.path.join(BASE, 'vstack_384_normals.npy')).astype(np.float64)
    print('stack', stack.shape, flush=True)

    # sample positions (x,y,z) per voxel: surface + offs*N
    print('recomputing sample positions...', flush=True)
    Px = tif_window(os.path.join(BASE4, 'tifxyz_2399_x.tif'), WU0, WV0, WS, WS).astype(np.float64)
    Py = tif_window(os.path.join(BASE4, 'tifxyz_2399_y.tif'), WU0, WV0, WS, WS).astype(np.float64)
    Pz = tif_window(os.path.join(BASE4, 'tifxyz_2399_z.tif'), WU0, WV0, WS, WS).astype(np.float64)
    offs = np.arange(NLAY, dtype=np.float64) - (NLAY - 1) / 2.0
    Sx = Px[:, :, None] + offs[None, None, :] * normals[:, :, 0:1]
    Sy = Py[:, :, None] + offs[None, None, :] * normals[:, :, 1:2]
    Sz = Pz[:, :, None] + offs[None, None, :] * normals[:, :, 2:3]
    del Px, Py, Pz
    s = (Sx + Sy + Sz) / 32.0  # level-5-equivalent absolute indices
    del Sx, Sy, Sz
    print('s range: [%.1f, %.1f]' % (s.min(), s.max()), flush=True)

    # F_C_new components (actual definitions from phi2_decoder/)
    th_g = s * GA; th_t = s * TA; th_r = s * 2 * math.pi
    psi = (np.exp(1j * PHI * th_g) + np.exp(1j * PHI * th_t) + np.exp(1j * PHI * th_r)) / math.sqrt(3)
    mpsi = np.abs(psi)
    amp = np.power(PHI, s / (math.pi * PHI2))
    V = stack  # real volume intensity
    Fc_abs = V * amp * mpsi * P3  # |F_C_new|; |exp(i*3*phi*th_g)| = 1
    rung_new = np.log(Fc_abs + 1e-300) / LN_PHI
    phase_new = ((rung_new - np.floor(rung_new)) * 360.0).astype(np.float32)
    np.save(os.path.join(BASE, 'Fc_rung_new.npy'), rung_new.astype(np.float32))
    np.save(os.path.join(BASE, 'Fc_phase_new.npy'), phase_new)
    print('rung_new range: [%.2f, %.2f]' % (rung_new.min(), rung_new.max()), flush=True)
    json.dump({'s_min': float(s.min()), 's_max': float(s.max()),
               'rung_new_min': float(rung_new.min()), 'rung_new_max': float(rung_new.max()),
               'V_mean': float(V.mean()), 'V_max': float(V.max()),
               'amp_min': float(amp.min()), 'amp_max': float(amp.max())},
              open(os.path.join(BASE, 'Fc_new_meta.json'), 'w'), indent=1)

    # CRUX: is phase_new still a function of s? (with V:=1 it was exact)
    si = np.floor(s).astype(np.int64)
    uniq = np.unique(si)
    viol = 0; maxrange = 0.0; n_multi = 0
    for sv in uniq:
        ph = phase_new[si == sv]
        if ph.size > 1:
            n_multi += 1
            r = float(ph.max() - ph.min())
            # circular range
            maxrange = max(maxrange, r)
            if r > 1e-6: viol += 1
    print('CRUX phase_new vs s: %d s-levels, %d multi-voxel, %d with phase range>1e-6 deg, max range %.3f deg'
          % (len(uniq), n_multi, viol, maxrange), flush=True)
    json.dump({'n_s_levels': int(len(uniq)), 'n_multi': int(n_multi),
               'violating': int(viol), 'max_phase_range_deg': float(maxrange)},
              open(os.path.join(BASE, 'Fc_phase_crux.json'), 'w'), indent=1)

    # character pipeline on S = |F_C_new| / (amp*|Psi|*P3) — the literal amplitude spec.
    # Algebraically S = V (the payload lane); computed via the formula, not the shortcut.
    S = (Fc_abs / (amp * mpsi * P3)).astype(np.float32)
    print('S: mean %.2f (V mean %.2f), max %.1f' % (S.mean(), V.mean(), S.max()), flush=True)
    del V, psi, mpsi, amp, Fc_abs, rung_new
    tvecs = np.stack([render_letter(ch).ravel() for ch in GREEK])
    tnorm = np.linalg.norm(tvecs, axis=1, keepdims=True) + 1e-12
    struct26 = np.ones((3, 3, 3))

    def score_labels(labels, ncomp):
        out = []
        # PERF FIX (2026-09-15): the original loop did np.argwhere(labels == cid)
        # for EVERY cid — O(ncomp*N) full-array scans, ~12 min per shuffled trial
        # (46K components). np.unique sizes components in ONE pass; argwhere runs
        # only for components with >=10 voxels. Identical output: same cid set,
        # same ascending order, same per-component math.
        _vals, _counts = np.unique(labels, return_counts=True)
        _big = _vals[(_vals > 0) & (_counts >= 10)]
        for cid in _big:
            coords = np.argwhere(labels == cid)
            proj = project_component(coords)
            v = proj.ravel(); vn = np.linalg.norm(v) + 1e-12
            scores = (tvecs @ v) / (tnorm[:, 0] * vn)
            order = np.argsort(scores)[::-1]
            out.append({'id': int(cid), 'size': int(len(coords)),
                        'cu': float(coords[:, 1].mean()), 'cv': float(coords[:, 0].mean()),
                        'top1': NAMES[int(order[0])], 's1': float(scores[order[0]]),
                        'top2': NAMES[int(order[1])], 's2': float(scores[order[1]]),
                        'top3': NAMES[int(order[2])], 's3': float(scores[order[2]])})
        return out

    pos = S[S > 0]
    thresholds = [float(np.percentile(pos, q)) for q in (90, 95, 98)]
    allres = {}
    for thr in thresholds:
        lab, n = ndimage.label(S >= thr, structure=struct26)
        res = score_labels(lab, n)
        # stabilizer: 20 shuffled trials, seed 42
        rng = np.random.default_rng(42)
        ceil = 0
        flat = S.ravel().copy()
        for t in range(20):
            rng.shuffle(flat)
            lab_s, n_s = ndimage.label((flat.reshape(S.shape) >= thr), structure=struct26)
            r_s = score_labels(lab_s, n_s)
            ceil = max(ceil, sum(1 for r in r_s if r['s1'] >= 0.60))
        leg = [r for r in res if r['s1'] >= 0.60]
        allres[str(round(thr, 3))] = {'threshold': thr, 'n_components': int(n),
                                     'candidates_ge10': len(res), 'legible': len(leg),
                                     'null_ceiling': int(ceil),
                                     'best': max([r['s1'] for r in res], default=0.0),
                                     'cands': res}
        print('thr %.3f: comps %d, >=10 %d, legible %d, null_ceiling %d, best %.4f' % (
            thr, n, len(res), len(leg), ceil, allres[str(round(thr, 3))]['best']), flush=True)
    json.dump(allres, open(os.path.join(BASE, 'Fc_chars.json'), 'w'))

    # published column-15 Greek comparison + passage
    pub = open('/home/hatch/workspace/pherc1667/hit_columns_greek.txt').read()
    col15 = pub.split('COL 15')[1].split('COL 16')[0]
    import re
    pub_letters = set(re.findall(r'[αβγδεζηθικλμνξοπρστυφχψω]', col15))
    best_thr = max(allres, key=lambda k: allres[k]['legible'] if allres[k]['legible'] else -1)
    # pick threshold with most legible; tie -> lowest threshold
    ranked = sorted(allres, key=lambda k: (-allres[k]['legible'], allres[k]['threshold']))
    best_thr = ranked[0]
    leg = [r for r in allres[best_thr]['cands'] if r['s1'] >= 0.60]
    leg.sort(key=lambda r: (r['cv'] // 40, r['cu']))
    greek_of = dict(zip(NAMES, GREEK))
    passage = ''.join(greek_of[r['top1']] for r in leg)
    in_pub = sum(1 for r in leg if greek_of[r['top1']] in pub_letters)
    print('published COL15 distinct letters: %d' % len(pub_letters), flush=True)
    print('legible letters also in published COL15: %d/%d' % (in_pub, len(leg)), flush=True)
    print('passage (thr %s): %s' % (best_thr, passage if passage else '(none)'), flush=True)
    json.dump({'passage': passage, 'threshold': best_thr, 'n_legible': len(leg),
               'n_in_published': int(in_pub),
               'published_col15_letters': sorted(pub_letters)},
              open(os.path.join(BASE, 'Fc_passage.json'), 'w'), indent=1)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
