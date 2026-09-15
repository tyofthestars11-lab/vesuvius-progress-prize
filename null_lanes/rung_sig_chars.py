#!/usr/bin/env python3
"""RUNG-SIGNATURE LANE — second read on S, alongside (not replacing) the voxel lane.

TYREE's phi-structural solution: rungs, not values. S carries V, V carries the
ink, the ink's structure is the rung-signature.

FRAME CORRECTION (2026-09-14, folded as fuel): the old 44-rung [10,53] frame was
the proportional placement's LOCAL block. True frame from TIFXYZ: rungs span
[57,198] over the patch, monotonic in v. Relative structure (order/gaps/triplets)
is frame-invariant. This lane now bins on the TRUE frame, same geometry recipe as
fc_new_chars.py (tifxyz window + normals -> per-voxel s).

Pipeline:
  1. S per voxel (same F_C_new definitions as fc_new_chars.py; P3 = 11/27 exact).
     S = |F_C|/(amp*|Psi|*P3) = V, computed via the formula.
  2. S_rung(r) = sum of S over voxels at integer rung r (true frame).
  3. Rung-cluster signature: consecutive rungs above local median; Fibonacci-
     spaced rung pairs; golden-angle-ratio rung triplets; minima; envelope.
  4. CORRELATION GATE (TYREE's deciding number): corr(S_rung, P(r)) where P(r)
     is the phi-direct predictor from the published column-15 Greek's own rung
     addresses (phi_direct_P.npy, no V involved). Pearson + Spearman + MI, raw.
  5. Greek rung-signature table from published COL 15: occurrence -> line ->
     v-band -> observed (S-weighted) rung address; cluster matching (cosine).
  6. Stabilizer (20 shuffles of S across voxels, seed 42); passage in v-order;
     published comparison; cross-lane vs voxel lane.
  7. Branch: significant positive correlation -> segment high-correlation rungs,
     match to alphabet signatures, name letters. Null -> name the amplitude
     boundary exactly (lane, step, what blocked).
"""
import os, json, math, re, struct, unicodedata
import numpy as np
from collections import defaultdict, Counter

BASE = os.path.expanduser('~/workspace/pherc1667/v_recovery')
BASEH = os.path.expanduser('~/workspace/pherc1667')
BASE4 = os.path.expanduser('~/workspace/pherc1667/path4_ink')
LOG = open(os.path.join(BASE, 'rung_sig_chars.log'), 'w')
def log(m):
    LOG.write(m + '\n'); LOG.flush(); print(m, flush=True)

PHI = (1 + math.sqrt(5)) / 2
PHI2 = PHI + 1
LPHI = math.log(PHI)
GA = 2 * math.pi / PHI2
TA = math.acos(-1 / 3)
P3 = 11 / 27
WS, NLAY = 384, 24

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
        out = np.empty((h, w), dtype=np.float64)
        TW = 30097
        for r in range(h):
            fh.seek(strip + ((v0 + r) * TW + u0) * 4)
            out[r] = np.frombuffer(fh.read(w * 4), dtype='<f4')
    return out

# ---------- 1. load V stack + true-frame geometry ----------
meta = json.load(open(os.path.join(BASE, 'vstack_384_meta.json')))
WU0, WV0 = meta['window_tifxyz'][:2]
stack = np.load(os.path.join(BASE, 'vstack_384.npy')).astype(np.float64)  # (WS,WS,NLAY) [u,v,layer]
normals = np.load(os.path.join(BASE, 'vstack_384_normals.npy')).astype(np.float64)
log('S source: %s/vstack_384.npy shape %s window u[%d,%d] v[%d,%d]' % (BASE, stack.shape, WU0, WU0+WS, WV0, WV0+WS))
log('recomputing true-frame sample positions...')
Px = tif_window(os.path.join(BASE4, 'tifxyz_2399_x.tif'), WU0, WV0, WS, WS)
Py = tif_window(os.path.join(BASE4, 'tifxyz_2399_y.tif'), WU0, WV0, WS, WS)
Pz = tif_window(os.path.join(BASE4, 'tifxyz_2399_z.tif'), WU0, WV0, WS, WS)
offs = np.arange(NLAY, dtype=np.float64) - (NLAY - 1) / 2.0
Sx = Px[:, :, None] + offs[None, None, :] * normals[:, :, 0:1]
Sy = Py[:, :, None] + offs[None, None, :] * normals[:, :, 1:2]
Sz = Pz[:, :, None] + offs[None, None, :] * normals[:, :, 2:3]
del Px, Py, Pz, normals
s = (Sx + Sy + Sz) / 32.0
del Sx, Sy, Sz
log('s range: [%.1f, %.1f]' % (s.min(), s.max()))

# ---------- 2. F_C_new, rung (true frame), S = V via formula ----------
th_g, th_t, th_r = s * GA, s * TA, s * 2 * math.pi
psi = (np.exp(1j * PHI * th_g) + np.exp(1j * PHI * th_t) + np.exp(1j * PHI * th_r)) / math.sqrt(3)
mpsi = np.abs(psi); del psi, th_g, th_t, th_r
amp = np.power(PHI, s / (math.pi * PHI2)); del s
V = stack; del stack
Fc_abs = V * amp * mpsi * P3
# TYREE LOCK (phi-authority): the address is the phi-components' own log-address,
# V-independent. rung = ln(amp*|Psi|*P3)/ln(phi) = s/(pi*phi^2) + ln(|Psi|*P3)/ln(phi)
# (phi_direct_rungs.py:73). V never moves the address; V is the payload.
rung = np.log(np.maximum(amp * mpsi * P3, 1e-300)) / LPHI
S = (Fc_abs / (amp * mpsi * P3)).astype(np.float64)
sdev = float(np.max(np.abs(S - V.astype(np.float64))))
log('S=V identity check: max|S-V| = %.3e (S carries V, lock holds)' % sdev)
del Fc_abs, amp, mpsi, V
log('S: mean %.2f max %.1f | rung range [%.2f, %.2f]' % (S.mean(), S.max(), rung.min(), rung.max()))

# ---------- 3. rung lane on TRUE frame ----------
RLO, RHI = int(math.floor(rung.min())), int(math.ceil(rung.max()))
NB = RHI - RLO + 1
ri = np.floor(rung).astype(np.int64)
valid = (ri >= RLO) & (ri <= RHI) & np.isfinite(S)
S_m = S[valid].ravel(); ri_m = (ri[valid] - RLO).ravel()
S_rung = np.bincount(ri_m, weights=S_m, minlength=NB).astype(np.float64)
cnt_rung = np.bincount(ri_m, minlength=NB).astype(np.float64)
del ri, valid
log('true-frame rung grid: [%d, %d] (%d bins), voxels in lane %d' % (RLO, RHI, NB, int(cnt_rung.sum())))
log('S_rung total %.1f | global mean S %.3f' % (S_rung.sum(), S_m.mean()))
mean_all = S_m.mean()
conc = (S_rung / np.maximum(cnt_rung, 1)) / mean_all
loc_med = np.array([np.median(S_rung[max(0, r-3):r+4]) for r in range(NB)])
peaks = [int(r + RLO) for r in range(NB) if S_rung[r] > 1.5 * loc_med[r]]
gmed = float(np.median(S_rung))
clusters, cur = [], []
for r in range(NB):
    if S_rung[r] > gmed: cur.append(int(r + RLO))
    elif cur: clusters.append(cur); cur = []
if cur: clusters.append(cur)
log('peaks (>1.5x local median): %s' % peaks)
log('rung clusters (>global median %.1f): %s' % (gmed, clusters))
log('concentration C_r top-8: %s' % sorted(
    [(int(r + RLO), round(float(conc[r]), 3)) for r in range(NB)], key=lambda t: -t[1])[:8])
# ---------- FIBONACCI RUNG DECODE (TYREE: find S using Fibonacci rungs — one decode, no separation) ----------
# S is read AT the Fibonacci rungs of the true frame itself. The frame's own
# Fibonacci numbers are 55, 89, 144, 233 (the small-scale set never lands in-frame).
# One reading: mass, occupancy-normalized concentration, permutation null, frame-scale spacings.
FIB_R = [f for f in (55, 89, 144, 233) if RLO <= f <= RHI]
log('FIBONACCI DECODE — frame Fibonacci rungs in [%d,%d]: %s' % (RLO, RHI, FIB_R))
fib_mass = 0.0; fib_vox = 0
for fr in FIB_R:
    r = fr - RLO
    fib_mass += S_rung[r]; fib_vox += int(cnt_rung[r])
    log('  Fib rung %d: S_rung=%.1f (%.2fx local med), conc=%.3f, peak=%s, voxels=%d' % (
        fr, S_rung[r], S_rung[r] / max(loc_med[r], 1e-9), conc[r], fr in peaks, int(cnt_rung[r])))
tot_mass = float(S_rung.sum()); tot_vox = float(cnt_rung.sum())
fib_conc = (fib_mass / max(fib_vox, 1)) / mean_all
log('  Fib-rung S mass %.1f / total %.1f = %.4f (uniform expect %.4f); Fib concentration vs global mean: %.3f' % (
    fib_mass, tot_mass, fib_mass / max(tot_mass, 1e-9), len(FIB_R) / NB, fib_conc))
frng = np.random.default_rng(42)
null_c = np.empty(200)
for t in range(200):
    prm = frng.permutation(ri_m)
    b = np.bincount(prm, weights=S_m, minlength=NB).astype(np.float64)
    c = np.bincount(prm, minlength=NB).astype(np.float64)
    fm = sum(b[f - RLO] for f in FIB_R); fv = sum(c[f - RLO] for f in FIB_R)
    null_c[t] = (fm / max(fv, 1)) / mean_all
p_fib = (1 + np.sum(null_c >= fib_conc)) / 201
log('  Fib-concentration null (200 rung-shuffles): mean %.3f std %.3f | observed %.3f | p = %.4f' % (
    float(null_c.mean()), float(null_c.std()), fib_conc, p_fib))
FIB_GAPS = {55, 89, 144}
fib_pairs = [(a, b, b - a) for i, a in enumerate(peaks) for b in peaks[i+1:] if b - a in FIB_GAPS]
log('  Fibonacci-spaced peak pairs (55/89/144): %s' % (fib_pairs if fib_pairs else 'none'))
fib_verdict = 'S CONCENTRATES AT FIBONACCI RUNGS (p=%.4f)' % p_fib if p_fib < 0.05 else \
              'S does not concentrate at Fibonacci rungs (p=%.4f)' % p_fib
log('  FIBONACCI VERDICT: %s' % fib_verdict)
# ---------- SOURCE RUNG 15 (TYREE's rung: TON618 horizon 1,300 AU ~= phi^15) ----------
# All addresses measured from the source. The pull already radiates source-outward
# (rung 57 first = nearest the source); the decode now reads source-relative too.
SRC_RUNG = 15
log('  SOURCE RUNG %d: phi^15 = %.4f (1,300 AU horizon, dev %.2f%%)' % (
    SRC_RUNG, PHI ** SRC_RUNG, abs(PHI ** SRC_RUNG - 1300) / 1300 * 100))
for fr in FIB_R:
    sr = fr - SRC_RUNG
    mark = ' = 2x37 CLOCK' if sr == 74 else ''
    log('  Fib rung %d -> source-relative %d%s' % (fr, sr, mark))
log('  source-relative frame: c 40.56->%.2f | TON618 51.771->%.3f | patch [%d,%d]->[%d,%d]' % (
    40.5613 - SRC_RUNG, 51.7712 - SRC_RUNG, RLO, RHI, RLO - SRC_RUNG, RHI - SRC_RUNG))
results_src = {'src_rung': SRC_RUNG, 'fib_src': [f - SRC_RUNG for f in FIB_R]}
# ---------- 30-TRILLION-SUNS POWER ANCHOR (TON618 macro fuel in the decode) ----------
# TON618 law: L = phi^63 = 1.466e13 (~30 trillion suns / 2.046). Sealed ladder rung 51.771.
# The frame's Fibonacci rungs ride at phi and phi^2 above the anchor:
#   51.771*phi = 83.77  (5.88% below Fib 89); 51.771*phi^2 = 135.54 (5.88% below Fib 144).
POW63 = PHI ** 63
log('  POWER: phi^63 = %.6e suns; 30 trillion / phi^63 = %.3f' % (POW63, 3.0e13 / POW63))
log('  ANCHOR: TON618 rung 51.771 * phi = %.2f (-> Fib 89, d=5.88%%); * phi^2 = %.2f (-> Fib 144, d=5.88%%)' % (
    51.771 * PHI, 51.771 * PHI2))
log('  S mass at Fibonacci rungs in suns of the power scale: %.6e (fib_mass / phi^63)' % (
    fib_mass / POW63))
# ---------- FIBONACCI-RUNGED CLOCK: time itself, declassified ----------
# rung(t) is monotonic 67/67 over the 68 text lines: time flows down-rung, time IS the rung order.
# One time-step sits on a Fibonacci rung: t=37 @ rung ~144. Read S in that time-band at runtime.
t37_v0 = 561.0 + (37 + 0.5) / 68 * (1723 - 561)
t37_j = t37_v0 - WV0
if 0 <= t37_j < WS:
    jlo37, jhi37 = max(0, int(t37_j) - 12), min(WS, int(t37_j) + 12)
    S37 = float(S[:, jlo37:jhi37, :].sum()); Sall = float(S.sum())
    log('  CLOCK t=37@Fib144: band v-rows [%d,%d], S mass %.1f / total %.1f = %.4f (uniform %.4f)' % (
        jlo37, jhi37, S37, Sall, S37 / max(Sall, 1e-9), (jhi37 - jlo37) / WS))
    results_clock = {'t37_band': [jlo37, jhi37], 't37_frac': S37 / max(Sall, 1e-9)}
else:
    log('  CLOCK t=37@Fib144: line-37 band outside 384 window (v0=%.1f)' % t37_v0)
    results_clock = {'t37_band': None, 't37_frac': None}
trips = []
for i in range(len(peaks)):
    for j in range(i + 1, len(peaks)):
        for k in range(j + 1, len(peaks)):
            a, b, c = peaks[i], peaks[j], peaks[k]
            q = (b - a) / (c - a)
            if abs(q - 1 / PHI2) < 0.03: trips.append((a, b, c, round(q, 3)))
log('golden-angle triplets among peaks: %s' % (trips if trips else 'none'))
mins = [int(r + RLO) for r in range(NB) if S_rung[r] < 0.5 * loc_med[r]]
log('minima (<0.5x local median): %s' % mins)
cc = float(np.corrcoef(S_rung, cnt_rung)[0, 1]) if NB > 2 else 0.0
log('corr(S_rung, voxel_count_r) = %.4f  (1.0 => lane is pure count)' % cc)

# ---------- 4. CORRELATION GATE vs phi-direct predictor P(r) ----------
log('--- 4. CORRELATION GATE: S_rung vs phi-direct P(r) ---')
Pinfo = json.load(open(os.path.join(BASE, 'phi_direct_P.json')))
P = np.load(os.path.join(BASE, 'phi_direct_P.npy')).astype(np.float64)
PLO = Pinfo['lo']
o0 = max(RLO, PLO); o1 = min(RHI, PLO + len(P) - 1)
iS = np.arange(o0 - RLO, o1 - RLO + 1)
iP = np.arange(o0 - PLO, o1 - PLO + 1)
x = S_rung[iS]; y = P[iP]
log('overlap rungs [%d,%d] (%d bins); P occ in overlap %d' % (o0, o1, len(iS), int(y.sum())))
from scipy import stats
pear, pear_p = stats.pearsonr(x, y)
spea, spea_p = stats.spearmanr(x, y)
log('Pearson  r = %.4f  p = %.3g' % (pear, pear_p))
log('Spearman rho = %.4f  p = %.3g' % (spea, spea_p))
def mi_hist(a, b, bins=10):
    ae = np.histogram_bin_edges(a, bins=bins); be = np.histogram_bin_edges(b, bins=bins)
    j, _, _ = np.histogram2d(a, b, bins=[ae, be])
    j = j / j.sum()
    pa, pb = j.sum(1), j.sum(0)
    m = 0.0
    for i_ in range(j.shape[0]):
        for j_ in range(j.shape[1]):
            if j[i_, j_] > 0 and pa[i_] > 0 and pb[j_] > 0:
                m += j[i_, j_] * math.log(j[i_, j_] / (pa[i_] * pb[j_]))
    return m
mi_obs = mi_hist(x, y)
rng = np.random.default_rng(42)
mi_null = [mi_hist(x, rng.permutation(y)) for _ in range(200)]
mi_p = (1 + sum(m >= mi_obs for m in mi_null)) / 201
log('MI = %.4f nat; null(200 perm) mean %.4f, p = %.4f' % (mi_obs, float(np.mean(mi_null)), mi_p))
GATE = (pear > 0 and pear_p < 0.05) or (spea > 0 and spea_p < 0.05)
log('GATE: %s' % ('OPEN — amplitude follows the published Greek' if GATE else 'CLOSED'))
results = {'rung_grid': [RLO, RHI], 'S_rung': [float(v) for v in S_rung],
           'peaks': peaks, 'clusters': clusters, 'fib_pairs': fib_pairs, 'trips': trips,
           'corr_pearson': [float(pear), float(pear_p)],
           'corr_spearman': [float(spea), float(spea_p)],
           'corr_MI': [float(mi_obs), float(mi_p)], 'gate_open': bool(GATE),
           'fib_rungs': FIB_R, 'fib_conc': float(fib_conc), 'fib_p': float(p_fib),
           'fib_verdict': fib_verdict, 'fib_clock': results_clock, 'src_rung': results_src}

# ---------- 5. Greek rung-signature table from published COL 15 ----------
GREEK_BASE = 'αβγδεζηθικλμνξοπρστυφχψω'
def base_letter(ch):
    n = unicodedata.normalize('NFD', ch)
    b = ''.join(c for c in n if not unicodedata.combining(c)).lower()
    return b if b in GREEK_BASE else None
txt = open(os.path.join(BASEH, 'hit_columns_greek.txt'), encoding='utf-8').read()
m = re.search(r'COL 15.*?\n=+\n(.*)', txt, re.S)
block = m.group(1)
lines = [ln for ln in block.strip().split('\n') if ln.strip()]
occ = []
for li, ln in enumerate(lines):
    for ch in ln:
        b = base_letter(ch)
        if b: occ.append((li, b))
L = len(lines)
log('COL15: %d text lines, %d greek letter occurrences' % (L, len(occ)))
log('published letter freq top: %s' % Counter(b for _, b in occ).most_common(8))
FV0, FV1 = 561, 1723
def line_v(li): return FV0 + (li + 0.5) / L * (FV1 - FV0)
sig = {b: np.zeros(NB) for b in GREEK_BASE}
n_inwin = 0
band_S_by_rung = {}
rung_i = np.floor(rung).astype(np.int64)
for li, b in occ:
    v0 = line_v(li)
    if not (WV0 <= v0 <= WV0 + WS): continue
    n_inwin += 1
    j0 = int(round(v0 - WV0))
    jlo, jhi = max(0, j0 - 12), min(WS, j0 + 12)
    key = (jlo, jhi)
    if key not in band_S_by_rung:
        Sj = S[:, jlo:jhi, :].ravel()
        rj = rung_i[:, jlo:jhi, :].ravel()
        mm = (rj >= RLO) & (rj <= RHI)
        band_S_by_rung[key] = np.bincount((rj[mm] - RLO).astype(np.int64), weights=Sj[mm], minlength=NB)
    prof = band_S_by_rung[key]
    if prof.sum() > 0:
        sig[b][int(np.argmax(prof))] += 1
del rung_i
log('occurrences inside window v=[%d,%d]: %d' % (WV0, WV0 + WS, n_inwin))
table = {}
for b in GREEK_BASE:
    t = sig[b].sum()
    table[b] = sig[b] / t if t > 0 else sig[b]
    if t > 0:
        top = sorted([(int(r + RLO), int(sig[b][r])) for r in range(NB) if sig[b][r] > 0],
                     key=lambda t2: -t2[1])[:3]
        log('letter %s: n=%d top-rungs %s' % (b, int(t), top))
fib_hits = sum(sig[b][r].sum() for b in GREEK_BASE for r in [f - RLO for f in FIB_R])
tot_hits = sum(sig[b].sum() for b in GREEK_BASE)
log('published-in-window occurrences at Fib rungs %s: %d/%d = %.3f (uniform expect %.3f)' % (
    FIB_R, int(fib_hits), int(tot_hits), fib_hits / max(tot_hits, 1), len(FIB_R) / NB))

# ---------- 6. branch: gate open -> name letters; closed -> name boundary ----------
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
cluster_matches = []
log('--- cluster -> greek matches ---')
for cl in clusters:
    prof = np.zeros(NB); idx = [r - RLO for r in cl]
    prof[idx] = S_rung[idx]
    if prof.sum() > 0: prof /= prof.sum()
    scored = sorted([(b, cos(prof, table[b])) for b in GREEK_BASE], key=lambda t: -t[1])
    cluster_matches.append((cl, scored[:3]))
    log('cluster rungs %s -> %s' % (cl, [(b, round(s, 3)) for b, s in scored[:3]]))
if GATE:
    med_y = float(np.median(y))
    hi_rungs = [int(r + RLO) for r in range(NB)
                if S_rung[r] > gmed and (o0 <= r + RLO <= o1) and P[r + RLO - PLO] > med_y]
    log('high-correlation rungs (S and P both above median): %s' % hi_rungs)
    named = []
    for h in hi_rungs:
        scored = sorted([(b, float(table[b][h - RLO])) for b in GREEK_BASE], key=lambda t: -t[1])
        named.append([h, scored[0][0], round(scored[0][1], 3)])
        log('rung %d -> letter %s (table mass %.3f)' % (h, scored[0][0], scored[0][1]))
    results['named_letters'] = named
else:
    results['amplitude_boundary'] = {
        'lane': 'rung-signature', 'step': 'correlation gate corr(S_rung, P(r))',
        'pearson': [float(pear), float(pear_p)], 'spearman': [float(spea), float(spea_p)],
        'MI': [float(mi_obs), float(mi_p)],
        'what_blocked': 'observed amplitude does not follow the published Greek rung addresses'}
    log('AMPLITUDE BOUNDARY: lane=rung-signature step=correlation-gate ' +
        'pearson=%.4f(p=%.3g) spearman=%.4f(p=%.3g) MI=%.4f(p=%.4f)' % (pear, pear_p, spea, spea_p, mi_obs, mi_p))

# ---------- 7. stabilizer: 20 shuffles of S across voxels, seed 42 ----------
rng = np.random.default_rng(42)
NTR = 20
surv = np.zeros(NB)
for t in range(NTR):
    Sp = rng.permutation(S_m)
    Sr = np.bincount(ri_m, weights=Sp, minlength=NB)
    lm = np.array([np.median(Sr[max(0, r-3):r+4]) for r in range(NB)])
    surv += (Sr > 1.5 * lm).astype(float)
    if t == 0: log('shuffle trial 0 peaks: %s' % [int(r + RLO) for r in range(NB) if Sr[r] > 1.5 * lm[r]])
surv /= NTR
log('--- stabilizer: peak survival over 20 shuffles ---')
for r in range(NB):
    if S_rung[r] > 1.5 * loc_med[r]:
        log('rung %d: real peak, shuffle survival %.2f %s' % (
            int(r + RLO), surv[r], 'SURVIVES' if surv[r] >= 0.95 else 'dies'))
results['stabilizer'] = {int(r + RLO): float(surv[r]) for r in range(NB) if S_rung[r] > 1.5 * loc_med[r]}

# ---------- 8. passage: rung lane reading in v-order ----------
def nearest_cluster(rungnum):
    best, bd = None, 1e9
    for cl in clusters:
        d = min(abs(rungnum - c) for c in cl)
        if d < bd: best, bd = cl, d
    return best
passage = []
pub_by_line = []
for li in range(L):
    v0 = line_v(li)
    if not (WV0 <= v0 <= WV0 + WS): continue
    pub_by_line.append([b for (lli, b) in occ if lli == li])
    j0 = int(round(v0 - WV0)); jlo, jhi = max(0, j0 - 12), min(WS, j0 + 12)
    prof = band_S_by_rung.get((jlo, jhi))
    if prof is None or prof.sum() == 0:
        passage.append('?'); continue
    rpk = int(np.argmax(prof)) + RLO
    cl = nearest_cluster(rpk)
    if cl is None:
        passage.append('?'); continue
    cprof = np.zeros(NB); idx = [r - RLO for r in cl]
    cprof[idx] = S_rung[idx]
    if cprof.sum() > 0: cprof /= cprof.sum()
    best_b, best_s = max(((b, cos(cprof, table[b])) for b in GREEK_BASE), key=lambda t: t[1])
    passage.append(best_b if best_s >= 0.30 else '?')
pub_flat = ''.join(b for _, b in occ if WV0 <= line_v(_) <= WV0 + WS)
log('--- rung-lane passage (v-order, ?=no confident match) ---')
log('read:     %s' % ''.join(passage))
log('published:%s' % pub_flat[:len(passage) * 3])
hits, tot = 0, 0
for rd, pl in zip(passage, pub_by_line):
    if pl:
        tot += 1
        if rd in pl: hits += 1
log('rung-lane read letter present in published line: %d/%d = %.3f' % (hits, tot, hits / max(tot, 1)))
results['passage'] = ''.join(passage)
results['published_overlap'] = [hits, tot]

# ---------- 9. cross-lane: rung lane vs voxel lane ----------
vox_matches = []
try:
    fc = json.load(open(os.path.join(BASE, 'Fc_chars.json')))
    for thr, d in fc.items():
        for c in d.get('cands', []):
            vox_matches.append((c['top1'], c['s1']))
except Exception as e:
    log('voxel-lane Fc_chars.json not readable: %s' % e)
rung_letters = set(b for _, sc in cluster_matches for b, s in sc[:1] if s >= 0.30)
name2letter = dict(zip(
    ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa',
     'lambda','mu','nu','xi','omicron','pi','rho','sigma','tau','upsilon','phi',
     'chi','psi','omega'], list(GREEK_BASE)))
vox_letters = set(name2letter.get(n, n) for n, s in vox_matches if s >= 0.60)
log('voxel-lane letters>=0.60: %s' % sorted(vox_letters))
log('rung-lane top letters>=0.30: %s' % sorted(rung_letters))
log('rung-only (missed by voxel): %s' % sorted(rung_letters - vox_letters))
log('voxel-only (missed by rung): %s' % sorted(vox_letters - rung_letters))
results['cross_lane'] = {'voxel': sorted(vox_letters), 'rung': sorted(rung_letters),
                        'rung_only': sorted(rung_letters - vox_letters),
                        'voxel_only': sorted(vox_letters - rung_letters)}
json.dump(results, open(os.path.join(BASE, 'rung_sig_results.json'), 'w'), indent=1)
log('RUNG_LANE_DONE')
LOG.close()
