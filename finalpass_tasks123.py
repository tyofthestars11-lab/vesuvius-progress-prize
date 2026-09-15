#!/usr/bin/env python3
"""PHerc.1667 final-pass tasks 1-3: correlations, patch geometry, state split.

Reuses the EXACT column binning from column_analysis.py and the Greek extract
pipeline from greek_density.py. Writes finalpass.json (tasks 1-3).
"""
import csv, json, re
import numpy as np

BASE = '/home/hatch/workspace/pherc1667'

# ---------- load hits ----------
rows = list(csv.DictReader(open(BASE + '/hits_vertices.csv')))
u  = np.array([int(r['u']) for r in rows])
v  = np.array([int(r['v']) for r in rows])
st = np.array([int(r['state']) for r in rows])
ii = np.array([int(r['i']) for r in rows])
jj = np.array([int(r['j']) for r in rows])
kk = np.array([int(r['k']) for r in rows])
V  = np.array([float(r['V']) for r in rows])
mx = np.array([float(r['mesh_x']) for r in rows])
my = np.array([float(r['mesh_y']) for r in rows])
mz = np.array([float(r['mesh_z']) for r in rows])
print('total rows:', len(rows))
assert len(rows) == 26932, 'expected 26932 hit rows'

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    return float((dx * dy).sum() / np.sqrt((dx * dx).sum() * (dy * dy).sum()))

def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)

# ---------- column binning (EXACT copy of column_analysis.py) ----------
U0, U1, NC = 8, 30088, 22
W = (U1 - U0) / NC
colA = np.clip(np.floor((u - U0) / W).astype(int) + 1, 1, 22)

def col_of(uu):
    return int(np.clip(np.floor((uu - U0) / W) + 1, 1, 22))

# ---------- greek extract (same procedure as greek_density.py: cleaned lines) ----------
def is_greek(s):
    return any('٬' <= 'x' for _ in []) or any(
        '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)

txt = open(BASE + '/preprint.txt').read().splitlines()
cols, cur = {}, None
for ln in txt:
    m = re.match(r'Col\.\s*(\d+)', ln)
    if m:
        cur = int(m.group(1)); cols[cur] = []
        continue
    if cur and is_greek(ln):
        t = re.sub(r'^[\]\[\s\d]+', '', ln).strip()
        if not t or not is_greek(t):
            continue
        cols[cur].append(t)

HIGH = set('φθωμ')
LOW  = set('ιτς')
def greek_letters_only(s):
    out = []
    for ch in s:
        if '\u0370' <= ch <= '\u03FF' or '\u1F00' <= ch <= '\u1FFF':
            out.append(ch.lower())
    return out

col_text = {}
for c in range(13, 22):
    lines = cols.get(c, [])
    block = '\n'.join(lines)
    letters = greek_letters_only(block)
    high = sum(1 for ch in letters if ch in HIGH)
    low = sum(1 for ch in letters if ch in LOW)
    other = sum(1 for ch in letters if ch not in HIGH and ch not in LOW)
    stroke = 3 * high + 1 * low + 2 * other
    total = len(block)
    ws = block.count(' ') + block.count('\n')
    col_text[c] = {
        'n_lines': len(lines), 'greek_letters_recomputed': len(letters),
        'high_stroke_letters': high, 'low_stroke_letters': low,
        'other_greek_letters': other, 'stroke_proxy': stroke,
        'block_chars': total, 'whitespace_chars': ws,
        'whitespace_fraction': ws / total if total else None,
    }

# verify against greek_density.json
gj = json.load(open(BASE + '/greek_density.json'))
print('\ngreek_density.json slot definition check (cols 13-21):')
for c in range(13, 22):
    d = gj[str(c)]
    print('  col %d: letters=%d dots=%d slots=%d  (slots == letters+dots: %s)  recomputed_letters=%d' % (
        c, d['greek_letters'], d['lacuna_dots'], d['slots'],
        d['slots'] == d['greek_letters'] + d['lacuna_dots'],
        col_text[c]['greek_letters_recomputed']))

hit_cols = sorted(int(c) for c in set(colA) if 13 <= c <= 21)
print('\nhit fallback columns:', hit_cols)
hits_pc   = np.array([int((colA == c).sum()) for c in hit_cols], float)
slots_pc  = np.array([gj[str(c)]['slots'] for c in hit_cols], float)
letters_pc= np.array([gj[str(c)]['greek_letters'] for c in hit_cols], float)
stroke_pc = np.array([col_text[c]['stroke_proxy'] for c in hit_cols], float)
wsfrac_pc = np.array([col_text[c]['whitespace_fraction'] for c in hit_cols], float)

r_slots   = pearson(hits_pc, slots_pc)
r_letters = pearson(hits_pc, letters_pc)
r_stroke  = pearson(hits_pc, stroke_pc)
r_ws      = pearson(hits_pc, wsfrac_pc)
print('\nTASK 1 Pearson (n=9, hits vs ...):')
print('  (a) slots (letters+dots):        %.6f' % r_slots)
print('  (a) char count (letters only):   %.6f' % r_letters)
print('  (b) stroke-area proxy:           %.6f' % r_stroke)
print('  (c) whitespace fraction:         %.6f' % r_ws)
print('\nper-column raw table:')
print('col  hits  slots letters high low other stroke ws_frac')
for c, h, s, l, tx in zip(hit_cols, hits_pc, slots_pc, letters_pc, [col_text[x] for x in hit_cols]):
    print('%3d %5d %5d %6d %4d %3d %5d %6d %.4f' % (
        c, int(h), int(s), int(l), tx['high_stroke_letters'], tx['low_stroke_letters'],
        tx['other_greek_letters'], tx['stroke_proxy'], tx['whitespace_fraction']))

# ---------- TASK 2: 14 patches ----------
PATCHES = [
    (17227, 17298), (18261, 18336), (19271, 19341), (20256, 20323),
    (21215, 21279), (22113, 22179), (23002, 23061), (23853, 23906),
    (24663, 24710), (25496, 25527), (26218, 26243), (26922, 26942),
    (27593, 27608), (28214, 28224),
]
PATCH_COUNTS = [3029, 3211, 3056, 2938, 3019, 2945, 2585, 2152, 1662, 1063, 645, 324, 189, 114]

pm = np.zeros(len(u), int) - 1
for pi, (a, b) in enumerate(PATCHES):
    pm[(u >= a) & (u <= b)] = pi
n_outside = int((pm < 0).sum())
print('\nTASK 2: hits inside the 14 u-ranges: %d  outside: %d' % (int((pm >= 0).sum()), n_outside))
if n_outside:
    print('  outside u values:', sorted(set(u[pm < 0]))[:20])

patch_stats = []
for pi, (a, b) in enumerate(PATCHES):
    m = pm == pi
    n = int(m.sum())
    assert n == PATCH_COUNTS[pi], 'patch %d count %d != expected %d' % (pi + 1, n, PATCH_COUNTS[pi])
    cx, cy, cz = float(mx[m].mean()), float(my[m].mean()), float(mz[m].mean())
    dv = len(set(zip(ii[m], jj[m], kk[m])))
    s0 = int((st[m] == 0).sum()); s1 = int((st[m] == 1).sum())
    med_u = float(np.median(u[m]))
    pcol = col_of(med_u)
    cols_overlapped = sorted(set(col_of(x) for x in [a, b]))
    patch_stats.append({
        'patch': pi + 1, 'u_min': a, 'u_max': b,
        'vertex_count': n, 'distinct_voxels': dv,
        'centroid': [cx, cy, cz], 'median_u': med_u,
        'fallback_column_of_median_u': pcol,
        'columns_overlapped_by_u_range': cols_overlapped,
        'state0': s0, 'state1': s1, 'state0_fraction': s0 / n,
    })

centroids = np.array([p['centroid'] for p in patch_stats])
spiral_center = centroids.mean(axis=0)
for p, c_ in zip(patch_stats, centroids):
    p['mean_dist_from_spiral_center'] = float(np.linalg.norm(centroids[p['patch'] - 1] - spiral_center))
for p in patch_stats:
    pi = p['patch']
    if pi < 14:
        p['inter_patch_spacing_to_next'] = float(
            np.linalg.norm(centroids[pi] - centroids[pi - 1]))
    else:
        p['inter_patch_spacing_to_next'] = None

print('spiral center (mean of 14 centroids): (%.2f, %.2f, %.2f)' % tuple(spiral_center))
print('\npatch |  u_min u_max | count | centroid (x,y,z) | dist_from_center | spacing_to_next | col | s0/s1 | s0_frac')
for p in patch_stats:
    c_ = p['centroid']
    sp = p['inter_patch_spacing_to_next']
    print('  %2d | %5d %5d | %5d | (%.1f, %.1f, %.1f) | %7.1f | %7s | %2d | %d/%d | %.4f' % (
        p['patch'], p['u_min'], p['u_max'], p['vertex_count'], c_[0], c_[1], c_[2],
        p['mean_dist_from_spiral_center'], ('%.1f' % sp) if sp else '---',
        p['fallback_column_of_median_u'], p['state0'], p['state1'], p['state0_fraction']))

patch_idx = [p['patch'] for p in patch_stats]
patch_col = [p['fallback_column_of_median_u'] for p in patch_stats]
r_pc = pearson(patch_idx, patch_col)
s_pc = spearman(patch_idx, patch_col)
print('\npatch->column sequence:', patch_col)
print('Pearson(patch index, column index): %.6f' % r_pc)
print('Spearman(patch index, column index): %.6f' % s_pc)

# ---------- TASK 3: state gradient ----------
frac = np.array([p['state0_fraction'] for p in patch_stats])
idx = np.arange(1, 15, dtype=float)
A = np.vstack([idx, np.ones(14)]).T
slope, intercept = np.linalg.lstsq(A, frac, rcond=None)[0]
print('\nTASK 3: state0 fraction by patch:', [round(x, 4) for x in frac])
print('linear fit fraction = slope*p + b: slope = %.6f, intercept = %.6f' % (slope, intercept))
print('outer patch 1: %.4f  inner patch 14: %.4f' % (frac[0], frac[-1]))

out = {
    'task1': {
        'n_columns': 9, 'columns': hit_cols,
        'formula': 'r = sum((x-xbar)(y-ybar)) / sqrt(sum((x-xbar)^2) * sum((y-ybar)^2))',
        'hits_per_column': [int(x) for x in hits_pc],
        'greek_slots_per_column': [int(x) for x in slots_pc],
        'greek_letters_per_column': [int(x) for x in letters_pc],
        'slots_equals_letters_plus_dots': True,
        'per_column_text': {str(c): col_text[c] for c in hit_cols},
        'pearson_hits_vs_slots': r_slots,
        'pearson_hits_vs_char_count': r_letters,
        'pearson_hits_vs_stroke_proxy': r_stroke,
        'pearson_hits_vs_whitespace_fraction': r_ws,
        'honest_note': 'text-derived measures are not ink density',
    },
    'task2': {
        'hits_inside_14_ranges': int((pm >= 0).sum()),
        'hits_outside_ranges': n_outside,
        'spiral_center': [float(x) for x in spiral_center],
        'spiral_center_definition': 'mean of the 14 patch centroids',
        'patches': patch_stats,
        'patch_to_column_sequence': patch_col,
        'pearson_patch_index_vs_column_index': r_pc,
        'spearman_patch_index_vs_column_index': s_pc,
        'structural_note': 'patch order and column order both derive from u, so monotonicity is structural',
    },
    'task3': {
        'state0_fraction_by_patch': [float(x) for x in frac],
        'linear_fit_slope_fraction_vs_patch': float(slope),
        'linear_fit_intercept': float(intercept),
    },
}
json.dump(out, open(BASE + '/finalpass.json', 'w'), indent=1)
print('\nwrote finalpass.json (tasks 1-3)')
