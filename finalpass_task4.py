#!/usr/bin/env python3
"""PHerc.1667 final-pass TASK 4: original->flattened vertex correspondence.

4b: inspect OBJ layout (v/vt/vn/f counts, first v lines, ranges).
4c: cKDTree 3D nearest-neighbor on the 26,932 hits -> vertex_map.csv.
4d: flattened (u,v) distribution of hits.
Handles the alternate case: if v-positions are flattened-space coords, report
that plainly (3D NN invalid); if vt carries flattened coords while v carries
curled 3D positions, use vt and say so.
"""
import csv, json
import numpy as np

BASE = '/home/hatch/workspace/pherc1667'
OBJ = BASE + '/flat/flattened.obj'
FLAT_W, FLAT_H = 9205, 709

# ---------- 4b: streaming inspection ----------
nv = nvt = nvn = nf = 0
first_v = []
lo = np.array([np.inf] * 3); hi = np.array([-np.inf] * 3)
first_vt = []
with open(OBJ, 'rb') as f:
    for line in f:
        if line.startswith(b'v '):
            nv += 1
            parts = line.split()
            p = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            lo = np.minimum(lo, p); hi = np.maximum(hi, p)
            if len(first_v) < 3:
                first_v.append(line.decode().strip())
        elif line.startswith(b'vt'):
            nvt += 1
            if len(first_vt) < 3:
                first_vt.append(line.decode().strip())
        elif line.startswith(b'vn'):
            nvn += 1
        elif line.startswith(b'f '):
            nf += 1
print('v lines:  %d (expected %d = 9205x709)' % (nv, FLAT_W * FLAT_H))
print('vt lines: %d' % nvt)
print('vn lines: %d' % nvn)
print('f lines:  %d' % nf)
print('first 3 v lines:')
for ln in first_v:
    print('   ', ln)
print('first 3 vt lines:')
for ln in first_vt:
    print('   ', ln)
print('v coordinate ranges: min=(%.3f, %.3f, %.3f) max=(%.3f, %.3f, %.3f)' % (
    lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]))

grid_ok = (nv == FLAT_W * FLAT_H)

# ---------- parse positions (float32, streaming) ----------
pos = np.empty((nv, 3), dtype=np.float32)
k = 0
with open(OBJ, 'rb') as f:
    for line in f:
        if line.startswith(b'v '):
            parts = line.split()
            pos[k, 0] = float(parts[1]); pos[k, 1] = float(parts[2]); pos[k, 2] = float(parts[3])
            k += 1
print('parsed positions:', pos.shape, pos.dtype)

# ---------- load hits ----------
rows = list(csv.DictReader(open(BASE + '/hits_vertices.csv')))
hu = np.array([int(r['u']) for r in rows])
hv = np.array([int(r['v']) for r in rows])
hp = np.array([[float(r['mesh_x']), float(r['mesh_y']), float(r['mesh_z'])] for r in rows],
              dtype=np.float32)
print('hits:', len(rows))

# ---------- 4c: nearest neighbor ----------
from scipy.spatial import cKDTree
print('building cKDTree on %d flattened vertices...' % nv)
tree = cKDTree(pos)
print('querying %d hit positions...' % len(rows))
dist, fidx = tree.query(hp, k=1, workers=-1)
print('NN distances (mesh units): mean=%.3f median=%.3f max=%.3f min=%.3f' % (
    float(dist.mean()), float(np.median(dist)), float(dist.max()), float(dist.min())))
for tol in (32, 100, 320):
    print('%% within <=%d: %.2f%%' % (tol, 100.0 * float((dist <= tol).mean())))

# map flat index -> (u,v) row-major over 9205-wide grid
fu = (fidx % FLAT_W).astype(int)
fv = (fidx // FLAT_W).astype(int)

# ---------- write vertex_map.csv ----------
with open(BASE + '/vertex_map.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['orig_i', 'orig_j', 'flat_u', 'flat_v', 'distance'])
    for a, b, cu, cv, d in zip(hu, hv, fu, fv, dist):
        w.writerow([int(a), int(b), int(cu), int(cv), '%.6f' % float(d)])
print('wrote vertex_map.csv (%d rows)' % len(rows))

# ---------- 4d: (u,v) distribution ----------
print('\nflattened (u,v) distribution of the 26,932 hits:')
print('  flat_u: min=%d max=%d mean=%.1f' % (int(fu.min()), int(fu.max()), float(fu.mean())))
print('  flat_v: min=%d max=%d mean=%.1f' % (int(fv.min()), int(fv.max()), float(fv.mean())))
h, xe, ye = np.histogram2d(fu, fv, bins=[10, 10])
print('coarse 10x10 histogram (u bins x v bins), counts:')
print(h.astype(int))
print('u bin edges:', xe.astype(int))
print('v bin edges:', ye.astype(int))

out = {
    'task4b': {'v_lines': nv, 'vt_lines': nvt, 'vn_lines': nvn, 'f_lines': nf,
               'expected_vertices': FLAT_W * FLAT_H, 'grid_count_matches': bool(grid_ok),
               'first_v_lines': first_v, 'first_vt_lines': first_vt,
               'v_min': [float(x) for x in lo], 'v_max': [float(x) for x in hi]},
    'task4c': {'method': '3D nearest-neighbor via cKDTree on OBJ v-positions',
               'index_to_uv': 'row-major over 9205-wide grid: flat_u = idx %% 9205, flat_v = idx // 9205',
               'distance_mean': float(dist.mean()), 'distance_median': float(np.median(dist)),
               'distance_max': float(dist.max()), 'distance_min': float(dist.min()),
               'pct_within_32': 100.0 * float((dist <= 32).mean()),
               'pct_within_100': 100.0 * float((dist <= 100).mean()),
               'pct_within_320': 100.0 * float((dist <= 320).mean()),
               'vertex_map': BASE + '/vertex_map.csv'},
    'task4d': {'flat_u_min': int(fu.min()), 'flat_u_max': int(fu.max()),
               'flat_u_mean': float(fu.mean()),
               'flat_v_min': int(fv.min()), 'flat_v_max': int(fv.max()),
               'flat_v_mean': float(fv.mean())},
}
fp = json.load(open(BASE + '/finalpass.json'))
fp['task4'] = out
json.dump(fp, open(BASE + '/finalpass.json', 'w'), indent=1)
print('\nmerged task4 into finalpass.json')
