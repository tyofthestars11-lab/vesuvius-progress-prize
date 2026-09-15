#!/usr/bin/env python3
"""PHerc.1667 final-pass TASK 4 (corrected): validated OBJ->flattened-grid correspondence.

Finding: OBJ vt/20 = integer (u,v) on the 9205x709 flattened grid.
- vt u in [0, 184060] -> u in [0, 9203]
- vt v in [1600, 14140] -> v in [80, 707]
Writes vertex_map.csv: obj_idx, flat_u, flat_v (validated).
3D nearest-neighbor across curled/straightened spaces is INVALID (documented with stats).
"""
import csv, json, os
import numpy as np

BASE = os.environ.get("PHERC1667_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OBJ = BASE + '/flat/flattened.obj'
FLAT_W, FLAT_H = 9205, 709

fu = []
fv = []
n = 0
nonint = 0
with open(OBJ, 'rb') as f:
    for line in f:
        if line.startswith(b'vt'):
            p = line.split()
            u = float(p[1]) / 20.0
            v = float(p[2]) / 20.0
            if u != int(u) or v != int(v):
                nonint += 1
            fu.append(int(round(u)))
            fv.append(int(round(v)))
            n += 1
fu = np.array(fu, dtype=np.int32)
fv = np.array(fv, dtype=np.int32)
print('vt parsed: %d' % n)
print('non-integer vt/20 values: %d' % nonint)
print('flat_u range: %d..%d (grid width %d)' % (int(fu.min()), int(fu.max()), FLAT_W))
print('flat_v range: %d..%d (grid height %d)' % (int(fv.min()), int(fv.max()), FLAT_H))

# uniqueness of (u,v) pairs
pairs = np.stack([fu, fv], axis=1)
upairs, counts = np.unique(pairs, axis=0, return_counts=True)
print('distinct (u,v) cells covered: %d of %d vertices' % (len(upairs), n))
print('max vertices sharing one cell: %d' % int(counts.max()))
print('cells with >1 vertex: %d' % int((counts > 1).sum()))

# write validated vertex_map.csv
with open(BASE + '/vertex_map.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['obj_idx', 'flat_u', 'flat_v'])
    for i, (u, v) in enumerate(zip(fu, fv)):
        w.writerow([i, int(u), int(v)])
print('wrote vertex_map.csv (%d rows): obj_idx -> flattened-grid (u,v) via vt/20' % n)

# 4d: distribution of OBJ vertices over the flattened grid
print('\nOBJ-vertex coverage of the 9205x709 flattened grid:')
print('  u: min=%d max=%d mean=%.1f' % (int(fu.min()), int(fu.max()), float(fu.mean())))
print('  v: min=%d max=%d mean=%.1f' % (int(fv.min()), int(fv.max()), float(fv.mean())))
h, xe, ye = np.histogram2d(fu, fv, bins=[10, 10])
print('coarse 10x10 histogram of OBJ vertices on flattened grid:')
print(h.astype(int))

out = {
    'task4b': {
        'v_lines': 4406818, 'vt_lines': 4406818, 'vn_lines': 4406818, 'f_lines': 8779784,
        'expected_vertices_9205x709': 6526445,
        'vertex_count_matches_grid': False,
        'note': 'OBJ mesh (4,406,818 verts) is a subset/decimation of the 9205x709 flattened grid, not the full grid.',
        'v_min': [26.209, -25.819, 14.532], 'v_max': [3335.845, 3226.142, 10871.079],
        'v_space': 'straightened-segment 3D (matches flattened TIFXYZ ranges, not curled-mesh space)',
    },
    'task4c': {
        'method_3d_nn': 'INVALID across spaces',
        'nn_evidence': {
            'note': 'cKDTree 3D NN between 26,932 curled-mesh hit positions and 4,406,818 straightened OBJ v-positions',
            'distance_mean': 6797.621, 'distance_median': 6765.071,
            'distance_min': 6196.753, 'distance_max': 7483.912,
            'pct_within_32': 0.0, 'pct_within_100': 0.0, 'pct_within_320': 0.0,
            'conclusion': 'OBJ v-positions live in straightened space (x 26-3336); hits live in curled-mesh space (x 9216-10239). No overlap; 3D NN is not a valid topology map.',
        },
        'validated_map': {
            'method': 'OBJ vt/20 -> integer (u,v) on the 9205x709 flattened grid',
            'vt_u_range': [0.0, 184060.0], 'vt_v_range': [1600.0, 14140.0],
            'flat_u_range': [int(fu.min()), int(fu.max())],
            'flat_v_range': [int(fv.min()), int(fv.max())],
            'non_integer_vt20': int(nonint),
            'distinct_cells': int(len(upairs)),
            'max_verts_per_cell': int(counts.max()),
            'cells_with_gt1_vertex': int((counts > 1).sum()),
            'vertex_map': BASE + '/vertex_map.csv',
            'columns': 'obj_idx,flat_u,flat_v',
            'rows': int(n),
        },
        'remaining_blocker': 'original-mesh (curled) hit -> flattened-grid (u,v) still requires the flatboi straightening deformation field (original-TIFXYZ-to-flattened-OBJ source-index correspondence), which is not in the public release.',
    },
    'task4d': {
        'flat_u_min': int(fu.min()), 'flat_u_max': int(fu.max()), 'flat_u_mean': float(fu.mean()),
        'flat_v_min': int(fv.min()), 'flat_v_max': int(fv.max()), 'flat_v_mean': float(fv.mean()),
        'note': 'Distribution of the 4,406,818 OBJ vertices over the flattened grid (validated vt/20 map). Per-hit flattened positions remain blocked (see task4c).',
    },
}
fp = json.load(open(BASE + '/finalpass.json'))
fp['task4'] = out
json.dump(fp, open(BASE + '/finalpass.json', 'w'), indent=1)
print('\nmerged corrected task4 into finalpass.json')
