#!/usr/bin/env python3
"""Column assignment + Greek comparison for the 26,932 vertex hits.

Column model (LABELED FALLBACK): 22 equal columns over valid u in [8,30088]
on the ORIGINAL mesh. Direction A (primary): Col 1 at low u, reading order
left-to-right. Direction B (mirror) reported for completeness.
Per-vertex original->flattened correspondence is NOT published (Manus confirmed);
proportional column assignment is justified by the preprint's "near isometric"
parameterization (Angelotti et al., Technical strategy).
"""
import csv, json, os, re
import numpy as np

BASE = os.environ.get("PHERC1667_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
rows = list(csv.DictReader(open(BASE + '/hits_vertices.csv')))
u = np.array([int(r['u']) for r in rows])
v = np.array([int(r['v']) for r in rows])
st = np.array([int(r['state']) for r in rows])

U0, U1, NC = 8, 30088, 22
W = (U1 - U0) / NC
colA = np.clip(np.floor((u - U0) / W).astype(int) + 1, 1, 22)

greek = json.load(open(BASE + '/greek_density.json'))

def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)

# extract raw transcription lines per column from preprint
txt = open(BASE + '/preprint.txt').read().splitlines()
cols, cur = {}, None
for ln in txt:
    m = re.match(r'Col\.\s*(\d+)', ln)
    if m:
        cur = int(m.group(1)); cols[cur] = []
    elif cur and is_greek(ln):
        cols[cur].append(ln)

result = {'direction': 'A (Col 1 at low u)', 'columns': {}}
for c in range(1, 23):
    m = colA == c
    n = int(m.sum())
    if n == 0:
        continue
    uu = u[m]
    result['columns'][c] = {
        'hits': n,
        'state0': int((st[m] == 0).sum()),
        'state1': int((st[m] == 1).sum()),
        'u_min': int(uu.min()), 'u_max': int(uu.max()),
        'v_min': int(v[m].min()), 'v_max': int(v[m].max()),
        'greek_slots': greek.get(str(c), {}).get('slots', 0),
        'greek_letters': greek.get(str(c), {}).get('greek_letters', 0),
    }

hits = np.array([result['columns'][c]['hits'] for c in sorted(result['columns'])], float)
grk = np.array([result['columns'][c]['greek_slots'] for c in sorted(result['columns'])], float)
xh, xg = hits - hits.mean(), grk - grk.mean()
pearson = float((xh * xg).sum() / np.sqrt((xh * xh).sum() * (xg * xg).sum()))
result['pearson_hit_vs_greek'] = pearson
result['n_hit_columns'] = len(result['columns'])

json.dump(result, open(BASE + '/column_analysis.json', 'w'), indent=1)

print('Hit columns (Dir A):', sorted(result['columns']))
print('Pearson(hit density, greek slots): %.3f' % pearson)
print()
for c in sorted(result['columns']):
    d = result['columns'][c]
    print('Col %2d: %5d hits (s0=%5d s1=%5d) u=[%d,%d] v=[%d,%d] greek_slots=%d' % (
        c, d['hits'], d['state0'], d['state1'], d['u_min'], d['u_max'],
        d['v_min'], d['v_max'], d['greek_slots']))

# dump Greek text of hit columns
with open(BASE + '/hit_columns_greek.txt', 'w') as f:
    for c in sorted(result['columns']):
        f.write('=' * 60 + '\nCOL %d  (%d hits, %d greek slots)\n' % (
            c, result['columns'][c]['hits'], result['columns'][c]['greek_slots']) + '=' * 60 + '\n')
        for ln in cols.get(c, ['(no transcription lines extracted)']):
            f.write(ln + '\n')
        f.write('\n')
print('\nwrote column_analysis.json, hit_columns_greek.txt')
