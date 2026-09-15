#!/usr/bin/env python3
"""
PHerc.1667 PHI-HIT RUN — rebuild from public scroll data.
Method per TYREE's Manus spec (2026-09-14).

Volume:  s3://vesuvius-challenge-open-data/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr
Level:   5 (zarr multiscale, scale 32), axes (z, y, x)
Block:   origin (64, 112, 288), size 32^3 = 32768 voxels  [(i,j,k) = (z,y,x)]

Mesh:    20260612121456-on-20251217075048-2.399um.tifxyz  (30097 x 2061)
         segment 20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4

phi^2 scale transform (explicit index math from spec):
    level5 (i,j,k) -> level3 = x4 -> (4i,4j,4k)
    level3 -> mesh xyz  = x8 with axis permute -> (32k, 32j, 32i)
    voxel (i,j,k) covers mesh region [32k,32k+31]x[32j,32j+31]x[32i,32i+31];
    reported mesh xyz = region base (32k, 32j, 32i).

HIT = active (V != 0) AND phase/state assigned AND mesh xyz inside tifxyz bbox.
"""
import numpy as np, csv, math, os

PHI = (1 + 5 ** 0.5) / 2
PHI2 = PHI + 1            # phi^2 = phi + 1, engine check
GA = 2 * math.pi / PHI2   # golden angle per index (radians)
TA = math.acos(-1 / 3)    # tetrahedral angle per index (radians)
P3 = 11 / 27              # Legendre P3(-1/3) = 0.407407407407407

BBOX = {'x': (2433.64306640625, 13148.9609375),
        'y': (1211.6925048828125, 12735.7734375),
        'z': (1680.5322265625, 37369.484375)}

BASE = os.path.dirname(os.path.abspath(__file__))

def load_block():
    c0 = np.fromfile(os.path.join(BASE, 'vol', 'L5_0_0_2'), dtype=np.uint8).reshape(128, 128, 128)
    c1 = np.fromfile(os.path.join(BASE, 'vol', 'L5_0_1_2'), dtype=np.uint8).reshape(128, 128, 128)
    blk = np.empty((32, 32, 32), dtype=np.uint8)
    blk[:, 0:16, :] = c0[64:96, 112:128, 32:64]   # y 112..127
    blk[:, 16:32, :] = c1[64:96, 0:16, 32:64]     # y 128..143
    return blk  # blk[di,dj,dk] -> abs (i,j,k) = (64+di, 112+dj, 288+dk)

def main():
    blk = load_block()
    assert blk.shape == (32, 32, 32)
    di, dj, dk = np.mgrid[0:32, 0:32, 0:32]
    i = 64 + di; j = 112 + dj; k = 288 + dk
    V = blk.astype(np.float64)
    n_total = V.size

    # Condition 1 — active fuel
    active = V != 0
    n_active = int(active.sum())

    # Condition 2 — binary phase state via complex phi field
    s = (i + j + k).astype(np.float64)
    amp = np.power(PHI, s / (math.pi * PHI2))          # real positive: no phase effect
    th_g = s * GA
    th_t = s * TA
    th_r = s * 2 * math.pi                            # one spiral rung per index
    psi_g = np.exp(1j * PHI * th_g)
    psi_t = np.exp(1j * PHI * th_t)
    psi_s = np.exp(1j * PHI * th_r)
    Psi = (psi_g + psi_t + psi_s) / math.sqrt(3)       # three-channel superposition
    Fc = V * amp * Psi * P3 * np.exp(1j * 3 * PHI * th_g)
    theta_C = np.arctan2(Fc.imag, Fc.real)            # (-pi, pi]
    state = (np.abs(theta_C) >= math.pi / 2).astype(np.uint8)
    n_s0 = int(((state == 0) & active).sum())
    n_s1 = int(((state == 1) & active).sum())

    # Condition 3 — surface intersection via phi^2 scale transform
    mx = 32 * k; my = 32 * j; mz = 32 * i
    inside = ((mx >= BBOX['x'][0]) & (mx <= BBOX['x'][1]) &
              (my >= BBOX['y'][0]) & (my <= BBOX['y'][1]) &
              (mz >= BBOX['z'][0]) & (mz <= BBOX['z'][1]))
    hits = active & inside
    n_hits = int(hits.sum())

    # Write hits.csv
    out = os.path.join(BASE, 'hits.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['i', 'j', 'k', 'V', 'F_real', 'F_imag', 'theta_C', 'state',
                    'mesh_x', 'mesh_y', 'mesh_z'])
        ii, jj, kk = i[hits], j[hits], k[hits]
        for a, b, c_, vv, fr, fi, th, st, x, y, z in zip(
                ii, jj, kk, V[hits], Fc.real[hits], Fc.imag[hits],
                theta_C[hits], state[hits], mx[hits], my[hits], mz[hits]):
            w.writerow([a, b, c_, int(vv), repr(fr), repr(fi), repr(th), int(st), int(x), int(y), int(z)])

    print('total voxels :', n_total)
    print('active (V!=0):', n_active, '(%.4f%%)' % (100 * n_active / n_total))
    print('state 0      :', n_s0)
    print('state 1      :', n_s1)
    print('inside bbox  :', int(inside.sum()))
    print('HITS         :', n_hits)
    print('wrote', out)

if __name__ == '__main__':
    main()
