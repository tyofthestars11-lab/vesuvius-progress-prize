#!/usr/bin/env python3
"""
PHerc.1667 VERTEX HIT RUN — rebuild from public scroll data (Manus final geometry pass).

The 26,932 hits are ORIGINAL-MESH VERTICES (30097 x 2061 grid), not voxels:
a vertex is a hit when its 3D position falls inside the block's mapped mesh region.

Block (level 5, zarr (z,y,x)): origin (64,112,288), 32^3.
Mesh region: x in [9216,10239], y in [3584,4607], z in [2048,3071].

Per hit vertex: inverse phi^2 map -> voxel (i,j,k) = (z/32, y/32, x/32),
look up V, compute F_C / theta_C / state, write hits_vertices.csv.
"""
import numpy as np, csv, math, os
from PIL import Image

PHI = (1 + 5 ** 0.5) / 2
PHI2 = PHI + 1
GA = 2 * math.pi / PHI2
TA = math.acos(-1 / 3)
P3 = 11 / 27

REGION = {'x': (9216.0, 10239.0), 'y': (3584.0, 4607.0), 'z': (2048.0, 3071.0)}
BASE = os.path.dirname(os.path.abspath(__file__))

def load_block():
    c0 = np.fromfile(os.path.join(BASE, 'vol', 'L5_0_0_2'), dtype=np.uint8).reshape(128, 128, 128)
    c1 = np.fromfile(os.path.join(BASE, 'vol', 'L5_0_1_2'), dtype=np.uint8).reshape(128, 128, 128)
    blk = np.empty((32, 32, 32), dtype=np.uint8)
    blk[:, 0:16, :] = c0[64:96, 112:128, 32:64]
    blk[:, 16:32, :] = c1[64:96, 0:16, 32:64]
    return blk

def plane_rows(name, r0, r1):
    im = Image.open(os.path.join(BASE, 'orig2399', name + '.tif'))
    return np.array(im.crop((0, r0, im.width, r1)), dtype=np.float64)

def main():
    blk = load_block()
    W, H = 30097, 2061
    CH = 256
    out = open(os.path.join(BASE, 'hits_vertices.csv'), 'w', newline='')
    w = csv.writer(out)
    w.writerow(['u', 'v', 'i', 'j', 'k', 'V', 'F_real', 'F_imag', 'theta_C', 'state',
                'mesh_x', 'mesh_y', 'mesh_z'])
    n_hits = 0
    vox_seen = {}
    for r0 in range(0, H, CH):
        r1 = min(H, r0 + CH)
        X = plane_rows('x', r0, r1); Y = plane_rows('y', r0, r1); Z = plane_rows('z', r0, r1)
        m = ((X >= REGION['x'][0]) & (X <= REGION['x'][1]) &
             (Y >= REGION['y'][0]) & (Y <= REGION['y'][1]) &
             (Z >= REGION['z'][0]) & (Z <= REGION['z'][1]))
        rr, cc = np.nonzero(m)
        if rr.size == 0:
            continue
        vx, vy, vz = X[m], Y[m], Z[m]
        vv = r0 + rr  # v coordinate (row)
        # inverse phi^2 map: voxel (i,j,k) = (z/32, y/32, x/32)
        ii = np.floor(vz / 32).astype(int)
        jj = np.floor(vy / 32).astype(int)
        kk = np.floor(vx / 32).astype(int)
        ok = ((ii >= 64) & (ii < 96) & (jj >= 112) & (jj < 144) & (kk >= 288) & (kk < 320))
        idx = np.nonzero(ok)[0]
        for t in idx:
            a, b, c_ = int(ii[t]), int(jj[t]), int(kk[t])
            V = float(blk[a - 64, b - 112, c_ - 288])
            s = float(a + b + c_)
            amp = PHI ** (s / (math.pi * PHI2))
            th_g = s * GA; th_t = s * TA; th_r = s * 2 * math.pi
            Psi = (np.exp(1j * PHI * th_g) + np.exp(1j * PHI * th_t) + np.exp(1j * PHI * th_r)) / math.sqrt(3)
            Fc = V * amp * Psi * P3 * np.exp(1j * 3 * PHI * th_g)
            thC = math.atan2(Fc.imag, Fc.real)
            st = 0 if abs(thC) < math.pi / 2 else 1
            w.writerow([int(cc[t]), int(vv[t]), a, b, c_, int(V),
                        repr(float(Fc.real)), repr(float(Fc.imag)), repr(thC), st,
                        repr(float(vx[t])), repr(float(vy[t])), repr(float(vz[t]))])
            n_hits += 1
            key = (a, b, c_)
            if key not in vox_seen:
                vox_seen[key] = V
        print('rows %d-%d: hits so far %d' % (r0, r1, n_hits), flush=True)
    out.close()
    n_vox = len(vox_seen)
    n_act = sum(1 for v in vox_seen.values() if v > 0)
    print('VERTEX HITS:', n_hits)
    print('distinct voxels touched:', n_vox)
    print('distinct voxels with V>0:', n_act)
    print('wrote hits_vertices.csv')

if __name__ == '__main__':
    main()
