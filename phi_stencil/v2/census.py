#!/usr/bin/env python3
"""V2 setup: read tif window, compute vertices + central-difference normals,
census the 128^3 chunk keys needed for the 24-layer stack (nearest-neighbor,
same as build_vstack.py)."""
import os, struct, json, sys
import numpy as np

BASE4 = os.path.expanduser('~/workspace/pherc1667/path4_ink')
TW = 30097
WS, PAD = 384, 2
NLAY, CHUNK = 24, 128
HALF = (NLAY - 1) / 2.0

def tif_window(f, u0, v0, w, h):
    with open(f, 'rb') as fh:
        head = fh.read(8)
        bo = '<' if head[:2] == b'II' else '>'
        off = struct.unpack(bo + 'I', head[4:8])[0]
        fh.seek(off); n = struct.unpack(bo + 'H', fh.read(2))[0]
        strip = None
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(bo + 'HHI4s', fh.read(12))
            if tag == 273:
                strip = struct.unpack(bo + 'I', val)[0]
        out = np.empty((h, w), dtype=np.float32)
        for r in range(h):
            fh.seek(strip + ((v0 + r) * TW + u0) * 4)
            out[r] = np.frombuffer(fh.read(w * 4), dtype='<f4')
    return out

def main(u0, v0, outbase):
    os.makedirs(outbase, exist_ok=True)
    P = {}
    for ax in 'xyz':
        P[ax] = tif_window(os.path.join(BASE4, 'tifxyz_2399_%s.tif' % ax),
                           u0 - PAD, v0 - PAD, WS + 2 * PAD, WS + 2 * PAD).astype(np.float64)
    X, Y, Z = P['x'][2:-2, 2:-2], P['y'][2:-2, 2:-2], P['z'][2:-2, 2:-2]
    okv = (X > 0) & (Y > 0) & (Z > 0)
    print('valid frac: %.4f' % okv.mean(), flush=True)
    dPu = np.stack([P['x'][2:-2, 3:-1] - P['x'][2:-2, 1:-3],
                    P['y'][2:-2, 3:-1] - P['y'][2:-2, 1:-3],
                    P['z'][2:-2, 3:-1] - P['z'][2:-2, 1:-3]], -1)
    dPv = np.stack([P['x'][3:-1, 2:-2] - P['x'][1:-3, 2:-2],
                    P['y'][3:-1, 2:-2] - P['y'][1:-3, 2:-2],
                    P['z'][3:-1, 2:-2] - P['z'][1:-3, 2:-2]], -1)
    N = np.cross(dPu, dPv)
    N /= (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)
    del P, dPu, dPv
    nn = np.linalg.norm(N, axis=-1)
    print('window u[%d,%d] v[%d,%d]: |N| min/mean/max %.6f/%.6f/%.6f' %
          (u0, u0 + WS, v0, v0 + WS, nn.min(), nn.mean(), nn.max()), flush=True)
    np.save(os.path.join(outbase, 'vertices_xyz.npy'),
            np.stack([X, Y, Z], -1).astype(np.float32))
    np.save(os.path.join(outbase, 'normals.npy'), N.astype(np.float32))

    offs = np.arange(NLAY, dtype=np.float64) - HALF
    Sb = np.stack([X, Y, Z], -1).astype(np.float32)[:, :, None, :] + \
        offs[None, None, :, None] * N[:, :, None, :].astype(np.float32)
    Szyx = np.stack([Sb[..., 2], Sb[..., 1], Sb[..., 0]], -1)
    del Sb
    ci = np.floor(Szyx / CHUNK).astype(np.int64)
    keys = np.unique(ci.reshape(-1, 3), axis=0)
    del ci, Szyx
    keylist = [tuple(int(x) for x in k) for k in keys]
    json.dump(keylist, open(os.path.join(outbase, 'chunk_keys.json'), 'w'))
    json.dump({'window_tifxyz': [u0, v0, WS, WS], 'n_chunks': len(keylist),
               'valid_frac': float(okv.mean())},
              open(os.path.join(outbase, 'window_meta.json'), 'w'), indent=1)
    print('chunk census: %d chunks (%.1f MB)' % (len(keylist), len(keylist) * CHUNK**3 / 1e6), flush=True)

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
