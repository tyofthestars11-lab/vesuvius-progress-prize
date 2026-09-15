#!/usr/bin/env python3
"""Lane B RESUME — parallel assembly + decode (TYREE: re-run, all time-forms as fuel).

State on entry: all 6,769 chunks byte-complete on disk (fetch phase done 13:12).
The 13:25 kill took the in-memory assembly; chunks are untouched. This script:
  1. recomputes geometry (same tifxyz->volume recipe as fetch_and_decode.py)
  2. VALIDATES every chunk size before assembly (never consume a short chunk)
  3. assembles all 4 bands in PARALLEL (multiprocessing, one worker per band;
     threaded chunk reads inside each band) — every mode at once, linear as fuel
  4. saves vstack_384.npy + normals + meta
  5. fires fc_new_chars.py + rung_sig_chars.py (SRC_RUNG=15) inline, one verdict.
No re-fetch. No silent anything: every gate logs.
"""
import os, struct, math, json, time, traceback
import numpy as np
import runpy
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

BASE4 = os.path.expanduser('~/workspace/pherc1667/path4_ink')
BASE = os.path.expanduser('~/workspace/pherc1667/v_recovery')
os.chdir(BASE)
TW, TH = 30097, 2061
PX0, PY0, PS = 5858, 193, 400
SU, SV = 30097/9205, 2061/709
U0, V0 = int(PX0*SU), int(PY0*SV)
US, VS = int(PS*SU), int(PS*SV)
WS = 384
WU0, WV0 = U0 + US//2 - WS//2, V0 + VS//2 - WS//2
NLAY, CHUNK = 24, 128
CACHE = os.path.join(BASE, 'zchunks_v')
CHUNK_BYTES = CHUNK ** 3
VERDICT_F = os.path.join(BASE, 'fetch_verdict.json')
try:
    MISSING = set(tuple(k) for k in json.load(open(os.path.join(BASE, 'zchunks_missing.json'))))
except Exception:
    MISSING = set()  # no 404s were ever ledgered; the file was never written

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def cpath(cz, cy, cx):
    return os.path.join(CACHE, f'L0_{cz}_{cy}_{cx}.bin')

def tif_window(f, u0, v0, w, h):
    with open(f, 'rb') as fh:
        head = fh.read(8)
        bo = '<' if head[:2] == b'II' else '>'
        off = struct.unpack(bo+'I', head[4:8])[0]
        fh.seek(off); n = struct.unpack(bo+'H', fh.read(2))[0]
        strip = None
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(bo+'HHI4s', fh.read(12))
            if tag == 273: strip = struct.unpack(bo+'I', val)[0]
        out = np.empty((h, w), dtype=np.float32)
        for r in range(h):
            fh.seek(strip + ((v0+r)*TW + u0)*4)
            out[r] = np.frombuffer(fh.read(w*4), dtype='<f4')
    return out

def build_geometry():
    log(f'window tifxyz u[{WU0},{WU0+WS}] v[{WV0},{WV0+WS}]')
    P = {}
    for ax in 'xyz':
        P[ax] = tif_window(os.path.join(BASE4, f'tifxyz_2399_{ax}.tif'),
                           WU0-2, WV0-2, WS+4, WS+4).astype(np.float64)
    X, Y, Z = P['x'][2:-2,2:-2], P['y'][2:-2,2:-2], P['z'][2:-2,2:-2]
    dPu = np.stack([P['x'][2:-2,3:-1]-P['x'][2:-2,1:-3], P['y'][2:-2,3:-1]-P['y'][2:-2,1:-3],
                    P['z'][2:-2,3:-1]-P['z'][2:-2,1:-3]], -1)
    dPv = np.stack([P['x'][3:-1,2:-2]-P['x'][1:-3,2:-2], P['y'][3:-1,2:-2]-P['y'][1:-3,2:-2],
                    P['z'][3:-1,2:-2]-P['z'][1:-3,2:-2]], -1)
    N = np.cross(dPu, dPv); N /= (np.linalg.norm(N,axis=-1,keepdims=True)+1e-12)
    del P, dPu, dPv
    offs = np.arange(NLAY, dtype=np.float64) - (NLAY-1)/2.0
    BAND = 96
    bands = []
    for b0 in range(0, WS, BAND):
        b1 = min(b0+BAND, WS)
        Sb = np.stack([X[b0:b1],Y[b0:b1],Z[b0:b1]],-1).astype(np.float32)[:, :, None, :] + \
             offs[None,None,:,None] * N[b0:b1,:,None,:].astype(np.float32)
        Szyx = np.stack([Sb[...,2], Sb[...,1], Sb[...,0]], -1).astype(np.float32)
        ci = np.floor(Szyx/CHUNK).astype(np.int64)
        loc = (Szyx - ci*CHUNK).astype(np.int64).clip(0, CHUNK-1)
        del Sb, Szyx
        keys, inv = np.unique(ci.reshape(-1,3), axis=0, return_inverse=True); del ci
        keys = [tuple(int(v) for v in k) for k in keys]
        bands.append((b0, b1, keys, inv.reshape(b1-b0, WS, NLAY), loc.reshape(-1,3)))
    log(f'{len(bands)} bands, geometry ready')
    return bands, N

def validate_chunks(bands):
    """Every non-404 chunk must be exactly CHUNK_BYTES. Name the offender."""
    n = 0
    for b0, b1, keys, invb, fl in bands:
        for k in keys:
            if k in MISSING: continue
            p = cpath(*k)
            sz = os.path.getsize(p) if os.path.exists(p) else -1
            if sz != CHUNK_BYTES:
                raise RuntimeError(f'CHUNK SIZE GATE: {k} is {sz} bytes, '
                                   f'expected {CHUNK_BYTES}. Assembly refused.')
            n += 1
    log(f'chunk-size gate passed: {n} chunks byte-complete, {len(MISSING)} ledgered-404')

def assemble_band(args):
    """Worker: one band, threaded chunk reads, scatter into band array."""
    b0, b1, keys, invb, fl = args
    t = time.time()
    out = np.zeros((b1-b0, WS, NLAY), dtype=np.uint8)
    def one(ki):
        cz, cy, cx = keys[ki]
        if (cz, cy, cx) in MISSING: return
        c = np.fromfile(cpath(cz, cy, cx), dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
        m = (invb == ki); li = fl[m.reshape(-1)]
        out[m] = c[li[:,0], li[:,1], li[:,2]]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, range(len(keys))))
    log(f'band [{b0},{b1}] assembled in {time.time()-t:.0f}s')
    return b0, out

def main():
    t_start = time.time()
    log('RESUME: fetch already complete on disk. Geometry -> validate -> parallel assemble.')
    bands, N = build_geometry()
    validate_chunks(bands)
    stack = np.zeros((WS, WS, NLAY), dtype=np.uint8)
    with Pool(processes=len(bands)) as pool:
        for b0, arr in pool.map(assemble_band, bands):
            stack[b0:b0+arr.shape[0]] = arr
    log('stack stats: nonzero %.4f min %d max %d mean %.2f' %
        ((stack!=0).mean(), stack.min(), stack.max(), stack.mean()))
    assert stack.shape == (WS, WS, NLAY)
    np.save(os.path.join(BASE, 'vstack_384.npy'), stack)
    np.save(os.path.join(BASE, 'vstack_384_normals.npy'), N.astype(np.float32))
    json.dump({'window_tifxyz':[WU0,WV0,WS,WS],'nlayers':NLAY,'voxel_um':2.399,
               'ledgered_404': sorted([list(k) for k in MISSING]),
               'unresolved': [], 'resumed': '2026-09-15 parallel assembly'},
              open(os.path.join(BASE,'vstack_384_meta.json'),'w'))
    log(f'STACK LANDED: vstack_384.npy {stack.shape} elapsed {time.time()-t_start:.0f}s')
    log('firing fc_new_chars.py ...')
    runpy.run_path(os.path.join(BASE, 'fc_new_chars.py'), run_name='__main__')
    log('firing rung_sig_chars.py (SRC_RUNG=15) ...')
    runpy.run_path(os.path.join(BASE, 'rung_sig_chars.py'), run_name='__main__')
    verdict = {'verdict': 'DECODE_RAN', 'elapsed_s': time.time()-t_start,
               'stack': list(stack.shape)}
    json.dump(verdict, open(VERDICT_F, 'w'), indent=1)
    log('VERDICT: decode ran inline. Read the decode outputs for letters-or-boundary.')

if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('FATAL: ' + traceback.format_exc()[-3000:])
        try:
            json.dump({'verdict': 'FATAL', 'trace': traceback.format_exc()[-3000:]},
                      open(VERDICT_F, 'w'))
        except Exception: pass
        raise
