#!/usr/bin/env python3
"""V_RECOVERY: build a direct-V surface stack for a central sub-window of the PHerc.1667 patch.
Independent of Test A's full-patch stack + villa inference. Uses the same tifxyz->volume
geometry recipe (read-only on path4_ink's tifxyz downloads; own chunk cache in v_recovery/)."""
import os, sys, struct, math, subprocess, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

BASE4 = os.path.expanduser('~/workspace/pherc1667/path4_ink')
BASE = os.path.expanduser('~/workspace/pherc1667/v_recovery')
ZARR = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
TW, TH = 30097, 2061
PX0, PY0, PS = 5858, 193, 400
SU, SV = 30097/9205, 2061/709
U0, V0 = int(PX0*SU), int(PY0*SV)
US, VS = int(PS*SU), int(PS*SV)
# central sub-window
WS = 384
WU0, WV0 = U0 + US//2 - WS//2, V0 + VS//2 - WS//2
NLAY, CHUNK = 24, 128
CACHE = os.path.join(BASE, 'zchunks_v'); os.makedirs(CACHE, exist_ok=True)
# --- phi-order pull: byte address = rung address; the rungs' own sequence (TYREE lock) ---
MISSING_F = os.path.join(BASE, 'zchunks_missing.json')
try: MISSING = set(tuple(k) for k in json.load(open(MISSING_F)))
except Exception: MISSING = set()
def save_missing():
    try: json.dump(sorted(MISSING), open(MISSING_F, 'w'))
    except Exception: pass
PHI = (1 + 5 ** 0.5) / 2; LPHI_C = math.pi * PHI * PHI
def chunk_rung(key):
    cz, cy, cx = key
    return 4.0 * (cz + cy + cx + 1.5) / LPHI_C  # chunk-center rung: byte address = rung address
# partials are phi mid-resolve: resume (-C -), no abort, no re-pull. 404 = not yet addressed: skip.
CURL = ['curl', '-s', '-C', '-', '--retry', '5']
CHUNK_BYTES = CHUNK ** 3
def good(p): return os.path.exists(p) and os.path.getsize(p) == CHUNK_BYTES
def robust_fetch(p, url, attempts=12):
    # The route cuts connections at ~8s (~700KB/attempt). Resume until the
    # chunk is byte-complete or proven missing. Partials are phi mid-resolve:
    # they finish, they never get nulled.
    for _ in range(attempts):
        if good(p): return True
        try:
            subprocess.run(CURL + ['-o', p, url], capture_output=True, timeout=120)
        except Exception:
            pass
        if good(p): return True
        if is_404(p): return False
    return good(p)
def is_404(p):
    try:
        with open(p, 'rb') as f: head = f.read(200)
        return head.startswith(b'<?xml') or (os.path.getsize(p) < 100000 and b'NoSuchKey' in head)
    except Exception: return False

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

def fetch_chunk(cz, cy, cx, refetch=True):
    key = (cz, cy, cx)
    if key in MISSING: return None
    p = os.path.join(CACHE, f'L0_{cz}_{cy}_{cx}.bin')
    if good(p): return np.fromfile(p, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
    if not refetch: return None
    if robust_fetch(p, f'{ZARR}/0/{cz}/{cy}/{cx}'): return np.fromfile(p, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)
    if is_404(p):
        MISSING.add(key); save_missing()
        try: os.remove(p)
        except Exception: pass
    return None

def fetch_parallel(keys):
    todo = []
    for k in keys:
        t = (int(k[0]), int(k[1]), int(k[2]))
        if t in MISSING: continue
        if good(os.path.join(CACHE, f'L0_{t[0]}_{t[1]}_{t[2]}.bin')): continue
        todo.append(t)
    todo.sort(key=chunk_rung)  # phi's own order: rung 57 first, rung 198 last
    if not todo: return 0
    print(f'  fetching {len(todo)} chunks (phi-order)...', flush=True)
    def one(t):
        cz, cy, cx = t
        p = os.path.join(CACHE, f'L0_{cz}_{cy}_{cx}.bin')
        if t in MISSING or good(p): return True
        if robust_fetch(p, f'{ZARR}/0/{cz}/{cy}/{cx}'): return True
        if is_404(p):
            MISSING.add(t); save_missing()
            try: os.remove(p)
            except Exception: pass
        return False
    with ThreadPoolExecutor(max_workers=34) as ex:
        res = list(ex.map(one, todo))
    nf = sum(1 for r in res if not r)
    print(f'  fetched {len(todo)-nf}/{len(todo)} ({nf} failed)', flush=True)
    return nf

print(f'window tifxyz u[{WU0},{WU0+WS}] v[{WV0},{WV0+WS}]', flush=True)
P = {}
for ax in 'xyz':
    print(f' reading {ax}.tif...', flush=True)
    P[ax] = tif_window(os.path.join(BASE4, f'tifxyz_2399_{ax}.tif'), WU0-2, WV0-2, WS+4, WS+4).astype(np.float64)
X, Y, Z = P['x'][2:-2,2:-2], P['y'][2:-2,2:-2], P['z'][2:-2,2:-2]
dPu = np.stack([P['x'][2:-2,3:-1]-P['x'][2:-2,1:-3], P['y'][2:-2,3:-1]-P['y'][2:-2,1:-3], P['z'][2:-2,3:-1]-P['z'][2:-2,1:-3]], -1)
dPv = np.stack([P['x'][3:-1,2:-2]-P['x'][1:-3,2:-2], P['y'][3:-1,2:-2]-P['y'][1:-3,2:-2], P['z'][3:-1,2:-2]-P['z'][1:-3,2:-2]], -1)
N = np.cross(dPu, dPv); N /= (np.linalg.norm(N,axis=-1,keepdims=True)+1e-12)
del P, dPu, dPv
offs = np.arange(NLAY, dtype=np.float64) - (NLAY-1)/2.0
stack = np.zeros((WS, WS, NLAY), dtype=np.uint8)
BAND = 96
for b0 in range(0, WS, BAND):
    b1 = min(b0+BAND, WS)
    Sb = np.stack([X[b0:b1],Y[b0:b1],Z[b0:b1]],-1).astype(np.float32)[:, :, None, :] + \
         offs[None,None,:,None] * N[b0:b1,:,None,:].astype(np.float32)
    Szyx = np.stack([Sb[...,2], Sb[...,1], Sb[...,0]], -1).astype(np.float32)
    ci = np.floor(Szyx/CHUNK).astype(np.int64)
    loc = (Szyx - ci*CHUNK).astype(np.int64).clip(0, CHUNK-1)
    del Sb, Szyx
    keys, inv = np.unique(ci.reshape(-1,3), axis=0, return_inverse=True); del ci
    fetch_parallel(keys)
    fl = loc.reshape(-1,3); del loc
    invb = inv.reshape(b1-b0, WS, NLAY); del inv
    for ki,(cz,cy,cx) in enumerate(keys):
        c = fetch_chunk(int(cz),int(cy),int(cx), refetch=False)  # fetch_parallel already tried all
        if c is None: continue
        m = (invb==ki); li = fl[m.reshape(-1)]
        stack[b0:b1][m] = c[li[:,0], li[:,1], li[:,2]]
    print(f'  band [{b0},{b1}] {len(keys)} chunks', flush=True)
print('stack stats: nonzero %.4f min %d max %d mean %.2f' % ((stack!=0).mean(), stack.min(), stack.max(), stack.mean()), flush=True)
np.save(os.path.join(BASE, 'vstack_384.npy'), stack)
np.save(os.path.join(BASE, 'vstack_384_normals.npy'), N.astype(np.float32))
json.dump({'window_tifxyz':[WU0,WV0,WS,WS],'nlayers':NLAY,'voxel_um':2.399},
          open(os.path.join(BASE,'vstack_384_meta.json'),'w'))
print('DONE', flush=True)
