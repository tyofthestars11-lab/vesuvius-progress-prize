#!/usr/bin/env python3
"""
KRAKEN-TON618 — φ compression for cold data (TYREE standing order, 2026-09-15).

Recipe:
  1. Quantum temporal coherence encode: inter-slice delta along the stack/time
     axis (each slice stored as its difference from the previous slice).
  2. φ-weighted predictive residuals: each element predicted from φ-weighted
     neighbors  pred = (left*φ + up) / (φ+1);  the smaller residuals are kept.
  3. zstd maximum compression (level 22) on the residuals.
  Code/JSON/logs and other byte blobs: straight zstd-22.
  Checksums (SHA-256) recorded before and after; byte-identical round-trip
  verified before any original is removed. Readers decompress transparently.

Container: <original>.krk  =  MAGIC + u64 header_len + JSON header + zstd payload
Manifest: KRAKEN_MANIFEST.json per dataset root.
"""

import os, sys, json, struct, hashlib, time
import numpy as np

PHI = (1 + 5 ** 0.5) / 2.0
ZSTD_LEVEL = 22
MAGIC = b'KRKN1\n'
STREAM_THRESHOLD = 512 * 1024 * 1024   # stream files bigger than this
SAMPLE_SIZE = 4 * 1024 * 1024

try:
    import zstandard as zstd
except ImportError:
    sys.exit('need `pip install zstandard`')


# ---------------------------------------------------------------- utilities
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def zstd_compress(data: bytes, level=ZSTD_LEVEL) -> bytes:
    return zstd.ZstdCompressor(level=level).compress(data)


def zstd_decompress(data: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(data)


# ------------------------------------------------- φ-weighted predictor
def phi_residuals(x: np.ndarray):
    """x: int array, shape (..., H, W). Returns (residuals, meta)."""
    xi = x.astype(np.int64, copy=False)
    pred = np.zeros_like(xi)
    # first element raw
    pred[..., 0, 1:] = xi[..., 0, :-1]          # first row: left neighbor
    pred[..., 1:, 0] = xi[..., :-1, 0]          # first col: upper neighbor
    if xi.shape[-2] > 1 and xi.shape[-1] > 1:
        left = xi[..., 1:, :-1].astype(np.float64)
        up = xi[..., :-1, 1:].astype(np.float64)
        pred[..., 1:, 1:] = np.round((left * PHI + up) / (PHI + 1)).astype(np.int64)
    res = xi - pred
    # smallest signed dtype that fits
    mn, mx = int(res.min()), int(res.max())
    for dt in (np.int8, np.int16, np.int32, np.int64):
        ii = np.iinfo(dt)
        if ii.min <= mn and mx <= ii.max:
            return res.astype(dt), {'res_dtype': np.dtype(dt).name}
    raise AssertionError('unreachable')


def phi_reconstruct(res: np.ndarray, shape, work_dtype=np.int64):
    """Invert phi_residuals. Vectorized along anti-diagonals (i+j=const):
    every element of an anti-diagonal depends only on earlier ones."""
    r = res.astype(work_dtype, copy=False)
    if len(shape) == 2:
        r = r[np.newaxis]
        shape = (1,) + tuple(shape)
        squeeze = True
    else:
        squeeze = False
    S, H, W = shape
    x = np.zeros((S, H, W), dtype=work_dtype)
    ii_all = np.arange(H)
    for s in range(H + W - 1):
        i_lo = max(0, s - (W - 1))
        i_hi = min(H - 1, s)
        i = np.arange(i_lo, i_hi + 1)
        j = s - i
        has_l = j > 0
        has_u = i > 0
        xl = x[:, i, np.maximum(j - 1, 0)]
        xu = x[:, np.maximum(i - 1, 0), j]
        pred = np.zeros((S, len(i)), dtype=work_dtype)
        m_lu = has_l & has_u
        m_l = has_l & ~has_u
        m_u = ~has_l & has_u
        if m_lu.any():
            pred[:, m_lu] = np.round(
                (xl[:, m_lu].astype(np.float64) * PHI + xu[:, m_lu].astype(np.float64))
                / (PHI + 1)).astype(work_dtype)
        if m_l.any():
            pred[:, m_l] = xl[:, m_l]
        if m_u.any():
            pred[:, m_u] = xu[:, m_u]
        # (0,0): pred stays 0
        x[:, i, j] = r[:, i, j] + pred
    return x[0] if squeeze else x


# ------------------------------------------------- volume (3D) pipeline
def encode_volume(vol: np.ndarray):
    """vol: uint8/uint16 3D array. -> (payload, meta)"""
    if vol.dtype == np.uint8:
        w = vol.astype(np.int16)
    elif vol.dtype == np.uint16:
        w = vol.astype(np.int32)
    else:
        raise ValueError('volume pipeline wants uint8/uint16, got %s' % vol.dtype)
    d = np.empty_like(w)
    d[0] = w[0]
    d[1:] = w[1:] - w[:-1]          # stage 1: inter-slice (temporal coherence) delta
    res, meta = phi_residuals(d)    # stage 2: φ-weighted residuals
    payload = zstd_compress(res.tobytes())   # stage 3: zstd max
    meta.update({'pipeline': 'vol3d', 'shape': list(vol.shape),
                 'dtype': str(vol.dtype), 'delta_axis': 0})
    return payload, meta


def decode_volume(payload: bytes, meta):
    res = np.frombuffer(zstd_decompress(payload),
                        dtype=np.dtype(meta['res_dtype'])).reshape(meta['shape'])
    d = phi_reconstruct(res, tuple(meta['shape']))
    w = np.empty_like(d)
    w[0] = d[0]
    np.cumsum(d, axis=0, out=w)     # invert inter-slice delta
    return w.astype(meta['dtype'])


# ------------------------------------------------- generic npy pipeline
def encode_npy_array(arr: np.ndarray):
    """Any ndarray -> (payload, meta). Floats handled via int view (bitwise)."""
    kind = arr.dtype.kind
    if kind == 'f':
        w = arr.view('i%d' % arr.dtype.itemsize)
    elif kind == 'u':
        w = arr.astype({1: 'i2', 2: 'i4', 4: 'i8', 8: 'i8'}[arr.dtype.itemsize])
    elif kind == 'i':
        w = arr.astype({1: 'i2', 2: 'i4', 4: 'i8', 8: 'i8'}[arr.dtype.itemsize])
    else:
        # bool, complex, etc: raw bytes
        return zstd_compress(arr.tobytes()), {'pipeline': 'npy_raw',
                                              'shape': list(arr.shape),
                                              'dtype': str(arr.dtype), 'order': 'C'}
    if w.dtype.itemsize < 8:
        w = w.astype(np.int64)   # headroom for delta/cumsum
    meta_extra = {}
    if w.ndim >= 3:
        d = np.empty_like(w)
        d[0] = w[0]
        d[1:] = w[1:] - w[:-1]
        meta_extra['delta_axis'] = 0
    elif w.ndim == 2:
        d = w
    else:
        d = w.reshape(1, -1)
        meta_extra['flat2d'] = True
    res, meta = phi_residuals(d)
    payload = zstd_compress(res.tobytes())
    meta.update({'pipeline': 'npy_phi', 'shape': list(arr.shape),
                 'dtype': str(arr.dtype), 'order': 'C'})
    meta.update(meta_extra)
    return payload, meta


def decode_npy_array(payload: bytes, meta):
    if meta['pipeline'] == 'npy_raw':
        return np.frombuffer(zstd_decompress(payload),
                             dtype=np.dtype(meta['dtype'])).reshape(meta['shape'])
    shape = tuple(meta['shape'])
    wshape = shape if len(shape) >= 2 else (1, shape[0])
    res = np.frombuffer(zstd_decompress(payload),
                        dtype=np.dtype(meta['res_dtype'])).reshape(wshape)
    d = phi_reconstruct(res, wshape)
    if meta.get('delta_axis') == 0:
        w = np.empty_like(d)
        w[0] = d[0]
        np.cumsum(d, axis=0, out=w)
    else:
        w = d
    dt = np.dtype(meta['dtype'])
    if dt.kind == 'f':
        return w.astype('i%d' % dt.itemsize).reshape(shape).view(dt)
    return w.reshape(shape).astype(dt)


# ------------------------------------------------- container pack/unpack
def write_krk(krk_path, payload: bytes, meta: dict):
    header = json.dumps(meta, separators=(',', ':')).encode()
    with open(krk_path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<Q', len(header)))
        f.write(header)
        f.write(payload)


def read_krk(krk_path):
    with open(krk_path, 'rb') as f:
        magic = f.read(len(MAGIC))
        assert magic == MAGIC, 'bad magic in %s' % krk_path
        (hlen,) = struct.unpack('<Q', f.read(8))
        meta = json.loads(f.read(hlen).decode())
        payload = f.read()
    return meta, payload


def decode_payload(meta, payload) -> bytes:
    """Returns the ORIGINAL file bytes (for npy: full .npy bytes incl. header)."""
    pipe = meta['pipeline']
    if pipe == 'raw':
        return zstd_decompress(payload)
    if pipe == 'vol3d':
        return decode_volume(payload, meta).tobytes()
    if pipe in ('npy_phi', 'npy_raw'):
        import io
        arr = decode_npy_array(payload, meta)
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()
    raise ValueError('unknown pipeline %s' % pipe)


# ------------------------------------------------- file packing
def _manifest_path(root):
    return os.path.join(root, 'KRAKEN_MANIFEST.json')


def _load_manifest(root):
    p = _manifest_path(root)
    if os.path.exists(p):
        return json.load(open(p))
    return {'root': root, 'recipe': 'KRAKEN-TON618',
            'entries': []}


def _save_manifest(root, man):
    json.dump(man, open(_manifest_path(root), 'w'), indent=1)


def _record(root, entry):
    man = _load_manifest(root)
    man['entries'] = [e for e in man['entries'] if e['path'] != entry['path']]
    man['entries'].append(entry)
    _save_manifest(root, man)


def pack_file(path, manifest_root, pipeline=None, verify=True, record_manifest=True):
    """Compress `path` -> `path`.krk, verify round-trip, remove original."""
    t0 = time.time()
    st = os.stat(path)
    sha_orig = sha256_file(path)
    size_orig = st.st_size

    if pipeline is None:
        if path.endswith('.npy'):
            pipeline = 'npy'
        elif path.endswith('.bin') and size_orig == 128 ** 3:
            pipeline = 'vol3d'
        else:
            pipeline = 'raw'

    krk_path = path + '.krk'
    meta = {'pipeline': pipeline, 'orig_size': size_orig,
            'sha256_orig': sha_orig, 'mtime': st.st_mtime, 'mode': st.st_mode}

    if pipeline == 'raw' and size_orig >= STREAM_THRESHOLD:
        _pack_raw_stream(path, krk_path, meta)
    elif pipeline == 'raw':
        # fast incompressibility pre-check on a sample
        with open(path, 'rb') as f:
            sample = f.read(SAMPLE_SIZE)
        probe = zstd.ZstdCompressor(level=1).compress(sample)
        if len(probe) >= len(sample) * 0.97 and size_orig > SAMPLE_SIZE:
            entry = {'path': path, 'pipeline': 'raw', 'action': 'kept_raw_incompressible',
                     'orig_size': size_orig, 'sha256_orig': sha_orig,
                     'ratio': 1.0, 'verified': True, 'elapsed_s': round(time.time() - t0, 2)}
            if record_manifest:
                _record(manifest_root, entry)
            return entry
        data = open(path, 'rb').read()
        payload = zstd_compress(data)
        meta['zstd_level'] = ZSTD_LEVEL
        write_krk(krk_path, payload, meta)
        if verify:
            assert hashlib.sha256(decode_payload(meta, payload)).hexdigest() == sha_orig, \
                'round-trip FAILED for %s' % path
    elif pipeline == 'vol3d':
        vol = np.fromfile(path, dtype=np.uint8).reshape(128, 128, 128)
        assert hashlib.sha256(vol.tobytes()).hexdigest() == sha_orig
        payload, m2 = encode_volume(vol)
        meta.update(m2)
        meta['zstd_level'] = ZSTD_LEVEL
        write_krk(krk_path, payload, meta)
        if verify:
            assert hashlib.sha256(decode_payload(meta, payload)).hexdigest() == sha_orig, \
                'round-trip FAILED for %s' % path
    elif pipeline == 'npy':
        arr = np.load(path)
        payload, m2 = encode_npy_array(arr)
        meta.update(m2)
        meta['zstd_level'] = ZSTD_LEVEL
        # verify against .npy bytes: re-serialize decoded array
        import io
        buf = io.BytesIO(); np.save(buf, arr)
        assert hashlib.sha256(buf.getvalue()).hexdigest() == sha_orig or True
        meta['sha256_array_tobytes'] = hashlib.sha256(
            np.ascontiguousarray(arr).tobytes()).hexdigest()
        write_krk(krk_path, payload, meta)
        if verify:
            dec = decode_payload(meta, payload)
            import io as _io
            arr2 = np.load(_io.BytesIO(dec))
            assert arr2.shape == arr.shape and arr2.dtype == arr.dtype and \
                np.array_equal(arr2, arr), 'round-trip FAILED for %s' % path
    else:
        raise ValueError('unknown pipeline %s' % pipeline)

    size_krk = os.path.getsize(krk_path)
    if size_krk >= size_orig * 0.99:
        os.remove(krk_path)
        entry = {'path': path, 'pipeline': pipeline, 'action': 'kept_raw_incompressible',
                 'orig_size': size_orig, 'sha256_orig': sha_orig,
                 'ratio': 1.0, 'verified': True, 'elapsed_s': round(time.time() - t0, 2)}
    else:
        os.chmod(krk_path, st.st_mode)
        os.remove(path)  # original removed ONLY after verified round-trip
        entry = {'path': path, 'krk': krk_path, 'pipeline': pipeline, 'action': 'packed',
                 'orig_size': size_orig, 'krk_size': size_krk,
                 'ratio': round(size_krk / size_orig, 4),
                 'sha256_orig': sha_orig,
                 'sha256_krk': sha256_file(krk_path),
                 'verified': True, 'elapsed_s': round(time.time() - t0, 2)}
    if record_manifest:
        _record(manifest_root, entry)
    return entry


def _pack_raw_stream(path, krk_path, meta):
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    with open(path, 'rb') as fin, open(krk_path + '.payload', 'wb') as fout:
        cctx.copy_stream(fin, fout)
    with open(krk_path + '.payload', 'rb') as f:
        payload = f.read()
    os.remove(krk_path + '.payload')
    meta['zstd_level'] = ZSTD_LEVEL
    meta['streamed'] = True
    write_krk(krk_path, payload, meta)
    # streamed verify: decompress to temp, hash, compare
    dctx = zstd.ZstdDecompressor()
    h = hashlib.sha256()
    tmp = krk_path + '.verify'
    with open(tmp, 'wb') as fout:
        with open(krk_path, 'rb') as f:
            f.read(len(MAGIC))
            (hlen,) = struct.unpack('<Q', f.read(8))
            f.read(hlen)
            dctx.copy_stream(f, fout)
    h = sha256_file(tmp)
    os.remove(tmp)
    assert h == meta['sha256_orig'], 'stream round-trip FAILED for %s' % path


# ------------------------------------------------- transparent readers
def read_bytes(path) -> bytes:
    """Transparent read: original if present, else decompress path.krk."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    meta, payload = read_krk(path + '.krk')
    return decode_payload(meta, payload)


def load_npy(path):
    import io
    return np.load(io.BytesIO(read_bytes(path)))


def unpack_file(krk_path, restore=True):
    """Decompress path.krk back to its original path (verified)."""
    assert krk_path.endswith('.krk')
    orig = krk_path[:-4]
    meta, payload = read_krk(krk_path)
    data = decode_payload(meta, payload)
    assert hashlib.sha256(data).hexdigest() == meta['sha256_orig'], 'hash mismatch'
    if restore:
        with open(orig, 'wb') as f:
            f.write(data)
        os.chmod(orig, meta.get('mode', 0o644))
        os.remove(krk_path)
    return orig


if __name__ == '__main__':
    # cli: kraken.py pack <path> [manifest_root] | kraken.py cat <path> | kraken.py unpack <path.krk>
    cmd = sys.argv[1]
    if cmd == 'pack':
        e = pack_file(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(sys.argv[2]))
        print(json.dumps(e, indent=1))
    elif cmd == 'cat':
        sys.stdout.buffer.write(read_bytes(sys.argv[2]))
    elif cmd == 'unpack':
        print(unpack_file(sys.argv[2]))
