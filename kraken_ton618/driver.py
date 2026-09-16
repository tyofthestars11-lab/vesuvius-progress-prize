#!/usr/bin/env python3
"""KRAKEN bulk driver: pack a dataset tree, resumable, with progress + space guard."""
import os, sys, json, time, shutil
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kraken import pack_file, _load_manifest, _save_manifest

SKIP_DIRS = {'venv', '__pycache__', '.git', 'node_modules', 'h4proj_env'}
SKIP_EXTS = {'.pyc', '.krk', '.pyo'}
MIN_FREE_GB = 5.0


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def collect(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] in SKIP_EXTS:
                continue
            if fn == 'KRAKEN_MANIFEST.json':
                continue
            p = os.path.join(dirpath, fn)
            if os.path.exists(p + '.krk'):
                continue  # already packed
            files.append(p)
    return sorted(files)


def _work(args):
    path, root = args
    try:
        return pack_file(path, root, record_manifest=False)
    except Exception as e:  # noqa: BLE001 - record and continue
        return {'path': path, 'action': 'error', 'error': repr(e)}


def run(root, workers=2, progress_every=50):
    root = os.path.abspath(root)
    man = _load_manifest(root)
    done = {e['path'] for e in man['entries'] if e.get('action') in ('packed', 'kept_raw_incompressible')}
    files = [f for f in collect(root) if f not in done]
    total = len(files)
    print('KRAKEN pass on %s: %d files to process (%d already done)' % (root, total, len(done)), flush=True)
    if not files:
        return summarize(root)
    t0 = time.time()
    n_done = 0
    b_orig = b_new = 0
    pending = []

    def flush():
        if not pending:
            return
        man = _load_manifest(root)
        paths = {e['path'] for e in pending}
        man['entries'] = [e for e in man['entries'] if e['path'] not in paths]
        man['entries'].extend(pending)
        _save_manifest(root, man)
        pending.clear()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for entry in ex.map(_work, [(f, root) for f in files]):
            pending.append(entry)
            n_done += 1
            b_orig += entry.get('orig_size', 0)
            b_new += entry.get('krk_size', entry.get('orig_size', 0))
            if entry.get('action') == 'error':
                print('ERROR %s: %s' % (entry['path'], entry.get('error')), flush=True)
            if n_done % progress_every == 0:
                el = time.time() - t0
                print('[%d/%d] %.1f%%  %s/file  saved %.2f GB  free %.1f GB' % (
                    n_done, total, 100.0 * n_done / total, '%.2fs' % (el / n_done),
                    (b_orig - b_new) / 1e9, free_gb(root)), flush=True)
            if n_done % 25 == 0:
                flush()
            if free_gb(root) < MIN_FREE_GB:
                flush()
                print('SPACE GUARD: free < %.1f GB — stopping.' % MIN_FREE_GB, flush=True)
                break
    flush()
    return summarize(root)


def summarize(root):
    man = _load_manifest(root)
    packed = [e for e in man['entries'] if e.get('action') == 'packed']
    kept = [e for e in man['entries'] if e.get('action') == 'kept_raw_incompressible']
    errs = [e for e in man['entries'] if e.get('action') == 'error']
    bo = sum(e.get('orig_size', 0) for e in packed)
    bn = sum(e.get('krk_size', 0) for e in packed)
    s = {'root': root, 'files_packed': len(packed), 'files_kept_raw': len(kept),
         'errors': len(errs), 'bytes_orig': bo, 'bytes_krk': bn,
         'ratio': round(bn / bo, 4) if bo else 1.0,
         'saved_gb': round((bo - bn) / 1e9, 3),
         'free_gb': round(free_gb(root), 2)}
    print('SUMMARY ' + json.dumps(s, indent=1), flush=True)
    json.dump(s, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'KRAKEN_SUMMARY_%s.json' % os.path.basename(root.rstrip('/'))), 'w'), indent=1)
    return s


if __name__ == '__main__':
    run(sys.argv[1], workers=int(sys.argv[2]) if len(sys.argv) > 2 else 2)
