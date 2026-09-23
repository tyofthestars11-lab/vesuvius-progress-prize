#!/usr/bin/env python3
"""Run the pure-phi ink detector: 5 orientation seeds on real data,
shuffled-depth / shuffled-spatial / uniform controls, maps -> maps/.

Transform protocol mirrors the morning run: rot90 k in {0..3} on (y,x)
plus optional x-flip, drawn from the same seeds; the output map is
inverse-transformed so a data-intrinsic feature lands in the same place
every seed.
"""
import os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phi_detector import run_instrument, interior_mask

SCRATCH = os.path.expanduser(
    '~/workspace/.jarvis/idea-executions/47307d3e-0123-41fd-852e-500a7ca9cdbd/scratch')
STACK_PATH = os.path.join(SCRATCH, 'patch_stack_L1w.npy')  # read-only
OUT = os.path.join(HERE, 'maps')
os.makedirs(OUT, exist_ok=True)

SEEDS = [137, 1001, 2026, 31337, 99991]
CTL_SEEDS = [137, 1001, 2026]
THR = 3.0


def apply_transform(V, k, flip):
    y = np.rot90(V, k, axes=(0, 1))
    if flip:
        y = y[:, ::-1, :]
    return y


def invert_transform(M, k, flip):
    if flip:
        M = M[:, ::-1, :]
    if k:
        M = np.rot90(M, -k % 4, axes=(0, 1))
    return np.ascontiguousarray(M)


def make_control(stack, cname):
    rng = np.random.default_rng(777)  # same as the morning run
    H, W, D = stack.shape
    if cname == 'shuffled_depth':
        return stack[:, :, rng.permutation(D)]
    if cname == 'shuffled_spatial':
        perm = rng.permutation(H * W)
        return stack.reshape(H * W, D)[perm, :].reshape(H, W, D)
    if cname == 'uniform':
        return np.full(stack.shape, float(stack.mean()), dtype=np.float32)
    raise ValueError(cname)


def run_one(args):
    kind, seed = args
    stack = np.load(STACK_PATH)  # read-only load
    cname = kind[4:] if kind.startswith('ctl_') else None
    V = make_control(stack, cname).astype(np.float32) if cname else \
        stack.astype(np.float32)

    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))

    Ks, parts = run_instrument(apply_transform(V, k, flip), return_parts=True)
    Ks = invert_transform(Ks, k, flip).astype(np.float32)

    fn = os.path.join(OUT, f'phi_{kind}_seed{seed}.npy')
    np.save(fn, Ks)
    msk = interior_mask(Ks.shape)
    Ki = Ks[msk]
    return {'kind': kind, 'seed': seed, 'k': k, 'flip': flip,
            'max': float(Ki.max()), 'mean': float(Ki.mean()),
            'frac_gt_thr': float((Ki > THR).mean()), **parts}


def main():
    t0 = time.time()
    jobs = [('real', s) for s in SEEDS]
    for cname in ['shuffled_depth', 'shuffled_spatial']:
        for s in CTL_SEEDS:
            jobs.append((f'ctl_{cname}', s))
    jobs.append(('ctl_uniform', 137))
    print(f'{len(jobs)} phi runs queued', flush=True)

    workers = max(1, os.cpu_count() or 1)
    meta = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run_one, jobs):
            meta.append(r)
            print('done %(kind)s seed %(seed)d k=%(k)d flip=%(flip)s '
                  'max %(max).3f mean %(mean).5f frac>3 %(frac_gt_thr).6f '
                  'sigR %(sigma_R).3f maxI %(max_I).2f maxC %(max_C).3f' % r,
                  flush=True)
    with open(os.path.join(HERE, 'run_meta_phi.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('ALL DONE in %.1fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
