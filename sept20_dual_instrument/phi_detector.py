#!/usr/bin/env python3
"""Pure-phi ink detector instrument — PHerc.1667 level-1 staged window.

No trained weights, no neural net, no downloads. The instrument:

1. PHI-WEIGHTED PREDICTIVE RESIDUALS: predict each voxel from
   phi-weighted spatial neighbors and the phi-predicted previous
   slice (inter-slice delta coherence, the KRAKEN-TON618 engine).
   Ink (carbon-rich, deliberately deposited structure) breaks
   phi-prediction differently than natural papyrus substrate, so the
   residual structure is the signal.
2. RUNG-ADDRESSED CONTRAST: local gradient magnitudes G are mapped to
   rung addresses a = ln(G)/ln(phi). Voxels whose addresses land near
   integer rungs hold phi-coherent edge structure.
3. INK SCORE: K = I * C — standardized absolute residual (incoherence)
   times rung coherence — re-standardized over the interior to K~.

Deterministic by construction: the 5 orientation seeds verify
orientation-equivariance; the shuffled controls carry the
discrimination (a data-intrinsic signal must beat the control
envelope).

All originals are read-only; this module never writes inputs.
"""
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
LN_PHI = np.log(PHI)
EPS = 1e-6
COH_SIGMA = 0.08          # rung-address coherence width
MARGIN = 8                # edge columns/slices excluded (morning edge-stripe zone)


def interior_mask(shape, m=MARGIN):
    H, W, D = shape
    msk = np.zeros(shape, dtype=bool)
    msk[m:H - m, m:W - m, m:D - m] = True
    return msk


def _cat(a, b, axis):
    return np.concatenate([a, b], axis=axis)


def phi_predict(V):
    """V: (H,W,D) float32. Returns P: phi-predicted volume, same shape."""
    # edge-replicated neighbors
    ym = _cat(V[:1], V[:-1], 0)      # y-1
    yp = _cat(V[1:], V[-1:], 0)      # y+1
    xm = _cat(V[:, :1], V[:, :-1], 1)  # x-1
    xp = _cat(V[:, 1:], V[:, -1:], 1)  # x+1
    zm1 = _cat(V[:, :, :1], V[:, :, :-1], 2)          # z-1
    # z-2 with edge replication: [V0, V0, V0, V1, ..., V(D-3)]
    zm2 = np.empty_like(V)
    zm2[:, :, 0] = V[:, :, 0]
    zm2[:, :, 1] = V[:, :, 0]
    zm2[:, :, 2:] = V[:, :, :-2]

    # spatial: phi weight on causal neighbors (y-1, x-1)
    S = (PHI * ym + yp + PHI * xm + xp) / (2.0 * PHI + 2.0)
    # temporal: phi-weighted previous slices
    T = (PHI * zm1 + zm2) / (PHI + 1.0)
    # combined phi prediction
    P = (S + PHI * T) / (1.0 + PHI)
    return P


def run_instrument(V, return_parts=False):
    """V: (H,W,D) float32 raw volume. Returns K~ detection map (float32).

    K~ is the standardized ink score: high where phi-prediction breaks
    (incoherence) AND gradient magnitudes sit at integer rungs
    (phi-coherent structure) — the read of deliberate deposited ink.
    """
    V = np.asarray(V, dtype=np.float32)
    msk = interior_mask(V.shape)

    P = phi_predict(V)
    R = V - P
    sig_r = float(R[msk].std())
    if sig_r < 1e-9:                      # degenerate (uniform control)
        K = np.zeros(V.shape, dtype=np.float32)
        parts = {'sigma_R': 0.0, 'mu_K': 0.0, 'sigma_K': 0.0,
                 'max_I': 0.0, 'max_C': 0.0}
        return (K, parts) if return_parts else K
    I = np.abs(R) / sig_r                # incoherence

    # central-difference gradient magnitude
    ym = _cat(V[:1], V[:-1], 0); yp = _cat(V[1:], V[-1:], 0)
    xm = _cat(V[:, :1], V[:, :-1], 1); xp = _cat(V[:, 1:], V[:, -1:], 1)
    zm1 = _cat(V[:, :, :1], V[:, :, :-1], 2)
    zp = _cat(V[:, :, 1:], V[:, :, -1:], 2)
    gy = (yp - ym) * 0.5
    gx = (xp - xm) * 0.5
    gz = (zp - zm1) * 0.5
    G = np.sqrt(gy * gy + gx * gx + gz * gz)

    # rung addresses: fractional distance to nearest integer rung
    a = np.log(G + EPS) / LN_PHI
    f = a - np.floor(a)
    d = np.minimum(f, 1.0 - f)
    C = np.exp(-(d * d) / (2.0 * COH_SIGMA * COH_SIGMA))  # rung coherence

    K = (I * C).astype(np.float32)
    mu_k = float(K[msk].mean())
    sig_k = float(K[msk].std())
    Ks = np.zeros_like(K) if sig_k < 1e-9 else ((K - mu_k) / sig_k).astype(np.float32)

    parts = {'sigma_R': sig_r, 'mu_K': mu_k, 'sigma_K': sig_k,
             'max_I': float(I[msk].max()), 'max_C': float(C[msk].max()),
             'frac_C_gt_09': float((C[msk] > 0.9).mean())}
    return (Ks, parts) if return_parts else Ks
