# φ Ink Detector — PHerc.1667 — Full Report

**Date:** 2026-09-20. **Instrument:** pure φ, no trained weights, no neural net, no downloads.
**Data:** staged Level-1 PHerc.1667 window, 256×256×248 uint8, 4.8 µm/voxel, read-only from the
morning's verified staged copy. Cold data only; originals untouched.

## The instrument (exact)

For volume V (float32), with φ = (1+√5)/2:

1. **φ-weighted prediction.** Spatial (φ on causal neighbors y−1, x−1):
   S = (φ·V[y−1] + V[y+1] + φ·V[x−1] + V[x+1]) / (2φ+2).
   Temporal (inter-slice delta coherence): T = (φ·V[z−1] + V[z−2]) / (φ+1).
   Combined: P = (S + φ·T) / (1+φ). Residual R = V − P.
   Incoherence I = |R| / σ_R (σ_R over the interior, 8-voxel margins excluded).
2. **Rung-addressed contrast.** Gradient magnitude G from central differences.
   Rung address a = ln(G+10⁻⁶)/ln(φ); fractional distance to nearest integer rung
   d = min(frac(a), 1−frac(a)); coherence C = exp(−d²/(2·0.08²)).
3. **Ink score.** K = I·C, standardized over the interior → K~.
   Theory of the instrument: deposited carbon ink is deliberate structure — it breaks
   φ-prediction (high I) while its edges hold φ-coherent gradients (high C).

## Run configuration

- 5 orientation seeds: 137, 1001, 2026, 31337, 99991. Transform per seed: rot90 k∈{0..3}
  on (y,x) + optional x-flip (same draws as the morning run), output map inverse-transformed.
- Controls: shuffled-depth ×3 seeds, shuffled-spatial ×3 seeds, uniform ×1 (Stage C parity).
- Fixed threshold: K~ > 3.0. 12 runs, 2 workers, 13.6 s total. Maps in `maps/`.

## Key numbers

**Per-seed (real, interior):** max K~ 24.95–25.78, mean 0.00000, fraction > 3.0 = 0.02409
(all 5 seeds). σ_R = 18.93–18.94. Max incoherence I = 11.5σ.

**Orientation response:** max seed-pair |ΔK~| = 2.30 (on a ~25 scale). Diagonal-class pairs
(137 vs 31337) agree to 6×10⁻⁶ — the instrument is exactly equivariant under diagonal
reflection (the {−y,−x} causal φ-weighting is preserved by y↔x swap) and mildly
orientation-sensitive otherwise, by design: φ privileges causal directions. No moving
artifacts; nothing like the morning's edge stripe (no full-width stripe morphology found).

**Components (consensus > 3.0, 26-connectivity):** 118,018 components. Size histogram:
65,752 of size 1; 37,181 of size 2–4; 10,160 of size 5–9; 3,547 of size 10–19;
798 of size 20–29; 82 of size 30–316. Largest: 316 voxels.
Top-20 morphology: all roughly isotropic (dy≈dx≈dz, 10–29 voxels per axis),
fill fraction 1.5–4.5%, xy/z extent ratio 0.5–1.7. **Zero stroke-like components**
(stroke bar: fill ≥ 15%, xy/z ≥ 3, area ≥ 100).
40 wispy tail-filaments pass seed-stability (5/5) and the control screen but fail morphology.

**Controls (distributional):** shuffled-depth → 195,459 components, max size 60, max K~ 17.96,
frac>3 = 0.0266. shuffled-spatial → 201,867 components, max size 51, max K~ 23.93,
frac>3 = 0.0241. Real data yields FEWER but LARGER components (118k, max 316) — genuine
spatial coherence the shuffles destroy — but it takes wisp/fiber form, never stroke form.
Uniform control: K~ = 0 everywhere (instrument sanity holds).

**y=160 band (y 144–179):** K~ mean 0.062 vs background −0.0045; frac>3 0.0305 vs 0.0236;
max 24.68 vs 25.78. BUT the shuffled-depth control shows the same elevation (0.060):
the band is a y-spatial density feature, not depth-coherent ink. Raw data: band mean
intensity 64.47 vs background 59.45 (+8% denser — a denser papyrus layer/fold).
The morning trained detector independently cleared the band (below-background ink response).

**Deep interior (z 40–207, band excluded):** mean 0.0083, max 24.5–25.8, frac>3 0.0252 —
indistinguishable from global; nothing localized. Extreme voxels (K~>10: 15,129) are
spread proportionally through the volume (17.1% in band vs 15% band volume fraction).

## Bottom line

The φ instrument **does not detect ink** in PHerc.1667. It reads genuine substrate
φ-structure — wispy fiber/density filaments and the denser y=160 band layer — but nothing
with deposited-ink morphology: no strokes, no letters, no surface-bound structures, no
band-associated writing. The trained detector (morning) and the φ instrument (now) agree:
no ink detected; the y=160 band is substrate, not ink. Ink is not ruled out in the scroll —
both instruments simply do not see it.

## Files

- `phi_detector.py` — the instrument
- `run_phi.py` — 12-run harness (seeds, transforms, controls)
- `analyze_phi.py` — consensus, components, morphology, band/interior, verdict
- `maps/` — 12 float32 K~ maps (256×256×248)
- `run_meta_phi.json` — per-run stats
- `verdicts_phi.json` — full verdict data (components, band, interior, controls)
