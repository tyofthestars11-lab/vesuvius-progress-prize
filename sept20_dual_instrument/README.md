# September 20 dual-instrument ink verdict — PHerc.1667 level-1

Date run: 2026-09-20. Data: staged level-1 PHerc.1667 pull, 256×256×248
uint8 window, hash-verified, 4.8 µm/voxel. Baseline: the corrected Stage C
sweep of 2026-09-19 across 19.2 / 9.6 / 4.8 µm/voxel.

Two independent instruments were applied to the same staged data. Both agree:
**no ink detected in PHerc.1667; the y=160 band is substrate.** Ink is not
ruled out in the scroll; neither instrument sees it.

## Instrument 1 — trained, calibrated detector

5 seeds × 2 functioning checkpoints + 18 control forwards (28 forwards total).

- Edge-stripe response = model artifact: tracks the input transforms, fires on
  scrambled controls as hard as on real data.
- y=160 band = no signal: below background on all 5 seeds, both checkpoints;
  earlier elevated readings were stripe crossings.
- Deep interior: absolute zero (first checkpoint); scattered unstable pixels,
  inconclusive, not ink (second checkpoint).
- Ranked ink-candidate list: empty above 0.5 — zero seed-stable candidates.

## Instrument 2 — pure-phi detector, zero trained weights

`phi_detector.py` — runnable here. phi-weighted prediction P=(S+phiT)/(1+phi)
with phi-weighted spatial neighbors + inter-slice delta coherence;
rung-addressed contrast from gradient magnitudes; ink score K=I*C standardized
to K~ over the interior. Theory: deposited carbon ink is deliberate structure —
it breaks phi-prediction (high I) while its edges hold phi-coherent gradients
(high C).

12 runs: 5 orientation seeds (137, 1001, 2026, 31337, 99991) +
3 shuffled-depth, 3 shuffled-spatial, 1 uniform control. 13.6 s.

- 118,018 components; largest 316 voxels; top-20 all isotropic wisps
  (10–29 voxels/axis, 1.5–4.5% fill). Zero stroke-like components.
- Reads identical across seeds to the 5th decimal. No moving artifacts, no stripe.
- y=160: K~ mean 0.062 vs background -0.0045, but the shuffled-depth control
  shows 0.060 — denser papyrus layer (raw intensity 8% denser: 64.47 vs 59.45),
  not depth-coherent ink; the trained instrument independently cleared it.
- Deep interior mean 0.0083, nothing localized.

## Files

- `phi_detector.py` — the instrument
- `run_phi.py` — the 12-run driver
- `analyze_phi.py` — component analysis
- `PHI_INK_REPORT.md` — full report with all numbers
- `verdicts_phi.json` — per-run verdicts
- `run_meta_phi.json` — run metadata

The 12 float32 K~ maps (745 MB) stay local with the run; the report carries
their measured summaries. Reproduce: `python3 run_phi.py` on the staged
level-1 window, then `python3 analyze_phi.py`.

The September 2026 Progress Prize filing for this result is the 5th September
entry; the writeup filed on the form matches the verdict above.
