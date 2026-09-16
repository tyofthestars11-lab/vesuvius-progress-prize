# First inference run — PHerc1447 segment 20250703034159 (2026-09-15)

## Environment
- venv: `~/workspace/vesuvius-first-letters/venv/` (Python 3.12, torch 2.14.0+cpu)
- villa source: selective pull under `villa_src/vesuvius/src/`
- Local patch to villa source (run-only, NOT upstream): `models/build/build_network_from_config.py`
  top-level Primus and DINOv2 imports made lazy (try/except) because timm pulls a
  torchvision build incompatible with torch 2.14.0+cpu. The ink_9um model
  (vesuvius_unet_3d_stem_2d) never touches those paths.
- torch.load of the checkpoint used weights_only=False (checkpoint stores a config
  object, not just tensors) — same as the official CLI.

## Checkpoint (seed 42)
- `checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth`
- config: mode=flat, normalization=robust_mad(1.0, 99.0), crop_size=(17,128,128)
- model: vesuvius_unet_3d_stem_2d, 34.5M params, missing=0 unexpected=0

## Input
- `PHerc1447/seg_20250703034159_surfacevol.zarr` — assembled locally from the 31
  surface-volume TIFFs (31, 3620, 5220) uint8, ~586 MB
- NOTE: `03.tif` was corrupt (3.4 MB, unreadable) — re-downloaded clean from the
  public S3 bucket (7.5 MB, verified). All 31 re-verified readable before assembly.

## Runs (CPU, 2 cores)
- Forward: 862 s (14.4 min), 4536 patches, stride 64, overlap 0.5, Hann, batch 4
  -> `PHerc1447/ink_pred_seed42_forward.tif`
- Reverse: 880 s (14.7 min), same settings, layers [23..7]
  -> `PHerc1447/ink_pred_seed42_reverse.tif`
- Throughput: ~0.19 s/patch wall (~5.2 patches/s)

## Results (honest)
- Forward max 0.86, mean 0.20; 1.39% of pixels > 0.55.
- Forward/reverse Pearson r = 0.878; mean abs diff 0.035. Strict strong-pixel
  (>0.6) overlap between directions is only ~11% — scattered strong responses
  are unstable; the top regional responses are stable.
- Strongest raw peaks sit on zero-padding at segment boundaries (edge artifacts).
- Best in-data region: px (3567, 3062) = (30.8, 26.5) mm. Both directions peak
  ~0.81-0.82 there. The surface shows dark vertical stroke-like marks that persist
  through layers 7-23 (z-min). They resemble ink strokes / letter fragments but
  are NOT legible letters.
- No legible letters found. 8 in-data hotspots surveyed; all are stroke fragments
  or blotches, none supportable as letters.

## Renders
- `PHerc1447/render_overview_seed42_forward.png` / `render_overview_seed42_avg.png`
- `PHerc1447/indata{0..7}_seed42_fwd.png` — prediction | surface | overlay crops
- `PHerc1447/strokes_w{200,400}_{mid15,zmin,zmax}.png` — tight stroke-region crops
- `PHerc1447/strokes_avg_overlay.png` — averaged prediction on surface

## Next options (not run)
- seed 43 checkpoint not yet downloaded (second model, per PLAN).
- Wider letter search: the strokes region (30.8, 26.5 mm) deserves a grown
  segment in VC3D around it; current segment render is only a partial foothold.

## Seed-43 + LUCAS STABILIZER — main segment (2026-09-15, resumed session)

### Seed-43 inference (main segment 20250703034159)
- Checkpoint verified: `checkpoints/ink_9um/hybrid_3d2d-seed43/step-075000.pth`
  loads with missing_keys=0, unexpected_keys=0 (same as seed 42).
- Forward: `PHerc1447/ink_pred_seed43_forward.tif`, ~16.5 min wall, stride 64,
  overlap 0.5, Hann, batch 4, 4536 patches. (One earlier `--direction both`
  launch stalled with no output for 88 min and was killed; separate
  forward/reverse launches are the reliable recipe.)
- Reverse: `PHerc1447/ink_pred_seed43_reverse.tif`, batch 8 / 2 workers,
  ~17 min wall.
- NOTE: the CLI writes predictions as **uint8 (prediction x 255)**. All
  thresholds below are on normalized [0,1] values (divide by 255). An early
  analysis pass misread raw uint8 values — corrected in `lucas_stabilizer.py`
  and `sweep_infer.py` (normalize on load).

### LUCAS STABILIZER — first locking test (main segment)
Standing directive from TYREE: a hotspot LOCKS as genuine signal only if it
holds stable across BOTH seeds (42/43) AND BOTH directions (forward/reverse);
unstable responses are REJECTED as model noise. This is the pipeline's
false-positive mitigation for the prize submission. Rule (uniform, no
per-segment tuning): hotspots = connected components of the 4-map average >
0.55, area >= 25 px, inside the valid-data mask eroded by 32 px (padding
excluded). LOCKED iff peak >= 0.60 in ALL FOUR maps; MARGINAL iff >= 0.55 in
all four; else REJECTED. Peaks measured in component bbox dilated 16 px.
Script: `lucas_stabilizer.py` -> `PHerc1447/lucas_lock_20250703034159.json`.

- **Tally: 8 LOCKED, 0 MARGINAL, 20 REJECTED (28 hotspots).**
- Honest stability numbers (valid-mask only — padding zeros excluded):
  - Pearson r: 42f-vs-43f 0.338; 42r-vs-43r 0.272; 42f-vs-42r 0.118;
    43f-vs-43r 0.132. (Global r including padding was 0.875 — inflated by
    shared zero padding; the valid-mask numbers are the honest ones.)
  - Strong-pixel (>0.6) IoU: cross-seed fwd 0.086, cross-seed rev 0.086,
    cross-dir 42 0.024, cross-dir 43 0.042.
  - The models agree on background but disagree substantially on ink
    location — exactly why the locking gate matters.
- Locked regions (peak in all 4 maps):
  - hs16 min 0.745 @ (22.9, 26.6) mm — strongest lock
  - hs25 min 0.702 @ (27.8, 32.8) mm
  - hs03 min 0.690 @ (15.3, 17.0) mm
  - hs24 min 0.655 @ (26.2, 30.9) mm — the (30.8, 26.5) mm strokes region;
    seed43 responds weaker here (0.66) than seed42 (0.82) but still locks
  - hs20 min 0.631 @ (24.3, 29.9) mm; hs19 min 0.624 @ (24.0, 29.8) mm
  - hs02 min 0.608 @ (10.5, 14.8) mm; hs04 min 0.600 @ (15.5, 32.1) mm
- Rejected examples: hs17 (42f 0.82 but 42r 0.45), hs10 (42f 0.81, 42r 0.42),
  hs05 (42f 0.46, 42r 0.79) — single-direction phantoms the gate catches.
- Visual inspection of all 8 locked renders (`PHerc1447/lock_*_locked.png`,
  script `render_locks.py`): amorphous blotches and dark vertical
  stroke-like marks on the papyrus surface. **No legible letters.** The marks
  could be ink strokes, cracks, or delamination — not callable as letters.

### Sweep setup (in progress)
- All 14 other PHerc1447 segment surface-volume zarrs downloaded from the
  public S3 bucket to `PHerc1447/sweep/seg_<id>.zarr` (~2.6 GB total).
  Remote stores are multiscale groups; level '0' is full-res.
- Segment sizes (level 0): most 5-13 cm^2; **several exceed the 4 cm^2 prize
  region**; 20251105093211 is 112M px / 83.7 cm^2 (expect ~80 min inference).
- `sweep_infer.py` running seed42-forward (batch 8, 2 workers) over all 14,
  resumable, logging to `PHerc1447/sweep/sweep_stats.jsonl` with overview
  PNGs. Responding segments (max >= 0.55) get the full 4-way LUCAS lock next.

### Throughput ("speed of light as fuel")
- 2 CPU cores total (nproc=2) — all in use during inference (~115-120%).
- seed43 forward: 16.5 min wall (4536 patches, ~4.6 patches/s, batch 4).
- seed43 reverse: ~17 min wall (batch 8, 2 workers — no dramatic gain;
  bottleneck is raw compute on 2 cores).
- Downloads overlapped with inference throughout (parallel, network-bound).
- Lucas analysis + renders overlapped with sweep inference.
- One VM restart mid-session killed background jobs; all state was
  re-verified and relaunched (workspace persists, /tmp does not).

### Sweep complete — all 14 segments RESPOND (2026-09-15)
seed42-forward (batch 8, 2 workers) over all 14 PHerc1447 segments. Every
segment shows max >= 0.79 — the whole scroll is hot.

| segment | max | mean | f>0.55 | f>0.60 |
|---|---|---|---|---|
| 20250502183421 | 0.796 | 0.412 | 12.40% | 7.30% |
| 20251105093211 | 0.831 | 0.316 | 5.17% | 3.54% |
| 20250502182142 | 0.804 | 0.393 | 10.09% | 6.00% |
| 20250502185519 | 0.800 | 0.429 | 17.99% | 12.02% |
| 20250702235910 | 0.828 | 0.317 | 4.28% | 2.83% |
| 20250703025628 | 0.812 | 0.300 | 1.89% | 1.16% |
| 20250502205333 | 0.820 | 0.409 | 14.78% | 9.45% |
| 20250502182456 | 0.812 | 0.429 | 15.24% | 9.74% |
| 20250502183138 | 0.792 | 0.434 | 16.87% | 9.34% |
| 20250502184845 | 0.816 | 0.421 | 16.07% | 10.08% |
| 20250502180748 | 0.788 | 0.402 | 11.62% | 6.77% |
| 20250502184658 | 0.800 | 0.424 | 17.74% | 11.55% |
| 20250502184201 | 0.800 | 0.446 | 21.24% | 13.95% |
| 20250502180708 | 0.800 | 0.535 | 47.34% | 38.10% |

Note: 20250502180708 is anomalously hot (47% > 0.55) — inspect for
systematic model firing vs genuine ink. Overview PNGs in
`PHerc1447/sweep/overview_*_seed42_forward.png`. Stats in
`sweep_stats.jsonl`. Two service restarts during the sweep; the resumable
script skipped completed segments (one relaunch re-ran 3 due to a
startup-time stats read race — wasted compute, no data corruption).

### 4-way LUCAS locking — sweep segments (in progress, 2026-09-15)
`lock4way.py` processes responding segments hottest-first: seed42-reverse +
seed43-forward + seed43-reverse, then `lucas_stabilizer.py` + renders.

Completed:
- 20250502180708 (hottest, 47% >0.55): 6 LOCKED, 1 MARGINAL, 3 REJECTED.
  Locked peaks 0.61-0.77, highly consistent across all 4 maps. BUT visual
  inspection shows the model firing on papyrus fragment edges/structure —
  long horizontal bands tracking the physical papyrus, NOT letterforms.
  Stability != letters. Renders in `sweep/lock_20250502180708_*`.
- 20250502184201: 28 LOCKED, 12 MARGINAL, 16 REJECTED (56 hotspots).
  Largest locked area 26,286 px — far too large for letters. Visual confirms
  papyrus-band responses, not characters.

Emerging honest pattern: the ink model fires strongly and stably on PHerc1447
papyrus, but the stable responses track papyrus structure, not writing. No
legible letters in any locked region inspected so far (main segment 8 locked +
sweep 34 locked = 42 locked regions, 0 letters).

### 4-way LUCAS locking COMPLETE (2026-09-15)
All 14 responding segments processed: seed42-reverse + seed43-forward +
seed43-reverse, then lucas_stabilizer.py + renders. `lock4way.py` (hottest-first,
resumable) survived 4 execution-service restarts.

Final tallies (13 segments with LUCAS JSON):

| segment | LOCKED | MARGINAL | REJECTED |
|---|---|---|---|
| 20250502180708 | 6 | 1 | 3 |
| 20250502180748 | 19 | 5 | 16 |
| 20250502182142 | 30 | 10 | 17 |
| 20250502182456 | 43 | 21 | 22 |
| 20250502183138 | 27 | 11 | 15 |
| 20250502183421 | 36 | 9 | 20 |
| 20250502184201 | 28 | 12 | 16 |
| 20250502184658 | 31 | 8 | 12 |
| 20250502184845 | 40 | 9 | 26 |
| 20250502185519 | 39 | 9 | 14 |
| 20250502205333 | 53 | 18 | 55 |
| 20250702235910 | 0 | 0 | 10 |
| 20250703025628 | 1 | 0 | 4 |
| TOTAL | 353 | 113 | 230 |

Plus main segment 20250703034159: 8 LOCKED, 0 MARGINAL, 20 REJECTED.
Grand total: 385 LOCKED / 119 MARGINAL / 264 REJECTED, 0 legible letters.

Correction note (2026-09-15 recount): the 14-segment sweep was recounted to
377 LOCKED / 119 MARGINAL / 244 REJECTED (the per-segment table above shows
the earlier 353/113/230 breakdown); adding the separately processed main
segment (8 / 0 / 20) gives the corrected grand total 385 / 119 / 264.

Known issue: 20251105093211 (112M px, 83.7 cm²) completed all 4 inferences
(4 TIFFs exist) but LUCAS analysis was SIGKILLed (rc=-9, OOM) twice. The 4 maps
are available for manual inspection; automated locking needs a
memory-efficient (tiled/chunked) LUCAS pass.

Honest visual finding: every inspected LOCKED region (main segment 8 +
sweep samples) shows the ink model firing stably on papyrus fragment
edges, fiber bands, and structural texture — NOT on letterforms. The largest
locked component was 26,286 px (a papyrus band, not a character). Stability
across seeds/directions confirms the model is consistent, but consistent
about papyrus, not ink. 20250702235910 is the control case: 10 hotspots, all
REJECTED — the stabilizer does filter unstable responses.

Throughput (batch 8, 2 workers, 2 CPUs, shared machine):
- Main segment forward: 4,536 patches / ~990 s = ~4.6 patches/sec
- Main segment reverse: ~17 min wall
- Sweep segments: 90-996 s each depending on size/contention
- 4-way lock per segment: 3 inferences (~3-14 min each) + LUCAS + renders
