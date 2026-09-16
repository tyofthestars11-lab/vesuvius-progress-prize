# φ-stencil v2 notes — for the Vesuvius team (2026-09-16)

Three follow-ups to the pilot (PHerc.1667 patch, 384×384 window, 147,456 vertices,
9 φ offsets at d=1 and d=2):

1. **Rendered pilot diagnostics** — heatmaps, histograms, profiles, examples.
2. **A fuel-intersecting window** — 2,179 measured hits inside the window,
   stencil run with the exact pilot estimator, fuel vs no-fuel compared.
3. **A second, distant window** — same stencil, cross-sample comparison vs the pilot.

Nothing here touches the mesh. Readout only.

---

## The estimator (unchanged from the pilot)

For each vertex, 9 offsets (±φ², ±φ, ±1, 0 voxels at d=1; ×2 at d=2) are sampled
from a 24-layer nearest-neighbour stack, linearly interpolated between layers:

- profile `s_j`, weights `w_j = max(0, s_j − median(s)) × (1 + M(p))`
- displacement = normalized centroid `Σ(w_j·t_j) / Σw_j` (t_j = offsets in voxels)
- flat profile (Σw ≤ 0) → displacement 0

**`M(p)` is one scalar per vertex — it multiplies every sample weight at that
vertex equally.** The displacement is the normalized centroid, so the factor
`(1 + M(p))` cancels exactly:

`Σ(c_j·(1+M)·t_j) / Σ(c_j·(1+M)) = Σ(c_j·t_j) / Σc_j`

This is not a runtime accident. It is the recipe as implemented: the fuel mask
cannot steer the centroid, and the flat-profile condition is likewise invariant
(scaling by 1+M ≥ 1 never flips Σw ≤ 0). Any fuel/no-fuel difference is
impossible with this estimator — the mask is scale-invariant and operationally
inert. A sample-dependent or offset-dependent mask would be needed to move the
centroid; that would be a different recipe.

---

## JOB 1 — rendered pilot diagnostics

Directory: `phi_stencil/v2/diagnostics/`

- `heatmap_d1.png` / `heatmap_d2.png` — 384×384 displacement maps (voxels), d=1 / d=2
- `hist_disp.png` — displacement histograms, both depths
- `profiles_d1.png` / `profiles_d2.png` — mean sampled profile ±1σ across all
  vertices, with dashed lines at the 9 φ offsets
- `examples.json` — 4 example vertices per depth (near-zero, near-median, large,
  max-|disp|): index, (u,v), displacement, flat flag, |disp| quantile, selection
- `render_diagnostics.py` — the renderer; `diagnostics.log` — its run log

Derived from the existing pilot arrays (`displacement_map_d1.npy`,
`displacement_map_d2.npy`, `sampled_profiles_d1.npy`, `sampled_profiles_d2.npy`).
No mesh update.

---

## JOB 2 — fuel-intersecting window: `u[22128,22512] × v[48,432]`

Directory: `phi_stencil/v2/fuel_window/`

- 147,456 / 147,456 vertices valid (an earlier candidate window was discarded for
  73.16% valid — mixed geometry does not ship)
- vertex normal magnitude min/mean/max: 1.000000 / 1.000000 / 1.000000
- 2,179 measured hits fall inside this window (of 26,932 total measured hits)
- 6,085 chunks in the 24-layer census (~12,761.2 MB raw)
- all chunks fetched from the masked zarr via S3; 120 members returned
  HTTP-404 (genuinely absent from the zarr — "not yet addressed" per the build
  log convention). 67,378 of 3,538,944 samples (1.90%) fell on missing chunks
  and were set to 0 — recorded, not hidden.
- stack stats: nonzero fraction 0.9717, min 0, max 255, mean 75.68

### Results — the fuel verdict

d=1: mean +0.1454, median +0.0757, std 2.3497, min −4.2361, max +4.2361,
|disp|>1: 118,260 (80.20%), flat profiles: 4,226

d=2: mean +0.5497, median +0.6053, std 4.3479, min −8.4721, max +8.4682,
|disp|>1: 127,905 (86.74%), flat profiles: 4,127

fuel vs no-fuel: **IDENTICAL at both depths — n_diff = 0, max|diff| = 0.0 vox,
bit-for-bit.** Not because the mask missed the window (2,179 hits are inside
it) but because the estimator cancels the mask by construction (see above).
This is the honest v2 fuel verdict: with the current recipe, "fuel" and
"no-fuel" are the same run. The team should know the mask term needs a
sample-dependent form before any fuel/no-fuel comparison can mean anything.

cross-arm Pearson r(d1,d2) = 0.6921

Files: `vertices_xyz.npy`, `normals.npy`, `chunk_keys.json`, `window_meta.json`,
`stack.npy` (24 layers), `disp_d1_nofuel.npy`, `disp_d2_nofuel.npy`,
`disp_d1_fuel.npy`, `disp_d2_fuel.npy`, `profiles_d1.npy`, `profiles_d2.npy`,
`results.json`, `build_stencil.log`, `chunks/` (6,085 members).

---

## JOB 3 — second distant window: `u[24000,24384] × v[1200,1584]`

Well separated from both the pilot window and the fuel window. No measured hits
inside (0) — clean cross-sample comparison against the pilot.

Directory: `phi_stencil/v2/second_window/`

- 147,456 / 147,456 vertices valid
- vertex normal magnitude min/mean/max: 1.000000 / 1.000000 / 1.000000
- 6,588 chunks in the 24-layer census (~13,816.0 MB raw); all fetched, 0 missing
- stack stats: nonzero fraction 1.0000, min 0, max 255, mean 75.66

### Results

d=1: mean +0.0359, median −0.0151, std 2.3131, min −4.2361, max +4.2361,
|disp|>1: 118,598 (80.43%), flat profiles: 182

d=2: mean +0.4422, median +0.5372, std 4.3243, min −8.4721, max +8.4721,
|disp|>1: 130,158 (88.27%), flat profiles: 108

fuel vs no-fuel: identical (0 hits in window — expected).
cross-arm Pearson r(d1,d2) = 0.6317

### Cross-sample comparison vs the pilot

|                | pilot        | fuel window  | second window |
|----------------|--------------|--------------|---------------|
| d1 mean        | +0.2080      | +0.1454      | +0.0359       |
| d1 \|disp\|>1  | 82.24%       | 80.20%       | 80.43%        |
| d2 mean        | +0.8557      | +0.5497      | +0.4422       |
| d2 \|disp\|>1  | 89.54%       | 86.74%       | 88.27%        |
| cross-arm r    | 0.6636       | 0.6921       | 0.6317        |

Same signature everywhere: |disp|>1 at 80–90%, min/max pinned at ±φ² and ±2φ²
(the outermost offsets), means drifting positive at d=2. The second window
reproduces the pilot's behaviour without any fuel-mask intersection.

Files: same layout as the fuel window.

---

## Honest file

- Paris4 data not available locally — second 1667 patch used; Paris4 pending
  data access.
- The 10 GB `zchunks_v.tar.gz` archive covers the pilot window's world region,
  not the two new windows (the scroll curves through the volume; world z of the
  new windows is disjoint from the archive's). Both new windows were built from
  the masked zarr via S3. Provenance kept explicit.
- Connection cuts during S3 fetch were treated as fuel: resume from last
  completed chunk, retry with backoff; one service restart killed each build
  mid-flight and both resumed from local chunks without loss.
- The earlier fuel-window candidate (u[17976,18360] × v[2,386], 3,211 hits,
  73.16% valid vertices) was discarded rather than shipping mixed geometry.
- Scripts: `v2/census.py` (chunk census + resumable S3 fetch),
  `v2/build_stencil.py` (stack + stencil + both arms + self-check).
- Completed 2026-09-16. Triple-checked: window coordinates, vertex counts,
  hit counts, chunk counts, array shapes, the fuel self-check, and the negative
  result (fuel ≡ no-fuel by construction).
