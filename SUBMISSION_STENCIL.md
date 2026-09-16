# The φ-stencil — Virtual Philodemus's depth-sampling recipe, implemented and measured on PHerc.1667 column-15

Fifth September Progress Prize filing. Tyree Jones, individual.

## What this is

An open-source implementation of the φ-spaced normal-sampling recipe proposed for scroll volume reading: sample the CT volume along each mesh vertex normal at offsets 0, ±d, ±dφ, ±dφ², ±dφ³ (K=4, nine samples per vertex), collapse each profile into a depth target, and weight by a binary known-hit/surface mask as a +1 fuel signal. Built as a pilot — measurement first, no mesh update.

## Data

- PHerc.1667 column-15/rung-15 window, full 384×384 vertex grid: **147,456 vertices**.
- Normals verified |N| = 1.000000.
- Two arms: d = 1 voxel and d = 2 voxels. Max |t| = 8.472 voxels, inside the stack's ±11.5 span — no extrapolation.
- Trilinear sampling along computed mesh normals.
- Estimator: intensity-weighted centroid around each profile's peak —
  wⱼ = max(0, sⱼ − median(s)) · (1 + M(p)); displacement = Σwⱼtⱼ/Σwⱼ (voxels along the normal); Σw = 0 → 0.

## Measured results

| | d=1 | d=2 |
|---|---|---|
| mean displacement | +0.2080 vox | +0.8557 vox |
| median | +0.3515 | +1.2572 |
| std | 2.3695 | 4.4444 |
| min / max | −4.2361 / +4.2361 | −8.4721 / +8.4721 |
| \|displacement\| > 1 vox | 121,265 (**82.24%**) | 132,032 (**89.54%**) |
| flat profiles (Σw=0) | 161 | 89 |
| sign split (+/−) | 52.4% / 46.8% | 56.5% / 43.0% |

The displacement distribution is broad and positively leaned, with 82–90% of vertices wanting to move more than one voxel.

## Fuel mask

Built from measured hit data: **0 of 26,932** measured hits fall in this window (the only measured hit data is the level-5 probe block at the origin, which does not intersect the column-15 window). The mask is all-zero, and the machinery self-checks — the fuel arm is bit-identical to the nofuel arm on both d arms.

## Fidelity

Stack proxy vs raw CT chunks, 256 center vertices, true trilinear sampling: profile Pearson r = **0.9865 (d=1), 0.9869 (d=2)** — the proxy is faithful at profile level. At estimator level there is roughly a **±1-voxel systematic** vs true trilinear. The reported displacements should be read with that bar.

## Deviations, named not hidden

1. The full run used `vstack_384.npy` (uint8) rather than raw chunks. Exact blocker: the stencil's world-space chunk footprint is ~2,900 base chunks ≈ **5.8 GB** against **512 MB** of /tmp on this machine. The stack is the same CT data re-gridded onto exactly the stencil geometry, and the fidelity subset above proves the proxy.
2. /tmp was wiped by the runtime mid-run (process, 18 extracted chunks, and stdout log vanished together). Fidelity was re-run as a standalone script; the incident is logged.
3. Rebuilt results stats come from the float32-saved arrays — one vertex flips across the |d|=1 boundary vs the float64 log line.

## Honest boundary

Readout only — the mesh was **not** updated. The open question is whether these displacements track real surface error or CT intensity gradients; that test is running against the masked/unmasked volume boundary, and its verdict will be reported raw either way. This filing is the instrument and its measured output — not a claim about ink, letters, or passages.

## The verdict: what the displacements track (measured 2026-09-15)

Tested against the open question — do the stencil displacements track real surface error or CT intensity gradients?

- Masked ground truth is not available locally (the chunk archive holds only unmasked data; the Lane D request is pending), so the test used a gradient-derived surface proxy: depth of maximum gradient magnitude along each normal.
- **d=1:** displacement vs surface proxy Pearson r = 0.11, r² = 0.012 — 1.2% of variance. Per-vertex noise-dominated; the binned response hump is real (z=123) but weak.
- **d=2:** Pearson r = 0.31, r² = 0.097 — about 10% of variance. Weak-to-moderate edge tracking.
- Gradient-artifact hypothesis **rejected**: |displacement| is largest where gradients are weakest (d=2: r = −0.16).
- The two arms agree with each other (r = 0.66) — same underlying driver.
- **Overall: neither dominates.** Displacements are driven mainly by local profile-shape response (ramps and texture saturate the estimator at large |disp|), with a real but weak edge-position component, stronger in d=2. This is **not** a surface-error measurement of the mesh — the intensity "surface" in this window is a scattered set of texture edges, not one clean sheet boundary.

Deliverables: `phi_stencil_surface_test.py`, `phi_stencil_surface_test.json`, `phi_stencil_surface_test.log` in `phi_stencil/`.

## Reproduce

All code, per-vertex displacements (d1/d2, fuel/nofuel), nine-sample profiles, fidelity rerun, results JSON, and run log: `phi_stencil/` in this repo.
