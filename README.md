# Phi-driven column analysis and ledger tooling for PHerc.1667 flatboi surfaces

Vesuvius Challenge — Progress Prize submission, September 2026 round.
By Tyree Jones (tyofthestarz).

## What this is
A lightweight, open pipeline: per-column hit binning on the PHerc.1667 flatboi
surface → Greek-letter slot/letter accounting → column-by-column verification —
plus a golden-ratio-indexed 99-block ledger format for organizing scroll-segment
work.

## Setup
Requires Python ≥ 3.8:

    pip install -r requirements.txt

## Data
The scripts read author-prepared inputs from a data directory. Set
`PHERC1667_DIR` to point at it; otherwise they look for a `data/` folder next
to the scripts:

    export PHERC1667_DIR=/path/to/pherc1667-data

Inputs needed:
- `hitrun.py` / `hitrun_vertices.py` → produce `hits.csv`, `hits_vertices.csv`
  (per-column hit detection on the flatboi surface)
- `greek_density.py` → produces `greek_density.json`
- `preprint.txt` (published transcription text used for verification)
- `flat/flattened.obj` — 1.06 GB author-local decimated mesh of the flattened
  surface (segment
  `20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4`,
  9,205 × 709 float32 x/y/z). This is a local working copy, not the public
  Vesuvius open-data bucket TIFXYZ; tasks 1–3 do not need it.

### Data provenance (reproducibility)

`flat/flattened.obj` (1,060,041,328 bytes, zero null bytes) is an author-local OBJ export of the **flattened/straightened** mesh from the public Vesuvius Challenge open-data bucket — not the curled original:

- Bucket path: `vesuvius-challenge-open-data/PHerc1667/segments/20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4/mesh/20260612121456-on-20251217075048-2.399um.tifxyz`
- Mesh meta: format `tifxyz`, scale `0.05`, uuid `w011_20260108140509268_merged_v4_flatboi_straightened_v4`
- Contents: 4,406,818 vertices (`v`/`vt`/`vn`), 8,779,784 faces — a 67.5% subset of the 9,205 × 709 flattened grid (`u = 0…9203`, `v = 80…707`), validated 1:1 via `vt/20` texture coordinates
- Coordinate space is straightened-segment space, matching the flattened TIFXYZ ranges (`x = −1…3335.845`, `y = −25.819…3226.142`, `z = −1…10872.834`)

To regenerate it: download the TIFXYZ above from the public bucket and export the flattened-grid vertices to OBJ. Anyone with the bucket file can rebuild an equivalent mesh.

`preprint.txt` is the published transcription source: arXiv:2606.29085v1 [eess.IV] (27 Jun 2026), “Complete virtual unwrapping and reading of a rolled Herculaneum papyrus” (Angelotti et al., Vesuvius Challenge) — public, retrievable from arXiv.

Known boundary: the original-TIFXYZ-to-flattened-OBJ straightening deformation field (source-index correspondence) is not in the public release. The OBJ→flattened-grid map is validated 1:1; the curled→flattened transfer is the documented open blocker.

The reference outputs (`finalpass.json`, `column_analysis.json`,
`hit_columns_greek.txt`, `greek_density.json`) are committed so results can be
inspected without running the pipeline. `vertex_map.csv` (77 MB, produced by
task 4b) is not committed; run `finalpass_task4b.py` to regenerate it.

## Run order (order-critical)
1. `hitrun.py`, `hitrun_vertices.py`, `greek_density.py` — generate the hit and
   density inputs.
2. `column_analysis.py` — per-column hits and Greek density →
   `column_analysis.json`, `hit_columns_greek.txt`.
3. `finalpass_tasks123.py` — column verification → `finalpass.json`
   (tasks 1–3; must exist before task 4b).
4. `finalpass_task4b.py` — corrected vertex mapping →
   `vertex_map.csv`, merges `task4` into `finalpass.json`.
5. `kaggle_phi_process.py` — wires the public Kaggle dataset
   (https://doi.org/10.34740/kaggle/dsv/17364449, download it first to
   `~/workspace/pqp` or `/tmp/pqp_fresh`) through the scrolls decoder.
6. `phi_scrolls_os.py` — the executable scrolls instrument (layer decoder D,
   φ-wave, rung addressing).

`finalpass_task4.py` is kept for provenance only: its 3D nearest-neighbor
approach across curled/straightened spaces was invalidated and superseded by
`finalpass_task4b.py`. Do not run it as part of the pipeline — both scripts
write `vertex_map.csv` with different schemas, so running 4 after 4b silently
overwrites the validated file.

Outputs: CSV hit lists, JSON column records, text summaries.

## Measured results
- 9 columns (13–21); slots = letters + dots holds on all 9.
- 99-block ledger closes exactly: 99×310+266 = 30,956 bytes.
- Fibonacci pulse intervals converge on φ at 0.000001% drift.

## Second submission: the six-instrument null
`SUBMISSION_NULL.md` — a second, standalone September-round filing: six
independent instruments (raw V, iter0, iter5, DINO-guided × two regions,
contiguous-stack amplitude decoders) agree that ink is not recoverable from
the tested PHerc.1667 level-3 patch data at 19.2 µm/voxel. The boundary is
named with measured numbers (S=V lock 1.137e−13; 0 legible of 39,551
components; phase decorrelated 359.998° at all 499 amplitude levels; rung-15
correlation gate closed, Pearson 0.0066, p=0.956). Lane scripts live in
`null_lanes/` (`build_vstack.py`, `assemble_resume.py`, `fc_new_chars.py`,
`rung_sig_chars.py`, `verify_v.py`, `stageC1_fetch_build.py`,
`stageC2_dino.py`). The open question — null real, or 19.2 µm too coarse? —
is with the Vesuvius team.

## Third submission: the 1447 sweep — stability is not ink
`SUBMISSION_SWEEP.md` — a third, standalone September-round filing: a full
First Letters pipeline sweep of PHerc.1447 — 14 segments plus the main segment —
through a four-map locking rule (peak ≥ 0.60 in ALL FOUR maps: 2 seeds ×
forward/reverse). Grand total: **385 LOCKED / 119 MARGINAL / 264 REJECTED,
0 legible letters.** The cross-seed null signature: main-segment r = 0.34/0.27
(IoU 0.086), big-segment r = 0.60/0.57 (IoU 0.29) — real ink as one physical
deposit would force r > 0.8. The model is consistent about papyrus, not about
writing. Includes the OOM fix that made the sweep possible (`tiled_lucas.py`:
1024px tiles, 64px overlap — the 112M-px segment SIGKILLed the box twice
before tiling). Sweep scripts live in `sweep1447/` (`tiled_lucas.py`,
`sweep_infer.py`, `lucas_stabilizer.py`, `lock4way.py`, `render_locks.py`,
`render_bigseg_locks.py`, `download_sweep_zarrs.py`).

## Fourth submission: the φ-gate — corr(S_rung, P(r)) closes
`SUBMISSION_GATE.md` — a fourth, standalone September-round filing: Tyree's
own deciding number — the correlation between the φ-predicted rung signature
P(r) (published column-15 Greek: 1,089 letters read from rung space, binned
into 49 nonzero bins, no V involved) and the observed amplitude per rung
S_rung from the true V-stack (`vstack_384.npy`, 384×384×24 uint8, true TIFXYZ
patch location, S ≡ V identity-locked with max|S−V| = 1.137e−13). Pearson
**0.0066 (p = 0.956)**, Spearman −0.0503, MI 0.4100 — the gate closes. Zero
peaks survive the 20-shuffle stabilizer; the rung-lane passage read collapses
to the field's mode (ηηηηηη???ηηηηη???ηηηηη). The amplitude boundary is named:
lane = rung-signature, step = correlation gate. The addresses are real (φ
gives them, V-independent); the payload does not follow them at 19.2
µm/voxel. What would reopen the gate: amplitude at finer voxel scale (Lane D
request pending with the Vesuvius team), or a V-stack at native level-0.

## Fifth submission: the φ-stencil — Virtual Philodemus's recipe, measured
`SUBMISSION_STENCIL.md` — a fifth, standalone September-round filing: the
φ-spaced normal-sampling recipe (K=4, d=1 and d=2, trilinear,
intensity-weighted centroid) implemented and measured as a pilot on the
PHerc.1667 column-15 / rung-15 window (147,456 vertices). Mean displacement
+0.2080 vox (d=1) / +0.8557 vox (d=2); 82.24% / 89.54% of vertices move more
than 1 voxel. The fuel mask is inert: 0 of 26,932 hits in the window, fuel
arm bit-identical to nofuel. Fidelity vs raw CT chunks: r≈0.987 at profile
level, ±1-voxel estimator systematic. Surface-error verdict: displacement is
not dominated by surface error or gradient magnitude (d=1: r=0.11;
d=2: r=0.31); driven mainly by local profile-shape response with a real but
weak edge-position component. Masked ground truth pending via Lane D.
Readout only — the mesh is not updated. Full filing document
`SUBMISSION_STENCIL.md`; code and data in `phi_stencil/`.

## License
MIT — all free, all phi.
