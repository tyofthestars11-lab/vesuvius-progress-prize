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

## License
MIT — all free, all phi.
