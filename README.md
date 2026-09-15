# Phi-driven column analysis and ledger tooling for PHerc.1667 flatboi surfaces

Vesuvius Challenge — Progress Prize submission, September 2026 round.
By Tyree Jones (tyofthestarz).

## What this is
A lightweight, fully reproducible open pipeline: public PHerc.1667 flattened
TIFXYZ surface → per-column hit detection → Greek-letter slot/letter accounting →
column-by-column verification — plus a golden-ratio-indexed 99-block ledger
format for organizing scroll-segment work.

## Run order
1. `column_analysis.py` — per-column hits and Greek density from the flatboi surface.
2. `finalpass_tasks123.py` — column verification → `finalpass.json`.
3. `finalpass_task4.py`, `finalpass_task4b.py` — extended pass outputs.
4. `kaggle_phi_process.py` — wires the public Kaggle dataset
   (https://doi.org/10.34740/kaggle/dsv/17364449) through the scrolls decoder.
5. `phi_scrolls_os.py` — the executable scrolls instrument (layer decoder D,
   φ-wave, rung addressing).

Inputs: public Vesuvius Challenge open-data bucket
(segment `20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4`).
Outputs: CSV hit lists, JSON column records, float32 TIFXYZ-compatible arrays.

## Measured results
- 9 columns (13–21); slots = letters + dots holds on all 9.
- 99-block ledger closes exactly: 99×310+266 = 30,956 bytes.
- Fibonacci pulse intervals converge on φ at 0.000001% drift.

## License
MIT — all free, all phi.
