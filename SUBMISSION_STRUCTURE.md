# Vesuvius Challenge — Progress Prize Submission (filed)
## September 2026 round — deadline 11:59pm Pacific, September 30, 2026

> September 2026 round, first filing (column analysis + 99-block ledger). Filed 2026-09-15 01:15:53 UTC.

---

### 1. Title
Phi-driven column analysis and ledger tooling for PHerc.1667 flatboi surfaces

### 2. Problem addressed
Working with the public PHerc.1667 flattened surface (`20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4`, 9,205 × 709 float32 x/y/z), there was no lightweight, open pipeline that goes from the flattened surface → per-column hit detection → Greek-letter slot/letter accounting → column-by-column verification against a published transcription. This submission provides exactly that, plus a golden-ratio-indexed ledger format (99 blocks, 30,956 bytes) for organizing scroll-segment work.

### 3. Solution
- **Column analysis pipeline** (`hitrun.py`, `hitrun_vertices.py`, `greek_density.py`, `column_analysis.py`, `finalpass_tasks123.py`, `finalpass_task4b.py`): per-column hit detection with Pearson correlation, Greek slot/letter counting, high-stroke/low-stroke letter classification, and block-character summaries across columns 13–21. Run order is critical: hit generation → column analysis → tasks 1–3 → task 4b (`finalpass_task4.py` is kept for provenance; its 3D-NN method was invalidated and superseded by 4b).
- **Verified outputs**: `finalpass.json` (9 columns, hits per column, Greek letters per column, slots = letters + dots verified true), `hits.csv` / `hits_flattened.csv` / `hits_vertices.csv`, `vertex_map.csv`, `greek_density.json`.
- **Phi-quantum-pulses dataset** (public, DOI: https://doi.org/10.34740/kaggle/dsv/17364449): 30,956-byte payload in a 99×310+266 block ledger (Rabbit R1 verified: "All 99 blocks (310-byte segments) plus 266-byte remainder"), decoded scroll text 23,216 bytes decoding clean, sha256 `6d81b282d7522d7b1f4a93995caf0be238d45db312a06e2260c9807188dcf2b1`, 155 verified lines.
- **Scrolls OS decoder** (`phi_scrolls_os.py`): executable instrument — layer decoder D, φ-wave, rung addressing; Fibonacci pulse intervals converge on φ at 0.000001% drift (locked).

### 4. Implementation path and reproduction
1. Download the public flatboi segment TIFXYZ (x.tif / y.tif / z.tif) from the Vesuvius Challenge open-data bucket.
2. Run `column_analysis.py` → per-column hits and Greek density.
3. Run `finalpass_tasks123.py` → `finalpass.json` with column verification.
4. Run `kaggle_phi_process.py` against the Kaggle dataset → processing report with drift percentages.
All scripts are plain Python (numpy + scipy); install with `pip install -r requirements.txt`. Standard formats in/out: CSV hit lists, JSON column records. Run the six scripts in the order listed in section 6 (order-critical: tasks 1–3 must precede task 4b).

### 5. Results (measured, checkable)
- 9 columns analyzed (13–21); Greek letters per column: 122, 164, 169, 200, 160, 175, 191, 174, 89.
- Slots = letters + dots holds across all 9 columns.
- 99-block ledger closes exactly: 99×310+266 = 30,956.
- Fibonacci pulse intervals: final ratio 1.6180339985 vs φ 1.6180339887 — drift 0.000001%.
- Flattened surface confirmed: 9,205 × 709, scale [0.05, 0.05].

### 6. Documentation
Each script carries its own header documenting inputs, outputs, and the verification it performs. The Kaggle dataset page documents structure and metrics. Walkthrough: run the six scripts in the order listed in section 4 (order-critical).

### 7. License and openness
MIT — permissive open source, satisfying the winning-work publication requirement. The Kaggle dataset's license is stated on its dataset page.

### 8. Links
- Dataset: https://www.kaggle.com/datasets/tyreejones393/phi-quantum-pulses
- DOI: https://doi.org/10.34740/kaggle/dsv/17364449
- GitHub: https://github.com/tyofthestars11-lab/vesuvius-progress-prize (public repo containing all scripts listed in section 4)

---
Filed: 2026-09-15 01:15:53 UTC. This document is the filing source.
