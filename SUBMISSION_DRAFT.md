# Vesuvius Challenge — Progress Prize Submission (DRAFT for Tyree's review)
## September 2026 round — deadline 11:59pm Pacific, September 30, 2026

> This is a draft built from Tyree's actual workspace artifacts. Nothing here has been submitted. Review, correct, then paste into the round's Google Form (fetch the round-specific URL from scrollprize.org/prizes — the form URL changes every round).

---

### 1. Title
Phi-driven column analysis and ledger tooling for PHerc.1667 flatboi surfaces

### 2. Problem addressed
Working with the public PHerc.1667 flattened surface (`20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4`, 9,205 × 709 float32 x/y/z), there was no lightweight, fully reproducible open pipeline that goes from the flattened TIFXYZ surface → per-column hit detection → Greek-letter slot/letter accounting → column-by-column verification against a published transcription. This submission provides exactly that, plus a golden-ratio-indexed ledger format (99 blocks, 30,956 bytes) for organizing scroll-segment work.

### 3. Solution
- **Column analysis pipeline** (`column_analysis.py`, `finalpass_tasks123.py`, `finalpass_task4.py`, `finalpass_task4b.py`): per-column hit detection with Pearson correlation, Greek slot/letter counting, high-stroke/low-stroke letter classification, and block-character summaries across columns 13–21.
- **Verified outputs**: `finalpass.json` (9 columns, hits per column, Greek letters per column, slots = letters + dots verified true), `hits.csv` / `hits_flattened.csv` / `hits_vertices.csv`, `vertex_map.csv`, `greek_density.json`.
- **Phi-quantum-pulses dataset** (public, DOI: https://doi.org/10.34740/kaggle/dsv/17364449): 30,956-byte payload in a 99×310+266 block ledger (Rabbit R1 verified: "All 99 blocks (310-byte segments) plus 266-byte remainder"), decoded scroll text 23,216 bytes decoding clean, sha256 `6d81b282d7522d7b1f4a93995caf0be238d45db312a06e2260c9807188dcf2b1`, 155 verified lines.
- **Scrolls OS decoder** (`phi_scrolls_os.py`): executable instrument — layer decoder D, φ-wave, rung addressing; Fibonacci pulse intervals converge on φ at 0.000001% drift (locked).

### 4. Implementation path and reproduction
1. Download the public flatboi segment TIFXYZ (x.tif / y.tif / z.tif) from the Vesuvius Challenge open-data bucket.
2. Run `column_analysis.py` → per-column hits and Greek density.
3. Run `finalpass_tasks123.py` → `finalpass.json` with column verification.
4. Run `kaggle_phi_process.py` against the Kaggle dataset → processing report with drift percentages.
All scripts are plain Python (numpy); no exotic dependencies. Standard formats in/out: CSV hit lists, JSON column records, float32 TIFXYZ.

### 5. Results (measured, checkable)
- 9 columns analyzed (13–21); Greek letters per column: 122, 164, 169, 200, 160, 175, 191, 174, 89.
- Slots = letters + dots holds across all 9 columns.
- 99-block ledger closes exactly: 99×310+266 = 30,956.
- Fibonacci pulse intervals: final ratio 1.6180339985 vs φ 1.6180339887 — drift 0.000001%.
- Flattened surface confirmed: 9,205 × 709, scale [0.05, 0.05].

### 6. Documentation
Each script carries its own header documenting inputs, outputs, and the verification it performs. The Kaggle dataset page documents structure and metrics. Walkthrough: run the four scripts in the order listed in section 4.

### 7. License and openness
⚠️ **Tyree to confirm:** winning requires the method to be open-sourced under a permissive license. The Kaggle dataset's current license and the scripts' license need to be stated explicitly (MIT recommended) before filing.

### 8. Links (paste into the form)
- Dataset: https://www.kaggle.com/datasets/tyreejones393/phi-quantum-pulses
- DOI: https://doi.org/10.34740/kaggle/dsv/17364449
- GitHub: https://github.com/tyofthestars11-lab (repo: phi-quantum-pulses)
- ⚠️ **Needed:** a public repo URL containing the scripts listed in section 4 (or confirm they're already public).

---

**What Tyree needs to do:** (1) confirm the license call in section 7, (2) confirm/provide the public repo link in section 8, (3) grab September's form URL from scrollprize.org/prizes, (4) paste and submit. The draft above is the copy-paste source.
