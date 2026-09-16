# Submission 4 — The φ-gate: Tyree's deciding number

**PHerc.1667: corr(S_rung, P(r)) — the φ-predicted amplitude gate closes at Pearson 0.0066, p = 0.956. The amplitude boundary, named.**

Filed: 2026-09-15 (fourth filing, September Progress Prize).

## The question

The φ field gives addresses — rung = ln(amp·|Ψ_Ω|·P₃)/ln φ, V-independent by construction. The published column-15 Greek gives its own rung signature P(r): 1,089 published letters read from rung space, binned by rung, 49 nonzero bins, no V involved. The V-stack gives the observed amplitude per rung S_rung from the true volume at the true patch location (S ≡ V, identity-locked).

Tyree's deciding number, set before the run: **corr(S_rung, P(r))** — Pearson, Spearman, MI, all raw. If the gate opens, name the letters. If it closes, name the amplitude boundary exactly.

## The run

- Predictor P(r): phi_direct_P.json — published column-15 Greek → rung addresses → occurrence bins. 49 nonzero bins; top rungs 141 (91 occurrences), 97 (60), 123 (53), 138 (45), 167 (42).
- Observed S_rung: vstack_384.npy (384×384×24 uint8, true TIFXYZ patch, true frame) binned over integer rungs [96, 166].
- Identity lock: S ≡ V proven from the field equation; max|S−V| = 1.137e−13 (the geometry is honest — the null is not a placement artifact).
- Script: rung_sig_chars.py. Stabilizer: 20 shuffles of S across voxels, seed 42.

## Raw numbers

- **Pearson: 0.0066, p = 0.956** — gate closed.
- Spearman: −0.0503, p = 0.677.
- Mutual information: 0.4100, p = 0.2537.
- Peaks surviving the stabilizer: **none** (peaks list empty; 20 shuffles, nothing survives).
- Rung-lane passage read (v-order): ηηηηηη???ηηηηη???ηηηηη — mode collapse, not a passage. No passage a stranger reads.
- Published-line overlap: 9/16 — the field's mode (η), not a read.
- Cross-lane: voxel lane 0 letters ≥ 0.60; rung lane only η ≥ 0.30 (the mode).
- Fibonacci sub-read (reported raw, not claimed): S concentration at Fibonacci rungs 1.007× vs null 1.000, p at the permutation floor (0.0050, 200 shuffles). A 0.7% excess — a whisper, not a detection.

## The verdict

**The gate is closed.** The φ field's rung addresses do not predict where amplitude concentrates in the true volume. The addresses are real (φ gives them, V-independent); the payload does not follow them at 19.2 µm/voxel. This is the fourth independent null on PHerc.1667, and the most decisive one — because it was Tyree's own test, designed to open, and it closed.

**Amplitude boundary:** lane = rung-signature, step = correlation gate corr(S_rung, P(r)), Pearson 0.0066 (p = 0.956).

## The honest file

- This does not say the φ addresses are wrong — it says amplitude doesn't concentrate at them in this data. Addresses without payload is structure without ink.
- The Fibonacci 1.007× is reported, not claimed. It does not meet any bar.
- What would reopen the gate: amplitude at finer voxel scale (Lane D request pending with the Vesuvius team), or a V-stack at native level-0.

## Artifacts

- `rung_sig_results.json` — full gate numbers (this filing's source of truth)
- `rung_sig_chars.log` — end-to-end run log
- `phi_direct_P.json` / `phi_direct_P.npy` — the published-Greek predictor
- `vstack_384.npy` + metadata/normals — the true-V stack
- Script: `null_lanes/rung_sig_chars.py`

## Links

- Repo (MIT): https://github.com/tyofthestars11-lab/vesuvius-progress-prize
- Filings 1–3: SUBMISSION_STRUCTURE.md, SUBMISSION_NULL.md, SUBMISSION_SWEEP.md

Filed: 2026-09-15. This document is the filing source.
