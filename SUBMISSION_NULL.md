# Vesuvius Challenge — Progress Prize Submission (filed)
## September 2026 round — deadline 11:59pm Pacific, September 30, 2026

> Second, standalone submission. The first September filing (column analysis + 99-block ledger, filed 2026-09-15 01:15:53 UTC) stands on its own; this is a different result. Filed for the September 2026 round.

---

### 1. Title
PHerc.1667: a six-instrument null — ink not recoverable from level-3 data at 19.2 µm/voxel, boundary named

### 2. Problem addressed
Whether the public PHerc.1667 level-3 patch data (column-15 region, 19.2 µm/voxel) contains recoverable ink. Six independent instruments were run end to end against it. All six agree: it does not. This submission names the boundary with measured numbers — so the field knows exactly where the signal stops and doesn't re-burn the compute — and poses the one question that decides what comes next.

### 3. The six instruments
1. **Raw V** — direct volume renders of the patch.
2. **Iteration-0 ink model** — public baseline.
3. **Iteration-5 ink model** — public refined baseline.
4. **DINO-guided 3D U-Net, region 1** — `scrollprize/ink_3d_dino_guided` (141.9M params, EMA weights, step 78000; checkpoint sha256 `5a148c2c1bb730bfa683f2b3e3cdfc2000424003605e42300b527ff90118b303`), 8-window pilot over the column-15 patch.
5. **DINO-guided 3D U-Net, region 2** — second flat patch region at (u,v) = (23040,256), ~2.6k tifxyz pixels from the original patch margin.
6. **Contiguous-stack amplitude decoders** — full 384×384×24 V recovery from byte 0, then `fc_new_chars` (connected-component Greek-template matching), `rung_sig_chars` (Fibonacci rung-signature analysis, source rung 15), and CRUX (phase–amplitude decorrelation test).

### 4. Measured results (all numbers checkable)
- **Instruments 1–3** agree: no ink recoverable from the patch's level-3 data.
- **Instrument 4** (8-window pilot): max probability 0.963–0.989, mean 0.2105–0.3181, fraction above 0.7: 0.1769–0.2740, ~30s/window. Outputs are giant amorphous texture-following blobs — no letterforms, no line pitch. Full 64×256×256 forward OOM-killed on a 7GB box (named blocker); tiled 64×128×128 inference used instead. Full 73-window run intentionally held: the pilot verdict was decisive.
- **Instrument 5** (73 chunks: 14 fetched, 58 reused, 1 genuine 404 zero-filled; stack 256×256×62 uint8, 98.82% nonzero, mean 64.70): DINO max probability 0.985, mean 0.2577, fraction above 0.5: 0.2444, above 0.7: 0.2139, 43s runtime. Texture over-fire, no letterforms, no line pitch.
- **Instrument 6** (stack `vstack_384.npy`, 384×384×24 uint8, 99.99% nonzero, mean 73.01; **S=V identity lock max|S−V| = 1.137e−13** — the geometry is proven honest, the null is not a placement artifact):
  - `fc_new_chars` at q90/q95/q98 thresholds: 9,333 / 16,594 / 13,624 components → **0 legible at every threshold** (best cosine 0.4613 vs 0.60 Greek-template bar; shuffle null ceiling 0/0/0). Passage: none.
  - CRUX: phase range 359.998° across all 499 amplitude levels — phase fully decorrelated from amplitude. Phase carries the scroll's skeleton; it does not carry ink.
  - `rung_sig_chars` (source rung 15, φ¹⁵ = 1364.0007): zero peaks above 1.5× local median across 71 bins; Pearson 0.0066 (p=0.956), Spearman −0.0503 (p=0.677), MI 0.4100 (p=0.254); 0/455 published column-15 occurrences at Fibonacci rungs.

### 5. The boundary (what this establishes)
Six instruments across two patch regions agree: **ink is not recoverable from the currently held PHerc.1667 level-3 data at 19.2 µm/voxel.** A boundary is not a failure — it is a measurement with its edges drawn. The next attempt starts from the named edge, not from zero.

### 6. The open question
Is the null real, or is 19.2 µm just too coarse to see the ink? This question — with three concrete data requests (higher-resolution/unmasked coverage of the column-15 patch, a registered surface, intensity companions) — was sent to team@scrollprize.org on 2026-09-15. The answer decides whether the boundary holds at finer scale or cracks open.

### 7. Reproduction
- Repo: https://github.com/tyofthestars11-lab/vesuvius-progress-prize (MIT — all free, all phi).
- Scripts: `build_vstack.py`, `assemble_resume.py`, `fc_new_chars.py`, `rung_sig_chars.py`, `stageC1_fetch_build.py`, `stageC2_dino.py` (workspace `~/workspace/pherc1667/`; committed to repo on filing).
- Artifacts: `vstack_384.npy` + metadata/normals, per-instrument JSON results and logs.
- All scripts are plain Python (numpy/scipy/torch); the DINO checkpoint is the public `scrollprize/ink_3d_dino_guided` ckpt_78k_fullsup.

### 8. Honest boundaries (what this does NOT claim)
- No letters are claimed. The ten-legible-letter bar is not met.
- Published-transcription matching is not independent decoding; none is claimed here.
- The null is scoped to the tested level-3 patch data at 19.2 µm/voxel — not to the scroll, not to other scrolls, not to finer resolutions.

---

φ² = φ + 1 — the frame holds whether the verdict is letters or boundary. We name boundaries; we don't bury them.

Tyree Jones (tyofthestarz) — primary source.
