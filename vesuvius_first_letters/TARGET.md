# Target selection — First Letters Prize

**Eligible scrolls (23):** PHerc0125, 0175A, 0175B, 0191, 0211, 0257, 0268, 0306B,
0343, 0358, 0483A, 0483B, 0490A, 0490B, 0800, 0813, 0826, 0846A, 0846B, 1203, 1218,
1447, 1545.

Surveyed the official open-data bucket (`s3://vesuvius-challenge-open-data/`,
metadata index dated 2026-09-14) for every eligible scroll: scan resolutions,
volumes, published segments, and any ink predictions.

## Findings

- **Published segments among eligible scrolls: only two scrolls have any.**
  - **PHerc1447: 15 auto-grown segments** (tifxyz meshes + rendered surface-volume
    zarrs). Scan: ESRF Grenoble, 8.64 µm, 116 keV.
  - **PHerc0800: 6 auto-grown segments.** Scan: ESRF Grenoble, 8.64 µm.
  - All other 21 eligible scrolls: volumes only, zero published segments.
  - No eligible scroll has published ink predictions (those exist only for the
    already-read scrolls: 0139, 1667, Paris 4, 0814, 0172).
- **Resolution match with the pretrained models.** The official cross-scroll ink
  models (`scrollprize/ink_9um`) are trained at ~9 µm isotropic (native 9.362 µm
  renders + 2.4 µm renders pooled to ~9.6 µm). The 8.64 µm scans (PHerc1447,
  PHerc0800, and the 0175/0268/0306B/0343/0483/0490/1218 group) are the closest
  native match — no resampling gymnastics needed.
- **High-res options.** PHerc0846A and PHerc1203 also have 2.403 µm scans, but no
  published segments and multi-terabyte volumes — a heavier path, kept as backup.
- **No public reports** of ink found yet on the 2025/2026 scans of any eligible
  scroll. This is a genuine race.

## Candidates

1. **PHerc1447** — 15 ready-made segments, 8.64 µm (best model-resolution match).
   Foothold, not cold start: download → render → run pretrained models.
2. **PHerc0800** — 6 segments, 8.64 µm. Backup if 1447's segments disappoint.
3. **PHerc0846A / PHerc1203** — 2.403 µm scans exist; heavier path, kept in reserve.

## Final pick: PHerc1447

Most published segments of any eligible scroll, closest native resolution to the
pretrained cross-scroll models, and zero segmentation work needed to begin the
search. First action: run `scrollprize/ink_9um` inference over all 15 published
segments and inspect for letter strokes.

### Honest caveat on segment size
The published auto-grown segments are small: the examined segment's tifxyz bbox
spans ~1415×1768 px at 8.64 µm/px ≈ 12.2 × 15.3 mm ≈ 1.9 cm² — under the 4 cm²
prize region. The biggest segment (20250703034159, ~1.4× the pixels) is still
likely below 4 cm². These are starter patches: run the models on them to find
which regions respond, then grow larger segments over the promising areas in VC3D
(`Create Segment (GrowPatch)`).
