# PHerc.1447 First Letters sweep: stability is not ink — 385 locked regions, 0 letters, cross-seed null signature

**Submitter:** Tyree Jones (tyofthestarz), tyofthestars11@gmail.com
**Repository:** https://github.com/tyofthestars11-lab/vesuvius-progress-prize
**Prize round:** September 2026 Progress Prizes (third standalone September submission; the first was the column-analysis/99-block-ledger filing, the second was the PHerc.1667 six-instrument null — multiple submissions per month are permitted)

## What this is

A full First Letters pipeline sweep of PHerc.1447: 14 segments plus the main segment, run through a four-map locking rule, with a cross-seed stability test that produces a **null signature** — the model is consistent about papyrus, not about writing.

## Method

- **Four-map locking rule:** a hotspot locks only if its peak ≥ 0.60 in ALL FOUR maps (2 model seeds × forward/reverse passes). MARGINAL at ≥ 0.55 in all four. Everything else REJECTED. Peaks measured in component bounding boxes dilated 16 px.
- **OOM fix:** the largest segment (20251105093211, 112M px) SIGKILLed the box twice (float64 maps = 3.6GB+ on a 7GB machine). Fixed with `tiled_lucas.py` — 1024px tiles, 64px overlap, core-centroid ownership, streaming Pearson/IoU accumulators, uint8 reads. Same locking rule, no science changed.
- **Visual verification:** every locked region rendered and inspected; the largest locked component was 26,286 px — a papyrus band, not a character.

## Results

| Scope | LOCKED | MARGINAL | REJECTED |
|---|---|---|---|
| 14-segment sweep | 377 | 119 | 244 |
| Main segment 20250703034159 | 8 | 0 | 20 |
| **Grand total** | **385** | **119** | **264** |

Legible letters: **0**.

## The null signature (cross-seed)

- Main segment cross-seed r = **0.34/0.27**, strong-pixel IoU ~0.086.
- Big segment (20251105093211, highest agreement of any segment) cross-seed r = **0.60/0.57**, IoU ~0.29.
- Physics: real ink is one physical deposit that both inits observe through the same volume — it would force cross-seed r > 0.8 (both nets must see the same strokes). 0.34 vs 0.8 is the gap: the model responds to texture, not to a shared physical signal.

**Stability ≠ ink.** The model is consistent about papyrus, not about writing.

## Scripts (all in `sweep1447/`)

- `tiled_lucas.py` — OOM-proof tiled inference (1024px tiles, 64px overlap)
- `sweep_infer.py` — 14-segment sweep driver
- `lucas_stabilizer.py` — the φ-LUCAS locking rule
- `lock4way.py` — four-map lock/marginal/reject classifier
- `render_locks.py`, `render_bigseg_locks.py` — lock renderers
- `download_sweep_zarrs.py` — segment data fetcher

## Honest boundaries

- No letters claimed. This is First Letters pipeline infrastructure and a measured negative, not a First Letters claim.
- The null signature is specific to the tested segments and models (iter0/iter5-class inits on PHerc.1447); it does not pre-judge other scrolls or finer data.
- Visual inspection covered the locked regions; 385 locks inspected at the render level, 0 letterforms.
