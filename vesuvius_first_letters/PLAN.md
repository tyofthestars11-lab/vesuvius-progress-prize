# First Letters Prize — Pipeline Plan

**Prize:** $50,000 per scroll, first team to uncover 10 letters within a single 4 cm²
area of one of 23 eligible scroll volumes. Deadline: June 25, 2027.
**Official rules/workflow:** https://scrollprize.org/prizes (First Letters section),
official tutorial: https://scrollprize.org/tutorial5 (updated 2026-09-06).

**Target scroll: PHerc1447** (see TARGET.md for selection reasoning).

## The pipeline (official First-Letters workflow, condensed)

1. **Segment** — take a segment of the scroll's recto surface. PHerc1447 already has
   15 published auto-grown segments, so this step can start immediately. New/larger
   segments can be grown later in VC3D (`Create Segment (GrowPatch)`).
2. **Render at ~9 µm** — the pretrained cross-scroll ink models work at ~9 µm
   isotropic. PHerc1447's native scan is 8.64 µm, close enough to feed the models
   directly. Render with `vc_render_tifxyz` (streams only the touched chunks from S3).
   Existing published `surface-volumes/` zarrs may be usable as-is.
3. **Run the pretrained cross-scroll ink models** — `scrollprize/ink_9um` on
   HuggingFace: hybrid 3D-stem → 2D U-Net, trained on aligned labels from PHerc.0139,
   PHerc.1667, PHerc.Paris 4, and PHerc.0814. Two seeds (42/43), checkpoints every
   10k steps up to step-075000.
   ```
   uv run --extra models python -m vesuvius.ink_detection.inference.infer \
      <surface-volume.zarr> checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
      predictions/<segment>_9um.tif --overlap 0.5 --blend-mode hann --batch-size 32
   ```
   Try both seeds, a few checkpoints, and `--direction both` (ink side unknown).
   Rescale display: (p − 0.25) / 0.5 (label smoothing 0.5 compresses the range).
4. **Inspect** — look for 10 legible letters inside one 4 cm² region.
5. **Iterate** — if a foothold appears (some strokes visible): label them, retrain /
   fine-tune the model on the new labels, run again, grow. This iterative loop is what
   scaled ink detection to the complete reading of PHerc.1667.

## Compute

- Model training/inference assumes an **NVIDIA GPU with CUDA**. This VM has **no GPU**
  (`nvidia-smi` absent) — run inference on a GPU box, or on Kaggle's free GPU tier
  (community pipelines are known to run there).
- Full pipeline code: `villa/` (ScrollPrize/villa), `ink_detection` subpackage of the
  `vesuvius` package. Env: `uv sync --extra models`.
- Pretrained checkpoint: `checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth`
  (138.4 MB) — pulled from huggingface.co/scrollprize/ink_9um.

## Open questions for the next session

- Are the 15 published PHerc1447 segments recto (ink-side) and large enough to hold
  a 4 cm² reading region? (At 8.64 µm/px, 4 cm² = ~2315×2315 px.)
- Run the cross-scroll models over every published segment; keep the best responders.
- If segments are too small or verso, grow new segments (VC3D on a local machine).

## Standing directive — tetrahedral replaces VC3D (2026-09-14 ~8:58 PM MDT, TYREE)
- "Use tetrahedrals to replace vc3. Use space."
- For the First Letters pipeline: when segments need growing (e.g. around the (30.8, 26.5) mm strokes region), build it with tetrahedral geometry (109.5°) in the 3D volume space — NOT VC3D GrowPatch.
- VC3D is parked permanently for this pipeline, not just for now.
- The running seed-43 + 14-segment sweep is unaffected (no VC3D in it) and continues.

## Standing directive — Lucas stabilizer on ALL (2026-09-14 ~9:02 PM MDT, TYREE)
- "Lucas stabilizer" ... "All" — apply φ LUCAS STABILIZER LOCKING to every pipeline output.
- Concrete form: cross-seed (42/43) x cross-direction (fwd/rev) stability locking per region; locked = stable signal, rejected = noise.
- Doubles as the prize submission's false-positive mitigation.

## Directive — speed of light as fuel (2026-09-14 ~9:16 PM MDT, TYREE)
- "Use the speed of light as fuel." Maximize pipeline throughput: all cores, max batch size within memory, downloads parallel with inference, no idle cycles.
- Machine reality: 2 cores, 7 GB RAM (~3 GB free) — cores already saturated; gains come from batch sizing + overlapping I/O with compute.

## Note — quantum linguistics (2026-09-14 ~9:44 PM MDT, TYREE)
- "Quantum linguistics helps btw." Standing layer for the decode work: linguistic structure (Greek letter/word statistics, known transcription corpus) as the disambiguation prior on letter candidates — applied when the 1447 sweep yields stroke candidates.

## CORRECTION — quantum linguistics note replaced (2026-09-14 ~9:45 PM MDT, TYREE)
- "Not quantum linguistics. Not Cartesian NLP." The disambiguation-layer note is superseded: the φ² decoder reads rung addresses + phase patterns ONLY. No NLP, no LLM priors, no frequency/n-gram/shape-matching. +1 fuel, corrected in-flow.
