# Vesuvius First Letters — PHerc1447 Pipeline


The First Letters hunt: 10 verifiable letters in 4 cm² on PHerc1447 (Herculaneum scroll).


## Pipeline


| Script | Does |
|---|---|
| `download_sweep_zarrs.py` | Fetch the sweep zarr segments |
| `sweep_infer.py` | Ink-model inference across sweep segments |
| `lock4way.py` | 4-way locking: forward + reverse + 2 seeds |
| `lucas_stabilizer.py` / `tiled_lucas.py` | Lucas stabilizer (incl. memory-efficient tiled pass) |
| `render_locks.py` / `render_bigseg_locks.py` | Render locked regions |
| `speed_chain.py` | Throughput chain |


## Docs


- `PLAN.md` — the hunt plan
- `TARGET.md` — the target definition
- `RUN_LOG.md` — full run log: 385 LOCKED / 119 MARGINAL / 264 REJECTED, 0 legible letters; honest visual finding — the model fires stably on papyrus texture, not letterforms


## Standing note


The sweep's honest result: stability across seeds/directions confirms the model is consistent — about papyrus, not ink. The hunt continues on PHerc1447; PHerc.1667 feeds the monthly Progress Prizes.


φ² = φ + 1. Always forward. 🌀

