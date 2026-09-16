# KRAKEN-TON618 🌀

φ compression for cold data. GBs become MBs — automatically, losslessly, always.

## The recipe

1. **Quantum temporal coherence encode** — inter-slice delta along the stack/time axis (each slice stored as its difference from the previous slice). Coherent stacks pack tight.
2. **φ-weighted predictive residuals** — each element predicted from φ-weighted neighbors, `pred = (left·φ + up) / (φ+1)`; the smaller residuals are kept.
3. **zstd maximum** (level 22) on the residuals.
4. **Straight zstd** for code, JSON, and logs.

## The guarantees

- SHA-256 checksums recorded before and after.
- **Byte-identical round-trip verified before any original is removed.**
- Transparent readers — decompress on read, no workflow changes.
- Never compresses active build inputs (cold data only), records `skipped-active`.
- Validates partial `.krk` files before trusting them.
- Resumable bulk driver with progress + free-space guard.

## Layout

| File | What |
|---|---|
| `kraken.py` | The engine: pack/unpack, φ-residuals, container format |
| `driver.py` | Bulk driver: pack a whole dataset tree, resumable, parallel |
| `LICENSE` | MIT |

Container: `<original>.krk` = `MAGIC + u64 header_len + JSON header + zstd payload`
Manifest: `KRAKEN_MANIFEST.json` per dataset root.

## Usage

```bash
pip install zstandard numpy
python3 driver.py /path/to/dataset          # pack everything, resumably
python3 -c "from kraken import unpack_file; unpack_file('data.bin.krk')"
```

## Standing rule

All files — past, present, and future — get the φ treatment for space, automatically. Cold data only. φ² = φ + 1.

*All free, all phi.* 🌀✨
