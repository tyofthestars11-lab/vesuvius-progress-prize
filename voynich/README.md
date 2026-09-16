# Voynich Manuscript — φ Structural Analysis 📜

Full φ-rung structural analysis of the Voynich manuscript (Beinecke MS 408), from the EVA/RF transliteration.

## Pipeline

| Script | Does |
|---|---|
| `full_voynich_phi_ton618.py` | Rung-addresses every line and page by cumulative symbol count, relative to TON618 |
| `voynich_phi_spectral.py` | φ-transition operator spectrum: FFT power aggregated into φ-rungs of inverse wavelength |
| `extract_voynich_generator.py` | Long-range φ-scaled transition-field generator from dominant modes |
| `map_voynich_residuals.py` | Maps φ-residuals to folio/locus order (227 folio residuals, 20 layout bands) |
| `correlate_residual_quire_proxy.py` | Folio-boundary vs interior residual energy (permutation test) |
| `correlate_true_codicology.py` | Model-based quire/bifolio boundary analysis (standard quire = 4 bifolios = 8 folios) |
| `inverse_bifolio_filter.py` | Inverse-FFT isolation of the bifolio band → pure bifolio + secondary harmonic components |
| `synthesize_voynich_structure.py` | Three-layer structural reconstruction (base + bifolio band + secondary harmonic) |

## Key results (all in `*_results.json`)

- **188,672 symbols → rung 25.24**; manuscript sits **25.52 rungs below TON618**
- **52.7% of 40,408 words carry Fibonacci lengths** (mode length 5)
- Spectral peak at **rung −2** — the φ⁻² weight rung of the transition operator
- Boundary excess holds at all three physical scales: folio 0.87% (p=0.0), bifolio 0.64% (p=0.0075), quire 1.04% (p=0.0345)
- Bifolio band (rungs 9.36–9.72) carries 2.99% of residual variance; secondary harmonic (rung 15.48) 0.11%
- Three-layer sum reconstructs the residual field to floating-point precision

## Paper

`Structural_Reconstruction_of_the_Voynich_Manuscrip.pdf` — the full structural reconstruction writeup (honest boundaries included: not a plaintext translation).

## Data

- `voynich_rf1b-e.txt` — EVA/RF transliteration source
- `voynich_pure_bifolio_component.bin` / `voynich_secondary_harmonic_component.bin` — isolated components (.npy format)
- `voynich_structural_reconstruction.bin` — synthesized three-layer field
- `5451_54_glk0.png` — φ-transition Morlet scalogram of the residual field

φ² = φ + 1. The manuscript was φ all along. 🌀
