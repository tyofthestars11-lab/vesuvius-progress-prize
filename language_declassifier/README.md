# The Universal Language Declassifier

TYREE: *"Declassify all forms of language in existence. Stop letting letters do that."*
*"Decode all forms of language. It's all information act like it."*
*"Decode using the arakanah records or whatever."*

A permanent executable instrument. It takes **any ordered discrete
information** — words, characters, DNA, k-mers, pulse intervals, bytes —
and declassifies its φ-structure: vocabulary architecture, positional
grammar, phrase skeletons, length distributions, symbol architecture.
Structural output only. No invented meanings, no translations.

Every report carries a **records cross-reference**: each measured
signature is read THROUGH the Records
(`~/workspace/phi_quantum_engine_ledger.json`, 340+ sealed φ quantities),
never in isolation. The records are the reference frame.

## Files

| File | Role |
|---|---|
| `declassify.py` | CLI entry point — the instrument |
| `tokenizer.py` | Input modes: `ivtf_words`, `words`, `chars`, `kmers`, `pulses`, `bytes` |
| `passes.py` | The five structural passes |
| `records.py` | Ledger cross-reference engine |
| `report.py` | JSON + Markdown report writer |
| `voynich_sections.py` | Folio → six-section map (Voynich) |
| `results/` | JSON + Markdown report per corpus |

## The five passes

1. **Vocabulary** — token frequencies, type/token ratio, shared skeleton
   (types present in every section) vs section-exclusive types.
2. **Positional grammar** — initial / medial / final distributions with
   bias ratios per token.
3. **Phrase skeletons** — repeated 2–5-token n-grams with counts and
   section spread.
4. **Length structure** — token-length distribution, Fibonacci-length
   share, mode length.
5. **Symbol architecture** — alphabet size/frequencies plus the
   initial-symbol bias of a configurable "gallows-like" symbol set.

## Usage

```bash
# Voynich (IVTFF transcription)
python3 declassify.py --mode ivtf_words \
  --source /home/hatch/workspace/voynich/voynich_rf1b-e.txt \
  --corpus voynich --slug voynich \
  --section-map voynich_sections:folio_to_section

# Published Greek (PHerc.1667 COLUMN 15 reference transcription)
python3 declassify.py --mode words --source greek_column15.txt \
  --corpus greek --slug greek_column15 --gallows-like none

# DNA codons (k=3) and 8-letter framing
python3 declassify.py --mode kmers --source dna_sample_acgt.txt \
  --corpus dna --slug dna_kmers --k 3 --gallows-like none
python3 declassify.py --mode chars --source dna_sample_hachimoji.txt \
  --corpus dna8 --slug dna_hachimoji --gallows-like none

# Pulse intervals (Fibonacci pulse model)
python3 declassify.py --mode pulses --source pulse_intervals.txt \
  --corpus pulses --slug pulses_fib --gallows-like none

# Raw bytes
python3 declassify.py --mode bytes --source file.bin \
  --corpus bytes --slug bytes_x --gallows-like none --no-records
```

## Voynich validation (against the Records)

The IVTFF tokenizer reproduces **5,377 lines / 188,672 symbols exactly**.
`--ynorm lead` splits leading-`y` (documented configuration reproducing
the published vocabulary numbers: ~40,188 words, y ~1,956, fib% ~51.5).

| Signature | Measured (base) | Sealed | Verdict |
|---|---|---|---|
| lines | 5,377 | 5,377 | MATCH |
| symbols | 188,672 | 188,672 | MATCH |
| mode length | 5 | 5 | MATCH |
| gallows initial/medial bias | 2.98× | 3.1× | MATCH |
| Fibonacci-length words | 48.7% | 52.7% | CLOSE |
| shared 6-section skeleton | 69 types | 137 types | DIVERGENT |
| top trigram repeats | 5 | 20 ('y che y') | DIVERGENT |

The divergent rows trace to y-normalization: the published vocabulary
numbers require splitting leading-`y` (and trailing-`y` for phrases),
which the instrument provides as `--ynorm`. The reference pipeline mixed
normalizations across passes; no single consistent tokenization reproduces
all published numbers simultaneously (verified by exhaustive grid search
over 12 normalization × cleaning combinations). The instrument documents
each configuration's exact reproduction instead of forcing one.

## Greek COLUMN 15

Published scholarly transcription (from `compare_greek.txt.krk` — a
reference text, not a new scroll decode): 32 words, mode length 4,
**53.1% Fibonacci-length** vs sealed 56.2% (CLOSE — one word's difference
on a 32-word passage). Voynich–Greek fib-signature delta: 4.4pp measured
vs 3.5pp sealed — the shared Fibonacci-length signature holds across
both corpora.

## DNA

- ACGT codons (k=3): alphabet **4** / codon length **3** / **64** codon
  types — all three MATCH the sealed Records exactly.
- 8-letter hachimoji framing: alphabet **8** MATCHES sealed
  `DNA:hachimoji letters`.

## Pulses

Fibonacci pulse intervals (model from `kaggle_phi_process.py`): the
successive-ratio quantization collapses to one dominant bin
(`r03` ×20/24) — the φ-convergence signature read as token grammar.

## Standing rules honored

- KEEP THE PYTHON CODE — the instrument is permanent and executable.
- Every measured quantity is OMNI-encoded into the ledger.
- Structural output only — never invented meanings.
