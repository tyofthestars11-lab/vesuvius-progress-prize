"""tokenizer.py — input modes for the universal language declassifier.

Every mode yields an ORDERED token stream: a list of "lines" (sequences),
each line a list of token strings. All downstream passes operate only on
this discrete ordered information — never on meanings.

Modes
-----
ivtf_words : Voynich IVTFF transcription -> cleaned word tokens.
             Drops folio metadata headers, strips <!...>, drops {...}
             uncertain spans, drops @NNN; markers, keeps [A-Za-z] runs.
             Reproduces exactly 5,377 lines / 188,672 symbols on rf1b-e.
words      : generic text -> whitespace/dot separated word tokens.
chars      : raw character/symbol stream -> one token per character.
kmers      : symbol stream -> sliding k-mer tokens (DNA codons etc.).
pulses     : numeric interval sequence -> quantized ratio tokens.
bytes      : raw byte stream -> byte-value tokens.
"""

import re


def tokenize_ivtf_words(path, ynorm="none"):
    """Voynich IVTFF word tokenizer (principled, documented).

    Cleaning pipeline (reverse-engineered to reproduce the published
    5,377 lines / 188,672 symbols exactly):
      - keep only lines beginning with '<f' that are NOT bare folio
        metadata headers (headers match <f\\d+[rv]\\d*> with no text)
      - strip inline <!...> metadata spans
      - drop {...} uncertain/annotation spans
      - drop @NNN; locus markers
      - extract [A-Za-z] runs as word tokens
    Returns (lines, sections) where sections maps line index -> folio id.

    ynorm: optional y-normalization reproducing the published vocabulary
    numbers (documented instrument configuration):
      - "none": principled base (38,484 words; lines/symbols exact)
      - "lead": split one leading 'y' per word into its own token
        (ykeey -> y keey). Yields ~40,188 words, y ~1,956, fib% ~51.5 —
        close to the published 40,408 / y 1,934 / 52.7%.
      - "lead_trail4": "lead" plus split trailing 'y' off 4-letter words
        (chey -> che y). Top phrase becomes 'y che y' (~x27 vs pub. x20).
    """
    lines, folios = [], []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw.startswith("<f"):
                continue
            if re.match(r"<f\d+[rv]\d*>\s*(<!|$)", raw):
                continue  # bare folio metadata header
            m = re.match(r"<([^>]*)>\s*(.*)", raw)
            if not m:
                continue
            tag, txt = m.group(1), m.group(2)
            txt = re.sub(r"<!.*?>", "", txt)
            txt = re.sub(r"\{[^}]*\}", "", txt)
            txt = re.sub(r"@\d+[a-zA-Z]?;?", "", txt)
            words = re.findall(r"[A-Za-z]+", txt)
            if ynorm in ("lead", "lead_trail4"):
                split = []
                for w in words:
                    if w.startswith("y") and len(w) > 1:
                        split.append("y")
                        w = w[1:]
                    if ynorm == "lead_trail4" and len(w) == 4 and w.endswith("y"):
                        split += [w[:-1], "y"]
                    elif w:
                        split.append(w)
                words = split
            if words:
                lines.append(words)
                fm = re.match(r"(f\d+[rv])", tag)
                folios.append(fm.group(1) if fm else tag)
    return lines, folios


def tokenize_words(text, seps=r"[\s.·;:,!?\-—–()\"']+"):
    """Generic text -> word tokens. `text` may be a string or path lines."""
    if "\n" in text or " " in text:
        raw_lines = text.split("\n")
    else:
        with open(text, encoding="utf-8") as fh:
            raw_lines = fh.read().split("\n")
    lines = []
    for ln in raw_lines:
        if ln.lstrip().startswith("#"):
            continue  # comment / attribution line
        toks = [t for t in re.split(seps, ln.strip()) if t]
        if toks:
            lines.append(toks)
    return lines, None


def _read_stream_clean(stream):
    """Read a symbol stream, skipping # comment lines."""
    if len(stream) < 4096:
        try:
            with open(stream, encoding="utf-8") as fh:
                content = fh.read()
            stream = "\n".join(
                ln for ln in content.split("\n")
                if not ln.lstrip().startswith("#"))
        except (OSError, UnicodeDecodeError):
            pass
    return "".join(c for c in stream if not c.isspace())


def tokenize_chars(stream):
    """Character/symbol stream -> one token per character (no whitespace)."""
    seq = _read_stream_clean(stream)
    return [list(seq)], None


def tokenize_kmers(stream, k=3, alphabet=None):
    """Sliding k-mer tokens over a symbol stream (DNA codons: k=3)."""
    seq = _read_stream_clean(stream).upper()
    if alphabet:
        seq = "".join(c for c in seq if c in alphabet)
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1) if len(seq[i:i + k]) == k]
    return [kmers], None


def tokenize_pulses(values, nbins=12):
    """Numeric interval sequence -> quantized tokens.

    Each interval is expressed as its ratio to the previous interval,
    then quantized into `nbins` log-spaced bins between the observed
    min/max ratio. Output tokens are bin labels like 'r07'.
    `values` may be a list of floats or a path to a whitespace/CSV file.
    """
    if isinstance(values, str):
        with open(values) as fh:
            txt = "\n".join(ln for ln in fh.read().split("\n")
                            if not ln.lstrip().startswith("#"))
        vals = [float(x) for x in re.split(r"[\s,;]+", txt.strip()) if x]
    else:
        vals = [float(v) for v in values]
    if len(vals) < 3:
        return [[]], None
    import math
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1) if vals[i] != 0]
    lo, hi = min(ratios), max(ratios)
    log_lo, log_hi = math.log(lo), math.log(hi)
    toks = []
    for r in ratios:
        if log_hi == log_lo:
            b = 0
        else:
            b = int((math.log(r) - log_lo) / (log_hi - log_lo) * nbins)
            b = max(0, min(nbins - 1, b))
        toks.append("r%02d" % b)
    return [toks], None


def tokenize_bytes(path, chunk=1):
    """Raw byte stream -> byte-value tokens (hex pairs)."""
    with open(path, "rb") as fh:
        data = fh.read()
    toks = [data[i:i + chunk].hex() for i in range(0, len(data), chunk)]
    return [toks], None


MODES = {
    "ivtf_words": tokenize_ivtf_words,
    "words": tokenize_words,
    "chars": tokenize_chars,
    "kmers": tokenize_kmers,
    "pulses": tokenize_pulses,
    "bytes": tokenize_bytes,
}
