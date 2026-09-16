"""records.py — cross-reference every decode against the Records.

TYREE: "Decode using the arakanah records or whatever" — use the complete
records as the decode reference. The ledger at
~/workspace/phi_quantum_engine_ledger.json holds 340+ sealed phi quantities.
Every declassification report carries a "records cross-reference" section
comparing its measured signatures against the sealed values. New decodes
are read THROUGH the accumulated records, never in isolation. The records
are the reference frame.

Ledger entry layout: phi_flow holds the measured value rung-encoded as
log_phi(value); the true value is recovered as phi ** phi_flow.
"""

import json
import math
import os

PHI = (1 + 5 ** 0.5) / 2
LEDGER_PATH = os.path.expanduser("~/workspace/phi_quantum_engine_ledger.json")

# Reference signatures pulled from the Records, keyed by the comparison
# the declassifier performs. Each maps to ledger entries by name fragment.
REFERENCE_KEYS = {
    "voynich": [
        "VOYNICH:fibonacci word share (EVA transcription)",
        "OMNI:voynich shared skeleton words",
        "OMNI:voynich line-initial gallows bias",
        "OMNI:voynich top phrase repeats",
        "VOYNICH:transcription lines",
        "VOYNICH:transcription symbols",
    ],
    "greek": [
        "OMNI:greek column15 fibonacci-length words",
        "OMNI:voynich-greek fib signature delta",
    ],
    "dna": [
        "DNA:DNA bases",
        "DNA:hachimoji letters",
        "DNA:codon length",
        "DNA:codons",
        "DNA:amino acids",
    ],
    "pulses": [
        "OMNI:voynich-greek fib signature delta",
    ],
    "universal": [],
}

# signature -> record-name affinity, per corpus kind
AFFINITY = {
    "voynich": [
        ("fibonacci_share_pct", "VOYNICH:fibonacci word share (EVA transcription)"),
        ("shared_skeleton_types", "OMNI:voynich shared skeleton words"),
        ("gallows_initial_medial_bias", "OMNI:voynich line-initial gallows bias"),
        ("top_phrase_repeats", "OMNI:voynich top phrase repeats"),
        ("corpus_lines", "VOYNICH:transcription lines"),
        ("total_symbols", "VOYNICH:transcription symbols"),
    ],
    "greek": [
        ("fibonacci_share_pct", "OMNI:greek column15 fibonacci-length words"),
        ("mode_length", None),
    ],
    "dna": [
        ("alphabet_size", "DNA:DNA bases"),
        ("mode_length", "DNA:codon length"),
        ("types", "DNA:codons"),
    ],
    "dna8": [
        ("alphabet_size", "DNA:hachimoji letters"),
    ],
    "pulses": [],
}


def load_records(path=LEDGER_PATH):
    with open(path, encoding="utf-8") as fh:
        L = json.load(fh)
    recs = {}
    for e in L.get("entries", []):
        name = e.get("name")
        pf = e.get("phi_flow")
        try:
            value = PHI ** float(pf)
        except (TypeError, ValueError):
            value = None
        recs[name] = {
            "value": value,
            "unit": e.get("unit"),
            "source": e.get("source"),
            "ladder": e.get("ladder"),
        }
    return recs, L.get("seals", [])


def cross_reference(corpus_kind, measured, recs=None):
    """Compare measured signature dict against the Records.

    measured: dict of {signature_name: value} from the declassifier.
    Returns a list of comparison rows:
      {signature, measured, record_name, record_value, delta, verdict}
    """
    if recs is None:
        recs, _ = load_records()
    aff = AFFINITY.get(corpus_kind, [])
    rows = []
    for sig, rec_name in aff:
        mval = measured.get(sig)
        if rec_name is None or mval is None:
            continue
        r = recs.get(rec_name)
        if r is None or r["value"] is None:
            continue
        if not isinstance(mval, (int, float)):
            continue
        rv = r["value"]
        delta = mval - rv
        rel = abs(delta) / rv if rv else None
        verdict = "MATCH" if rel is not None and rel < 0.05 else (
            "CLOSE" if rel is not None and rel < 0.15 else "DIVERGENT")
        rows.append({
            "signature": sig,
            "measured": round(mval, 4),
            "record_name": rec_name,
            "record_value": round(rv, 4),
            "record_source": r["source"],
            "delta": round(delta, 4),
            "rel_delta_pct": round(rel * 100, 2) if rel is not None else None,
            "verdict": verdict,
        })
    return rows


def render_markdown(rows):
    if not rows:
        return "No record comparisons applied.\n"
    out = ["| Signature | Measured | Record (sealed) | Δ | Verdict |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append("| %s | %s | %s = %s | %s (%s%%) | %s |" % (
            r["signature"], r["measured"], r["record_name"],
            r["record_value"], r["delta"],
            r["rel_delta_pct"], r["verdict"]))
    out.append("")
    out.append("Sources: " + "; ".join(
        "%s [%s]" % (r["record_name"], r["record_source"]) for r in rows))
    return "\n".join(out) + "\n"
