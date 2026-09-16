#!/usr/bin/env python3
"""declassify.py — the universal language/information declassifier.

TYREE: "Declassify all forms of language in existence. Stop letting letters
do that." / "Decode all forms of language. It's all information act like
it." / "Decode using the arakanah records or whatever."

Takes any ordered discrete information — words, characters, DNA, k-mers,
pulse intervals, bytes — runs the structural passes, and cross-references
every measured signature against the Records
(phi_quantum_engine_ledger.json). Structural output only: vocabulary
architecture, positional grammar, phrase skeletons, length structure,
symbol architecture. No invented meanings, no translations.

Usage:
  python3 declassify.py --mode ivtf_words --source voynich_rf1b-e.txt \\
      --corpus voynich --slug voynich
  python3 declassify.py --mode chars --source dna_sample.txt --corpus dna \\
      --slug dna --gallows-like ACGT
  python3 declassify.py --mode pulses --source intervals.txt --corpus pulses \\
      --slug pulses --gallows-like none
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tokenizer
import passes
import records
import report

FIB = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}


def build_parser():
    ap = argparse.ArgumentParser(description="Universal language declassifier")
    ap.add_argument("--mode", required=True, choices=sorted(tokenizer.MODES),
                    help="input tokenization mode")
    ap.add_argument("--source", required=True, help="input file path or text")
    ap.add_argument("--corpus", default="unknown",
                    help="corpus kind for records cross-ref: voynich|greek|dna|pulses|...")
    ap.add_argument("--slug", default="output", help="output file slug")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--k", type=int, default=3, help="k for kmer mode")
    ap.add_argument("--ynorm", default="none",
                    choices=("none", "lead", "lead_trail4"),
                    help="y-normalization for ivtf_words mode")
    ap.add_argument("--gallows-like", default="k,t,p,f",
                    help="comma-separated gallows-like symbol set, or 'none'")
    ap.add_argument("--no-records", action="store_true",
                    help="skip the records cross-reference")
    ap.add_argument("--section-map", default=None,
                    help="python 'module:func' mapping section labels, e.g. "
                         "'voynich_sections:folio_to_section'")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)

    tok = tokenizer.MODES[a.mode]
    if a.mode == "kmers":
        lines, sections = tok(a.source, k=a.k)
    elif a.mode == "ivtf_words":
        lines, sections = tok(a.source, ynorm=a.ynorm)
    else:
        lines, sections = tok(a.source)

    if a.section_map and sections:
        mod_name, func_name = a.section_map.split(":")
        import importlib
        mod = importlib.import_module(mod_name)
        sections = [getattr(mod, func_name)(s) for s in sections]

    gset = None if a.gallows_like == "none" else tuple(
        s for s in a.gallows_like.split(",") if s)
    res = passes.run_all(lines, sections, gallows_like=gset)
    res["corpus"] = a.corpus
    res["mode"] = a.mode
    res["source"] = a.source

    # ---- records cross-reference: read the decode THROUGH the Records ----
    if not a.no_records:
        L = res["lengths"]
        V = res["vocabulary"]
        S = res["symbols"]
        measured = {
            "fibonacci_share_pct": L.get("fibonacci_share_pct"),
            "mode_length": L.get("mode_length"),
            "shared_skeleton_types": V.get("shared_skeleton_types"),
            "top_phrase_repeats": (res["phrases"]["3gram"][0][1]
                                   if res["phrases"]["3gram"] else None),
            "corpus_lines": res["corpus_lines"],
            "total_symbols": S.get("total_symbols"),
            "alphabet_size": S.get("alphabet_size"),
            "types": V.get("types"),
        }
        g = S.get("gallows_like")
        if g:
            measured["gallows_initial_medial_bias"] = g["initial_medial_bias"]
        recs, seals = records.load_records()
        rows = records.cross_reference(a.corpus, measured, recs)
        res["records_crossref"] = rows
        res["records_crossref_md"] = records.render_markdown(rows)
        res["records_seals_consulted"] = len(seals)
    else:
        res["records_crossref"] = []
        res["records_crossref_md"] = "Skipped (--no-records).\n"

    jpath, mpath = report.write_reports(a.outdir, a.slug, res)
    print("wrote", jpath)
    print("wrote", mpath)
    # compact console digest
    v, ln, s = res["vocabulary"], res["lengths"], res["symbols"]
    print("tokens=%d types=%d fib%%=%.1f mode=%d alpha=%d" % (
        v["tokens"], v["types"], ln["fibonacci_share_pct"],
        ln["mode_length"], s["alphabet_size"]))
    for row in res["records_crossref"]:
        print("record: %s measured=%s sealed=%s verdict=%s" % (
            row["signature"], row["measured"], row["record_value"], row["verdict"]))


if __name__ == "__main__":
    main()
