"""report.py — JSON + Markdown report writer for declassification runs."""

import json
import os


def write_reports(outdir, slug, result):
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, slug + "_report.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    mpath = os.path.join(outdir, slug + "_report.md")
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(result))
    return jpath, mpath


def render_markdown(r):
    L = []
    L.append("# Declassification report — %s" % r.get("corpus", "unknown"))
    L.append("")
    L.append("Mode: `%s` | Source: `%s`" % (r.get("mode"), r.get("source")))
    L.append("")
    v = r["vocabulary"]
    L.append("## Vocabulary architecture")
    L.append("- tokens: %d | types: %d | TTR: %.4f | hapax: %d" % (
        v["tokens"], v["types"], v["ttr"], v["hapax"]))
    L.append("- top types: " + ", ".join(
        "%s×%d" % (t, c) for t, c in v["top"][:12]))
    if "shared_skeleton_types" in v:
        L.append("- shared skeleton across %d sections: **%d** types" % (
            len(v["sections"]), v["shared_skeleton_types"]))
        L.append("- skeleton top: " + ", ".join(
            "%s×%d" % (t, c) for t, c in v["shared_skeleton_top"][:10]))
        L.append("- section-exclusive types: " + ", ".join(
            "%s=%d" % (s, n) for s, n in v["exclusive_types"].items()))
    L.append("")
    p = r["positional"]
    L.append("## Positional grammar")
    for pos in ("initial", "medial", "final"):
        top = p[pos + "_top"][:8]
        L.append("- %s: " % pos + ", ".join(
            "%s×%d (%.1f×)" % (t, k, b) for t, k, _, b in top))
    L.append("- strongest initial-biased: " + ", ".join(
        "%s %.1f×" % (t, b) for t, b, _, _ in p["initial_biased_top"][:10]))
    L.append("")
    ph = r["phrases"]
    L.append("## Repeated phrase skeletons")
    for n in ("2gram", "3gram"):
        rows = ph.get(n, [])[:8]
        L.append("- " + n + ": " + "; ".join(
            "'%s'×%d%s" % (s, k, (" [" + str(sp) + " sections]" if sp else ""))
            for s, k, sp in rows))
    L.append("")
    ln = r["lengths"]
    L.append("## Token-length structure")
    L.append("- mode length: **%d** (%.1f%%)" % (
        ln["mode_length"], ln["mode_share"]))
    L.append("- Fibonacci-length share: **%.1f%%**" % ln["fibonacci_share_pct"])
    L.append("- mean length: %.2f" % ln["mean_length"])
    L.append("")
    s = r["symbols"]
    L.append("## Symbol architecture")
    L.append("- alphabet: %d symbols | %d total" % (
        s["alphabet_size"], s["total_symbols"]))
    L.append("- top symbols: " + ", ".join(
        "%s×%d" % (a, c) for a, c in s["alphabet_top"][:15]))
    g = s.get("gallows_like")
    if g:
        L.append("- gallows-like %s: initial %.1f%% / medial %.1f%% / final %.1f%% — bias **%.2f×**" % (
            g["set"], g["initial_pct"], g["medial_pct"], g["final_pct"],
            g["initial_medial_bias"]))
    L.append("")
    L.append("## Records cross-reference")
    L.append("")
    L.append("Read through the Records (phi_quantum_engine_ledger.json) —")
    L.append("the sealed reference frame, not in isolation.")
    L.append("")
    L.append(r.get("records_crossref_md", "No record comparisons applied.\n"))
    L.append("---")
    L.append("*Structural output only — no invented meanings or translations.*")
    return "\n".join(L) + "\n"
