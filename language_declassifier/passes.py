"""passes.py — structural passes for the universal language declassifier.

Every pass takes ordered token lines and returns plain data. No meanings,
no translations — structure only.

Passes
------
1. vocabulary      token frequencies, rank list, type/token counts,
                   shared skeleton (types in every section) and
                   section-exclusive types.
2. positional      initial / medial / final token distributions and
                   bias ratios per token.
3. phrases         repeated 2-5-token skeletons with counts and spread.
4. lengths         token-length distribution, Fibonacci-length share,
                   mode length.
5. symbols         character/symbol alphabet, frequencies, and the
                   initial-symbol bias of a configurable "gallows-like"
                   symbol set (default: Voynich gallows k t p f).
"""

from collections import Counter, defaultdict

FIBONACCI = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}


def pass_vocabulary(lines, sections=None):
    freq = Counter(t for ln in lines for t in ln)
    n_tok = sum(freq.values())
    n_typ = len(freq)
    out = {
        "tokens": n_tok,
        "types": n_typ,
        "ttr": n_typ / n_tok if n_tok else 0,
        "top": freq.most_common(25),
        "hapax": sum(1 for c in freq.values() if c == 1),
    }
    if sections:
        per = defaultdict(Counter)
        for ln, s in zip(lines, sections):
            per[s].update(ln)
        sec_names = sorted(per)
        shared = set(per[sec_names[0]])
        for s in sec_names[1:]:
            shared &= set(per[s])
        out["sections"] = sec_names
        out["shared_skeleton_types"] = len(shared)
        out["shared_skeleton_top"] = Counter(
            {t: sum(per[s][t] for s in sec_names) for t in shared}
        ).most_common(15)
        out["exclusive_types"] = {
            s: len([t for t in per[s] if all(t not in per[o] for o in sec_names if o != s)])
            for s in sec_names
        }
    return out


def pass_positional(lines, min_count=10):
    """Initial / medial / final distributions; bias = share_pos / share_all."""
    total = Counter(t for ln in lines for t in ln)
    init = Counter()
    med = Counter()
    fin = Counter()
    n_init = n_med = n_fin = 0
    for ln in lines:
        if not ln:
            continue
        init[ln[0]] += 1
        n_init += 1
        fin[ln[-1]] += 1
        n_fin += 1
        for t in ln[1:-1]:
            med[t] += 1
            n_med += 1
    n_all = n_init + n_med + n_fin
    out = {"n_initial": n_init, "n_medial": n_med, "n_final": n_fin}
    for name, c, n in (("initial", init, n_init), ("medial", med, n_med),
                       ("final", fin, n_fin)):
        top = []
        for t, k in c.most_common(15):
            share_pos = k / n if n else 0
            share_all = total[t] / n_all if n_all else 0
            top.append([t, k, round(share_pos, 4),
                        round(share_pos / share_all, 2) if share_all else 0])
        out[name + "_top"] = top
    biased = []
    for t, k in init.items():
        if total[t] >= min_count:
            share_i = k / n_init
            share_m = med[t] / n_med if n_med else 0
            if share_m > 0:
                biased.append([t, round(share_i / share_m, 2), k, med[t]])
    biased.sort(key=lambda r: -r[1])
    out["initial_biased_top"] = biased[:15]
    return out


def pass_phrases(lines, ns=(2, 3, 4, 5), top=20, sections=None):
    out = {}
    for n in ns:
        c = Counter()
        spread = defaultdict(set)
        for li, ln in enumerate(lines):
            for j in range(len(ln) - n + 1):
                ng = tuple(ln[j:j + n])
                c[ng] += 1
                if sections:
                    spread[ng].add(sections[li])
        rows = []
        for ng, k in c.most_common(top):
            rows.append([" ".join(ng), k, len(spread[ng]) if sections else None])
        out["%dgram" % n] = rows
    return out


def pass_lengths(lines):
    lens = [len(t) for ln in lines for t in ln]
    if not lens:
        return {}
    c = Counter(lens)
    n = len(lens)
    fib_n = sum(1 for L in lens if L in FIBONACCI)
    mode = c.most_common(1)[0][0]
    return {
        "n": n,
        "mode_length": mode,
        "mode_share": round(c[mode] / n * 100, 2),
        "fibonacci_share_pct": round(fib_n / n * 100, 2),
        "mean_length": round(sum(lens) / n, 3),
        "length_dist": sorted([[L, k, round(k / n * 100, 2)] for L, k in c.items()]),
    }


def pass_symbols(lines, gallows_like=None):
    """Alphabet frequencies + initial-symbol bias of a symbol set.

    gallows_like: iterable of symbols treated as the "gallows-like" set
    (default {'k','t','p','f'} — the Voynich gallows). For other corpora
    pass the corpus's own structurally salient symbol set, or None to
    skip the bias computation.
    """
    alpha = Counter()
    init_alpha = Counter()
    n_init = 0
    for ln in lines:
        for t in ln:
            for ch in t:
                alpha[ch] += 1
        if ln and ln[0]:
            init_alpha[ln[0][0]] += 1
            n_init += 1
    out = {
        "alphabet_size": len(alpha),
        "alphabet_top": alpha.most_common(30),
        "total_symbols": sum(alpha.values()),
    }
    if gallows_like:
        gset = set(gallows_like)
        gi = sum(init_alpha.get(g, 0) for g in gset)
        init_share = gi / n_init if n_init else 0
        nm = nf = gm = gf = 0
        for ln in lines:
            for j, t in enumerate(ln):
                if not t:
                    continue
                g = t[0] in gset
                if 0 < j < len(ln) - 1:
                    nm += 1
                    gm += g
                elif j == len(ln) - 1 and len(ln) > 1:
                    nf += 1
                    gf += g
        med_share = gm / nm if nm else 0
        fin_share = gf / nf if nf else 0
        out["gallows_like"] = {
            "set": sorted(gset),
            "initial_pct": round(init_share * 100, 2),
            "medial_pct": round(med_share * 100, 2),
            "final_pct": round(fin_share * 100, 2),
            "initial_medial_bias": round(init_share / med_share, 2) if med_share else None,
        }
    return out


def run_all(lines, sections=None, gallows_like=("k", "t", "p", "f")):
    return {
        "corpus_lines": len(lines),
        "vocabulary": pass_vocabulary(lines, sections),
        "positional": pass_positional(lines),
        "phrases": pass_phrases(lines, sections=sections),
        "lengths": pass_lengths(lines),
        "symbols": pass_symbols(lines, gallows_like),
    }
