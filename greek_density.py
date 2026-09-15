#!/usr/bin/env python3
"""Parse PHerc.1667 preprint transcription: per-column Greek character density.
Counts Greek-script character slots (letters + lacuna dots) per column 5-22."""
import re, json, unicodedata

txt = open('/home/hatch/workspace/pherc1667/preprint.txt').read().splitlines()

def is_greek(s):
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in s)

cols = {}
cur = None
for ln in txt:
    m = re.match(r'Col\.\s*(\d+)', ln)
    if m:
        cur = int(m.group(1)); cols[cur] = []
        continue
    if cur and is_greek(ln):
        # strip line-number prefixes like "5 " or "] " keep content
        cols[cur].append(ln)

out = {}
for c in sorted(cols):
    greek_chars = 0
    dot_slots = 0
    lines = 0
    for ln in cols[c]:
        # remove bracket/line-number artifacts at line start
        t = re.sub(r'^[\]\[\s\d]+', '', ln).strip()
        if not t or not is_greek(t):
            continue
        lines += 1
        for ch in t:
            if '\u0370' <= ch <= '\u03FF' or '\u1F00' <= ch <= '\u1FFF':
                greek_chars += 1
            elif ch in '̣.':
                dot_slots += 1
    out[c] = {'greek_letters': greek_chars, 'lacuna_dots': dot_slots,
              'slots': greek_chars + dot_slots, 'n_lines': lines}
    print('Col %2d: letters=%5d dots=%5d slots=%5d lines=%d' % (
        c, greek_chars, dot_slots, greek_chars + dot_slots, lines))

json.dump(out, open('/home/hatch/workspace/pherc1667/greek_density.json', 'w'), indent=1)
print('wrote greek_density.json')
