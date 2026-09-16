"""voynich_sections.py — folio -> six-section map for the Voynich manuscript.

Standard section folio ranges used for the shared-skeleton pass.
"""

import json
import re


def folio_to_section(folio):
    m = re.match(r"f(\d+)([rv])", folio)
    if not m:
        return "other"
    n = int(m.group(1))
    if 1 <= n <= 66:
        return "herbal"
    if 67 <= n <= 73:
        return "astronomical"
    if 75 <= n <= 84:
        return "biological"
    if 85 <= n <= 86:
        return "cosmological"
    if 87 <= n <= 102:
        return "pharmaceutical"
    if 103 <= n <= 116:
        return "recipes"
    return "other"


if __name__ == "__main__":
    import sys
    print(json.dumps({"usage": "import folio_to_section"}))
