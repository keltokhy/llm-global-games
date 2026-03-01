#!/usr/bin/env python3
"""
Lint: find bare decimal numbers in paper.tex that should be LaTeX macros.

Reports lines containing patterns like 0.83, -0.42, 43.6 that are NOT
inside \\providecommand, \\newcommand, \\def, macro invocations, or
common TeX constructs (coordinates, font sizes, tikz, etc.).

Usage: uv run python analysis/check_paper_numbers.py
"""

import re
import sys
from pathlib import Path

PAPER_PATH = Path(__file__).resolve().parent.parent / "paper" / "paper.tex"

# Patterns to skip: lines that are clearly not hand-typed statistics
SKIP_LINE_PATTERNS = [
    re.compile(r"\\(providecommand|newcommand|renewcommand|def)\b"),
    re.compile(r"^%"),                          # comments
    re.compile(r"\\input\{"),                   # input directives
    re.compile(r"\\bibliography"),
    re.compile(r"\\(label|ref|cite|eqref)\{"),
    re.compile(r"\\(begin|end)\{"),
    re.compile(r"\\(usepackage|documentclass)"),
]

# Bare decimal: digit(s) . digit(s), possibly preceded by a sign
BARE_DECIMAL = re.compile(r"(?<![\\a-zA-Z{])(-?\d+\.\d+)")

# Patterns that are OK in context (not paper statistics)
OK_CONTEXT = [
    re.compile(r"\\(textwidth|columnwidth|linewidth|baselineskip)"),
    re.compile(r"(width|height|scale|trim|clip)\s*="),
    re.compile(r"\\(hspace|vspace|kern|skip)"),
    re.compile(r"\\(fontsize|setlength|addtolength)"),
    re.compile(r"(em|ex|pt|mm|cm|in|bp|pc)\b"),
    re.compile(r"\\tikz"),
    re.compile(r"pgf"),
    re.compile(r"\\(geometry|margin)"),
    re.compile(r"\\captionsetup"),
    re.compile(r"\\setcounter"),
    re.compile(r"\\(small|footnotesize|scriptsize|tiny|large|Large)"),
]


def check_paper():
    if not PAPER_PATH.exists():
        print(f"ERROR: {PAPER_PATH} not found")
        sys.exit(1)

    lines = PAPER_PATH.read_text(encoding="utf-8").splitlines()
    warnings = []

    for i, line in enumerate(lines, 1):
        # Skip structural lines
        if any(p.search(line) for p in SKIP_LINE_PATTERNS):
            continue

        # Find bare decimals
        matches = BARE_DECIMAL.findall(line)
        if not matches:
            continue

        # Skip if line is clearly a layout/style context
        if any(p.search(line) for p in OK_CONTEXT):
            continue

        for m in matches:
            warnings.append((i, m, line.strip()[:100]))

    print(f"Bare decimal numbers in paper.tex: {len(warnings)}")
    print(f"(These may be hand-typed statistics that should use LaTeX macros)")
    print()

    for lineno, number, context in warnings:
        print(f"  L{lineno:4d}: {number:>8s}  {context}")

    return len(warnings)


if __name__ == "__main__":
    n = check_paper()
    sys.exit(0)  # informational only, don't fail the build
