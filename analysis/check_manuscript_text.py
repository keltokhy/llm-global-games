#!/usr/bin/env python3
"""Lightweight manuscript text checks that do not depend on a journal style."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHECKED_FILES = [
    ROOT / "paper" / "paper.tex",
    ROOT / "REPLICATION_README.txt",
    ROOT / "DATA_MANIFEST.txt",
]

COMMON_TYPOS = {
    "adn",
    "avaliable",
    "becuase",
    "calucl",
    "coordiant",
    "dependant",
    "enviroment",
    "equilibir",
    "futher",
    "goverment",
    "hte",
    "implcit",
    "lenght",
    "manuscritp",
    "occurence",
    "publically",
    "recieve",
    "represion",
    "seperate",
    "signfic",
    "similiar",
    "statments",
    "strenght",
    "surveillence",
    "teh",
    "thier",
    "varible",
    "wich",
}

STALE_PHRASES = [
    "reported in supplementary materials",
    "to be added",
    "placeholder",
    "todo",
    "fixme",
    "tbd",
]


def _strip_tex(line: str) -> str:
    line = re.sub(r"\\cite\w*(?:\[[^\]]*\])*\{[^}]*\}", " ", line)
    line = re.sub(r"\$[^$]*\$", " ", line)
    line = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r" \1 ", line)
    line = re.sub(r"[{}]", " ", line)
    return line


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    text = path.read_text(errors="replace")

    for stale in STALE_PHRASES:
        if re.search(rf"\b{re.escape(stale)}\b", text, re.I):
            problems.append(f"{path.relative_to(ROOT)}: stale phrase found: {stale}")

    for lineno, line in enumerate(text.splitlines(), 1):
        if path.suffix == ".tex" and re.match(
            r"\s*\\(documentclass|usepackage|newenvironment|newtheorem|newif|ifdefined|else|fi)\b",
            line,
        ):
            continue
        stripped = _strip_tex(line)

        for match in re.finditer(r"\b([A-Za-z]{3,})\s+\1\b", stripped, re.I):
            problems.append(
                f"{path.relative_to(ROOT)}:{lineno}: repeated word: {match.group(0)}"
            )

        lowered_words = {word.lower() for word in re.findall(r"[A-Za-z]+", stripped)}
        for typo in sorted(COMMON_TYPOS & lowered_words):
            problems.append(f"{path.relative_to(ROOT)}:{lineno}: possible typo: {typo}")

    return problems


def main() -> int:
    missing = [path for path in CHECKED_FILES if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing checked file: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1

    problems: list[str] = []
    for path in CHECKED_FILES:
        problems.extend(check_file(path))

    if problems:
        print("Manuscript text check failed:", file=sys.stderr)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    print("Manuscript text checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
