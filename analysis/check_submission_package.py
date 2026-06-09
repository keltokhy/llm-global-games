#!/usr/bin/env python3
"""Submission-package checks for built manuscript PDFs and LaTeX logs."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
MAX_SUBMISSION_PAGES = 30

LOG_PROBLEM_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"undefined",
        r"Citation .* undefined",
        r"Reference .* undefined",
        r"There were undefined",
        r"Package natbib Warning",
        r"LaTeX Warning",
        r"pdfTeX warning",
        r"Overfull",
        r"Rerun",
    ]
]

ANONYMOUS_FORBIDDEN_PATTERNS = [
    r"Khaled",
    r"Eltokhy",
    r"keltokhy",
    r"github",
    r"Graduate Center",
    r"CUNY",
    r"/Users/khaled",
    r"llm-global-games",
    r"publicly available",
]


def _need_tool(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"Dependency needed: install `{tool}` or add it to PATH")
    return path


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def _pdfinfo(pdf_path: Path) -> dict[str, str]:
    output = _run([_need_tool("pdfinfo"), str(pdf_path)])
    info: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def _pdftotext(pdf_path: Path) -> str:
    return _run([_need_tool("pdftotext"), str(pdf_path), "-"])


def _check_pdf_exists(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise AssertionError(f"{pdf_path.relative_to(ROOT)} is missing")
    if not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise AssertionError(f"{pdf_path.relative_to(ROOT)} is not a valid PDF")


def _check_latex_log(log_path: Path) -> None:
    if not log_path.exists():
        raise AssertionError(f"{log_path.relative_to(ROOT)} is missing")

    problems: list[str] = []
    for lineno, line in enumerate(log_path.read_text(errors="replace").splitlines(), 1):
        lowered = line.lower()
        if "rerunfilecheck" in lowered or "rerunfilechec" in lowered:
            continue
        if any(pattern.search(line) for pattern in LOG_PROBLEM_PATTERNS):
            problems.append(f"{log_path.relative_to(ROOT)}:{lineno}: {line}")

    if problems:
        raise AssertionError("LaTeX log problems found:\n" + "\n".join(problems))


def _check_main_pdf() -> None:
    pdf_path = PAPER_DIR / "paper.pdf"
    _check_pdf_exists(pdf_path)
    info = _pdfinfo(pdf_path)
    text = _pdftotext(pdf_path)

    if info.get("Author") != "Khaled Eltokhy":
        raise AssertionError(f"paper.pdf Author metadata is {info.get('Author')!r}")
    pages = int(info.get("Pages") or 0)
    if pages <= 0 or pages > MAX_SUBMISSION_PAGES:
        raise AssertionError(f"paper.pdf Pages metadata is {info.get('Pages')!r}")
    if "https://github.com/keltokhy/llm-global-games" not in text:
        raise AssertionError("paper.pdf is missing the public replication URL")
    if "de-identified replication archive" in text:
        raise AssertionError("paper.pdf contains anonymous archive language")


def _check_anonymous_pdf() -> None:
    pdf_path = PAPER_DIR / "paper_anonymous.pdf"
    _check_pdf_exists(pdf_path)
    info = _pdfinfo(pdf_path)
    text = _pdftotext(pdf_path)

    if info.get("Author") != "Anonymous Author(s)":
        raise AssertionError(f"paper_anonymous.pdf Author metadata is {info.get('Author')!r}")
    pages = int(info.get("Pages") or 0)
    if pages <= 0 or pages > MAX_SUBMISSION_PAGES:
        raise AssertionError(f"paper_anonymous.pdf Pages metadata is {info.get('Pages')!r}")
    for pattern in ANONYMOUS_FORBIDDEN_PATTERNS:
        if re.search(pattern, text, re.I):
            raise AssertionError(f"paper_anonymous.pdf leaks identifying text: {pattern}")
    if "de-identified replication archive" not in text:
        raise AssertionError("paper_anonymous.pdf is missing de-identified archive language")


def main() -> int:
    checks = [
        lambda: _check_latex_log(PAPER_DIR / "paper.log"),
        lambda: _check_latex_log(PAPER_DIR / "paper_anonymous.log"),
        _check_main_pdf,
        _check_anonymous_pdf,
    ]

    try:
        for check in checks:
            check()
    except Exception as exc:
        print(f"Submission package check failed: {exc}", file=sys.stderr)
        return 1

    print("Submission package checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
