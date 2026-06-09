#!/usr/bin/env python3
"""JEBO-specific submission checks."""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

from build_jebo_submission_package import PACKAGE_ZIP, source_files


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
HIGHLIGHTS_TXT = PAPER_DIR / "highlights_jebo.txt"
HIGHLIGHTS_DOCX = PAPER_DIR / "highlights_jebo.docx"
MAX_HIGHLIGHT_CHARS = 85
AI_DECLARATION_TITLE = (
    "Declaration of Generative AI and AI-Assisted Technologies in the Writing Process"
)


def read_highlights(path: Path = HIGHLIGHTS_TXT) -> list[str]:
    if not path.exists():
        raise AssertionError(f"{path.relative_to(ROOT)} is missing")

    highlights: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if not line.startswith("- "):
            raise AssertionError(
                f"{path.relative_to(ROOT)}:{lineno}: highlight must start with '- '"
            )
        highlight = line[2:].strip()
        if not highlight:
            raise AssertionError(
                f"{path.relative_to(ROOT)}:{lineno}: empty highlight"
            )
        if len(highlight) > MAX_HIGHLIGHT_CHARS:
            raise AssertionError(
                f"{path.relative_to(ROOT)}:{lineno}: {len(highlight)} characters"
            )
        if re.search(r"\\cite|doi:|https?://|\([12][0-9]{3}\)", highlight, re.I):
            raise AssertionError(
                f"{path.relative_to(ROOT)}:{lineno}: references are not allowed"
            )
        highlights.append(highlight)

    if not 3 <= len(highlights) <= 5:
        raise AssertionError(
            f"{path.relative_to(ROOT)} must contain three to five highlights"
        )

    return highlights


def _check_highlights_docx(highlights: list[str]) -> None:
    if not HIGHLIGHTS_DOCX.exists():
        raise AssertionError(
            f"{HIGHLIGHTS_DOCX.relative_to(ROOT)} is missing; run "
            "`uv run python analysis/make_jebo_highlights_docx.py`"
        )

    try:
        with zipfile.ZipFile(HIGHLIGHTS_DOCX) as archive:
            document_xml = archive.read("word/document.xml").decode()
    except Exception as exc:
        raise AssertionError(f"{HIGHLIGHTS_DOCX.relative_to(ROOT)} is not a valid docx") from exc

    for highlight in highlights:
        if highlight not in document_xml:
            raise AssertionError(
                f"{HIGHLIGHTS_DOCX.relative_to(ROOT)} is missing highlight: {highlight}"
            )


def _check_ai_declaration() -> None:
    tex_path = PAPER_DIR / "paper.tex"
    tex = tex_path.read_text()
    declaration = rf"\section*{{{AI_DECLARATION_TITLE}}}"

    declaration_pos = tex.find(declaration)
    references_pos = tex.find("%% REFERENCES")
    conclusion_pos = tex.find(r"\section{Conclusion}")

    if declaration_pos == -1:
        raise AssertionError(f"{tex_path.relative_to(ROOT)} is missing the AI declaration")
    if not (conclusion_pos < declaration_pos < references_pos):
        raise AssertionError("AI declaration must appear after the conclusion and before references")

    declaration_block = tex[declaration_pos:references_pos]
    required_phrases = [
        "OpenAI Codex",
        "reviewed and edited",
        "takes full responsibility",
        "manuscript preparation only",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in declaration_block]
    if missing:
        raise AssertionError(f"AI declaration missing phrases: {missing}")


def _check_cover_letter() -> None:
    tex_path = PAPER_DIR / "cover_letter_jebo.tex"
    pdf_path = PAPER_DIR / "cover_letter_jebo.pdf"

    if not tex_path.exists():
        raise AssertionError(f"{tex_path.relative_to(ROOT)} is missing")
    if not pdf_path.exists():
        raise AssertionError(
            f"{pdf_path.relative_to(ROOT)} is missing; run `make jebo-cover-letter`"
        )
    if not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise AssertionError(f"{pdf_path.relative_to(ROOT)} is not a valid PDF")

    tex = tex_path.read_text()
    required_phrases = [
        "Journal of Economic Behavior \\& Organization",
        "Speaking in Code: Surveillance and Coordinated Dissent in a Language-Based Global Game with LLM Agents",
        "global game",
        "sender-side surveillance effect",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in tex]
    if missing:
        raise AssertionError(f"Cover letter missing phrases: {missing}")

    forbidden = re.compile(r"\b(Funding:|competing interest|suggested reviewer|opposed reviewer)\b", re.I)
    if forbidden.search(tex):
        raise AssertionError("Cover letter contains portal-only material")


def _check_submission_source_zip() -> None:
    if not PACKAGE_ZIP.exists():
        raise AssertionError(
            f"{PACKAGE_ZIP.relative_to(ROOT)} is missing; run "
            "`uv run python analysis/build_jebo_submission_package.py`"
        )

    expected = {path.relative_to(PAPER_DIR).as_posix() for path in source_files()}
    with zipfile.ZipFile(PACKAGE_ZIP) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}

    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise AssertionError(
            "Submission source zip contents differ from expected files: "
            f"missing={missing}, extra={extra}"
        )

    required_names = {
        "cover_letter_jebo.pdf",
        "cover_letter_jebo.tex",
        "paper.tex",
        "references.bib",
        "paper.pdf",
        "highlights_jebo.docx",
    }
    missing_required = sorted(required_names - names)
    if missing_required:
        raise AssertionError(
            f"Submission source zip is missing required files: {missing_required}"
        )

    if not any(name.startswith("figures/") and name.endswith(".pdf") for name in names):
        raise AssertionError("Submission source zip does not include separate figure PDFs")
    if not any(name.startswith("tables/") and name.endswith(".tex") for name in names):
        raise AssertionError("Submission source zip does not include editable table sources")


def main() -> int:
    try:
        highlights = read_highlights()
        _check_highlights_docx(highlights)
        _check_ai_declaration()
        _check_cover_letter()
        _check_submission_source_zip()
    except Exception as exc:
        print(f"JEBO submission check failed: {exc}", file=sys.stderr)
        return 1

    print("JEBO submission checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
