#!/usr/bin/env python3
"""Build a deterministic JEBO submission-source package."""

from __future__ import annotations

import re
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
PACKAGE_ZIP = PAPER_DIR / "jebo_submission_source.zip"

BASE_FILES = [
    "cover_letter_jebo.pdf",
    "cover_letter_jebo.tex",
    "paper.tex",
    "references.bib",
    "placeins.sty",
    "multirow.sty",
    "paper.pdf",
    "highlights_jebo.docx",
]


def _collect_inputs(path: Path, seen: set[Path]) -> set[Path]:
    if path in seen:
        return set()
    seen.add(path)

    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))

    text = path.read_text(errors="replace")
    files = {path}
    for raw in re.findall(r"\\input\{([^}]+)\}", text):
        input_path = PAPER_DIR / raw
        if not input_path.suffix:
            input_path = input_path.with_suffix(".tex")
        files.update(_collect_inputs(input_path, seen))

    for raw in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        graphic_path = PAPER_DIR / raw
        if not graphic_path.suffix:
            if graphic_path.with_suffix(".pdf").exists():
                graphic_path = graphic_path.with_suffix(".pdf")
            elif graphic_path.with_suffix(".png").exists():
                graphic_path = graphic_path.with_suffix(".png")
        if not graphic_path.exists():
            raise FileNotFoundError(graphic_path.relative_to(ROOT))
        files.add(graphic_path)

    return files


def source_files() -> list[Path]:
    files = {PAPER_DIR / name for name in BASE_FILES}
    files.update(_collect_inputs(PAPER_DIR / "paper.tex", seen=set()))

    missing = [path for path in files if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path.relative_to(ROOT)) for path in sorted(missing))
        raise FileNotFoundError(f"Missing submission package files: {missing_list}")

    return sorted(files, key=lambda path: path.relative_to(PAPER_DIR).as_posix())


def build_package(output_path: Path = PACKAGE_ZIP) -> None:
    files = source_files()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(PAPER_DIR).as_posix())


def main() -> int:
    build_package()
    print(f"Wrote {PACKAGE_ZIP.relative_to(ROOT)} with {len(source_files())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
