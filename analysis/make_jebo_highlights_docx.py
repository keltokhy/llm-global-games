#!/usr/bin/env python3
"""Build the JEBO Highlights Word file from the plain-text source."""

from __future__ import annotations

from html import escape
from pathlib import Path
import zipfile

from check_jebo_submission import HIGHLIGHTS_DOCX, read_highlights


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""


def _rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def _document_xml(highlights: list[str]) -> str:
    bullet = "&#8226;"
    paragraphs = [
        "<w:p><w:r><w:t>Highlights</w:t></w:r></w:p>",
    ]
    for highlight in highlights:
        paragraphs.append(
            f"<w:p><w:r><w:t>{bullet} {escape(highlight)}</w:t></w:r></w:p>"
        )

    body = "\n".join(paragraphs)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>
"""


def build_docx(output_path: Path = HIGHLIGHTS_DOCX) -> None:
    highlights = read_highlights()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _rels_xml())
        archive.writestr("word/document.xml", _document_xml(highlights))


def main() -> int:
    build_docx()
    print(f"Wrote {HIGHLIGHTS_DOCX.relative_to(HIGHLIGHTS_DOCX.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
