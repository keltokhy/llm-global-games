#!/usr/bin/env python3
"""Check human-supplied JEBO portal inputs that cannot be inferred from code."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
METADATA_PATH = PAPER_DIR / "jebo_submission_metadata.json"
METADATA_SCHEMA_PATH = PAPER_DIR / "jebo_submission_metadata.schema.json"
COMPETING_INTEREST_DOCX = PAPER_DIR / "declaration_of_competing_interest.docx"

REQUIRED_METADATA_FIELDS = [
    "corresponding_author.name",
    "corresponding_author.email",
    "corresponding_author.postal_address",
    "corresponding_author.phone",
    "funding_statement",
    "acknowledgments_statement",
    "submission_declaration.approved_by_all_authors",
    "submission_declaration.not_under_consideration_elsewhere",
    "submission_declaration.not_previously_published_except_allowed_preprint",
]


class DependencyNeeded(RuntimeError):
    """Raised when final submission inputs require author-supplied facts."""


def _flatten_schema_required_fields(schema: dict[str, Any], prefix: str = "") -> list[str]:
    fields: list[str] = []
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for key in required:
        dotted_key = f"{prefix}.{key}" if prefix else key
        subschema = properties.get(key, {})
        if subschema.get("type") == "object" and subschema.get("required"):
            fields.extend(_flatten_schema_required_fields(subschema, dotted_key))
        else:
            fields.append(dotted_key)

    return fields


def _check_metadata_schema() -> None:
    if not METADATA_SCHEMA_PATH.exists():
        raise AssertionError(f"{METADATA_SCHEMA_PATH.relative_to(ROOT)} is missing")

    try:
        schema = json.loads(METADATA_SCHEMA_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{METADATA_SCHEMA_PATH.relative_to(ROOT)} is not valid JSON: {exc}"
        ) from exc

    schema_fields = set(_flatten_schema_required_fields(schema))
    required_fields = set(REQUIRED_METADATA_FIELDS)
    if schema_fields != required_fields:
        raise AssertionError(
            "Metadata schema required fields differ from checker fields: "
            f"schema_only={sorted(schema_fields - required_fields)}, "
            f"checker_only={sorted(required_fields - schema_fields)}"
        )


def _nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        fields = ", ".join(REQUIRED_METADATA_FIELDS)
        raise DependencyNeeded(
            f"create {METADATA_PATH.relative_to(ROOT)} following "
            f"{METADATA_SCHEMA_PATH.relative_to(ROOT)} with fields: {fields}"
        )

    try:
        data = json.loads(METADATA_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{METADATA_PATH.relative_to(ROOT)} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise AssertionError(f"{METADATA_PATH.relative_to(ROOT)} must contain a JSON object")
    return data


def _check_metadata(data: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_METADATA_FIELDS if _nested_get(data, field) in (None, "")]
    if missing:
        raise DependencyNeeded(
            f"complete {METADATA_PATH.relative_to(ROOT)} fields: {', '.join(missing)}"
        )

    for field in REQUIRED_METADATA_FIELDS:
        value = _nested_get(data, field)
        if isinstance(value, str) and re.search(r"\b(TODO|TBD|PLACEHOLDER|xxxx|\[|\])\b", value, re.I):
            raise AssertionError(
                f"{METADATA_PATH.relative_to(ROOT)} field {field} contains placeholder text"
            )

    email = str(_nested_get(data, "corresponding_author.email"))
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise AssertionError(f"corresponding_author.email is not a valid email address")

    phone = str(_nested_get(data, "corresponding_author.phone"))
    if len(re.findall(r"\d", phone)) < 7:
        raise AssertionError("corresponding_author.phone must contain at least seven digits")

    postal_address = str(_nested_get(data, "corresponding_author.postal_address"))
    if len(postal_address) < 20:
        raise AssertionError("corresponding_author.postal_address is too short for a full address")

    funding_statement = str(_nested_get(data, "funding_statement"))
    if "Funding:" not in funding_statement and "did not receive any specific grant" not in funding_statement:
        raise AssertionError(
            "funding_statement must be a complete funding disclosure or the standard no-funding sentence"
        )

    for field in [
        "submission_declaration.approved_by_all_authors",
        "submission_declaration.not_under_consideration_elsewhere",
        "submission_declaration.not_previously_published_except_allowed_preprint",
    ]:
        if _nested_get(data, field) is not True:
            raise DependencyNeeded(
                f"set {field}=true in {METADATA_PATH.relative_to(ROOT)} after confirming it is accurate"
            )


def _check_competing_interest_docx() -> None:
    if not COMPETING_INTEREST_DOCX.exists():
        raise DependencyNeeded(
            "export the Elsevier declarations-tool Word document to "
            f"{COMPETING_INTEREST_DOCX.relative_to(ROOT)}"
        )

    try:
        with zipfile.ZipFile(COMPETING_INTEREST_DOCX) as archive:
            archive.read("word/document.xml")
    except Exception as exc:
        raise AssertionError(
            f"{COMPETING_INTEREST_DOCX.relative_to(ROOT)} is not a valid docx"
        ) from exc


def main() -> int:
    dependencies: list[str] = []
    errors: list[str] = []

    metadata: dict[str, Any] | None = None
    try:
        _check_metadata_schema()
        metadata = _load_metadata()
    except DependencyNeeded as exc:
        dependencies.append(str(exc))
    except Exception as exc:
        errors.append(str(exc))

    if metadata is not None:
        try:
            _check_metadata(metadata)
        except DependencyNeeded as exc:
            dependencies.append(str(exc))
        except Exception as exc:
            errors.append(str(exc))

    try:
        _check_competing_interest_docx()
    except DependencyNeeded as exc:
        dependencies.append(str(exc))
    except Exception as exc:
        errors.append(str(exc))

    if errors:
        print("JEBO final input check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if dependencies:
        print("Dependency needed:", file=sys.stderr)
        for dependency in dependencies:
            print(f"- {dependency}", file=sys.stderr)
        return 1

    print("JEBO final portal inputs check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
