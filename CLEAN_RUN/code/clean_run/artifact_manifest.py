"""Build a hash manifest for CLEAN_RUN artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .config import CLEAN_RUN_ROOT


EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".tmp"}


def build_artifact_manifest(root: str | Path = CLEAN_RUN_ROOT) -> pd.DataFrame:
    root_path = Path(root).resolve()
    rows = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root_path).as_posix()
        if _excluded(path, rel):
            continue
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def _excluded(path: Path, rel: str) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if rel == "artifacts/artifact_manifest.tsv":
        return True
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CLEAN_RUN artifact hash manifest")
    parser.add_argument("--root", default=str(CLEAN_RUN_ROOT))
    parser.add_argument("--output", default=str(CLEAN_RUN_ROOT / "artifacts" / "artifact_manifest.tsv"))
    args = parser.parse_args()

    df = build_artifact_manifest(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, sep="\t", index=False)
    print(f"wrote {output} ({len(df)} files)")


if __name__ == "__main__":
    main()
