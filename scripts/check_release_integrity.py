#!/usr/bin/env python3
"""Check source-preservation and artifact-boundary invariants for this release."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PUBLIC_SOURCE_MANIFEST.csv"
BLOCKED_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".docx",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".html",
    ".ipynb",
    ".pkl",
    ".joblib",
    ".dill",
    ".cpkt",
    ".whl",
    ".zip",
    ".log",
}
BLOCKED_PATTERNS = [
    "after_imputation",
    "before_imputation",
    "prediction",
    "predictions",
    "df_cleaned",
    "df_ppv",
]
BLOCKED_TEXT_PATTERNS = [
    "/content/drive/" + "My" + "Drive",
    "001_" + "\u5927\u5b66\u9662",
    "Graduate" + "School",
    "DSDP_" + "JRVS",
    "RRD_" + "Prediction_" + "Code",
]
TEXT_SUFFIXES = {
    ".csv",
    ".md",
    ".py",
    ".txt",
    ".yml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    failures: list[str] = []

    for row in rows:
        rel_path = row["release_relative_path"]
        path = ROOT / rel_path
        if not path.is_file():
            failures.append(f"missing manifest file: {rel_path}")
            continue
        if sha256_file(path) != row["release_sha256"]:
            failures.append(f"release hash changed: {rel_path}")
        if rel_path.endswith(".py") and row["source_root"] == "review":
            if row["source_sha256"] != row["release_sha256"]:
                failures.append(f"analysis-used Python not byte-identical: {rel_path}")
        if row["notebook_code_source_sha256"] or row["notebook_code_release_sha256"]:
            hashes_match = row["notebook_code_source_sha256"] == row["notebook_code_release_sha256"]
            role_notes_sanitized = "sanitized" in row["role"].lower()
            if not hashes_match and not role_notes_sanitized:
                failures.append(f"notebook code-cell hash changed without sanitized role note: {rel_path}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked generated/binary suffix: {rel}")
        lowered = rel.lower()
        if path.suffix.lower() == ".csv" and any(pattern in lowered for pattern in BLOCKED_PATTERNS):
            failures.append(f"blocked patient-derived filename pattern: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            for pattern in BLOCKED_TEXT_PATTERNS:
                if pattern in text:
                    failures.append(f"blocked private path text pattern {pattern!r}: {rel}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("Release integrity check passed.")


if __name__ == "__main__":
    main()
