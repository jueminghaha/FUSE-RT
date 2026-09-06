#!/usr/bin/env python3
"""Merge RepoRT per-method metadata TSVs into the configured global CSV.

Uses only the Python standard library. Fields are matched by header name,
method IDs remain four-digit strings, and source blanks are not imputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_method_metadata(path: Path, method_id: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = reader.fieldnames
        if not headers or "id" not in headers:
            raise ValueError(f"Metadata must have an id column: {path}")
        if any(not header.strip() for header in headers) or len(set(headers)) != len(headers):
            raise ValueError(f"Empty or duplicate metadata column name: {path}")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row, found {len(rows)}: {path}")
    row = rows[0]
    if None in row or any(value is None for value in row.values()):
        raise ValueError(f"Metadata row width does not match its header: {path}")
    source_id = row["id"].strip()
    if not re.fullmatch(r"[0-9]{1,4}", source_id) or source_id.zfill(4) != method_id:
        raise ValueError(f"Folder ID {method_id} differs from metadata ID {source_id!r}: {path}")
    row["id"] = method_id
    return row


def build_global_metadata(processed_dir: Path, output_csv: Path, *, overwrite: bool = False) -> int:
    processed_dir = Path(processed_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"RepoRT processed_data directory not found: {processed_dir}")
    if output_csv.suffix.lower() != ".csv":
        raise ValueError(f"Output must be a .csv file: {output_csv}")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite to replace it: {output_csv}")
    folders = sorted(path for path in processed_dir.iterdir()
                     if path.is_dir() and re.fullmatch(r"[0-9]{4}", path.name))
    if not folders:
        raise ValueError(f"No four-digit method folders found: {processed_dir}")

    # Validate all inputs before opening the output, including on --overwrite.
    rows = []
    fieldnames = ["id"]
    seen_fields = {"id"}
    for folder in folders:
        path = folder / f"{folder.name}_metadata.tsv"
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file missing for method {folder.name}: {path}")
        row = read_method_metadata(path, folder.name)
        rows.append(row)
        for field in row:
            if field not in seen_fields:
                fieldnames.append(field)
                seen_fields.add(field)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w" if overwrite else "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path,
                        help="RepoRT directory containing processed_data; defaults to config/paths.json.")
    parser.add_argument("--output", type=Path,
                        help="Output CSV; defaults to metadata_csv in config/paths.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / "config/paths.json").read_text(encoding="utf-8"))
    report_root = (args.report_root.expanduser().resolve() if args.report_root is not None
                   else project_path(config["report_root"]))
    output_csv = (args.output.expanduser().resolve() if args.output is not None
                  else project_path(config["metadata_csv"]))
    try:
        count = build_global_metadata(report_root / "processed_data", output_csv,
                                      overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Merged {count} methods from: {report_root / 'processed_data'}")
    print(f"Global metadata CSV: {output_csv}")
    print("Original field values and blanks are preserved; no missing values were imputed.")


if __name__ == "__main__":
    main()
