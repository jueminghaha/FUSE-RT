#!/usr/bin/env python3
"""Rebuild the frozen 179-method training registry from a local RepoRT snapshot.

This is the script form of
``Strict_RP_RadonPy_Overlap_gt50_From_Full_RepoRT.ipynb``.  It deliberately
does not read an existing split or a ``global_mol_key`` directory.

Frozen construction:

1. strict RP + no observed variable flow + binary A/B + at least 50 RT rows;
2. unique-molecule RadonPy overlap greater than 50%;
3. valid gradient information: 190 -> 178 methods;
4. move 0097 and 0238 from the 178 into internal OOD;
5. add the documented exceptions 0053, 0069, and 0126: 176 + 3 = 179.

Method 0055 is retained explicitly because the frozen notebook treated its
missing flow information as eligible.  Some RepoRT snapshots encode that row
differently, so the raw flow classification and the frozen-protocol override
are both written to the audit instead of silently changing the 179 methods.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from rdkit import Chem, RDLogger, rdBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_CONFIG = PROJECT_ROOT / "config" / "paths.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "method_selection_179"

RT_TSV_TEMPLATE = "{method_id}_rtdata_canonical_success.tsv"
GRADIENT_TSV_TEMPLATE = "{method_id}_gradient.tsv"
INFO_TSV_TEMPLATE = "{method_id}_info.tsv"
METADATA_TSV_TEMPLATE = "{method_id}_metadata.tsv"

SMILES_COLUMN = "smiles.std"
RT_COLUMN = "rt"
FLOW_COLUMN = "flow rate [ml/min]"
STRICT_METHOD_TYPE = "RP"
MIN_RT_ROWS = 50
OVERLAP_THRESHOLD = 0.50
SOURCE_MAX_METHOD_ID = 392
EXPECTED_SOURCE_METHOD_COUNT = 376

INTERNAL_OOD_IDS = ("0097", "0238")
REMOVED_FROM_STRICT_178 = ("0097", "0238")
MANUALLY_ADDED_IDS = ("0053", "0069", "0126")
FROZEN_FLOW_INCLUSION_IDS = ("0055",)
EXPECTED_INVALID_GRADIENT_IDS = (
    "0011",
    "0012",
    "0017",
    "0062",
    "0211",
    "0212",
    "0213",
    "0217",
    "0220",
    "0221",
    "0223",
    "0224",
)

TIME_COLUMN_CANDIDATES = ("t [min]", "time [min]", "time[min]", "time", "t")
A_COLUMN_CANDIDATES = ("a [%]", "a[%]", "a", "eluent a [%]", "eluent.a")
B_COLUMN_CANDIDATES = ("b [%]", "b[%]", "b", "eluent b [%]", "eluent.b")
BLANK_TOKENS = {"", "nan", "none", "na", "n/a", "<na>", "__na__", "null"}

RDLogger.DisableLog("rdApp.*")


def load_path_config() -> dict[str, str]:
    return json.loads(PATHS_CONFIG.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_method_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Cannot normalize method ID: {value!r}")
    return match.group(1).zfill(4)


def resolve_report_root(path: Path) -> Path:
    candidates = [path, path / "RepoRT_latest"]
    for candidate in candidates:
        if (candidate / "processed_data").is_dir() and (candidate / "raw_data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "RepoRT_latest must contain processed_data/ and raw_data/. Checked:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


def resolve_radonpy_csv(data_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(path)

    candidates = [
        data_dir / "RadonPy_20260611" / "RadonPySM_checkeq_masked.csv",
        data_dir / "RadonPySM_checkeq_masked.csv",
    ]
    candidates.extend(sorted(data_dir.glob("**/*RadonPy*masked*.csv")))
    candidates.extend(sorted(data_dir.glob("**/*checkeq*masked*.csv")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find RadonPySM_checkeq_masked.csv under " + str(data_dir.resolve())
    )


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def clean_smiles(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in BLANK_TOKENS:
        return None
    return text


@lru_cache(maxsize=None)
def connectivity_key(smiles: str) -> str | None:
    """Return the first InChIKey block after selecting the largest fragment."""
    smiles = clean_smiles(smiles)
    if smiles is None:
        return None
    fragment = max(smiles.split("."), key=len)
    molecule = Chem.MolFromSmiles(fragment)
    if molecule is None:
        return None
    try:
        inchikey = Chem.MolToInchiKey(molecule)
    except Exception:
        return None
    return inchikey.split("-")[0] if inchikey else None


def build_metadata_from_folders(processed_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for folder in sorted(processed_dir.glob("[0-9][0-9][0-9][0-9]")):
        method_id = folder.name
        if int(method_id) > SOURCE_MAX_METHOD_ID:
            continue
        record: dict[str, object] = {"dataset_id": method_id}
        path = folder / METADATA_TSV_TEMPLATE.format(method_id=method_id)
        if path.is_file():
            try:
                frame = pd.read_csv(path, sep="\t", dtype=str)
                if len(frame):
                    record.update(frame.iloc[0].to_dict())
            except Exception as exc:
                record["_metadata_read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    if not rows:
        raise RuntimeError(f"No four-digit method folders found under {processed_dir}")
    return pd.DataFrame(rows).set_index("dataset_id").sort_index()


def load_metadata(metadata_csv: Path | None, processed_dir: Path) -> tuple[pd.DataFrame, str]:
    if metadata_csv is None or not metadata_csv.is_file():
        return build_metadata_from_folders(processed_dir), "built_from_processed_data_folders"

    frame = pd.read_csv(metadata_csv, index_col=0, low_memory=False)
    if "id" in frame.columns:
        frame["dataset_id"] = frame["id"].map(normalize_method_id)
        frame = frame.set_index("dataset_id", drop=True)
    else:
        frame.index = pd.Index(
            [normalize_method_id(value) for value in frame.index],
            name="dataset_id",
        )
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    frame = frame.loc[[method_id for method_id in frame.index if int(method_id) <= SOURCE_MAX_METHOD_ID]]
    return frame, str(metadata_csv.resolve())


def read_method_type(processed_dir: Path, method_id: str) -> dict[str, object]:
    path = processed_dir / method_id / INFO_TSV_TEMPLATE.format(method_id=method_id)
    result: dict[str, object] = {
        "method_type_raw": "",
        "info_exists": path.is_file(),
        "info_read_ok": False,
        "info_error": "",
        "info_path": str(path),
    }
    if not path.is_file():
        result["info_error"] = "info_file_missing"
        return result
    try:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as exc:
        result["info_error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["info_read_ok"] = True
    if frame.empty:
        result["info_error"] = "info_file_empty"
    elif "method.type" not in frame.columns:
        result["info_error"] = "method.type_column_missing"
    else:
        result["method_type_raw"] = str(frame["method.type"].iloc[0]).strip()
    return result


def count_rt_rows(processed_dir: Path, method_id: str) -> tuple[int, str]:
    path = processed_dir / method_id / RT_TSV_TEMPLATE.format(method_id=method_id)
    if not path.is_file():
        return 0, "rt_file_missing"
    try:
        return int(len(pd.read_csv(path, sep="\t", index_col=0))), "ok"
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def constant_flow_check(processed_dir: Path, method_id: str) -> tuple[bool, str]:
    path = processed_dir / method_id / GRADIENT_TSV_TEMPLATE.format(method_id=method_id)
    if not path.is_file():
        return False, "gradient_file_missing"
    try:
        gradient = pd.read_csv(path, sep="\t", index_col=0)
    except Exception as exc:
        return False, f"gradient_read_error:{type(exc).__name__}"
    if FLOW_COLUMN not in gradient.columns:
        return True, "flow_column_missing_assumed_constant"
    values = pd.to_numeric(gradient[FLOW_COLUMN], errors="coerce").dropna()
    if values.empty:
        return True, "flow_values_missing_allowed_by_original_rule"
    return (True, "constant_flow") if values.nunique() <= 1 else (False, "variable_flow")


def binary_ab_check(
    metadata: pd.DataFrame,
    method_id: str,
    cd_columns: list[str],
) -> tuple[bool, str]:
    if not cd_columns:
        return True, "no_C_D_columns_in_metadata"
    if method_id not in metadata.index:
        return False, "dataset_missing_from_metadata"
    row = metadata.loc[method_id, cd_columns]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    values = pd.to_numeric(row, errors="coerce").fillna(0)
    active = bool((values > 0).any())
    return (not active), ("active_C_or_D" if active else "binary_AB")


def build_pre_overlap_audit(processed_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    cd_columns = [
        str(column)
        for column in metadata.columns
        if str(column).startswith("eluent.C.") or str(column).startswith("eluent.D.")
    ]
    rows: list[dict[str, object]] = []
    for method_id in sorted(map(normalize_method_id, metadata.index)):
        method_info = read_method_type(processed_dir, method_id)
        observed_constant_pass, observed_constant_reason = constant_flow_check(
            processed_dir, method_id
        )
        frozen_flow_override = bool(
            method_id in FROZEN_FLOW_INCLUSION_IDS and not observed_constant_pass
        )
        constant_pass = bool(observed_constant_pass or frozen_flow_override)
        constant_reason = (
            "frozen_protocol_include_0055_flow_unknown"
            if frozen_flow_override
            else observed_constant_reason
        )
        binary_pass, binary_reason = binary_ab_check(metadata, method_id, cd_columns)
        n_rows, rt_status = count_rt_rows(processed_dir, method_id)
        rows.append(
            {
                "dataset_id": method_id,
                "constant_flow_pass": bool(constant_pass),
                "constant_flow_reason": constant_reason,
                "constant_flow_observed_pass": bool(observed_constant_pass),
                "constant_flow_observed_reason": observed_constant_reason,
                "constant_flow_frozen_override": frozen_flow_override,
                "binary_AB_pass": bool(binary_pass),
                "binary_AB_reason": binary_reason,
                **method_info,
                "strict_method_type_RP_pass": method_info["method_type_raw"] == STRICT_METHOD_TYPE,
                "n_rt_rows_raw": int(n_rows),
                "rt_file_status": rt_status,
                "min_rows_pass": bool(n_rows >= MIN_RT_ROWS),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def load_radonpy_keys(radonpy_csv: Path) -> tuple[set[str], pd.DataFrame]:
    frame = pd.read_csv(radonpy_csv, index_col=0, low_memory=False)
    smiles_column = next(
        (
            column
            for column in ("smiles_list_canonical", "smiles", "SMILES", "canonical_smiles")
            if column in frame.columns
        ),
        None,
    )
    if smiles_column is None:
        raise KeyError("Cannot find a supported SMILES column in the RadonPy CSV")
    values = frame[smiles_column].map(clean_smiles).dropna().astype(str)
    keys = values.map(connectivity_key)
    key_set = set(keys.dropna().astype(str))
    audit = pd.DataFrame(
        [
            {
                "radonpy_csv": str(radonpy_csv),
                "raw_rows": int(len(frame)),
                "smiles_column": smiles_column,
                "nonempty_smiles_rows": int(len(values)),
                "rdkit_parseable_rows": int(keys.notna().sum()),
                "unique_connectivity_keys": int(len(key_set)),
            }
        ]
    )
    return key_set, audit


def build_overlap_audit(
    processed_dir: Path,
    eligible_ids: Iterable[str],
    radonpy_keys: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_id in eligible_ids:
        path = processed_dir / method_id / RT_TSV_TEMPLATE.format(method_id=method_id)
        frame = pd.read_csv(path, sep="\t", index_col=0)
        missing = [column for column in (RT_COLUMN, SMILES_COLUMN) if column not in frame.columns]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")
        rt = pd.to_numeric(frame[RT_COLUMN], errors="coerce")
        smiles = frame[SMILES_COLUMN].map(clean_smiles)
        valid = rt.notna() & (rt > 0) & smiles.notna()
        selected_smiles = smiles.loc[valid].astype(str)
        keys = selected_smiles.map(connectivity_key)
        unique_keys = set(keys.dropna().astype(str))
        overlap_keys = unique_keys & radonpy_keys
        fraction = len(overlap_keys) / len(unique_keys) if unique_keys else np.nan
        rows.append(
            {
                "dataset_id": method_id,
                "n_rt_rows_raw": int(len(frame)),
                "n_valid_rt_and_smiles_rows": int(valid.sum()),
                "n_rows_with_valid_mol_key": int(keys.notna().sum()),
                "n_unique_mol_keys": int(len(unique_keys)),
                "n_overlap_unique_mol_keys": int(len(overlap_keys)),
                "frac_overlap_unique_mol_keys": float(fraction),
                "overlap_gt_0p50": bool(np.isfinite(fraction) and fraction > OVERLAP_THRESHOLD),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)


def normalize_column_name(column: object) -> str:
    return re.sub(r"\s+", " ", str(column).strip().lower())


def find_column(columns: Iterable[object], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_column_name(column): str(column) for column in columns}
    for candidate in candidates:
        if normalize_column_name(candidate) in normalized:
            return normalized[normalize_column_name(candidate)]
    return None


def audit_gradient_information(processed_dir: Path, method_id: str) -> dict[str, object]:
    path = processed_dir / method_id / GRADIENT_TSV_TEMPLATE.format(method_id=method_id)
    result: dict[str, object] = {
        "dataset_id": method_id,
        "gradient_path": str(path),
        "gradient_exists": path.is_file(),
        "gradient_read_ok": False,
        "n_raw_rows": 0,
        "n_nonblank_rows": 0,
        "n_valid_gradient_rows": 0,
        "time_column": "",
        "A_column": "",
        "B_column": "",
        "valid_gradient_information": False,
        "gradient_status": "",
    }
    if not path.is_file():
        result["gradient_status"] = "EXCLUDED_gradient_file_missing"
        return result
    try:
        gradient = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        result["gradient_read_ok"] = True
    except EmptyDataError:
        result["gradient_status"] = "EXCLUDED_gradient_file_empty"
        return result
    except Exception as exc:
        result["gradient_status"] = f"EXCLUDED_gradient_read_error:{type(exc).__name__}"
        return result

    result["n_raw_rows"] = int(len(gradient))
    if gradient.empty:
        result["gradient_status"] = "EXCLUDED_gradient_header_only"
        return result
    gradient.columns = [str(column).strip() for column in gradient.columns]
    cleaned = gradient.apply(lambda column: column.astype("string").str.strip())
    nonblank = np.zeros(len(cleaned), dtype=bool)
    for column in cleaned.columns:
        nonblank |= ~cleaned[column].str.lower().isin(BLANK_TOKENS).to_numpy()
    cleaned = cleaned.loc[nonblank].copy()
    result["n_nonblank_rows"] = int(len(cleaned))
    if cleaned.empty:
        result["gradient_status"] = "EXCLUDED_gradient_all_rows_blank"
        return result

    time_column = find_column(cleaned.columns, TIME_COLUMN_CANDIDATES)
    a_column = find_column(cleaned.columns, A_COLUMN_CANDIDATES)
    b_column = find_column(cleaned.columns, B_COLUMN_CANDIDATES)
    result.update(
        {
            "time_column": time_column or "",
            "A_column": a_column or "",
            "B_column": b_column or "",
        }
    )
    if time_column is None:
        result["gradient_status"] = "EXCLUDED_gradient_time_column_missing"
        return result
    if a_column is None and b_column is None:
        result["gradient_status"] = "EXCLUDED_gradient_A_B_columns_missing"
        return result

    time_values = pd.to_numeric(cleaned[time_column], errors="coerce")
    a_values = (
        pd.to_numeric(cleaned[a_column], errors="coerce")
        if a_column is not None
        else pd.Series(np.nan, index=cleaned.index, dtype=float)
    )
    b_values = (
        pd.to_numeric(cleaned[b_column], errors="coerce")
        if b_column is not None
        else pd.Series(np.nan, index=cleaned.index, dtype=float)
    )
    valid_time = time_values.notna() & np.isfinite(time_values) & (time_values >= 0)
    valid_a_or_b = (
        (a_values.notna() & np.isfinite(a_values))
        | (b_values.notna() & np.isfinite(b_values))
    )
    ab_total = a_values.fillna(0.0) + b_values.fillna(0.0)
    valid_rows = valid_time & valid_a_or_b & np.isfinite(ab_total) & (ab_total > 0)
    result["n_valid_gradient_rows"] = int(valid_rows.sum())
    if valid_rows.any():
        result["valid_gradient_information"] = True
        result["gradient_status"] = "SELECTED_valid_gradient"
    elif not valid_time.any():
        result["gradient_status"] = "EXCLUDED_gradient_no_valid_time_values"
    elif not valid_a_or_b.any():
        result["gradient_status"] = "EXCLUDED_gradient_no_valid_A_B_values"
    else:
        result["gradient_status"] = "EXCLUDED_gradient_no_valid_time_composition_row"
    return result


def validation_record(
    label: str,
    actual: object,
    expected: object,
    passed: bool,
) -> dict[str, object]:
    """Return a non-fatal reproducibility check and warn when it differs."""

    def format_value(value: object) -> str:
        if isinstance(value, set):
            value = sorted(value)
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    actual_text = format_value(actual)
    expected_text = format_value(expected)
    if not passed:
        warnings.warn(
            f"Validation differs for {label}: expected={expected_text}, actual={actual_text}",
            stacklevel=2,
        )
    return {
        "check": label,
        "passed": bool(passed),
        "expected": expected_text,
        "actual": actual_text,
    }


def exact_set_validation(
    label: str,
    actual: Iterable[str],
    expected: Iterable[str],
) -> dict[str, object]:
    actual_set = set(actual)
    expected_set = set(expected)
    return validation_record(
        label,
        sorted(actual_set),
        sorted(expected_set),
        actual_set == expected_set,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen 179-method registry from RepoRT_latest and RadonPy."
    )
    parser.add_argument("--report-root", type=Path, help="Path to RepoRT_latest.")
    parser.add_argument("--radonpy-csv", type=Path, help="Path to RadonPySM_checkeq_masked.csv.")
    parser.add_argument("--metadata-csv", type=Path, help="Optional consolidated metadata CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_path_config()
    report_root = resolve_report_root(
        (args.report_root or project_path(paths["report_root"])).expanduser().resolve()
    )
    processed_dir = report_root / "processed_data"
    data_dir = project_path(paths["data_dir"])
    radonpy_csv = resolve_radonpy_csv(data_dir, args.radonpy_csv)
    metadata_csv = args.metadata_csv
    if metadata_csv is None and paths.get("metadata_csv"):
        candidate = project_path(paths["metadata_csv"])
        metadata_csv = candidate if candidate.is_file() else None
    elif metadata_csv is not None:
        metadata_csv = metadata_csv.expanduser().resolve()
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, metadata_source = load_metadata(metadata_csv, processed_dir)
    base_audit = build_pre_overlap_audit(processed_dir, metadata)
    eligible_mask = (
        base_audit["constant_flow_pass"]
        & base_audit["binary_AB_pass"]
        & base_audit["strict_method_type_RP_pass"]
        & base_audit["min_rows_pass"]
    )
    eligible_ids = base_audit.loc[eligible_mask, "dataset_id"].tolist()

    radonpy_keys, radonpy_audit = load_radonpy_keys(radonpy_csv)
    overlap_audit = build_overlap_audit(processed_dir, eligible_ids, radonpy_keys)
    strict_190_ids = overlap_audit.loc[
        overlap_audit["overlap_gt_0p50"], "dataset_id"
    ].astype(str).tolist()

    gradient_audit = pd.DataFrame(
        [audit_gradient_information(processed_dir, method_id) for method_id in strict_190_ids]
    ).sort_values("dataset_id").reset_index(drop=True)
    strict_178_ids = gradient_audit.loc[
        gradient_audit["valid_gradient_information"], "dataset_id"
    ].astype(str).tolist()
    invalid_gradient_ids = gradient_audit.loc[
        ~gradient_audit["valid_gradient_information"], "dataset_id"
    ].astype(str).tolist()

    strict_178_set = set(strict_178_ids)
    missing_manual_inputs = [
        method_id
        for method_id in MANUALLY_ADDED_IDS
        if not (processed_dir / method_id).is_dir()
    ]
    if missing_manual_inputs:
        raise FileNotFoundError(f"Manual-addition RepoRT folders are missing: {missing_manual_inputs}")

    training_176_ids = sorted(strict_178_set - set(REMOVED_FROM_STRICT_178))
    final_179_ids = sorted(set(training_176_ids) | set(MANUALLY_ADDED_IDS))

    final_registry = pd.DataFrame({"method_id": final_179_ids})
    final_registry["selection_source"] = np.select(
        [
            final_registry["method_id"].isin(MANUALLY_ADDED_IDS),
            final_registry["method_id"].isin(FROZEN_FLOW_INCLUSION_IDS),
        ],
        [
            "documented_manual_addition",
            "frozen_protocol_flow_unknown_inclusion",
        ],
        default="strict_RP_overlap_gradient_valid",
    )
    internal_ood_registry = pd.DataFrame({"method_id": INTERNAL_OOD_IDS})
    internal_ood_registry["selection_source"] = [
        "removed_from_strict_178",
        "removed_from_strict_178",
    ]

    full_audit = base_audit.merge(overlap_audit, on=["dataset_id", "n_rt_rows_raw"], how="left")
    full_audit = full_audit.merge(
        gradient_audit[["dataset_id", "valid_gradient_information", "gradient_status"]],
        on="dataset_id",
        how="left",
    )
    full_audit["in_final_179"] = full_audit["dataset_id"].isin(final_179_ids)
    full_audit["in_internal_ood"] = full_audit["dataset_id"].isin(INTERNAL_OOD_IDS)

    stages = pd.DataFrame(
        [
            ("all methods in metadata", len(base_audit)),
            ("constant-flow", int(base_audit["constant_flow_pass"].sum())),
            (
                "constant-flow + binary A/B",
                int((base_audit["constant_flow_pass"] & base_audit["binary_AB_pass"]).sum()),
            ),
            (
                "method.type exactly RP",
                int(
                    (
                        base_audit["constant_flow_pass"]
                        & base_audit["binary_AB_pass"]
                        & base_audit["strict_method_type_RP_pass"]
                    ).sum()
                ),
            ),
            (f"RT rows >= {MIN_RT_ROWS}", len(eligible_ids)),
            ("RadonPy unique-molecule overlap > 0.50", len(strict_190_ids)),
            ("valid gradient information", len(strict_178_ids)),
            ("remove 0097 and 0238 for internal OOD", len(training_176_ids)),
            ("add 0053, 0069, and 0126", len(final_179_ids)),
        ],
        columns=["stage", "n_methods"],
    )

    expected_stage_counts = {
        "all methods in metadata": EXPECTED_SOURCE_METHOD_COUNT,
        "constant-flow": 323,
        "constant-flow + binary A/B": 321,
        "method.type exactly RP": 258,
        f"RT rows >= {MIN_RT_ROWS}": 211,
        "RadonPy unique-molecule overlap > 0.50": 190,
        "valid gradient information": 178,
        "remove 0097 and 0238 for internal OOD": 176,
        "add 0053, 0069, and 0126": 179,
    }
    validation_rows = [
        validation_record(
            row.stage,
            int(row.n_methods),
            expected_stage_counts[row.stage],
            int(row.n_methods) == expected_stage_counts[row.stage],
        )
        for row in stages.itertuples(index=False)
    ]
    validation_rows.extend(
        [
            exact_set_validation(
                "invalid-gradient method IDs",
                invalid_gradient_ids,
                EXPECTED_INVALID_GRADIENT_IDS,
            ),
            exact_set_validation(
                "internal OOD methods present in strict 178",
                strict_178_set & set(INTERNAL_OOD_IDS),
                REMOVED_FROM_STRICT_178,
            ),
            exact_set_validation(
                "manual additions absent from strict 178",
                set(MANUALLY_ADDED_IDS) & strict_178_set,
                (),
            ),
            exact_set_validation(
                "final registry has no internal OOD methods",
                set(final_179_ids) & set(INTERNAL_OOD_IDS),
                (),
            ),
            validation_record(
                "0055 retained by frozen flow rule",
                "0055" in set(eligible_ids),
                True,
                "0055" in set(eligible_ids),
            ),
        ]
    )
    validation_frame = pd.DataFrame(validation_rows)

    save_table(stages, output_dir / "method_selection_stages.csv")
    save_table(validation_frame, output_dir / "method_selection_validation.csv")
    save_table(radonpy_audit, output_dir / "radonpy_key_audit.csv")
    save_table(base_audit, output_dir / "strict_RP_pre_overlap_full_audit.csv")
    save_table(overlap_audit, output_dir / "strict_RP_per_dataset_RadonPy_overlap.csv")
    save_table(gradient_audit, output_dir / "strict_RP_gradient_information_audit.csv")
    save_table(full_audit, output_dir / "full_method_selection_audit.csv")
    save_table(pd.DataFrame({"method_id": strict_178_ids}), output_dir / "strict_178_method_ids.csv")
    save_table(final_registry, output_dir / "our_179_method_ids.csv")
    save_table(internal_ood_registry, output_dir / "internal_ood_method_ids.csv")
    save_table(
        pd.DataFrame({"method_id": invalid_gradient_ids}),
        output_dir / "removed_invalid_gradient_method_ids.csv",
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_notebook": "Strict_RP_RadonPy_Overlap_gt50_From_Full_RepoRT.ipynb",
        "report_root": str(report_root),
        "metadata_source": metadata_source,
        "source_method_rule": f"RepoRT method ID <= {SOURCE_MAX_METHOD_ID:04d}",
        "source_method_count": len(metadata),
        "radonpy_csv": str(radonpy_csv),
        "rdkit_version": rdBase.rdkitVersion,
        "molecular_overlap_key": "largest SMILES fragment; first InChIKey block",
        "strict_method_rule": "method.type.strip() == 'RP'",
        "flow_selection_rule": "exclude observed variable flow except frozen protocol inclusion 0055",
        "frozen_flow_inclusion_ids": list(FROZEN_FLOW_INCLUSION_IDS),
        "flow_overrides_applied": base_audit.loc[
            base_audit["constant_flow_frozen_override"], "dataset_id"
        ].astype(str).tolist(),
        "minimum_rt_rows": MIN_RT_ROWS,
        "overlap_rule": "unique RepoRT molecule overlap with RadonPy > 0.50",
        "strict_before_gradient": len(strict_190_ids),
        "strict_after_gradient": len(strict_178_ids),
        "internal_ood_ids": list(INTERNAL_OOD_IDS),
        "removed_from_strict_178": list(REMOVED_FROM_STRICT_178),
        "manually_added_ids": list(MANUALLY_ADDED_IDS),
        "final_training_method_count": len(final_179_ids),
        "final_method_ids_csv": str(output_dir / "our_179_method_ids.csv"),
        "validation_passed": bool(validation_frame["passed"].all()),
        "validation_csv": str(output_dir / "method_selection_validation.csv"),
    }
    write_json(manifest, output_dir / "method_selection_manifest.json")

    print(stages.to_string(index=False))
    failed_validation = validation_frame.loc[~validation_frame["passed"]]
    if len(failed_validation):
        print("\n[WARNING] Dataset validation differences were recorded:")
        print(failed_validation.to_string(index=False))
    else:
        print("\nAll frozen dataset validation checks passed.")
    print(f"\nFinal 179-method registry: {output_dir / 'our_179_method_ids.csv'}")
    print(f"Internal OOD registry: {output_dir / 'internal_ood_method_ids.csv'}")


if __name__ == "__main__":
    main()
