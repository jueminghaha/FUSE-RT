#!/usr/bin/env python3
"""Build the frozen ten 179-method molecule-clean 60/20/20 splits.

The script reads the relative method registry produced by
``data/select_179_methods.py`` and raw tables from ``RepoRT_latest``.  It does
not require an older split, a ``global_mol_key`` directory, or premerged
descriptor columns.  The training engine derives environment features from
the local RepoRT metadata at load time.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS_CONFIG = PROJECT_ROOT / "config" / "paths.json"
DEFAULT_METHOD_IDS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "method_selection_179" / "our_179_method_ids.csv"
)

INTERNAL_OOD_IDS = ("0097", "0238")
EXPECTED_METHOD_COUNT = 179
EXPECTED_SPLIT_SEEDS = (2004, 2006, 2011, 2012, 2016, 2020, 2022, 2027, 2032, 2034)
N_ACCEPTED_SPLITS = 10
SEED_START = 2000
MAX_TRIAL_SEEDS = 20000

TARGET_RATIOS = {"train": 0.60, "valid": 0.20, "internal_test": 0.20}
TASK_RATIO_BOUNDS = {
    "train": (0.50, 0.70),
    "valid": (0.10, 0.30),
    "internal_test": (0.10, 0.30),
}
MAX_BAD_TASKS = 2

FULL_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
SMILES_COLUMNS = ("pubchem.smiles.isomeric", "pubchem.smiles.canonical")
SPLIT_NAMES = ("train", "valid", "internal_test")

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
    for candidate in (path, path / "RepoRT_latest"):
        if (candidate / "raw_data").is_dir() and (candidate / "processed_data").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "RepoRT_latest must contain raw_data/ and processed_data/. Checked:\n"
        + "\n".join(str(candidate) for candidate in (path, path / "RepoRT_latest"))
    )


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def valid_full_inchikey(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(FULL_INCHIKEY_RE.fullmatch(str(value).strip().upper()))


def clean_inchikey(value: object) -> str | None:
    return str(value).strip().upper() if valid_full_inchikey(value) else None


def clean_smiles(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"} or text == "-":
        return None
    return text


def choose_smiles(frame: pd.DataFrame) -> pd.Series:
    selected = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    for column in SMILES_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].map(clean_smiles)
        take = selected.isna() & values.notna()
        selected.loc[take] = values.loc[take]
    return selected


@lru_cache(maxsize=None)
def exact_key_from_smiles(smiles: str) -> tuple[str | None, str | None, str]:
    smiles = clean_smiles(smiles)
    if smiles is None:
        return None, None, "missing_smiles"
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None, "invalid_smiles"
    try:
        key = clean_inchikey(Chem.MolToInchiKey(molecule))
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        return None, None, "inchi_failed"
    return (key, canonical, "rdkit_from_smiles") if key else (None, canonical, "invalid_inchikey")


def add_exact_molecular_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["smiles_selected"] = choose_smiles(out)
    raw_keys = (
        out["pubchem.inchikey"].map(clean_inchikey)
        if "pubchem.inchikey" in out.columns
        else pd.Series([None] * len(out), index=out.index, dtype="object")
    )
    out["mol_key_exact"] = raw_keys
    out["canonical_smiles_exact"] = None
    out["mol_key_exact_source"] = np.where(raw_keys.notna(), "pubchem.inchikey", None)

    for index in out.index[out["mol_key_exact"].isna()]:
        key, canonical, source = exact_key_from_smiles(out.at[index, "smiles_selected"])
        out.at[index, "mol_key_exact"] = key
        out.at[index, "canonical_smiles_exact"] = canonical
        out.at[index, "mol_key_exact_source"] = source

    out["mol_key_connectivity"] = (
        out["mol_key_exact"].astype("string").str.split("-", regex=False).str[0]
    )
    out["mol_key"] = out["mol_key_exact"]
    out["smiles"] = out["smiles_selected"]
    return out


def read_raw_rtdata(raw_dir: Path, method_id: str) -> pd.DataFrame:
    path = raw_dir / method_id / f"{method_id}_rtdata.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    if "rt" not in frame.columns:
        raise KeyError(f"{path} has no rt column")
    frame["dataset_id"] = method_id
    frame["dir"] = method_id
    frame["row_id"] = (
        frame["id"].astype(str)
        if "id" in frame.columns
        else [f"{method_id}_{index + 1:05d}" for index in range(len(frame))]
    )
    frame["rt"] = pd.to_numeric(frame["rt"], errors="coerce")
    return frame


def load_method_registry(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"179-method registry not found: {path}\n"
            "Run `python data/select_179_methods.py` first."
        )
    frame = pd.read_csv(path, dtype={"method_id": str})
    if "method_id" not in frame.columns:
        raise KeyError(f"{path} has no method_id column")
    method_ids = sorted({normalize_method_id(value) for value in frame["method_id"]})
    if len(method_ids) != EXPECTED_METHOD_COUNT:
        raise RuntimeError(f"Expected 179 unique training methods, found {len(method_ids)}")
    overlap = set(method_ids) & set(INTERNAL_OOD_IDS)
    if overlap:
        raise RuntimeError(f"Training registry overlaps internal OOD methods: {sorted(overlap)}")
    return method_ids


def prepare_all_rows(
    report_root: Path,
    method_ids: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = report_root / "raw_data"
    parts = [read_raw_rtdata(raw_dir, method_id) for method_id in method_ids + list(INTERNAL_OOD_IDS)]
    all_rows = pd.concat(parts, ignore_index=True)
    all_rows = all_rows[np.isfinite(all_rows["rt"]) & (all_rows["rt"] > 0)].copy()
    all_rows = add_exact_molecular_key(all_rows)
    dropped = all_rows[all_rows["mol_key_exact"].isna()].copy()
    all_rows = all_rows[all_rows["mol_key_exact"].notna()].copy().reset_index(drop=True)

    internal_pool = all_rows[all_rows["dataset_id"].isin(method_ids)].copy()
    internal_ood = all_rows[all_rows["dataset_id"].isin(INTERNAL_OOD_IDS)].copy()
    if internal_pool["dataset_id"].nunique() != EXPECTED_METHOD_COUNT:
        missing = sorted(set(method_ids) - set(internal_pool["dataset_id"]))
        raise RuntimeError(f"Internal pool is missing methods: {missing}")
    if set(internal_ood["dataset_id"].unique()) != set(INTERNAL_OOD_IDS):
        raise RuntimeError("The two internal OOD methods were not both loaded.")

    all_rows.to_csv(output_dir / "all_rt_rows_raw_exact_key.csv", index=False)
    dropped.to_csv(output_dir / "dropped_rows_without_exact_key.csv", index=False)
    (
        internal_pool.groupby("dataset_id")
        .agg(n_rows=("row_id", "size"), n_exact_mol_keys=("mol_key_exact", "nunique"))
        .reset_index()
        .sort_values("dataset_id")
        .to_csv(output_dir / "internal_179_method_size_exact_key.csv", index=False)
    )
    return all_rows, internal_pool, internal_ood


def make_exact_key_split(pool: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    molecule_groups = (
        pool.groupby("mol_key_exact", sort=False).size().rename("n_rows").reset_index()
    )
    molecule_groups = molecule_groups.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    total_rows = int(molecule_groups["n_rows"].sum())

    test_keys: list[str] = []
    accumulated = 0
    for row in molecule_groups.itertuples(index=False):
        if accumulated >= TARGET_RATIOS["internal_test"] * total_rows:
            break
        test_keys.append(row.mol_key_exact)
        accumulated += int(row.n_rows)

    remaining = molecule_groups[~molecule_groups["mol_key_exact"].isin(test_keys)]
    valid_keys: list[str] = []
    accumulated = 0
    for row in remaining.itertuples(index=False):
        if accumulated >= TARGET_RATIOS["valid"] * total_rows:
            break
        valid_keys.append(row.mol_key_exact)
        accumulated += int(row.n_rows)

    test_set = set(test_keys)
    valid_set = set(valid_keys)
    parts = {
        "train": pool[~pool["mol_key_exact"].isin(test_set | valid_set)].copy(),
        "valid": pool[pool["mol_key_exact"].isin(valid_set)].copy(),
        "internal_test": pool[pool["mol_key_exact"].isin(test_set)].copy(),
    }
    for split_name, frame in parts.items():
        frame["split"] = split_name
    return parts


def key_overlap_counts(parts: dict[str, pd.DataFrame], internal_ood: pd.DataFrame) -> dict[str, int]:
    exact = {
        name: set(frame["mol_key_exact"].astype(str))
        for name, frame in parts.items()
    }
    connectivity = {
        name: set(frame["mol_key_connectivity"].astype(str))
        for name, frame in parts.items()
    }
    ood_exact = set(internal_ood["mol_key_exact"].astype(str))
    ood_connectivity = set(internal_ood["mol_key_connectivity"].astype(str))
    return {
        "train_valid_exact_overlap": len(exact["train"] & exact["valid"]),
        "train_test_exact_overlap": len(exact["train"] & exact["internal_test"]),
        "valid_test_exact_overlap": len(exact["valid"] & exact["internal_test"]),
        "train_external_exact_overlap": len(exact["train"] & ood_exact),
        "valid_external_exact_overlap": len(exact["valid"] & ood_exact),
        "test_external_exact_overlap": len(exact["internal_test"] & ood_exact),
        "train_valid_connectivity_overlap": len(connectivity["train"] & connectivity["valid"]),
        "train_test_connectivity_overlap": len(connectivity["train"] & connectivity["internal_test"]),
        "valid_test_connectivity_overlap": len(connectivity["valid"] & connectivity["internal_test"]),
        "train_external_connectivity_overlap": len(connectivity["train"] & ood_connectivity),
        "valid_external_connectivity_overlap": len(connectivity["valid"] & ood_connectivity),
        "test_external_connectivity_overlap": len(connectivity["internal_test"] & ood_connectivity),
    }


def audit_candidate(
    parts: dict[str, pd.DataFrame],
    internal_pool: pd.DataFrame,
    internal_ood: pd.DataFrame,
    method_ids: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total = internal_pool.groupby("dataset_id").size().rename("total")
    counts = [
        parts[name].groupby("dataset_id").size().rename(name)
        for name in SPLIT_NAMES
    ]
    task_audit = pd.concat([total] + counts, axis=1).fillna(0).astype(int).reset_index()
    for name in SPLIT_NAMES:
        task_audit[f"{name}_frac"] = task_audit[name] / task_audit["total"]
    task_audit["method_present_all_splits"] = np.logical_and.reduce(
        [task_audit[name] > 0 for name in SPLIT_NAMES]
    )
    bad = ~task_audit["method_present_all_splits"]
    for name, (low, high) in TASK_RATIO_BOUNDS.items():
        bad |= ~task_audit[f"{name}_frac"].between(low, high, inclusive="both")
    task_audit["is_bad_task"] = bad
    task_audit["seed"] = int(seed)

    n_rows = sum(len(parts[name]) for name in SPLIT_NAMES)
    summary: dict[str, object] = {
        "seed": int(seed),
        "source": "exact_full_inchikey_6_2_2",
        "n_internal_rows": int(n_rows),
        "train_rows": int(len(parts["train"])),
        "valid_rows": int(len(parts["valid"])),
        "internal_test_rows": int(len(parts["internal_test"])),
        "train_frac": len(parts["train"]) / n_rows,
        "valid_frac": len(parts["valid"]) / n_rows,
        "internal_test_frac": len(parts["internal_test"]) / n_rows,
        "n_methods_train": int(parts["train"]["dataset_id"].nunique()),
        "n_methods_valid": int(parts["valid"]["dataset_id"].nunique()),
        "n_methods_internal_test": int(parts["internal_test"]["dataset_id"].nunique()),
        "n_bad_tasks": int(task_audit["is_bad_task"].sum()),
        **key_overlap_counts(parts, internal_ood),
    }
    summary["accepted"] = bool(
        summary["train_valid_exact_overlap"] == 0
        and summary["train_test_exact_overlap"] == 0
        and summary["valid_test_exact_overlap"] == 0
        and summary["n_bad_tasks"] <= MAX_BAD_TASKS
        and summary["n_methods_train"] == len(method_ids)
        and summary["n_methods_valid"] == len(method_ids)
        and summary["n_methods_internal_test"] == len(method_ids)
    )
    return task_audit, summary


def save_split_bundle(
    output_dir: Path,
    parts: dict[str, pd.DataFrame],
    internal_ood: pd.DataFrame,
    task_audit: pd.DataFrame,
    summary: dict[str, object],
) -> Path:
    seed_dir = output_dir / f"seed_{summary['seed']}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    raw_names = {
        "train": "train_raw_exact.csv",
        "valid": "valid_raw_exact.csv",
        "internal_test": "internal_test_raw_exact.csv",
    }
    for name in SPLIT_NAMES:
        parts[name].to_csv(seed_dir / raw_names[name], index=False)
        parts[name].to_csv(seed_dir / f"{name}.csv", index=False)
    internal_ood.to_csv(seed_dir / "external_ood_raw_exact.csv", index=False)
    internal_ood.to_csv(seed_dir / "external_ood.csv", index=False)
    task_audit.to_csv(seed_dir / "per_task_split_distribution.csv", index=False)
    write_json(summary, seed_dir / "split_audit.json")
    return seed_dir


def search_splits(
    output_dir: Path,
    internal_pool: pd.DataFrame,
    internal_ood: pd.DataFrame,
    method_ids: list[str],
) -> pd.DataFrame:
    accepted: list[dict[str, object]] = []
    candidate_log: list[dict[str, object]] = []
    task_audits: list[pd.DataFrame] = []
    for seed in range(SEED_START, SEED_START + MAX_TRIAL_SEEDS):
        parts = make_exact_key_split(internal_pool, seed)
        task_audit, summary = audit_candidate(parts, internal_pool, internal_ood, method_ids, seed)
        candidate_log.append(summary)
        if summary["accepted"]:
            split_dir = save_split_bundle(
                output_dir, parts, internal_ood, task_audit, summary
            )
            summary["split_dir"] = str(split_dir)
            accepted.append(summary)
            task_audits.append(task_audit)
            print(
                f"[ACCEPT {len(accepted):02d}/{N_ACCEPTED_SPLITS}] seed={seed} "
                f"rows=({summary['train_rows']}, {summary['valid_rows']}, "
                f"{summary['internal_test_rows']}) bad_tasks={summary['n_bad_tasks']}"
            )
        if len(accepted) == N_ACCEPTED_SPLITS:
            break

    if len(accepted) != N_ACCEPTED_SPLITS:
        raise RuntimeError(f"Found only {len(accepted)} accepted splits")
    accepted_frame = pd.DataFrame(accepted)
    actual_seeds = tuple(accepted_frame["seed"].astype(int))
    if actual_seeds != EXPECTED_SPLIT_SEEDS:
        raise RuntimeError(
            f"Accepted seeds changed from the frozen protocol: {actual_seeds}"
        )
    accepted_frame.to_csv(output_dir / "accepted_splits_summary.csv", index=False)
    pd.DataFrame(candidate_log).to_csv(output_dir / "candidate_seed_search_log.csv", index=False)
    pd.concat(task_audits, ignore_index=True).to_csv(
        output_dir / "accepted_per_task_split_distribution_all.csv", index=False
    )
    return accepted_frame


def write_final_audits(
    output_dir: Path,
    all_rows: pd.DataFrame,
    accepted: pd.DataFrame,
) -> None:
    collision = (
        all_rows.groupby("mol_key_connectivity")
        .agg(
            n_rows=("row_id", "size"),
            n_full_inchikey=("mol_key_exact", "nunique"),
            n_methods=("dataset_id", "nunique"),
        )
        .reset_index()
        .sort_values(["n_full_inchikey", "n_rows"], ascending=False)
    )
    collision.to_csv(output_dir / "full_inchikey_per_connectivity_key_audit.csv", index=False)

    first_seed_dir = Path(accepted.iloc[0]["split_dir"])
    for filename in (
        "train.csv",
        "valid.csv",
        "internal_test.csv",
        "external_ood.csv",
        "train_raw_exact.csv",
        "valid_raw_exact.csv",
        "internal_test_raw_exact.csv",
        "external_ood_raw_exact.csv",
        "per_task_split_distribution.csv",
        "split_audit.json",
    ):
        shutil.copy2(first_seed_dir / filename, output_dir / filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ten frozen 179-method exact-full-InChIKey splits."
    )
    parser.add_argument("--report-root", type=Path, help="Path to RepoRT_latest.")
    parser.add_argument("--method-ids-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, help="Override config.paths split_root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_path_config()
    report_root = resolve_report_root(
        (args.report_root or project_path(paths["report_root"])).expanduser().resolve()
    )
    method_ids_csv = (
        args.method_ids_csv.expanduser().resolve()
        if args.method_ids_csv is not None
        else project_path(paths.get("method_ids_csv", DEFAULT_METHOD_IDS_CSV))
    )
    output_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else project_path(paths["split_root"])
    )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_ids = load_method_registry(method_ids_csv)
    shutil.copy2(method_ids_csv, output_dir / "method_registry_used.csv")
    all_rows, internal_pool, internal_ood = prepare_all_rows(
        report_root, method_ids, output_dir
    )
    accepted = search_splits(output_dir, internal_pool, internal_ood, method_ids)
    write_final_audits(output_dir, all_rows, accepted)

    manifest = {
        "report_root": str(report_root),
        "method_ids_csv": str(method_ids_csv),
        "method_count": len(method_ids),
        "internal_ood_ids": list(INTERNAL_OOD_IDS),
        "split_unit": "exact full InChIKey",
        "target_ratios": TARGET_RATIOS,
        "accepted_seeds": accepted["seed"].astype(int).tolist(),
        "uses_existing_split": False,
        "uses_global_mol_key_directory": False,
        "training_csv_policy": "raw required columns; environment features rebuilt from RepoRT_latest",
    }
    write_json(manifest, output_dir / "split_generation_manifest.json")
    print(f"\nCompleted split root: {output_dir}")
    print("Accepted seeds:", accepted["seed"].astype(int).tolist())


if __name__ == "__main__":
    main()
