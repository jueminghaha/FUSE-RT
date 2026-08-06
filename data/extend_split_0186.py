def display(value):
    print(value)

# =========================
# Cell 1. Configuration
# =========================
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent
PATHS = json.loads((PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8"))


def configured_path(key):
    path = Path(PATHS[key]).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

# No modification is normally required. If automatic discovery fails, set only the three MANUAL_* variables below.
MANUAL_BASE_SPLIT_ROOT = None
MANUAL_REPORT_AUDIT_ROOT = None
MANUAL_METHOD_0186_RAW_FILE = None

BASE_SPLIT_CANDIDATES = [
    configured_path("split_root"),
    SOURCE_ROOT / 'RepoRT_PolyOmic/outputs/exact_full_inchikey_split_6_2_2',
    Path("outputs/exact_full_inchikey_split_6_2_2"),
    Path("exact_full_inchikey_split_6_2_2"),
]

# Continue to read method 0186 exclusively from the RepoRT_latest/raw_data snapshot frozen in the overlap audit;
# do not read from RepoRT/raw_data at the project root.
REPORT_AUDIT_CANDIDATES = [
    configured_path("report_root"),
    SOURCE_ROOT / 'outputs/report_laest_overlap_audit',
    Path("outputs/report_latest_overlap_audit"),
    Path("report_latest_overlap_audit"),
    Path("report_latest_overlap_audit.zip"),
    Path("/mnt/data/report_latest_overlap_audit.zip"),
]

OUT_DIR = configured_path("split_root_180")
OVERWRITE_OUT_DIR = True

ADDED_METHOD_ID = 186
ADDED_METHOD_NAME = "0186"
EXPECTED_BASE_METHODS = 179
EXPECTED_FINAL_METHODS = 180

FROZEN_SPLIT_SEEDS = [
    2004, 2006, 2011, 2012, 2016,
    2020, 2022, 2027, 2032, 2034,
]

TARGET_RATIOS = {
    "train": 0.60,
    "valid": 0.20,
    "internal_test": 0.20,
}

# Per-method acceptance bounds retained from the original split notebook.
TASK_RATIO_BOUNDS = {
    "train": (0.50, 0.70),
    "valid": (0.10, 0.30),
    "internal_test": (0.10, 0.30),
}
MAX_BAD_TASKS = 2
REQUIRE_ALL_METHODS_IN_EACH_SPLIT = True

SMILES_PRIORITY = [
    "pubchem.smiles.isomeric",
    "pubchem.smiles.canonical",
]
DROP_ROWS_WITHOUT_EXACT_KEY = True

# Store one molecular-key assignment audit for method 0186 per seed.
WRITE_KEY_ASSIGNMENT_AUDIT = True


# =========================
# Cell 2. Imports and utility functions
# =========================
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception as exc:
    HAS_RDKIT = False
    print("[WARN] RDKit import failed:", repr(exc))

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 160)

FULL_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
SPLIT_NAMES = ["train", "valid", "internal_test"]


def find_existing(candidates: Sequence[Path], name: str) -> Path:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            print(f"{name}: {path}")
            return path
    raise FileNotFoundError(
        f"Cannot find {name}. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def materialize_input(path: Path, cache_root: Path, label: str) -> Path:
    """Return a directory. Extract a zip into cache_root/label when needed."""
    path = Path(path)
    if path.is_dir():
        return path.resolve()

    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"{label} must be a directory or .zip: {path}")

    out = cache_root / label
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {path} -> {out}")
    with zipfile.ZipFile(path, "r") as archive:
        members = [m for m in archive.namelist() if not m.startswith("__MACOSX")]
        archive.extractall(out, members=members)

    top_dirs = [p for p in out.iterdir() if p.is_dir() and not p.name.startswith("__MACOSX")]
    top_files = [p for p in out.iterdir() if p.is_file()]
    if len(top_dirs) == 1 and not top_files:
        return top_dirs[0].resolve()
    return out.resolve()


def normalize_method_int(value) -> int:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return int(text)


def method_dir_name(method_id: int) -> str:
    return f"{int(method_id):04d}"


def valid_full_inchikey(value) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().upper()
    return bool(FULL_INCHIKEY_RE.fullmatch(text))


def clean_inchikey(value) -> Optional[str]:
    if not valid_full_inchikey(value):
        return None
    return str(value).strip().upper()


def clean_smiles_value(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"} or text == "-":
        return None
    return text


def choose_smiles_vectorized(df: pd.DataFrame) -> pd.Series:
    selected = pd.Series([None] * len(df), index=df.index, dtype="object")
    for column in SMILES_PRIORITY:
        if column not in df.columns:
            continue
        values = df[column].map(clean_smiles_value)
        take = selected.isna() & values.notna()
        selected.loc[take] = values.loc[take]
    return selected


def rdkit_exact_key_from_smiles(smiles: Optional[str]) -> Tuple[Optional[str], Optional[str], str]:
    if not HAS_RDKIT:
        return None, None, "rdkit_unavailable"
    smiles = clean_smiles_value(smiles)
    if smiles is None:
        return None, None, "missing_smiles"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "invalid_smiles"

    try:
        key = Chem.MolToInchiKey(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None, None, "inchi_failed"

    key = clean_inchikey(key)
    if key is None:
        return None, canonical, "invalid_inchikey"
    return key, canonical, "rdkit_from_smiles"


def add_exact_molecular_key_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Use raw full PubChem InChIKey first; run RDKit only for missing unique SMILES."""
    out = df.copy()
    out["smiles_selected"] = choose_smiles_vectorized(out)

    if "pubchem.inchikey" in out.columns:
        raw_keys = out["pubchem.inchikey"].map(clean_inchikey)
    else:
        raw_keys = pd.Series([None] * len(out), index=out.index, dtype="object")

    out["mol_key_exact"] = raw_keys
    out["canonical_smiles_exact"] = None
    out["mol_key_exact_source"] = np.where(
        out["mol_key_exact"].notna(),
        "pubchem.inchikey",
        None,
    )

    missing_mask = out["mol_key_exact"].isna()
    unique_missing_smiles = (
        out.loc[missing_mask, "smiles_selected"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    rdkit_cache: Dict[str, Tuple[Optional[str], Optional[str], str]] = {}
    for smiles in unique_missing_smiles:
        rdkit_cache[smiles] = rdkit_exact_key_from_smiles(smiles)

    if missing_mask.any():
        for idx in out.index[missing_mask]:
            smiles = out.at[idx, "smiles_selected"]
            key, canonical, source = rdkit_cache.get(
                smiles,
                (None, None, "missing_smiles"),
            )
            out.at[idx, "mol_key_exact"] = key
            out.at[idx, "canonical_smiles_exact"] = canonical
            out.at[idx, "mol_key_exact_source"] = source

    out["mol_key_connectivity"] = (
        out["mol_key_exact"]
        .astype("string")
        .str.split("-", regex=False)
        .str[0]
    )
    return out


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def canonical_identity_set(df: pd.DataFrame) -> set:
    """Identity used to prove the old 179 rows were not changed."""
    needed = ["dataset_id", "row_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing identity columns: {missing}")
    return set(zip(df["dataset_id"].map(normalize_method_int), df["row_id"].astype(str)))


# =========================
# Cell 3. Resolve inputs and create the new 180-dataset output root
# =========================
if OUT_DIR.exists() and OVERWRITE_OUT_DIR:
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUT_DIR / "_input_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if MANUAL_BASE_SPLIT_ROOT is not None:
    BASE_SPLIT_INPUT = Path(MANUAL_BASE_SPLIT_ROOT).expanduser()
    if not BASE_SPLIT_INPUT.exists():
        raise FileNotFoundError(BASE_SPLIT_INPUT)
else:
    BASE_SPLIT_INPUT = find_existing(BASE_SPLIT_CANDIDATES, "BASE_SPLIT_INPUT")

if MANUAL_REPORT_AUDIT_ROOT is not None:
    REPORT_AUDIT_INPUT = Path(MANUAL_REPORT_AUDIT_ROOT).expanduser()
    if not REPORT_AUDIT_INPUT.exists():
        raise FileNotFoundError(REPORT_AUDIT_INPUT)
else:
    REPORT_AUDIT_INPUT = find_existing(REPORT_AUDIT_CANDIDATES, "REPORT_AUDIT_INPUT")

BASE_SPLIT_ROOT = materialize_input(BASE_SPLIT_INPUT, CACHE_DIR, "base_179_split")
REPORT_ROOT = materialize_input(REPORT_AUDIT_INPUT, CACHE_DIR, "report_latest_overlap_audit")

if MANUAL_METHOD_0186_RAW_FILE is not None:
    METHOD_0186_RAW_FILE = Path(MANUAL_METHOD_0186_RAW_FILE).expanduser().resolve()
    if not METHOD_0186_RAW_FILE.exists():
        raise FileNotFoundError(METHOD_0186_RAW_FILE)
else:
    report_latest_root = (
        REPORT_ROOT
        if (REPORT_ROOT / "raw_data").is_dir()
        else REPORT_ROOT / "RepoRT_latest"
    )
    direct = report_latest_root / "raw_data" / ADDED_METHOD_NAME / f"{ADDED_METHOD_NAME}_rtdata.tsv"
    if direct.exists():
        METHOD_0186_RAW_FILE = direct.resolve()
    else:
        matches = sorted(
            REPORT_ROOT.glob(f"**/raw_data/{ADDED_METHOD_NAME}/{ADDED_METHOD_NAME}_rtdata.tsv")
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                "Cannot uniquely locate the frozen audit raw file for 0186.\n"
                f"Expected: {direct}\n"
                "Set MANUAL_METHOD_0186_RAW_FILE explicitly."
            )
        METHOD_0186_RAW_FILE = matches[0].resolve()

required_root_files = ["accepted_splits_summary.csv"]
for filename in required_root_files:
    if not (BASE_SPLIT_ROOT / filename).exists():
        raise FileNotFoundError(BASE_SPLIT_ROOT / filename)

print("\nResolved paths:")
print("BASE_SPLIT_ROOT     =", BASE_SPLIT_ROOT)
print("REPORT_ROOT         =", REPORT_ROOT)
print("METHOD_0186_RAW_FILE=", METHOD_0186_RAW_FILE)
print("OUT_DIR             =", OUT_DIR.resolve())


# =========================
# Cell 4. Load the fixed 179-method list and method 0186 raw rows
# =========================
method_ids_path = configured_path("method_ids_csv")
if not method_ids_path.exists():
    matches = sorted(PROJECT_ROOT.glob("**/our_179_method_ids.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Cannot uniquely locate our_179_method_ids.csv under REPORT_ROOT."
        )
    method_ids_path = matches[0]

base_method_ids = (
    pd.read_csv(method_ids_path)["method_id"]
    .map(normalize_method_int)
    .tolist()
)

if len(base_method_ids) != EXPECTED_BASE_METHODS or len(set(base_method_ids)) != EXPECTED_BASE_METHODS:
    raise RuntimeError(
        f"Expected {EXPECTED_BASE_METHODS} unique base methods, got {len(set(base_method_ids))}."
    )
if ADDED_METHOD_ID in base_method_ids:
    raise RuntimeError("0186 is already present in the frozen 179-method list.")

method_ids_180 = base_method_ids + [ADDED_METHOD_ID]
if len(method_ids_180) != EXPECTED_FINAL_METHODS or len(set(method_ids_180)) != EXPECTED_FINAL_METHODS:
    raise RuntimeError("179 + 0186 did not produce exactly 180 unique methods.")

raw_0186 = pd.read_csv(METHOD_0186_RAW_FILE, sep="\t", low_memory=False)
raw_0186["source_row_index"] = np.arange(len(raw_0186), dtype=np.int64)
raw_0186["dataset_id"] = ADDED_METHOD_ID
raw_0186["dir"] = ADDED_METHOD_ID

if "rt" not in raw_0186.columns:
    raise KeyError(f"{METHOD_0186_RAW_FILE} has no rt column.")
raw_0186["rt"] = pd.to_numeric(raw_0186["rt"], errors="coerce")

# Preserve a usable raw ID when it is unique; otherwise use a stable 0186_<source index> ID.
if "id" in raw_0186.columns:
    candidate_row_id = raw_0186["id"].astype("string").str.strip()
else:
    candidate_row_id = pd.Series([pd.NA] * len(raw_0186), index=raw_0186.index, dtype="string")

fallback_row_id = raw_0186["source_row_index"].map(lambda i: f"0186_{int(i):07d}")
invalid_id = candidate_row_id.isna() | candidate_row_id.eq("") | candidate_row_id.str.lower().eq("nan")
duplicate_id = candidate_row_id.duplicated(keep=False) & ~invalid_id
candidate_row_id = candidate_row_id.where(~(invalid_id | duplicate_id), fallback_row_id)
raw_0186["row_id"] = candidate_row_id.astype(str)

if raw_0186["row_id"].duplicated().any():
    raise RuntimeError("0186 row_id is still non-unique after stable fallback generation.")

before_filter_rows = len(raw_0186)
raw_0186 = raw_0186[np.isfinite(raw_0186["rt"]) & (raw_0186["rt"] > 0)].copy()
raw_0186 = add_exact_molecular_key_fast(raw_0186)

dropped_0186_no_key = raw_0186[raw_0186["mol_key_exact"].isna()].copy()
if DROP_ROWS_WITHOUT_EXACT_KEY:
    raw_0186 = raw_0186[raw_0186["mol_key_exact"].notna()].copy()

raw_0186["mol_key"] = raw_0186["mol_key_exact"]
raw_0186 = raw_0186.reset_index(drop=True)

raw_0186.to_csv(OUT_DIR / "method_0186_raw_exact_key.csv", index=False)
dropped_0186_no_key.to_csv(OUT_DIR / "method_0186_dropped_rows_without_exact_key.csv", index=False)

pd.DataFrame({"method_id": [f"{m:04d}" for m in method_ids_180]}).to_csv(
    OUT_DIR / "our_180_method_ids.csv",
    index=False,
)
with open(OUT_DIR / "dataset_180_method_ids.txt", "w", encoding="utf-8") as handle:
    handle.write("\n".join(f"{m:04d}" for m in method_ids_180) + "\n")

print("Base method count:", len(base_method_ids))
print("Final method count:", len(method_ids_180))
print("0186 raw rows before RT/key filtering:", before_filter_rows)
print("0186 rows after RT/key filtering:", len(raw_0186))
print("0186 unique exact full InChIKeys:", raw_0186["mol_key_exact"].nunique())
print("0186 dropped rows without exact key:", len(dropped_0186_no_key))
print("\n0186 exact-key source counts:")
display(raw_0186["mol_key_exact_source"].value_counts(dropna=False).rename_axis("source").reset_index(name="n_rows"))


# =========================
# Cell 5. Validate the frozen 179-method seed inputs
# =========================
BASE_SUMMARY = pd.read_csv(BASE_SPLIT_ROOT / "accepted_splits_summary.csv")
if "seed" not in BASE_SUMMARY.columns:
    raise KeyError("Base accepted_splits_summary.csv has no seed column.")

available_seeds = set(BASE_SUMMARY["seed"].dropna().astype(int))
missing_seeds = [seed for seed in FROZEN_SPLIT_SEEDS if seed not in available_seeds]
if missing_seeds:
    raise RuntimeError(
        f"Frozen seeds are missing from the 179-method split summary: {missing_seeds}"
    )

RAW_FILENAMES = {
    "train": "train_raw_exact.csv",
    "valid": "valid_raw_exact.csv",
    "internal_test": "internal_test_raw_exact.csv",
}
FEATURE_FILENAMES = {
    "train": "train.csv",
    "valid": "valid.csv",
    "internal_test": "internal_test.csv",
}


def seed_dir(root: Path, seed: int) -> Path:
    path = Path(root) / f"seed_{int(seed)}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_base_raw_split(seed: int, split_name: str) -> pd.DataFrame:
    sdir = seed_dir(BASE_SPLIT_ROOT, seed)
    raw_path = sdir / RAW_FILENAMES[split_name]
    fallback_path = sdir / FEATURE_FILENAMES[split_name]

    if raw_path.exists():
        df = pd.read_csv(raw_path, low_memory=False)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path, low_memory=False)
        if "mol_key_exact" not in df.columns:
            if "mol_key" not in df.columns:
                raise KeyError(f"{fallback_path} has neither mol_key_exact nor mol_key.")
            df["mol_key_exact"] = df["mol_key"].astype(str)
        if "mol_key_connectivity" not in df.columns:
            df["mol_key_connectivity"] = (
                df["mol_key_exact"].astype("string").str.split("-", regex=False).str[0]
            )
    else:
        raise FileNotFoundError(f"Missing both {raw_path} and {fallback_path}")

    df = df.copy()
    df["dataset_id"] = df["dataset_id"].map(normalize_method_int)
    df["row_id"] = df["row_id"].astype(str)
    df["mol_key_exact"] = df["mol_key_exact"].astype(str)
    df["mol_key"] = df["mol_key_exact"]
    df["split"] = split_name
    return df


def load_base_feature_split(seed: int, split_name: str) -> pd.DataFrame:
    path = seed_dir(BASE_SPLIT_ROOT, seed) / FEATURE_FILENAMES[split_name]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df["dataset_id"] = df["dataset_id"].map(normalize_method_int)
    df["row_id"] = df["row_id"].astype(str)
    return df


base_input_audits = []
for seed in FROZEN_SPLIT_SEEDS:
    raw_parts = {name: load_base_raw_split(seed, name) for name in SPLIT_NAMES}
    key_sets = {name: set(df["mol_key_exact"].astype(str)) for name, df in raw_parts.items()}

    overlaps = {
        "train_valid": len(key_sets["train"] & key_sets["valid"]),
        "train_test": len(key_sets["train"] & key_sets["internal_test"]),
        "valid_test": len(key_sets["valid"] & key_sets["internal_test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Base 179 split has exact-key leakage for seed={seed}: {overlaps}")

    method_counts = {name: df["dataset_id"].nunique() for name, df in raw_parts.items()}
    if any(count != EXPECTED_BASE_METHODS for count in method_counts.values()):
        raise RuntimeError(
            f"Base split seed={seed} does not contain all 179 methods in every split: {method_counts}"
        )
    if any((df["dataset_id"] == ADDED_METHOD_ID).any() for df in raw_parts.values()):
        raise RuntimeError(f"0186 is unexpectedly already present in base seed={seed}.")

    base_input_audits.append({
        "seed": seed,
        "train_rows": len(raw_parts["train"]),
        "valid_rows": len(raw_parts["valid"]),
        "internal_test_rows": len(raw_parts["internal_test"]),
        "n_methods_train": method_counts["train"],
        "n_methods_valid": method_counts["valid"],
        "n_methods_internal_test": method_counts["internal_test"],
        "train_valid_exact_overlap": overlaps["train_valid"],
        "train_test_exact_overlap": overlaps["train_test"],
        "valid_test_exact_overlap": overlaps["valid_test"],
    })

base_input_audit_df = pd.DataFrame(base_input_audits)
base_input_audit_df.to_csv(OUT_DIR / "base_179_seed_input_audit.csv", index=False)
print("Validated frozen 179-method seed inputs:", FROZEN_SPLIT_SEEDS)
display(base_input_audit_df)


# =========================
# Cell 6. 0186 overlap inheritance and seeded 6/2/2 deficit fill
# =========================
def integer_row_targets(total_rows: int) -> Dict[str, int]:
    """Integer 60/20/20 targets whose sum is exactly total_rows."""
    target_valid = int(round(TARGET_RATIOS["valid"] * total_rows))
    target_test = int(round(TARGET_RATIOS["internal_test"] * total_rows))
    target_train = int(total_rows - target_valid - target_test)
    return {
        "train": target_train,
        "valid": target_valid,
        "internal_test": target_test,
    }


def build_base_key_to_split(base_raw_parts: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    key_to_split: Dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for key in base_raw_parts[split_name]["mol_key_exact"].dropna().astype(str).unique():
            previous = key_to_split.get(key)
            if previous is not None and previous != split_name:
                raise RuntimeError(
                    f"Base exact key {key} appears in both {previous} and {split_name}."
                )
            key_to_split[key] = split_name
    return key_to_split


def assign_0186_for_seed(
    method_df: pd.DataFrame,
    base_raw_parts: Dict[str, pd.DataFrame],
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, object]]:
    """
    1) Keys overlapping the frozen 179 split inherit that split.
    2) Remaining keys are shuffled with `seed` and used to fill the row-count
       deficits of the 0186-wide 60/20/20 target. Exact keys remain indivisible.
    """
    key_to_base_split = build_base_key_to_split(base_raw_parts)

    key_table = (
        method_df.groupby("mol_key_exact", sort=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    key_table["inherited_split"] = key_table["mol_key_exact"].map(key_to_base_split)
    key_table["overlaps_base_179"] = key_table["inherited_split"].notna()
    key_table["assigned_split"] = key_table["inherited_split"]
    key_table["assignment_source"] = np.where(
        key_table["overlaps_base_179"],
        "inherited_from_base_179",
        None,
    )

    targets = integer_row_targets(len(method_df))
    fixed_counts = {
        split_name: int(
            key_table.loc[key_table["inherited_split"] == split_name, "n_rows"].sum()
        )
        for split_name in SPLIT_NAMES
    }

    remaining = key_table[key_table["assigned_split"].isna()].copy()
    remaining = remaining.sample(frac=1.0, random_state=int(seed)).reset_index()

    # Preserve the original split notebook's ordering: test, then valid, then train.
    # The fixed inherited rows are subtracted first, so these are true deficits for 0186 overall.
    cursor = 0
    for split_name in ["internal_test", "valid"]:
        need = max(0, targets[split_name] - fixed_counts[split_name])
        accumulated = 0
        while cursor < len(remaining) and accumulated < need:
            row_index = int(remaining.at[cursor, "index"])
            n_rows = int(remaining.at[cursor, "n_rows"])
            key_table.at[row_index, "assigned_split"] = split_name
            key_table.at[row_index, "assignment_source"] = "seeded_deficit_fill"
            accumulated += n_rows
            cursor += 1

    # All remaining non-overlap keys go to train.
    while cursor < len(remaining):
        row_index = int(remaining.at[cursor, "index"])
        key_table.at[row_index, "assigned_split"] = "train"
        key_table.at[row_index, "assignment_source"] = "seeded_deficit_fill"
        cursor += 1

    if key_table["assigned_split"].isna().any():
        raise RuntimeError(f"Unassigned 0186 exact keys remain for seed={seed}.")

    key_to_final_split = dict(
        zip(key_table["mol_key_exact"].astype(str), key_table["assigned_split"].astype(str))
    )
    assigned_rows = method_df.copy()
    assigned_rows["split"] = assigned_rows["mol_key_exact"].astype(str).map(key_to_final_split)

    if assigned_rows["split"].isna().any():
        raise RuntimeError(f"Unassigned 0186 rows remain for seed={seed}.")

    parts = {
        split_name: assigned_rows[assigned_rows["split"] == split_name].copy()
        for split_name in SPLIT_NAMES
    }

    final_counts = {split_name: int(len(parts[split_name])) for split_name in SPLIT_NAMES}
    final_key_counts = {
        split_name: int(parts[split_name]["mol_key_exact"].nunique())
        for split_name in SPLIT_NAMES
    }

    # Exact-key disjointness within 0186.
    final_sets = {
        split_name: set(parts[split_name]["mol_key_exact"].astype(str))
        for split_name in SPLIT_NAMES
    }
    if (
        final_sets["train"] & final_sets["valid"]
        or final_sets["train"] & final_sets["internal_test"]
        or final_sets["valid"] & final_sets["internal_test"]
    ):
        raise RuntimeError(f"0186 exact-key leakage after assignment for seed={seed}.")

    # Every overlapping key must exactly inherit the base 179 split.
    inherited_bad = key_table[
        key_table["overlaps_base_179"]
        & (key_table["assigned_split"] != key_table["inherited_split"])
    ]
    if len(inherited_bad):
        raise RuntimeError(
            f"Inherited 0186 keys changed split for seed={seed}: {len(inherited_bad)} keys."
        )

    key_table["seed"] = int(seed)
    key_table["target_train_rows"] = targets["train"]
    key_table["target_valid_rows"] = targets["valid"]
    key_table["target_internal_test_rows"] = targets["internal_test"]

    summary = {
        "seed": int(seed),
        "method_0186_rows": int(len(method_df)),
        "method_0186_unique_exact_keys": int(key_table["mol_key_exact"].nunique()),
        "method_0186_overlap_rows": int(
            key_table.loc[key_table["overlaps_base_179"], "n_rows"].sum()
        ),
        "method_0186_overlap_exact_keys": int(key_table["overlaps_base_179"].sum()),
        "method_0186_nonoverlap_rows": int(
            key_table.loc[~key_table["overlaps_base_179"], "n_rows"].sum()
        ),
        "method_0186_nonoverlap_exact_keys": int((~key_table["overlaps_base_179"]).sum()),
        **{f"method_0186_target_{name}_rows": int(targets[name]) for name in SPLIT_NAMES},
        **{f"method_0186_fixed_{name}_rows": int(fixed_counts[name]) for name in SPLIT_NAMES},
        **{f"method_0186_{name}_rows": int(final_counts[name]) for name in SPLIT_NAMES},
        **{f"method_0186_{name}_keys": int(final_key_counts[name]) for name in SPLIT_NAMES},
        **{
            f"method_0186_{name}_frac": final_counts[name] / max(len(method_df), 1)
            for name in SPLIT_NAMES
        },
    }
    return parts, key_table, summary


# =========================
# Cell 7. Output-schema alignment and audits
# =========================
def make_0186_feature_rows(
    raw_split: pd.DataFrame,
    base_template: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Create rows with exactly the same columns and order as the existing 179-method
    train.csv/valid.csv/internal_test.csv. Known raw columns are copied; all optional
    legacy descriptor columns absent for 0186 remain NaN. Current E1-E9 notebooks
    rebuild environment features from method 0186 metadata at load time.
    """
    out = pd.DataFrame(index=np.arange(len(raw_split)), columns=base_template.columns)

    # Copy columns with identical names first.
    for column in out.columns.intersection(raw_split.columns):
        out[column] = raw_split[column].to_numpy()

    smiles = raw_split["smiles_selected"].copy()
    if smiles.isna().any() and "pubchem.smiles.canonical" in raw_split.columns:
        smiles = smiles.fillna(raw_split["pubchem.smiles.canonical"].map(clean_smiles_value))
    if smiles.isna().any() and "pubchem.smiles.isomeric" in raw_split.columns:
        smiles = smiles.fillna(raw_split["pubchem.smiles.isomeric"].map(clean_smiles_value))

    forced_values = {
        "dataset_id": np.full(len(raw_split), ADDED_METHOD_ID),
        "dir": np.full(len(raw_split), ADDED_METHOD_ID),
        "row_id": raw_split["row_id"].astype(str).to_numpy(),
        "rt": raw_split["rt"].to_numpy(),
        "smiles": smiles.to_numpy(),
        "mol_key": raw_split["mol_key_exact"].astype(str).to_numpy(),
        "mol_key_exact": raw_split["mol_key_exact"].astype(str).to_numpy(),
        "mol_key_connectivity": raw_split["mol_key_connectivity"].astype(str).to_numpy(),
        "mol_key_old_connectivity": raw_split["mol_key_connectivity"].astype(str).to_numpy(),
        "mol_key_exact_source": raw_split["mol_key_exact_source"].to_numpy(),
        "canonical_smiles_exact": raw_split["canonical_smiles_exact"].to_numpy(),
        "smiles_selected": raw_split["smiles_selected"].to_numpy(),
        "smiles_reference": smiles.to_numpy(),
        "split": np.full(len(raw_split), split_name),
    }

    for column, values in forced_values.items():
        if column in out.columns:
            out[column] = values

    required = ["dataset_id", "dir", "row_id", "rt", "smiles", "mol_key"]
    missing_required = [column for column in required if column not in out.columns]
    if missing_required:
        raise KeyError(
            f"Base feature template is missing required training columns: {missing_required}"
        )

    null_required = {
        column: int(pd.isna(out[column]).sum())
        for column in required
    }
    if any(null_required.values()):
        raise RuntimeError(
            f"0186 feature-compatible rows contain missing required fields: {null_required}"
        )

    return out


def exact_key_leakage_audit(train_df, valid_df, test_df, external_df=None) -> Dict[str, int]:
    sets = {
        "train": set(train_df["mol_key_exact"].astype(str)),
        "valid": set(valid_df["mol_key_exact"].astype(str)),
        "internal_test": set(test_df["mol_key_exact"].astype(str)),
    }
    result = {
        "train_valid_exact_overlap": len(sets["train"] & sets["valid"]),
        "train_test_exact_overlap": len(sets["train"] & sets["internal_test"]),
        "valid_test_exact_overlap": len(sets["valid"] & sets["internal_test"]),
    }
    if external_df is not None and len(external_df):
        external = set(external_df["mol_key_exact"].astype(str))
        result.update({
            "train_external_exact_overlap": len(sets["train"] & external),
            "valid_external_exact_overlap": len(sets["valid"] & external),
            "test_external_exact_overlap": len(sets["internal_test"] & external),
        })
    return result


def connectivity_overlap_audit(train_df, valid_df, test_df, external_df=None) -> Dict[str, int]:
    column = "mol_key_connectivity"
    sets = {
        "train": set(train_df[column].astype(str)),
        "valid": set(valid_df[column].astype(str)),
        "internal_test": set(test_df[column].astype(str)),
    }
    result = {
        "train_valid_connectivity_overlap": len(sets["train"] & sets["valid"]),
        "train_test_connectivity_overlap": len(sets["train"] & sets["internal_test"]),
        "valid_test_connectivity_overlap": len(sets["valid"] & sets["internal_test"]),
    }
    if external_df is not None and len(external_df):
        external = set(external_df[column].astype(str))
        result.update({
            "train_external_connectivity_overlap": len(sets["train"] & external),
            "valid_external_connectivity_overlap": len(sets["valid"] & external),
            "test_external_connectivity_overlap": len(sets["internal_test"] & external),
        })
    return result


def task_ratio_audit_180(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    all_internal = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    total = all_internal.groupby("dataset_id").size().rename("total")

    parts = []
    for split_name, frame in [
        ("train", train_df),
        ("valid", valid_df),
        ("internal_test", test_df),
    ]:
        parts.append(frame.groupby("dataset_id").size().rename(split_name))

    audit = pd.concat([total] + parts, axis=1).fillna(0).astype(int).reset_index()
    for split_name in SPLIT_NAMES:
        audit[f"{split_name}_frac"] = audit[split_name] / audit["total"].replace(0, np.nan)

    audit["method_present_all_splits"] = (
        (audit["train"] > 0)
        & (audit["valid"] > 0)
        & (audit["internal_test"] > 0)
    )

    bad = pd.Series(False, index=audit.index)
    for split_name, (lower, upper) in TASK_RATIO_BOUNDS.items():
        bad = bad | ~audit[f"{split_name}_frac"].between(lower, upper, inclusive="both")
    if REQUIRE_ALL_METHODS_IN_EACH_SPLIT:
        bad = bad | ~audit["method_present_all_splits"]

    audit["is_bad_task"] = bad
    audit["seed"] = int(seed)
    audit["source"] = "frozen_179_plus_0186_overlap_inherit_deficit_fill"

    n_total = len(all_internal)
    summary = {
        "seed": int(seed),
        "source": "frozen_179_plus_0186_overlap_inherit_deficit_fill",
        "n_internal_rows": int(n_total),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "internal_test_rows": int(len(test_df)),
        "train_frac": len(train_df) / n_total,
        "valid_frac": len(valid_df) / n_total,
        "internal_test_frac": len(test_df) / n_total,
        "n_methods_train": int(train_df["dataset_id"].nunique()),
        "n_methods_valid": int(valid_df["dataset_id"].nunique()),
        "n_methods_internal_test": int(test_df["dataset_id"].nunique()),
        "n_bad_tasks": int(audit["is_bad_task"].sum()),
    }
    return audit, summary


def save_split_bundle_180(
    seed: int,
    combined_raw: Dict[str, pd.DataFrame],
    combined_feature: Dict[str, pd.DataFrame],
    task_audit: pd.DataFrame,
    summary: Dict[str, object],
    key_assignment: pd.DataFrame,
) -> Path:
    output_seed_dir = OUT_DIR / f"seed_{int(seed)}"
    output_seed_dir.mkdir(parents=True, exist_ok=True)

    combined_raw["train"].to_csv(output_seed_dir / "train_raw_exact.csv", index=False)
    combined_raw["valid"].to_csv(output_seed_dir / "valid_raw_exact.csv", index=False)
    combined_raw["internal_test"].to_csv(output_seed_dir / "internal_test_raw_exact.csv", index=False)

    combined_feature["train"].to_csv(output_seed_dir / "train.csv", index=False)
    combined_feature["valid"].to_csv(output_seed_dir / "valid.csv", index=False)
    combined_feature["internal_test"].to_csv(output_seed_dir / "internal_test.csv", index=False)

    # External OOD is intentionally unchanged.
    base_seed_dir = seed_dir(BASE_SPLIT_ROOT, seed)
    for filename in ["external_ood.csv", "external_ood_raw_exact.csv"]:
        source = base_seed_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_seed_dir / filename)

    task_audit.to_csv(output_seed_dir / "per_task_split_distribution.csv", index=False)
    if WRITE_KEY_ASSIGNMENT_AUDIT:
        key_assignment.to_csv(output_seed_dir / "method_0186_key_assignment.csv", index=False)
    save_json(summary, output_seed_dir / "split_audit.json")
    return output_seed_dir


# =========================
# Cell 8. Build the 10 frozen 180-method splits
# =========================
accepted_summaries: List[Dict[str, object]] = []
all_task_audits: List[pd.DataFrame] = []
assignment_summaries: List[Dict[str, object]] = []
feature_schema_audits: List[Dict[str, object]] = []

for seed in FROZEN_SPLIT_SEEDS:
    base_raw = {name: load_base_raw_split(seed, name) for name in SPLIT_NAMES}
    base_feature = {name: load_base_feature_split(seed, name) for name in SPLIT_NAMES}

    method_0186_parts, key_assignment, assignment_summary = assign_0186_for_seed(
        raw_0186,
        base_raw,
        seed,
    )

    combined_raw: Dict[str, pd.DataFrame] = {}
    combined_feature: Dict[str, pd.DataFrame] = {}

    for split_name in SPLIT_NAMES:
        added_raw = method_0186_parts[split_name].copy()
        added_raw["dataset_id"] = ADDED_METHOD_ID
        added_raw["dir"] = ADDED_METHOD_ID
        added_raw["mol_key"] = added_raw["mol_key_exact"]
        added_raw["split"] = split_name

        combined_raw[split_name] = pd.concat(
            [base_raw[split_name], added_raw],
            ignore_index=True,
            sort=False,
        )

        added_feature = make_0186_feature_rows(
            added_raw,
            base_feature[split_name],
            split_name,
        )
        combined_feature[split_name] = pd.concat(
            [base_feature[split_name], added_feature],
            ignore_index=True,
            sort=False,
        )
        combined_feature[split_name] = combined_feature[split_name][base_feature[split_name].columns]

        # Prove that the original 179 rows were copied without reassignment or deletion.
        base_identity = canonical_identity_set(base_feature[split_name])
        combined_base_identity = canonical_identity_set(
            combined_feature[split_name][combined_feature[split_name]["dataset_id"].map(normalize_method_int) != ADDED_METHOD_ID]
        )
        if base_identity != combined_base_identity:
            raise RuntimeError(
                f"Base 179 row identities changed for seed={seed}, split={split_name}."
            )

        feature_schema_audits.append({
            "seed": int(seed),
            "split": split_name,
            "base_columns": int(len(base_feature[split_name].columns)),
            "output_columns": int(len(combined_feature[split_name].columns)),
            "schema_preserved": list(base_feature[split_name].columns) == list(combined_feature[split_name].columns),
            "base_179_rows": int(len(base_feature[split_name])),
            "added_0186_rows": int(len(added_feature)),
            "output_rows": int(len(combined_feature[split_name])),
            "missing_0186_smiles": int(pd.isna(added_feature["smiles"]).sum()),
            "missing_0186_rt": int(pd.isna(added_feature["rt"]).sum()),
            "missing_0186_mol_key": int(pd.isna(added_feature["mol_key"]).sum()),
        })

    external_raw_path = seed_dir(BASE_SPLIT_ROOT, seed) / "external_ood_raw_exact.csv"
    external_raw = pd.read_csv(external_raw_path, low_memory=False)
    if "mol_key_exact" not in external_raw.columns and "mol_key" in external_raw.columns:
        external_raw["mol_key_exact"] = external_raw["mol_key"].astype(str)
    if "mol_key_connectivity" not in external_raw.columns:
        external_raw["mol_key_connectivity"] = (
            external_raw["mol_key_exact"].astype("string").str.split("-", regex=False).str[0]
        )

    task_audit, summary = task_ratio_audit_180(
        combined_raw["train"],
        combined_raw["valid"],
        combined_raw["internal_test"],
        seed,
    )
    summary.update(exact_key_leakage_audit(
        combined_raw["train"],
        combined_raw["valid"],
        combined_raw["internal_test"],
        external_raw,
    ))
    summary.update(connectivity_overlap_audit(
        combined_raw["train"],
        combined_raw["valid"],
        combined_raw["internal_test"],
        external_raw,
    ))
    summary.update(assignment_summary)

    summary["base_179_train_rows"] = int(len(base_raw["train"]))
    summary["base_179_valid_rows"] = int(len(base_raw["valid"]))
    summary["base_179_internal_test_rows"] = int(len(base_raw["internal_test"]))
    summary["base_179_rows_unchanged"] = True

    summary["accepted"] = bool(
        summary["train_valid_exact_overlap"] == 0
        and summary["train_test_exact_overlap"] == 0
        and summary["valid_test_exact_overlap"] == 0
        and summary["n_bad_tasks"] <= MAX_BAD_TASKS
        and (
            not REQUIRE_ALL_METHODS_IN_EACH_SPLIT
            or (
                summary["n_methods_train"] == EXPECTED_FINAL_METHODS
                and summary["n_methods_valid"] == EXPECTED_FINAL_METHODS
                and summary["n_methods_internal_test"] == EXPECTED_FINAL_METHODS
            )
        )
    )

    if not summary["accepted"]:
        display(task_audit[task_audit["is_bad_task"]])
        raise RuntimeError(
            f"Generated 180-method split failed acceptance checks for seed={seed}: {summary}"
        )

    output_seed_dir = save_split_bundle_180(
        seed,
        combined_raw,
        combined_feature,
        task_audit,
        summary,
        key_assignment,
    )
    summary["split_dir"] = str(output_seed_dir)

    accepted_summaries.append(summary)
    all_task_audits.append(task_audit)
    assignment_summaries.append(assignment_summary)

    print(
        f"[DONE {len(accepted_summaries):02d}/{len(FROZEN_SPLIT_SEEDS)}] seed={seed} "
        f"combined rows=({summary['train_rows']}, {summary['valid_rows']}, {summary['internal_test_rows']}); "
        f"0186 rows=({summary['method_0186_train_rows']}, {summary['method_0186_valid_rows']}, "
        f"{summary['method_0186_internal_test_rows']}); "
        f"0186 inherited keys={summary['method_0186_overlap_exact_keys']}"
    )

accepted_df = pd.DataFrame(accepted_summaries)
accepted_task_audit_df = pd.concat(all_task_audits, ignore_index=True)
assignment_summary_df = pd.DataFrame(assignment_summaries)
feature_schema_audit_df = pd.DataFrame(feature_schema_audits)

accepted_df.to_csv(OUT_DIR / "accepted_splits_summary.csv", index=False)
# Compatibility filename: there is no candidate search now; the frozen 10 seeds are the only candidates.
accepted_df.to_csv(OUT_DIR / "candidate_seed_search_log.csv", index=False)
accepted_task_audit_df.to_csv(
    OUT_DIR / "accepted_per_task_split_distribution_all.csv",
    index=False,
)
assignment_summary_df.to_csv(
    OUT_DIR / "method_0186_assignment_summary.csv",
    index=False,
)
feature_schema_audit_df.to_csv(
    OUT_DIR / "feature_schema_audit.csv",
    index=False,
)

display(accepted_df)


# =========================
# Cell 9. Final audits and convenience copy
# =========================
summary = pd.read_csv(OUT_DIR / "accepted_splits_summary.csv")

expected_seed_order = FROZEN_SPLIT_SEEDS
observed_seed_order = summary["seed"].astype(int).tolist()
if observed_seed_order != expected_seed_order:
    raise RuntimeError(
        f"Output seed order changed. Expected {expected_seed_order}, got {observed_seed_order}."
    )

print("=== Final 180-method sanity checklist ===")
print("Output seeds:", observed_seed_order)
print("All accepted:", bool(summary["accepted"].astype(bool).all()))
print("All exact train-valid leaks zero:", bool((summary["train_valid_exact_overlap"] == 0).all()))
print("All exact train-test leaks zero:", bool((summary["train_test_exact_overlap"] == 0).all()))
print("All exact valid-test leaks zero:", bool((summary["valid_test_exact_overlap"] == 0).all()))
print("All methods in train:", bool((summary["n_methods_train"] == EXPECTED_FINAL_METHODS).all()))
print("All methods in valid:", bool((summary["n_methods_valid"] == EXPECTED_FINAL_METHODS).all()))
print("All methods in internal_test:", bool((summary["n_methods_internal_test"] == EXPECTED_FINAL_METHODS).all()))
print("All base-179 identities unchanged:", bool(summary["base_179_rows_unchanged"].astype(bool).all()))

show_columns = [
    "seed",
    "train_rows", "valid_rows", "internal_test_rows",
    "train_frac", "valid_frac", "internal_test_frac",
    "n_methods_train", "n_methods_valid", "n_methods_internal_test",
    "method_0186_overlap_exact_keys", "method_0186_overlap_rows",
    "method_0186_train_rows", "method_0186_valid_rows", "method_0186_internal_test_rows",
    "method_0186_train_frac", "method_0186_valid_frac", "method_0186_internal_test_frac",
    "n_bad_tasks",
    "train_valid_exact_overlap", "train_test_exact_overlap", "valid_test_exact_overlap",
    "split_dir",
]
display(summary[show_columns])

# Copy the first frozen seed to OUT_DIR root for one-seed smoke tests.
first_seed_dir = OUT_DIR / f"seed_{FROZEN_SPLIT_SEEDS[0]}"
root_copy_files = [
    "train.csv", "valid.csv", "internal_test.csv", "external_ood.csv",
    "train_raw_exact.csv", "valid_raw_exact.csv", "internal_test_raw_exact.csv", "external_ood_raw_exact.csv",
    "per_task_split_distribution.csv", "method_0186_key_assignment.csv", "split_audit.json",
]
for filename in root_copy_files:
    source = first_seed_dir / filename
    if source.exists():
        shutil.copy2(source, OUT_DIR / filename)

# Rebuild a convenience feature-table pickle from the first 180-method split.
feature_parts = []
for split_name, filename in [
    ("train", "train.csv"),
    ("valid", "valid.csv"),
    ("internal_test", "internal_test.csv"),
    ("external_ood", "external_ood.csv"),
]:
    frame = pd.read_csv(OUT_DIR / filename, low_memory=False)
    frame["__reference_split"] = split_name
    feature_parts.append(frame)

reference_feature_table_180 = pd.concat(feature_parts, ignore_index=True, sort=False)
reference_feature_table_180 = reference_feature_table_180.drop_duplicates(
    subset=["dataset_id", "row_id"],
    keep="first",
)
reference_feature_table_180.to_pickle(OUT_DIR / "reference_feature_table.pkl")

print("\nNew 180-dataset output directory:", OUT_DIR.resolve())
print("First frozen seed copied to OUT_DIR root:", FROZEN_SPLIT_SEEDS[0])


# =========================
# Cell 10. Root-level 180-dataset audit files
# =========================
first_seed = FROZEN_SPLIT_SEEDS[0]
first_seed_dir = OUT_DIR / f"seed_{first_seed}"

first_internal_parts = [
    pd.read_csv(first_seed_dir / "train_raw_exact.csv", low_memory=False),
    pd.read_csv(first_seed_dir / "valid_raw_exact.csv", low_memory=False),
    pd.read_csv(first_seed_dir / "internal_test_raw_exact.csv", low_memory=False),
]
internal_180_all = pd.concat(first_internal_parts, ignore_index=True, sort=False)
external_ood_first = pd.read_csv(first_seed_dir / "external_ood_raw_exact.csv", low_memory=False)

internal_180_all.to_csv(OUT_DIR / "all_180_internal_rows_raw_exact_key.csv", index=False)
pd.concat([internal_180_all, external_ood_first], ignore_index=True, sort=False).to_csv(
    OUT_DIR / "all_rt_rows_raw_exact_key.csv",
    index=False,
)

# Combine old dropped rows with 0186 dropped rows when the old file exists.
dropped_parts = []
base_dropped = BASE_SPLIT_ROOT / "dropped_rows_without_exact_key.csv"
if base_dropped.exists():
    old_dropped = pd.read_csv(base_dropped, low_memory=False)
    old_dropped["drop_source"] = "base_179_notebook"
    dropped_parts.append(old_dropped)
if len(dropped_0186_no_key):
    new_dropped = dropped_0186_no_key.copy()
    new_dropped["drop_source"] = "method_0186"
    dropped_parts.append(new_dropped)

if dropped_parts:
    pd.concat(dropped_parts, ignore_index=True, sort=False).to_csv(
        OUT_DIR / "dropped_rows_without_exact_key.csv",
        index=False,
    )
else:
    pd.DataFrame().to_csv(OUT_DIR / "dropped_rows_without_exact_key.csv", index=False)

method_size = (
    internal_180_all.groupby("dataset_id")
    .agg(
        n_rows=("row_id", "size"),
        n_exact_mol_keys=("mol_key_exact", "nunique"),
    )
    .reset_index()
    .sort_values("n_rows")
)
method_size.to_csv(OUT_DIR / "internal_180_method_size_exact_key.csv", index=False)

exact_collision = (
    internal_180_all.groupby("mol_key_connectivity")
    .agg(
        n_rows=("row_id", "size"),
        n_full_inchikey=("mol_key_exact", "nunique"),
        n_methods=("dataset_id", "nunique"),
    )
    .reset_index()
    .sort_values(["n_full_inchikey", "n_rows"], ascending=False)
)
exact_collision.to_csv(OUT_DIR / "full_inchikey_per_connectivity_key_audit.csv", index=False)

connectivity_overlap_records = []
for seed in FROZEN_SPLIT_SEEDS:
    sdir = OUT_DIR / f"seed_{seed}"
    split_frames = {
        "train": pd.read_csv(
            sdir / "train_raw_exact.csv",
            usecols=["dataset_id", "row_id", "mol_key_exact", "mol_key_connectivity"],
        ),
        "valid": pd.read_csv(
            sdir / "valid_raw_exact.csv",
            usecols=["dataset_id", "row_id", "mol_key_exact", "mol_key_connectivity"],
        ),
        "internal_test": pd.read_csv(
            sdir / "internal_test_raw_exact.csv",
            usecols=["dataset_id", "row_id", "mol_key_exact", "mol_key_connectivity"],
        ),
    }

    for split_a, split_b in [
        ("train", "valid"),
        ("train", "internal_test"),
        ("valid", "internal_test"),
    ]:
        frame_a = split_frames[split_a]
        frame_b = split_frames[split_b]
        overlap_keys = sorted(
            set(frame_a["mol_key_connectivity"].astype(str))
            & set(frame_b["mol_key_connectivity"].astype(str))
        )
        for key in overlap_keys:
            mask_a = frame_a["mol_key_connectivity"].astype(str).eq(key)
            mask_b = frame_b["mol_key_connectivity"].astype(str).eq(key)
            connectivity_overlap_records.append({
                "seed": int(seed),
                "split_a": split_a,
                "split_b": split_b,
                "mol_key_connectivity": key,
                "n_full_keys_a": int(frame_a.loc[mask_a, "mol_key_exact"].nunique()),
                "n_full_keys_b": int(frame_b.loc[mask_b, "mol_key_exact"].nunique()),
                "n_rows_a": int(mask_a.sum()),
                "n_rows_b": int(mask_b.sum()),
            })

connectivity_overlap_df = pd.DataFrame(connectivity_overlap_records)
connectivity_overlap_df.to_csv(
    OUT_DIR / "connectivity_overlap_keys_between_splits.csv",
    index=False,
)

run_config = {
    "protocol": "frozen 179 exact-full-InChIKey splits + method 0186 overlap inheritance + seeded deficit fill",
    "base_split_root": str(BASE_SPLIT_ROOT),
    "report_root": str(REPORT_ROOT),
    "method_0186_raw_file": str(METHOD_0186_RAW_FILE),
    "output_dir": str(OUT_DIR.resolve()),
    "base_method_count": EXPECTED_BASE_METHODS,
    "added_method": ADDED_METHOD_NAME,
    "final_method_count": EXPECTED_FINAL_METHODS,
    "frozen_split_seeds": FROZEN_SPLIT_SEEDS,
    "target_ratios": TARGET_RATIOS,
    "exact_key_definition": "full pubchem.inchikey; fallback RDKit MolToInchiKey from isomeric/canonical SMILES; stereochemistry preserved",
    "overlap_rule": "0186 exact keys overlapping base 179 inherit the base split",
    "nonoverlap_rule": "remaining 0186 exact keys are seed-shuffled and fill 0186-wide 60/20/20 row deficits",
    "external_ood_rule": "copied unchanged from the base 179 split",
}
save_json(run_config, OUT_DIR / "run_config.json")

print("180-method size rows:", len(method_size))
print("Connectivity-level overlap records:", len(connectivity_overlap_df))
print("\nSmallest methods:")
display(method_size.head(15))
print("\n0186 method size:")
display(method_size[method_size["dataset_id"].map(normalize_method_int) == ADDED_METHOD_ID])


# =========================
# Cell 11. Quick usage notes
# =========================
print(
    "Generation complete.\n\n"
    "Output directory:\n"
    "  outputs/exact_full_inchikey_split_6_2_2_180dataset/\n\n"
    "Each seed_<seed>/ directory retains the original output filenames:\n"
    "  train.csv\n"
    "  valid.csv\n"
    "  internal_test.csv\n"
    "  external_ood.csv\n"
    "  train_raw_exact.csv\n"
    "  valid_raw_exact.csv\n"
    "  internal_test_raw_exact.csv\n"
    "  external_ood_raw_exact.csv\n"
    "  per_task_split_distribution.csv\n"
    "  split_audit.json\n"
    "Additional file:\n"
    "  method_0186_key_assignment.csv\n\n"
    "Protocol invariants:\n"
    "  - Splits for the original 179 methods remain fully frozen; seeds are not re-optimized.\n"
    "  - Exact full InChIKeys shared by method 0186 and the original 179 methods inherit their existing assignments.\n"
    "  - Remaining exact keys from method 0186 are assigned with the same seed to approximate the 60/20/20 target.\n"
    "  - An exact full InChIKey never crosses the train, validation, and internal-test partitions.\n"
    "  - External-OOD files remain unchanged.\n"
)
