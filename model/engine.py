"""Shared RepoRT/PolyOmics model and training engine.

The original E1-E9 and E8 scaling notebooks repeated the same implementation.
This module keeps one final definition of each helper/class and reads experiment
differences from ``config/experiments.json``.
"""

from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import re
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdchem import BondType, HybridizationType
    RDLogger.DisableLog("rdApp.warning")
except Exception as exc:
    raise ImportError("RDKit is required.") from exc

try:
    from torch_geometric.data import Batch as PyGBatch, Data as PyGData
    from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
    from torch_geometric.utils import to_dense_batch
except Exception as exc:
    raise ImportError("torch_geometric is required.") from exc

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 240)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKDIR = PROJECT_ROOT


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_CONFIG = _load_json(PROJECT_ROOT / "config" / "experiments.json")
_PATHS = _load_json(PROJECT_ROOT / "config" / "paths.json")


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


REPORT_ROOT = _project_path(_PATHS["report_root"])
PROCESSED_DIR = REPORT_ROOT / "processed_data"
DATA_DIR = _project_path(_PATHS["data_dir"])
META_PATH = _project_path(_PATHS["metadata_csv"])
SPLIT_ROOT = _project_path(_PATHS["split_root"])

EXP_KEY = "E8"
EXP_ID = "E8"
MOLECULE_MODE = "M2_joint"
RT_ARCHITECTURE = "R1_device_multi"
EXP_NAME = "M2 joint multitask × R1 device encoder + multitask heads"
USE_DEVICE_METADATA = True
RT_HEAD_TYPE = "multi"
JOINT_MULTITASK = True
WEIGHTS_ONLY = True
EVALUATE_INTERNAL_TEST = False
EVALUATE_EXTERNAL_OOD = False
RADONPY_PERCENT = 100.0
RADONPY_FRACTION = 1.0
RADONPY_PRETRAIN_EPOCHS = 0
RADONPY_VALID_FRACTION = 0.10
RT_EPOCHS = 200
SPLIT_SEEDS = list(_CONFIG["common"]["seeds"])
RESUME_COMPLETED = True
FAIL_FAST = False
BEST_CKPT_NAME = "best_joint.pt"

CUDA_DEVICE_INDEX = int(os.environ.get("CUDA_DEVICE_INDEX", "0"))
DEVICE = torch.device(f"cuda:{CUDA_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu")

ROOT_OUT = PROJECT_ROOT / "checkpoints" / "experiments" / EXP_ID
FRACTION_OUT = ROOT_OUT
OUT_DIR = FRACTION_OUT / "_shared"
DEDUPLICATE_SMILES_WITHIN_TASK = False
df_meta = pd.DataFrame()

_model_cfg = _CONFIG["common"]["model"]
_optim_cfg = _CONFIG["common"]["optimization"]
CFG = {
    **_model_cfg,
    **_optim_cfg,
    "rt_loss_huber_weight": 0.7,
    "rt_loss_mse_weight": 0.3,
    "rt_huber_beta": 0.5,
    "composite_valid_weights": {"all": 0.4, "clean_overlap": 0.3, "nonoverlap": 0.3},
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first_existing(paths):
    for path in map(Path, paths):
        if path.exists():
            return path
    return Path(paths[0])


# Cell 3 — RepoRT readers, Graphormer-style cleaning rules, descriptors, and selected env features
# This cell intentionally builds metadata from RepoRT files directly instead of relying on GraphormerRT_*_metadata.pickle.

MOBILE_SOLVENT_KEYS = ["h2o", "meoh", "acn", "iproh", "acetone", "ace"]


def to_dataset_id(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(4) if s.isdigit() else s


def _norm_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _as_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        s = str(x).strip().replace("spp", "")
        if s == "" or s.lower() in {"nan", "none", "na", "n/a", "__na__"}:
            return default
        v = float(s)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def scale_or_missing(value, denom: float, default=-1.0):
    v = _as_float(value, default=np.nan)
    if not np.isfinite(v) or denom == 0:
        return float(default)
    return float(v / denom)


def pct_or_missing(value, default=-1.0):
    return scale_or_missing(value, 100.0, default=default)


def ph_or_missing(value, default=-1.0):
    return scale_or_missing(value, 14.0, default=default)


def cat_or_na(value) -> str:
    s = _norm_text(value)
    if s == "" or s.lower() in {"nan", "none", "na", "n/a", "__na__"}:
        return "NA"
    return s


def normalize_solvent_name(x) -> str:
    s = cat_or_na(x).lower()
    if s == "na":
        return "NA"
    if s in {"water", "h2o", "aqueous"} or "water" in s:
        return "h2o"
    if "methanol" in s or "meoh" in s:
        return "meoh"
    if "acetonitrile" in s or "acn" in s:
        return "acn"
    if "isoprop" in s or "iproh" in s or "ipa" in s:
        return "iproh"
    if "acetone" in s or s == "ace":
        return "acetone"
    return "Other"


def normalize_phase_type(*values) -> str:
    s = " ".join([cat_or_na(v) for v in values]).lower()
    if s.strip() == "" or s.strip() == "na":
        return "NA"
    # Keep chemistry-like information, not brand/company names.
    if "hilic" in s:
        return "HILIC"
    if "amide" in s:
        return "amide"
    if "biphenyl" in s:
        return "biphenyl"
    if "phenyl" in s:
        return "phenyl"
    if "c18" in s or re.search(r"\bl1\b", s):
        return "C18/L1"
    if "c8" in s or re.search(r"\bl7\b", s):
        return "C8/L7"
    m = re.search(r"\bl\d{1,3}\b", s)
    if m:
        return m.group(0).upper()
    return "Other"


def list_report_method_ids(processed_dir=PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"RepoRT processed_data folder not found: {processed_dir.resolve()}")
    return sorted([p.name for p in processed_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)])


def read_report_global_metadata(meta_path=META_PATH):
    meta_path = Path(meta_path)
    if not meta_path.exists():
        print(f"Global metadata file not found: {meta_path}. Falling back to per-method TSV files.")
        return pd.DataFrame()
    df = pd.read_csv(meta_path, index_col=0)
    df.index = df.index.map(to_dataset_id)
    return df




def _case_insensitive_get(row, names, default=np.nan):
    if row is None or len(row) == 0:
        return default
    if not isinstance(names, (list, tuple)):
        names = [names]
    key_map = {str(k).lower(): k for k in row.index}
    for name in names:
        key = key_map.get(str(name).lower())
        if key is not None:
            return row[key]
    return default


def get_meta_row(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    if len(df_meta) and dataset_id in df_meta.index:
        return df_meta.loc[dataset_id]
    return pd.Series(dtype=object)


def read_per_method_metadata_matrix(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    path = PROCESSED_DIR / dataset_id / f"{dataset_id}_metadata.tsv"
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore").replace("/%", "percent")
    rows = []
    for line in txt.splitlines():
        line = line.rstrip("\r\n")
        if line.strip() == "":
            continue
        rows.append(line.split("\t"))
    if not rows:
        return None
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    return np.asarray(rows, dtype=object)


def get_matrix_value(dataset_id, col_idx, default=np.nan):
    mat = read_per_method_metadata_matrix(dataset_id)
    if mat is not None and mat.shape[0] > 1 and mat.shape[1] > col_idx:
        return mat[1, col_idx]
    return default


def get_meta_or_matrix(dataset_id, names, matrix_col=None, default=np.nan):
    meta_row = get_meta_row(dataset_id)
    val = _case_insensitive_get(meta_row, names, default=default)
    if _norm_text(val) != "" and not (isinstance(val, float) and np.isnan(val)):
        return val
    if matrix_col is not None:
        return get_matrix_value(dataset_id, matrix_col, default=default)
    return default


def get_column_name(dataset_id, meta_row=None):
    dataset_id = to_dataset_id(dataset_id)
    if meta_row is None:
        meta_row = get_meta_row(dataset_id)
    val = _case_insensitive_get(meta_row, ["column.name", "column", "column_name", "name"], default="")
    if str(val).strip():
        return str(val).strip()
    return str(get_matrix_value(dataset_id, 1, default="")).strip()


def read_info_hplc_type(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    path = PROCESSED_DIR / dataset_id / f"{dataset_id}_info.tsv"
    if not path.exists():
        return ""
    try:
        arr = pd.read_csv(path, sep="\t", header=None, dtype=str).fillna("").values
        if arr.shape[0] > 1 and arr.shape[1] > 2:
            return str(arr[1, 2]).strip()
    except Exception:
        pass
    return ""


def infer_family_from_metadata(dataset_id, meta_row=None):
    dataset_id = to_dataset_id(dataset_id)
    if meta_row is None:
        meta_row = get_meta_row(dataset_id)
    column_name = get_column_name(dataset_id, meta_row)
    hplc_type = read_info_hplc_type(dataset_id)
    text = f"{column_name} {hplc_type}".lower()
    if "hilic" in text or "amide" in text:
        return "HILIC", column_name, hplc_type
    return "RP", column_name, hplc_type


def read_gradient_table(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    path = PROCESSED_DIR / dataset_id / f"{dataset_id}_gradient.tsv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep="\t")
    df = df.dropna(axis=1, how="all")
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed"):
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() == len(df):
                df = df.drop(columns=[c])

    num = df.apply(pd.to_numeric, errors="coerce")
    usable = [c for c in num.columns if num[c].notna().sum() >= 2]
    if len(usable) < 3:
        arr = np.loadtxt(path, delimiter="\t", dtype=str)[1:, :].astype(float)
        out = pd.DataFrame({"time": arr[:, 0], "A": arr[:, 1], "B": arr[:, 2], "flow": arr[:, -1]})
        if arr.shape[1] > 5:
            out["C"] = arr[:, -3]
            out["D"] = arr[:, -2]
        else:
            out["C"] = 0.0
            out["D"] = 0.0
        return out

    def find_by_name(patterns, candidates):
        for pat in patterns:
            rgx = re.compile(pat, flags=re.I)
            hits = [c for c in candidates if rgx.search(str(c))]
            if hits:
                return hits[0]
        return None

    time_col = find_by_name([r"time", r"t\s*\["], usable) or usable[0]
    flow_col = find_by_name([r"flow"], usable)
    candidates = [c for c in usable if c != time_col and c != flow_col]

    A_col = find_by_name([r"(^|[^A-Za-z])A([^A-Za-z]|$)", r"%\s*A", r"A\s*\[%"], candidates)
    B_col = find_by_name([r"(^|[^A-Za-z])B([^A-Za-z]|$)", r"%\s*B", r"B\s*\[%"], candidates)
    C_col = find_by_name([r"(^|[^A-Za-z])C([^A-Za-z]|$)", r"%\s*C", r"C\s*\[%"], candidates)
    D_col = find_by_name([r"(^|[^A-Za-z])D([^A-Za-z]|$)", r"%\s*D", r"D\s*\[%"], candidates)

    if A_col is None or B_col is None:
        if len(candidates) < 2:
            raise ValueError(f"Cannot identify A/B gradient columns for {dataset_id}")
        A_col, B_col = candidates[0], candidates[1]
        if len(candidates) >= 4:
            C_col, D_col = candidates[2], candidates[3]

    flow = num[flow_col] if flow_col is not None else pd.Series(np.nan, index=df.index)
    out = pd.DataFrame({
        "time": pd.to_numeric(num[time_col], errors="coerce"),
        "A": pd.to_numeric(num[A_col], errors="coerce"),
        "B": pd.to_numeric(num[B_col], errors="coerce"),
        "flow": pd.to_numeric(flow, errors="coerce"),
    }).dropna(subset=["time", "A", "B"]).reset_index(drop=True)
    out["C"] = pd.to_numeric(num[C_col], errors="coerce") if C_col is not None else 0.0
    out["D"] = pd.to_numeric(num[D_col], errors="coerce") if D_col is not None else 0.0
    out["C"] = out["C"].fillna(0.0)
    out["D"] = out["D"].fillna(0.0)
    return out


def canonicalize_gradient(grad_df, family):
    grad = grad_df.copy()
    threshold = 30.0 if family == "HILIC" else 50.0
    switched = bool(len(grad) > 0 and _as_float(grad["B"].iloc[0]) > threshold)
    if switched:
        grad[["A", "B"]] = grad[["B", "A"]]
    return grad, switched


def graphormer_inflections(pB, times, close_delta=0.3):
    pB = np.asarray(pB, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(pB) < 2 or np.all(~np.isfinite(pB)):
        return []
    max_ind = int(np.nanargmax(pB))
    points = []
    for i in range(1, len(pB)):
        if not np.isfinite(pB[i]) or not np.isfinite(pB[i - 1]):
            continue
        if pB[i] != pB[i - 1] and i <= max_ind:
            points.append((float(times[i - 1]), float(pB[i - 1])))
            points.append((float(times[i]), float(pB[i])))
    points = sorted(set(points), key=lambda x: x[0])
    if len(times) and len(pB):
        start = (float(times[0]), float(pB[0]))
        points = [pt for pt in points if not (abs(pt[0] - start[0]) < 1e-9 and abs(pt[1] - start[1]) < 1e-9)]
    collapsed = []
    for pt in points:
        if not collapsed:
            collapsed.append(pt)
        elif abs(pt[0] - collapsed[-1][0]) >= close_delta:
            collapsed.append(pt)
    return collapsed


def get_t0(dataset_id, meta_row=None):
    if meta_row is None:
        meta_row = get_meta_row(dataset_id)
    v = _case_insensitive_get(meta_row, ["column.t0", "column.dead.time", "col_dead", "dead time", "t0"], default=np.nan)
    val = _as_float(v)
    if np.isfinite(val):
        return val
    return _as_float(get_matrix_value(dataset_id, 8, default=np.nan))


def dominant_solvent_from_matrix(dataset_id, channel):
    mat = read_per_method_metadata_matrix(dataset_id)
    if mat is None or mat.shape[0] < 2:
        return ""
    if channel == "A":
        start, end = 9, 18
    else:
        start, end = 49, 58
    if mat.shape[1] <= start:
        return ""
    labels = mat[0, start:min(end, mat.shape[1])]
    vals = [_as_float(x, default=0.0) for x in mat[1, start:min(end, mat.shape[1])]]
    if not vals or max(vals) == 0:
        return ""
    return str(labels[int(np.argmax(vals))]).split(".")[-1].strip()


def dominant_solvent(dataset_id, channel, meta_row=None):
    if meta_row is None:
        meta_row = get_meta_row(dataset_id)
    for key in MOBILE_SOLVENT_KEYS:
        v = _case_insensitive_get(meta_row, [f"eluent.{channel}.{key}", f"{channel}_{key}"], default=np.nan)
        if np.isfinite(_as_float(v)) and abs(_as_float(v)) > 0:
            return key
    return dominant_solvent_from_matrix(dataset_id, channel)


def solvent_is_excluded(solvent):
    s = _norm_text(solvent).lower()
    return any(bad in s for bad in ["iproh", "isoprop", "ipa", "acetone", "ace"])


def compute_method_properties(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    rec = {"dataset_id": dataset_id}
    try:
        meta_row = get_meta_row(dataset_id)
        family, column_name, hplc_type = infer_family_from_metadata(dataset_id, meta_row)
        rec.update({"family": family, "column_name": column_name, "hplc_type": hplc_type})
        rec["t0_min"] = get_t0(dataset_id, meta_row)

        grad0 = read_gradient_table(dataset_id)
        grad, switched = canonicalize_gradient(grad0, family)
        rec["gradient_ok"] = True
        rec["AB_switched"] = switched
        rec["has_C_or_D_gradient"] = bool(
            pd.to_numeric(grad.get("C", 0), errors="coerce").fillna(0).abs().gt(1e-9).any()
            or pd.to_numeric(grad.get("D", 0), errors="coerce").fillna(0).abs().gt(1e-9).any()
        )
        flow = pd.to_numeric(grad["flow"], errors="coerce").dropna()
        rec["flow_nunique"] = int(flow.nunique(dropna=True)) if len(flow) else 0
        pB = pd.to_numeric(grad["B"], errors="coerce").to_numpy(float)
        times = pd.to_numeric(grad["time"], errors="coerce").to_numpy(float)
        if len(pB) and np.isfinite(pB).any():
            max_idx = int(np.nanargmax(pB))
            rec["pB_max"] = float(np.nanmax(pB))
            rec["t_pB_max"] = float(times[max_idx])
            rec["pB_start"] = float(pB[0])
            rec["pB_end"] = float(pB[-1])
        else:
            rec["pB_max"] = np.nan
            rec["t_pB_max"] = np.nan
            rec["pB_start"] = np.nan
            rec["pB_end"] = np.nan
        infl = graphormer_inflections(pB, times)
        rec["n_inflections"] = int(len(infl))

        rec["A_solv"] = dominant_solvent(dataset_id, "A", meta_row)
        rec["B_solv"] = dominant_solvent(dataset_id, "B", meta_row)
        if switched:
            rec["A_solv"], rec["B_solv"] = rec["B_solv"], rec["A_solv"]
        rec["has_excluded_solvent_iPrOH_or_acetone"] = bool(
            solvent_is_excluded(rec["A_solv"]) or solvent_is_excluded(rec["B_solv"])
        )
        rec["method_read_error"] = ""
    except Exception as e:
        rec.update({
            "family": "unknown",
            "column_name": "",
            "hplc_type": "",
            "t0_min": np.nan,
            "gradient_ok": False,
            "AB_switched": False,
            "has_C_or_D_gradient": True,
            "flow_nunique": 999,
            "pB_max": np.nan,
            "t_pB_max": np.nan,
            "pB_start": np.nan,
            "pB_end": np.nan,
            "n_inflections": 999,
            "A_solv": "",
            "B_solv": "",
            "has_excluded_solvent_iPrOH_or_acetone": True,
            "method_read_error": f"{type(e).__name__}: {e}",
        })
    return rec


def to_numeric_df(df: pd.DataFrame, prefix: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    if prefix is not None:
        out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def find_smiles_col(df: pd.DataFrame):
    candidates = [c for c in df.columns if str(c).lower() in {"smiles", "canonical_smiles", "canonical"}]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if "smiles" in str(c).lower():
            return c
    raise KeyError("Could not find SMILES column")


def find_rt_col(df: pd.DataFrame):
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"rt", "retention_time", "retention time"} or "retention" in lc:
            return c
    raise KeyError("Could not find RT column")


def find_name_col(df: pd.DataFrame):
    for c in df.columns:
        if str(c).lower() in {"name", "compound", "compound_name", "id"}:
            return c
    return None


def read_rt_table(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    path = PROCESSED_DIR / dataset_id / f"{dataset_id}_rtdata_canonical_success.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, sep="\t", index_col=0)
    except Exception:
        txt = path.read_text(errors="ignore").replace("#", "Q")
        tmp = OUT_DIR / f"__tmp_rt_{dataset_id}.tsv"
        tmp.write_text(txt)
        try:
            df = pd.read_csv(tmp, sep="\t", index_col=0)
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass
    df.index = df.index.astype(str)
    return df


def read_descriptor_tables(dataset_id):
    dataset_id = to_dataset_id(dataset_id)
    d = PROCESSED_DIR / dataset_id
    rdk_path = d / f"{dataset_id}_RDKdescriptor_canonical_success.tsv"
    ff_path = d / f"{dataset_id}_FFdescPolar_canonical_success.tsv"
    if not rdk_path.exists() or not ff_path.exists():
        raise FileNotFoundError(f"Missing descriptor files for {dataset_id}")
    rdk = pd.read_csv(rdk_path, sep="\t", index_col=0)
    ff = pd.read_csv(ff_path, sep="\t", index_col=0)
    rdk.index = rdk.index.astype(str)
    ff.index = ff.index.astype(str)
    return to_numeric_df(rdk, prefix="rdk__"), to_numeric_df(ff, prefix="ff__")


def load_method_rows_with_descriptors(dataset_id: str) -> pd.DataFrame:
    dataset_id = to_dataset_id(dataset_id)
    rt = read_rt_table(dataset_id)
    rdk, ff = read_descriptor_tables(dataset_id)
    smiles_col = find_smiles_col(rt)
    rt_col = find_rt_col(rt)
    name_col = find_name_col(rt)
    out = pd.DataFrame({
        "dir": dataset_id,
        "dataset_id": dataset_id,
        "row_id": rt.index.astype(str),
        "name": rt[name_col].astype(str).values if name_col is not None else rt.index.astype(str),
        "smiles": rt[smiles_col].astype(str).str.strip().values,
        "rt": pd.to_numeric(rt[rt_col], errors="coerce").values,
    })
    has_rdk = out["row_id"].isin(rdk.index)
    has_ff = out["row_id"].isin(ff.index)
    out = out.loc[has_rdk & has_ff].copy()
    if len(out) == 0:
        return out
    ids = out["row_id"].astype(str).tolist()
    desc = pd.concat([rdk.loc[ids].reset_index(drop=True), ff.loc[ids].reset_index(drop=True)], axis=1)
    valid_desc = (~desc.isna().all(axis=1).values)
    valid_basic = out["smiles"].ne("") & out["smiles"].str.lower().ne("nan") & out["rt"].notna()
    out = out.loc[valid_desc & valid_basic.values].reset_index(drop=True)
    desc = desc.loc[valid_desc & valid_basic.values].reset_index(drop=True)
    if DEDUPLICATE_SMILES_WITHIN_TASK and len(out):
        keep = ~out["smiles"].duplicated(keep="first")
        out = out.loc[keep].reset_index(drop=True)
        desc = desc.loc[keep.values].reset_index(drop=True)
    return pd.concat([out, desc.astype(np.float32)], axis=1)


def restrict_to_method_ids(df, method_ids):
    method_ids = set(map(to_dataset_id, method_ids))
    return df.loc[df["dataset_id"].map(to_dataset_id).isin(method_ids)].copy().reset_index(drop=True)


def remove_nonretained_rows(df, method_info, policy):
    if policy in {None, False, "none", "off"}:
        return df.copy().reset_index(drop=True)
    out_parts = []
    info = method_info.set_index("dataset_id")
    for dataset_id, sub in df.groupby("dataset_id", sort=False):
        dataset_id = to_dataset_id(dataset_id)
        keep = pd.Series(True, index=sub.index)
        if policy == "repo_code_0186_200s":
            if dataset_id == "0186":
                keep = pd.to_numeric(sub["rt"], errors="coerce") >= (200.0 / 60.0)
        elif policy == "t0_plus_1pct_gradient":
            if dataset_id in info.index:
                t0 = _as_float(info.loc[dataset_id, "t0_min"])
                t_pBmax = _as_float(info.loc[dataset_id, "t_pB_max"])
                if np.isfinite(t0) and np.isfinite(t_pBmax):
                    threshold = float(t0) + 0.01 * float(t_pBmax)
                    keep = pd.to_numeric(sub["rt"], errors="coerce") >= threshold
        else:
            raise ValueError(f"Unknown NONRETAINED_POLICY: {policy}")
        out_parts.append(sub.loc[keep].copy())
    return pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()


def summarize_rt_rows(df, source, family, stage):
    smiles = df["smiles"].astype(str).str.strip() if len(df) else pd.Series(dtype=str)
    return {
        "source": source,
        "family": family,
        "stage": stage,
        "n_methods": int(df["dataset_id"].nunique()) if len(df) else 0,
        "n_rt_rows": int(len(df)),
        "n_unique_smiles": int(smiles.nunique()) if len(smiles) else 0,
    }


def select_gradient_points(dataset_id, family):
    try:
        grad0 = read_gradient_table(dataset_id)
        grad, _ = canonicalize_gradient(grad0, family)
        pB = pd.to_numeric(grad["B"], errors="coerce").to_numpy(float)
        times = pd.to_numeric(grad["time"], errors="coerce").to_numpy(float)
        if len(pB) == 0:
            return [(np.nan, np.nan)] * 4
        pts = []
        if np.isfinite(times[0]) and np.isfinite(pB[0]):
            pts.append((float(times[0]), float(pB[0])))
        pts.extend(graphormer_inflections(pB, times))
        if np.isfinite(pB).any():
            max_idx = int(np.nanargmax(pB))
            pts.append((float(times[max_idx]), float(pB[max_idx])))
        if np.isfinite(times[-1]) and np.isfinite(pB[-1]):
            pts.append((float(times[-1]), float(pB[-1])))

        # Deduplicate by approximate time and percent.
        clean = []
        seen = set()
        for t, b in sorted(pts, key=lambda x: x[0]):
            if not (np.isfinite(t) and np.isfinite(b)):
                continue
            key = (round(float(t), 4), round(float(b), 4))
            if key not in seen:
                clean.append((float(t), float(b)))
                seen.add(key)
        if len(clean) > 4:
            # Keep start, end, and earliest internal points. This preserves fixed length.
            clean = clean[:3] + [clean[-1]]
        while len(clean) < 4:
            clean.append((np.nan, np.nan))
        return clean[:4]
    except Exception:
        return [(np.nan, np.nan)] * 4




# ============================================================
    # Cell 3. Graph cache, datasets, and Pretrain-Multi model
    # ============================================================
    # The model definition is inherited from CURATED_04_RadonPyPretrain_multi.
    # Per-run globals (y_mean, y_std, method_vocab, cat_vocabs) are assigned
    # inside run_one_seed before datasets are instantiated.

# ============================================================
# Cell 5. Graph cache, datasets, final model definition, and dataloaders
# ============================================================
ATOM_CAT_DIMS = [119, 7, 11, 2, 9, 9, 2, 8, 2, 9, 9]
ATOM_FLOAT_DIM = 1
BOND_CAT_DIMS = [8, 2, 2, 8]
HYBRIDIZATION_TO_INT = {
    HybridizationType.UNSPECIFIED: 0,
    HybridizationType.S: 1,
    HybridizationType.SP: 2,
    HybridizationType.SP2: 3,
    HybridizationType.SP3: 4,
    HybridizationType.SP3D: 5,
    HybridizationType.SP3D2: 6,
}
BOND_TO_INT = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

def _safe_int_clamp(x, lo, hi):
    try:
        return int(max(lo, min(hi, int(x))))
    except Exception:
        return int(lo)

def _torch_load_full(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

class PrecomputedPyGGraphCache:
    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            try:
                loaded = _torch_load_full(self.cache_path, map_location='cpu')
                if not isinstance(loaded, dict):
                    raise TypeError(f'Expected a dictionary cache, found {type(loaded).__name__}.')
                self.cache = loaded
                print(f'Loaded graph cache: {self.cache_path}, entries={len(self.cache)}')
                # Older PyG Data objects may not have explicit num_nodes. Because
                # these graphs use x_cat/x_float instead of x, PyG can otherwise
                # infer an incorrect node count for molecules with isolated atoms.
                self.fix_num_nodes_in_cache(save=False)
            except Exception as exc:
                corrupt_path = self.cache_path.with_name(f'{self.cache_path.name}.corrupt')
                if corrupt_path.exists():
                    corrupt_path = self.cache_path.with_name(
                        f'{self.cache_path.name}.corrupt.{time.time_ns()}'
                    )
                try:
                    self.cache_path.replace(corrupt_path)
                    quarantine_note = f' Archived the unreadable file as {corrupt_path}.'
                except OSError as quarantine_exc:
                    quarantine_note = (
                        ' The unreadable file could not be archived '
                        f'({type(quarantine_exc).__name__}: {quarantine_exc}).'
                    )
                print(
                    '[WARN] Graph cache could not be loaded '
                    f'({type(exc).__name__}: {exc}).{quarantine_note} '
                    'A new cache will be built automatically.'
                )

    def _save_atomic(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_name(
            f'.{self.cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}'
        )
        try:
            torch.save(self.cache, temp_path)
            temp_path.replace(self.cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _ensure_num_nodes(self, data):
        if hasattr(data, 'x_cat'):
            data.num_nodes = int(data.x_cat.size(0))
        return data
    def fix_num_nodes_in_cache(self, save: bool = False):
        fixed = 0
        bad = []
        for smi, data in list(self.cache.items()):
            if not hasattr(data, 'x_cat'):
                bad.append((smi, 'missing x_cat'))
                continue
            n = int(data.x_cat.size(0))
            old = getattr(data, 'num_nodes', None)
            if old != n:
                data.num_nodes = n
                fixed += 1
            if hasattr(data, 'x_float') and int(data.x_float.size(0)) != n:
                bad.append((smi, f'x_float={data.x_float.size(0)} vs x_cat={n}'))
            if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
                max_node = int(data.edge_index.max().item())
                if max_node >= n:
                    bad.append((smi, f'edge_index max={max_node} >= num_nodes={n}'))
        if fixed:
            print(f'Fixed num_nodes for {fixed} cached graphs.')
        if bad:
            print('[WARN] possible bad cached graphs:', bad[:5], '... total=', len(bad))
        if save:
            self._save_atomic()
        return fixed, bad
    def build_one(self, smiles: str):
        smi = str(smiles).strip()
        if smi in self.cache:
            return self._ensure_num_nodes(self.cache[smi])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles('C')
        atom_cat, atom_float = [], []
        for atom in mol.GetAtoms():
            hyb = HYBRIDIZATION_TO_INT.get(atom.GetHybridization(), 0)
            atom_cat.append([
                _safe_int_clamp(atom.GetAtomicNum(), 0, 118),
                _safe_int_clamp(atom.GetDegree(), 0, 6),
                _safe_int_clamp(atom.GetFormalCharge() + 5, 0, 10),
                1 if atom.GetIsAromatic() else 0,
                _safe_int_clamp(atom.GetTotalNumHs(), 0, 8),
                _safe_int_clamp(atom.GetTotalValence(), 0, 8),
                1 if atom.IsInRing() else 0,
                _safe_int_clamp(hyb, 0, 7),
                1 if atom.HasProp('_ChiralityPossible') else 0,
                _safe_int_clamp(atom.GetImplicitValence(), 0, 8),
                _safe_int_clamp(atom.GetExplicitValence(), 0, 8),
            ])
            atom_float.append([float(atom.GetMass()) / 200.0])
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            btype = BOND_TO_INT.get(bond.GetBondType(), 0)
            stereo = _safe_int_clamp(int(bond.GetStereo()), 0, 7)
            attr = [btype, 1 if bond.GetIsConjugated() else 0, 1 if bond.IsInRing() else 0, stereo]
            edge_index += [[i, j], [j, i]]
            edge_attr += [attr, attr]
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        n_atoms = len(atom_cat)
        data = PyGData(
            x_cat=torch.tensor(atom_cat, dtype=torch.long),
            x_float=torch.tensor(atom_float, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smi,
            num_nodes=n_atoms,
        )
        self.cache[smi] = data
        return data
    def build(self, smiles_list, save=True):
        for smi in tqdm(pd.Series(smiles_list).dropna().astype(str).unique(), desc='Building PyG graph cache'):
            self.build_one(smi)
        if save:
            self.fix_num_nodes_in_cache(save=False)
            self._save_atomic()
    def get(self, smiles):
        return self._ensure_num_nodes(self.build_one(smiles))



def move_batch_to_device(batch):
    out = {}
    for k, v in batch.items():
        if k == 'graph':
            out[k] = v.to(DEVICE)
        elif torch.is_tensor(v):
            out[k] = v.to(DEVICE)
        else:
            out[k] = v
    return out

class GraphEncoder(nn.Module):
    def __init__(self, d_model=256, layers=5, dropout=0.18):
        super().__init__()
        self.d_model = d_model
        self.atom_embeddings = nn.ModuleList([nn.Embedding(dim, d_model) for dim in ATOM_CAT_DIMS])
        self.atom_float_proj = nn.Linear(ATOM_FLOAT_DIM, d_model)
        self.atom_norm = nn.LayerNorm(d_model)
        self.atom_ffn = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*d_model, d_model))
        self.bond_embeddings = nn.ModuleList([nn.Embedding(dim, d_model) for dim in BOND_CAT_DIMS])
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*d_model, d_model))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(d_model))
        self.g0_proj = nn.Sequential(nn.Linear(3*d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
    def _encode_atoms(self, graph_batch):
        x_cat = graph_batch.x_cat.long()
        h = 0.0
        for j, emb in enumerate(self.atom_embeddings):
            h = h + emb(x_cat[:, j].clamp(min=0, max=emb.num_embeddings-1))
        x_float = graph_batch.x_float.float()
        if x_float.dim() == 1:
            x_float = x_float[:, None]
        h = h + self.atom_float_proj(x_float[:, :ATOM_FLOAT_DIM])
        h = self.atom_norm(h)
        h = h + self.dropout(self.atom_ffn(h))
        return h
    def _encode_edges(self, edge_attr, h):
        if edge_attr is None or edge_attr.numel() == 0:
            return h.new_zeros((0, h.size(-1)))
        edge_attr = edge_attr.long()
        e = 0.0
        for j, emb in enumerate(self.bond_embeddings):
            e = e + emb(edge_attr[:, j].clamp(min=0, max=emb.num_embeddings-1))
        return e.to(dtype=h.dtype)
    def forward(self, graph_batch):
        edge_index, edge_attr, batch = graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        h = self._encode_atoms(graph_batch)
        for conv, norm in zip(self.convs, self.norms):
            edge_emb = self._encode_edges(edge_attr, h)
            h_new = conv(h, edge_index, edge_emb)
            h = norm(h + self.dropout(h_new))
        B = int(batch.max().item()) + 1 if batch.numel() else 1
        g_mean = global_mean_pool(h, batch)
        n_atoms = torch.bincount(batch, minlength=B).float().to(h.device).clamp(min=1.0)
        g_sum = global_add_pool(h, batch) / torch.sqrt(n_atoms).unsqueeze(-1)
        g_max = global_max_pool(h, batch)
        g0 = self.g0_proj(torch.cat([g_mean, g_sum, g_max], dim=-1))
        dense_h, atom_mask = to_dense_batch(h, batch)  # mask True for real atoms
        return dense_h, atom_mask, g0

class MolecularTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4, layers=1, dropout=0.18):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4*d_model, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(d_model)
    def forward(self, g0, H_atom, atom_mask):
        B = H_atom.size(0)
        x = torch.cat([g0[:, None, :], H_atom], dim=1)
        key_padding = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=H_atom.device), ~atom_mask], dim=1)
        y = self.encoder(x, src_key_padding_mask=key_padding)
        y = self.out_norm(y)
        g_fused = y[:, 0, :]
        H_mol = y[:, 1:, :]
        return g_fused, H_mol


class TypedAttentionBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dropout=0.18, num_types=6):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.type_emb = nn.Embedding(num_types, d_model)
        self.type_bias = nn.Parameter(torch.zeros(num_types, num_types))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, type_ids, key_padding_mask=None):
        x = x + self.type_emb(type_ids)
        h = self.ln1(x)
        B, L, D = h.shape
        qkv = self.qkv(h).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.type_bias[type_ids[:, :, None], type_ids[:, None, :]]
        scores = scores + bias[:, None, :, :]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.dropout(self.out(y))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TypedInteractionTransformer(nn.Module):
    def __init__(self, d_model=256, layers=3, num_heads=4, dropout=0.18, num_types=6):
        super().__init__()
        self.blocks = nn.ModuleList([TypedAttentionBlock(d_model, num_heads, dropout, num_types) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, type_ids, key_padding_mask=None):
        for blk in self.blocks:
            x = blk(x, type_ids, key_padding_mask)
        return self.norm(x)

class RTDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dropout=0.18):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_m = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, memory, memory_key_padding_mask=None):
        qn = self.ln_q(q)
        mn = self.ln_m(memory)
        y, attn = self.cross(qn, mn, mn, key_padding_mask=memory_key_padding_mask, need_weights=True, average_attn_weights=False)
        q = q + self.dropout(y)
        q = q + self.dropout(self.ffn(self.ln_ffn(q)))
        return q, attn

class RTQueryDecoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4, layers=2, dropout=0.18):
        super().__init__()
        self.layers = nn.ModuleList([RTDecoderLayer(d_model, num_heads, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d_model)
    def forward(self, g_fused, memory, memory_key_padding_mask=None):
        q0 = g_fused[:, None, :]
        q = q0
        last_attn = None
        for layer in self.layers:
            q, last_attn = layer(q, memory, memory_key_padding_mask)
        # One final RT vector. Keep pure molecular anchor through residual.
        r_rt = self.out_norm(g_fused + q.squeeze(1))
        return r_rt, last_attn.detach() if last_attn is not None else None

class SingleRTHead(nn.Module):
    def __init__(self, d_model=256, dropout=0.18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, 1),
        )
    def forward(self, z, method_idx=None):
        return self.net(z).squeeze(-1)

class MultiRTHead(nn.Module):
    def __init__(self, d_model=256, num_methods=1, hidden=64, dropout=0.18):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)) for _ in range(num_methods)])
        self.default_head = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, z, method_idx=None):
        h = self.trunk(z)
        pred = self.default_head(h).squeeze(-1)
        if method_idx is None:
            return pred
        method_idx = method_idx.to(h.device)
        valid = method_idx >= 0
        if valid.any():
            for m in method_idx[valid].unique().tolist():
                m_int = int(m)
                if 0 <= m_int < len(self.heads):
                    mask = method_idx == m_int
                    pred[mask] = self.heads[m_int](h[mask]).squeeze(-1)
        return pred



# ============================================================
# Cell 4. Curated-10 RadonPy row-level split and transforms
# ============================================================


def find_radonpy_file():
    candidates = [
        DATA_DIR / 'RadonPy_20260611' / 'RadonPySM_checkeq_masked.csv',
        DATA_DIR / 'RadonPySM_checkeq_masked.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    files = sorted(DATA_DIR.glob('**/*RadonPy*masked*.csv'))
    if files:
        return files[0]
    raise FileNotFoundError('Cannot find RadonPySM_checkeq_masked.csv')

def apply_radonpy_transform(raw_values, transform: str):
    raw = np.asarray(raw_values, dtype=np.float32)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(raw)
    if transform == 'log':
        valid = finite & (raw > 0)
        out[valid] = np.log(raw[valid]).astype(np.float32)
    elif transform == 'signed_log':
        valid = finite
        out[valid] = (np.sign(raw[valid]) * np.log1p(np.abs(raw[valid]))).astype(np.float32)
    elif transform == 'none':
        valid = finite
        out[valid] = raw[valid]
    else:
        raise ValueError(f'Unknown RadonPy transform: {transform}')
    return out

class RadonPyDataset(Dataset):
    def __init__(self, df, y_std_arr, y_mask, graph_cache):
        self.df = df.reset_index(drop=True).copy()
        self.y = y_std_arr.astype(np.float32)
        self.mask = y_mask.astype(np.float32)
        self.graph_cache = graph_cache
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return {
            'graph': self.graph_cache.get(self.df.iloc[idx]['smiles']),
            'y': self.y[idx],
            'mask': self.mask[idx],
            'smiles': self.df.iloc[idx]['smiles'],
        }

def collate_radon(batch):
    graphs = []
    for b in batch:
        g = b['graph']
        if hasattr(g, 'x_cat'):
            g.num_nodes = int(g.x_cat.size(0))
        graphs.append(g)
    return {
        'graph': PyGBatch.from_data_list(graphs),
        'y': torch.tensor(np.stack([b['y'] for b in batch]), dtype=torch.float32),
        'mask': torch.tensor(np.stack([b['mask'] for b in batch]), dtype=torch.float32),
        'smiles': [b['smiles'] for b in batch],
    }



# ============================================================
# Cell 5. Losses, evaluation, checkpoint selection, and metrics
# ============================================================
def unscale_rt(y_std_arr):
    return np.asarray(y_std_arr, dtype=np.float32) * y_std + y_mean

def rt_loss_fn(pred, y):
    return (
        CFG['rt_loss_huber_weight'] * F.smooth_l1_loss(pred, y, beta=CFG['rt_huber_beta'])
        + CFG['rt_loss_mse_weight'] * F.mse_loss(pred, y)
    )

def masked_radon_loss(pred, y, mask):
    m = mask > 0.5
    if m.sum() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[m], y[m], beta=1.0)

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    try:
        if len(y_true) < 2 or np.std(y_true) < 1e-12:
            return np.nan
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan

def safe_mape_pct(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if valid.sum() == 0:
        return np.nan, 0
    value = np.mean(np.abs((y_pred[valid] - y_true[valid]) / y_true[valid])) * 100.0
    return float(value), int(valid.sum())


@torch.no_grad()

@torch.no_grad()
def evaluate_radon(model, loader):
    if loader is None or model.radon_heads is None:
        return np.nan
    model.eval()
    losses = []
    for batch in loader:
        graph = batch['graph'].to(DEVICE)
        y = batch['y'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        pred = model.forward_radon(graph)
        losses.append(float(masked_radon_loss(pred, y, mask).detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan

def valid_score_from_metrics(m):
    w = CFG['composite_valid_weights']
    values_weights = [
        (m.get('mae_sec', np.nan), w['all']),
        (m.get('clean_overlap_mae_sec', np.nan), w['clean_overlap']),
        (m.get('nonoverlap_mae_sec', np.nan), w['nonoverlap']),
    ]
    vals, weights = [], []
    for value, weight in values_weights:
        if np.isfinite(value):
            vals.append(value)
            weights.append(weight)
    return float(np.average(vals, weights=weights)) if vals else np.inf

def train_one_rt_epoch(model, optimizer, loader):
    model.train()
    losses = []
    for batch in tqdm(loader, desc='RT train', leave=False):
        batch = move_batch_to_device(batch)
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model.forward_rt(batch)
        loss = rt_loss_fn(pred.float(), batch['y'].float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan

def train_one_radon_epoch(model, optimizer, loader, desc='RadonPy'):
    model.train()
    losses = []
    for rb in tqdm(loader, desc=desc, leave=False):
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_radon(rb['graph'].to(DEVICE))
        loss = masked_radon_loss(pred, rb['y'].to(DEVICE), rb['mask'].to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan

def make_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG['lr'], weight_decay=CFG['weight_decay'],
    )

def save_per_task_metrics(pred_df, out_dir, split_name):
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame()
    rows = []
    for d, g in pred_df.groupby('dir'):
        met = compute_metrics_from_arrays(g['true_min'].to_numpy(float), g['pred_min'].to_numpy(float))
        rows.append({
            'dir': str(d).zfill(4),
            'n_rows': int(len(g)),
            'n_mols': int(g['mol_key'].nunique()),
            **met,
        })
    out = pd.DataFrame(rows).sort_values('mae_sec', ascending=False)
    out.to_csv(Path(out_dir) / f'per_task_metrics_{split_name}.csv', index=False)
    summary_rows = []
    for min_rows in [1, 5, 10]:
        sub = out[out['n_rows'] >= min_rows]
        summary_rows.append({
            'split': split_name,
            'min_rows': min_rows,
            'n_tasks': int(len(sub)),
            'macro_mae_sec': float(sub['mae_sec'].mean()) if len(sub) else np.nan,
            'macro_mape_pct': float(sub['mape_pct'].mean()) if len(sub) else np.nan,
            'macro_r2': float(sub['r2'].mean()) if len(sub) else np.nan,
            'median_task_mae_sec': float(sub['mae_sec'].median()) if len(sub) else np.nan,
        })
    pd.DataFrame(summary_rows).to_csv(Path(out_dir) / f'per_task_metrics_{split_name}_summary.csv', index=False)
    return out


# ============================================================
# Cell 6. RT split loading, mol_key audit, and one-seed runner
# ============================================================
ENV_FEATURE_CACHE = {}
RADON_KEY_CACHE = None


def smiles_to_mol_key(smiles, strip_stereo=True):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    if strip_stereo:
        Chem.RemoveStereochemistry(mol)
    try:
        ik = Chem.MolToInchiKey(mol)
        return ik.split('-')[0] if ik else None
    except Exception:
        return None




def load_radon_keys_for_flags():
    global RADON_KEY_CACHE
    if RADON_KEY_CACHE is not None:
        return RADON_KEY_CACHE
    try:
        p = find_radonpy_file()
        df = pd.read_csv(p, index_col=0, low_memory=False)
        smiles_col = 'smiles_list_canonical' if 'smiles_list_canonical' in df.columns else ('smiles' if 'smiles' in df.columns else None)
        if smiles_col is None:
            RADON_KEY_CACHE = set()
        else:
            RADON_KEY_CACHE = set(df[smiles_col].map(smiles_to_mol_key).dropna().astype(str))
    except Exception:
        RADON_KEY_CACHE = set()
    return RADON_KEY_CACHE


def torch_load_compat(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)



# ============================================================
# Cell 6b. Weight-only Methods-v2 overrides:
# 10 split seeds, brand encoding, RadonPy CORR_PROPS=all, E1–E9 training modes.
# ============================================================

# -------------------------
# Split seed discovery from the 00 data-split notebook
# -------------------------
def discover_split_seeds(max_seeds: int = 10) -> List[int]:
    summary_path = Path(SPLIT_ROOT) / 'accepted_splits_summary.csv'
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        seeds = df['seed'].dropna().astype(int).head(max_seeds).tolist()
        if len(seeds) < max_seeds:
            raise RuntimeError(f'Need {max_seeds} accepted split seeds, found {len(seeds)} in {summary_path}')
        print(f'Loaded {len(seeds)} split seeds from {summary_path}:', seeds)
        return seeds
    dirs = sorted(Path(SPLIT_ROOT).glob('seed_*')) if Path(SPLIT_ROOT).exists() else []
    seeds = []
    for d in dirs:
        m = re.search(r'seed_(\d+)', d.name)
        if m and all((d / fn).exists() for fn in ['train.csv', 'valid.csv', 'internal_test.csv', 'external_ood.csv']):
            seeds.append(int(m.group(1)))
    if len(seeds) >= max_seeds:
        seeds = sorted(seeds)[:max_seeds]
        print(f'Loaded {len(seeds)} split seeds from split directories:', seeds)
        return seeds
    print('[WARN] Could not discover split summary/directories; using fallback SPLIT_SEEDS:', SPLIT_SEEDS[:max_seeds])
    return list(SPLIT_SEEDS[:max_seeds])


# -------------------------
# Brand normalization and environment features
# -------------------------
BRAND_PATTERNS = [
    (r'\bagilent\b|\bzorbax\b|\beclipse\b|\bporoshell\b|\bbonus[- ]?rp\b', 'Agilent'),
    (r'\bwaters\b|\bacquity\b|\bxbridge\b|\bxselect\b|\bsunfire\b|\batlantis\b|\bcortecs\b|\bsymmetry\b|\bbeh\b', 'Waters'),
    (r'\bshimadzu\b|\bshim[- ]?pack\b', 'Shimadzu'),
    (r'\bthermo\b|\bhypersil\b|\baccucore\b|\bsyncronis\b', 'Thermo'),
    (r'\bphenomenex\b|\bkinetex\b|\bluna\b|\bgemini\b|\bsynergi\b|\bjupiter\b', 'Phenomenex'),
    (r'\bmerck\b|\bsupelco\b|\bsigma\b|\bascentis\b|\blicrospher\b|\bchromolith\b|\bsequant\b', 'Merck/Supelco'),
    (r'\bymc\b', 'YMC'),
    (r'\btosoh\b|\btskgel\b|\btsk[- ]?gel\b', 'Tosoh'),
    (r'\bgl sciences\b|\binertsil\b', 'GL Sciences'),
    (r'\brestek\b|\braptor\b', 'Restek'),
    (r'\bmacherey\b|\bnagel\b|\bnucleodur\b|\bnucleosil\b', 'Macherey-Nagel'),
    (r'\bace\b|\badvanced chromatography technologies\b', 'ACE'),
    (r'\bcosmosil\b|\bnacalai\b', 'Nacalai/COSMOSIL'),
    (r'\bkromasil\b', 'Kromasil'),
    (r'\bimtak\b|\bscherzo\b', 'Imtakt'),
    (r'\bhilic\.?com\b|\bhilicon\b', 'HILICON'),
]

def normalize_brand_name(*values) -> str:
    text = ' '.join([cat_or_na(v) for v in values if cat_or_na(v) != 'NA']).strip()
    if text == '':
        return 'NA'
    s = text.lower()
    for pattern, brand in BRAND_PATTERNS:
        if re.search(pattern, s, flags=re.I):
            return brand
    return 'Other'

def get_column_brand(dataset_id, meta_row=None, column_name=None):
    dataset_id = to_dataset_id(dataset_id)
    if meta_row is None:
        meta_row = get_meta_row(dataset_id)
    if column_name is None:
        column_name = get_column_name(dataset_id, meta_row)
    values = []
    for names in [
        ['column.manufacturer', 'column.company', 'column.brand', 'column.vendor', 'manufacturer', 'company', 'brand', 'vendor'],
        ['instrument.manufacturer', 'instrument.company', 'device.manufacturer', 'device.company'],
    ]:
        values.append(_case_insensitive_get(meta_row, names, default=''))
    values.append(column_name)
    return normalize_brand_name(*values)

def build_selected_env_features(dataset_id: str, rec: Optional[dict] = None) -> Dict[str, Any]:
    dataset_id = to_dataset_id(dataset_id)
    meta_row = get_meta_row(dataset_id)
    if rec is None:
        rec = compute_method_properties(dataset_id)
    family = rec.get('family', 'unknown')
    switched = bool(rec.get('AB_switched', False))
    column_name = rec.get('column_name', get_column_name(dataset_id, meta_row))
    usp_code = get_meta_or_matrix(dataset_id, ['column.usp.code', 'column.usp', 'usp_code', 'col_usp_code'], matrix_col=1, default='')
    col_length = get_meta_or_matrix(dataset_id, ['column.length', 'column.length.mm', 'col_length'], matrix_col=2, default=np.nan)
    col_diam = get_meta_or_matrix(dataset_id, ['column.id', 'column.innerdiam', 'column.inner_diameter', 'col_innerdiam'], matrix_col=3, default=np.nan)
    part_size = get_meta_or_matrix(dataset_id, ['column.particle.size', 'column.particle_size', 'col_part_size'], matrix_col=4, default=np.nan)
    temp = get_meta_or_matrix(dataset_id, ['column.temperature', 'temperature', 'temp'], matrix_col=5, default=np.nan)
    flow = np.nan
    try:
        grad, _ = canonicalize_gradient(read_gradient_table(dataset_id), family)
        flow_vals = pd.to_numeric(grad['flow'], errors='coerce').dropna()
        if len(flow_vals):
            flow = float(flow_vals.median())
    except Exception:
        flow = get_meta_or_matrix(dataset_id, ['column.flowrate', 'flowrate', 'flow'], matrix_col=6, default=np.nan)
    t0 = rec.get('t0_min', get_t0(dataset_id, meta_row))
    A_solv = rec.get('A_solv', dominant_solvent(dataset_id, 'A', meta_row))
    B_solv = rec.get('B_solv', dominant_solvent(dataset_id, 'B', meta_row))
    pH_A = get_meta_or_matrix(dataset_id, ['eluent.A.pH', 'A_pH', 'pH_A'], matrix_col=19, default=np.nan)
    pH_B = get_meta_or_matrix(dataset_id, ['eluent.B.pH', 'B_pH', 'pH_B'], matrix_col=20, default=np.nan)
    if switched:
        pH_A, pH_B = pH_B, pH_A
    gradient_cont = []
    for t, b in select_gradient_points(dataset_id, family):
        gradient_cont.extend([scale_or_missing(t, 60.0), pct_or_missing(b)])
    return {
        'column_cont': [
            scale_or_missing(col_length, 250.0),
            scale_or_missing(col_diam, 4.6),
            scale_or_missing(part_size, 10.0),
            scale_or_missing(temp, 100.0),
            scale_or_missing(t0, 10.0),
        ],
        'column_cat': [normalize_phase_type(usp_code, column_name, family)],
        'brand_cat': [get_column_brand(dataset_id, meta_row, column_name)],
        'solvent_cont': [pct_or_missing(rec.get('pB_start', np.nan)), ph_or_missing(pH_A), ph_or_missing(pH_B)],
        'solvent_cat': [normalize_solvent_name(A_solv), normalize_solvent_name(B_solv)],
        'gradient_cont': gradient_cont,
        'operation_cont': [scale_or_missing(flow, 2.0)],
        'family': family,
        'AB_switched': switched,
    }

def attach_env_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    env_cols = ['column_cont', 'column_cat', 'brand_cat', 'solvent_cont', 'solvent_cat', 'gradient_cont', 'operation_cont', 'family', 'AB_switched']
    for sid in sorted(df['dir'].astype(str).unique()):
        sid = str(sid).zfill(4)
        if sid not in ENV_FEATURE_CACHE:
            ENV_FEATURE_CACHE[sid] = build_selected_env_features(sid)
    records = [ENV_FEATURE_CACHE[str(sid).zfill(4)] for sid in df['dir'].astype(str)]
    env_df = pd.DataFrame(records)
    for col in env_cols:
        df[col] = env_df[col].values
    return df

def collect_vocab(df: pd.DataFrame, col: str, pos: int) -> Dict[str, int]:
    vals = ['<UNK>', 'NA']
    for xs in df[col].values:
        try:
            v = str(xs[pos])
            vals.append(v if v and v.lower() != 'nan' else 'NA')
        except Exception:
            vals.append('NA')
    ordered = ['<UNK>', 'NA'] + sorted(set(vals) - {'<UNK>', 'NA'})
    return {v: i for i, v in enumerate(ordered)}

def _cat_index(vocab_name: str, value: str) -> int:
    vocab = cat_vocabs.get(vocab_name, {'<UNK>': 0, 'NA': 1})
    return int(vocab.get(str(value), vocab.get('<UNK>', 0)))

# -------------------------
# Dataset / model overrides with brand_cat and no-device R2 support
# -------------------------
class RTGraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, graph_cache: PrecomputedPyGGraphCache):
        self.df = df.reset_index(drop=True).copy()
        self.graph_cache = graph_cache
        self.y = ((self.df['rt'].to_numpy(np.float32) - y_mean) / y_std).astype(np.float32)
        self.y_raw = self.df['rt'].to_numpy(np.float32)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d = str(row['dir']).zfill(4)
        brand_value = row['brand_cat'][0] if 'brand_cat' in row and isinstance(row['brand_cat'], (list, tuple, np.ndarray)) else 'NA'
        return {
            'graph': self.graph_cache.get(row['smiles']),
            'y': self.y[idx], 'y_raw': self.y_raw[idx],
            'dir': d, 'method_idx': method_vocab.get(d, -1),
            'row_id': row['row_id'], 'smiles': row['smiles'], 'mol_key': row['mol_key'],
            'clean_radonpy_overlap': bool(row.get('clean_radonpy_overlap', False)),
            'non_radonpy_overlap': bool(row.get('non_radonpy_overlap', False)),
            'column_cont': np.asarray(row['column_cont'], dtype=np.float32),
            'column_cat': np.asarray([_cat_index('column_cat0', str(row['column_cat'][0]))], dtype=np.int64),
            'brand_cat': np.asarray([_cat_index('brand_cat0', str(brand_value))], dtype=np.int64),
            'solvent_cont': np.asarray(row['solvent_cont'], dtype=np.float32),
            'solvent_cat': np.asarray([_cat_index('solvent_cat0', str(row['solvent_cat'][0])), _cat_index('solvent_cat1', str(row['solvent_cat'][1]))], dtype=np.int64),
            'gradient_cont': np.asarray(row['gradient_cont'], dtype=np.float32),
            'operation_cont': np.asarray(row['operation_cont'], dtype=np.float32),
        }

def collate_rt(batch):
    graphs = []
    for b in batch:
        g = b['graph']
        if hasattr(g, 'x_cat'):
            g.num_nodes = int(g.x_cat.size(0))
        graphs.append(g)
    return {
        'graph': PyGBatch.from_data_list(graphs),
        'y': torch.tensor([b['y'] for b in batch], dtype=torch.float32),
        'y_raw': torch.tensor([b['y_raw'] for b in batch], dtype=torch.float32),
        'dir': [b['dir'] for b in batch],
        'method_idx': torch.tensor([b['method_idx'] for b in batch], dtype=torch.long),
        'row_id': [b['row_id'] for b in batch],
        'smiles': [b['smiles'] for b in batch],
        'mol_key': [b['mol_key'] for b in batch],
        'clean_radonpy_overlap': torch.tensor([b['clean_radonpy_overlap'] for b in batch], dtype=torch.bool),
        'non_radonpy_overlap': torch.tensor([b['non_radonpy_overlap'] for b in batch], dtype=torch.bool),
        'column_cont': torch.tensor(np.stack([b['column_cont'] for b in batch]), dtype=torch.float32),
        'column_cat': torch.tensor(np.stack([b['column_cat'] for b in batch]), dtype=torch.long),
        'brand_cat': torch.tensor(np.stack([b['brand_cat'] for b in batch]), dtype=torch.long),
        'solvent_cont': torch.tensor(np.stack([b['solvent_cont'] for b in batch]), dtype=torch.float32),
        'solvent_cat': torch.tensor(np.stack([b['solvent_cat'] for b in batch]), dtype=torch.long),
        'gradient_cont': torch.tensor(np.stack([b['gradient_cont'] for b in batch]), dtype=torch.float32),
        'operation_cont': torch.tensor(np.stack([b['operation_cont'] for b in batch]), dtype=torch.float32),
    }

class EnvironmentEncoder(nn.Module):
    def __init__(self, d_model=256, dropout=0.18, cat_vocabs=None):
        super().__init__()
        cat_vocabs = cat_vocabs or {}
        emb_dim = 16
        self.column_emb = nn.Embedding(len(cat_vocabs.get('column_cat0', {'NA':0})), emb_dim)
        self.brand_emb = nn.Embedding(len(cat_vocabs.get('brand_cat0', {'NA':0})), emb_dim)
        self.solv_a_emb = nn.Embedding(len(cat_vocabs.get('solvent_cat0', {'NA':0})), emb_dim)
        self.solv_b_emb = nn.Embedding(len(cat_vocabs.get('solvent_cat1', {'NA':0})), emb_dim)
        self.column_mlp = nn.Sequential(nn.Linear(5 + 2*emb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, d_model), nn.LayerNorm(d_model))
        self.solvent_mlp = nn.Sequential(nn.Linear(3 + 2*emb_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, d_model), nn.LayerNorm(d_model))
        self.gradient_mlp = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, d_model), nn.LayerNorm(d_model))
        self.operation_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, d_model), nn.LayerNorm(d_model))
    def forward(self, batch):
        col_cat = self.column_emb(batch['column_cat'][:, 0])
        brand_cat = self.brand_emb(batch['brand_cat'][:, 0])
        e_col = self.column_mlp(torch.cat([batch['column_cont'], col_cat, brand_cat], dim=-1))
        solv_a = self.solv_a_emb(batch['solvent_cat'][:, 0])
        solv_b = self.solv_b_emb(batch['solvent_cat'][:, 1])
        e_solv = self.solvent_mlp(torch.cat([batch['solvent_cont'], solv_a, solv_b], dim=-1))
        e_grad = self.gradient_mlp(batch['gradient_cont'])
        e_oper = self.operation_mlp(batch['operation_cont'])
        return torch.stack([e_col, e_solv, e_grad, e_oper], dim=1)

class GraphEnvRTModel(nn.Module):
    def __init__(self, cat_vocabs, radon_targets=0, cfg=CFG, head_type='single', num_methods=1, use_device_metadata=True):
        super().__init__()
        d = int(cfg['d_model']); dropout = float(cfg['dropout'])
        self.head_type = head_type
        self.use_device_metadata = bool(use_device_metadata)
        self.gnn = GraphEncoder(d, int(cfg['gnn_layers']), dropout)
        self.mol_encoder = MolecularTransformerEncoder(d, int(cfg['num_heads']), int(cfg.get('molecular_transformer_layers', 1)), dropout)
        self.env_encoder = EnvironmentEncoder(d, dropout, cat_vocabs) if self.use_device_metadata else None
        self.interaction = TypedInteractionTransformer(d, int(cfg['transformer_layers']), int(cfg['num_heads']), dropout, num_types=6)
        self.decoder = RTQueryDecoder(d, int(cfg['num_heads']), int(cfg['decoder_layers']), dropout)
        self.rt_head = MultiRTHead(d, num_methods=num_methods, hidden=64, dropout=dropout) if head_type == 'multi' else SingleRTHead(d, dropout=dropout)
        self.radon_heads = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, radon_targets)) if radon_targets > 0 else None
    def encode_molecule(self, graph_batch):
        H_atom, atom_mask, g0 = self.gnn(graph_batch)
        g_fused, H_mol = self.mol_encoder(g0, H_atom, atom_mask)
        return g_fused, H_mol, atom_mask
    def encode_interaction(self, batch):
        g_fused, H_mol, atom_mask = self.encode_molecule(batch['graph'])
        B, Na, D = H_mol.shape
        if self.use_device_metadata:
            E_env = self.env_encoder(batch)
            x = torch.cat([g_fused[:, None, :], H_mol, E_env], dim=1)
            type_ids = torch.cat([
                torch.zeros(B, 1, dtype=torch.long, device=x.device),
                torch.ones(B, Na, dtype=torch.long, device=x.device),
                torch.tensor([2,3,4,5], dtype=torch.long, device=x.device).view(1,4).expand(B,-1),
            ], dim=1)
            key_padding = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=x.device), ~atom_mask, torch.zeros(B,4,dtype=torch.bool,device=x.device)], dim=1)
        else:
            x = torch.cat([g_fused[:, None, :], H_mol], dim=1)
            type_ids = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), torch.ones(B, Na, dtype=torch.long, device=x.device)], dim=1)
            key_padding = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=x.device), ~atom_mask], dim=1)
        x_env = self.interaction(x, type_ids, key_padding)
        return g_fused, x_env, type_ids, key_padding, {'g_fused': g_fused, 'atom_mask': atom_mask}
    def encode_representation(self, batch):
        g_fused, x_env, type_ids, key_padding, extra = self.encode_interaction(batch)
        r_rt, attn = self.decoder(g_fused, x_env, key_padding)
        extra.update({'r_rt': r_rt, 'attn': attn})
        return r_rt, extra
    def forward_rt(self, batch):
        z, extra = self.encode_representation(batch)
        return self.rt_head(z, method_idx=batch.get('method_idx', None)), extra
    def forward_radon(self, graph_batch):
        g_fused, _, _ = self.encode_molecule(graph_batch)
        if self.radon_heads is None:
            raise RuntimeError('Model has no RadonPy heads.')
        return self.radon_heads(g_fused)

# -------------------------
# RadonPy CORR_PROPS='all' registry and loader
# -------------------------
RADONPY_TARGET_REGISTRY = [
    {'target':'mol_weight_monomer1','category':'monomer size','lser':True,'transform':'log'},
    {'target':'vdw_volume_monomer1','category':'monomer size','lser':True,'transform':'log'},
    {'target':'qm_total_energy_monomer1','category':'QM monomer','lser':True,'transform':'signed_log'},
    {'target':'qm_homo_monomer1','category':'QM monomer','lser':True,'transform':'none'},
    {'target':'qm_lumo_monomer1','category':'QM monomer','lser':True,'transform':'none'},
    {'target':'qm_dipole_x_monomer1','category':'QM monomer','lser':False,'transform':'signed_log'},
    {'target':'qm_dipole_y_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_dipole_z_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_polarizability_monomer1','category':'QM monomer','lser':True,'transform':'log'},
    {'target':'qm_polarizability_xx_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_polarizability_yy_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_polarizability_zz_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_polarizability_xy_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_polarizability_xz_monomer1','category':'QM monomer','lser':False,'transform':'none'},
    {'target':'qm_polarizability_yz_monomer1','category':'QM monomer','lser':False,'transform':'signed_log'},
    {'target':'mol_weight','category':'bulk/structure','lser':True,'transform':'none'},
    {'target':'density','category':'bulk/structure','lser':True,'transform':'none'},
    {'target':'Rg','category':'bulk/structure','lser':True,'transform':'none'},
    {'target':'self-diffusion','category':'thermo/transport','lser':True,'transform':'signed_log'},
    {'target':'Cp','category':'thermo/transport','lser':True,'transform':'log'},
    {'target':'Cv','category':'thermo/transport','lser':True,'transform':'none'},
    {'target':'compressibility','category':'thermo/transport','lser':True,'transform':'log'},
    {'target':'isentropic_compressibility','category':'thermo/transport','lser':True,'transform':'log'},
    {'target':'bulk_modulus','category':'thermo/transport','lser':True,'transform':'log'},
    {'target':'isentropic_bulk_modulus','category':'thermo/transport','lser':True,'transform':'log'},
    {'target':'volume_expansion','category':'thermo/transport','lser':True,'transform':'signed_log'},
    {'target':'linear_expansion','category':'thermo/transport','lser':True,'transform':'signed_log'},
    {'target':'r2','category':'bulk/structure','lser':False,'transform':'none'},
    {'target':'static_dielectric_const','category':'dielectric/optical','lser':True,'transform':'log'},
    {'target':'nematic_order_parameter','category':'bulk/structure','lser':True,'transform':'log'},
    {'target':'refractive_index','category':'dielectric/optical','lser':True,'transform':'none'},
    {'target':'thermal_conductivity','category':'thermal conductivity','lser':True,'transform':'log'},
    {'target':'thermal_diffusivity','category':'thermal conductivity','lser':True,'transform':'none'},
    {'target':'TC_ke','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_pe','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_pair','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_bond','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_angle','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_dihed','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_improper','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'TC_kspace','category':'thermal conductivity','lser':False,'transform':'signed_log'},
    {'target':'dipole_mag','category':'dielectric/optical','lser':True,'transform':'log'},
]
RADONPY_RECOMMENDED_TARGETS = [r['target'] for r in RADONPY_TARGET_REGISTRY]
RADONPY_TARGET_TRANSFORMS = {r['target']: r['transform'] for r in RADONPY_TARGET_REGISTRY}

def _resolve_radon_target_column(df: pd.DataFrame, target: str) -> Optional[str]:
    for c in [target, target.replace('-', '_'), target.replace('_', '-')]:
        if c in df.columns:
            return c
        low = {str(x).lower(): x for x in df.columns}
        if c.lower() in low:
            return low[c.lower()]
    return None

def _iqr_outlier_count(vals) -> int:
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if len(v) < 4: return 0
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0: return 0
    return int(((v < q1 - 1.5*iqr) | (v > q3 + 1.5*iqr)).sum())

def prepare_radonpy_loaders(fraction: float, seed: int, graph_cache, out_dir: Path):
    out_dir = Path(out_dir)
    if fraction <= 0 or MOLECULE_MODE == 'M0_no_aux':
        cfg = {'enabled': False, 'target_policy': 'RadonPy disabled for M0', 'registered_targets': RADONPY_RECOMMENDED_TARGETS, 'used_targets': []}
        with open(out_dir / 'radonpy_target_config.json', 'w', encoding='utf-8') as f: json.dump(cfg, f, ensure_ascii=False, indent=2)
        pd.DataFrame(RADONPY_TARGET_REGISTRY).to_csv(out_dir / 'radonpy_property_transform_outlier_report.csv', index=False)
        return {'train_loader': None, 'valid_loader': None, 'targets': [], 'transforms': [], 'target_mean': None, 'target_std': None, 'n_train_used': 0, 'n_train_full': 0, 'n_valid': 0, 'radon_path': None}
    radon_path = find_radonpy_file()
    raw_df = pd.read_csv(radon_path, index_col=0, low_memory=False).copy()
    raw_df['source_row_id'] = raw_df.index.astype(str)
    smiles_col = 'smiles_list_canonical' if 'smiles_list_canonical' in raw_df.columns else ('smiles' if 'smiles' in raw_df.columns else None)
    if smiles_col is None: raise KeyError('Cannot find RadonPy smiles column.')
    raw_df['smiles'] = raw_df[smiles_col].astype(str).str.strip()
    df = raw_df[raw_df['smiles'].notna() & ~raw_df['smiles'].str.lower().isin(['nan', 'none', ''])].copy().reset_index(drop=True)
    y_cols, rows = [], []
    for meta in RADONPY_TARGET_REGISTRY:
        target = meta['target']; source_col = _resolve_radon_target_column(df, target)
        raw = pd.to_numeric(df[source_col], errors='coerce').to_numpy(np.float32) if source_col is not None else np.full(len(df), np.nan, dtype=np.float32)
        y = apply_radonpy_transform(raw, meta['transform'])
        y_cols.append(y)
        rows.append({'target': target, 'source_column': source_col or '', 'column_status': 'present' if source_col else 'missing_column', 'category': meta['category'], 'LSER': bool(meta['lser']), 'recommended_transform': meta['transform'], 'n_raw_finite': int(np.isfinite(raw).sum()), 'n_transformed_finite': int(np.isfinite(y).sum()), 'iqr_outlier_count_after_transform': _iqr_outlier_count(y)})
    targets = [r['target'] for r in rows]; transforms = [r['recommended_transform'] for r in rows]
    Y_trans = np.stack(y_cols, axis=1).astype(np.float32); mask = np.isfinite(Y_trans)
    idx = np.arange(len(df), dtype=np.int64); rng = np.random.default_rng(seed); rng.shuffle(idx)
    n_full_train = int((1.0 - RADONPY_VALID_FRACTION) * len(idx)); full_train_idx = idx[:n_full_train]; valid_idx = idx[n_full_train:]
    n_use = max(1, min(len(full_train_idx), int(round(float(fraction) * len(full_train_idx))))); train_idx = full_train_idx[:n_use]
    mean = np.nanmean(Y_trans[train_idx], axis=0).astype(np.float32); std = np.nanstd(Y_trans[train_idx], axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0.0; std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    Y_std = (Y_trans - mean) / std; Y_std[~np.isfinite(Y_std)] = 0.0
    needed_idx = np.concatenate([train_idx, valid_idx]); graph_cache.build(df.iloc[needed_idx]['smiles'].values, save=True)
    train_ds = RadonPyDataset(df.iloc[train_idx], Y_std[train_idx], mask[train_idx], graph_cache)
    valid_ds = RadonPyDataset(df.iloc[valid_idx], Y_std[valid_idx], mask[valid_idx], graph_cache)
    gen = torch.Generator(); gen.manual_seed(seed + 100_000)
    train_loader = DataLoader(train_ds, batch_size=CFG['radon_batch_size'], shuffle=True, collate_fn=collate_radon, num_workers=CFG['num_workers'], generator=gen)
    valid_loader = DataLoader(valid_ds, batch_size=CFG['radon_batch_size'], shuffle=False, collate_fn=collate_radon, num_workers=CFG['num_workers'])
    stats_df = pd.DataFrame(rows); stats_df['mean_after_transform_selected_train_only'] = mean; stats_df['std_after_transform_selected_train_only'] = std; stats_df['n_labels_selected_train'] = mask[train_idx].sum(axis=0).astype(int); stats_df['n_labels_valid'] = mask[valid_idx].sum(axis=0).astype(int); stats_df['auxiliary_loss_active_current_data'] = stats_df['n_labels_selected_train'] > 0
    stats_df.to_csv(out_dir / 'radonpy_target_stats.csv', index=False); stats_df.to_csv(out_dir / 'radonpy_property_transform_outlier_report.csv', index=False)
    role = np.full(len(df), 'train_unused', dtype=object); role[valid_idx] = 'valid'; role[train_idx] = 'train_used'
    pd.DataFrame({'radon_row_position': np.arange(len(df)), 'source_row_id': df['source_row_id'].astype(str), 'smiles': df['smiles'].astype(str), 'role': role}).to_csv(out_dir / 'radonpy_row_split.csv', index=False)
    cfg = {'enabled': True, 'target_policy': 'CORR_PROPS=all; LSER marker only; masked loss skips missing labels', 'registered_targets': RADONPY_RECOMMENDED_TARGETS, 'used_targets': targets, 'transforms': dict(zip(targets, transforms)), 'n_rows_total': int(len(df)), 'n_train_used': int(len(train_idx)), 'n_valid_10pct': int(len(valid_idx)), 'radon_path': str(radon_path)}
    with open(out_dir / 'radonpy_target_config.json', 'w', encoding='utf-8') as f: json.dump(cfg, f, ensure_ascii=False, indent=2)
    report_columns = [
        'target',
        'recommended_transform',
        'n_transformed_finite',
        'n_labels_selected_train',
        'auxiliary_loss_active_current_data',
    ]
    print(stats_df[report_columns].to_string(index=False))
    return {'train_loader': train_loader, 'valid_loader': valid_loader, 'targets': targets, 'transforms': transforms, 'target_mean': mean, 'target_std': std, 'n_train_used': int(len(train_idx)), 'n_train_full': int(len(full_train_idx)), 'n_valid': int(len(valid_idx)), 'radon_path': str(radon_path)}

# -------------------------
# Metrics override; validation-only selection still needs valid predictions.
# -------------------------
def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2: return np.nan
    r1 = pd.Series(y_true[valid]).rank(method='average').to_numpy(dtype=float); r2 = pd.Series(y_pred[valid]).rank(method='average').to_numpy(dtype=float)
    if np.std(r1) < 1e-12 or np.std(r2) < 1e-12: return np.nan
    return float(np.corrcoef(r1, r2)[0, 1])

def compute_metrics_from_arrays(true_min, pred_min):
    true_min = np.asarray(true_min, dtype=float); pred_min = np.asarray(pred_min, dtype=float); err_sec = (pred_min - true_min) * 60.0
    mape, mape_n = safe_mape_pct(true_min, pred_min)
    return {'mae_sec': float(np.mean(np.abs(err_sec))) if len(err_sec) else np.nan, 'median_ae_sec': float(np.median(np.abs(err_sec))) if len(err_sec) else np.nan, 'rmse_sec': float(np.sqrt(np.mean(err_sec ** 2))) if len(err_sec) else np.nan, 'mape_pct': mape, 'mape_n_rows': mape_n, 'r2': safe_r2(true_min, pred_min), 'spearman': safe_spearman(true_min, pred_min), 'n_rows': int(len(true_min))}

@torch.no_grad()
def evaluate_rt(model, loader, split_name='valid', save_predictions=False):
    model.eval(); y_true_std, y_pred_std, rows = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch); pred, _ = model.forward_rt(batch)
        y_true_std.append(batch['y'].detach().cpu().numpy()); y_pred_std.append(pred.detach().cpu().numpy())
        if save_predictions:
            true_min = unscale_rt(batch['y'].detach().cpu().numpy()); pred_min = unscale_rt(pred.detach().cpu().numpy())
            for i in range(len(true_min)):
                dataset_id = str(batch['dir'][i]).zfill(4); denom = abs(float(true_min[i])); ape = abs(float(pred_min[i] - true_min[i])) / denom * 100.0 if denom > 1e-8 else np.nan; abs_err_sec = float(abs(pred_min[i] - true_min[i]) * 60.0)
                rows.append({'split': split_name, 'dataset_id': dataset_id, 'dir': dataset_id, 'row_id': batch['row_id'][i], 'smiles': batch['smiles'][i], 'mol_key': batch['mol_key'][i], 'clean_radonpy_overlap': bool(batch['clean_radonpy_overlap'][i].detach().cpu().item()), 'non_radonpy_overlap': bool(batch['non_radonpy_overlap'][i].detach().cpu().item()), 'true_rt': float(true_min[i]), 'pred_rt': float(pred_min[i]), 'true_min': float(true_min[i]), 'pred_min': float(pred_min[i]), 'true_sec': float(true_min[i] * 60.0), 'pred_sec': float(pred_min[i] * 60.0), 'absolute_error_sec': abs_err_sec, 'abs_error_sec': abs_err_sec, 'APE_pct': float(ape) if np.isfinite(ape) else np.nan, 'ape_pct': float(ape) if np.isfinite(ape) else np.nan})
    if len(y_true_std) == 0: return compute_metrics_from_arrays([], []), (pd.DataFrame(rows) if save_predictions else None)
    true_min = unscale_rt(np.concatenate(y_true_std)); pred_min = unscale_rt(np.concatenate(y_pred_std)); metrics = compute_metrics_from_arrays(true_min, pred_min)
    pred_df = pd.DataFrame(rows) if save_predictions else None
    if save_predictions and pred_df is not None and len(pred_df):
        for label, mask_arr in {'clean_overlap': pred_df['clean_radonpy_overlap'].values, 'nonoverlap': pred_df['non_radonpy_overlap'].values}.items():
            if mask_arr.sum() > 0:
                sub = pred_df.loc[mask_arr]; sub_m = compute_metrics_from_arrays(sub['true_min'].to_numpy(float), sub['pred_min'].to_numpy(float)); metrics[f'{label}_mae_sec'] = sub_m['mae_sec']; metrics[f'{label}_mape_pct'] = sub_m['mape_pct']; metrics[f'{label}_r2'] = sub_m['r2']; metrics[f'{label}_spearman'] = sub_m['spearman']; metrics[f'{label}_n_rows'] = int(mask_arr.sum())
            else:
                metrics[f'{label}_mae_sec'] = np.nan; metrics[f'{label}_mape_pct'] = np.nan; metrics[f'{label}_r2'] = np.nan; metrics[f'{label}_spearman'] = np.nan; metrics[f'{label}_n_rows'] = 0
    return metrics, pred_df

# -------------------------
# Joint multitask epoch for M2
# -------------------------
def train_one_joint_epoch(model, optimizer, rt_loader, aux_loader, aux_weight: float = 1.0):
    model.train(); total_losses, rt_losses, aux_losses = [], [], []; rt_batches = aux_batches = rt_rows = aux_rows = 0; aux_iter = iter(aux_loader) if aux_loader is not None else None
    for batch in tqdm(rt_loader, desc='Joint RT+RadonPy train', leave=False):
        batch = move_batch_to_device(batch); optimizer.zero_grad(set_to_none=True); pred, _ = model.forward_rt(batch); rt_loss = rt_loss_fn(pred.float(), batch['y'].float()); aux_loss = pred.sum() * 0.0; used_aux = False
        if aux_iter is not None and model.radon_heads is not None:
            try: rb = next(aux_iter)
            except StopIteration: aux_iter = iter(aux_loader); rb = next(aux_iter)
            aux_pred = model.forward_radon(rb['graph'].to(DEVICE)); aux_loss = masked_radon_loss(aux_pred, rb['y'].to(DEVICE), rb['mask'].to(DEVICE)); used_aux = True
        loss = rt_loss + float(aux_weight) * aux_loss; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip']); optimizer.step()
        total_losses.append(float(loss.detach().cpu())); rt_losses.append(float(rt_loss.detach().cpu())); rt_batches += 1; rt_rows += int(batch['y'].shape[0])
        if used_aux: aux_losses.append(float(aux_loss.detach().cpu())); aux_batches += 1; aux_rows += int(rb['mask'].shape[0])
    return {'total_loss': float(np.mean(total_losses)) if total_losses else np.nan, 'rt_loss': float(np.mean(rt_losses)) if rt_losses else np.nan, 'aux_loss': float(np.mean(aux_losses)) if aux_losses else np.nan, 'rt_batches': int(rt_batches), 'aux_batches': int(aux_batches), 'rt_rows': int(rt_rows), 'aux_rows': int(aux_rows), 'aux_loss_weight': float(aux_weight)}

# -------------------------
# Split loading / vocab saving
# -------------------------
def normalize_split_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if 'smiles' not in df.columns:
        if 'smiles_raw' in df.columns: df['smiles'] = df['smiles_raw'].astype(str)
        elif 'SMILES' in df.columns: df['smiles'] = df['SMILES'].astype(str)
        elif 'smiles_selected' in df.columns: df['smiles'] = df['smiles_selected'].astype(str)
        else: raise KeyError('Cannot find smiles/smiles_raw/SMILES column.')
    df['smiles'] = df['smiles'].astype(str).str.strip()
    if 'dir' not in df.columns:
        if 'dataset_id' in df.columns: df['dir'] = df['dataset_id']
        else: raise KeyError('Cannot find dir or dataset_id column.')
    df['dir'] = df['dir'].astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    if 'dataset_id' not in df.columns: df['dataset_id'] = df['dir']
    if 'row_id' not in df.columns: df['row_id'] = [f'{split_name}_{i}' for i in range(len(df))]
    if 'mol_key' not in df.columns:
        df['mol_key'] = df['mol_key_exact'].astype(str) if 'mol_key_exact' in df.columns else df['smiles'].map(smiles_to_mol_key).astype(str)
    if 'rt' not in df.columns:
        rt_cols = [c for c in df.columns if c.lower() in {'rt', 'rt_min', 'retention_time'} or 'rt' in c.lower()]
        if not rt_cols: raise KeyError('Cannot infer RT column.')
        df['rt'] = pd.to_numeric(df[rt_cols[0]], errors='coerce')
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
    return df.dropna(subset=['rt', 'smiles', 'mol_key']).copy().reset_index(drop=True)

def resolve_split_dir(seed: int) -> Path:
    roots = [SPLIT_ROOT]
    checked = []
    for root in roots:
        p = Path(root) / f'seed_{seed}'; checked.append(str(p))
        if all((p / fn).exists() for fn in ['train.csv', 'valid.csv', 'internal_test.csv', 'external_ood.csv']): return p
    raise FileNotFoundError('Cannot resolve split directory. Checked:\n' + '\n'.join(checked))

def _write_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def prepare_rt_split(seed: int, run_dir: Path, graph_cache):
    global y_mean, y_std, cat_vocabs, method_vocab
    split_dir = resolve_split_dir(seed)
    train_df_raw = normalize_split_df(pd.read_csv(split_dir / 'train.csv'), 'train'); valid_df_raw = normalize_split_df(pd.read_csv(split_dir / 'valid.csv'), 'valid'); test_df_raw = normalize_split_df(pd.read_csv(split_dir / 'internal_test.csv'), 'internal_test'); external_df_raw = normalize_split_df(pd.read_csv(split_dir / 'external_ood.csv'), 'external_ood')
    train_df_raw = attach_env_features(train_df_raw); valid_df_raw = attach_env_features(valid_df_raw); test_df_raw = attach_env_features(test_df_raw); external_df_raw = attach_env_features(external_df_raw)
    y_mean = float(train_df_raw['rt'].mean()); y_std = float(train_df_raw['rt'].std(ddof=0)); y_std = y_std if np.isfinite(y_std) and y_std >= 1e-8 else 1.0
    cat_vocabs = {'column_cat0': collect_vocab(train_df_raw, 'column_cat', 0), 'brand_cat0': collect_vocab(train_df_raw, 'brand_cat', 0), 'solvent_cat0': collect_vocab(train_df_raw, 'solvent_cat', 0), 'solvent_cat1': collect_vocab(train_df_raw, 'solvent_cat', 1)}
    train_methods = sorted(train_df_raw['dir'].astype(str).unique().tolist()); method_vocab = {m: i for i, m in enumerate(train_methods)}
    radon_keys = load_radon_keys_for_flags(); train_keys = set(train_df_raw['mol_key'].astype(str))
    def add_flags(df):
        df = df.copy(); df['in_radonpy'] = df['mol_key'].astype(str).isin(radon_keys) if radon_keys else False; df['appears_in_train'] = df['mol_key'].astype(str).isin(train_keys); df['clean_radonpy_overlap'] = df['in_radonpy'] & (~df['appears_in_train']); df['non_radonpy_overlap'] = ~df['in_radonpy']; return df
    train_df_raw = add_flags(train_df_raw); valid_df_raw = add_flags(valid_df_raw); test_df_raw = add_flags(test_df_raw); external_df_raw = add_flags(external_df_raw)
    tr_m, va_m, te_m = set(train_df_raw['mol_key'].astype(str)), set(valid_df_raw['mol_key'].astype(str)), set(test_df_raw['mol_key'].astype(str)); internal_dirs = set(train_df_raw['dir']) | set(valid_df_raw['dir']) | set(test_df_raw['dir']); ood_dirs = set(external_df_raw['dir'])
    leakage = pd.DataFrame([{'check':'train_valid_mol_overlap','value':len(tr_m & va_m)}, {'check':'train_test_mol_overlap','value':len(tr_m & te_m)}, {'check':'valid_test_mol_overlap','value':len(va_m & te_m)}, {'check':'ood_internal_method_overlap','value':len(ood_dirs & internal_dirs)}, {'check':'vocabulary_uses_test_or_ood','value':0}])
    if (leakage['value'] != 0).any(): raise RuntimeError(f'Leakage audit failed for seed={seed}:\n{leakage}')
    if len(method_vocab) != 179: raise RuntimeError(f'Expected 179 train methods, found {len(method_vocab)} for seed={seed}.')
    stage0 = run_dir / 'stage0_data'; stage0.mkdir(parents=True, exist_ok=True); leakage.to_csv(stage0 / 'leakage_audit.csv', index=False)
    pd.DataFrame([{'split':'train','rows':len(train_df_raw),'methods':train_df_raw['dir'].nunique(),'mol_keys':train_df_raw['mol_key'].nunique()}, {'split':'valid','rows':len(valid_df_raw),'methods':valid_df_raw['dir'].nunique(),'mol_keys':valid_df_raw['mol_key'].nunique()}, {'split':'internal_test','rows':len(test_df_raw),'methods':test_df_raw['dir'].nunique(),'mol_keys':test_df_raw['mol_key'].nunique()}, {'split':'external_ood','rows':len(external_df_raw),'methods':external_df_raw['dir'].nunique(),'mol_keys':external_df_raw['mol_key'].nunique()}]).to_csv(stage0 / 'split_summary.csv', index=False)
    _write_json(cat_vocabs, stage0 / 'cat_vocabs.json'); _write_json(cat_vocabs.get('brand_cat0', {}), stage0 / 'brand_vocab.json'); _write_json(method_vocab, stage0 / 'method_vocab.json')
    _write_json({'EXP_ID': EXP_ID, 'EXP_NAME': EXP_NAME, 'MOLECULE_MODE': MOLECULE_MODE, 'RT_ARCHITECTURE': RT_ARCHITECTURE, 'USE_DEVICE_METADATA': USE_DEVICE_METADATA, 'RT_HEAD_TYPE': RT_HEAD_TYPE, 'JOINT_MULTITASK': JOINT_MULTITASK, 'WEIGHTS_ONLY': True, 'CFG': CFG, 'split_dir': str(split_dir), 'y_mean': y_mean, 'y_std': y_std}, stage0 / 'experiment_config.json')
    all_smiles = pd.concat([train_df_raw['smiles'], valid_df_raw['smiles'], test_df_raw['smiles'], external_df_raw['smiles']], ignore_index=True); graph_cache.build(all_smiles, save=True)
    train_ds = RTGraphDataset(train_df_raw, graph_cache); valid_ds = RTGraphDataset(valid_df_raw, graph_cache); test_ds = RTGraphDataset(test_df_raw, graph_cache); external_ds = RTGraphDataset(external_df_raw, graph_cache)
    gen = torch.Generator(); gen.manual_seed(seed + 200_000)
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, collate_fn=collate_rt, num_workers=CFG['num_workers'], generator=gen); valid_loader = DataLoader(valid_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_rt, num_workers=CFG['num_workers']); test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_rt, num_workers=CFG['num_workers']); external_loader = DataLoader(external_ds, batch_size=CFG['batch_size'], shuffle=False, collate_fn=collate_rt, num_workers=CFG['num_workers'])
    tmp = next(iter(train_loader)); assert tmp['graph'].x_cat.size(0) == tmp['graph'].batch.size(0)
    return {'split_dir': split_dir, 'train_df': train_df_raw, 'valid_df': valid_df_raw, 'test_df': test_df_raw, 'external_df': external_df_raw, 'train_ds': train_ds, 'valid_ds': valid_ds, 'test_ds': test_ds, 'external_ds': external_ds, 'train_loader': train_loader, 'valid_loader': valid_loader, 'test_loader': test_loader, 'external_loader': external_loader}

# -------------------------
# Weight-only run: train and save checkpoints; skip internal-test/OOD final evaluation.
# -------------------------
def _build_ckpt_payload(model, epoch, best_score, seed, data, radon):
    return {'model': model.state_dict(), 'cfg': CFG, 'epoch': int(epoch), 'best_score': float(best_score) if np.isfinite(best_score) else np.inf, 'seed': int(seed), 'split_seed': int(seed), 'EXP_ID': EXP_ID, 'EXP_NAME': EXP_NAME, 'MOLECULE_MODE': MOLECULE_MODE, 'RT_ARCHITECTURE': RT_ARCHITECTURE, 'USE_DEVICE_METADATA': USE_DEVICE_METADATA, 'rt_head_type': RT_HEAD_TYPE, 'JOINT_MULTITASK': JOINT_MULTITASK, 'WEIGHTS_ONLY': True, 'radonpy_fraction': RADONPY_FRACTION, 'radonpy_percent': RADONPY_PERCENT, 'radonpy_rows_used': radon.get('n_train_used', 0), 'radonpy_rows_full_train': radon.get('n_train_full', 0), 'y_mean': y_mean, 'y_std': y_std, 'radon_targets': radon_targets, 'radon_target_transforms': radon_target_transforms, 'radon_target_mean': radon_target_mean, 'radon_target_std': radon_target_std, 'method_vocab': method_vocab, 'cat_vocabs': cat_vocabs, 'training_mode': MOLECULE_MODE, 'split_dir': str(data['split_dir'])}

def run_one_seed(seed: int, graph_cache):
    global radon_train_loader, radon_valid_loader, radon_targets, radon_target_transforms, radon_target_mean, radon_target_std
    set_all_seeds(seed); run_dir = FRACTION_OUT / f'seed_{seed}'; run_dir.mkdir(parents=True, exist_ok=True); complete_path = run_dir / 'run_complete.json'; summary_path = run_dir / 'run_summary.csv'; best_path = run_dir / BEST_CKPT_NAME
    if RESUME_COMPLETED and complete_path.exists() and summary_path.exists() and best_path.exists(): print(f'[SKIP complete weights] {EXP_ID} seed={seed}'); return pd.read_csv(summary_path).iloc[0].to_dict()
    start_time = time.time(); print('\n' + '='*100); print(f'WEIGHT-ONLY RUN {EXP_ID}: {EXP_NAME} | seed={seed}'); print('='*100)
    data = prepare_rt_split(seed, run_dir, graph_cache); radon = prepare_radonpy_loaders(RADONPY_FRACTION, seed, graph_cache, run_dir); radon_train_loader = radon['train_loader']; radon_valid_loader = radon['valid_loader']; radon_targets = radon['targets']; radon_target_transforms = radon['transforms']; radon_target_mean = radon['target_mean']; radon_target_std = radon['target_std']
    model = GraphEnvRTModel(cat_vocabs=cat_vocabs, radon_targets=len(radon_targets), cfg=CFG, head_type=RT_HEAD_TYPE, num_methods=len(method_vocab), use_device_metadata=USE_DEVICE_METADATA).to(DEVICE); print('Total parameters:', sum(p.numel() for p in model.parameters()))
    if RADONPY_PRETRAIN_EPOCHS > 0:
        if radon_train_loader is None: raise RuntimeError('M1 pretraining requested, but RadonPy loader is disabled.')
        opt_pre = make_optimizer(model); pre_rows = []
        for ep in range(1, RADONPY_PRETRAIN_EPOCHS + 1):
            tr_loss = train_one_radon_epoch(model, opt_pre, radon_train_loader, desc=f'RadonPy pretrain {EXP_ID} seed {seed} {ep}/{RADONPY_PRETRAIN_EPOCHS}'); va_loss = evaluate_radon(model, radon_valid_loader); pre_rows.append({'epoch': ep, 'radon_train_loss': tr_loss, 'radon_valid_loss': va_loss}); print(f'[Pretrain {ep:03d}] train={tr_loss:.5f} valid={va_loss:.5f}')
        pd.DataFrame(pre_rows).to_csv(run_dir / 'radonpy_pretrain_log.csv', index=False); pretrain_payload = {'model': model.state_dict(), 'epoch': RADONPY_PRETRAIN_EPOCHS, 'seed': seed, 'EXP_ID': EXP_ID, 'radon_targets': radon_targets, 'radon_target_transforms': radon_target_transforms, 'radon_target_mean': radon_target_mean, 'radon_target_std': radon_target_std, 'WEIGHTS_ONLY': True}; torch.save(pretrain_payload, run_dir / 'pretrain_final.pt'); torch.save(pretrain_payload, run_dir / 'radonpy_pretrain_final.pt')
    optimizer = make_optimizer(model); scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CFG['scheduler_factor'], patience=CFG['scheduler_patience'], min_lr=CFG['min_lr']); best_score = np.inf; best_epoch = 0; no_improve = 0; log_rows = []
    for epoch in range(1, RT_EPOCHS + 1):
        if JOINT_MULTITASK:
            train_stats = train_one_joint_epoch(model, optimizer, data['train_loader'], radon_train_loader, aux_weight=CFG.get('aux_loss_weight', 1.0)); train_loss = train_stats['total_loss']
        else:
            train_loss = train_one_rt_epoch(model, optimizer, data['train_loader']); train_stats = {'total_loss': train_loss, 'rt_loss': train_loss, 'aux_loss': np.nan, 'rt_batches': np.nan, 'aux_batches': 0, 'rt_rows': np.nan, 'aux_rows': 0, 'aux_loss_weight': 0.0}
        aux_valid_loss = evaluate_radon(model, radon_valid_loader) if (radon_valid_loader is not None and model.radon_heads is not None) else np.nan; valid_metrics, valid_pred = evaluate_rt(model, data['valid_loader'], 'valid', save_predictions=True); score = valid_score_from_metrics(valid_metrics); scheduler.step(score); improved = score < best_score - 1e-6
        if improved:
            best_score = score; best_epoch = epoch; no_improve = 0; ckpt_payload = _build_ckpt_payload(model, epoch, best_score, seed, data, radon); torch.save(ckpt_payload, best_path); torch.save(ckpt_payload, run_dir / 'best_model.pt'); valid_pred.to_csv(run_dir / 'best_valid_predictions.csv', index=False)
        else: no_improve += 1
        row = {'epoch': epoch, 'train_loss': train_loss, 'rt_train_loss': train_stats.get('rt_loss', train_loss), 'aux_train_loss': train_stats.get('aux_loss', np.nan), 'aux_valid_loss': aux_valid_loss, 'rt_batches': train_stats.get('rt_batches', np.nan), 'aux_batches': train_stats.get('aux_batches', np.nan), 'rt_rows_exposed': train_stats.get('rt_rows', np.nan), 'aux_rows_exposed': train_stats.get('aux_rows', np.nan), 'aux_loss_weight': train_stats.get('aux_loss_weight', 0.0), 'valid_score': score, 'best_score': best_score, 'best_epoch': best_epoch, 'no_improve': no_improve, 'lr': float(optimizer.param_groups[0]['lr']), **{f'valid_{k}': v for k, v in valid_metrics.items()}}
        log_rows.append(row); pd.DataFrame(log_rows).to_csv(run_dir / 'training_log.csv', index=False); print(f"[RT {epoch:03d}/{RT_EPOCHS}] total={train_loss:.5f} rt={row['rt_train_loss']:.5f} aux={row['aux_train_loss'] if np.isfinite(row['aux_train_loss']) else np.nan} valid_MAE={valid_metrics['mae_sec']:.2f}s score={score:.2f} best={best_score:.2f}@{best_epoch}")
        if no_improve >= CFG['early_stop_patience']: print('Early stopping triggered.'); break
    if not best_path.exists():
        best_epoch = int(epoch); best_score = float(score) if 'score' in locals() and np.isfinite(score) else np.inf; ckpt_payload = _build_ckpt_payload(model, best_epoch, best_score, seed, data, radon); torch.save(ckpt_payload, best_path); torch.save(ckpt_payload, run_dir / 'best_model.pt')
    final_payload = _build_ckpt_payload(model, epoch, best_score, seed, data, radon); final_payload['checkpoint_type'] = 'final_model'; torch.save(final_payload, run_dir / 'final_model.pt')
    elapsed_min = (time.time() - start_time) / 60.0; summary = {'EXP_ID': EXP_ID, 'EXP_NAME': EXP_NAME, 'MOLECULE_MODE': MOLECULE_MODE, 'RT_ARCHITECTURE': RT_ARCHITECTURE, 'use_device_metadata': USE_DEVICE_METADATA, 'rt_head_type': RT_HEAD_TYPE, 'joint_multitask': JOINT_MULTITASK, 'weight_only': True, 'ood_evaluation': False, 'internal_test_evaluation': False, 'radonpy_percent': RADONPY_PERCENT, 'radonpy_fraction': RADONPY_FRACTION, 'seed': seed, 'split_dir': str(data['split_dir']), 'best_epoch': int(best_epoch), 'best_valid_score': float(best_score) if np.isfinite(best_score) else np.inf, 'radonpy_rows_used': int(radon['n_train_used']), 'radonpy_rows_full_train': int(radon['n_train_full']), 'radonpy_valid_rows': int(radon['n_valid']), 'elapsed_min': elapsed_min, 'best_checkpoint': str(best_path), 'best_model_alias': str(run_dir / 'best_model.pt'), 'final_checkpoint': str(run_dir / 'final_model.pt'), 'training_log': str(run_dir / 'training_log.csv'), 'best_valid_predictions': str(run_dir / 'best_valid_predictions.csv')}
    pd.DataFrame([summary]).to_csv(run_dir / 'run_summary.csv', index=False); manifest = {**summary, 'cfg': CFG, 'split_seeds': SPLIT_SEEDS, 'radon_targets': radon_targets, 'radon_target_transforms': dict(zip(radon_targets, radon_target_transforms)), 'radonpy_row_split': '90/10 by row; nested training fraction prefix', 'brand_vocab_policy': 'training split only; unseen valid/test brand maps to <UNK>'}
    with open(run_dir / 'weights_manifest.json', 'w', encoding='utf-8') as f: json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(run_dir / 'run_manifest.json', 'w', encoding='utf-8') as f: json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(complete_path, 'w', encoding='utf-8') as f: json.dump({'complete': True, **summary}, f, ensure_ascii=False, indent=2)
    del model, optimizer, scheduler, data; gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summary


def configure_experiment(experiment_key: str) -> dict:
    """Apply one E1-E9 or E8 scaling configuration to the shared engine."""
    global EXP_KEY, EXP_ID, MOLECULE_MODE, RT_ARCHITECTURE, EXP_NAME
    global USE_DEVICE_METADATA, RT_HEAD_TYPE, JOINT_MULTITASK
    global RADONPY_PERCENT, RADONPY_FRACTION, RADONPY_PRETRAIN_EPOCHS
    global RT_EPOCHS, RADONPY_VALID_FRACTION, SPLIT_SEEDS
    global RESUME_COMPLETED, FAIL_FAST, BEST_CKPT_NAME
    global ROOT_OUT, FRACTION_OUT, OUT_DIR, df_meta
    global ENV_FEATURE_CACHE, RADON_KEY_CACHE

    try:
        experiment = dict(_CONFIG["experiments"][experiment_key])
    except KeyError as exc:
        choices = ", ".join(sorted(_CONFIG["experiments"]))
        raise KeyError(f"Unknown experiment {experiment_key!r}. Choose one of: {choices}") from exc

    common = _CONFIG["common"]
    EXP_KEY = experiment_key
    EXP_ID = experiment["exp_id"]
    MOLECULE_MODE = experiment["molecule_mode"]
    RT_ARCHITECTURE = experiment["rt_architecture"]
    EXP_NAME = experiment["name"]
    USE_DEVICE_METADATA = bool(experiment["use_device_metadata"])
    RT_HEAD_TYPE = experiment["rt_head_type"]
    JOINT_MULTITASK = bool(experiment["joint_multitask"])
    RADONPY_PERCENT = float(experiment["radonpy_percent"])
    RADONPY_FRACTION = RADONPY_PERCENT / 100.0
    RADONPY_PRETRAIN_EPOCHS = int(experiment["radonpy_pretrain_epochs"])
    RT_EPOCHS = int(common["rt_epochs"])
    RADONPY_VALID_FRACTION = float(common["radonpy_valid_fraction"])
    SPLIT_SEEDS = [int(seed) for seed in common["seeds"]]
    RESUME_COMPLETED = bool(common["resume_completed"])
    FAIL_FAST = bool(common["fail_fast"])
    BEST_CKPT_NAME = "best_joint.pt" if JOINT_MULTITASK else "best_rt.pt"

    ROOT_OUT = PROJECT_ROOT / "checkpoints" / experiment["output_group"] / experiment["output_name"]
    FRACTION_OUT = ROOT_OUT
    OUT_DIR = FRACTION_OUT / "_shared"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ENV_FEATURE_CACHE = {}
    RADON_KEY_CACHE = None
    df_meta = read_report_global_metadata(META_PATH) if META_PATH.exists() else pd.DataFrame()
    return experiment


def _numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    row = {
        "experiment_key": EXP_KEY,
        "EXP_ID": EXP_ID,
        "EXP_NAME": EXP_NAME,
        "n_completed_seeds": int(frame["seed"].nunique()) if "seed" in frame else int(len(frame)),
    }
    for column in ("best_valid_score", "best_epoch", "elapsed_min"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
        row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return pd.DataFrame([row])


def run_training(experiment_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train all configured seeds and write the two curated result tables."""
    configure_experiment(experiment_key)
    shared_cache = PrecomputedPyGGraphCache(OUT_DIR / "pyg_graph_cache.pt")
    completed, failed = [], []
    for split_seed in SPLIT_SEEDS:
        try:
            completed.append(run_one_seed(split_seed, shared_cache))
        except Exception as exc:
            error = traceback.format_exc()
            failed.append({"EXP_ID": EXP_ID, "seed": split_seed, "error_type": type(exc).__name__, "error": str(exc)})
            fail_dir = FRACTION_OUT / f"seed_{split_seed}"
            fail_dir.mkdir(parents=True, exist_ok=True)
            failure_path = fail_dir / "run_failed.txt"
            failure_path.write_text(error, encoding="utf-8")
            print(
                f"[ERROR] {EXP_ID} seed={split_seed} failed with "
                f"{type(exc).__name__}: {exc}\nTraceback: {failure_path}"
            )
            if FAIL_FAST:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed_df = pd.DataFrame(completed)
    failed_df = pd.DataFrame(failed)
    result_dir = PROJECT_ROOT / "result" / "training" / experiment_key
    result_dir.mkdir(parents=True, exist_ok=True)
    completed_df.to_csv(result_dir / "per_model_seed.csv", index=False)
    _numeric_summary(completed_df).to_csv(result_dir / "summary.csv", index=False)
    if len(failed_df):
        failed_df.to_csv(FRACTION_OUT / "failed_runs.csv", index=False)
    return completed_df, failed_df
