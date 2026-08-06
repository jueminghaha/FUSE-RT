from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CUDA_DEVICE_INDEX = int(__import__("os").environ.get("CUDA_DEVICE_INDEX", "0"))
from model.unirt import *

def display(value):
    print(value)

# ============================================================
# Cell 1 — imports, configuration, GPU-only
# ============================================================


import os
import re
import gc
import json
import math
import time
import random
import hashlib
import warnings
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score





warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 240)
pd.set_option("display.max_rows", 240)

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except Exception as e:
    raise ImportError("RDKit is required.") from e

try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINConv, global_add_pool, BatchNorm, GCNConv, GATConv
except Exception as e:
    raise ImportError(
        "torch_geometric is required. Run this notebook in the same environment used for UniRT training."
    ) from e

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. This notebook is GPU-only.")

DEVICE = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WORKDIR = PROJECT_ROOT

# Exact architecture settings used by the reproduction notebook.
CFG = {
    "mode": "RPLC",
    "model_type": "GINKAN",
    "hidden": 64,
    "layers": 4,
    "dropout": 0.02,
    "adapter_reduction": 4,
    "use_cross_stitch": True,
    "task_emb_dim_mode": "num_tasks",
}

RPLC_IDS = [
    "0019", "0052", "0179", "0180", "0234", "0235", "0260",
    "0261", "0264", "0317", "0319", "0321", "0323", "0331",
]

TARGET_OOD_METHODS = ["0391", "0390", "0437", "0420", "0419", "0411"]
K_VALUES = [0, 20, 100, 2000]

# Fixed current 10-seed split protocol. Do not auto-replace it from an older FUSE-RT CSV.
DEFAULT_K_REPEAT_SEEDS = [2004, 2006, 2011, 2012, 2016, 2020, 2022, 2027, 2032, 2034]
AUTO_DETECT_REPEAT_SEEDS_FROM_FUSE_RT = False

# Exact support-index formula used in the FUSE-RT OOD notebook.
MATCH_FUSE_RT_SUPPORT_SPLITS = True
FUSE_RT_REFERENCE_FRACTION = 100
FUSE_RT_REFERENCE_MODEL_SEED = 1000

HEAD_EPOCHS = 200
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4
HEAD_HIDDEN = 64
BATCH_SIZE = 256

PREBUILD_GRAPH_CACHE = True
USE_GRAPH_CACHE = True
USE_EMBEDDING_CACHE = True
RESUME_EXISTING = True
SAVE_QUERY_PREDICTIONS = True

# Optional manual checkpoint override. Leave None for automatic validation-based selection.
MANUAL_CHECKPOINT = Path(os.environ["UNIRT_CHECKPOINT"]) if os.environ.get("UNIRT_CHECKPOINT") else None

OUT_DIR = PROJECT_ROOT / "result" / "_runs" / "unirt_external_ood"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNI_RT_OUTPUT_ROOT_CANDIDATES = [
    Path(os.environ["UNIRT_OUTPUT_ROOT"]) if os.environ.get("UNIRT_OUTPUT_ROOT") else WORKDIR / "outputs/uni_rt_reproduction/RPLC/GINKAN",
    WORKDIR / "outputs/uni_rt_reproduction/RPLC/GINKAN",
    WORKDIR / "../outputs/uni_rt_reproduction/RPLC/GINKAN",
    WORKDIR / "../../outputs/uni_rt_reproduction/RPLC/GINKAN",
]

# Prefer the exact clean OOD rows produced by the FUSE-RT OOD evaluation pipeline.
EXTERNAL_ROWS_CANDIDATES = [
    PROJECT_ROOT.parent / "RepoRT_PolyOmic/outputs/evaluation_E1_E9_OOD_kshot_lowoverlap_shimadzu/ood_rows_master_lowoverlap_shimadzu.csv",
    WORKDIR / "outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/external_methods_clean_molecule_rows.csv",
    WORKDIR / "../outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/external_methods_clean_molecule_rows.csv",
    WORKDIR / "../../outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/external_methods_clean_molecule_rows.csv",
]

FUSE_RT_METRICS_CANDIDATES = [
    PROJECT_ROOT / "result/external_ood/per_model_seed.csv",
    WORKDIR / "outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/new_report_low_overlap_ood_kshot_metrics_all_runs.csv",
    WORKDIR / "../outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/new_report_low_overlap_ood_kshot_metrics_all_runs.csv",
]

# Fallback paths if the clean OOD CSV is unavailable.
REPORT_ROOT_CANDIDATES = [
    PROJECT_ROOT.parent / "outputs/report_laest_overlap_audit/RepoRT_latest",
    WORKDIR / "outputs/report_latest_overlap_audit/RepoRT_latest",
    WORKDIR / "../outputs/report_latest_overlap_audit/RepoRT_latest",
    WORKDIR / "RepoRT_latest",
    WORKDIR / "RepoRT",
]
SPLIT_DIR_CANDIDATES = [
    PROJECT_ROOT.parent / "RepoRT_PolyOmic/outputs/exact_full_inchikey_split_6_2_2",
    WORKDIR / "outputs/global_molkey_disjoint_split_no_forced_radonpy",
    WORKDIR / "../outputs/global_molkey_disjoint_split_no_forced_radonpy",
]

print("DEVICE:", DEVICE)
print("GPU:", torch.cuda.get_device_name(CUDA_DEVICE_INDEX))
print("OUT_DIR:", OUT_DIR)
print("TARGET_OOD_METHODS:", TARGET_OOD_METHODS)
print("K_VALUES:", K_VALUES)

# ============================================================
# Cell 4 — discover and load the best UniRT-GINKAN seed checkpoint
# ============================================================
def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def first_existing(paths, kind=None):
    for p in paths:
        p = Path(p)
        if p.exists():
            if kind == "dir" and not p.is_dir():
                continue
            if kind == "file" and not p.is_file():
                continue
            return p
    return None


def find_validation_column(df):
    candidates = [
        "val_MAE", "valid_MAE", "validation_MAE",
        "val_mae", "valid_mae", "validation_mae",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", c.lower())
        if "val" in key and "mae" in key:
            return c
    return None


def discover_unirt_checkpoints():
    roots = [p for p in UNI_RT_OUTPUT_ROOT_CANDIDATES if Path(p).exists()]
    records = []

    for root in roots:
        for ckpt_path in sorted(Path(root).glob("**/seed_*/best_model.pth")):
            seed_dir = ckpt_path.parent
            hist_path = seed_dir / "history.csv"
            stats_path = seed_dir / "transform_stats.json"

            seed_match = re.search(r"seed_(\d+)", seed_dir.name)
            seed = int(seed_match.group(1)) if seed_match else np.nan

            best_val = np.nan
            best_epoch = np.nan
            val_col = None

            if hist_path.exists():
                try:
                    hist = pd.read_csv(hist_path)
                    val_col = find_validation_column(hist)
                    if val_col is not None:
                        vals = pd.to_numeric(hist[val_col], errors="coerce")
                        if vals.notna().any():
                            idx = vals.idxmin()
                            best_val = float(vals.loc[idx])
                            if "epoch" in hist.columns:
                                best_epoch = int(hist.loc[idx, "epoch"])
                except Exception:
                    pass

            records.append({
                "checkpoint": str(ckpt_path),
                "seed_dir": str(seed_dir),
                "run_dir": str(seed_dir.parent),
                "seed": seed,
                "history": str(hist_path) if hist_path.exists() else "",
                "transform_stats": str(stats_path) if stats_path.exists() else "",
                "best_valid_mae_sec": best_val,
                "best_epoch": best_epoch,
                "checkpoint_mtime": ckpt_path.stat().st_mtime,
            })

    return pd.DataFrame(records)


checkpoint_table = discover_unirt_checkpoints()

if MANUAL_CHECKPOINT is not None:
    selected_ckpt_path = Path(MANUAL_CHECKPOINT)
    if not selected_ckpt_path.exists():
        raise FileNotFoundError(selected_ckpt_path)
    selected_seed_dir = selected_ckpt_path.parent
    selected_seed = int(re.search(r"seed_(\d+)", selected_seed_dir.name).group(1))
else:
    if checkpoint_table.empty:
        raise FileNotFoundError(
            "No UniRT checkpoint found under:\n" +
            "\n".join(map(str, UNI_RT_OUTPUT_ROOT_CANDIDATES))
        )

    # Validation-only selection. If some rows have no validation history, they are placed last.
    ranked = checkpoint_table.copy()
    ranked["_missing_val"] = ranked["best_valid_mae_sec"].isna()
    ranked = ranked.sort_values(
        ["_missing_val", "best_valid_mae_sec", "checkpoint_mtime"],
        ascending=[True, True, False],
    )
    selected_row = ranked.iloc[0]
    selected_ckpt_path = Path(selected_row["checkpoint"])
    selected_seed_dir = selected_ckpt_path.parent
    selected_seed = int(selected_row["seed"])

checkpoint_table.to_csv(OUT_DIR / "unirt_checkpoint_candidates.csv", index=False)

print("=== UniRT checkpoint candidates ===")
display(checkpoint_table.sort_values(["best_valid_mae_sec", "seed"], na_position="last"))

print("\nSELECTED checkpoint:", selected_ckpt_path)
print("SELECTED seed:", selected_seed)

stats_path = selected_seed_dir / "transform_stats.json"
if not stats_path.exists():
    raise FileNotFoundError(f"Missing transform stats: {stats_path}")

with open(stats_path, "r") as f:
    stats_json = json.load(f)

TRANSFORM_STATS = (
    float(stats_json["mean"]),
    float(stats_json["std"]),
    float(stats_json["lambda"]),
)
print("Box-Cox transform stats (mean, std, lambda):", TRANSFORM_STATS)

state_obj = torch_load_compat(selected_ckpt_path, map_location="cpu")
state_dict = state_obj["model"] if isinstance(state_obj, dict) and "model" in state_obj else state_obj

# Infer dimensions from the checkpoint.
num_tasks, task_emb_dim = state_dict["task_embed.weight"].shape
hidden, node_dim_from_ckpt = state_dict["embed.weight"].shape
layer_ids = sorted({
    int(m.group(1))
    for k in state_dict.keys()
    for m in [re.match(r"convs\.(\d+)\.", k)]
    if m
})
n_layers = max(layer_ids) + 1 if layer_ids else CFG["layers"]
adapter_down_out = state_dict["adapters.0.down.weight"].shape[0]
adapter_reduction = int(hidden // adapter_down_out)
use_cross_stitch = "cross_stitch.alpha" in state_dict

if int(node_dim_from_ckpt) != int(node_dim):
    raise ValueError(
        f"Node feature mismatch: checkpoint={node_dim_from_ckpt}, current featurization={node_dim}"
    )

model = GINKANmultiRegressor(
    node_dim=node_dim,
    num_tasks=int(num_tasks),
    task_emb_dim=int(task_emb_dim),
    hidden=int(hidden),
    layers=int(n_layers),
    grid_size=5,
    spline_order=3,
    dropout=float(CFG["dropout"]),
    adapter_reduction=int(adapter_reduction),
    use_cross_stitch=bool(use_cross_stitch),
).to(DEVICE)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    raise RuntimeError("Checkpoint did not load exactly; inspect architecture mismatch.")

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

print("\nLoaded UniRT-GINKAN:")
print("num_tasks:", num_tasks)
print("task_emb_dim:", task_emb_dim)
print("hidden:", hidden)
print("layers:", n_layers)
print("adapter_reduction:", adapter_reduction)
print("use_cross_stitch:", use_cross_stitch)
print("trainable original-model params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("model device:", next(model.parameters()).device)

selection_record = {
    "selected_checkpoint": str(selected_ckpt_path),
    "selected_seed": int(selected_seed),
    "transform_mean": TRANSFORM_STATS[0],
    "transform_std": TRANSFORM_STATS[1],
    "transform_lambda": TRANSFORM_STATS[2],
    "num_tasks": int(num_tasks),
    "task_emb_dim": int(task_emb_dim),
    "hidden": int(hidden),
    "layers": int(n_layers),
}
pd.DataFrame([selection_record]).to_csv(OUT_DIR / "selected_unirt_checkpoint.csv", index=False)



# ============================================================
# Cell 5 — load exactly the same molecule-clean New RepoRT OOD rows
# ============================================================
def normalize_mol_key(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return np.nan
    return s.split("-")[0].upper()


def normalize_method_id(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"\d+", str(x))
    return f"{int(m.group(0)):04d}" if m else str(x)


def find_col_fuzzy(df, candidates):
    norm = lambda x: re.sub(r"[^a-z0-9]+", "", str(x).lower())
    mapping = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in mapping:
            return mapping[norm(cand)]
    for c in df.columns:
        ck = norm(c)
        for cand in candidates:
            q = norm(cand)
            if q and (q in ck or ck in q):
                return c
    return None


def load_same_clean_ood_rows():
    existing = first_existing(EXTERNAL_ROWS_CANDIDATES, kind="file")
    if existing is not None:
        print("Using exact clean OOD rows saved by FUSE-RT notebook:", existing)
        df = pd.read_csv(existing, dtype=str)
        df["source_method_id"] = df["source_method_id"].map(normalize_method_id)
        df = df[df["source_method_id"].isin(TARGET_OOD_METHODS)].copy()
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
        df["mol_key"] = df["mol_key"].map(normalize_mol_key)
        df = df.dropna(subset=["smiles", "rt", "mol_key"])
        df = df[df["rt"] > 0].copy()
        return df.reset_index(drop=True), str(existing)

    print("Exact clean OOD CSV not found. Rebuilding from RepoRT as fallback.")
    report_root = first_existing(REPORT_ROOT_CANDIDATES, kind="dir")
    split_dir = first_existing(SPLIT_DIR_CANDIDATES, kind="dir")
    if report_root is None or split_dir is None:
        raise FileNotFoundError(
            "Need either external_methods_clean_molecule_rows.csv, or both RepoRT clone and original split."
        )

    split_frames = []
    for fn in ["train.csv", "valid.csv", "internal_test.csv"]:
        p = split_dir / fn
        one = pd.read_csv(p, dtype=str)
        split_frames.append(one)
    old = pd.concat(split_frames, ignore_index=True)
    old_mol_col = find_col_fuzzy(old, ["mol_key", "inchikey", "inchikey.std"])
    if old_mol_col is None:
        raise ValueError("Cannot find mol_key in original split.")
    old_keys = set(old[old_mol_col].map(normalize_mol_key).dropna())

    processed = report_root / "processed_data"
    frames = []
    known_columns = {
        "0391": "Waters ACQUITY UPLC BEH C18",
        "0390": "Waters ACQUITY UPLC BEH C18",
        "0437": "Agilent ZORBAX 300 SB-C18",
        "0420": "Waters SunFire C18",
        "0419": "Waters SunFire C18",
        "0411": "Thermo Scientific Hypersil GOLD aQ",
    }

    for mid in TARGET_OOD_METHODS:
        method_dir = processed / mid
        hits = sorted(method_dir.glob("*rtdata*canonical*success*.tsv"))
        if not hits:
            hits = sorted(method_dir.glob("*rtdata*.tsv"))
        if not hits:
            raise FileNotFoundError(f"No rtdata for {mid}")
        raw = pd.read_csv(hits[0], sep="\t", dtype=str)
        smi_col = find_col_fuzzy(raw, ["smiles.std", "smiles", "SMILES"])
        key_col = find_col_fuzzy(raw, ["inchikey.std", "inchikey", "mol_key"])
        rt_col = find_col_fuzzy(raw, ["rt", "RT", "rt_min"])
        if smi_col is None or rt_col is None:
            raise ValueError(f"{mid}: required columns missing")

        one = pd.DataFrame({
            "source_method_id": mid,
            "smiles": raw[smi_col].astype(str).str.strip(),
            "rt": pd.to_numeric(raw[rt_col], errors="coerce"),
        })
        if key_col is not None:
            one["mol_key"] = raw[key_col].map(normalize_mol_key)
        else:
            one["mol_key"] = one["smiles"].map(
                lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0]
                if Chem.MolFromSmiles(s) is not None else np.nan
            )
        one = one.dropna(subset=["smiles", "rt", "mol_key"])
        one = one[(one["rt"] > 0) & (~one["mol_key"].isin(old_keys))]
        one = one.drop_duplicates("mol_key", keep="first").reset_index(drop=True)
        one["column.name"] = known_columns.get(mid, "")
        one["row_id"] = [f"{mid}_OOD_{i:06d}" for i in range(len(one))]
        frames.append(one)

    return pd.concat(frames, ignore_index=True), "fallback_rebuilt"


external_df, external_source = load_same_clean_ood_rows()
external_df["source_method_id"] = external_df["source_method_id"].map(normalize_method_id)
external_df["rt_sec"] = pd.to_numeric(external_df["rt"], errors="coerce") * 60.0
external_df = external_df.dropna(subset=["rt_sec", "smiles", "mol_key"])
external_df = external_df[external_df["rt_sec"] > 0].reset_index(drop=True)

# Preserve the existing FUSE-RT OOD row order. This is required to reproduce identical support indices.
external_df["_row_order"] = np.arange(len(external_df))

method_summary = (
    external_df.groupby("source_method_id", sort=False)
    .agg(
        n_rows=("source_method_id", "size"),
        n_mol_keys=("mol_key", "nunique"),
        column_name=("column.name", lambda x: x.dropna().iloc[0] if x.dropna().size else ""),
        rt_sec_mean=("rt_sec", "mean"),
        rt_sec_median=("rt_sec", "median"),
    )
    .reset_index()
)

print("External source:", external_source)
print("External rows:", len(external_df))
print("External unique mol_key:", external_df["mol_key"].nunique())
display(method_summary)

external_df.to_csv(OUT_DIR / "unirt_ood_external_rows_exact.csv", index=False)
method_summary.to_csv(OUT_DIR / "unirt_ood_external_method_summary.csv", index=False)

# Fixed current 10-seed split protocol.
# AUTO_DETECT_REPEAT_SEEDS_FROM_FUSE_RT remains False by default so an older
# result CSV cannot silently replace the requested seeds.
K_REPEAT_SEEDS = [int(s) for s in DEFAULT_K_REPEAT_SEEDS]

if AUTO_DETECT_REPEAT_SEEDS_FROM_FUSE_RT:
    fuse_rt_metrics_path = first_existing(FUSE_RT_METRICS_CANDIDATES, kind="file")
    if fuse_rt_metrics_path is not None:
        fuse_rt_metrics = pd.read_csv(fuse_rt_metrics_path)
        if "repeat_seed" in fuse_rt_metrics.columns:
            detected = sorted(
                pd.to_numeric(fuse_rt_metrics["repeat_seed"], errors="coerce")
                .dropna().astype(int).loc[lambda s: s >= 0].unique().tolist()
            )
            if detected:
                K_REPEAT_SEEDS = detected

EXPECTED_K_REPEAT_SEEDS = [
    2004, 2006, 2011, 2012, 2016,
    2020, 2022, 2027, 2032, 2034,
]

if K_REPEAT_SEEDS != EXPECTED_K_REPEAT_SEEDS:
    raise ValueError(
        "K_REPEAT_SEEDS does not match the current 10-seed protocol. "
        f"Got {K_REPEAT_SEEDS}, expected {EXPECTED_K_REPEAT_SEEDS}."
    )

seed_protocol_df = pd.DataFrame({
    "split_index": np.arange(len(K_REPEAT_SEEDS), dtype=int),
    "repeat_seed": K_REPEAT_SEEDS,
})
seed_protocol_df.to_csv(OUT_DIR / "ood_kshot_seed_protocol.csv", index=False)

print("K_REPEAT_SEEDS:", K_REPEAT_SEEDS)
print("Number of K>0 splits:", len(K_REPEAT_SEEDS))
display(seed_protocol_df)



# ============================================================
# Cell 6 — UniRT graph cache and OOD PyG dataset
# ============================================================
class UniRTGraphCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            try:
                self.cache = torch_load_compat(self.path, map_location="cpu")
                print(f"Loaded graph cache: {len(self.cache):,} molecules")
            except Exception as e:
                print("Could not load old graph cache; rebuilding. Error:", repr(e))
                self.cache = {}

    def get(self, smiles):
        key = str(smiles)
        if key not in self.cache:
            data = smiles_to_data(key, y=None, task_id=None)
            data.num_nodes = int(data.x.size(0))
            # y/task are not needed in the cache.
            if hasattr(data, "y"):
                del data.y
            if hasattr(data, "task"):
                del data.task
            self.cache[key] = data.cpu()
        return self.cache[key].clone()

    def save(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(self.cache, tmp)
        tmp.replace(self.path)
        print("Saved graph cache:", self.path, "n=", len(self.cache))


GRAPH_CACHE_PATH = OUT_DIR / "_shared" / "unirt_ginkan_graph_cache.pt"
graph_cache = UniRTGraphCache(GRAPH_CACHE_PATH)

if PREBUILD_GRAPH_CACHE:
    unique_smiles = external_df["smiles"].astype(str).drop_duplicates().tolist()
    failed = []
    for smi in tqdm(unique_smiles, desc="Prebuild UniRT graphs"):
        try:
            graph_cache.get(smi)
        except Exception as e:
            failed.append({"smiles": smi, "error": repr(e)})
    graph_cache.save()
    print("Graph failures:", len(failed))
    if failed:
        pd.DataFrame(failed).to_csv(OUT_DIR / "unirt_graph_failures.csv", index=False)
        display(pd.DataFrame(failed).head())


def forward_boxcox_normalize(y_sec, stats):
    mean, std, lmbda = stats
    y_sec = np.asarray(y_sec, dtype=float)
    if np.any(y_sec <= 0):
        raise ValueError("Box-Cox requires positive RT seconds.")
    transformed = boxcox(y_sec, lmbda=lmbda)
    return (transformed - mean) / std


def inverse_boxcox_normalized(y_norm, stats):
    mean, std, lmbda = stats
    z = np.asarray(y_norm, dtype=float) * std + mean
    if abs(lmbda) < 1e-12:
        return np.exp(z)
    base = lmbda * z + 1.0
    base = np.maximum(base, 1e-12)
    return np.power(base, 1.0 / lmbda)


class UniRTOODDataset(Dataset):
    def __init__(self, frame, graph_cache, transform_stats):
        super().__init__()
        self.frame = frame.reset_index(drop=True).copy()
        self.graph_cache = graph_cache
        self.y_norm = forward_boxcox_normalize(self.frame["rt_sec"].to_numpy(float), transform_stats)

    def len(self):
        return len(self.frame)

    def get(self, idx):
        row = self.frame.iloc[int(idx)]
        data = self.graph_cache.get(row["smiles"])
        data.y = torch.tensor([float(self.y_norm[int(idx)])], dtype=torch.float)
        data.row_index = torch.tensor([int(idx)], dtype=torch.long)
        return data


def make_loader(frame, shuffle=False):
    ds = UniRTOODDataset(frame, graph_cache, TRANSFORM_STATS)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )

print("Graph cache ready:", GRAPH_CACHE_PATH)



# ============================================================
# Cell 7 — frozen-backbone embeddings and K=0 source-task ensemble
# ============================================================
@torch.no_grad()
def extract_backbone_embedding(model, batch):
    """
    Shared GINKAN representation before Adapter/Cross-Stitch/task embedding/FiLM.
    """
    x, edge_index, graph_batch = batch.x, batch.edge_index, batch.batch
    h = model.embed(x)
    for conv, bn in zip(model.convs, model.bns):
        h = conv(h, edge_index)
        h = bn(h)
        h = F.relu(h)
    return global_add_pool(h, graph_batch)


@torch.no_grad()
def predict_source_task_ensemble(model, g):
    """
    Calibration-free K=0 baseline:
    run the pooled shared embedding through every trained source-task Adapter
    and FiLM-conditioned shared head, then average predictions.
    Cross-Stitch is intentionally excluded because it is not defined for an unseen task.
    """
    preds = []
    batch_size = g.size(0)
    for task_id in range(model.num_tasks):
        task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=g.device)
        g_t = model.adapters[task_id](g)
        task_emb = model.task_embed(task_ids)
        p_t = model.head(g_t, task_emb).view(-1)
        preds.append(p_t)
    return torch.stack(preds, dim=0).mean(dim=0)


def checkpoint_fingerprint(path):
    p = Path(path)
    text = f"{p.resolve()}|{p.stat().st_size}|{p.stat().st_mtime_ns}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


CKPT_FINGERPRINT = checkpoint_fingerprint(selected_ckpt_path)
EMBED_CACHE_DIR = OUT_DIR / "_embedding_cache" / f"seed_{selected_seed}_{CKPT_FINGERPRINT}"
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def get_or_build_method_embeddings(method_id, frame):
    cache_path = EMBED_CACHE_DIR / f"method_{method_id}.pt"
    if USE_EMBEDDING_CACHE and cache_path.exists():
        obj = torch_load_compat(cache_path, map_location="cpu")
        if len(obj["meta"]) == len(frame):
            return obj

    loader = make_loader(frame, shuffle=False)
    H_list, y_list, zero_list = [], [], []
    row_indices = []

    model.eval()
    for batch in tqdm(loader, desc=f"Backbone embeddings {method_id}", leave=False):
        batch = batch.to(DEVICE)
        g = extract_backbone_embedding(model, batch)
        p0 = predict_source_task_ensemble(model, g)
        H_list.append(g.detach().cpu())
        y_list.append(batch.y.view(-1).detach().cpu())
        zero_list.append(p0.detach().cpu())
        row_indices.extend(batch.row_index.view(-1).detach().cpu().numpy().astype(int).tolist())

    H = torch.cat(H_list, dim=0)
    y = torch.cat(y_list, dim=0)
    zero_pred = torch.cat(zero_list, dim=0)

    # Restore original order explicitly.
    order = np.argsort(np.asarray(row_indices))
    H = H[order]
    y = y[order]
    zero_pred = zero_pred[order]
    meta = frame.iloc[order].reset_index(drop=True).copy()

    obj = {
        "H": H,
        "y_norm": y,
        "zero_pred_norm": zero_pred,
        "meta": meta,
        "method_id": method_id,
        "checkpoint": str(selected_ckpt_path),
        "checkpoint_fingerprint": CKPT_FINGERPRINT,
    }
    if USE_EMBEDDING_CACHE:
        torch.save(obj, cache_path)
    return obj


method_frames = {
    mid: external_df[external_df["source_method_id"] == mid]
    .sort_values("_row_order")
    .reset_index(drop=True)
    for mid in TARGET_OOD_METHODS
    if (external_df["source_method_id"] == mid).any()
}

embedding_objects = {}
for mid, frame in method_frames.items():
    embedding_objects[mid] = get_or_build_method_embeddings(mid, frame)
    print(mid, "H shape:", tuple(embedding_objects[mid]["H"].shape))

print("All frozen-backbone embeddings are cached.")



# ============================================================
# Cell 8 — fresh OOD head, support protocol, metrics
# ============================================================
class FreshOODHead(nn.Module):
    def __init__(self, in_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_head_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def fit_fresh_head(H_support, y_support, seed):
    set_head_seed(seed)
    H_support = H_support.to(DEVICE)
    y_support = y_support.to(DEVICE)

    head = FreshOODHead(in_dim=H_support.shape[1], hidden=HEAD_HIDDEN).to(DEVICE)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=HEAD_LR,
        weight_decay=HEAD_WEIGHT_DECAY,
    )

    for _ in range(HEAD_EPOCHS):
        head.train()
        pred = head(H_support)
        loss = (
            0.7 * F.smooth_l1_loss(pred, y_support, beta=0.5)
            + 0.3 * F.mse_loss(pred, y_support)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    head.eval()
    return head


def support_rng_seed(repeat_seed, K):
    if MATCH_FUSE_RT_SUPPORT_SPLITS:
        return int(
            repeat_seed
            + 1_000_000
            + FUSE_RT_REFERENCE_MODEL_SEED * 13
            + FUSE_RT_REFERENCE_FRACTION * 1000
            + K
        )
    return int(repeat_seed + 1_000_000 + selected_seed * 13 + K)


def head_init_seed(repeat_seed, K):
    if MATCH_FUSE_RT_SUPPORT_SPLITS:
        return int(
            repeat_seed
            + FUSE_RT_REFERENCE_MODEL_SEED
            + FUSE_RT_REFERENCE_FRACTION
            + K
        )
    return int(repeat_seed + selected_seed + K)


def compute_metrics_seconds(true_sec, pred_sec):
    true_sec = np.asarray(true_sec, dtype=float)
    pred_sec = np.asarray(pred_sec, dtype=float)
    mask = np.isfinite(true_sec) & np.isfinite(pred_sec)
    true_sec, pred_sec = true_sec[mask], pred_sec[mask]
    err = pred_sec - true_sec
    nonzero = np.abs(true_sec) > 1e-8

    return {
        "n_query": int(len(true_sec)),
        "mae_sec": float(np.mean(np.abs(err))) if len(err) else np.nan,
        "rmse_sec": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
        "medae_sec": float(np.median(np.abs(err))) if len(err) else np.nan,
        "mape_pct": float(np.mean(np.abs(err[nonzero] / true_sec[nonzero])) * 100.0)
                    if np.any(nonzero) else np.nan,
        "r2": float(r2_score(true_sec, pred_sec))
              if len(true_sec) >= 2 and np.std(true_sec) > 1e-12 else np.nan,
    }


def prediction_frame(meta, true_sec, pred_sec, K, repeat_seed, mode):
    out = meta.copy()
    out["true_sec"] = np.asarray(true_sec, dtype=float)
    out["pred_sec"] = np.asarray(pred_sec, dtype=float)
    out["abs_error_sec"] = np.abs(out["pred_sec"] - out["true_sec"])
    out["ape_pct"] = np.where(
        np.abs(out["true_sec"]) > 1e-8,
        out["abs_error_sec"] / np.abs(out["true_sec"]) * 100.0,
        np.nan,
    )
    out["K"] = int(K)
    out["repeat_seed"] = int(repeat_seed)
    out["mode"] = mode
    out["unirt_seed"] = int(selected_seed)
    return out


print("Fresh-head trainable parameters:",
      sum(p.numel() for p in FreshOODHead(in_dim=hidden, hidden=HEAD_HIDDEN).parameters()))
print("Original UniRT trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))



# ============================================================
# Cell 9 — run OOD K-shot evaluation with resume
# ============================================================
METRICS_PATH = OUT_DIR / "unirt_ginkan_ood_kshot_metrics_all_runs.csv"
SUPPORT_MANIFEST_PATH = OUT_DIR / "unirt_ginkan_support_manifest.csv"
PRED_DIR = OUT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

if RESUME_EXISTING and METRICS_PATH.exists():
    metrics_existing = pd.read_csv(METRICS_PATH)

    # Keep only K=0 and the current requested 10 repeat seeds.
    # This prevents legacy seed rows from being mixed into the new mean/SD.
    existing_k = pd.to_numeric(metrics_existing["K"], errors="coerce")
    existing_seed = pd.to_numeric(metrics_existing["repeat_seed"], errors="coerce")
    keep_existing = (
        ((existing_k == 0) & (existing_seed == -1))
        | ((existing_k > 0) & existing_seed.isin(K_REPEAT_SEEDS))
    )
    dropped_old_rows = int((~keep_existing).sum())
    metrics_existing = metrics_existing.loc[keep_existing].copy()

    metric_rows = metrics_existing.to_dict("records")
    done_keys = set(
        (
            str(r["method_id"]).zfill(4),
            int(r["K"]),
            int(r["repeat_seed"]),
        )
        for _, r in metrics_existing.iterrows()
    )
    print(
        "Resume: loaded",
        len(done_keys),
        "current-protocol completed metric rows;",
        "discarded legacy rows:",
        dropped_old_rows,
    )
else:
    metric_rows = []
    done_keys = set()

support_manifest_rows = []
if SUPPORT_MANIFEST_PATH.exists() and RESUME_EXISTING:
    try:
        support_existing = pd.read_csv(SUPPORT_MANIFEST_PATH)
        support_seed = pd.to_numeric(
            support_existing["repeat_seed"], errors="coerce"
        )
        support_existing = support_existing.loc[
            support_seed.isin(K_REPEAT_SEEDS)
        ].copy()
        support_manifest_rows = support_existing.to_dict("records")
    except Exception:
        support_manifest_rows = []

for method_id, obj in embedding_objects.items():
    H_all = obj["H"]
    y_all = obj["y_norm"]
    zero_pred_all = obj["zero_pred_norm"]
    meta_all = obj["meta"].reset_index(drop=True)
    n_total = len(meta_all)
    column_name = (
        meta_all["column.name"].dropna().iloc[0]
        if "column.name" in meta_all.columns and meta_all["column.name"].notna().any()
        else ""
    )

    print(f"\n=== UniRT OOD method={method_id}, n={n_total}, column={column_name} ===")

    # K=0: source-task ensemble baseline.
    if 0 in K_VALUES:
        key = (method_id, 0, -1)
        if key not in done_keys:
            true_sec = inverse_boxcox_normalized(y_all.numpy(), TRANSFORM_STATS)
            pred_sec = inverse_boxcox_normalized(zero_pred_all.numpy(), TRANSFORM_STATS)
            metrics = compute_metrics_seconds(true_sec, pred_sec)
            metrics.update({
                "method_id": method_id,
                "column.name": column_name,
                "K": 0,
                "repeat_seed": -1,
                "n_total": n_total,
                "n_support": 0,
                "mode": "zero_shot_source_task_ensemble",
                "unirt_seed": selected_seed,
                "checkpoint": str(selected_ckpt_path),
            })
            metric_rows.append(metrics)
            done_keys.add(key)

            if SAVE_QUERY_PREDICTIONS:
                pred_df = prediction_frame(
                    meta_all, true_sec, pred_sec, 0, -1,
                    "zero_shot_source_task_ensemble"
                )
                pred_df.to_csv(PRED_DIR / f"{method_id}_K000_zero_shot.csv", index=False)

    # K > 0 fresh-head adaptation.
    for K in [k for k in K_VALUES if k > 0]:
        if K >= n_total:
            print(f"Skip method={method_id}, K={K}: n_total={n_total}")
            continue

        for repeat_seed in K_REPEAT_SEEDS:
            key = (method_id, int(K), int(repeat_seed))
            if key in done_keys:
                continue

            rng = np.random.default_rng(support_rng_seed(repeat_seed, K))
            support_idx = np.sort(rng.choice(np.arange(n_total), size=int(K), replace=False))
            support_mask = np.zeros(n_total, dtype=bool)
            support_mask[support_idx] = True
            query_idx = np.where(~support_mask)[0]

            H_support = H_all[support_idx]
            y_support = y_all[support_idx]
            H_query = H_all[query_idx]
            y_query = y_all[query_idx]
            meta_query = meta_all.iloc[query_idx].reset_index(drop=True)

            head = fit_fresh_head(
                H_support,
                y_support,
                seed=head_init_seed(repeat_seed, K),
            )

            with torch.no_grad():
                pred_norm = head(H_query.to(DEVICE)).detach().cpu().numpy()

            true_sec = inverse_boxcox_normalized(y_query.numpy(), TRANSFORM_STATS)
            pred_sec = inverse_boxcox_normalized(pred_norm, TRANSFORM_STATS)

            metrics = compute_metrics_seconds(true_sec, pred_sec)
            metrics.update({
                "method_id": method_id,
                "column.name": column_name,
                "K": int(K),
                "repeat_seed": int(repeat_seed),
                "n_total": int(n_total),
                "n_support": int(K),
                "mode": "frozen_backbone_fresh_head",
                "unirt_seed": int(selected_seed),
                "checkpoint": str(selected_ckpt_path),
                "support_rng_seed": support_rng_seed(repeat_seed, K),
                "head_init_seed": head_init_seed(repeat_seed, K),
            })
            metric_rows.append(metrics)
            done_keys.add(key)

            # Save support mol_keys so FUSE-RT/UniRT support sets can be audited.
            support_meta = meta_all.iloc[support_idx]
            for _, r in support_meta.iterrows():
                support_manifest_rows.append({
                    "method_id": method_id,
                    "K": int(K),
                    "repeat_seed": int(repeat_seed),
                    "row_id": r.get("row_id", ""),
                    "mol_key": r.get("mol_key", ""),
                    "smiles": r.get("smiles", ""),
                    "support_rng_seed": support_rng_seed(repeat_seed, K),
                })

            if SAVE_QUERY_PREDICTIONS:
                pred_df = prediction_frame(
                    meta_query, true_sec, pred_sec, K, repeat_seed,
                    "frozen_backbone_fresh_head"
                )
                pred_dir = PRED_DIR / method_id / f"K_{K:04d}"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_df.to_csv(pred_dir / f"repeat_{repeat_seed}.csv", index=False)

            pd.DataFrame(metric_rows).to_csv(METRICS_PATH, index=False)
            pd.DataFrame(support_manifest_rows).drop_duplicates(
                ["method_id", "K", "repeat_seed", "mol_key"]
            ).to_csv(SUPPORT_MANIFEST_PATH, index=False)

metrics_all = pd.DataFrame(metric_rows)
metrics_all.to_csv(METRICS_PATH, index=False)

support_manifest = pd.DataFrame(support_manifest_rows)
if not support_manifest.empty:
    support_manifest = support_manifest.drop_duplicates(
        ["method_id", "K", "repeat_seed", "mol_key"]
    )
    support_manifest.to_csv(SUPPORT_MANIFEST_PATH, index=False)

print("\n=== UniRT OOD metric rows ===")
display(metrics_all)
print("Saved:", METRICS_PATH)
print("Saved:", SUPPORT_MANIFEST_PATH)

# ============================================================
# Cell 10 — summaries and headless-safe plots
# ============================================================
metrics_all = pd.read_csv(METRICS_PATH)

# Audit that every K>0 summary uses only the current 10 split seeds.
observed_positive_seeds = sorted(
    pd.to_numeric(
        metrics_all.loc[pd.to_numeric(metrics_all["K"], errors="coerce") > 0, "repeat_seed"],
        errors="coerce",
    ).dropna().astype(int).unique().tolist()
)
unexpected_seeds = sorted(set(observed_positive_seeds) - set(K_REPEAT_SEEDS))
if unexpected_seeds:
    raise RuntimeError(
        f"Unexpected legacy repeat seeds remain in metrics: {unexpected_seeds}"
    )
print("Observed K>0 repeat seeds:", observed_positive_seeds)

for c in ["K", "repeat_seed", "n_query", "mae_sec", "rmse_sec", "medae_sec", "mape_pct", "r2"]:
    if c in metrics_all.columns:
        metrics_all[c] = pd.to_numeric(metrics_all[c], errors="coerce")


def summarize(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        for metric in ["mae_sec", "rmse_sec", "medae_sec", "mape_pct", "r2"]:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{metric}_n"] = int(len(vals))
        row["n_query_mean"] = float(g["n_query"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


summary_by_method = summarize(metrics_all, ["method_id", "column.name", "K"])
summary_overall = summarize(metrics_all, ["K"])

summary_by_method.to_csv(OUT_DIR / "unirt_ood_summary_by_method_k.csv", index=False)
summary_overall.to_csv(OUT_DIR / "unirt_ood_overall_summary_by_k.csv", index=False)

print("=== Overall summary ===")
display(summary_overall)
print("=== Per-method summary ===")
display(summary_by_method)

# ============================================================
# Cell 11 — optional direct comparison with existing FUSE-RT OOD results
# ============================================================
FUSE_RT_BEST_BY_METHOD_CANDIDATES = [
    Path("outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/best_fraction_by_method_and_K_mae.csv"),
    Path("../outputs/randonpy_f6/09_new_report_low_overlap_ood_kshot_layer2_cached/best_fraction_by_method_and_K_mae.csv"),
]

fuse_rt_best_path = first_existing(FUSE_RT_BEST_BY_METHOD_CANDIDATES, kind="file")
if fuse_rt_best_path is None:
    print("FUSE-RT best-by-method table not found; comparison skipped.")
else:
    fuse_rt_best = pd.read_csv(fuse_rt_best_path)
    fuse_rt_best["method_id"] = fuse_rt_best["method_id"].astype(str).str.extract(r"(\d+)")[0].astype(int).astype(str).str.zfill(4)
    fuse_rt_best["K"] = pd.to_numeric(fuse_rt_best["K"], errors="coerce")

    uni = summary_by_method.copy()
    uni["method_id"] = uni["method_id"].astype(str).str.zfill(4)

    compare = uni.merge(
        fuse_rt_best[
            [
                "method_id", "K", "radonpy_percent",
                "mae_sec_mean", "mae_sec_std", "r2_mean"
            ]
        ].rename(columns={
            "radonpy_percent": "fuse_rt_best_radonpy_percent",
            "mae_sec_mean": "fuse_rt_mae_sec_mean",
            "mae_sec_std": "fuse_rt_mae_sec_std",
            "r2_mean": "fuse_rt_r2_mean",
        }),
        on=["method_id", "K"],
        how="left",
    )

    compare = compare.rename(columns={
        "mae_sec_mean": "unirt_mae_sec_mean",
        "mae_sec_std": "unirt_mae_sec_std",
        "r2_mean": "unirt_r2_mean",
    })
    compare["unirt_minus_fuse_rt_mae_sec"] = (
        compare["unirt_mae_sec_mean"] - compare["fuse_rt_mae_sec_mean"]
    )

    compare.to_csv(OUT_DIR / "unirt_vs_fuse_rt_best_ood_comparison.csv", index=False)
    print("=== UniRT vs best FUSE-RT fraction ===")
    display(compare)


# Refresh the two report-facing tables after a successful evaluation.
CURATED_ROOT = PROJECT_ROOT / "result" / "unirt_external_ood"
CURATED_ROOT.mkdir(parents=True, exist_ok=True)
for source_name, target_name in [
    ("unirt_ood_summary_by_method_k.csv", "summary.csv"),
    ("unirt_ginkan_ood_kshot_metrics_all_runs.csv", "per_model_seed.csv"),
]:
    source_path = OUT_DIR / source_name
    if source_path.exists():
        __import__("shutil").copy2(source_path, CURATED_ROOT / target_name)
