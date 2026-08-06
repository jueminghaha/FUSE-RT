#!/usr/bin/env python3
"""Build the refactored local project from the downloaded notebook archive.

This migration is intentionally conservative: source notebooks and raw outputs are
never modified. Large checkpoint tensors are hard-linked when the filesystem allows
it, while reports and metadata are copied into the curated project tree.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence


SEEDS = [2004, 2006, 2011, 2012, 2016, 2020, 2022, 2027, 2032, 2034]


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_model_name(source: str) -> str:
    """Prevent legacy notebook sources from restoring the former model name."""
    legacy_name = "".join(("R", "E", "M", "T"))
    source = source.replace(legacy_name, "FUSE_RT").replace(legacy_name.lower(), "fuse_rt")
    return re.sub(r"\bFUSE_RT\b", "FUSE-RT", source)


def code_from_cells(path: Path, cell_indices: Sequence[int] | None = None) -> str:
    notebook = read_notebook(path)
    selected = set(cell_indices) if cell_indices is not None else None
    blocks = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code" or (selected is not None and index not in selected):
            continue
        blocks.append("".join(cell.get("source", [])))
    return normalize_model_name("\n\n".join(blocks).strip() + "\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def strip_ipython(source: str) -> str:
    source = re.sub(r"^from IPython\.display import .*$", "", source, flags=re.MULTILINE)
    return source


def strip_server_specific_paths(source: str) -> str:
    """Remove obsolete absolute paths while retaining local/configured candidates."""
    server_home = re.compile(r"/(?:home|workspace)/[^/\s\"']+/")
    lines = [
        line
        for line in source.splitlines()
        if server_home.search(line) is None
    ]
    return "\n".join(lines).strip() + "\n"


def keep_before_marker(source: str, marker: str) -> str:
    """Keep the computational part of a notebook cell and drop its plotting tail."""
    return source.split(marker, 1)[0].rstrip() + "\n"


def node_target_name(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
    return target.id if isinstance(target, ast.Name) else None


def clean_engine_source(source: str) -> str:
    """Keep the final version of duplicated notebook definitions and assignments."""
    tree = ast.parse(source)
    last_definition: dict[str, int] = {}
    last_assignment: dict[str, int] = {}
    dedupe_assignments = {"RADONPY_RECOMMENDED_TARGETS", "RADONPY_TARGET_TRANSFORMS"}
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            last_definition[node.name] = index
        name = node_target_name(node)
        if name in dedupe_assignments:
            last_assignment[name] = index

    remove_ranges: list[tuple[int, int]] = []
    for index, node in enumerate(tree.body):
        remove = False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            remove = last_definition[node.name] != index
        name = node_target_name(node)
        if name in dedupe_assignments and last_assignment[name] != index:
            remove = True
        if name == "df_meta":
            remove = True
        if name == "SPLIT_SEEDS" and isinstance(node, ast.Assign):
            remove = True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            rendered = ast.unparse(node.value)
            if "Weight-only runner active" in rendered or "Global metadata shape" in rendered:
                remove = True
        if remove:
            remove_ranges.append((node.lineno, node.end_lineno or node.lineno))

    lines = source.splitlines()
    removed = set()
    for start, end in remove_ranges:
        removed.update(range(start - 1, end))
    cleaned = [line for index, line in enumerate(lines) if index not in removed]
    return "\n".join(cleaned).strip() + "\n"


def build_configs(project_root: Path) -> None:
    common = {
        "seeds": SEEDS,
        "rt_epochs": 200,
        "radonpy_valid_fraction": 0.10,
        "resume_completed": True,
        "fail_fast": False,
        "model": {
            "d_model": 256,
            "gnn_layers": 5,
            "molecular_transformer_layers": 1,
            "transformer_layers": 3,
            "decoder_layers": 2,
            "num_heads": 4,
            "dropout": 0.18,
        },
        "optimization": {
            "batch_size": 96,
            "radon_batch_size": 128,
            "num_workers": 0,
            "lr": 0.0002,
            "weight_decay": 0.0001,
            "grad_clip": 5.0,
            "early_stop_patience": 30,
            "scheduler_patience": 8,
            "scheduler_factor": 0.5,
            "min_lr": 0.000001,
            "aux_loss_weight": 1.0,
        },
    }
    experiment_rows = [
        ("E1", "M0_no_aux", "R0_device_single", "M0 no PolyOmics × R0 device encoder + single head", True, "single", False, 0.0, 0),
        ("E2", "M0_no_aux", "R1_device_multi", "M0 no PolyOmics × R1 device encoder + multitask heads", True, "multi", False, 0.0, 0),
        ("E3", "M0_no_aux", "R2_nodevice_multi", "M0 no PolyOmics × R2 no device + multitask heads", False, "multi", False, 0.0, 0),
        ("E4", "M1_pretrain", "R0_device_single", "M1 RadonPy pretrain × R0 device encoder + single head", True, "single", False, 100.0, 50),
        ("E5", "M1_pretrain", "R1_device_multi", "M1 RadonPy pretrain × R1 device encoder + multitask heads", True, "multi", False, 100.0, 50),
        ("E6", "M1_pretrain", "R2_nodevice_multi", "M1 RadonPy pretrain × R2 no device + multitask heads", False, "multi", False, 100.0, 50),
        ("E7", "M2_joint", "R0_device_single", "M2 joint multitask × R0 device encoder + single head", True, "single", True, 100.0, 0),
        ("E8", "M2_joint", "R1_device_multi", "M2 joint multitask × R1 device encoder + multitask heads", True, "multi", True, 100.0, 0),
        ("E9", "M2_joint", "R2_nodevice_multi", "M2 joint multitask × R2 no device + multitask heads", False, "multi", True, 100.0, 0),
    ]
    experiments: dict[str, dict] = {}
    for exp_id, mode, architecture, name, device_meta, head, joint, percent, pretrain_epochs in experiment_rows:
        experiments[exp_id] = {
            "exp_id": exp_id,
            "molecule_mode": mode,
            "rt_architecture": architecture,
            "name": name,
            "use_device_metadata": device_meta,
            "rt_head_type": head,
            "joint_multitask": joint,
            "radonpy_percent": percent,
            "radonpy_pretrain_epochs": pretrain_epochs,
            "output_group": "experiments",
            "output_name": exp_id,
        }

    for key, percent, source_label in [
        ("E8_p01", 0.05, "point_01_pct_000p050000"),
        ("E8_p02", 0.334370152488211, "point_02_pct_000p334370"),
        ("E8_p03", 2.23606797749979, "point_03_pct_002p236068"),
        ("E8_p04", 14.9534878122122, "point_04_pct_014p953488"),
    ]:
        experiments[key] = {
            **experiments["E8"],
            "name": f"E8 auxiliary scaling at {percent:.12g}%",
            "radonpy_percent": percent,
            "radonpy_pretrain_epochs": 0,
            "output_group": "scaling",
            "output_name": key,
            "source_label": source_label,
        }

    write_json(project_root / "config" / "experiments.json", {"common": common, "experiments": experiments})
    write_json(
        project_root / "config" / "paths.json",
        {
            "report_root": "data/RepoRT_latest",
            "data_dir": "data/raw",
            "metadata_csv": "data/proc_metadata_sw_20250405.csv",
            "method_ids_csv": "data/processed/method_selection_179/our_179_method_ids.csv",
            "split_root": "data/processed/exact_full_inchikey_split_6_2_2",
            "split_root_180": "data/processed/exact_full_inchikey_split_6_2_2_180dataset",
        },
    )


ENGINE_HEADER = r'''"""Shared RepoRT/PolyOmics model and training engine.

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
'''


ENGINE_TAIL = r'''

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
            (fail_dir / "run_failed.txt").write_text(error, encoding="utf-8")
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
'''


def build_engine(source_root: Path, project_root: Path) -> None:
    notebook = source_root / "RepoRT_PolyOmic" / "E8_M2_joint_R1_device_multi_weights.ipynb"
    shared = code_from_cells(notebook, [2, 3, 4, 5, 6, 7])
    shared = clean_engine_source(shared)
    shared = re.sub(
        r"^\s*roots = \[SPLIT_ROOT, WORKDIR / 'outputs/exact_full_inchikey_split_6_2_2'.*$",
        "    roots = [SPLIT_ROOT]",
        shared,
        flags=re.MULTILINE,
    )
    write_text(project_root / "model" / "engine.py", ENGINE_HEADER + "\n\n" + shared + ENGINE_TAIL)
    write_text(
        project_root / "model" / "__init__.py",
        '"""Shared model implementations for the RepoRT/PolyOmics experiments."""\n',
    )


def build_data_scripts(source_root: Path, project_root: Path) -> None:
    # These files are now curated script implementations rather than direct
    # notebook dumps.  Preserve them when refreshing the rest of the project.
    curated_targets = [
        project_root / "data" / "select_179_methods.py",
        project_root / "data" / "build_exact_splits.py",
        project_root / "data" / "extend_split_0186.py",
    ]
    if all(path.exists() for path in curated_targets):
        print("Preserving curated data preparation scripts.")
        return

    mappings = [
        ("00_data_split_exact_full_inchikey_10seeds.ipynb", "build_exact_splits.py"),
        ("00_data_split_exact_full_inchikey_180dataset_add0186_10seeds.ipynb", "extend_split_0186.py"),
    ]
    for notebook_name, output_name in mappings:
        source = strip_server_specific_paths(
            strip_ipython(code_from_cells(source_root / "RepoRT_PolyOmic" / notebook_name))
        )
        source = source.replace(
            "from pathlib import Path",
            "from pathlib import Path\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nSOURCE_ROOT = PROJECT_ROOT.parent",
            1,
        )
        source = source.replace("REPORT_AUDIT_CANDIDATES = [", "REPORT_AUDIT_CANDIDATES = [\n    SOURCE_ROOT / 'outputs/report_laest_overlap_audit',")
        source = source.replace("BASE_SPLIT_CANDIDATES = [", "BASE_SPLIT_CANDIDATES = [\n    SOURCE_ROOT / 'RepoRT_PolyOmic/outputs/exact_full_inchikey_split_6_2_2',")
        source = source.replace(
            'OUT_DIR = Path("outputs/exact_full_inchikey_split_6_2_2")',
            'OUT_DIR = PROJECT_ROOT / "data/processed/exact_full_inchikey_split_6_2_2"',
        )
        source = source.replace(
            'OUT_DIR = Path("outputs/exact_full_inchikey_split_6_2_2_180dataset")',
            'OUT_DIR = PROJECT_ROOT / "data/processed/exact_full_inchikey_split_6_2_2_180dataset"',
        )
        source = "def display(value):\n    print(value)\n\n" + source
        write_text(project_root / "data" / output_name, source)
    write_text(project_root / "data" / "__init__.py", '"""Data preparation scripts and dataset locations."""\n')


EVAL_HEADER = r'''from __future__ import annotations

import ast
import csv
import gc
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
import traceback
import warnings
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import engine
engine.configure_experiment("E1")
from model.engine import *

def display(value):
    print(value)

class Markdown(str):
    pass
'''


UNIRT_MODEL_HEADER = r'''"""UniRT-GINKAN architecture used by the external OOD baseline.

Upstream implementation: https://github.com/hcji/Uni-RT
Reproducibility reference: b981fc118b1264f937626b8ec980b426100537b1
Uni-RT is distributed under the MIT License; this module keeps the upstream
provenance visible while adapting paths and evaluation entry points locally.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, GINConv, global_add_pool

CUDA_DEVICE_INDEX = int(os.environ.get("CUDA_DEVICE_INDEX", "0"))
DEVICE = torch.device(f"cuda:{CUDA_DEVICE_INDEX}" if torch.cuda.is_available() else "cpu")
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
'''


def build_unirt_model_and_eval(source_root: Path, project_root: Path) -> None:
    notebook = source_root / "RepoRT_PolyOmic" / "10_UniRT_GINKAN_NewRepoRT_OOD_KShot_FrozenBackbone_seed10.ipynb"
    model_source = code_from_cells(notebook, [2, 3])
    write_text(project_root / "model" / "unirt.py", UNIRT_MODEL_HEADER + "\n" + model_source)

    config_source = code_from_cells(notebook, [1])
    config_source = config_source.replace("from __future__ import annotations", "")
    config_source = strip_ipython(config_source)
    config_source = config_source.replace("from matplotlib.figure import Figure", "")
    config_source = config_source.replace("from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas", "")
    config_source = config_source.replace('DEVICE = torch.device("cuda:0")', 'DEVICE = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")')
    config_source = config_source.replace("WORKDIR = Path.cwd()", "WORKDIR = PROJECT_ROOT")
    config_source = config_source.replace(
        'MANUAL_CHECKPOINT = None',
        'MANUAL_CHECKPOINT = Path(os.environ["UNIRT_CHECKPOINT"]) if os.environ.get("UNIRT_CHECKPOINT") else None',
    )
    config_source = config_source.replace(
        'OUT_DIR = Path("outputs/unirt_ginkan_new_report_ood_kshot_frozen_backbone")',
        'OUT_DIR = PROJECT_ROOT / "result" / "_runs" / "unirt_external_ood"',
    )
    config_source = config_source.replace(
        "UNI_RT_OUTPUT_ROOT_CANDIDATES = [",
        'UNI_RT_OUTPUT_ROOT_CANDIDATES = [\n    Path(os.environ["UNIRT_OUTPUT_ROOT"]) if os.environ.get("UNIRT_OUTPUT_ROOT") else WORKDIR / "outputs/uni_rt_reproduction/RPLC/GINKAN",',
    )
    config_source = config_source.replace(
        "EXTERNAL_ROWS_CANDIDATES = [",
        'EXTERNAL_ROWS_CANDIDATES = [\n    PROJECT_ROOT.parent / "RepoRT_PolyOmic/outputs/evaluation_E1_E9_OOD_kshot_lowoverlap_shimadzu/ood_rows_master_lowoverlap_shimadzu.csv",',
    )
    config_source = config_source.replace(
        "FUSE_RT_METRICS_CANDIDATES = [",
        'FUSE_RT_METRICS_CANDIDATES = [\n    PROJECT_ROOT / "result/external_ood/per_model_seed.csv",',
    )
    config_source = config_source.replace(
        "REPORT_ROOT_CANDIDATES = [",
        'REPORT_ROOT_CANDIDATES = [\n    PROJECT_ROOT.parent / "outputs/report_laest_overlap_audit/RepoRT_latest",',
    )
    config_source = config_source.replace(
        "SPLIT_DIR_CANDIDATES = [",
        'SPLIT_DIR_CANDIDATES = [\n    PROJECT_ROOT.parent / "RepoRT_PolyOmic/outputs/exact_full_inchikey_split_6_2_2",',
    )
    config_source = config_source.replace("torch.cuda.get_device_name(0)", "torch.cuda.get_device_name(CUDA_DEVICE_INDEX)")
    config_source = strip_server_specific_paths(config_source)

    summary_source = keep_before_marker(code_from_cells(notebook, [10]), "def save_curve(")
    evaluation_source = (
        code_from_cells(notebook, [4, 5, 6, 7, 8, 9])
        + "\n"
        + summary_source
        + "\n"
        + code_from_cells(notebook, [11])
    )
    evaluation_source = strip_ipython(strip_server_specific_paths(evaluation_source))

    header = r'''from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CUDA_DEVICE_INDEX = int(__import__("os").environ.get("CUDA_DEVICE_INDEX", "0"))
from model.unirt import *

def display(value):
    print(value)
'''
    curated_tail = r'''

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
'''
    write_text(
        project_root / "script" / "evaluate_unirt_external_ood.py",
        header + "\n" + config_source + "\n" + evaluation_source + curated_tail,
    )


def build_evaluation_scripts(source_root: Path, project_root: Path) -> None:
    external_nb = source_root / "RepoRT_PolyOmic" / "13Evaluate_E1_E9_OOD_KShot_LowOverlap_Shimadzu_REALLY_FIXED.ipynb"
    external_source = (
        code_from_cells(external_nb, [10, 11, 12, 13, 14])
        + "\n"
        + keep_before_marker(code_from_cells(external_nb, [15]), "# Plot helpers.")
        + "\n"
        + code_from_cells(external_nb, [16, 17])
    )
    external = strip_server_specific_paths(
        strip_ipython(external_source)
    )
    external = external.replace("import matplotlib.pyplot as plt", "")
    external = external.replace("(EVAL_ROOT / 'figures').mkdir(parents=True, exist_ok=True)", "")
    external = external.replace(
        "# Cell B6. Summaries and separated errorbar + shaded-area plots",
        "# Cell B6. Summary tables and ranks",
    )
    external = external.replace(
        "#   Bypass matplotlib / inline plotting errors.",
        "#   Aggregate the final CSV result tables.",
    )
    external = external.replace(
        "EVAL_ROOT = WORKDIR / 'outputs' / 'evaluation_E1_E9_OOD_kshot_lowoverlap_shimadzu'",
        "EVAL_ROOT = PROJECT_ROOT / 'result' / '_runs' / 'external_ood'",
    ).replace("WEIGHT_ROOT = WORKDIR / 'outputs'", "WEIGHT_ROOT = PROJECT_ROOT / 'checkpoints' / 'experiments'")
    external = re.sub(
        r"(def configure_experiment_globals\([^\n]+\):\n)",
        r"\1    engine.configure_experiment(str(meta['EXP_ID']))\n",
        external,
        count=1,
    )
    external += '''\n\n# Refresh the two report-facing tables after a successful evaluation.\nCURATED_ROOT = PROJECT_ROOT / "result" / "external_ood"\nCURATED_ROOT.mkdir(parents=True, exist_ok=True)\nfor source_name, target_name in [\n    ("summary_by_ood_tier_experiment_K.csv", "summary.csv"),\n    ("ood_kshot_metrics_all_runs.csv", "per_model_seed.csv"),\n]:\n    source_path = EVAL_ROOT / source_name\n    if source_path.exists():\n        shutil.copy2(source_path, CURATED_ROOT / target_name)\n'''
    write_text(project_root / "script" / "evaluate_external_ood.py", EVAL_HEADER + "\n" + external)

    internal_nb = source_root / "RepoRT_PolyOmic" / "14_Evaluate_E1_E9_InternalOOD_KShot_K0_5_20_30_100.ipynb"
    internal_source = (
        code_from_cells(internal_nb, [10, 11, 12, 13, 14])
        + "\n"
        + keep_before_marker(code_from_cells(internal_nb, [15]), "# Saved figures:")
        + "\n"
        + code_from_cells(internal_nb, [16, 17])
    )
    internal = strip_server_specific_paths(
        strip_ipython(internal_source)
    )
    internal = re.sub(
        r"# Use a non-interactive backend.*?import matplotlib\.pyplot as plt\s*",
        "",
        internal,
        count=1,
        flags=re.DOTALL,
    )
    internal = internal.replace("(EVAL_ROOT / 'figures').mkdir(parents=True, exist_ok=True)", "")
    internal = internal.replace(
        "# Cell B6. Per-dataset summaries, task-macro summaries, ranks, and figures",
        "# Cell B6. Per-dataset summaries, task-macro summaries, and ranks",
    )
    internal = internal.replace(
        "EVAL_ROOT = WORKDIR / 'outputs' / 'evaluation_E1_E9_internal_heldout_OOD_kshot'",
        "EVAL_ROOT = PROJECT_ROOT / 'result' / '_runs' / 'internal_ood'",
    ).replace("WEIGHT_ROOT = WORKDIR / 'outputs'", "WEIGHT_ROOT = PROJECT_ROOT / 'checkpoints' / 'experiments'")
    internal = re.sub(
        r"(def configure_experiment_globals\([^\n]+\):\n)",
        r"\1    engine.configure_experiment(str(meta['EXP_ID']))\n",
        internal,
        count=1,
    )
    internal += '''\n\n# Refresh the two report-facing tables after a successful evaluation.\nCURATED_ROOT = PROJECT_ROOT / "result" / "internal_ood"\nCURATED_ROOT.mkdir(parents=True, exist_ok=True)\nfor source_name, target_name in [\n    ("summary_by_internal_ood_dataset_experiment_K.csv", "summary.csv"),\n    ("internal_ood_kshot_metrics_all_runs.csv", "per_model_seed.csv"),\n]:\n    source_path = EVAL_ROOT / source_name\n    if source_path.exists():\n        shutil.copy2(source_path, CURATED_ROOT / target_name)\n'''
    write_text(project_root / "script" / "evaluate_internal_ood.py", EVAL_HEADER + "\n" + internal)

    scaling_nb = source_root / "RepoRT_PolyOmic" / "E2_to_E8_RadonPy_scaling_internal_test_report.ipynb"
    scaling = strip_server_specific_paths(
        strip_ipython(code_from_cells(scaling_nb, [9, 10, 11, 13, 15]))
    )
    scaling = scaling.replace("'relative_dir': 'E3'", "'relative_dir': 'experiments/E3'")
    scaling = scaling.replace("'relative_dir': 'E2'", "'relative_dir': 'experiments/E2'")
    for old, new in [
        ("PROJECT_OUTPUT_ROOT / 'E8_scaling_log5' / 'point_01_pct_000p050000'", "PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p01'"),
        ("PROJECT_OUTPUT_ROOT / 'E8_scaling_log5' / 'point_02_pct_000p334370'", "PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p02'"),
        ("PROJECT_OUTPUT_ROOT / 'E8_scaling_log5' / 'point_03_pct_002p236068'", "PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p03'"),
        ("PROJECT_OUTPUT_ROOT / 'E8_scaling_log5' / 'point_04_pct_014p953488'", "PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p04'"),
        ("PROJECT_OUTPUT_ROOT / 'E8'", "PROJECT_OUTPUT_ROOT / 'experiments' / 'E8'"),
    ]:
        scaling = scaling.replace(old, new)
    scaling = scaling.replace("# E3 → E8 RadonPy Scaling", "# E2 → E8 RadonPy Scaling")
    for old, new in [
        ("\u968f auxiliary \u6bd4\u4f8b\u589e\u52a0\u5355\u8c03\u6539\u5584", "improves monotonically as the auxiliary fraction increases"),
        ("\u968f auxiliary \u6bd4\u4f8b\u589e\u52a0\u5355\u8c03\u6076\u5316", "degrades monotonically as the auxiliary fraction increases"),
        ("\u4e0d\u662f\u4e25\u683c\u5355\u8c03\u5173\u7cfb\uff0c\u5b58\u5728\u4e2d\u95f4\u6bd4\u4f8b\u6700\u4f18\u6216 seed \u6ce2\u52a8", "is not strictly monotonic; an intermediate fraction is optimal or seed-level variability is present"),
        ("# E2 \u2192 E8 RadonPy Scaling\uff1aInternal Test \u81ea\u52a8\u62a5\u544a", "# E2 \u2192 E8 RadonPy Scaling: Automated Internal-Test Report"),
        ("- \u5b8c\u6210\u70b9\u6570\uff1a", "- Completed points: "),
        ("\uff1b\u6bcf\u70b9 seeds\uff1a", "; seeds per point: "),
        ("- \u6a2a\u8f74\u91c7\u7528\u516d\u4e2a\u7b49\u8ddd log-step\uff1a0% baseline + \u4e94\u4e2a log10 \u7b49\u8ddd\u6b63\u6bd4\u4f8b\u70b9\u3002", "- The horizontal axis uses six uniformly spaced log-scale steps: a 0% baseline plus five positive fractions equally spaced in log10 space."),
        ("- **\u8bbe\u8ba1\u9650\u5236\uff1a0% \u4f7f\u7528 E3-R2\uff0c\u800c\u6b63\u6bd4\u4f8b\u70b9\u4f7f\u7528 E8-R1\uff1b\u56e0\u6b64 0%\u21920.05% \u7684\u53d8\u5316\u540c\u65f6\u5305\u542b architecture effect\uff0c\u4e0d\u80fd\u4f5c\u4e3a\u7eaf data-scaling \u56e0\u679c\u7ed3\u8bba\u3002**", "- **Design constraint: the 0% point uses E3-R2, whereas positive fractions use E8-R1; the 0%\u21920.05% contrast therefore includes an architectural effect and does not isolate the causal effect of data scaling.**"),
        ("- 0% \u4f7f\u7528 E2-R1\uff0c\u4e0e E8 \u6b63\u6bd4\u4f8b\u70b9\u4fdd\u6301\u76f8\u540c RT architecture\uff0c\u53ef\u4f5c\u4e3a\u4e25\u683c auxiliary scaling \u66f2\u7ebf\u3002", "- The 0% point uses E2-R1 and therefore matches the RT architecture used by the positive-fraction E8 points, yielding a controlled auxiliary-data scaling curve."),
        ("## Overall internal-test \u6700\u4f73\u70b9", "## Best Overall Internal-Test Point"),
        ("## 100% \u76f8\u5bf9 0% endpoint", "## Endpoint Comparison: 100% versus 0%"),
        ("\uff08\u6b63\u503c\u8868\u793a 100% \u66f4\u597d\uff09", "(positive values indicate that the 100% endpoint is better)"),
        ("## Method-equal macro \u7ed3\u679c", "## Method-Equal Macro Results"),
        ("{label} \u6700\u4f73", "Best {label}"),
        ("## \u89e3\u8bfb\u539f\u5219", "## Interpretation"),
        ("- Overall micro \u6307\u6807\u6309\u6240\u6709 internal-test rows \u8ba1\u7b97\uff0c\u5927\u6570\u636e\u96c6\u6743\u91cd\u66f4\u9ad8\u3002", "- Overall micro metrics are computed over all internal-test observations and therefore assign greater weight to larger datasets."),
        ("- Method-macro \u6307\u6807\u5148\u5728\u6bcf\u4e2a method \u5185\u8ba1\u7b97\uff0c\u518d\u5bf9 methods \u7b49\u6743\u5e73\u5747\uff0c\u66f4\u9002\u5408\u5224\u65ad\u8de8\u8272\u8c31\u65b9\u6cd5\u7684\u666e\u904d\u6027\u3002", "- Method-macro metrics are first computed within each chromatographic method and then averaged with equal method weights, providing a measure of cross-method generality."),
        ("- \u6c47\u603b\u8868\u540c\u65f6\u62a5\u544a seed SD \u4e0e seed mean \u7684 95% CI\u3002", "- Summary tables report both the standard deviation across seeds and the 95% confidence interval of the seed mean."),
        ("- \u5224\u65ad\u67d0\u4e2a\u6bd4\u4f8b\u662f\u5426\u771f\u6b63\u4f18\u4e8e baseline\uff0c\u5e94\u540c\u65f6\u67e5\u770b paired_comparisons_vs_zero.csv\uff0c\u800c\u4e0d\u662f\u53ea\u6bd4\u8f83\u5747\u503c\u3002", "- Claims that a fraction improves upon the baseline should be evaluated using paired_comparisons_vs_zero.csv rather than mean values alone."),
        ("- Error bar \u4e3a seed SD\uff1b\u9634\u5f71\u4e3a seed mean \u7684 95% CI\u3002", "- Summary tables report both the standard deviation across seeds and the 95% confidence interval of the seed mean."),
    ]:
        scaling = scaling.replace(old, new)
    for old, new in [("\uff1a", ":"), ("\uff0c", ", "), ("\uff1b", "; "), ("\u3002", ".")]:
        scaling = scaling.replace(old, new)
    scaling = scaling.replace("    FIG_DIR,\n", "")
    scaling = scaling.replace(
        'Path("outputs/evaluation_E3_to_E8_scaling_internal_test")',
        'PROJECT_ROOT / "result" / "_runs" / "scaling_internal_test"',
    )
    scaling_header = EVAL_HEADER.replace(
        'engine.configure_experiment("E1")',
        'engine.configure_experiment("E8")',
    ) + "\nPROJECT_OUTPUT_ROOT = PROJECT_ROOT / 'checkpoints'\nROOT_OUT = PROJECT_ROOT / 'result' / '_runs' / 'scaling_internal_test'\nROOT_OUT.mkdir(parents=True, exist_ok=True)\n"
    scaling += '''\n\n# Refresh the two report-facing tables after a successful evaluation.\nCURATED_ROOT = PROJECT_ROOT / "result" / "scaling_internal_test"\nCURATED_ROOT.mkdir(parents=True, exist_ok=True)\nfor source_name, target_name in [\n    ("internal_test_scaling_summary.csv", "summary.csv"),\n    ("internal_test_seed_metrics.csv", "per_model_seed.csv"),\n]:\n    source_path = ROOT_OUT / source_name\n    if source_path.exists():\n        shutil.copy2(source_path, CURATED_ROOT / target_name)\n'''
    write_text(project_root / "script" / "evaluate_scaling_internal_test.py", scaling_header + "\n" + scaling)


def link_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_checkpoint_set(source_dir: Path, destination_dir: Path, key: str, index_rows: list[dict]) -> None:
    for seed in SEEDS:
        source_seed = source_dir / f"seed_{seed}"
        destination_seed = destination_dir / f"seed_{seed}"
        weight = source_seed / "best_model.pt"
        if not weight.exists():
            raise FileNotFoundError(weight)
        link_checkpoint(weight, destination_seed / "best_model.pt")
        for name in ("run_manifest.json", "weights_manifest.json", "radonpy_target_config.json", "run_complete.json"):
            copy_if_exists(source_seed / name, destination_seed / name)
        stage0 = source_seed / "stage0_data"
        if stage0.is_dir():
            shutil.copytree(stage0, destination_seed / "stage0_data", dirs_exist_ok=True)
        index_rows.append(
            {
                "experiment_key": key,
                "seed": seed,
                "checkpoint": str((destination_seed / "best_model.pt").relative_to(destination_dir.parents[1])),
                "source": os.path.relpath(weight, destination_dir.parents[2]),
                "size_bytes": weight.stat().st_size,
            }
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def create_training_summary(per_seed: Path, destination: Path, exp_key: str) -> None:
    rows = read_csv_rows(per_seed)
    summary: dict[str, object] = {"experiment_key": exp_key, "n_completed_seeds": len({row.get("seed") for row in rows})}
    for column in ("best_valid_score", "best_epoch", "elapsed_min"):
        values = []
        for row in rows:
            try:
                value = float(row.get(column, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            mean = sum(values) / len(values)
            summary[f"{column}_mean"] = mean
            summary[f"{column}_std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def copy_two_table_result(source_dir: Path, destination_dir: Path, summary_name: str, per_seed_name: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copy_if_exists(source_dir / summary_name, destination_dir / "summary.csv")
    copy_if_exists(source_dir / per_seed_name, destination_dir / "per_model_seed.csv")


def curate_artifacts(source_root: Path, project_root: Path) -> None:
    outputs = source_root / "RepoRT_PolyOmic" / "outputs"
    checkpoint_rows: list[dict] = []
    for exp_id in [f"E{index}" for index in range(1, 10)]:
        source_dir = outputs / exp_id
        destination_dir = project_root / "checkpoints" / "experiments" / exp_id
        copy_checkpoint_set(source_dir, destination_dir, exp_id, checkpoint_rows)
        result_dir = project_root / "result" / "training" / exp_id
        copy_if_exists(source_dir / "weights_run_summary_all_seeds.csv", result_dir / "per_model_seed.csv")
        if (source_dir / "weights_summary.csv").exists():
            copy_if_exists(source_dir / "weights_summary.csv", result_dir / "summary.csv")
        else:
            create_training_summary(source_dir / "weights_run_summary_all_seeds.csv", result_dir / "summary.csv", exp_id)

    scaling_sources = {
        "E2_zero": outputs / "E2",
        "E8_p01": outputs / "E8_scaling_log5" / "point_01_pct_000p050000",
        "E8_p02": outputs / "E8_scaling_log5" / "point_02_pct_000p334370",
        "E8_p03": outputs / "E8_scaling_log5" / "point_03_pct_002p236068",
        "E8_p04": outputs / "E8_scaling_log5" / "point_04_pct_014p953488",
        "E8_full": outputs / "E8",
    }
    for key, source_dir in scaling_sources.items():
        destination_dir = project_root / "checkpoints" / "scaling" / key
        copy_checkpoint_set(source_dir, destination_dir, key, checkpoint_rows)
        result_dir = project_root / "result" / "scaling_training" / key
        copy_if_exists(source_dir / "weights_run_summary_all_seeds.csv", result_dir / "per_model_seed.csv")
        if (source_dir / "weights_summary.csv").exists():
            copy_if_exists(source_dir / "weights_summary.csv", result_dir / "summary.csv")
        else:
            create_training_summary(source_dir / "weights_run_summary_all_seeds.csv", result_dir / "summary.csv", key)

    index_path = project_root / "checkpoints" / "checkpoint_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    copy_two_table_result(
        outputs / "evaluation_E1_E9_OOD_kshot_lowoverlap_shimadzu",
        project_root / "result" / "external_ood",
        "summary_by_ood_tier_experiment_K.csv",
        "ood_kshot_metrics_all_runs.csv",
    )
    copy_two_table_result(
        outputs / "evaluation_E1_E9_internal_heldout_OOD_kshot",
        project_root / "result" / "internal_ood",
        "summary_by_internal_ood_dataset_experiment_K.csv",
        "internal_ood_kshot_metrics_all_runs.csv",
    )
    copy_two_table_result(
        outputs / "evaluation_E2_to_E8_scaling_internal_test",
        project_root / "result" / "scaling_internal_test",
        "internal_test_scaling_summary.csv",
        "internal_test_seed_metrics.csv",
    )
    copy_two_table_result(
        outputs / "unirt_ginkan_new_report_ood_kshot_frozen_backbone",
        project_root / "result" / "unirt_external_ood",
        "unirt_ood_summary_by_method_k.csv",
        "unirt_ginkan_ood_kshot_metrics_all_runs.csv",
    )
    original_internal_result = project_root / "result" / "internal_test_original"
    copy_if_exists(
        outputs / "evaluation_E1_E9" / "aggregate" / "summary_by_experiment_split.csv",
        original_internal_result / "summary.csv",
    )
    copy_if_exists(
        outputs / "evaluation_E1_E9" / "all_evaluation_metrics.csv",
        original_internal_result / "per_model_seed.csv",
    )

    graphormer_results = source_root / "graphormer_rt_scratch179_final_10seeds"
    copy_if_exists(
        graphormer_results / "internal_test_10seeds_final_aggregate.csv",
        project_root / "result" / "graphormer_rt" / "internal_test" / "summary.csv",
    )
    copy_if_exists(
        graphormer_results / "internal_test_10seeds_final.csv",
        project_root / "result" / "graphormer_rt" / "internal_test" / "per_model_seed.csv",
    )

    data_sources = [
        source_root / "Data" / "shimazu_data_20241223_integrated.csv",
        source_root / "Data" / "shimazu_data_20251120_integraded.csv",
        source_root / "Data" / "RadonPy_20260611" / "RadonPySM_checkeq_masked.csv",
    ]
    data_manifest = []
    for source in data_sources:
        relative = source.relative_to(source_root / "Data")
        destination = project_root / "data" / "raw" / relative
        link_checkpoint(source, destination)
        data_manifest.append(
            {
                "local_path": str(destination.relative_to(project_root)),
                "source_path": os.path.relpath(source, project_root),
                "size_bytes": source.stat().st_size,
            }
        )
    with (project_root / "data" / "source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data_manifest[0]))
        writer.writeheader()
        writer.writerows(data_manifest)


def build_entrypoint(project_root: Path) -> None:
    write_text(
        project_root / "script" / "train.py",
        '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one E1-E9 or E8 scaling experiment.")
    parser.add_argument("experiment", help="E1-E9 or E8_p01/E8_p02/E8_p03/E8_p04")
    args = parser.parse_args()
    run_training(args.experiment)


if __name__ == "__main__":
    main()
''',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    project_root = args.project_root.resolve()
    build_configs(project_root)
    build_engine(source_root, project_root)
    build_data_scripts(source_root, project_root)
    build_evaluation_scripts(source_root, project_root)
    build_unirt_model_and_eval(source_root, project_root)
    build_entrypoint(project_root)
    curate_artifacts(source_root, project_root)
    print(f"Refactored project created at: {project_root}")


if __name__ == "__main__":
    main()
