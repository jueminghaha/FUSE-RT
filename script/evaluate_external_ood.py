from __future__ import annotations

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

# ============================================================
# Cell B1. Correct OOD K-shot configuration
# ============================================================
from datetime import datetime
import zlib
import shutil


# Main output. Keep separate from the earlier zero-shot-only OOD audit.
EVAL_ROOT = PROJECT_ROOT / 'result' / '_runs' / 'external_ood'
EVAL_ROOT.mkdir(parents=True, exist_ok=True)


# Saved weights are expected under outputs/E1/seed_xxxx, ..., outputs/E9/seed_xxxx.
WEIGHT_ROOT = PROJECT_ROOT / 'checkpoints' / 'experiments'
PREFER_CHECKPOINTS = ['best_model.pt', 'best_joint.pt', 'best_rt.pt', 'final_model.pt']

# OOD sets: use the selected low-overlap RepoRT methods and Shimadzu external data.
LOW_OVERLAP_METHOD_IDS = ['0391', '0390', '0437', '0420', '0419', '0411']
INCLUDE_LOW_OVERLAP_OOD = True
INCLUDE_SHIMADZU_OOD = True

# Main protocol requested:
# - R0 single-head models: zero-shot only.
# - R1/R2 multi-head models: K-shot only with a fresh OOD head on frozen backbone.
K_VALUES_MULTIHEAD = [5, 20, 100, 1000]
RUN_SINGLE_ZERO_SHOT_ONLY = True
RUN_MULTIHEAD_KSHOT_ONLY = True

# For K-shot repeats, the default is one support/query split per model seed.
# This keeps the run tractable and matches the 10 model split repeats.
PAIR_SUPPORT_SEED_WITH_MODEL_SEED = True
SUPPORT_RANDOM_SEEDS = SPLIT_SEEDS[:]  # used only when PAIR_SUPPORT_SEED_WITH_MODEL_SEED=False

# Fresh OOD head hyperparameters.
OOD_HEAD_EPOCHS = 300
OOD_HEAD_LR = 1e-3
OOD_HEAD_WEIGHT_DECAY = 1e-4
OOD_HEAD_HIDDEN = 64
BATCH_SIZE_OOD = 256
MIN_QUERY_ROWS = 2

# Resume / outputs.
RESUME_EXISTING = True
SAVE_QUERY_PREDICTIONS = True
SAVE_SUPPORT_INDEX_FILES = True
FAIL_FAST = False

# Low-overlap OOD cleaning.
FILTER_LOW_OVERLAP_BY_OUR179_MOLKEYS = True
DEDUP_LOW_OVERLAP_BY_MOL_KEY = True

# Shimadzu settings. 0.52 and 0.60 are both included; unavailable conditions are skipped with an audit table.
SHIMADZU_TARGET_TEMP = 50.0
SHIMADZU_TARGET_ACN_VALUES = [0.60]
SHIMADZU_TEMP_TOL = 1e-6
SHIMADZU_ACN_TOL = 1e-6
SHIMADZU_COLUMN_HINT = 'C18'
# Keep this False to match the earlier Shimadzu K-shot notebook protocol.
# Set True only if you explicitly want molecule-clean Shimadzu OOD against the 179 RepoRT pool.
FILTER_SHIMADZU_BY_OUR179_MOLKEYS = False
DEDUP_SHIMADZU_BY_MOL_KEY = False

# Experiment matrix.
EXPERIMENTS = [
    {'EXP_ID':'E1', 'order':1, 'MOLECULE_MODE':'M0_no_aux',   'RT_ARCHITECTURE':'R0_device_single',  'EXP_NAME':'M0 no PolyOmics × R0 device encoder + single head',      'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'single', 'JOINT_MULTITASK':False, 'RADONPY_PERCENT':0,   'expected_best':'best_rt.pt',    'name_short':'M0-R0'},
    {'EXP_ID':'E2', 'order':2, 'MOLECULE_MODE':'M0_no_aux',   'RT_ARCHITECTURE':'R1_device_multi',   'EXP_NAME':'M0 no PolyOmics × R1 device encoder + multitask heads',  'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':0,   'expected_best':'best_rt.pt',    'name_short':'M0-R1'},
    {'EXP_ID':'E3', 'order':3, 'MOLECULE_MODE':'M0_no_aux',   'RT_ARCHITECTURE':'R2_nodevice_multi', 'EXP_NAME':'M0 no PolyOmics × R2 no-device + multitask heads',        'USE_DEVICE_METADATA':False, 'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':0,   'expected_best':'best_rt.pt',    'name_short':'M0-R2'},
    {'EXP_ID':'E4', 'order':4, 'MOLECULE_MODE':'M1_pretrain', 'RT_ARCHITECTURE':'R0_device_single',  'EXP_NAME':'M1 RadonPy pretrain × R0 device encoder + single head',    'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'single', 'JOINT_MULTITASK':False, 'RADONPY_PERCENT':100, 'expected_best':'best_rt.pt',    'name_short':'M1-R0'},
    {'EXP_ID':'E5', 'order':5, 'MOLECULE_MODE':'M1_pretrain', 'RT_ARCHITECTURE':'R1_device_multi',   'EXP_NAME':'M1 RadonPy pretrain × R1 device encoder + multitask heads', 'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':100, 'expected_best':'best_rt.pt',    'name_short':'M1-R1'},
    {'EXP_ID':'E6', 'order':6, 'MOLECULE_MODE':'M1_pretrain', 'RT_ARCHITECTURE':'R2_nodevice_multi', 'EXP_NAME':'M1 RadonPy pretrain × R2 no-device + multitask heads',      'USE_DEVICE_METADATA':False, 'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':100, 'expected_best':'best_rt.pt',    'name_short':'M1-R2'},
    {'EXP_ID':'E7', 'order':7, 'MOLECULE_MODE':'M2_joint',    'RT_ARCHITECTURE':'R0_device_single',  'EXP_NAME':'M2 joint multitask × R0 device encoder + single head',      'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'single', 'JOINT_MULTITASK':True,  'RADONPY_PERCENT':100, 'expected_best':'best_joint.pt', 'name_short':'M2-R0'},
    {'EXP_ID':'E8', 'order':8, 'MOLECULE_MODE':'M2_joint',    'RT_ARCHITECTURE':'R1_device_multi',   'EXP_NAME':'M2 joint multitask × R1 device encoder + multitask heads',  'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':True,  'RADONPY_PERCENT':100, 'expected_best':'best_joint.pt', 'name_short':'M2-R1'},
    {'EXP_ID':'E9', 'order':9, 'MOLECULE_MODE':'M2_joint',    'RT_ARCHITECTURE':'R2_nodevice_multi', 'EXP_NAME':'M2 joint multitask × R2 no-device + multitask heads',        'USE_DEVICE_METADATA':False, 'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':True,  'RADONPY_PERCENT':100, 'expected_best':'best_joint.pt', 'name_short':'M2-R2'},
]
EXP_BY_ID = {e['EXP_ID']: e for e in EXPERIMENTS}
EXP_SHORT = {e['EXP_ID']: e['name_short'] for e in EXPERIMENTS}
EXP_ORDER = {e['EXP_ID']: int(e['order']) for e in EXPERIMENTS}

# Discover the ten split seeds if possible; fall back to whatever the definitions cell set.
try:
    SPLIT_SEEDS = discover_split_seeds(10)
except Exception as exc:
    print('[WARN] Cannot rediscover split seeds; using current SPLIT_SEEDS:', exc)

print('EVAL_ROOT:', EVAL_ROOT)
print('WEIGHT_ROOT:', WEIGHT_ROOT)
print('SPLIT_SEEDS:', SPLIT_SEEDS)
print('LOW_OVERLAP_METHOD_IDS:', LOW_OVERLAP_METHOD_IDS)
print('K_VALUES_MULTIHEAD:', K_VALUES_MULTIHEAD)
print('single-head protocol: K=0 zero-shot only')
print('multi-head protocol: frozen backbone + fresh OOD head, K only =', K_VALUES_MULTIHEAD)
print('Shimadzu temp/acn:', SHIMADZU_TARGET_TEMP, SHIMADZU_TARGET_ACN_VALUES)



# ============================================================
# Cell B2. Shared utilities: paths, checkpoint restore, metrics
# ============================================================

def safe_filename(x: Any, max_len: int = 160) -> str:
    s = str(x)
    s = re.sub(r'[^A-Za-z0-9._=-]+', '_', s).strip('_')
    return s[:max_len] if len(s) > max_len else s


def stable_int_hash(x: Any) -> int:
    return int(zlib.crc32(str(x).encode('utf-8')) & 0xffffffff)


def first_existing_path_strict(paths, must_be_dir=None):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists() and (must_be_dir is None or p.is_dir() == must_be_dir):
            return p
    raise FileNotFoundError('None of these paths exist:\n' + '\n'.join(map(str, [p for p in paths if p is not None])))


def first_existing_file(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists() and p.is_file():
            return p
    return None


def _read_csv_flexible(path: Path, columns=None):
    """Read a CSV whose later appended rows may have extra columns.

    Older versions of this notebook appended zero-shot and K-shot metric rows with
    different schemas, so pandas' default parser can fail with e.g.
    "Expected 32 fields, saw 35".  This fallback uses the csv module and, when
    rows have extra trailing fields, assigns them to the missing expected columns.
    """
    import csv
    path = Path(path)
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=columns or [])
        rows = [r for r in reader]
    if not header:
        return pd.DataFrame(columns=columns or [])
    max_len = max([len(header)] + [len(r) for r in rows]) if rows else len(header)
    final_cols = list(header)
    if columns is not None:
        for c in columns:
            if c not in final_cols and len(final_cols) < max_len:
                final_cols.append(c)
    while len(final_cols) < max_len:
        final_cols.append(f'__extra_col_{len(final_cols) - len(header) + 1}')
    norm_rows = []
    for r in rows:
        if len(r) < len(final_cols):
            r = r + [''] * (len(final_cols) - len(r))
        elif len(r) > len(final_cols):
            r = r[:len(final_cols)-1] + [','.join(r[len(final_cols)-1:])]
        norm_rows.append(r)
    return pd.DataFrame(norm_rows, columns=final_cols)


def safe_read_csv(path, columns=None):
    path = Path(path)
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    except pd.errors.ParserError as exc:
        print(f'[WARN] ParserError while reading {path.name}: {exc}. Trying flexible CSV reader.')
        try:
            return _read_csv_flexible(path, columns=columns)
        except Exception as flex_exc:
            print(f'[WARN] Flexible CSV reader failed for {path.name}: {flex_exc}. Falling back to on_bad_lines="skip".')
            try:
                return pd.read_csv(path, engine='python', on_bad_lines='skip')
            except Exception:
                return pd.DataFrame(columns=columns or [])
    except Exception as exc:
        print(f'[WARN] Could not read {path.name}: {exc}')
        return pd.DataFrame(columns=columns or [])


def _backup_path(path: Path, suffix: str) -> Path:
    path = Path(path)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return path.with_name(f'{path.stem}.{suffix}.{stamp}{path.suffix}')


def normalize_existing_csv_to_schema(path: Path, columns=None, label: str = 'csv'):
    """Rewrite an existing CSV into a stable schema if it is corrupt or has drifted.

    This prevents resume reads from crashing after earlier appends with variable
    column counts. The original file is copied to *.schema_backup.* before any
    rewrite.
    """
    path = Path(path)
    if (not path.exists()) or path.stat().st_size == 0:
        return
    # Detect parser problems with the strict parser.
    strict_ok = True
    try:
        strict_df = pd.read_csv(path)
    except Exception:
        strict_ok = False
        strict_df = None
    df = safe_read_csv(path, columns=columns)
    if df is None:
        df = pd.DataFrame(columns=columns or [])
    if columns is not None:
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        # Keep known columns first, but preserve any extra columns at the end.
        df = df[[c for c in columns if c in df.columns] + [c for c in df.columns if c not in columns]]
    need_rewrite = (not strict_ok)
    if strict_ok and columns is not None:
        current_cols = list(strict_df.columns)
        desired_prefix = [c for c in columns if c in df.columns]
        need_rewrite = current_cols[:len(desired_prefix)] != desired_prefix or len(current_cols) != len(df.columns)
    if need_rewrite:
        backup = _backup_path(path, 'schema_backup')
        try:
            shutil.copy2(path, backup)
        except Exception:
            backup = None
        df.to_csv(path, index=False)
        print(f'[FIX] Normalized {label} CSV schema: {path}' + (f' | backup={backup}' if backup else ''))


def append_df_to_csv(path: Path, df: pd.DataFrame, columns=None):
    """Append rows while keeping a parseable, stable CSV schema.

    If the output CSV already exists with a different header, it is normalized
    before appending. This is important because zero-shot rows and K-shot rows
    have different metadata fields unless a full schema is enforced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        return
    out = df.copy()
    if len(out) == 0:
        if not path.exists():
            pd.DataFrame(columns=columns or list(out.columns)).to_csv(path, index=False)
        return
    if columns is not None:
        for c in columns:
            if c not in out.columns:
                out[c] = np.nan
        out = out[[c for c in columns if c in out.columns] + [c for c in out.columns if c not in columns]]
    if (not path.exists()) or path.stat().st_size == 0:
        out.to_csv(path, index=False)
        return
    normalize_existing_csv_to_schema(path, columns=columns, label=path.stem)
    try:
        header_cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        # Last-resort quarantine and restart this CSV; resume will recompute missing keys.
        backup = _backup_path(path, 'unreadable_before_append')
        shutil.copy2(path, backup)
        out.to_csv(path, index=False)
        print(f'[FIX] Restarted unreadable CSV before append: {path} | backup={backup}')
        return
    all_cols = list(header_cols)
    for c in out.columns:
        if c not in all_cols:
            all_cols.append(c)
    if all_cols != header_cols:
        existing = safe_read_csv(path, columns=all_cols)
        for c in all_cols:
            if c not in existing.columns:
                existing[c] = np.nan
        backup = _backup_path(path, 'expanded_schema_backup')
        shutil.copy2(path, backup)
        existing[all_cols].to_csv(path, index=False)
        print(f'[FIX] Expanded CSV schema before append: {path} | backup={backup}')
    out = out.reindex(columns=all_cols)
    out.to_csv(path, mode='a', header=False, index=False)



# ---- Metric scalarization helpers -------------------------------------------------
# In the training notebooks, safe_mape_pct returns (mape_value, n_valid_rows).
# This OOD notebook needs the scalar MAPE value in metrics['mape_pct']; otherwise
# print formatting like {metrics["mape_pct"]:.3f} fails with tuple.__format__.
import ast

def metric_to_float(x, default=np.nan) -> float:
    """Convert floats, numpy scalars, tuples/lists, or tuple-like strings to a scalar float."""
    try:
        if isinstance(x, (tuple, list, np.ndarray)):
            if len(x) == 0:
                return float(default)
            return metric_to_float(x[0], default=default)
        if isinstance(x, str):
            s = x.strip()
            if s == '' or s.lower() in {'nan', 'none', 'null', '<na>'}:
                return float(default)
            if (s.startswith('(') and s.endswith(')')) or (s.startswith('[') and s.endswith(']')):
                try:
                    parsed = ast.literal_eval(s)
                    return metric_to_float(parsed, default=default)
                except Exception:
                    pass
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def metric_to_int(x, default=0) -> int:
    try:
        if isinstance(x, (tuple, list, np.ndarray)):
            if len(x) >= 2:
                return metric_to_int(x[1], default=default)
            return int(default)
        if isinstance(x, str):
            s = x.strip()
            if (s.startswith('(') and s.endswith(')')) or (s.startswith('[') and s.endswith(']')):
                try:
                    parsed = ast.literal_eval(s)
                    return metric_to_int(parsed, default=default)
                except Exception:
                    pass
        v = float(x)
        return int(v) if np.isfinite(v) else int(default)
    except Exception:
        return int(default)


def split_mape_return(x, default=np.nan, default_n=0):
    """Return (scalar_mape, n_valid_rows) from safe_mape_pct output."""
    if isinstance(x, (tuple, list, np.ndarray)):
        mape = metric_to_float(x[0], default=default) if len(x) >= 1 else float(default)
        n = metric_to_int(x[1], default=default_n) if len(x) >= 2 else int(default_n)
        return mape, n
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith('(') and s.endswith(')')) or (s.startswith('[') and s.endswith(']')):
            try:
                return split_mape_return(ast.literal_eval(s), default=default, default_n=default_n)
            except Exception:
                pass
    return metric_to_float(x, default=default), int(default_n)


def fmt_metric(x, digits: int = 3, default='nan') -> str:
    v = metric_to_float(x, default=np.nan)
    if not np.isfinite(v):
        return str(default)
    return f'{v:.{digits}f}'




# ---- Stable output schemas ---------------------------------------------------------
# Full metric schema for both protocols.  Using one schema avoids corrupt CSVs when
# R0 zero-shot rows and R1/R2 K-shot rows are appended to the same file.
OOD_METRIC_COLUMNS = [
    'mae_sec','median_ae_sec','rmse_sec','mape_pct','mape_n_rows','r2','spearman','n_query',
    'EXP_ID','EXP_NAME','name_short','order','MOLECULE_MODE','RT_ARCHITECTURE','RT_HEAD_TYPE',
    'USE_DEVICE_METADATA','JOINT_MULTITASK','seed','ood_tier','ood_task_id','ood_method_id',
    'K','repeat_seed','repeat_index','protocol','mode','n_total','n_support',
    'checkpoint_path','checkpoint_name','checkpoint_epoch','checkpoint_best_score','model_load_status',
    'support_indices_path','ood_head_epochs','ood_head_lr'
]
OOD_MISSING_COLUMNS = ['EXP_ID','EXP_NAME','seed','expected_run_dir','reason','preferred_names']
OOD_FAILED_COLUMNS = ['EXP_ID','EXP_NAME','seed','checkpoint_path','error_type','error']
OOD_SKIPPED_COLUMNS = ['EXP_ID','seed','ood_tier','ood_task_id','K','reason','n_total','min_query_rows']

def sanitize_metric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy tuple-valued MAPE cells to scalar mape_pct + mape_n_rows."""
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    if 'mape_pct' in out.columns:
        mape_vals = []
        mape_ns = []
        for x in out['mape_pct'].tolist():
            mv, mn = split_mape_return(x, default=np.nan, default_n=0)
            mape_vals.append(mv)
            mape_ns.append(mn)
        out['mape_pct'] = mape_vals
        if 'mape_n_rows' not in out.columns:
            out['mape_n_rows'] = mape_ns
        else:
            existing_n = pd.to_numeric(out['mape_n_rows'], errors='coerce')
            out['mape_n_rows'] = existing_n.where(existing_n.notna(), pd.Series(mape_ns, index=out.index)).fillna(0).astype(int)
    return out


def repair_prior_tuple_format_outputs(metrics_path: Path, failed_path: Path):
    """Repair CSVs created by the previous tuple-format bug, without rerunning completed rows."""
    metrics_path = Path(metrics_path)
    failed_path = Path(failed_path)
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        raw = safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS)
        fixed = sanitize_metric_dataframe(raw)
        if not raw.equals(fixed):
            backup = metrics_path.with_name(metrics_path.stem + '.before_tuple_fix_backup.csv')
            if not backup.exists():
                raw.to_csv(backup, index=False)
            fixed.to_csv(metrics_path, index=False)
            print(f'[FIX] Repaired tuple-valued MAPE entries in {metrics_path}; backup={backup}')
    if failed_path.exists() and failed_path.stat().st_size > 0:
        failed = safe_read_csv(failed_path, columns=OOD_FAILED_COLUMNS)
        if len(failed) and 'error' in failed.columns:
            mask_print_bug = failed['error'].astype(str).str.contains('unsupported format string passed to tuple.__format__', regex=False, na=False)
            if mask_print_bug.any():
                backup = failed_path.with_name(failed_path.stem + '.before_tuple_fix_backup.csv')
                if not backup.exists():
                    failed.to_csv(backup, index=False)
                failed.loc[~mask_print_bug].to_csv(failed_path, index=False)
                print(f'[FIX] Removed {int(mask_print_bug.sum())} print-only tuple-format failure rows from {failed_path}; backup={backup}')


def write_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def torch_load_compat_local(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_method_id(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    m = re.search(r'\d+', s)
    if m:
        return f'{int(m.group(0)):04d}'
    return s


def normalize_mol_key(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in {'nan', 'none', 'null'}:
        return np.nan
    if '-' in s:
        s = s.split('-')[0]
    return s.upper()


def find_col_fuzzy(df: pd.DataFrame, candidates: List[str]):
    norm = lambda s: re.sub(r'[^a-z0-9]+', '', str(s).lower())
    cmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in cmap:
            return cmap[key]
    for c in df.columns:
        ck = norm(c)
        for cand in candidates:
            q = norm(cand)
            if q and (q in ck or ck in q):
                return c
    return None


def read_table_loose(path: Path):
    for sep in ['\t', ',', ';', None]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, engine='python', on_bad_lines='skip')
            if df is not None and df.shape[1] >= 2:
                return df
        except Exception:
            pass
    raise ValueError(f'Could not read table: {path}')


def get_reference_split_dir() -> Path:
    for seed in SPLIT_SEEDS:
        try:
            p = resolve_split_dir(int(seed))
            if all((p / fn).exists() for fn in ['train.csv', 'valid.csv', 'internal_test.csv']):
                return p
        except Exception:
            pass
    # Last resort: search under SPLIT_ROOT.
    hits = []
    for p in sorted(Path(SPLIT_ROOT).glob('**/train.csv')):
        d = p.parent
        if (d / 'valid.csv').exists() and (d / 'internal_test.csv').exists():
            hits.append(d)
    if hits:
        return hits[0]
    raise FileNotFoundError('Cannot locate any split directory with train/valid/internal_test.csv')


def load_our179_molkey_pool() -> Dict[str, Any]:
    split_dir = get_reference_split_dir()
    frames = []
    for split_name, fn in [('train','train.csv'), ('valid','valid.csv'), ('internal_test','internal_test.csv')]:
        df = pd.read_csv(split_dir / fn, dtype=str)
        df['split'] = split_name
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    method_col = find_col_fuzzy(pool, ['dir', 'method', 'method_id', 'dataset_id', 'task_id'])
    mol_col = find_col_fuzzy(pool, ['mol_key', 'inchikey', 'inchi_key', 'inchikey.std'])
    if method_col is None or mol_col is None:
        raise ValueError(f'Cannot find method/mol_key columns in split files. Columns={list(pool.columns)}')
    pool['method_id'] = pool[method_col].map(normalize_method_id)
    pool['mol_key_norm'] = pool[mol_col].map(normalize_mol_key)
    out = {
        'split_dir': split_dir,
        'pool_df': pool,
        'all_molkeys': set(pool['mol_key_norm'].dropna()),
        'train_molkeys': set(pool.loc[pool['split']=='train', 'mol_key_norm'].dropna()),
        'valid_molkeys': set(pool.loc[pool['split']=='valid', 'mol_key_norm'].dropna()),
        'test_molkeys': set(pool.loc[pool['split']=='internal_test', 'mol_key_norm'].dropna()),
        'methods': sorted(pool['method_id'].dropna().unique().tolist()),
    }
    print('Reference split dir:', split_dir)
    print('our179 methods:', len(out['methods']), 'unique mol_keys:', len(out['all_molkeys']))
    return out


def configure_experiment_globals(meta: Dict[str, Any]):
    engine.configure_experiment(str(meta['EXP_ID']))
    global EXP_ID, EXP_NAME, MOLECULE_MODE, RT_ARCHITECTURE, USE_DEVICE_METADATA, RT_HEAD_TYPE
    global JOINT_MULTITASK, RADONPY_PERCENT, RADONPY_FRACTION, BEST_CKPT_NAME, ROOT_OUT, FRACTION_OUT, OUT_DIR
    EXP_ID = meta['EXP_ID']
    EXP_NAME = meta['EXP_NAME']
    MOLECULE_MODE = meta['MOLECULE_MODE']
    RT_ARCHITECTURE = meta['RT_ARCHITECTURE']
    USE_DEVICE_METADATA = bool(meta['USE_DEVICE_METADATA'])
    RT_HEAD_TYPE = meta['RT_HEAD_TYPE']
    JOINT_MULTITASK = bool(meta['JOINT_MULTITASK'])
    RADONPY_PERCENT = int(meta.get('RADONPY_PERCENT', 100 if MOLECULE_MODE != 'M0_no_aux' else 0))
    RADONPY_FRACTION = RADONPY_PERCENT / 100.0
    BEST_CKPT_NAME = meta.get('expected_best', 'best_joint.pt' if JOINT_MULTITASK else 'best_rt.pt')
    ROOT_OUT = WEIGHT_ROOT / EXP_ID
    FRACTION_OUT = ROOT_OUT
    OUT_DIR = FRACTION_OUT / '_shared'


def find_checkpoint_path(exp_id: str, seed: int, prefer: Optional[List[str]] = None) -> Optional[Path]:
    prefer = prefer or PREFER_CHECKPOINTS
    run_dir = WEIGHT_ROOT / exp_id / f'seed_{seed}'
    for name in prefer:
        p = run_dir / name
        if p.exists():
            return p
    for p in sorted(run_dir.glob('*.pt')):
        if p.name.startswith('best'):
            return p
    return None


def infer_radon_target_count_from_state(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith('radon_heads.4.weight') and hasattr(value, 'shape') and len(value.shape) >= 1:
            return int(value.shape[0])
    return 0


def checkpoint_radon_target_count(ckpt: Dict[str, Any]) -> int:
    targets = ckpt.get('radon_targets', [])
    if targets is None:
        targets = []
    if isinstance(targets, (list, tuple)):
        n = len(targets)
    else:
        try:
            n = int(targets)
        except Exception:
            n = 0
    return max(n, infer_radon_target_count_from_state(ckpt.get('model', {})))


def restore_metadata_globals_from_checkpoint(ckpt: Dict[str, Any]):
    global y_mean, y_std, cat_vocabs, method_vocab, CFG
    if 'y_mean' not in ckpt or 'y_std' not in ckpt:
        raise RuntimeError('Checkpoint missing y_mean/y_std; cannot standardize OOD labels correctly.')
    y_mean = float(ckpt['y_mean'])
    y_std = float(ckpt['y_std'])
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0
    cat_vocabs = ckpt.get('cat_vocabs') or {'column_cat0': {'<UNK>': 0, 'NA': 1}, 'brand_cat0': {'<UNK>': 0, 'NA': 1}, 'solvent_cat0': {'<UNK>': 0, 'NA': 1}, 'solvent_cat1': {'<UNK>': 0, 'NA': 1}}
    method_vocab = {str(k).zfill(4): int(v) for k, v in (ckpt.get('method_vocab') or {}).items()}
    if ckpt.get('cfg'):
        CFG.update(dict(ckpt['cfg']))


def build_model_from_checkpoint(ckpt: Dict[str, Any], meta: Dict[str, Any]) -> nn.Module:
    restore_metadata_globals_from_checkpoint(ckpt)
    cfg = dict(ckpt.get('cfg', CFG))
    head_type = str(ckpt.get('rt_head_type', meta['RT_HEAD_TYPE']))
    use_device = bool(ckpt.get('USE_DEVICE_METADATA', ckpt.get('use_device_metadata', meta['USE_DEVICE_METADATA'])))
    n_methods = len(ckpt.get('method_vocab', method_vocab))
    n_radon = checkpoint_radon_target_count(ckpt)
    model = GraphEnvRTModel(
        cat_vocabs=ckpt.get('cat_vocabs', cat_vocabs),
        radon_targets=n_radon,
        cfg=cfg,
        head_type=head_type,
        num_methods=n_methods,
        use_device_metadata=use_device,
    ).to(DEVICE)
    try:
        model.load_state_dict(ckpt['model'], strict=True)
        model._load_status = 'strict'
        model._missing_keys = []
        model._unexpected_keys = []
    except RuntimeError as exc:
        print('[WARN] strict=True load failed; falling back to strict=False:', exc)
        load_result = model.load_state_dict(ckpt['model'], strict=False)
        model._load_status = 'non_strict'
        model._missing_keys = list(load_result.missing_keys)
        model._unexpected_keys = list(load_result.unexpected_keys)
    model.eval()
    return model


def safe_spearman(y_true, y_pred):
    try:
        s = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        return float(s) if pd.notna(s) else np.nan
    except Exception:
        return np.nan


def compute_metric_dict(true_min, pred_min) -> Dict[str, Any]:
    true_min = np.asarray(true_min, dtype=float)
    pred_min = np.asarray(pred_min, dtype=float)
    m = np.isfinite(true_min) & np.isfinite(pred_min)
    true_min = true_min[m]
    pred_min = pred_min[m]
    err_sec = (pred_min - true_min) * 60.0
    abs_err_sec = np.abs(err_sec)
    mape_value, mape_n = split_mape_return(safe_mape_pct(true_min, pred_min), default=np.nan, default_n=0)
    return {
        'n_query': int(len(true_min)),
        'mae_sec': float(np.mean(abs_err_sec)) if len(abs_err_sec) else np.nan,
        'median_ae_sec': float(np.median(abs_err_sec)) if len(abs_err_sec) else np.nan,
        'rmse_sec': float(np.sqrt(np.mean(err_sec ** 2))) if len(err_sec) else np.nan,
        'mape_pct': mape_value,
        'mape_n_rows': int(mape_n),
        'r2': safe_r2(true_min, pred_min),
        'spearman': safe_spearman(true_min, pred_min),
    }



# ============================================================
# Cell B3. Load selected Low-overlap RepoRT OOD and Shimadzu OOD rows
# ============================================================
SMILES_CANDIDATES = [
    'smiles', 'SMILES', 'smiles.std', 'smiles_std', 'smiles.canonical',
    'smiles_list_canonical', 'canonical_smiles', 'smiles_canonical',
]
INCHIKEY_CANDIDATES = [
    'mol_key', 'inchikey', 'InChIKey', 'inchikey.std', 'inchikey_std',
    'inchi_key', 'inchi.key', 'std.inchikey',
]
RT_CANDIDATES = [
    'rt', 'RT', 'rt.min', 'rt_min', 'RT_min', 'retention_time', 'retention.time',
    'retention_time_min', 'retention.time.min', 'retention time', 'RT(min)',
]

our179 = load_our179_molkey_pool()


def choose_report_root_for_methods(method_ids: List[str]) -> Path:
    candidates = [
        WORKDIR / '../outputs/report_latest_overlap_audit/RepoRT_latest',
        WORKDIR / 'outputs/report_latest_overlap_audit/RepoRT_latest',
        WORKDIR / 'RepoRT_latest',
        WORKDIR / '../RepoRT_latest',
        WORKDIR / 'RepoRT',
        WORKDIR / '../RepoRT',
        REPORT_ROOT,
    ]
    scored = []
    for root in candidates:
        if root is None:
            continue
        root = Path(root)
        proc = root / 'processed_data'
        if not proc.exists():
            continue
        count = sum((proc / str(mid).zfill(4)).exists() for mid in method_ids)
        scored.append((count, root))
    if not scored:
        raise FileNotFoundError('Cannot locate a RepoRT/RepoRT_latest processed_data folder for low-overlap OOD methods.')
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    best_count, best_root = scored[0]
    print('Low-overlap RepoRT root:', best_root, '| matched methods:', best_count, '/', len(method_ids))
    if best_count == 0:
        raise FileNotFoundError(f'No selected low-overlap methods found in candidate RepoRT roots: {method_ids}')
    return best_root


def find_rtdata_file(method_id: str) -> Path:
    method_id = normalize_method_id(method_id)
    method_dir = PROCESSED_DIR / method_id
    if not method_dir.exists():
        raise FileNotFoundError(method_dir)
    candidates = [
        method_dir / f'{method_id}_rtdata_canonical_success.tsv',
        method_dir / f'{method_id}_rtdata.tsv',
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted(method_dir.glob('*rtdata*canonical*success*.tsv')) + sorted(method_dir.glob('*rtdata*.tsv')) + sorted(method_dir.glob('*rtdata*.csv'))
    if hits:
        return hits[0]
    raise FileNotFoundError(f'No rtdata file found for {method_id} in {method_dir}')


def smiles_to_key_safe(smi):
    try:
        return normalize_mol_key(smiles_to_mol_key(smi))
    except Exception:
        return np.nan


def get_method_column_name(method_id):
    try:
        rec = compute_method_properties(method_id)
        meta = get_meta_row(method_id)
        val = rec.get('column_name', get_column_name(method_id, meta))
        if val is not None and str(val).strip() and str(val).lower() != 'nan':
            return str(val).strip()
    except Exception:
        pass
    return np.nan


def load_low_overlap_method_df(method_id: str):
    method_id = normalize_method_id(method_id)
    rt_path = find_rtdata_file(method_id)
    raw = read_table_loose(rt_path)
    smiles_col = find_col_fuzzy(raw, SMILES_CANDIDATES)
    inchikey_col = find_col_fuzzy(raw, INCHIKEY_CANDIDATES)
    rt_col = find_col_fuzzy(raw, RT_CANDIDATES)
    if rt_col is None:
        raise ValueError(f'No RT column found for {method_id}. Columns={list(raw.columns)}')
    if smiles_col is None:
        raise ValueError(f'{method_id} has no SMILES column; graph inference needs SMILES. Columns={list(raw.columns)}')
    df = raw.copy()
    df['rt'] = pd.to_numeric(df[rt_col], errors='coerce')
    df['smiles'] = df[smiles_col].astype(str).str.strip()
    if inchikey_col is not None:
        df['mol_key'] = df[inchikey_col].map(normalize_mol_key)
    else:
        df['mol_key'] = df['smiles'].map(smiles_to_key_safe)
    df = df.dropna(subset=['rt', 'smiles', 'mol_key']).copy()
    df = df[~df['smiles'].astype(str).str.lower().isin(['', 'nan', 'none'])]
    df['appears_in_our179_all'] = df['mol_key'].isin(our179['all_molkeys'])
    df['appears_in_our179_train'] = df['mol_key'].isin(our179['train_molkeys'])
    df['appears_in_our179_valid'] = df['mol_key'].isin(our179['valid_molkeys'])
    df['appears_in_our179_internal_test'] = df['mol_key'].isin(our179['test_molkeys'])
    n_before_filter = len(df)
    if FILTER_LOW_OVERLAP_BY_OUR179_MOLKEYS:
        df = df[~df['appears_in_our179_all']].copy()
    if DEDUP_LOW_OVERLAP_BY_MOL_KEY:
        df = df.drop_duplicates('mol_key', keep='first').copy()
    df = df.reset_index(drop=True)
    df['dir'] = method_id
    df['source_method_id'] = method_id
    df['ood_tier'] = 'new_report_low_overlap_ood'
    df['ood_method_id'] = method_id
    df['ood_task_id'] = method_id
    df['source'] = 'RepoRT_latest'
    df['row_id'] = [f'{method_id}_LOWOVERLAP_OOD_{i:06d}' for i in range(len(df))]
    df['split'] = 'new_report_low_overlap_ood'
    df['clean_radonpy_overlap'] = False
    df['non_radonpy_overlap'] = False
    df = attach_env_features(df)
    column_name = get_method_column_name(method_id)
    df['column.name'] = column_name
    df['rtdata_file'] = str(rt_path)
    summary = {
        'ood_tier': 'new_report_low_overlap_ood',
        'method_id': method_id,
        'rtdata_file': str(rt_path),
        'rt_col': rt_col,
        'smiles_col': smiles_col,
        'inchikey_col': inchikey_col,
        'column.name': column_name,
        'rows_before_our179_filter': int(n_before_filter),
        'rows_after_filter_dedup': int(len(df)),
        'unique_mol_keys_after_filter_dedup': int(df['mol_key'].nunique()),
        'FILTER_LOW_OVERLAP_BY_OUR179_MOLKEYS': bool(FILTER_LOW_OVERLAP_BY_OUR179_MOLKEYS),
        'DEDUP_LOW_OVERLAP_BY_MOL_KEY': bool(DEDUP_LOW_OVERLAP_BY_MOL_KEY),
    }
    return df, summary


def load_low_overlap_ood_all():
    global REPORT_ROOT, PROCESSED_DIR, ENV_FEATURE_CACHE, df_meta
    if not INCLUDE_LOW_OVERLAP_OOD:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected_root = choose_report_root_for_methods(LOW_OVERLAP_METHOD_IDS)
    REPORT_ROOT = selected_root
    PROCESSED_DIR = selected_root / 'processed_data'
    ENV_FEATURE_CACHE = {}
    try:
        df_meta = read_report_global_metadata(META_PATH)
    except Exception as e:
        print('[WARN] Could not reload global metadata for low-overlap RepoRT; per-method metadata fallback will be used:', repr(e))
        df_meta = pd.DataFrame()
    frames, summaries, failed = [], [], []
    for mid in LOW_OVERLAP_METHOD_IDS:
        try:
            df_mid, summary_mid = load_low_overlap_method_df(mid)
            frames.append(df_mid)
            summaries.append(summary_mid)
            print(f"Loaded low-overlap {mid}: rows={len(df_mid):,}, mols={df_mid['mol_key'].nunique():,}, column={summary_mid.get('column.name')}")
        except Exception as exc:
            failed.append({'ood_tier':'new_report_low_overlap_ood', 'method_id': str(mid).zfill(4), 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'FAILED low-overlap {mid}: {repr(exc)}')
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out_df, pd.DataFrame(summaries), pd.DataFrame(failed)


def normalize_acn_ratio(x):
    v = pd.to_numeric(x, errors='coerce')
    if pd.isna(v):
        return np.nan
    v = float(v)
    if v > 1.5:
        v = v / 100.0
    return v


def bool_close_to_any(x, values, tol=1e-6):
    if not np.isfinite(x):
        return False
    return any(abs(float(x) - float(v)) <= tol for v in values)


def build_manual_shimadzu_env(temp_c, acn_ratio, column_name=SHIMADZU_COLUMN_HINT):
    temp_scaled = float(temp_c) / 100.0 if np.isfinite(float(temp_c)) else -1.0
    acn_scaled = float(acn_ratio) if np.isfinite(float(acn_ratio)) else -1.0
    return {
        'column_cont': [-1.0, -1.0, -1.0, temp_scaled, -1.0],
        'column_cat': [normalize_phase_type(column_name)],
        'brand_cat': ['Shimadzu'],
        'solvent_cont': [acn_scaled, -1.0, -1.0],
        'solvent_cat': ['h2o', 'acn'],
        'gradient_cont': [0.0, acn_scaled, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        'operation_cont': [-1.0],
        'family': 'RP',
        'AB_switched': False,
    }


def find_data_dir_for_shimadzu() -> Path:
    candidates = [
        DATA_DIR if 'DATA_DIR' in globals() else None,
        WORKDIR / 'Data',
        WORKDIR / '../Data',
        WORKDIR / '../../Data',
        WORKDIR / '../../../Data',
    ]
    for d in candidates:
        if d is not None and Path(d).exists():
            return Path(d)
    raise FileNotFoundError('Cannot find Data directory for Shimadzu raw CSV files.')


def load_shimadzu_ood_all():
    if not INCLUDE_SHIMADZU_OOD:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data_dir_local = find_data_dir_for_shimadzu()
    print('Shimadzu DATA_DIR:', data_dir_local)
    specs = {
        'Shimadzu_20241223': {
            'raw_candidates': [
                data_dir_local / 'shimazu_data_20241223_integrated.csv',
                data_dir_local / 'shimadzu_data_20241223_integrated.csv',
            ],
            'smiles_col': 'smiles_canonical',
            'target_col': 'RT_min_subT0',
            'temp_col': 'temp(C)',
            'acn_col': 'ACN_ratio',
        },
        'Shimadzu_20251120': {
            'raw_candidates': [
                data_dir_local / 'shimazu_data_20251120_integraded.csv',
                data_dir_local / 'shimazu_data_20251120_integrated.csv',
                data_dir_local / 'shimadzu_data_20251120_integrated.csv',
            ],
            'smiles_col': 'smiles_canonical',
            'target_col': 'RT_min_subT0',
            'temp_col': 'temp(C)',
            'acn_col': 'ACN_ratio',
        },
    }
    frames, summaries, failed = [], [], []
    for source, spec in specs.items():
        try:
            raw_path = first_existing_file(spec['raw_candidates'])
            if raw_path is None:
                raise FileNotFoundError(f'Missing Shimadzu raw file for {source}. Tried={spec["raw_candidates"]}')
            raw = pd.read_csv(raw_path)
            smiles_col, y_col, temp_col, acn_col = spec['smiles_col'], spec['target_col'], spec['temp_col'], spec['acn_col']
            missing_cols = [c for c in [smiles_col, y_col, temp_col, acn_col] if c not in raw.columns]
            if missing_cols:
                raise ValueError(f'{raw_path} missing columns={missing_cols}. Available={list(raw.columns)}')
            df = raw.copy()
            df['_temp_num'] = pd.to_numeric(df[temp_col], errors='coerce')
            df['_acn_norm'] = df[acn_col].map(normalize_acn_ratio)
            df['_rt_num'] = pd.to_numeric(df[y_col], errors='coerce')
            df['_smiles'] = df[smiles_col].astype(str).str.strip()
            valid = (
                df['_smiles'].notna()
                & ~df['_smiles'].str.lower().isin(['', 'nan', 'none'])
                & df['_rt_num'].notna()
                & df['_temp_num'].apply(lambda x: np.isfinite(x) and abs(float(x) - SHIMADZU_TARGET_TEMP) <= SHIMADZU_TEMP_TOL)
                & df['_acn_norm'].apply(lambda x: bool_close_to_any(x, SHIMADZU_TARGET_ACN_VALUES, SHIMADZU_ACN_TOL))
            )
            selected = df.loc[valid].copy()
            if len(selected) == 0:
                avail = (df.dropna(subset=['_temp_num', '_acn_norm']).groupby(['_temp_num', '_acn_norm']).size().rename('rows').reset_index().sort_values('rows', ascending=False).head(20))
                summaries.append({'ood_tier':'shimadzu_external_ood', 'source':source, 'raw_file':str(raw_path), 'status':'no_rows_for_requested_condition', 'n_rows_after_filter_dedup':0, 'available_conditions_preview': avail.to_dict('records')})
                print(f'WARNING: {source} has 0 rows for temp={SHIMADZU_TARGET_TEMP} and ACN={SHIMADZU_TARGET_ACN_VALUES}.')
                display(avail)
                continue
            selected['mol_key'] = selected['_smiles'].map(smiles_to_key_safe)
            selected = selected.dropna(subset=['mol_key']).copy()
            selected['appears_in_our179_all'] = selected['mol_key'].isin(our179['all_molkeys'])
            if FILTER_SHIMADZU_BY_OUR179_MOLKEYS:
                selected = selected[~selected['appears_in_our179_all']].copy()
            if DEDUP_SHIMADZU_BY_MOL_KEY:
                selected = selected.drop_duplicates('mol_key', keep='first').copy()
            selected = selected.reset_index(drop=True)
            for acn_val, g in selected.groupby('_acn_norm', sort=True):
                condition_label = f"T{int(SHIMADZU_TARGET_TEMP)}_ACN{str(float(acn_val)).replace('.', 'p')}"
                ood_task_id = f'{source}_{condition_label}'
                env = build_manual_shimadzu_env(SHIMADZU_TARGET_TEMP, float(acn_val))
                out = pd.DataFrame({
                    'split': 'shimadzu_external_ood',
                    'ood_tier': 'shimadzu_external_ood',
                    'source': source,
                    'condition_label': condition_label,
                    'ood_method_id': ood_task_id,
                    'ood_task_id': ood_task_id,
                    'dir': ood_task_id,
                    'row_id': [f'{ood_task_id}_{i:06d}' for i in range(len(g))],
                    'smiles': g['_smiles'].values,
                    'mol_key': g['mol_key'].values,
                    'rt': g['_rt_num'].astype(float).values,
                    'temp_C': float(SHIMADZU_TARGET_TEMP),
                    'ACN_ratio': float(acn_val),
                    'appears_in_our179_all': g['appears_in_our179_all'].values,
                    'clean_radonpy_overlap': False,
                    'non_radonpy_overlap': False,
                })
                for k, v in env.items():
                    out[k] = [v for _ in range(len(out))]
                frames.append(out)
                summaries.append({
                    'ood_tier': 'shimadzu_external_ood',
                    'source': source,
                    'condition_label': condition_label,
                    'ood_task_id': ood_task_id,
                    'raw_file': str(raw_path),
                    'temp_C': float(SHIMADZU_TARGET_TEMP),
                    'ACN_ratio': float(acn_val),
                    'n_rows_after_filter_dedup': int(len(out)),
                    'n_unique_mol_key': int(out['mol_key'].nunique()),
                    'n_overlap_our179_all': int(out['appears_in_our179_all'].sum()),
                    'FILTER_SHIMADZU_BY_OUR179_MOLKEYS': bool(FILTER_SHIMADZU_BY_OUR179_MOLKEYS),
                    'DEDUP_SHIMADZU_BY_MOL_KEY': bool(DEDUP_SHIMADZU_BY_MOL_KEY),
                    'status': 'ok',
                })
                print(f'Loaded Shimadzu {ood_task_id}: rows={len(out):,}, mols={out["mol_key"].nunique():,}')
        except Exception as exc:
            failed.append({'ood_tier':'shimadzu_external_ood', 'source':source, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'FAILED Shimadzu {source}: {repr(exc)}')
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out_df, pd.DataFrame(summaries), pd.DataFrame(failed)

low_df, low_summary, low_failed = load_low_overlap_ood_all()
shim_df, shim_summary, shim_failed = load_shimadzu_ood_all()

all_ood_frames = [x for x in [low_df, shim_df] if x is not None and len(x)]
if not all_ood_frames:
    raise RuntimeError('No OOD rows were loaded. Check RepoRT latest and Shimadzu data paths.')

ood_all_df = pd.concat(all_ood_frames, ignore_index=True)
ood_all_df['ood_tier'] = ood_all_df['ood_tier'].astype(str)
ood_all_df['ood_task_id'] = ood_all_df['ood_task_id'].astype(str)
ood_all_df['ood_method_id'] = ood_all_df['ood_method_id'].astype(str)

ood_summary = pd.concat([low_summary, shim_summary], ignore_index=True) if len(low_summary) or len(shim_summary) else pd.DataFrame()
ood_failed = pd.concat([low_failed, shim_failed], ignore_index=True) if len(low_failed) or len(shim_failed) else pd.DataFrame(columns=['ood_tier','error_type','error'])

ood_all_df.to_csv(EVAL_ROOT / 'ood_rows_master_lowoverlap_shimadzu.csv', index=False)
ood_summary.to_csv(EVAL_ROOT / 'ood_task_loading_summary.csv', index=False)
ood_failed.to_csv(EVAL_ROOT / 'ood_task_loading_failed.csv', index=False)

print('\n=== OOD task loading summary ===')
display(ood_summary)
print('\n=== OOD row counts by tier/task ===')
display(ood_all_df.groupby(['ood_tier', 'ood_task_id']).size().rename('n_rows').reset_index())
if len(ood_failed):
    print('\nFailed OOD loads:')
    display(ood_failed[['ood_tier', 'error_type', 'error']].head(50))

# Build a shared graph cache for all selected OOD molecules.
OOD_GRAPH_CACHE_PATH = EVAL_ROOT / '_shared' / 'pyg_graph_cache_lowoverlap_shimadzu.pt'
OOD_GRAPH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
ood_graph_cache = PrecomputedPyGGraphCache(OOD_GRAPH_CACHE_PATH)
ood_graph_cache.build(ood_all_df['smiles'], save=True)
print('OOD graph cache:', OOD_GRAPH_CACHE_PATH)



# ============================================================
# Cell B4. Frozen-backbone embedding extraction and OOD head helpers
# ============================================================

def make_ood_loader(df: pd.DataFrame, graph_cache, batch_size: Optional[int] = None, shuffle: bool = False) -> DataLoader:
    ds = RTGraphDataset(df.reset_index(drop=True), graph_cache)
    return DataLoader(ds, batch_size=int(batch_size or BATCH_SIZE_OOD), shuffle=shuffle, collate_fn=collate_rt, num_workers=0)


@torch.no_grad()
def predict_direct_single_head(model: nn.Module, loader: DataLoader):
    model.eval()
    rows = []
    true_all, pred_all = [], []
    for batch in loader:
        batch = move_batch_to_device(batch)
        pred_std, _ = model.forward_rt(batch)
        true_std = batch['y'].detach().cpu().numpy()
        pred_std_np = pred_std.detach().cpu().numpy()
        true_min = unscale_rt(true_std)
        pred_min = unscale_rt(pred_std_np)
        true_all.append(true_min)
        pred_all.append(pred_min)
        for i in range(len(true_min)):
            rows.append({
                'dir': str(batch['dir'][i]),
                'row_id': batch['row_id'][i],
                'smiles': batch['smiles'][i],
                'mol_key': batch['mol_key'][i],
                'true_min': float(true_min[i]),
                'pred_min': float(pred_min[i]),
                'true_sec': float(true_min[i] * 60.0),
                'pred_sec': float(pred_min[i] * 60.0),
                'abs_error_sec': float(abs(pred_min[i] - true_min[i]) * 60.0),
                'ape_pct': float(abs(pred_min[i] - true_min[i]) / max(abs(true_min[i]), 1e-8) * 100.0),
            })
    true_all = np.concatenate(true_all) if true_all else np.array([], dtype=float)
    pred_all = np.concatenate(pred_all) if pred_all else np.array([], dtype=float)
    return true_all, pred_all, pd.DataFrame(rows)


@torch.no_grad()
def extract_frozen_features(model: nn.Module, loader: DataLoader):
    """Return frozen features for fresh OOD-head training.

    For multi-head models, use rt_head.trunk(r_rt), matching the input space of dataset-specific heads.
    For other models, fall back to r_rt. The current protocol calls this only for R1/R2 multi-head models.
    """
    model.eval()
    Hs, ys, yraws, rows = [], [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch)
        z, extra = model.encode_representation(batch)
        if getattr(model, 'head_type', 'single') == 'multi' and hasattr(model.rt_head, 'trunk'):
            H = model.rt_head.trunk(z)
        else:
            H = z
        Hs.append(H.detach().cpu())
        ys.append(batch['y'].detach().cpu())
        yraws.append(batch['y_raw'].detach().cpu())
        for i in range(len(batch['y'])):
            rows.append({
                'dir': str(batch['dir'][i]),
                'row_id': batch['row_id'][i],
                'smiles': batch['smiles'][i],
                'mol_key': batch['mol_key'][i],
                'true_min': float(batch['y_raw'][i].detach().cpu().item()),
                'true_sec': float(batch['y_raw'][i].detach().cpu().item() * 60.0),
            })
    if not Hs:
        return torch.empty(0, int(CFG.get('d_model', 256))), torch.empty(0), torch.empty(0), pd.DataFrame(rows)
    return torch.cat(Hs, dim=0), torch.cat(ys, dim=0), torch.cat(yraws, dim=0), pd.DataFrame(rows)


class FewShotOODHead(nn.Module):
    def __init__(self, d_model=256, hidden=OOD_HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_fewshot_head(H_support: torch.Tensor, y_support: torch.Tensor, seed: int) -> nn.Module:
    set_all_seeds(int(seed))
    H_support = H_support.to(DEVICE)
    y_support = y_support.to(DEVICE)
    head = FewShotOODHead(d_model=int(H_support.shape[1]), hidden=OOD_HEAD_HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=OOD_HEAD_LR, weight_decay=OOD_HEAD_WEIGHT_DECAY)
    for ep in range(1, OOD_HEAD_EPOCHS + 1):
        head.train()
        pred = head(H_support)
        loss = 0.7 * F.smooth_l1_loss(pred, y_support, beta=0.5) + 0.3 * F.mse_loss(pred, y_support)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict_with_fewshot_head(head: nn.Module, H_query: torch.Tensor, y_query: torch.Tensor, meta_query: pd.DataFrame):
    H_query = H_query.to(DEVICE)
    pred_std = head(H_query).detach().cpu().numpy()
    true_std = y_query.detach().cpu().numpy()
    true_min = unscale_rt(true_std)
    pred_min = unscale_rt(pred_std)
    pred_df = meta_query.copy()
    pred_df['pred_min'] = pred_min
    pred_df['true_sec'] = true_min * 60.0
    pred_df['pred_sec'] = pred_min * 60.0
    pred_df['abs_error_sec'] = np.abs(pred_min - true_min) * 60.0
    pred_df['ape_pct'] = np.abs(pred_min - true_min) / np.maximum(np.abs(true_min), 1e-8) * 100.0
    return true_min, pred_min, pred_df


def feature_cache_path(exp_id: str, seed: int, ood_tier: str, ood_task_id: str) -> Path:
    return EVAL_ROOT / '_feature_cache' / exp_id / f'seed_{seed}' / safe_filename(ood_tier) / f'{safe_filename(ood_task_id)}_features.pt'


def get_or_build_features(model: nn.Module, task_df: pd.DataFrame, graph_cache, exp_id: str, seed: int, ood_tier: str, ood_task_id: str):
    path = feature_cache_path(exp_id, seed, ood_tier, ood_task_id)
    if RESUME_EXISTING and path.exists():
        obj = torch_load_compat_local(path, map_location='cpu')
        return obj['H_all'], obj['y_all'], obj['yraw_all'], obj['meta_all']
    loader = make_ood_loader(task_df, graph_cache, shuffle=False)
    H_all, y_all, yraw_all, meta_all = extract_frozen_features(model, loader)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'H_all': H_all, 'y_all': y_all, 'yraw_all': yraw_all, 'meta_all': meta_all}, path)
    return H_all, y_all, yraw_all, meta_all


def support_indices_for_task(n_total: int, K: int, exp_id: str, model_seed: int, repeat_seed: int, ood_task_id: str):
    if K <= 0:
        return np.array([], dtype=int), np.arange(n_total, dtype=int)
    seed_value = int(repeat_seed) + int(model_seed) * 1009 + EXP_ORDER.get(exp_id, 0) * 100003 + int(K) * 9176 + stable_int_hash(ood_task_id)
    rng = np.random.default_rng(seed_value)
    support_idx = np.sort(rng.choice(np.arange(n_total), size=int(K), replace=False))
    support_set = set(support_idx.tolist())
    query_idx = np.array([i for i in range(n_total) if i not in support_set], dtype=int)
    return support_idx, query_idx


def save_support_indices(run_dir: Path, ood_tier: str, ood_task_id: str, K: int, repeat_seed: int, support_idx: np.ndarray, query_idx: np.ndarray):
    if not SAVE_SUPPORT_INDEX_FILES:
        return ''
    out = pd.DataFrame({
        'index': np.concatenate([support_idx, query_idx]),
        'role': ['support'] * len(support_idx) + ['query'] * len(query_idx),
    })
    p = run_dir / 'support_indices' / safe_filename(ood_tier) / f'{safe_filename(ood_task_id)}_K{K}_repeat{repeat_seed}.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return str(p)



# ============================================================
# Cell B5. Run OOD protocol: R0 zero-shot; R1/R2 frozen-backbone K-shot
# ============================================================
metrics_path = EVAL_ROOT / 'ood_kshot_metrics_all_runs.csv'
pred_path = EVAL_ROOT / 'ood_kshot_predictions_query_all.csv'
missing_path = EVAL_ROOT / 'ood_kshot_missing_checkpoints.csv'
failed_path = EVAL_ROOT / 'ood_kshot_failed_runs.csv'
skipped_path = EVAL_ROOT / 'ood_kshot_skipped_tasks.csv'

# Repair outputs from the previous tuple-format print bug and schema-drift bug before resuming.
repair_prior_tuple_format_outputs(metrics_path, failed_path)
normalize_existing_csv_to_schema(metrics_path, OOD_METRIC_COLUMNS, label='OOD metrics')
normalize_existing_csv_to_schema(failed_path, OOD_FAILED_COLUMNS, label='OOD failed runs')
normalize_existing_csv_to_schema(missing_path, OOD_MISSING_COLUMNS, label='OOD missing checkpoints')
normalize_existing_csv_to_schema(skipped_path, OOD_SKIPPED_COLUMNS, label='OOD skipped tasks')

# Resume keys.
metric_columns = ['EXP_ID','seed','ood_tier','ood_task_id','K','repeat_seed','protocol']
existing_metrics = sanitize_metric_dataframe(safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS))
done_keys = set()
if RESUME_EXISTING and len(existing_metrics):
    for _, r in existing_metrics.iterrows():
        try:
            done_keys.add((str(r['EXP_ID']), int(r['seed']), str(r['ood_tier']), str(r['ood_task_id']), int(r['K']), int(r['repeat_seed']), str(r['protocol'])))
        except Exception:
            pass
    print('Resume enabled. Completed metric keys:', len(done_keys))

metric_rows, missing_rows, failed_rows, skipped_rows = [], [], [], []

ood_task_table = (
    ood_all_df[['ood_tier', 'ood_task_id', 'ood_method_id']]
    .drop_duplicates()
    .sort_values(['ood_tier', 'ood_task_id'])
    .reset_index(drop=True)
)
print('OOD tasks to evaluate:')
display(ood_task_table.assign(n_rows=ood_task_table.apply(lambda r: len(ood_all_df[(ood_all_df['ood_tier']==r['ood_tier']) & (ood_all_df['ood_task_id']==r['ood_task_id'])]), axis=1)))

for meta in EXPERIMENTS:
    configure_experiment_globals(meta)
    exp_id = meta['EXP_ID']
    exp_protocol = 'single_zero_shot' if meta['RT_HEAD_TYPE'] == 'single' else 'frozen_backbone_kshot_new_head'
    print('\n' + '#'*110)
    print('Experiment:', exp_id, meta['name_short'], '| protocol:', exp_protocol, '|', meta['EXP_NAME'])
    print('#'*110)

    for seed in SPLIT_SEEDS:
        ckpt_path = find_checkpoint_path(exp_id, int(seed))
        if ckpt_path is None:
            row = {'EXP_ID': exp_id, 'EXP_NAME': meta['EXP_NAME'], 'seed': int(seed), 'expected_run_dir': str(WEIGHT_ROOT / exp_id / f'seed_{seed}'), 'reason': 'checkpoint_not_found', 'preferred_names': ','.join(PREFER_CHECKPOINTS)}
            missing_rows.append(row)
            append_df_to_csv(missing_path, pd.DataFrame([row]), columns=OOD_MISSING_COLUMNS)
            print(f'[MISSING] {exp_id} seed={seed}: no checkpoint')
            continue

        run_dir = EVAL_ROOT / exp_id / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f'\nLoading {exp_id} seed={seed}: {ckpt_path}')
            ckpt = torch_load_compat_local(ckpt_path, map_location='cpu')
            model = build_model_from_checkpoint(ckpt, meta)
            model.eval()

            ckpt_info = {
                'EXP_ID': exp_id,
                'EXP_NAME': meta['EXP_NAME'],
                'seed': int(seed),
                'checkpoint_path': str(ckpt_path),
                'checkpoint_name': ckpt_path.name,
                'checkpoint_epoch': ckpt.get('epoch', np.nan),
                'checkpoint_best_score': ckpt.get('best_score', np.nan),
                'model_load_status': getattr(model, '_load_status', 'unknown'),
                'missing_keys': ';'.join(getattr(model, '_missing_keys', [])),
                'unexpected_keys': ';'.join(getattr(model, '_unexpected_keys', [])),
            }
            write_json(ckpt_info, run_dir / 'checkpoint_restore_info.json')

            for _, task in ood_task_table.iterrows():
                ood_tier = str(task['ood_tier'])
                ood_task_id = str(task['ood_task_id'])
                ood_method_id = str(task['ood_method_id'])
                task_df = ood_all_df[(ood_all_df['ood_tier'] == ood_tier) & (ood_all_df['ood_task_id'] == ood_task_id)].reset_index(drop=True)
                n_total = int(len(task_df))
                if n_total < MIN_QUERY_ROWS:
                    row = {'EXP_ID': exp_id, 'seed': int(seed), 'ood_tier': ood_tier, 'ood_task_id': ood_task_id, 'reason': 'too_few_rows', 'n_total': n_total}
                    skipped_rows.append(row)
                    append_df_to_csv(skipped_path, pd.DataFrame([row]), columns=OOD_SKIPPED_COLUMNS)
                    continue

                # R0 single-head: zero-shot only.
                if meta['RT_HEAD_TYPE'] == 'single':
                    K = 0
                    repeat_seed = -1
                    protocol = 'single_zero_shot_direct'
                    key = (exp_id, int(seed), ood_tier, ood_task_id, int(K), int(repeat_seed), protocol)
                    if key in done_keys:
                        continue
                    loader = make_ood_loader(task_df, ood_graph_cache, shuffle=False)
                    true_min, pred_min, pred_df = predict_direct_single_head(model, loader)
                    metrics = compute_metric_dict(true_min, pred_min)
                    metrics.update({
                        'EXP_ID': exp_id,
                        'EXP_NAME': meta['EXP_NAME'],
                        'name_short': meta['name_short'],
                        'order': int(meta['order']),
                        'MOLECULE_MODE': meta['MOLECULE_MODE'],
                        'RT_ARCHITECTURE': meta['RT_ARCHITECTURE'],
                        'RT_HEAD_TYPE': meta['RT_HEAD_TYPE'],
                        'USE_DEVICE_METADATA': bool(meta['USE_DEVICE_METADATA']),
                        'JOINT_MULTITASK': bool(meta['JOINT_MULTITASK']),
                        'seed': int(seed),
                        'ood_tier': ood_tier,
                        'ood_task_id': ood_task_id,
                        'ood_method_id': ood_method_id,
                        'K': int(K),
                        'repeat_seed': int(repeat_seed),
                        'repeat_index': 0,
                        'protocol': protocol,
                        'mode': 'zero_shot_direct_single_head',
                        'n_total': n_total,
                        'n_support': 0,
                        'checkpoint_path': str(ckpt_path),
                        'checkpoint_name': ckpt_path.name,
                        'checkpoint_epoch': ckpt.get('epoch', np.nan),
                        'checkpoint_best_score': ckpt.get('best_score', np.nan),
                        'model_load_status': getattr(model, '_load_status', 'unknown'),
                    })
                    metric_rows.append(metrics)
                    append_df_to_csv(metrics_path, pd.DataFrame([metrics]), columns=OOD_METRIC_COLUMNS)
                    done_keys.add(key)
                    if SAVE_QUERY_PREDICTIONS:
                        pred_df = pred_df.copy()
                        for c, v in [('EXP_ID', exp_id), ('seed', int(seed)), ('ood_tier', ood_tier), ('ood_task_id', ood_task_id), ('ood_method_id', ood_method_id), ('K', int(K)), ('repeat_seed', int(repeat_seed)), ('protocol', protocol), ('mode', 'zero_shot_direct_single_head')]:
                            pred_df[c] = v
                        append_df_to_csv(pred_path, pred_df)
                    print(f'{exp_id} seed={seed} {ood_tier}/{ood_task_id} K=0 zero-shot: MAPE={fmt_metric(metrics["mape_pct"], 3)}% MAE={fmt_metric(metrics["mae_sec"], 2)}s n={metrics["n_query"]}')
                    continue

                # R1/R2 multi-head: no zero-shot here; K-shot fresh head only.
                if meta['RT_HEAD_TYPE'] == 'multi':
                    H_all, y_all, yraw_all, meta_all = get_or_build_features(model, task_df, ood_graph_cache, exp_id, int(seed), ood_tier, ood_task_id)
                    if len(H_all) != n_total:
                        raise RuntimeError(f'Feature cache size mismatch for {exp_id} seed={seed} {ood_task_id}: H={len(H_all)} rows={n_total}')
                    for K in K_VALUES_MULTIHEAD:
                        if int(K) >= n_total or (n_total - int(K)) < MIN_QUERY_ROWS:
                            row = {'EXP_ID': exp_id, 'seed': int(seed), 'ood_tier': ood_tier, 'ood_task_id': ood_task_id, 'K': int(K), 'reason': 'K_too_large_for_task', 'n_total': n_total, 'min_query_rows': MIN_QUERY_ROWS}
                            skipped_rows.append(row)
                            append_df_to_csv(skipped_path, pd.DataFrame([row]), columns=OOD_SKIPPED_COLUMNS)
                            continue
                        repeat_seed_list = [int(seed)] if PAIR_SUPPORT_SEED_WITH_MODEL_SEED else [int(x) for x in SUPPORT_RANDOM_SEEDS]
                        for rep_idx, repeat_seed in enumerate(repeat_seed_list):
                            protocol = 'frozen_backbone_kshot_new_head'
                            key = (exp_id, int(seed), ood_tier, ood_task_id, int(K), int(repeat_seed), protocol)
                            if key in done_keys:
                                continue
                            support_idx, query_idx = support_indices_for_task(n_total, int(K), exp_id, int(seed), int(repeat_seed), ood_task_id)
                            support_path = save_support_indices(run_dir, ood_tier, ood_task_id, int(K), int(repeat_seed), support_idx, query_idx)
                            H_support = H_all[support_idx]
                            y_support = y_all[support_idx]
                            H_query = H_all[query_idx]
                            y_query = y_all[query_idx]
                            meta_query = meta_all.iloc[query_idx].reset_index(drop=True)
                            head_seed = int(repeat_seed) + int(seed) * 17 + int(K) * 31 + stable_int_hash(f'{exp_id}:{ood_task_id}')
                            head = fit_fewshot_head(H_support, y_support, seed=head_seed)
                            true_min, pred_min, pred_df = predict_with_fewshot_head(head, H_query, y_query, meta_query)
                            metrics = compute_metric_dict(true_min, pred_min)
                            metrics.update({
                                'EXP_ID': exp_id,
                                'EXP_NAME': meta['EXP_NAME'],
                                'name_short': meta['name_short'],
                                'order': int(meta['order']),
                                'MOLECULE_MODE': meta['MOLECULE_MODE'],
                                'RT_ARCHITECTURE': meta['RT_ARCHITECTURE'],
                                'RT_HEAD_TYPE': meta['RT_HEAD_TYPE'],
                                'USE_DEVICE_METADATA': bool(meta['USE_DEVICE_METADATA']),
                                'JOINT_MULTITASK': bool(meta['JOINT_MULTITASK']),
                                'seed': int(seed),
                                'ood_tier': ood_tier,
                                'ood_task_id': ood_task_id,
                                'ood_method_id': ood_method_id,
                                'K': int(K),
                                'repeat_seed': int(repeat_seed),
                                'repeat_index': int(rep_idx),
                                'protocol': protocol,
                                'mode': 'kshot_fresh_head_frozen_backbone',
                                'n_total': n_total,
                                'n_support': int(len(support_idx)),
                                'support_indices_path': support_path,
                                'checkpoint_path': str(ckpt_path),
                                'checkpoint_name': ckpt_path.name,
                                'checkpoint_epoch': ckpt.get('epoch', np.nan),
                                'checkpoint_best_score': ckpt.get('best_score', np.nan),
                                'model_load_status': getattr(model, '_load_status', 'unknown'),
                                'ood_head_epochs': int(OOD_HEAD_EPOCHS),
                                'ood_head_lr': float(OOD_HEAD_LR),
                            })
                            metric_rows.append(metrics)
                            append_df_to_csv(metrics_path, pd.DataFrame([metrics]), columns=OOD_METRIC_COLUMNS)
                            done_keys.add(key)
                            if SAVE_QUERY_PREDICTIONS:
                                pred_df = pred_df.copy()
                                for c, v in [('EXP_ID', exp_id), ('seed', int(seed)), ('ood_tier', ood_tier), ('ood_task_id', ood_task_id), ('ood_method_id', ood_method_id), ('K', int(K)), ('repeat_seed', int(repeat_seed)), ('repeat_index', int(rep_idx)), ('protocol', protocol), ('mode', 'kshot_fresh_head_frozen_backbone')]:
                                    pred_df[c] = v
                                append_df_to_csv(pred_path, pred_df)
                            print(f'{exp_id} seed={seed} {ood_tier}/{ood_task_id} K={K}: MAPE={fmt_metric(metrics["mape_pct"], 3)}% MAE={fmt_metric(metrics["mae_sec"], 2)}s nQ={metrics["n_query"]}')
                            del head
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

            with open(run_dir / 'ood_evaluation_checkpoint_complete.json', 'w', encoding='utf-8') as f:
                json.dump({'complete': True, 'EXP_ID': exp_id, 'seed': int(seed), 'time_finished': datetime.now().isoformat(timespec='seconds')}, f, ensure_ascii=False, indent=2)
            del model, ckpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            err = traceback.format_exc()
            print(f'[FAILED] {exp_id} seed={seed}: {exc}')
            print(err)
            fail_dir = EVAL_ROOT / exp_id / f'seed_{seed}'
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / 'ood_evaluation_failed.txt').write_text(err, encoding='utf-8')
            row = {'EXP_ID': exp_id, 'EXP_NAME': meta['EXP_NAME'], 'seed': int(seed), 'checkpoint_path': str(ckpt_path), 'error_type': type(exc).__name__, 'error': str(exc)}
            failed_rows.append(row)
            append_df_to_csv(failed_path, pd.DataFrame([row]), columns=OOD_FAILED_COLUMNS)
            if FAIL_FAST:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Ensure audit files exist with headers even if empty.
if not missing_path.exists():
    pd.DataFrame(missing_rows, columns=OOD_MISSING_COLUMNS).to_csv(missing_path, index=False)
if not failed_path.exists():
    pd.DataFrame(failed_rows, columns=OOD_FAILED_COLUMNS).to_csv(failed_path, index=False)
if not skipped_path.exists():
    pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)

metrics_all = sanitize_metric_dataframe(safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS))
missing_df = safe_read_csv(missing_path, columns=OOD_MISSING_COLUMNS)
failed_df = safe_read_csv(failed_path, columns=OOD_FAILED_COLUMNS)
skipped_df = safe_read_csv(skipped_path, columns=OOD_SKIPPED_COLUMNS)

print('\n=== OOD K-shot metrics preview ===')
display(metrics_all.head(40))
print('metrics shape:', metrics_all.shape)
print('missing:', missing_df.shape, 'failed:', failed_df.shape, 'skipped:', skipped_df.shape)
print('Outputs:', EVAL_ROOT)

# ============================================================
# Cell B6. Summary tables and ranks
# ============================================================
metrics_all = sanitize_metric_dataframe(safe_read_csv(EVAL_ROOT / 'ood_kshot_metrics_all_runs.csv', columns=OOD_METRIC_COLUMNS))
if len(metrics_all) == 0:
    raise RuntimeError('No OOD metrics found. Run Cell B5 first.')

for c in ['seed', 'K', 'repeat_seed', 'n_query', 'n_total', 'n_support', 'order']:
    if c in metrics_all.columns:
        metrics_all[c] = pd.to_numeric(metrics_all[c], errors='coerce')
for c in ['mape_pct', 'mae_sec', 'rmse_sec', 'median_ae_sec', 'r2', 'spearman']:
    if c in metrics_all.columns:
        metrics_all[c] = pd.to_numeric(metrics_all[c], errors='coerce').replace([np.inf, -np.inf], np.nan)

METRICS_TO_SUMMARIZE = ['mape_pct', 'r2', 'mae_sec', 'rmse_sec', 'median_ae_sec', 'spearman']
METRIC_LABELS = {
    'mape_pct': 'MAPE (%)',
    'r2': 'R²',
    'mae_sec': 'MAE (sec)',
    'rmse_sec': 'RMSE (sec)',
    'median_ae_sec': 'Median AE (sec)',
    'spearman': 'Spearman',
}
LOWER_CLAMP_ZERO = {'mape_pct', 'mae_sec', 'rmse_sec', 'median_ae_sec'}


def summarize_metric_frame(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if df is None or len(df) == 0:
        return pd.DataFrame()
    for keys, g in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row['n_runs'] = int(len(g))
        row['n_completed_model_seeds'] = int(g['seed'].nunique()) if 'seed' in g.columns else np.nan
        row['model_seeds_present'] = ','.join(map(str, sorted(g['seed'].dropna().astype(int).unique()))) if 'seed' in g.columns else ''
        row['repeat_seeds_present'] = ','.join(map(str, sorted(g['repeat_seed'].dropna().astype(int).unique()))) if 'repeat_seed' in g.columns else ''
        row['n_query_mean'] = float(pd.to_numeric(g.get('n_query', pd.Series(dtype=float)), errors='coerce').mean()) if 'n_query' in g.columns else np.nan
        row['n_support_mean'] = float(pd.to_numeric(g.get('n_support', pd.Series(dtype=float)), errors='coerce').mean()) if 'n_support' in g.columns else np.nan
        for metric in METRICS_TO_SUMMARIZE:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            n = int(len(vals))
            mean = float(vals.mean()) if n else np.nan
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            sem = std / math.sqrt(n) if n > 1 and np.isfinite(std) else np.nan
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_sem'] = sem
            row[f'{metric}_ci95_low'] = mean - 1.96 * sem if np.isfinite(sem) else np.nan
            row[f'{metric}_ci95_high'] = mean + 1.96 * sem if np.isfinite(sem) else np.nan
            row[f'{metric}_n'] = n
        rows.append(row)
    out = pd.DataFrame(rows)
    if 'EXP_ID' in out.columns:
        out['order'] = out['EXP_ID'].map(EXP_ORDER).fillna(out.get('order', np.nan))
        out['name_short'] = out['EXP_ID'].map(EXP_SHORT).fillna(out['EXP_ID'])
    return out

summary_by_tier_exp_K = summarize_metric_frame(metrics_all, ['ood_tier', 'EXP_ID', 'EXP_NAME', 'RT_ARCHITECTURE', 'RT_HEAD_TYPE', 'protocol', 'K'])
summary_by_task_exp_K = summarize_metric_frame(metrics_all, ['ood_tier', 'ood_task_id', 'ood_method_id', 'EXP_ID', 'EXP_NAME', 'RT_ARCHITECTURE', 'RT_HEAD_TYPE', 'protocol', 'K'])
summary_by_tier_exp_K.to_csv(EVAL_ROOT / 'summary_by_ood_tier_experiment_K.csv', index=False)
summary_by_task_exp_K.to_csv(EVAL_ROOT / 'summary_by_ood_task_experiment_K.csv', index=False)

# Rank tables by OOD tier and K.
rank_rows = []
for (tier, K), g in summary_by_tier_exp_K.groupby(['ood_tier', 'K'], dropna=False):
    gg = g.copy()
    gg['rank_mape'] = gg['mape_pct_mean'].rank(method='min', ascending=True)
    gg['rank_mae'] = gg['mae_sec_mean'].rank(method='min', ascending=True)
    gg['rank_r2'] = gg['r2_mean'].rank(method='min', ascending=False)
    rank_rows.append(gg)
rank_by_tier_K = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
rank_by_tier_K.to_csv(EVAL_ROOT / 'rank_by_ood_tier_K.csv', index=False)

print('=== Summary by OOD tier / experiment / K ===')
display(summary_by_tier_exp_K.sort_values(['ood_tier', 'K', 'order']))
print('=== Rank by OOD tier / K ===')
display(rank_by_tier_K.sort_values(['ood_tier', 'K', 'rank_mape']).head(100))

# ============================================================
# Cell B7. Completion audit
# ============================================================
metrics_all = safe_read_csv(EVAL_ROOT / 'ood_kshot_metrics_all_runs.csv')
missing_df = safe_read_csv(EVAL_ROOT / 'ood_kshot_missing_checkpoints.csv')
failed_df = safe_read_csv(EVAL_ROOT / 'ood_kshot_failed_runs.csv')
skipped_df = safe_read_csv(EVAL_ROOT / 'ood_kshot_skipped_tasks.csv')

completion_rows = []
for meta in EXPERIMENTS:
    for seed in SPLIT_SEEDS:
        ckpt = find_checkpoint_path(meta['EXP_ID'], int(seed))
        g = metrics_all[(metrics_all.get('EXP_ID', pd.Series(dtype=str)).astype(str) == meta['EXP_ID']) & (pd.to_numeric(metrics_all.get('seed', pd.Series(dtype=float)), errors='coerce') == int(seed))] if len(metrics_all) else pd.DataFrame()
        completion_rows.append({
            'EXP_ID': meta['EXP_ID'],
            'name_short': meta['name_short'],
            'seed': int(seed),
            'checkpoint_exists': ckpt is not None,
            'checkpoint_path': str(ckpt) if ckpt is not None else '',
            'n_metric_rows': int(len(g)),
            'n_ood_tiers': int(g['ood_tier'].nunique()) if len(g) and 'ood_tier' in g.columns else 0,
            'n_ood_tasks': int(g['ood_task_id'].nunique()) if len(g) and 'ood_task_id' in g.columns else 0,
            'single_zero_shot_rows': int((g['protocol'] == 'single_zero_shot_direct').sum()) if len(g) and 'protocol' in g.columns else 0,
            'kshot_rows': int((g['protocol'] == 'frozen_backbone_kshot_new_head').sum()) if len(g) and 'protocol' in g.columns else 0,
        })
completion_df = pd.DataFrame(completion_rows)
completion_df.to_csv(EVAL_ROOT / 'ood_kshot_completion_audit.csv', index=False)

config_manifest = {
    'protocol': 'R0 single-head zero-shot only; R1/R2 multi-head frozen-backbone K-shot fresh head only',
    'low_overlap_method_ids': LOW_OVERLAP_METHOD_IDS,
    'k_values_multihead': K_VALUES_MULTIHEAD,
    'shimadzu_target_temp': SHIMADZU_TARGET_TEMP,
    'shimadzu_target_acn_values': SHIMADZU_TARGET_ACN_VALUES,
    'pair_support_seed_with_model_seed': PAIR_SUPPORT_SEED_WITH_MODEL_SEED,
    'support_random_seeds': SUPPORT_RANDOM_SEEDS,
    'ood_head_epochs': OOD_HEAD_EPOCHS,
    'ood_head_lr': OOD_HEAD_LR,
    'ood_head_weight_decay': OOD_HEAD_WEIGHT_DECAY,
    'ood_head_hidden': OOD_HEAD_HIDDEN,
    'created_at': datetime.now().isoformat(timespec='seconds'),
}
write_json(config_manifest, EVAL_ROOT / 'ood_kshot_protocol_manifest.json')

print('=== OOD K-shot completion audit ===')
display(completion_df)
print('Missing checkpoints:', missing_df.shape)
print('Failed runs:', failed_df.shape)
print('Skipped tasks/K:', skipped_df.shape)
if len(failed_df):
    display(failed_df.head(50))
if len(skipped_df):
    display(skipped_df.head(50))
print('Protocol manifest:', EVAL_ROOT / 'ood_kshot_protocol_manifest.json')


# ============================================================
# FINAL CELL — No-plot OOD K-shot aggregation / summary
# Purpose:
#   Aggregate the final CSV result tables.
#   Read finished OOD metrics, clean scalar metrics, summarize by tier/task/K,
#   rank models, and save CSV tables only.
# ============================================================

from pathlib import Path
import ast
import math
import json
import numpy as np
import pandas as pd


# -----------------------------
# Locate evaluation root
# -----------------------------
try:
    EVAL_ROOT = Path(EVAL_ROOT)
except NameError:
    EVAL_ROOT = Path("outputs/evaluation_E1_E9_OOD_kshot_lowoverlap_shimadzu")

NO_PLOT_OUT = EVAL_ROOT / "aggregate_no_plot"
NO_PLOT_OUT.mkdir(parents=True, exist_ok=True)

METRICS_PATH = EVAL_ROOT / "ood_kshot_metrics_all_runs.csv"
SKIPPED_PATH = EVAL_ROOT / "ood_kshot_skipped_tasks.csv"
MISSING_PATH = EVAL_ROOT / "ood_kshot_missing_checkpoints.csv"
FAILED_PATH = EVAL_ROOT / "ood_kshot_failed_runs.csv"

print("EVAL_ROOT:", EVAL_ROOT.resolve())
print("METRICS_PATH:", METRICS_PATH.resolve())
print("NO_PLOT_OUT:", NO_PLOT_OUT.resolve())

if not METRICS_PATH.exists():
    raise FileNotFoundError(f"Cannot find metrics CSV: {METRICS_PATH}")

# -----------------------------
# Robust readers
# -----------------------------
def robust_read_csv(path: Path, columns=None):
    path = Path(path)
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    except pd.errors.ParserError as e:
        print(f"[WARN] ParserError reading {path.name}; retry with python engine and skip bad lines.")
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except TypeError:
            # older pandas fallback
            return pd.read_csv(path, engine="python", error_bad_lines=False)

def scalarize_metric_value(x):
    """
    Convert values such as:
      12.34
      "12.34"
      "(12.34, 100)"
      [12.34, 100]
    into scalar float 12.34.
    """
    if x is None:
        return np.nan
    if isinstance(x, (tuple, list, np.ndarray)):
        return scalarize_metric_value(x[0]) if len(x) else np.nan
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return np.nan

    # tuple/list stored as string
    if (s.startswith("(") and "," in s) or (s.startswith("[") and "," in s):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (tuple, list)) and len(obj):
                return scalarize_metric_value(obj[0])
        except Exception:
            pass

    # normal numeric
    return pd.to_numeric(s, errors="coerce")

def safe_int_unique(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    try:
        return ",".join(map(str, sorted(vals.astype(int).unique())))
    except Exception:
        return ",".join(map(str, sorted(series.dropna().astype(str).unique())))

# -----------------------------
# Load metrics and clean numeric columns
# -----------------------------
metrics_df = robust_read_csv(METRICS_PATH)

if len(metrics_df) == 0:
    raise RuntimeError(f"Metrics file exists but has no rows: {METRICS_PATH}")

# Clean metric columns; this specifically fixes old tuple-style mape_pct values.
metric_cols = ["mape_pct", "r2", "mae_sec", "rmse_sec", "median_ae_sec", "spearman"]
for col in metric_cols:
    if col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].map(scalarize_metric_value).astype(float)

for col in ["K", "seed", "repeat_seed", "repeat_index", "n_query", "n_support", "n_total"]:
    if col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

# Ensure core columns exist
for col in ["ood_tier", "ood_task_id", "ood_method_id", "EXP_ID", "EXP_NAME", "name_short", "RT_ARCHITECTURE", "RT_HEAD_TYPE", "protocol"]:
    if col not in metrics_df.columns:
        metrics_df[col] = "NA"

metrics_df["EXP_ID"] = metrics_df["EXP_ID"].astype(str)
metrics_df["ood_tier"] = metrics_df["ood_tier"].astype(str)
metrics_df["ood_task_id"] = metrics_df["ood_task_id"].astype(str)
metrics_df["K"] = pd.to_numeric(metrics_df["K"], errors="coerce").astype("Int64")

# Sort for readability
sort_cols = [c for c in ["ood_tier", "K", "EXP_ID", "seed", "ood_task_id", "repeat_index"] if c in metrics_df.columns]
metrics_df = metrics_df.sort_values(sort_cols).reset_index(drop=True)

clean_metrics_path = NO_PLOT_OUT / "ood_kshot_metrics_all_runs_CLEANED_no_plot.csv"
metrics_df.to_csv(clean_metrics_path, index=False)

print("Cleaned metrics shape:", metrics_df.shape)
print("Saved cleaned metrics:", clean_metrics_path)

# -----------------------------
# Summarizer
# -----------------------------
def summarize_metrics(df: pd.DataFrame, group_cols):
    rows = []
    if df is None or len(df) == 0:
        return pd.DataFrame()

    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))

        row["n_runs"] = int(len(g))
        if "seed" in g.columns:
            row["n_completed_model_seeds"] = int(pd.to_numeric(g["seed"], errors="coerce").dropna().nunique())
            row["model_seeds_present"] = safe_int_unique(g["seed"])
        if "repeat_seed" in g.columns:
            row["n_repeat_seeds"] = int(pd.to_numeric(g["repeat_seed"], errors="coerce").dropna().nunique())
            row["repeat_seeds_present"] = safe_int_unique(g["repeat_seed"])
        if "ood_task_id" in g.columns:
            row["n_ood_tasks"] = int(g["ood_task_id"].dropna().astype(str).nunique())
            row["ood_tasks_present"] = ",".join(sorted(g["ood_task_id"].dropna().astype(str).unique()))
        if "n_query" in g.columns:
            row["n_query_mean"] = float(pd.to_numeric(g["n_query"], errors="coerce").mean())
        if "n_support" in g.columns:
            row["n_support_mean"] = float(pd.to_numeric(g["n_support"], errors="coerce").mean())

        for metric in metric_cols:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            n = int(len(vals))
            mean = float(vals.mean()) if n else np.nan
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            sem = float(std / math.sqrt(n)) if n > 1 and np.isfinite(std) else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci95_low"] = mean - 1.96 * sem if np.isfinite(sem) else np.nan
            row[f"{metric}_ci95_high"] = mean + 1.96 * sem if np.isfinite(sem) else np.nan
            row[f"{metric}_n"] = n

        rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------
# 1) Run-level summary by OOD tier / experiment / K
# -----------------------------
tier_group_cols = [
    "ood_tier", "EXP_ID", "EXP_NAME", "name_short",
    "RT_ARCHITECTURE", "RT_HEAD_TYPE", "protocol", "K"
]
tier_summary = summarize_metrics(metrics_df, tier_group_cols)

# Rank within each OOD tier and K
tier_summary["rank_mape"] = tier_summary.groupby(["ood_tier", "K"])["mape_pct_mean"].rank(method="min", ascending=True)
tier_summary["rank_mae"] = tier_summary.groupby(["ood_tier", "K"])["mae_sec_mean"].rank(method="min", ascending=True)
tier_summary["rank_r2"] = tier_summary.groupby(["ood_tier", "K"])["r2_mean"].rank(method="min", ascending=False)

tier_summary = tier_summary.sort_values(["ood_tier", "K", "rank_mape", "EXP_ID"]).reset_index(drop=True)

tier_summary_path = NO_PLOT_OUT / "summary_by_ood_tier_experiment_K_NO_PLOT.csv"
tier_summary.to_csv(tier_summary_path, index=False)

rank_tier_path = NO_PLOT_OUT / "rank_by_ood_tier_K_NO_PLOT.csv"
tier_summary.to_csv(rank_tier_path, index=False)

best_tier = tier_summary[tier_summary["rank_mape"].eq(1)].copy()
best_tier_path = NO_PLOT_OUT / "best_by_ood_tier_K_mape_NO_PLOT.csv"
best_tier.to_csv(best_tier_path, index=False)

# -----------------------------
# 2) Per-task summary by OOD task / experiment / K
# -----------------------------
task_group_cols = [
    "ood_tier", "ood_task_id", "ood_method_id",
    "EXP_ID", "EXP_NAME", "name_short",
    "RT_ARCHITECTURE", "RT_HEAD_TYPE", "protocol", "K"
]
task_summary = summarize_metrics(metrics_df, task_group_cols)
task_summary["rank_mape"] = task_summary.groupby(["ood_tier", "ood_task_id", "K"])["mape_pct_mean"].rank(method="min", ascending=True)
task_summary["rank_mae"] = task_summary.groupby(["ood_tier", "ood_task_id", "K"])["mae_sec_mean"].rank(method="min", ascending=True)
task_summary["rank_r2"] = task_summary.groupby(["ood_tier", "ood_task_id", "K"])["r2_mean"].rank(method="min", ascending=False)
task_summary = task_summary.sort_values(["ood_tier", "ood_task_id", "K", "rank_mape", "EXP_ID"]).reset_index(drop=True)

task_summary_path = NO_PLOT_OUT / "summary_by_ood_task_experiment_K_NO_PLOT.csv"
task_summary.to_csv(task_summary_path, index=False)

best_task = task_summary[task_summary["rank_mape"].eq(1)].copy()
best_task_path = NO_PLOT_OUT / "best_by_ood_task_K_mape_NO_PLOT.csv"
best_task.to_csv(best_task_path, index=False)

# -----------------------------
# 3) Task-macro summary by OOD tier / experiment / K
#    This gives each OOD task equal weight after task-level averaging.
#    Useful because 0391/0390 are huge while Shimadzu/0411 are tiny.
# -----------------------------
macro_rows = []
macro_source = task_summary.copy()

macro_group_cols = [
    "ood_tier", "EXP_ID", "EXP_NAME", "name_short",
    "RT_ARCHITECTURE", "RT_HEAD_TYPE", "protocol", "K"
]

for keys, g in macro_source.groupby(macro_group_cols, dropna=False, sort=True):
    if not isinstance(keys, tuple):
        keys = (keys,)
    row = dict(zip(macro_group_cols, keys))
    row["n_ood_tasks"] = int(g["ood_task_id"].nunique())
    row["ood_tasks_present"] = ",".join(sorted(g["ood_task_id"].dropna().astype(str).unique()))

    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        if mean_col not in g.columns:
            continue
        vals = pd.to_numeric(g[mean_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(vals))
        mean = float(vals.mean()) if n else np.nan
        std = float(vals.std(ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if n > 1 and np.isfinite(std) else np.nan
        row[f"{metric}_task_macro_mean"] = mean
        row[f"{metric}_task_macro_std"] = std
        row[f"{metric}_task_macro_sem"] = sem
        row[f"{metric}_task_macro_ci95_low"] = mean - 1.96 * sem if np.isfinite(sem) else np.nan
        row[f"{metric}_task_macro_ci95_high"] = mean + 1.96 * sem if np.isfinite(sem) else np.nan
        row[f"{metric}_task_macro_n_tasks"] = n

    macro_rows.append(row)

tier_task_macro = pd.DataFrame(macro_rows)
if len(tier_task_macro):
    tier_task_macro["rank_mape_task_macro"] = tier_task_macro.groupby(["ood_tier", "K"])["mape_pct_task_macro_mean"].rank(method="min", ascending=True)
    tier_task_macro["rank_mae_task_macro"] = tier_task_macro.groupby(["ood_tier", "K"])["mae_sec_task_macro_mean"].rank(method="min", ascending=True)
    tier_task_macro["rank_r2_task_macro"] = tier_task_macro.groupby(["ood_tier", "K"])["r2_task_macro_mean"].rank(method="min", ascending=False)
    tier_task_macro = tier_task_macro.sort_values(["ood_tier", "K", "rank_mape_task_macro", "EXP_ID"]).reset_index(drop=True)

tier_task_macro_path = NO_PLOT_OUT / "summary_by_ood_tier_task_macro_experiment_K_NO_PLOT.csv"
tier_task_macro.to_csv(tier_task_macro_path, index=False)

best_tier_task_macro = tier_task_macro[tier_task_macro["rank_mape_task_macro"].eq(1)].copy() if len(tier_task_macro) else pd.DataFrame()
best_tier_task_macro_path = NO_PLOT_OUT / "best_by_ood_tier_K_task_macro_mape_NO_PLOT.csv"
best_tier_task_macro.to_csv(best_tier_task_macro_path, index=False)

# -----------------------------
# Skipped / missing / failed audit
# -----------------------------
skipped_df = robust_read_csv(SKIPPED_PATH)
missing_df = robust_read_csv(MISSING_PATH)
failed_df = robust_read_csv(FAILED_PATH)

audit = {
    "metrics_rows": int(len(metrics_df)),
    "skipped_rows": int(len(skipped_df)),
    "missing_rows": int(len(missing_df)),
    "failed_rows": int(len(failed_df)),
    "paths": {
        "clean_metrics": str(clean_metrics_path),
        "tier_summary": str(tier_summary_path),
        "task_summary": str(task_summary_path),
        "tier_task_macro_summary": str(tier_task_macro_path),
        "best_tier": str(best_tier_path),
        "best_task": str(best_task_path),
        "best_tier_task_macro": str(best_tier_task_macro_path),
    }
}

audit_path = NO_PLOT_OUT / "no_plot_aggregation_audit.json"
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False)

# -----------------------------
# Display compact result tables
# -----------------------------
display_cols = [
    "ood_tier", "K", "rank_mape", "EXP_ID", "name_short",
    "n_runs", "n_completed_model_seeds", "n_ood_tasks",
    "mape_pct_mean", "mape_pct_std", "mae_sec_mean", "r2_mean", "spearman_mean"
]
display_cols = [c for c in display_cols if c in tier_summary.columns]

macro_display_cols = [
    "ood_tier", "K", "rank_mape_task_macro", "EXP_ID", "name_short",
    "n_ood_tasks", "mape_pct_task_macro_mean", "mape_pct_task_macro_std",
    "mae_sec_task_macro_mean", "r2_task_macro_mean", "spearman_task_macro_mean"
]
macro_display_cols = [c for c in macro_display_cols if c in tier_task_macro.columns]

best_display_cols = [
    "ood_tier", "K", "EXP_ID", "name_short",
    "mape_pct_mean", "mape_pct_std", "mae_sec_mean", "r2_mean", "spearman_mean", "n_runs"
]
best_display_cols = [c for c in best_display_cols if c in best_tier.columns]

display(Markdown("## No-plot OOD summary: best model by OOD tier and K"))
display(best_tier[best_display_cols].reset_index(drop=True))

display(Markdown("## No-plot OOD rank table by tier / K"))
display(tier_summary[display_cols].reset_index(drop=True))

display(Markdown("## Task-macro summary by tier / K"))
display(tier_task_macro[macro_display_cols].reset_index(drop=True))

if len(skipped_df):
    display(Markdown("## Skipped K audit"))
    skip_cols = [c for c in ["ood_tier", "ood_task_id", "K", "reason", "n_total", "min_query_rows"] if c in skipped_df.columns]
    skip_summary = skipped_df.groupby(skip_cols, dropna=False).size().reset_index(name="n_skipped_rows")
    display(skip_summary.sort_values([c for c in ["ood_tier", "ood_task_id", "K"] if c in skip_summary.columns]).reset_index(drop=True))

print("\nSaved no-plot summaries:")
for k, v in audit["paths"].items():
    print(f"  {k}: {v}")
print("Audit:", audit_path)
print("\nNO PLOTTING WAS PERFORMED IN THIS CELL.")


# Refresh the two report-facing tables after a successful evaluation.
CURATED_ROOT = PROJECT_ROOT / "result" / "external_ood"
CURATED_ROOT.mkdir(parents=True, exist_ok=True)
for source_name, target_name in [
    ("summary_by_ood_tier_experiment_K.csv", "summary.csv"),
    ("ood_kshot_metrics_all_runs.csv", "per_model_seed.csv"),
]:
    source_path = EVAL_ROOT / source_name
    if source_path.exists():
        shutil.copy2(source_path, CURATED_ROOT / target_name)
