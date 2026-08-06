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
# Cell B1. Internal OOD configuration
# ============================================================
from datetime import datetime
import zlib
import shutil
import csv
import ast

EVAL_ROOT = PROJECT_ROOT / 'result' / '_runs' / 'internal_ood'
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

(EVAL_ROOT / '_shared').mkdir(parents=True, exist_ok=True)

WEIGHT_ROOT = PROJECT_ROOT / 'checkpoints' / 'experiments'
PREFER_CHECKPOINTS = ['best_model.pt', 'best_joint.pt', 'best_rt.pt', 'final_model.pt']

# Frozen project definition: the two held-out RepoRT methods.
INTERNAL_OOD_METHOD_IDS = ['0097', '0238']
STRICT_INTERNAL_OOD_IDS = True
OOD_TIER_NAME = 'internal_heldout_repoRT_ood'

# Requested K grid. All E1-E9 receive K=0 and K>0 results.
K_VALUES = [0, 5, 20, 100, 200]

# The ten frozen E1-E9 seeds.
FROZEN_SPLIT_SEEDS = [2004, 2006, 2011, 2012, 2016, 2020, 2022, 2027, 2032, 2034]
try:
    _discovered = discover_split_seeds(10)
    missing_frozen = [s for s in FROZEN_SPLIT_SEEDS if s not in _discovered]
    if missing_frozen:
        raise RuntimeError(f'Frozen seeds missing from SPLIT_ROOT: {missing_frozen}; discovered={_discovered}')
except Exception as exc:
    print('[WARN] Split-seed discovery audit:', repr(exc))
SPLIT_SEEDS = list(FROZEN_SPLIT_SEEDS)

# Exactly one deterministic support split per model seed. Because the split generator
# does not use EXP_ID, E1-E9 use the same support/query molecules for paired comparison.
PAIR_SUPPORT_SEED_WITH_MODEL_SEED = True
SUPPORT_RANDOM_SEEDS = SPLIT_SEEDS[:]
SUPPORT_UNIT = 'mol_key'  # molecule-clean support/query; falls back to row if unavailable
MIN_QUERY_ROWS = 2

# Fresh OOD head settings, matching the previous frozen-backbone K-shot protocol.
OOD_HEAD_EPOCHS = 300
OOD_HEAD_LR = 1e-3
OOD_HEAD_WEIGHT_DECAY = 1e-4
OOD_HEAD_HIDDEN = 64
BATCH_SIZE_OOD = 256

RESUME_EXISTING = True
SAVE_QUERY_PREDICTIONS = True
SAVE_SUPPORT_INDEX_FILES = True
FAIL_FAST = False

EXPERIMENTS = [
    {'EXP_ID':'E1', 'order':1, 'MOLECULE_MODE':'M0_no_aux',   'RT_ARCHITECTURE':'R0_device_single',  'EXP_NAME':'M0 no PolyOmics × R0 device encoder + single head',       'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'single', 'JOINT_MULTITASK':False, 'RADONPY_PERCENT':0,   'expected_best':'best_rt.pt',    'name_short':'M0-R0'},
    {'EXP_ID':'E2', 'order':2, 'MOLECULE_MODE':'M0_no_aux',   'RT_ARCHITECTURE':'R1_device_multi',   'EXP_NAME':'M0 no PolyOmics × R1 device encoder + multitask heads',   'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':0,   'expected_best':'best_rt.pt',    'name_short':'M0-R1'},
    {'EXP_ID':'E3', 'order':3, 'MOLECULE_MODE':'M0_no_aux',   'RT_ARCHITECTURE':'R2_nodevice_multi', 'EXP_NAME':'M0 no PolyOmics × R2 no-device + multitask heads',         'USE_DEVICE_METADATA':False, 'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':0,   'expected_best':'best_rt.pt',    'name_short':'M0-R2'},
    {'EXP_ID':'E4', 'order':4, 'MOLECULE_MODE':'M1_pretrain', 'RT_ARCHITECTURE':'R0_device_single',  'EXP_NAME':'M1 RadonPy pretrain × R0 device encoder + single head',    'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'single', 'JOINT_MULTITASK':False, 'RADONPY_PERCENT':100, 'expected_best':'best_rt.pt',    'name_short':'M1-R0'},
    {'EXP_ID':'E5', 'order':5, 'MOLECULE_MODE':'M1_pretrain', 'RT_ARCHITECTURE':'R1_device_multi',   'EXP_NAME':'M1 RadonPy pretrain × R1 device encoder + multitask heads','USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':100, 'expected_best':'best_rt.pt',    'name_short':'M1-R1'},
    {'EXP_ID':'E6', 'order':6, 'MOLECULE_MODE':'M1_pretrain', 'RT_ARCHITECTURE':'R2_nodevice_multi', 'EXP_NAME':'M1 RadonPy pretrain × R2 no-device + multitask heads',      'USE_DEVICE_METADATA':False, 'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':False, 'RADONPY_PERCENT':100, 'expected_best':'best_rt.pt',    'name_short':'M1-R2'},
    {'EXP_ID':'E7', 'order':7, 'MOLECULE_MODE':'M2_joint',    'RT_ARCHITECTURE':'R0_device_single',  'EXP_NAME':'M2 joint multitask × R0 device encoder + single head',      'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'single', 'JOINT_MULTITASK':True,  'RADONPY_PERCENT':100, 'expected_best':'best_joint.pt', 'name_short':'M2-R0'},
    {'EXP_ID':'E8', 'order':8, 'MOLECULE_MODE':'M2_joint',    'RT_ARCHITECTURE':'R1_device_multi',   'EXP_NAME':'M2 joint multitask × R1 device encoder + multitask heads',  'USE_DEVICE_METADATA':True,  'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':True,  'RADONPY_PERCENT':100, 'expected_best':'best_joint.pt', 'name_short':'M2-R1'},
    {'EXP_ID':'E9', 'order':9, 'MOLECULE_MODE':'M2_joint',    'RT_ARCHITECTURE':'R2_nodevice_multi', 'EXP_NAME':'M2 joint multitask × R2 no-device + multitask heads',        'USE_DEVICE_METADATA':False, 'RT_HEAD_TYPE':'multi',  'JOINT_MULTITASK':True,  'RADONPY_PERCENT':100, 'expected_best':'best_joint.pt', 'name_short':'M2-R2'},
]
EXP_BY_ID = {e['EXP_ID']: e for e in EXPERIMENTS}
EXP_ORDER = {e['EXP_ID']: int(e['order']) for e in EXPERIMENTS}
EXP_SHORT = {e['EXP_ID']: e['name_short'] for e in EXPERIMENTS}

print('EVAL_ROOT:', EVAL_ROOT)
print('WEIGHT_ROOT:', WEIGHT_ROOT)
print('SPLIT_ROOT:', SPLIT_ROOT)
print('SPLIT_SEEDS:', SPLIT_SEEDS)
print('INTERNAL_OOD_METHOD_IDS:', INTERNAL_OOD_METHOD_IDS)
print('K_VALUES:', K_VALUES)
print('K=0 single-head: native shared head')
print('K=0 multi-head: mean ensemble of all trained seen-method heads')
print('K>0 all E1-E9: frozen representation + fresh OOD head')


# ============================================================
# Cell B2. Robust CSV, checkpoint, metric, and model utilities
# ============================================================

def safe_filename(x: Any, max_len: int = 160) -> str:
    s = re.sub(r'[^A-Za-z0-9._=-]+', '_', str(x)).strip('_')
    return s[:max_len]


def stable_int_hash(x: Any) -> int:
    return int(zlib.crc32(str(x).encode('utf-8')) & 0xffffffff)


def safe_read_csv(path, columns=None):
    path = Path(path)
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns or [])
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    except pd.errors.ParserError:
        try:
            return pd.read_csv(path, engine='python', on_bad_lines='skip')
        except TypeError:
            return pd.read_csv(path, engine='python', error_bad_lines=False)


def append_df_to_csv(path: Path, df: pd.DataFrame, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        if not path.exists() and columns is not None:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    out = df.copy()
    if columns is not None:
        for c in columns:
            if c not in out.columns:
                out[c] = np.nan
        out = out[columns]
    if (not path.exists()) or path.stat().st_size == 0:
        out.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    all_cols = list(header)
    for c in out.columns:
        if c not in all_cols:
            all_cols.append(c)
    if all_cols != header:
        existing = safe_read_csv(path)
        for c in all_cols:
            if c not in existing.columns:
                existing[c] = np.nan
        backup = path.with_name(path.stem + '.schema_backup.csv')
        if not backup.exists():
            shutil.copy2(path, backup)
        existing.reindex(columns=all_cols).to_csv(path, index=False)
    out.reindex(columns=all_cols).to_csv(path, mode='a', header=False, index=False)


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
    RADONPY_PERCENT = int(meta.get('RADONPY_PERCENT', 0))
    RADONPY_FRACTION = RADONPY_PERCENT / 100.0
    BEST_CKPT_NAME = meta.get('expected_best', 'best_joint.pt' if JOINT_MULTITASK else 'best_rt.pt')
    ROOT_OUT = WEIGHT_ROOT / EXP_ID
    FRACTION_OUT = ROOT_OUT
    OUT_DIR = FRACTION_OUT / '_shared'


def find_checkpoint_path(exp_id: str, seed: int) -> Optional[Path]:
    run_dir = WEIGHT_ROOT / exp_id / f'seed_{seed}'
    for name in PREFER_CHECKPOINTS:
        p = run_dir / name
        if p.exists():
            return p
    hits = sorted(run_dir.glob('best*.pt'))
    return hits[0] if hits else None


def infer_radon_target_count_from_state(state_dict: Dict[str, torch.Tensor]) -> int:
    for key, value in state_dict.items():
        if key.endswith('radon_heads.4.weight') and hasattr(value, 'shape'):
            return int(value.shape[0])
    return 0


def checkpoint_radon_target_count(ckpt: Dict[str, Any]) -> int:
    targets = ckpt.get('radon_targets', [])
    n = len(targets) if isinstance(targets, (list, tuple)) else 0
    return max(n, infer_radon_target_count_from_state(ckpt.get('model', {})))


def restore_metadata_globals_from_checkpoint(ckpt: Dict[str, Any]):
    global y_mean, y_std, cat_vocabs, method_vocab, CFG
    if 'y_mean' not in ckpt or 'y_std' not in ckpt:
        raise RuntimeError('Checkpoint missing y_mean/y_std.')
    y_mean = float(ckpt['y_mean'])
    y_std = float(ckpt['y_std'])
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0
    cat_vocabs = ckpt.get('cat_vocabs') or {
        'column_cat0': {'<UNK>': 0, 'NA': 1},
        'brand_cat0': {'<UNK>': 0, 'NA': 1},
        'solvent_cat0': {'<UNK>': 0, 'NA': 1},
        'solvent_cat1': {'<UNK>': 0, 'NA': 1},
    }
    method_vocab = {str(k).zfill(4): int(v) for k, v in (ckpt.get('method_vocab') or {}).items()}
    if ckpt.get('cfg'):
        CFG.update(dict(ckpt['cfg']))


def build_model_from_checkpoint(ckpt: Dict[str, Any], meta: Dict[str, Any]) -> nn.Module:
    restore_metadata_globals_from_checkpoint(ckpt)
    cfg = dict(ckpt.get('cfg', CFG))
    head_type = str(ckpt.get('rt_head_type', meta['RT_HEAD_TYPE']))
    use_device = bool(ckpt.get('USE_DEVICE_METADATA', ckpt.get('use_device_metadata', meta['USE_DEVICE_METADATA'])))
    n_methods = len(ckpt.get('method_vocab', method_vocab))
    model = GraphEnvRTModel(
        cat_vocabs=ckpt.get('cat_vocabs', cat_vocabs),
        radon_targets=checkpoint_radon_target_count(ckpt),
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
        print('[WARN] strict checkpoint load failed; using strict=False:', exc)
        result = model.load_state_dict(ckpt['model'], strict=False)
        model._load_status = 'non_strict'
        model._missing_keys = list(result.missing_keys)
        model._unexpected_keys = list(result.unexpected_keys)
    model.eval()
    return model


def metric_to_float(x, default=np.nan) -> float:
    try:
        if isinstance(x, (tuple, list, np.ndarray)):
            return metric_to_float(x[0], default) if len(x) else float(default)
        if isinstance(x, str) and x.strip().startswith(('(', '[')):
            try:
                return metric_to_float(ast.literal_eval(x), default)
            except Exception:
                pass
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def split_mape_return(x):
    if isinstance(x, (tuple, list, np.ndarray)):
        return metric_to_float(x[0]), int(x[1]) if len(x) > 1 else 0
    return metric_to_float(x), 0


def fmt_metric(x, digits=3):
    v = metric_to_float(x)
    return f'{v:.{digits}f}' if np.isfinite(v) else 'nan'


def safe_spearman(y_true, y_pred):
    try:
        s = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        return float(s) if pd.notna(s) else np.nan
    except Exception:
        return np.nan


def compute_metric_dict(true_min, pred_min) -> Dict[str, Any]:
    true_min = np.asarray(true_min, dtype=float)
    pred_min = np.asarray(pred_min, dtype=float)
    mask = np.isfinite(true_min) & np.isfinite(pred_min)
    true_min, pred_min = true_min[mask], pred_min[mask]
    err_sec = (pred_min - true_min) * 60.0
    abs_err_sec = np.abs(err_sec)
    mape_value, mape_n = split_mape_return(safe_mape_pct(true_min, pred_min))
    return {
        'n_query': int(len(true_min)),
        'mae_sec': float(abs_err_sec.mean()) if len(abs_err_sec) else np.nan,
        'median_ae_sec': float(np.median(abs_err_sec)) if len(abs_err_sec) else np.nan,
        'rmse_sec': float(np.sqrt(np.mean(err_sec ** 2))) if len(err_sec) else np.nan,
        'mape_pct': float(mape_value),
        'mape_n_rows': int(mape_n),
        'r2': safe_r2(true_min, pred_min),
        'spearman': safe_spearman(true_min, pred_min),
    }


OOD_METRIC_COLUMNS = [
    'mae_sec','median_ae_sec','rmse_sec','mape_pct','mape_n_rows','r2','spearman','n_query',
    'EXP_ID','EXP_NAME','name_short','order','MOLECULE_MODE','RT_ARCHITECTURE','RT_HEAD_TYPE',
    'USE_DEVICE_METADATA','JOINT_MULTITASK','seed','ood_tier','ood_task_id','ood_method_id',
    'K','repeat_seed','repeat_index','protocol','mode','zero_shot_strategy','support_unit',
    'n_total','n_unique_mol_keys','n_support','n_excluded_same_molecule_rows',
    'checkpoint_path','checkpoint_name','checkpoint_epoch','checkpoint_best_score','model_load_status',
    'support_indices_path','ood_head_epochs','ood_head_lr','ood_head_weight_decay'
]
OOD_MISSING_COLUMNS = ['EXP_ID','EXP_NAME','seed','expected_run_dir','reason','preferred_names']
OOD_FAILED_COLUMNS = ['EXP_ID','EXP_NAME','seed','ood_task_id','K','checkpoint_path','error_type','error']
OOD_SKIPPED_COLUMNS = ['EXP_ID','seed','ood_tier','ood_task_id','K','reason','n_total','n_unique_mol_keys','min_query_rows']


# ============================================================
# Cell B3. Load and audit the two fixed internal/held-out OOD datasets
# ============================================================

def _canonical_ood_frame_for_seed(seed: int) -> pd.DataFrame:
    split_dir = resolve_split_dir(int(seed))
    path = split_dir / 'external_ood.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = normalize_split_df(pd.read_csv(path), 'internal_heldout_ood')
    df['dir'] = df['dir'].astype(str).str.extract(r'(\d+)')[0].str.zfill(4)
    df['dataset_id'] = df['dir']
    df['row_id'] = df['row_id'].astype(str)
    df['mol_key'] = df['mol_key'].astype(str)
    df = df[df['dir'].isin(INTERNAL_OOD_METHOD_IDS)].copy()
    actual_ids = sorted(df['dir'].unique().tolist())
    if STRICT_INTERNAL_OOD_IDS and actual_ids != sorted(INTERNAL_OOD_METHOD_IDS):
        raise RuntimeError(
            f'Internal OOD IDs do not match the frozen definition. '
            f'expected={sorted(INTERNAL_OOD_METHOD_IDS)}, actual={actual_ids}, file={path}'
        )
    # Rebuild environment features with the same parser used by E1-E9, including brand_cat.
    df = attach_env_features(df)
    if 'clean_radonpy_overlap' not in df.columns:
        df['clean_radonpy_overlap'] = False
    if 'non_radonpy_overlap' not in df.columns:
        df['non_radonpy_overlap'] = False
    df['ood_tier'] = OOD_TIER_NAME
    df['ood_method_id'] = df['dir']
    df['ood_task_id'] = df['dir']
    df['split'] = OOD_TIER_NAME
    df['source_split_seed'] = int(seed)
    return df.sort_values(['dir', 'mol_key', 'row_id']).reset_index(drop=True)


def _row_signature(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['dir', 'row_id', 'mol_key', 'smiles', 'rt']
    out = df[cols].copy()
    out['rt'] = pd.to_numeric(out['rt'], errors='coerce').round(10)
    return out.sort_values(cols[:-1]).reset_index(drop=True)


reference_seed = int(SPLIT_SEEDS[0])
internal_ood_df = _canonical_ood_frame_for_seed(reference_seed)
reference_signature = _row_signature(internal_ood_df)

audit_rows = []
for seed in SPLIT_SEEDS:
    df_seed = _canonical_ood_frame_for_seed(int(seed))
    sig = _row_signature(df_seed)
    same_rows = bool(sig.equals(reference_signature))
    audit_rows.append({
        'seed': int(seed),
        'split_dir': str(resolve_split_dir(int(seed))),
        'n_rows': int(len(df_seed)),
        'n_methods': int(df_seed['dir'].nunique()),
        'n_unique_mol_keys': int(df_seed['mol_key'].nunique()),
        'method_ids': ','.join(sorted(df_seed['dir'].unique().tolist())),
        'identical_to_reference_seed': same_rows,
    })
    if not same_rows:
        raise RuntimeError(f'external_ood.csv is not identical across seeds: reference={reference_seed}, different seed={seed}')

internal_ood_audit = pd.DataFrame(audit_rows)
internal_ood_audit.to_csv(EVAL_ROOT / 'internal_ood_split_seed_audit.csv', index=False)
internal_ood_df.to_csv(EVAL_ROOT / 'internal_ood_rows_master.csv', index=False)

internal_ood_task_summary = (
    internal_ood_df.groupby(['ood_tier', 'ood_task_id', 'ood_method_id'])
    .agg(n_rows=('row_id', 'size'), n_unique_mol_keys=('mol_key', 'nunique'), rt_min=('rt', 'min'), rt_max=('rt', 'max'))
    .reset_index()
)
internal_ood_task_summary.to_csv(EVAL_ROOT / 'internal_ood_task_loading_summary.csv', index=False)

print('Reference split seed:', reference_seed)
print('All split seeds use identical held-out rows:', bool(internal_ood_audit['identical_to_reference_seed'].all()))
display(internal_ood_audit)
print('\nInternal OOD datasets:')
display(internal_ood_task_summary)

# Build one shared graph cache because the held-out rows are fixed across seeds.
OOD_GRAPH_CACHE_PATH = EVAL_ROOT / '_shared' / 'pyg_graph_cache_internal_heldout_ood.pt'
ood_graph_cache = PrecomputedPyGGraphCache(OOD_GRAPH_CACHE_PATH)
ood_graph_cache.build(internal_ood_df['smiles'], save=True)
print('OOD graph cache:', OOD_GRAPH_CACHE_PATH)


# ============================================================
# Cell B4. Zero-shot prediction, frozen features, and paired K-shot helpers
# ============================================================

def make_ood_loader(df: pd.DataFrame, batch_size: Optional[int] = None, shuffle: bool = False) -> DataLoader:
    ds = RTGraphDataset(df.reset_index(drop=True), ood_graph_cache)
    return DataLoader(ds, batch_size=int(batch_size or BATCH_SIZE_OOD), shuffle=shuffle, collate_fn=collate_rt, num_workers=0)


@torch.no_grad()
def predict_zero_shot(model: nn.Module, loader: DataLoader):
    """K=0 prediction for every architecture.

    R0 single: native shared RT head.
    R1/R2 multi: mean of all trained seen-method heads; never use the untrained default_head.
    """
    model.eval()
    true_all, pred_all, rows = [], [], []
    strategy = 'native_shared_single_head' if getattr(model, 'head_type', 'single') == 'single' else 'seen_head_mean_ensemble'
    for batch in loader:
        batch = move_batch_to_device(batch)
        if getattr(model, 'head_type', 'single') == 'single':
            pred_std, _ = model.forward_rt(batch)
        else:
            z, _ = model.encode_representation(batch)
            h = model.rt_head.trunk(z)
            if len(model.rt_head.heads) == 0:
                raise RuntimeError('Multi-head checkpoint has no trained seen-method heads.')
            stacked = torch.stack([head(h).squeeze(-1) for head in model.rt_head.heads], dim=0)
            pred_std = stacked.mean(dim=0)
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
    return (
        np.concatenate(true_all) if true_all else np.array([], dtype=float),
        np.concatenate(pred_all) if pred_all else np.array([], dtype=float),
        pd.DataFrame(rows),
        strategy,
    )


@torch.no_grad()
def extract_frozen_features(model: nn.Module, loader: DataLoader):
    """Features used by the fresh OOD head.

    Multi-head: use rt_head.trunk(r_rt), exactly matching the input of dataset-specific heads.
    Single-head: use r_rt directly.
    """
    model.eval()
    Hs, ys, yraws, rows = [], [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch)
        z, _ = model.encode_representation(batch)
        H = model.rt_head.trunk(z) if getattr(model, 'head_type', 'single') == 'multi' else z
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
            })
    if not Hs:
        return torch.empty(0, int(CFG.get('d_model', 256))), torch.empty(0), torch.empty(0), pd.DataFrame(rows)
    return torch.cat(Hs), torch.cat(ys), torch.cat(yraws), pd.DataFrame(rows)


class FewShotOODHead(nn.Module):
    def __init__(self, d_model=256, hidden=OOD_HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_fewshot_head(H_support: torch.Tensor, y_support: torch.Tensor, seed: int) -> nn.Module:
    set_all_seeds(int(seed))
    H_support = H_support.to(DEVICE)
    y_support = y_support.to(DEVICE)
    head = FewShotOODHead(int(H_support.shape[1]), OOD_HEAD_HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=OOD_HEAD_LR, weight_decay=OOD_HEAD_WEIGHT_DECAY)
    for _ in range(OOD_HEAD_EPOCHS):
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
    pred_std = head(H_query.to(DEVICE)).detach().cpu().numpy()
    true_std = y_query.detach().cpu().numpy()
    true_min = unscale_rt(true_std)
    pred_min = unscale_rt(pred_std)
    out = meta_query.copy()
    out['pred_min'] = pred_min
    out['true_sec'] = true_min * 60.0
    out['pred_sec'] = pred_min * 60.0
    out['abs_error_sec'] = np.abs(pred_min - true_min) * 60.0
    out['ape_pct'] = np.abs(pred_min - true_min) / np.maximum(np.abs(true_min), 1e-8) * 100.0
    return true_min, pred_min, out


def feature_cache_path(exp_id: str, seed: int, task_id: str) -> Path:
    return EVAL_ROOT / '_feature_cache' / exp_id / f'seed_{seed}' / f'{safe_filename(task_id)}_features.pt'


def get_or_build_features(model: nn.Module, task_df: pd.DataFrame, exp_id: str, seed: int, task_id: str):
    path = feature_cache_path(exp_id, seed, task_id)
    if RESUME_EXISTING and path.exists():
        obj = torch_load_compat_local(path, map_location='cpu')
        return obj['H_all'], obj['y_all'], obj['yraw_all'], obj['meta_all']
    H_all, y_all, yraw_all, meta_all = extract_frozen_features(model, make_ood_loader(task_df, shuffle=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'H_all': H_all, 'y_all': y_all, 'yraw_all': yraw_all, 'meta_all': meta_all}, path)
    return H_all, y_all, yraw_all, meta_all


def support_query_indices(task_df: pd.DataFrame, K: int, model_seed: int, repeat_seed: int, task_id: str):
    """Create a deterministic support/query split independent of EXP_ID.

    With SUPPORT_UNIT='mol_key', K means K labeled molecules. One representative row per selected
    molecule is used as support, and every other row sharing those molecule keys is excluded from query.
    """
    n_total = len(task_df)
    if K <= 0:
        return np.array([], dtype=int), np.arange(n_total, dtype=int), 0, int(task_df['mol_key'].nunique())
    seed_value = int(repeat_seed) * 1009 + int(model_seed) * 100003 + int(K) * 9176 + stable_int_hash(task_id)
    rng = np.random.default_rng(seed_value)
    if SUPPORT_UNIT == 'mol_key' and 'mol_key' in task_df.columns:
        mol_series = task_df['mol_key'].astype(str)
        unique_keys = np.array(sorted(mol_series.unique().tolist()), dtype=object)
        if K > len(unique_keys):
            raise ValueError(f'K={K} exceeds n_unique_mol_keys={len(unique_keys)}')
        selected_keys = set(rng.choice(unique_keys, size=int(K), replace=False).tolist())
        support_idx = []
        excluded_same_molecule_rows = 0
        for key in sorted(selected_keys):
            idxs = np.flatnonzero(mol_series.to_numpy() == key)
            support_idx.append(int(idxs[0]))
            excluded_same_molecule_rows += max(0, len(idxs) - 1)
        support_idx = np.array(sorted(support_idx), dtype=int)
        query_idx = np.flatnonzero(~mol_series.isin(selected_keys).to_numpy()).astype(int)
        return support_idx, query_idx, int(excluded_same_molecule_rows), int(len(unique_keys))
    if K > n_total:
        raise ValueError(f'K={K} exceeds n_total={n_total}')
    support_idx = np.sort(rng.choice(np.arange(n_total), size=int(K), replace=False))
    support_set = set(support_idx.tolist())
    query_idx = np.array([i for i in range(n_total) if i not in support_set], dtype=int)
    return support_idx, query_idx, 0, int(task_df['mol_key'].nunique()) if 'mol_key' in task_df else n_total


def save_support_indices(task_df: pd.DataFrame, model_seed: int, task_id: str, K: int, repeat_seed: int,
                         support_idx: np.ndarray, query_idx: np.ndarray) -> str:
    if not SAVE_SUPPORT_INDEX_FILES:
        return ''
    roles = pd.DataFrame({
        'row_index': np.concatenate([support_idx, query_idx]),
        'role': ['support'] * len(support_idx) + ['query'] * len(query_idx),
    })
    roles = roles.merge(
        task_df.reset_index().rename(columns={'index':'row_index'})[['row_index','row_id','mol_key','rt']],
        on='row_index', how='left'
    )
    p = EVAL_ROOT / '_support_splits' / f'seed_{model_seed}' / safe_filename(task_id) / f'K{K}_repeat{repeat_seed}.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        roles.to_csv(p, index=False)
    return str(p)


# ============================================================
# Cell B5. Run E1-E9 on 0097/0238 at K=0/5/20/100/200
# ============================================================
metrics_path = EVAL_ROOT / 'internal_ood_kshot_metrics_all_runs.csv'
pred_path = EVAL_ROOT / 'internal_ood_kshot_predictions_query_all.csv'
missing_path = EVAL_ROOT / 'internal_ood_missing_checkpoints.csv'
failed_path = EVAL_ROOT / 'internal_ood_failed_runs.csv'
skipped_path = EVAL_ROOT / 'internal_ood_skipped_tasks.csv'

existing_metrics = safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS)
done_keys = set()
if RESUME_EXISTING and len(existing_metrics):
    for _, r in existing_metrics.iterrows():
        try:
            done_keys.add((str(r['EXP_ID']), int(r['seed']), str(r['ood_task_id']), int(r['K']), int(r['repeat_seed']), str(r['protocol'])))
        except Exception:
            pass
    print('Resume enabled; completed metric keys:', len(done_keys))

missing_rows, failed_rows, skipped_rows = [], [], []

task_table = (
    internal_ood_df[['ood_tier','ood_task_id','ood_method_id']]
    .drop_duplicates().sort_values('ood_task_id').reset_index(drop=True)
)
print('Tasks:')
display(task_table.merge(internal_ood_task_summary, on=['ood_tier','ood_task_id','ood_method_id'], how='left'))

for meta in EXPERIMENTS:
    configure_experiment_globals(meta)
    exp_id = meta['EXP_ID']
    print('\n' + '#'*110)
    print('Experiment:', exp_id, meta['name_short'], '|', meta['EXP_NAME'])
    print('#'*110)

    for seed in SPLIT_SEEDS:
        ckpt_path = find_checkpoint_path(exp_id, int(seed))
        if ckpt_path is None:
            row = {
                'EXP_ID': exp_id, 'EXP_NAME': meta['EXP_NAME'], 'seed': int(seed),
                'expected_run_dir': str(WEIGHT_ROOT / exp_id / f'seed_{seed}'),
                'reason': 'checkpoint_not_found', 'preferred_names': ','.join(PREFER_CHECKPOINTS),
            }
            missing_rows.append(row)
            append_df_to_csv(missing_path, pd.DataFrame([row]), OOD_MISSING_COLUMNS)
            print(f'[MISSING] {exp_id} seed={seed}')
            continue

        run_dir = EVAL_ROOT / exp_id / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f'\nLoading {exp_id} seed={seed}: {ckpt_path}')
            ckpt = torch_load_compat_local(ckpt_path, map_location='cpu')
            model = build_model_from_checkpoint(ckpt, meta)
            write_json({
                'EXP_ID': exp_id, 'EXP_NAME': meta['EXP_NAME'], 'seed': int(seed),
                'checkpoint_path': str(ckpt_path), 'checkpoint_name': ckpt_path.name,
                'checkpoint_epoch': ckpt.get('epoch', np.nan),
                'checkpoint_best_score': ckpt.get('best_score', np.nan),
                'model_load_status': getattr(model, '_load_status', 'unknown'),
                'missing_keys': getattr(model, '_missing_keys', []),
                'unexpected_keys': getattr(model, '_unexpected_keys', []),
            }, run_dir / 'checkpoint_restore_info.json')

            for _, task in task_table.iterrows():
                task_id = str(task['ood_task_id'])
                task_df = internal_ood_df[internal_ood_df['ood_task_id'].eq(task_id)].reset_index(drop=True)
                n_total = int(len(task_df))
                n_unique = int(task_df['mol_key'].nunique())
                if n_total < MIN_QUERY_ROWS:
                    row = {'EXP_ID':exp_id,'seed':int(seed),'ood_tier':OOD_TIER_NAME,'ood_task_id':task_id,'K':np.nan,'reason':'too_few_rows','n_total':n_total,'n_unique_mol_keys':n_unique,'min_query_rows':MIN_QUERY_ROWS}
                    skipped_rows.append(row)
                    append_df_to_csv(skipped_path, pd.DataFrame([row]), OOD_SKIPPED_COLUMNS)
                    continue

                # Build frozen features once per checkpoint/task for all K>0.
                H_all = y_all = yraw_all = meta_all = None

                for K in K_VALUES:
                    if K == 0:
                        repeat_seed = -1
                        protocol = 'zero_shot_native_single' if meta['RT_HEAD_TYPE'] == 'single' else 'zero_shot_seen_head_mean_ensemble'
                        key = (exp_id, int(seed), task_id, 0, repeat_seed, protocol)
                        if key in done_keys:
                            continue
                        loader = make_ood_loader(task_df, shuffle=False)
                        true_min, pred_min, pred_df, zero_strategy = predict_zero_shot(model, loader)
                        metrics = compute_metric_dict(true_min, pred_min)
                        metrics.update({
                            'EXP_ID':exp_id,'EXP_NAME':meta['EXP_NAME'],'name_short':meta['name_short'],'order':int(meta['order']),
                            'MOLECULE_MODE':meta['MOLECULE_MODE'],'RT_ARCHITECTURE':meta['RT_ARCHITECTURE'],'RT_HEAD_TYPE':meta['RT_HEAD_TYPE'],
                            'USE_DEVICE_METADATA':bool(meta['USE_DEVICE_METADATA']),'JOINT_MULTITASK':bool(meta['JOINT_MULTITASK']),
                            'seed':int(seed),'ood_tier':OOD_TIER_NAME,'ood_task_id':task_id,'ood_method_id':task_id,
                            'K':0,'repeat_seed':repeat_seed,'repeat_index':0,'protocol':protocol,'mode':'zero_shot','zero_shot_strategy':zero_strategy,
                            'support_unit':SUPPORT_UNIT,'n_total':n_total,'n_unique_mol_keys':n_unique,'n_support':0,'n_excluded_same_molecule_rows':0,
                            'checkpoint_path':str(ckpt_path),'checkpoint_name':ckpt_path.name,'checkpoint_epoch':ckpt.get('epoch',np.nan),
                            'checkpoint_best_score':ckpt.get('best_score',np.nan),'model_load_status':getattr(model,'_load_status','unknown'),
                        })
                        append_df_to_csv(metrics_path, pd.DataFrame([metrics]), OOD_METRIC_COLUMNS)
                        done_keys.add(key)
                        if SAVE_QUERY_PREDICTIONS:
                            pred_df = pred_df.copy()
                            for c,v in [('EXP_ID',exp_id),('seed',int(seed)),('ood_tier',OOD_TIER_NAME),('ood_task_id',task_id),('K',0),('repeat_seed',repeat_seed),('protocol',protocol),('mode','zero_shot'),('zero_shot_strategy',zero_strategy)]:
                                pred_df[c] = v
                            append_df_to_csv(pred_path, pred_df)
                        print(f'{exp_id} seed={seed} dataset={task_id} K=0 [{zero_strategy}] MAPE={fmt_metric(metrics["mape_pct"])}% MAE={fmt_metric(metrics["mae_sec"],2)}s nQ={metrics["n_query"]}')
                        continue

                    # K>0 for every E1-E9 model: frozen model + fresh OOD head.
                    if K > n_unique or (n_total - K) < MIN_QUERY_ROWS:
                        row = {'EXP_ID':exp_id,'seed':int(seed),'ood_tier':OOD_TIER_NAME,'ood_task_id':task_id,'K':int(K),'reason':'K_too_large_for_task','n_total':n_total,'n_unique_mol_keys':n_unique,'min_query_rows':MIN_QUERY_ROWS}
                        skipped_rows.append(row)
                        append_df_to_csv(skipped_path, pd.DataFrame([row]), OOD_SKIPPED_COLUMNS)
                        continue

                    if H_all is None:
                        H_all, y_all, yraw_all, meta_all = get_or_build_features(model, task_df, exp_id, int(seed), task_id)
                        if len(H_all) != n_total:
                            raise RuntimeError(f'Feature cache mismatch: {exp_id} seed={seed} dataset={task_id}, H={len(H_all)}, rows={n_total}')

                    repeat_seed_list = [int(seed)] if PAIR_SUPPORT_SEED_WITH_MODEL_SEED else [int(x) for x in SUPPORT_RANDOM_SEEDS]
                    for rep_idx, repeat_seed in enumerate(repeat_seed_list):
                        protocol = 'frozen_backbone_fresh_head_kshot'
                        key = (exp_id, int(seed), task_id, int(K), int(repeat_seed), protocol)
                        if key in done_keys:
                            continue
                        support_idx, query_idx, n_excluded_dup, n_unique_check = support_query_indices(
                            task_df, int(K), int(seed), int(repeat_seed), task_id
                        )
                        if len(query_idx) < MIN_QUERY_ROWS:
                            row = {'EXP_ID':exp_id,'seed':int(seed),'ood_tier':OOD_TIER_NAME,'ood_task_id':task_id,'K':int(K),'reason':'query_too_small_after_molecule_clean_split','n_total':n_total,'n_unique_mol_keys':n_unique_check,'min_query_rows':MIN_QUERY_ROWS}
                            skipped_rows.append(row)
                            append_df_to_csv(skipped_path, pd.DataFrame([row]), OOD_SKIPPED_COLUMNS)
                            continue
                        support_path = save_support_indices(task_df, int(seed), task_id, int(K), int(repeat_seed), support_idx, query_idx)
                        H_support, y_support = H_all[support_idx], y_all[support_idx]
                        H_query, y_query = H_all[query_idx], y_all[query_idx]
                        meta_query = meta_all.iloc[query_idx].reset_index(drop=True)
                        # Same initialization seed across E1-E9 for a paired comparison.
                        head_seed = int(repeat_seed) * 17 + int(seed) * 31 + int(K) * 1009 + stable_int_hash(task_id)
                        head = fit_fewshot_head(H_support, y_support, head_seed)
                        true_min, pred_min, pred_df = predict_with_fewshot_head(head, H_query, y_query, meta_query)
                        metrics = compute_metric_dict(true_min, pred_min)
                        metrics.update({
                            'EXP_ID':exp_id,'EXP_NAME':meta['EXP_NAME'],'name_short':meta['name_short'],'order':int(meta['order']),
                            'MOLECULE_MODE':meta['MOLECULE_MODE'],'RT_ARCHITECTURE':meta['RT_ARCHITECTURE'],'RT_HEAD_TYPE':meta['RT_HEAD_TYPE'],
                            'USE_DEVICE_METADATA':bool(meta['USE_DEVICE_METADATA']),'JOINT_MULTITASK':bool(meta['JOINT_MULTITASK']),
                            'seed':int(seed),'ood_tier':OOD_TIER_NAME,'ood_task_id':task_id,'ood_method_id':task_id,
                            'K':int(K),'repeat_seed':int(repeat_seed),'repeat_index':int(rep_idx),'protocol':protocol,
                            'mode':'kshot_fresh_head_frozen_model','zero_shot_strategy':'','support_unit':SUPPORT_UNIT,
                            'n_total':n_total,'n_unique_mol_keys':n_unique_check,'n_support':int(len(support_idx)),
                            'n_excluded_same_molecule_rows':int(n_excluded_dup),'support_indices_path':support_path,
                            'checkpoint_path':str(ckpt_path),'checkpoint_name':ckpt_path.name,'checkpoint_epoch':ckpt.get('epoch',np.nan),
                            'checkpoint_best_score':ckpt.get('best_score',np.nan),'model_load_status':getattr(model,'_load_status','unknown'),
                            'ood_head_epochs':int(OOD_HEAD_EPOCHS),'ood_head_lr':float(OOD_HEAD_LR),'ood_head_weight_decay':float(OOD_HEAD_WEIGHT_DECAY),
                        })
                        append_df_to_csv(metrics_path, pd.DataFrame([metrics]), OOD_METRIC_COLUMNS)
                        done_keys.add(key)
                        if SAVE_QUERY_PREDICTIONS:
                            pred_df = pred_df.copy()
                            for c,v in [('EXP_ID',exp_id),('seed',int(seed)),('ood_tier',OOD_TIER_NAME),('ood_task_id',task_id),('K',int(K)),('repeat_seed',int(repeat_seed)),('repeat_index',int(rep_idx)),('protocol',protocol),('mode','kshot_fresh_head_frozen_model'),('support_unit',SUPPORT_UNIT)]:
                                pred_df[c] = v
                            append_df_to_csv(pred_path, pred_df)
                        print(f'{exp_id} seed={seed} dataset={task_id} K={K} MAPE={fmt_metric(metrics["mape_pct"])}% MAE={fmt_metric(metrics["mae_sec"],2)}s nQ={metrics["n_query"]}')
                        del head
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            write_json({'complete':True,'EXP_ID':exp_id,'seed':int(seed),'finished_at':datetime.now().isoformat(timespec='seconds')}, run_dir / 'internal_ood_evaluation_checkpoint_complete.json')
            del model, ckpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            err = traceback.format_exc()
            print(f'[FAILED] {exp_id} seed={seed}: {exc}')
            print(err)
            row = {'EXP_ID':exp_id,'EXP_NAME':meta['EXP_NAME'],'seed':int(seed),'ood_task_id':'','K':np.nan,'checkpoint_path':str(ckpt_path),'error_type':type(exc).__name__,'error':str(exc)}
            failed_rows.append(row)
            append_df_to_csv(failed_path, pd.DataFrame([row]), OOD_FAILED_COLUMNS)
            (run_dir / 'internal_ood_evaluation_failed.txt').write_text(err, encoding='utf-8')
            if FAIL_FAST:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Always create audit files with headers.
if not missing_path.exists():
    pd.DataFrame(columns=OOD_MISSING_COLUMNS).to_csv(missing_path, index=False)
if not failed_path.exists():
    pd.DataFrame(columns=OOD_FAILED_COLUMNS).to_csv(failed_path, index=False)
if not skipped_path.exists():
    pd.DataFrame(columns=OOD_SKIPPED_COLUMNS).to_csv(skipped_path, index=False)

metrics_all = safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS)
missing_df = safe_read_csv(missing_path, columns=OOD_MISSING_COLUMNS)
failed_df = safe_read_csv(failed_path, columns=OOD_FAILED_COLUMNS)
skipped_df = safe_read_csv(skipped_path, columns=OOD_SKIPPED_COLUMNS)

print('\nMetrics shape:', metrics_all.shape)
print('Missing:', missing_df.shape, 'Failed:', failed_df.shape, 'Skipped:', skipped_df.shape)
display(metrics_all.head(30))

# ============================================================
# Cell B6. Per-dataset summaries, task-macro summaries, and ranks
# ============================================================
METRIC_LABELS = {
    'mape_pct':'MAPE (%)', 'r2':'R²', 'mae_sec':'MAE (sec)',
    'rmse_sec':'RMSE (sec)', 'median_ae_sec':'Median AE (sec)', 'spearman':'Spearman'
}
METRIC_COLS = list(METRIC_LABELS)

metrics_all = safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS)
for c in METRIC_COLS + ['K','seed','n_query','n_support','n_total']:
    if c in metrics_all.columns:
        metrics_all[c] = pd.to_numeric(metrics_all[c], errors='coerce')
metrics_all['EXP_ID'] = metrics_all['EXP_ID'].astype(str)
metrics_all['ood_task_id'] = metrics_all['ood_task_id'].astype(str).str.zfill(4)


def summarize_frame(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row['n_runs'] = int(len(g))
        row['n_completed_model_seeds'] = int(g['seed'].nunique())
        row['seeds_present'] = ','.join(map(str, sorted(pd.to_numeric(g['seed'], errors='coerce').dropna().astype(int).unique())))
        row['n_query_mean'] = float(pd.to_numeric(g['n_query'], errors='coerce').mean())
        row['n_support_mean'] = float(pd.to_numeric(g['n_support'], errors='coerce').mean())
        for metric in METRIC_COLS:
            vals = pd.to_numeric(g[metric], errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
            n = len(vals)
            mean = float(vals.mean()) if n else np.nan
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            sem = std / math.sqrt(n) if n > 1 and np.isfinite(std) else np.nan
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
            row[f'{metric}_sem'] = sem
            row[f'{metric}_ci95_low'] = mean - 1.96*sem if np.isfinite(sem) else np.nan
            row[f'{metric}_ci95_high'] = mean + 1.96*sem if np.isfinite(sem) else np.nan
            row[f'{metric}_n'] = int(n)
        rows.append(row)
    return pd.DataFrame(rows)


summary_by_dataset = summarize_frame(
    metrics_all,
    ['ood_tier','ood_task_id','EXP_ID','EXP_NAME','name_short','order','MOLECULE_MODE','RT_ARCHITECTURE','RT_HEAD_TYPE','K']
)
summary_by_dataset['rank_mape'] = summary_by_dataset.groupby(['ood_task_id','K'])['mape_pct_mean'].rank(method='min', ascending=True)
summary_by_dataset['rank_mae'] = summary_by_dataset.groupby(['ood_task_id','K'])['mae_sec_mean'].rank(method='min', ascending=True)
summary_by_dataset['rank_r2'] = summary_by_dataset.groupby(['ood_task_id','K'])['r2_mean'].rank(method='min', ascending=False)
summary_by_dataset = summary_by_dataset.sort_values(['ood_task_id','K','rank_mape','order']).reset_index(drop=True)
summary_by_dataset.to_csv(EVAL_ROOT / 'summary_by_internal_ood_dataset_experiment_K.csv', index=False)

best_by_dataset_k = summary_by_dataset[summary_by_dataset['rank_mape'].eq(1)].copy()
best_by_dataset_k.to_csv(EVAL_ROOT / 'best_model_by_internal_ood_dataset_K_mape.csv', index=False)

# Task-macro: first average each dataset, then give 0097/0238 equal weight.
macro_rows = []
for keys, g in summary_by_dataset.groupby(['EXP_ID','EXP_NAME','name_short','order','MOLECULE_MODE','RT_ARCHITECTURE','RT_HEAD_TYPE','K'], sort=True):
    row = dict(zip(['EXP_ID','EXP_NAME','name_short','order','MOLECULE_MODE','RT_ARCHITECTURE','RT_HEAD_TYPE','K'], keys))
    row['n_ood_tasks'] = int(g['ood_task_id'].nunique())
    row['ood_tasks_present'] = ','.join(sorted(g['ood_task_id'].astype(str).unique()))
    for metric in METRIC_COLS:
        vals = pd.to_numeric(g[f'{metric}_mean'], errors='coerce').dropna()
        n = len(vals)
        mean = float(vals.mean()) if n else np.nan
        std = float(vals.std(ddof=1)) if n > 1 else np.nan
        sem = std / math.sqrt(n) if n > 1 and np.isfinite(std) else np.nan
        row[f'{metric}_task_macro_mean'] = mean
        row[f'{metric}_task_macro_std'] = std
        row[f'{metric}_task_macro_sem'] = sem
        row[f'{metric}_task_macro_ci95_low'] = mean - 1.96*sem if np.isfinite(sem) else np.nan
        row[f'{metric}_task_macro_ci95_high'] = mean + 1.96*sem if np.isfinite(sem) else np.nan
    macro_rows.append(row)

task_macro = pd.DataFrame(macro_rows)
task_macro['rank_mape_task_macro'] = task_macro.groupby('K')['mape_pct_task_macro_mean'].rank(method='min', ascending=True)
task_macro['rank_mae_task_macro'] = task_macro.groupby('K')['mae_sec_task_macro_mean'].rank(method='min', ascending=True)
task_macro['rank_r2_task_macro'] = task_macro.groupby('K')['r2_task_macro_mean'].rank(method='min', ascending=False)
task_macro = task_macro.sort_values(['K','rank_mape_task_macro','order']).reset_index(drop=True)
task_macro.to_csv(EVAL_ROOT / 'summary_internal_ood_task_macro_experiment_K.csv', index=False)

display_cols = ['ood_task_id','K','rank_mape','EXP_ID','name_short','n_completed_model_seeds','mape_pct_mean','mape_pct_std','mae_sec_mean','mae_sec_std','r2_mean','spearman_mean']
print('Best model per dataset and K by MAPE:')
display(best_by_dataset_k[[c for c in display_cols if c in best_by_dataset_k.columns]])
print('\nTask-macro ranking:')
display(task_macro[['K','rank_mape_task_macro','EXP_ID','name_short','mape_pct_task_macro_mean','mae_sec_task_macro_mean','r2_task_macro_mean','spearman_task_macro_mean']])

# ============================================================
# Cell B7. Completion audit and protocol manifest
# ============================================================
metrics_all = safe_read_csv(metrics_path, columns=OOD_METRIC_COLUMNS)
actual_keys = set()
for _, r in metrics_all.iterrows():
    try:
        actual_keys.add((str(r['EXP_ID']), int(r['seed']), str(r['ood_task_id']).zfill(4), int(r['K'])))
    except Exception:
        pass

expected_rows = []
for meta in EXPERIMENTS:
    for seed in SPLIT_SEEDS:
        for task_id in INTERNAL_OOD_METHOD_IDS:
            task_df = internal_ood_df[internal_ood_df['ood_task_id'].eq(task_id)]
            n_total = len(task_df); n_unique = task_df['mol_key'].nunique()
            for K in K_VALUES:
                feasible = (K == 0) or (K <= n_unique and (n_total - K) >= MIN_QUERY_ROWS)
                key = (meta['EXP_ID'], int(seed), task_id, int(K))
                expected_rows.append({
                    'EXP_ID':meta['EXP_ID'],'seed':int(seed),'ood_task_id':task_id,'K':int(K),
                    'n_total':int(n_total),'n_unique_mol_keys':int(n_unique),'feasible':bool(feasible),
                    'completed':key in actual_keys,
                })

completion = pd.DataFrame(expected_rows)
completion.to_csv(EVAL_ROOT / 'internal_ood_kshot_completion_audit.csv', index=False)
missing_expected = completion[completion['feasible'] & ~completion['completed']].copy()
missing_expected.to_csv(EVAL_ROOT / 'internal_ood_kshot_missing_expected_keys.csv', index=False)

manifest = {
    'created_at': datetime.now().isoformat(timespec='seconds'),
    'evaluation_root': str(EVAL_ROOT),
    'internal_ood_method_ids': INTERNAL_OOD_METHOD_IDS,
    'k_values': K_VALUES,
    'model_seeds': SPLIT_SEEDS,
    'experiments': [e['EXP_ID'] for e in EXPERIMENTS],
    'support_unit': SUPPORT_UNIT,
    'paired_support_across_E1_E9': True,
    'k0_single_strategy': 'native shared single head',
    'k0_multi_strategy': 'mean ensemble of all trained seen-method heads',
    'k_positive_strategy': 'freeze complete model/backbone; train fresh two-layer OOD RT head',
    'ood_head_epochs': OOD_HEAD_EPOCHS,
    'ood_head_lr': OOD_HEAD_LR,
    'ood_head_weight_decay': OOD_HEAD_WEIGHT_DECAY,
    'n_expected_feasible': int(completion['feasible'].sum()),
    'n_completed_feasible': int((completion['feasible'] & completion['completed']).sum()),
    'n_missing_feasible': int(len(missing_expected)),
    'all_complete': bool(len(missing_expected) == 0),
}
write_json(manifest, EVAL_ROOT / 'internal_ood_kshot_protocol_manifest.json')

print('Expected feasible keys:', manifest['n_expected_feasible'])
print('Completed feasible keys:', manifest['n_completed_feasible'])
print('Missing feasible keys:', manifest['n_missing_feasible'])
print('ALL COMPLETE:', manifest['all_complete'])
if len(missing_expected):
    display(missing_expected.head(100))
else:
    print('All E1-E9 × 10 seeds × 2 datasets × requested feasible K values are complete.')
print('Outputs:', EVAL_ROOT)


# ============================================================
# FINAL CELL — Internal OOD device/environment + molecule-overlap audit
# Held-out: 0097 / 0238
#
# Audits:
#   1. Whether each held-out chromatographic environment occurs among the 179 RT-training methods
#   2. The nearest training method for each held-out method
#   3. Whether held-out molecules occur in the actual RT-training partition
#   4. Whether held-out molecules occur in any split of the 179-method cohort
#   5. Whether held-out molecules occur in the RadonPy auxiliary dataset
#
# This cell performs NO training and NO model inference.
# ============================================================

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd



# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
HELDOUT_IDS = [
    str(x).zfill(4)
    for x in globals().get(
        "INTERNAL_OOD_METHOD_IDS",
        ["0097", "0238"],
    )
]

AUDIT_SEEDS = [
    int(x)
    for x in globals().get(
        "SPLIT_SEEDS",
        [
            2004, 2006, 2011, 2012, 2016,
            2020, 2022, 2027, 2032, 2034,
        ],
    )
]

REFERENCE_SEED = int(
    globals().get("reference_seed", AUDIT_SEEDS[0])
)

AUDIT_OUT = Path(
    globals().get(
        "EVAL_ROOT",
        "outputs/evaluation_E1_E9_internal_heldout_OOD_kshot",
    )
) / "device_molecule_overlap_audit"

AUDIT_OUT.mkdir(parents=True, exist_ok=True)

print("Held-out IDs:", HELDOUT_IDS)
print("Audit seeds:", AUDIT_SEEDS)
print("Reference seed:", REFERENCE_SEED)
print("Audit output:", AUDIT_OUT.resolve())


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------
def clean_method_id(x):
    match = re.search(r"(\d+)", str(x))
    if match:
        return match.group(1).zfill(4)
    return str(x)


def read_split(seed, filename, split_name):
    split_dir = resolve_split_dir(int(seed))
    path = split_dir / filename

    if not path.exists():
        raise FileNotFoundError(path)

    df = normalize_split_df(
        pd.read_csv(path),
        split_name,
    ).copy()

    df["dir"] = df["dir"].map(clean_method_id)
    df["dataset_id"] = df["dir"]
    df["mol_key"] = df["mol_key"].astype(str)
    df["row_id"] = df["row_id"].astype(str)

    return df


def first_value(x, default="NA"):
    if isinstance(x, (list, tuple, np.ndarray)) and len(x):
        return str(x[0])
    return default


def to_float_list(x):
    values = []

    if not isinstance(x, (list, tuple, np.ndarray)):
        return values

    for value in x:
        try:
            values.append(float(value))
        except Exception:
            values.append(np.nan)

    return values


def rounded_tuple(x, ndigits=6):
    output = []

    for value in to_float_list(x):
        if np.isfinite(value):
            output.append(round(float(value), ndigits))
        else:
            output.append(None)

    return tuple(output)


def unscale(value, denominator):
    try:
        value = float(value)
    except Exception:
        return np.nan

    # -1 represents a missing environment value in this model.
    if not np.isfinite(value) or value < 0:
        return np.nan

    return float(value * denominator)


def normalize_column_name(x):
    if x is None:
        return ""

    text = str(x).strip().lower()

    if text in {"", "nan", "none", "na", "n/a", "__na__"}:
        return ""

    return re.sub(r"\s+", " ", text)


# ------------------------------------------------------------
# Load the reference split
# ------------------------------------------------------------
reference_train = read_split(
    REFERENCE_SEED,
    "train.csv",
    "train",
)

reference_valid = read_split(
    REFERENCE_SEED,
    "valid.csv",
    "valid",
)

reference_internal_test = read_split(
    REFERENCE_SEED,
    "internal_test.csv",
    "internal_test",
)

reference_ood = read_split(
    REFERENCE_SEED,
    "external_ood.csv",
    "external_ood",
)

reference_ood = reference_ood[
    reference_ood["dir"].isin(HELDOUT_IDS)
].copy()

reference_all_179 = pd.concat(
    [
        reference_train.assign(_partition="train"),
        reference_valid.assign(_partition="valid"),
        reference_internal_test.assign(
            _partition="internal_test"
        ),
    ],
    ignore_index=True,
)

actual_train_method_ids = sorted(
    reference_train["dir"].unique().tolist()
)

all_179_method_ids = sorted(
    reference_all_179["dir"].unique().tolist()
)

heldout_method_ids_found = sorted(
    reference_ood["dir"].unique().tolist()
)

print("\nMethod-level audit")
print(
    "Methods with actual RT-training rows:",
    len(actual_train_method_ids),
)
print(
    "Methods in train + valid + internal-test union:",
    len(all_179_method_ids),
)
print(
    "Held-out methods found:",
    heldout_method_ids_found,
)
print(
    "Held-out method overlap with 179:",
    sorted(
        set(heldout_method_ids_found)
        & set(all_179_method_ids)
    ),
)

if heldout_method_ids_found != sorted(HELDOUT_IDS):
    raise RuntimeError(
        "Held-out IDs do not match the expected definition. "
        f"expected={sorted(HELDOUT_IDS)}, "
        f"found={heldout_method_ids_found}"
    )

if set(heldout_method_ids_found) & set(all_179_method_ids):
    raise RuntimeError(
        "Held-out method IDs leaked into the 179-method set."
    )


# ------------------------------------------------------------
# Build device/environment records
#
# The "full environment signature" uses exactly the feature
# groups supplied to the environment encoder:
#
#   column_cont
#   column_cat / phase
#   brand_cat
#   solvent_cont
#   solvent_cat
#   gradient_cont
#   operation_cont
#
# It does not use dataset ID as part of the signature.
# ------------------------------------------------------------
def build_environment_record(method_id):
    method_id = clean_method_id(method_id)

    env = build_selected_env_features(method_id)

    meta_row = get_meta_row(method_id)

    if isinstance(meta_row, pd.DataFrame):
        meta_row = meta_row.iloc[0]

    try:
        column_name = get_column_name(
            method_id,
            meta_row,
        )
    except Exception:
        column_name = ""

    column_cont = to_float_list(
        env.get("column_cont", [])
    )

    solvent_cont = to_float_list(
        env.get("solvent_cont", [])
    )

    gradient_cont = to_float_list(
        env.get("gradient_cont", [])
    )

    operation_cont = to_float_list(
        env.get("operation_cont", [])
    )

    solvent_cat = env.get(
        "solvent_cat",
        ["NA", "NA"],
    )

    phase = first_value(
        env.get("column_cat", ["NA"])
    )

    brand = first_value(
        env.get("brand_cat", ["NA"])
    )

    solvent_A = (
        str(solvent_cat[0])
        if len(solvent_cat) > 0
        else "NA"
    )

    solvent_B = (
        str(solvent_cat[1])
        if len(solvent_cat) > 1
        else "NA"
    )

    column_signature = (
        phase,
        brand,
        rounded_tuple(column_cont),
    )

    full_environment_signature = (
        phase,
        brand,
        solvent_A,
        solvent_B,
        rounded_tuple(column_cont),
        rounded_tuple(solvent_cont),
        rounded_tuple(gradient_cont),
        rounded_tuple(operation_cont),
    )

    gradient_text = []

    for index in range(0, len(gradient_cont), 2):
        if index + 1 >= len(gradient_cont):
            break

        time_scaled = gradient_cont[index]
        percent_scaled = gradient_cont[index + 1]

        if (
            not np.isfinite(time_scaled)
            or not np.isfinite(percent_scaled)
            or time_scaled < 0
            or percent_scaled < 0
        ):
            gradient_text.append("NA")
        else:
            gradient_text.append(
                f"{time_scaled * 60:.3f} min / "
                f"{percent_scaled * 100:.2f}% B"
            )

    return {
        "method_id": method_id,

        "column_name": str(column_name),
        "column_name_normalized": normalize_column_name(
            column_name
        ),

        "brand": brand,
        "phase": phase,
        "family": str(
            env.get("family", "unknown")
        ),

        "solvent_A": solvent_A,
        "solvent_B": solvent_B,
        "solvent_pair": (
            f"{solvent_A}|{solvent_B}"
        ),

        "column_length_mm": (
            unscale(column_cont[0], 250)
            if len(column_cont) > 0
            else np.nan
        ),

        "inner_diameter_mm": (
            unscale(column_cont[1], 4.6)
            if len(column_cont) > 1
            else np.nan
        ),

        "particle_size_um": (
            unscale(column_cont[2], 10)
            if len(column_cont) > 2
            else np.nan
        ),

        "temperature_C": (
            unscale(column_cont[3], 100)
            if len(column_cont) > 3
            else np.nan
        ),

        "t0_min": (
            unscale(column_cont[4], 10)
            if len(column_cont) > 4
            else np.nan
        ),

        "start_B_pct": (
            unscale(solvent_cont[0], 100)
            if len(solvent_cont) > 0
            else np.nan
        ),

        "pH_A": (
            unscale(solvent_cont[1], 14)
            if len(solvent_cont) > 1
            else np.nan
        ),

        "pH_B": (
            unscale(solvent_cont[2], 14)
            if len(solvent_cont) > 2
            else np.nan
        ),

        "flow_mL_min": (
            unscale(operation_cont[0], 2)
            if len(operation_cont) > 0
            else np.nan
        ),

        "gradient_points": " | ".join(
            gradient_text
        ),

        # Private fields used only for comparison.
        "_column_cont": column_cont,
        "_solvent_cont": solvent_cont,
        "_gradient_cont": gradient_cont,
        "_operation_cont": operation_cont,

        "_column_signature": column_signature,
        "_full_environment_signature": (
            full_environment_signature
        ),
    }


training_environment_records = [
    build_environment_record(method_id)
    for method_id in actual_train_method_ids
]

heldout_environment_records = [
    build_environment_record(method_id)
    for method_id in HELDOUT_IDS
]

training_environment_df = pd.DataFrame(
    training_environment_records
)

heldout_environment_df = pd.DataFrame(
    heldout_environment_records
)


# ------------------------------------------------------------
# Environment distance
#
# This is only an audit distance, not a learned model score.
#
# Lower = more similar.
# ------------------------------------------------------------
def calculate_environment_distance(
    heldout_record,
    training_record,
):
    heldout_continuous = np.asarray(
        heldout_record["_column_cont"]
        + heldout_record["_solvent_cont"]
        + heldout_record["_gradient_cont"]
        + heldout_record["_operation_cont"],
        dtype=float,
    )

    training_continuous = np.asarray(
        training_record["_column_cont"]
        + training_record["_solvent_cont"]
        + training_record["_gradient_cont"]
        + training_record["_operation_cont"],
        dtype=float,
    )

    n_features = min(
        len(heldout_continuous),
        len(training_continuous),
    )

    heldout_continuous = heldout_continuous[
        :n_features
    ]

    training_continuous = training_continuous[
        :n_features
    ]

    heldout_missing = (
        ~np.isfinite(heldout_continuous)
        | (heldout_continuous < 0)
    )

    training_missing = (
        ~np.isfinite(training_continuous)
        | (training_continuous < 0)
    )

    both_observed = ~(
        heldout_missing
        | training_missing
    )

    if both_observed.any():
        continuous_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        heldout_continuous[both_observed]
                        - training_continuous[both_observed]
                    )
                    ** 2
                )
            )
        )
    else:
        continuous_rmse = 1.0

    if n_features:
        missing_mismatch_rate = float(
            np.mean(
                heldout_missing
                != training_missing
            )
        )
    else:
        missing_mismatch_rate = 1.0

    categorical_matches = [
        heldout_record["brand"]
        == training_record["brand"],

        heldout_record["phase"]
        == training_record["phase"],

        heldout_record["solvent_A"]
        == training_record["solvent_A"],

        heldout_record["solvent_B"]
        == training_record["solvent_B"],
    ]

    categorical_match_fraction = float(
        np.mean(categorical_matches)
    )

    categorical_mismatch = (
        1.0
        - categorical_match_fraction
    )

    total_distance = (
        continuous_rmse
        + 0.5 * missing_mismatch_rate
        + categorical_mismatch
    )

    return {
        "environment_distance": float(
            total_distance
        ),

        "continuous_rmse": continuous_rmse,

        "missing_mismatch_rate": (
            missing_mismatch_rate
        ),

        "categorical_match_fraction": (
            categorical_match_fraction
        ),

        "same_column_name": bool(
            heldout_record[
                "column_name_normalized"
            ]
            and heldout_record[
                "column_name_normalized"
            ]
            == training_record[
                "column_name_normalized"
            ]
        ),

        "same_brand": bool(
            heldout_record["brand"]
            == training_record["brand"]
        ),

        "same_phase": bool(
            heldout_record["phase"]
            == training_record["phase"]
        ),

        "same_solvent_pair": bool(
            heldout_record["solvent_pair"]
            == training_record["solvent_pair"]
        ),

        "same_encoded_column_signature": bool(
            heldout_record["_column_signature"]
            == training_record[
                "_column_signature"
            ]
        ),

        "same_full_environment_signature": bool(
            heldout_record[
                "_full_environment_signature"
            ]
            == training_record[
                "_full_environment_signature"
            ]
        ),
    }


device_summary_rows = []
nearest_environment_rows = []

training_brand_values = set(
    training_environment_df["brand"]
)

training_phase_values = set(
    training_environment_df["phase"]
)

training_solvent_pairs = set(
    training_environment_df["solvent_pair"]
)


for heldout_record in heldout_environment_records:

    comparisons = []

    for training_record in training_environment_records:

        distance_result = (
            calculate_environment_distance(
                heldout_record,
                training_record,
            )
        )

        comparisons.append({
            "heldout_method_id": (
                heldout_record["method_id"]
            ),

            "training_method_id": (
                training_record["method_id"]
            ),

            "training_column_name": (
                training_record["column_name"]
            ),

            "training_brand": (
                training_record["brand"]
            ),

            "training_phase": (
                training_record["phase"]
            ),

            "training_solvent_pair": (
                training_record["solvent_pair"]
            ),

            **distance_result,
        })

    comparison_df = pd.DataFrame(
        comparisons
    ).sort_values(
        [
            "environment_distance",
            "training_method_id",
        ]
    ).reset_index(drop=True)

    comparison_df["nearest_rank"] = (
        np.arange(
            1,
            len(comparison_df) + 1,
        )
    )

    nearest_environment_rows.extend(
        comparison_df.head(15).to_dict(
            "records"
        )
    )

    heldout_column_name = heldout_record[
        "column_name_normalized"
    ]

    if heldout_column_name:
        same_column_name_mask = (
            training_environment_df[
                "column_name_normalized"
            ].eq(heldout_column_name)
        )
    else:
        same_column_name_mask = pd.Series(
            False,
            index=training_environment_df.index,
        )

    same_brand_phase_mask = (
        training_environment_df["brand"].eq(
            heldout_record["brand"]
        )
        & training_environment_df["phase"].eq(
            heldout_record["phase"]
        )
    )

    same_column_signature_mask = (
        training_environment_df[
            "_column_signature"
        ].map(
            lambda value:
            value
            == heldout_record[
                "_column_signature"
            ]
        )
    )

    same_full_environment_mask = (
        training_environment_df[
            "_full_environment_signature"
        ].map(
            lambda value:
            value
            == heldout_record[
                "_full_environment_signature"
            ]
        )
    )

    nearest_row = comparison_df.iloc[0]

    device_summary_rows.append({
        "heldout_method_id": (
            heldout_record["method_id"]
        ),

        "column_name": (
            heldout_record["column_name"]
        ),

        "brand": heldout_record["brand"],
        "phase": heldout_record["phase"],
        "family": heldout_record["family"],

        "solvent_A": (
            heldout_record["solvent_A"]
        ),

        "solvent_B": (
            heldout_record["solvent_B"]
        ),

        "column_length_mm": (
            heldout_record[
                "column_length_mm"
            ]
        ),

        "inner_diameter_mm": (
            heldout_record[
                "inner_diameter_mm"
            ]
        ),

        "particle_size_um": (
            heldout_record[
                "particle_size_um"
            ]
        ),

        "temperature_C": (
            heldout_record["temperature_C"]
        ),

        "flow_mL_min": (
            heldout_record["flow_mL_min"]
        ),

        "start_B_pct": (
            heldout_record["start_B_pct"]
        ),

        "pH_A": heldout_record["pH_A"],
        "pH_B": heldout_record["pH_B"],

        "gradient_points": (
            heldout_record["gradient_points"]
        ),

        "brand_seen_in_RT_train": (
            heldout_record["brand"]
            in training_brand_values
        ),

        "phase_seen_in_RT_train": (
            heldout_record["phase"]
            in training_phase_values
        ),

        "solvent_pair_seen_in_RT_train": (
            heldout_record["solvent_pair"]
            in training_solvent_pairs
        ),

        "n_same_raw_column_name": int(
            same_column_name_mask.sum()
        ),

        "same_raw_column_name_method_ids": (
            ",".join(
                training_environment_df.loc[
                    same_column_name_mask,
                    "method_id",
                ].astype(str)
            )
        ),

        "n_same_brand_and_phase": int(
            same_brand_phase_mask.sum()
        ),

        "same_brand_and_phase_method_ids": (
            ",".join(
                training_environment_df.loc[
                    same_brand_phase_mask,
                    "method_id",
                ].astype(str)
            )
        ),

        "n_same_encoded_column_signature": int(
            same_column_signature_mask.sum()
        ),

        "same_encoded_column_signature_method_ids": (
            ",".join(
                training_environment_df.loc[
                    same_column_signature_mask,
                    "method_id",
                ].astype(str)
            )
        ),

        "n_same_full_environment_signature": int(
            same_full_environment_mask.sum()
        ),

        "same_full_environment_signature_method_ids": (
            ",".join(
                training_environment_df.loc[
                    same_full_environment_mask,
                    "method_id",
                ].astype(str)
            )
        ),

        "nearest_training_method_id": (
            nearest_row[
                "training_method_id"
            ]
        ),

        "nearest_environment_distance": (
            nearest_row[
                "environment_distance"
            ]
        ),

        "nearest_same_column_name": (
            nearest_row["same_column_name"]
        ),

        "nearest_same_brand": (
            nearest_row["same_brand"]
        ),

        "nearest_same_phase": (
            nearest_row["same_phase"]
        ),

        "nearest_same_solvent_pair": (
            nearest_row[
                "same_solvent_pair"
            ]
        ),
    })


device_summary_df = pd.DataFrame(
    device_summary_rows
)

nearest_environment_df = pd.DataFrame(
    nearest_environment_rows
).sort_values(
    [
        "heldout_method_id",
        "nearest_rank",
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# Molecule overlap across all 10 seeds
#
# actual RT train:
#   The molecule's RT label was used for optimization under at
#   least one of the 179 training methods.
#
# any of 179:
#   The molecule occurs somewhere in train, validation or
#   internal test. This does not necessarily mean its RT label
#   was used for training.
# ------------------------------------------------------------
molecule_overlap_per_seed_rows = []

train_keys_by_seed = {}
all_179_keys_by_seed = {}


for seed in AUDIT_SEEDS:

    train_df = read_split(
        seed,
        "train.csv",
        "train",
    )

    valid_df = read_split(
        seed,
        "valid.csv",
        "valid",
    )

    internal_test_df = read_split(
        seed,
        "internal_test.csv",
        "internal_test",
    )

    ood_df = read_split(
        seed,
        "external_ood.csv",
        "external_ood",
    )

    ood_df = ood_df[
        ood_df["dir"].isin(HELDOUT_IDS)
    ].copy()

    all_179_df = pd.concat(
        [
            train_df,
            valid_df,
            internal_test_df,
        ],
        ignore_index=True,
    )

    train_keys = set(
        train_df["mol_key"].astype(str)
    )

    all_179_keys = set(
        all_179_df["mol_key"].astype(str)
    )

    train_keys_by_seed[seed] = train_keys
    all_179_keys_by_seed[seed] = (
        all_179_keys
    )

    for heldout_id in HELDOUT_IDS:

        heldout_df = ood_df[
            ood_df["dir"].eq(heldout_id)
        ].copy()

        heldout_keys = set(
            heldout_df["mol_key"].astype(str)
        )

        overlap_with_train = (
            heldout_keys
            & train_keys
        )

        overlap_with_all_179 = (
            heldout_keys
            & all_179_keys
        )

        row_seen_in_train = (
            heldout_df["mol_key"]
            .astype(str)
            .isin(train_keys)
        )

        row_seen_in_all_179 = (
            heldout_df["mol_key"]
            .astype(str)
            .isin(all_179_keys)
        )

        n_heldout_unique = len(
            heldout_keys
        )

        n_heldout_rows = len(
            heldout_df
        )

        molecule_overlap_per_seed_rows.append({
            "seed": int(seed),

            "heldout_method_id": (
                heldout_id
            ),

            "n_heldout_rows": int(
                n_heldout_rows
            ),

            "n_heldout_unique_molecules": int(
                n_heldout_unique
            ),

            "n_overlap_unique_with_actual_RT_train": int(
                len(overlap_with_train)
            ),

            "pct_unique_overlap_with_actual_RT_train": (
                100.0
                * len(overlap_with_train)
                / max(n_heldout_unique, 1)
            ),

            "n_rows_seen_in_actual_RT_train": int(
                row_seen_in_train.sum()
            ),

            "pct_rows_seen_in_actual_RT_train": (
                100.0
                * row_seen_in_train.mean()
                if n_heldout_rows
                else np.nan
            ),

            "n_overlap_unique_with_any_of_179": int(
                len(overlap_with_all_179)
            ),

            "pct_unique_overlap_with_any_of_179": (
                100.0
                * len(overlap_with_all_179)
                / max(n_heldout_unique, 1)
            ),

            "n_rows_seen_anywhere_in_179": int(
                row_seen_in_all_179.sum()
            ),

            "pct_rows_seen_anywhere_in_179": (
                100.0
                * row_seen_in_all_179.mean()
                if n_heldout_rows
                else np.nan
            ),

            "n_overlap_only_in_valid_or_internal_test": int(
                len(
                    overlap_with_all_179
                    - overlap_with_train
                )
            ),
        })


molecule_overlap_per_seed_df = pd.DataFrame(
    molecule_overlap_per_seed_rows
)


# ------------------------------------------------------------
# Summarize molecule overlap across seeds
# ------------------------------------------------------------
molecule_summary_rows = []


for heldout_id, group in (
    molecule_overlap_per_seed_df.groupby(
        "heldout_method_id"
    )
):

    result = {
        "heldout_method_id": heldout_id,

        "n_seeds": int(
            group["seed"].nunique()
        ),

        "n_heldout_rows": int(
            group["n_heldout_rows"].iloc[0]
        ),

        "n_heldout_unique_molecules": int(
            group[
                "n_heldout_unique_molecules"
            ].iloc[0]
        ),
    }

    columns_to_summarize = [
        "n_overlap_unique_with_actual_RT_train",
        "pct_unique_overlap_with_actual_RT_train",
        "pct_rows_seen_in_actual_RT_train",

        "n_overlap_unique_with_any_of_179",
        "pct_unique_overlap_with_any_of_179",
        "pct_rows_seen_anywhere_in_179",

        "n_overlap_only_in_valid_or_internal_test",
    ]

    for column in columns_to_summarize:

        values = pd.to_numeric(
            group[column],
            errors="coerce",
        )

        result[f"{column}_mean"] = float(
            values.mean()
        )

        result[f"{column}_std"] = float(
            values.std(ddof=1)
        )

        result[f"{column}_min"] = float(
            values.min()
        )

        result[f"{column}_max"] = float(
            values.max()
        )

    molecule_summary_rows.append(
        result
    )


molecule_summary_df = pd.DataFrame(
    molecule_summary_rows
)


# ------------------------------------------------------------
# Per-molecule details
# ------------------------------------------------------------
all_179_key_to_methods = (
    reference_all_179
    .groupby("mol_key")["dir"]
    .agg(
        lambda values:
        ",".join(
            sorted(
                set(
                    map(str, values)
                )
            )
        )
    )
    .to_dict()
)


# Optional auxiliary exposure.
try:
    radonpy_keys = set(
        load_radon_keys_for_flags()
    )
except Exception:
    radonpy_keys = set()


molecule_detail_rows = []


for heldout_id in HELDOUT_IDS:

    heldout_df = reference_ood[
        reference_ood["dir"].eq(
            heldout_id
        )
    ].copy()

    for mol_key, molecule_group in (
        heldout_df.groupby("mol_key")
    ):

        mol_key = str(mol_key)

        seeds_seen_in_actual_train = [
            int(seed)
            for seed in AUDIT_SEEDS
            if mol_key
            in train_keys_by_seed[seed]
        ]

        molecule_detail_rows.append({
            "heldout_method_id": (
                heldout_id
            ),

            "mol_key": mol_key,

            "representative_smiles": str(
                molecule_group[
                    "smiles"
                ].iloc[0]
            ),

            "n_heldout_rows_for_molecule": int(
                len(molecule_group)
            ),

            "appears_anywhere_in_179": bool(
                mol_key
                in all_179_keys_by_seed[
                    REFERENCE_SEED
                ]
            ),

            "method_ids_in_179": (
                all_179_key_to_methods.get(
                    mol_key,
                    "",
                )
            ),

            "n_of_10_seeds_seen_in_actual_RT_train": int(
                len(
                    seeds_seen_in_actual_train
                )
            ),

            "pct_of_seeds_seen_in_actual_RT_train": (
                100.0
                * len(
                    seeds_seen_in_actual_train
                )
                / len(AUDIT_SEEDS)
            ),

            "seeds_seen_in_actual_RT_train": (
                ",".join(
                    map(
                        str,
                        seeds_seen_in_actual_train,
                    )
                )
            ),

            "appears_in_RadonPy_auxiliary": bool(
                mol_key in radonpy_keys
            )
            if radonpy_keys
            else False,
        })


molecule_detail_df = pd.DataFrame(
    molecule_detail_rows
)


molecule_exposure_df = (
    molecule_detail_df.assign(
        exposure_class=np.select(
            [
                molecule_detail_df[
                    "n_of_10_seeds_seen_in_actual_RT_train"
                ].eq(0),

                molecule_detail_df[
                    "n_of_10_seeds_seen_in_actual_RT_train"
                ].eq(
                    len(AUDIT_SEEDS)
                ),
            ],

            [
                "never_seen_in_RT_train",
                "seen_in_RT_train_all_seeds",
            ],

            default=(
                "seen_in_RT_train_some_seeds"
            ),
        )
    )
    .groupby(
        [
            "heldout_method_id",
            "exposure_class",
        ]
    )
    .agg(
        n_unique_molecules=(
            "mol_key",
            "nunique",
        )
    )
    .reset_index()
)


molecule_exposure_df[
    "pct_within_heldout"
] = (
    molecule_exposure_df[
        "n_unique_molecules"
    ]
    / molecule_exposure_df.groupby(
        "heldout_method_id"
    )[
        "n_unique_molecules"
    ].transform("sum")
    * 100.0
)


# ------------------------------------------------------------
# Save all audit outputs
# ------------------------------------------------------------
device_summary_df.to_csv(
    AUDIT_OUT
    / "heldout_device_overlap_summary.csv",
    index=False,
)

nearest_environment_df.to_csv(
    AUDIT_OUT
    / "heldout_device_nearest_training_methods_top15.csv",
    index=False,
)

heldout_environment_df[
    [
        column
        for column in heldout_environment_df.columns
        if not column.startswith("_")
    ]
].to_csv(
    AUDIT_OUT
    / "heldout_device_environment_details.csv",
    index=False,
)

molecule_overlap_per_seed_df.to_csv(
    AUDIT_OUT
    / "heldout_molecule_overlap_per_seed.csv",
    index=False,
)

molecule_summary_df.to_csv(
    AUDIT_OUT
    / "heldout_molecule_overlap_summary_across_10_seeds.csv",
    index=False,
)

molecule_detail_df.to_csv(
    AUDIT_OUT
    / "heldout_molecule_overlap_per_molecule.csv",
    index=False,
)

molecule_exposure_df.to_csv(
    AUDIT_OUT
    / "heldout_molecule_exposure_classes.csv",
    index=False,
)


with open(
    AUDIT_OUT / "audit_manifest.json",
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        {
            "reference_seed": (
                REFERENCE_SEED
            ),

            "seeds": AUDIT_SEEDS,

            "heldout_ids": HELDOUT_IDS,

            "n_actual_RT_train_methods": (
                len(
                    actual_train_method_ids
                )
            ),

            "n_all_179_methods": (
                len(
                    all_179_method_ids
                )
            ),

            "exact_environment_definition": (
                "Exact equality after the same E1-E9 parser: "
                "column_cont, phase, brand, solvent_cont, "
                "solvent A/B, gradient_cont and operation_cont."
            ),

            "nearest_distance_definition": (
                "continuous RMSE + 0.5 × missingness mismatch "
                "+ categorical mismatch. This is an audit "
                "heuristic, not a learned model score."
            ),

            "molecule_identity": (
                "mol_key from frozen split CSV files"
            ),
        },
        handle,
        ensure_ascii=False,
        indent=2,
    )


# ------------------------------------------------------------
# Display concise results
# ------------------------------------------------------------
display(
    Markdown(
        "## 1. Held-out device/environment "
        "and matches in the 179 RT-training methods"
    )
)

display(
    device_summary_df
)


display(
    Markdown(
        "## 2. Top-10 nearest training methods "
        "in model environment space"
    )
)

display(
    nearest_environment_df[
        nearest_environment_df[
            "nearest_rank"
        ]
        <= 10
    ]
)


display(
    Markdown(
        "## 3. Molecule overlap across "
        "the 10 model seeds"
    )
)

display(
    molecule_summary_df
)


display(
    Markdown(
        "## 4. Held-out molecule "
        "RT-training exposure classes"
    )
)

display(
    molecule_exposure_df
)


display(
    Markdown(
        "## 5. Examples of overlapping molecules"
    )
)

display(
    molecule_detail_df[
        molecule_detail_df[
            "appears_anywhere_in_179"
        ]
    ]
    .sort_values(
        [
            "heldout_method_id",
            "n_of_10_seeds_seen_in_actual_RT_train",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .head(50)
)


print("\nInterpretation guide:")

print(
    "1. n_same_full_environment_signature > 0: "
    "an identical encoded environment was present "
    "during RT training."
)

print(
    "2. Exact match = 0 but nearest_environment_distance "
    "is small: the held-out environment is similar, "
    "but not exactly identical."
)

print(
    "3. actual_RT_train overlap means the model saw "
    "RT labels for the same molecule under another method."
)

print(
    "4. any_of_179 overlap also includes validation and "
    "internal-test molecules whose RT labels were not "
    "used for model optimization."
)

print(
    "5. RadonPy overlap is auxiliary-property exposure, "
    "not RT-label exposure."
)

print(
    "6. These results identify plausible explanations "
    "for strong zero-shot performance, but do not alone "
    "prove causality."
)

print(
    "\nSaved audit files to:",
    AUDIT_OUT.resolve(),
)


# Refresh the two report-facing tables after a successful evaluation.
CURATED_ROOT = PROJECT_ROOT / "result" / "internal_ood"
CURATED_ROOT.mkdir(parents=True, exist_ok=True)
for source_name, target_name in [
    ("summary_by_internal_ood_dataset_experiment_K.csv", "summary.csv"),
    ("internal_ood_kshot_metrics_all_runs.csv", "per_model_seed.csv"),
]:
    source_path = EVAL_ROOT / source_name
    if source_path.exists():
        shutil.copy2(source_path, CURATED_ROOT / target_name)
