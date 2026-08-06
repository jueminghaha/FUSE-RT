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
engine.configure_experiment("E8")
from model.engine import *

def display(value):
    print(value)

class Markdown(str):
    pass

PROJECT_OUTPUT_ROOT = PROJECT_ROOT / 'checkpoints'
ROOT_OUT = PROJECT_ROOT / 'result' / '_runs' / 'scaling_internal_test'
ROOT_OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Cell 8. Six-point grid, source directories, and checkpoint audit
# ============================================================
ZERO_BASELINE_EXP_ID = 'E2'   # requested default; change to 'E2' for pure R1 architecture scaling
RESUME_EVALUATION = True
SKIP_MISSING_CHECKPOINTS = False
SAVE_PREDICTIONS = True

ZERO_BASELINES = {
    'E3': {
        'relative_dir': 'experiments/E3',
        'model_label': 'E3 (M0-R2)',
        'design_note': 'Requested endpoint; architecture differs from E8 scaling points.',
    },
    'E2': {
        'relative_dir': 'experiments/E2',
        'model_label': 'E2 (M0-R1)',
        'design_note': 'Recommended pure-scaling endpoint; RT architecture matches E8.',
    },
}
if ZERO_BASELINE_EXP_ID not in ZERO_BASELINES:
    raise ValueError(f'ZERO_BASELINE_EXP_ID must be one of {list(ZERO_BASELINES)}, got {ZERO_BASELINE_EXP_ID!r}')

positive_percent = np.power(10.0, np.linspace(np.log10(0.05), 2.0, 5))
POINT_CONFIGS = [
    {
        'point_key': f'{ZERO_BASELINE_EXP_ID}_zero',
        'axis_step': 0,
        'radonpy_percent': 0.0,
        'display_label': f'{ZERO_BASELINE_EXP_ID}\n0%',
        'model_label': ZERO_BASELINES[ZERO_BASELINE_EXP_ID]['model_label'],
        'source_dir': PROJECT_OUTPUT_ROOT / ZERO_BASELINES[ZERO_BASELINE_EXP_ID]['relative_dir'],
        'is_zero_baseline': True,
    },
    {
        'point_key': 'E8_p01', 'axis_step': 1, 'radonpy_percent': float(positive_percent[0]),
        'display_label': 'P1\n0.05%', 'model_label': 'E8 scaling P1',
        'source_dir': PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p01', 'is_zero_baseline': False,
    },
    {
        'point_key': 'E8_p02', 'axis_step': 2, 'radonpy_percent': float(positive_percent[1]),
        'display_label': 'P2\n0.334370%', 'model_label': 'E8 scaling P2',
        'source_dir': PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p02', 'is_zero_baseline': False,
    },
    {
        'point_key': 'E8_p03', 'axis_step': 3, 'radonpy_percent': float(positive_percent[2]),
        'display_label': 'P3\n2.236068%', 'model_label': 'E8 scaling P3',
        'source_dir': PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p03', 'is_zero_baseline': False,
    },
    {
        'point_key': 'E8_p04', 'axis_step': 4, 'radonpy_percent': float(positive_percent[3]),
        'display_label': 'P4\n14.953488%', 'model_label': 'E8 scaling P4',
        'source_dir': PROJECT_OUTPUT_ROOT / 'scaling' / 'E8_p04', 'is_zero_baseline': False,
    },
    {
        'point_key': 'E8_full', 'axis_step': 5, 'radonpy_percent': float(positive_percent[4]),
        'display_label': 'E8\n100%', 'model_label': 'E8 full',
        'source_dir': PROJECT_OUTPUT_ROOT / 'experiments' / 'E8', 'is_zero_baseline': False,
    },
]
POINT_DF = pd.DataFrame(POINT_CONFIGS)
POINT_DF['positive_log10_percent'] = np.where(
    POINT_DF['radonpy_percent'] > 0,
    np.log10(POINT_DF['radonpy_percent'].clip(lower=np.finfo(float).tiny)),
    np.nan,
)
POINT_DF['source_exists'] = POINT_DF['source_dir'].map(Path.exists)

def find_checkpoint(source_dir: Path, seed: int) -> Optional[Path]:
    run_dir = Path(source_dir) / f'seed_{seed}'
    for name in ['best_model.pt', 'best_joint.pt', 'best_rt.pt', 'final_model.pt']:
        p = run_dir / name
        if p.exists():
            return p
    return None

audit_rows = []
for point in POINT_CONFIGS:
    for seed in SPLIT_SEEDS:
        ckpt_path = find_checkpoint(point['source_dir'], seed)
        audit_rows.append({
            'point_key': point['point_key'],
            'axis_step': point['axis_step'],
            'radonpy_percent': point['radonpy_percent'],
            'seed': seed,
            'source_dir': str(point['source_dir']),
            'checkpoint': str(ckpt_path) if ckpt_path else None,
            'checkpoint_exists': ckpt_path is not None,
        })
CHECKPOINT_AUDIT = pd.DataFrame(audit_rows)
CHECKPOINT_AUDIT.to_csv(ROOT_OUT / 'checkpoint_audit.csv', index=False)

display(POINT_DF[['axis_step','point_key','radonpy_percent','positive_log10_percent','model_label','source_dir','source_exists']])
display(CHECKPOINT_AUDIT.groupby('point_key', sort=False)['checkpoint_exists'].agg(['sum','count']).reset_index())

missing = CHECKPOINT_AUDIT.loc[~CHECKPOINT_AUDIT['checkpoint_exists']]
if len(missing):
    display(missing)
    if not SKIP_MISSING_CHECKPOINTS:
        raise FileNotFoundError(
            f'{len(missing)} checkpoints are missing. Check PROJECT_OUTPUT_ROOT and the source directory names above.'
        )

if ZERO_BASELINE_EXP_ID == 'E3':
    print('WARNING: E3 is R2/no-device, whereas positive scaling points are E8/R1/device. The first transition is architecture-confounded.')
else:
    print('Pure scaling mode: E2 and E8 use the same R1/device+multi-head RT architecture.')


# ============================================================
# Cell 9. Checkpoint loading and one-point/one-seed internal-test evaluator
# ============================================================
def torch_load_full(path: Path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_metadata(ckpt: dict) -> dict:
    return {
        'checkpoint_EXP_ID': ckpt.get('EXP_ID', ''),
        'checkpoint_EXP_NAME': ckpt.get('EXP_NAME', ''),
        'checkpoint_MOLECULE_MODE': ckpt.get('MOLECULE_MODE', ckpt.get('training_mode', '')),
        'checkpoint_RT_ARCHITECTURE': ckpt.get('RT_ARCHITECTURE', ''),
        'checkpoint_use_device_metadata': bool(ckpt.get('USE_DEVICE_METADATA', True)),
        'checkpoint_rt_head_type': ckpt.get('rt_head_type', 'multi'),
        'checkpoint_joint_multitask': bool(ckpt.get('JOINT_MULTITASK', False)),
        'checkpoint_radonpy_percent': float(ckpt.get('radonpy_percent', np.nan)),
        'checkpoint_radonpy_rows_used': int(ckpt.get('radonpy_rows_used', 0) or 0),
        'checkpoint_epoch': int(ckpt.get('epoch', -1)),
        'checkpoint_best_score': float(ckpt.get('best_score', np.nan)),
    }


def compute_method_macro_metrics(pred_df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    rows = []
    for dataset_id, g in pred_df.groupby('dataset_id', sort=True):
        m = compute_metrics_from_arrays(g['true_min'].to_numpy(float), g['pred_min'].to_numpy(float))
        rows.append({'dataset_id': str(dataset_id).zfill(4), 'n_rows': int(len(g)), **m})
    per_method = pd.DataFrame(rows)
    macro = {}
    for metric in ['mae_sec', 'rmse_sec', 'mape_pct', 'r2', 'spearman']:
        vals = pd.to_numeric(per_method.get(metric), errors='coerce').replace([np.inf, -np.inf], np.nan)
        macro[f'macro_{metric}'] = float(vals.mean()) if vals.notna().any() else np.nan
    macro['macro_n_methods'] = int(per_method['dataset_id'].nunique()) if len(per_method) else 0
    return macro, per_method


def evaluate_point_seed(point: dict, seed: int, graph_cache) -> dict:
    global y_mean, y_std, cat_vocabs, method_vocab, CFG
    global EXP_ID, EXP_NAME, MOLECULE_MODE, RT_ARCHITECTURE, USE_DEVICE_METADATA, RT_HEAD_TYPE, JOINT_MULTITASK

    ckpt_path = find_checkpoint(point['source_dir'], seed)
    if ckpt_path is None:
        raise FileNotFoundError(f'No checkpoint for {point["point_key"]}, seed={seed}, under {point["source_dir"]}')

    eval_seed_dir = ROOT_OUT / point['point_key'] / f'seed_{seed}'
    eval_seed_dir.mkdir(parents=True, exist_ok=True)
    metric_path = eval_seed_dir / 'internal_test_metrics.csv'
    pred_path = eval_seed_dir / 'predictions_internal_test.csv'

    if RESUME_EVALUATION and metric_path.exists() and (pred_path.exists() or not SAVE_PREDICTIONS):
        cached = pd.read_csv(metric_path).iloc[0].to_dict()
        print(f'[SKIP cached] {point["point_key"]} seed={seed}')
        return cached

    set_all_seeds(seed)
    ckpt = torch_load_full(ckpt_path, map_location='cpu')
    meta = checkpoint_metadata(ckpt)

    ckpt_seed = int(ckpt.get('split_seed', ckpt.get('seed', seed)))
    if ckpt_seed != int(seed):
        raise RuntimeError(f'Checkpoint seed mismatch: requested {seed}, checkpoint contains {ckpt_seed}: {ckpt_path}')

    # Make the inherited data audit reflect the actual checkpoint being evaluated.
    EXP_ID = str(ckpt.get('EXP_ID', point['point_key']))
    EXP_NAME = str(ckpt.get('EXP_NAME', point['model_label']))
    MOLECULE_MODE = str(ckpt.get('MOLECULE_MODE', ckpt.get('training_mode', 'unknown')))
    RT_ARCHITECTURE = str(ckpt.get('RT_ARCHITECTURE', 'unknown'))
    USE_DEVICE_METADATA = bool(ckpt.get('USE_DEVICE_METADATA', True))
    RT_HEAD_TYPE = str(ckpt.get('rt_head_type', 'multi'))
    JOINT_MULTITASK = bool(ckpt.get('JOINT_MULTITASK', False))

    # prepare_rt_split reconstructs the exact split and feature tables; the checkpoint values
    # are then restored so normalization/vocabulary indices exactly match training.
    data = prepare_rt_split(seed, eval_seed_dir, graph_cache)

    recomputed_y_mean, recomputed_y_std = float(y_mean), float(y_std)
    checkpoint_y_mean = float(ckpt['y_mean'])
    checkpoint_y_std = float(ckpt['y_std'])
    if not np.isclose(recomputed_y_mean, checkpoint_y_mean, rtol=0, atol=1e-8):
        raise RuntimeError(f'y_mean mismatch for {point["point_key"]} seed={seed}: split={recomputed_y_mean}, checkpoint={checkpoint_y_mean}')
    if not np.isclose(recomputed_y_std, checkpoint_y_std, rtol=0, atol=1e-8):
        raise RuntimeError(f'y_std mismatch for {point["point_key"]} seed={seed}: split={recomputed_y_std}, checkpoint={checkpoint_y_std}')

    y_mean = checkpoint_y_mean
    y_std = checkpoint_y_std
    cat_vocabs = ckpt['cat_vocabs']
    method_vocab = {str(k): int(v) for k, v in ckpt['method_vocab'].items()}
    CFG = dict(ckpt.get('cfg', CFG))

    radon_targets_ckpt = ckpt.get('radon_targets', [])
    n_radon_targets = len(radon_targets_ckpt) if radon_targets_ckpt is not None else 0
    model = GraphEnvRTModel(
        cat_vocabs=cat_vocabs,
        radon_targets=n_radon_targets,
        cfg=CFG,
        head_type=RT_HEAD_TYPE,
        num_methods=len(method_vocab),
        use_device_metadata=USE_DEVICE_METADATA,
    ).to(DEVICE)

    incompatible = model.load_state_dict(ckpt['model'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'Strict checkpoint load unexpectedly reported incompatibility: {incompatible}')

    metrics, pred_df = evaluate_rt(model, data['test_loader'], 'internal_test', save_predictions=True)
    pred_df.insert(0, 'point_key', point['point_key'])
    pred_df.insert(1, 'axis_step', point['axis_step'])
    pred_df.insert(2, 'radonpy_percent_requested', point['radonpy_percent'])
    pred_df.insert(3, 'seed', seed)

    macro_metrics, per_method = compute_method_macro_metrics(pred_df)
    per_method.insert(0, 'point_key', point['point_key'])
    per_method.insert(1, 'axis_step', point['axis_step'])
    per_method.insert(2, 'radonpy_percent_requested', point['radonpy_percent'])
    per_method.insert(3, 'seed', seed)

    if SAVE_PREDICTIONS:
        pred_df.to_csv(pred_path, index=False)
    per_method.to_csv(eval_seed_dir / 'per_method_internal_test_metrics.csv', index=False)

    row = {
        'point_key': point['point_key'],
        'axis_step': int(point['axis_step']),
        'display_label': point['display_label'],
        'model_label': point['model_label'],
        'radonpy_percent_requested': float(point['radonpy_percent']),
        'seed': int(seed),
        'checkpoint': str(ckpt_path),
        **meta,
        **metrics,
        **macro_metrics,
    }
    pd.DataFrame([row]).to_csv(metric_path, index=False)

    del model, data, ckpt, pred_df, per_method
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


# ============================================================
# Cell 10. Run all six points × ten paired split seeds
# ============================================================
shared_cache_path = ROOT_OUT / '_shared' / 'pyg_graph_cache.pt'
shared_cache_path.parent.mkdir(parents=True, exist_ok=True)
shared_graph_cache = PrecomputedPyGGraphCache(shared_cache_path)

result_rows, failed_rows = [], []
for point in POINT_CONFIGS:
    print('\n' + '=' * 110)
    print(f"POINT {point['axis_step']}: {point['point_key']} | requested RadonPy={point['radonpy_percent']:.12g}%")
    print('SOURCE:', point['source_dir'])
    print('=' * 110)
    for seed in SPLIT_SEEDS:
        try:
            result_rows.append(evaluate_point_seed(point, seed, shared_graph_cache))
        except Exception as exc:
            err = traceback.format_exc()
            print(f"[FAILED] {point['point_key']} seed={seed}: {exc}")
            print(err)
            failed_rows.append({
                'point_key': point['point_key'], 'axis_step': point['axis_step'],
                'radonpy_percent_requested': point['radonpy_percent'], 'seed': seed,
                'error_type': type(exc).__name__, 'error': str(exc), 'traceback': err,
            })
            if FAIL_FAST:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

seed_metrics_df = pd.DataFrame(result_rows)
failed_df = pd.DataFrame(failed_rows)
seed_metrics_df.to_csv(ROOT_OUT / 'internal_test_seed_metrics.csv', index=False)
failed_df.to_csv(ROOT_OUT / 'internal_test_failed_runs.csv', index=False)

print('\nCompleted rows:', len(seed_metrics_df), '/', len(POINT_CONFIGS) * len(SPLIT_SEEDS))
display(seed_metrics_df.sort_values(['axis_step','seed']))
if len(failed_df):
    display(failed_df)

if not SKIP_MISSING_CHECKPOINTS and len(seed_metrics_df) != len(POINT_CONFIGS) * len(SPLIT_SEEDS):
    raise RuntimeError('Evaluation is incomplete. Inspect internal_test_failed_runs.csv before interpreting the curve.')


# ============================================================
# Cell 11. Aggregate mean ± SD / 95% CI and paired tests
# ============================================================
seed_metrics_path = ROOT_OUT / 'internal_test_seed_metrics.csv'
seed_metrics_df = pd.read_csv(seed_metrics_path)
seed_metrics_df = seed_metrics_df.sort_values(['axis_step','seed']).reset_index(drop=True)

try:
    from scipy.stats import t as student_t
    from scipy.stats import ttest_rel
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

METRICS = [
    'mape_pct', 'mae_sec', 'rmse_sec', 'r2', 'spearman',
    'macro_mape_pct', 'macro_mae_sec', 'macro_rmse_sec', 'macro_r2', 'macro_spearman',
]
LOWER_IS_BETTER = {'mape_pct','mae_sec','rmse_sec','macro_mape_pct','macro_mae_sec','macro_rmse_sec'}

summary_rows = []
for point_key, g in seed_metrics_df.groupby('point_key', sort=False):
    first = g.iloc[0]
    row = {
        'point_key': point_key,
        'axis_step': int(first['axis_step']),
        'display_label': first['display_label'],
        'model_label': first['model_label'],
        'radonpy_percent_requested': float(first['radonpy_percent_requested']),
        'checkpoint_EXP_ID': first.get('checkpoint_EXP_ID', ''),
        'checkpoint_RT_ARCHITECTURE': first.get('checkpoint_RT_ARCHITECTURE', ''),
        'checkpoint_use_device_metadata': bool(first.get('checkpoint_use_device_metadata', True)),
        'n_seeds': int(g['seed'].nunique()),
    }
    for metric in METRICS:
        vals = pd.to_numeric(g[metric], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(vals))
        mean = float(vals.mean()) if n else np.nan
        sd = float(vals.std(ddof=1)) if n > 1 else np.nan
        sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
        crit = float(student_t.ppf(0.975, df=n-1)) if SCIPY_AVAILABLE and n > 1 else 1.96
        ci95 = float(crit * sem) if n > 1 else np.nan
        row[f'{metric}_mean'] = mean
        row[f'{metric}_std'] = sd
        row[f'{metric}_sem'] = sem
        row[f'{metric}_ci95'] = ci95
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values('axis_step').reset_index(drop=True)
summary_df.to_csv(ROOT_OUT / 'internal_test_scaling_summary.csv', index=False)

# Human-readable table.
report_table = summary_df[['axis_step','point_key','radonpy_percent_requested','n_seeds']].copy()
for metric, digits in [('mape_pct',2),('mae_sec',2),('rmse_sec',2),('r2',4),('spearman',4)]:
    report_table[metric] = summary_df.apply(
        lambda r: f"{r[f'{metric}_mean']:.{digits}f} ± {r[f'{metric}_std']:.{digits}f}", axis=1
    )
display(report_table)

# Paired comparisons against zero baseline.
base_key = summary_df.sort_values('axis_step').iloc[0]['point_key']
base = seed_metrics_df.loc[seed_metrics_df['point_key'].eq(base_key)].set_index('seed')
paired_rows = []
for point_key in summary_df.loc[summary_df['axis_step'] > 0, 'point_key']:
    cur = seed_metrics_df.loc[seed_metrics_df['point_key'].eq(point_key)].set_index('seed')
    common = base.index.intersection(cur.index)
    for metric in ['mape_pct','mae_sec','rmse_sec','r2','spearman']:
        b = pd.to_numeric(base.loc[common, metric], errors='coerce')
        c = pd.to_numeric(cur.loc[common, metric], errors='coerce')
        valid = b.notna() & c.notna()
        # Signed improvement: positive always means current point is better than baseline.
        improvement = (b[valid] - c[valid]) if metric in LOWER_IS_BETTER else (c[valid] - b[valid])
        p_value = float(ttest_rel(c[valid], b[valid]).pvalue) if SCIPY_AVAILABLE and valid.sum() > 1 else np.nan
        paired_rows.append({
            'baseline_point': base_key,
            'comparison_point': point_key,
            'metric': metric,
            'n_pairs': int(valid.sum()),
            'mean_signed_improvement': float(improvement.mean()) if valid.any() else np.nan,
            'sd_signed_improvement': float(improvement.std(ddof=1)) if valid.sum() > 1 else np.nan,
            'paired_t_p_value': p_value,
        })
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(ROOT_OUT / 'paired_comparisons_vs_zero.csv', index=False)
display(paired_df)

# Per-metric best point.
best_rows = []
for metric in ['mape_pct','mae_sec','rmse_sec','r2','spearman','macro_mape_pct','macro_mae_sec','macro_r2','macro_spearman']:
    mean_col = f'{metric}_mean'
    idx = summary_df[mean_col].idxmin() if metric in LOWER_IS_BETTER else summary_df[mean_col].idxmax()
    r = summary_df.loc[idx]
    best_rows.append({
        'metric': metric,
        'best_point': r['point_key'],
        'radonpy_percent': r['radonpy_percent_requested'],
        'mean': r[mean_col],
        'std': r[f'{metric}_std'],
    })
best_df = pd.DataFrame(best_rows)
best_df.to_csv(ROOT_OUT / 'best_point_by_metric.csv', index=False)
display(best_df)


# ============================================================
# Cell 13. Automatic Chinese interpretation and report export
# ============================================================
def fmt_point(row):
    return f"{row['point_key']} ({row['radonpy_percent_requested']:.6g}%)"

def best_row(metric: str):
    col = f'{metric}_mean'
    idx = summary_df[col].idxmin() if metric in LOWER_IS_BETTER else summary_df[col].idxmax()
    return summary_df.loc[idx]

def monotonic_status(metric: str) -> str:
    vals = summary_df[f'{metric}_mean'].to_numpy(float)
    diffs = np.diff(vals)
    if metric in LOWER_IS_BETTER:
        if np.all(diffs <= 0): return 'improves monotonically as the auxiliary fraction increases'
        if np.all(diffs >= 0): return 'degrades monotonically as the auxiliary fraction increases'
    else:
        if np.all(diffs >= 0): return 'improves monotonically as the auxiliary fraction increases'
        if np.all(diffs <= 0): return 'degrades monotonically as the auxiliary fraction increases'
    return 'is not strictly monotonic; an intermediate fraction is optimal or seed-level variability is present'

zero = summary_df.sort_values('axis_step').iloc[0]
full = summary_df.sort_values('axis_step').iloc[-1]

lines = []
lines.append('# E2 → E8 RadonPy Scaling: Automated Internal-Test Report')
lines.append('')
lines.append(f'- Completed points: {summary_df["point_key"].nunique()}; seeds per point: {summary_df["n_seeds"].min()}–{summary_df["n_seeds"].max()}.')
lines.append('- The horizontal axis uses six uniformly spaced log-scale steps: a 0% baseline plus five positive fractions equally spaced in log10 space.')
if ZERO_BASELINE_EXP_ID == 'E3':
    lines.append('- **Design constraint: the 0% point uses E3-R2, whereas positive fractions use E8-R1; the 0%→0.05% contrast therefore includes an architectural effect and does not isolate the causal effect of data scaling.**')
else:
    lines.append('- The 0% point uses E2-R1 and therefore matches the RT architecture used by the positive-fraction E8 points, yielding a controlled auxiliary-data scaling curve.')
lines.append('')
lines.append('## Best Overall Internal-Test Point')
for metric, label, digits in [
    ('mape_pct','MAPE (%)',2), ('mae_sec','MAE (s)',2), ('rmse_sec','RMSE (s)',2),
    ('r2','R²',4), ('spearman','Spearman',4),
]:
    r = best_row(metric)
    lines.append(f"- {label}: {fmt_point(r)}, {r[f'{metric}_mean']:.{digits}f} ± {r[f'{metric}_std']:.{digits}f}; {monotonic_status(metric)}.")

lines.append('')
lines.append('## Endpoint Comparison: 100% versus 0%')
for metric, label, digits in [
    ('mape_pct','MAPE percentage-point improvement',2),
    ('mae_sec','MAE improvement (s)',2),
    ('rmse_sec','RMSE improvement (s)',2),
    ('r2','R² improvement',4),
    ('spearman','Spearman improvement',4),
]:
    improvement = (zero[f'{metric}_mean'] - full[f'{metric}_mean']) if metric in LOWER_IS_BETTER else (full[f'{metric}_mean'] - zero[f'{metric}_mean'])
    lines.append(f'- {label}: {improvement:+.{digits}f} (positive values indicate that the 100% endpoint is better).')

lines.append('')
lines.append('## Method-Equal Macro Results')
for metric, label, digits in [
    ('macro_mape_pct','macro MAPE (%)',2), ('macro_mae_sec','macro MAE (s)',2),
    ('macro_r2','macro R²',4), ('macro_spearman','macro Spearman',4),
]:
    r = best_row(metric)
    lines.append(f"- Best {label}: {fmt_point(r)}, {r[f'{metric}_mean']:.{digits}f} ± {r[f'{metric}_std']:.{digits}f}.")

lines.append('')
lines.append('## Interpretation')
lines.append('- Overall micro metrics are computed over all internal-test observations and therefore assign greater weight to larger datasets.')
lines.append('- Method-macro metrics are first computed within each chromatographic method and then averaged with equal method weights, providing a measure of cross-method generality.')
lines.append('- Summary tables report both the standard deviation across seeds and the 95% confidence interval of the seed mean.')
lines.append('- Claims that a fraction improves upon the baseline should be evaluated using paired_comparisons_vs_zero.csv rather than mean values alone.')

report_text = '\n'.join(lines)
(ROOT_OUT / 'internal_test_scaling_report.md').write_text(report_text, encoding='utf-8')
display(Markdown(report_text))

print('\nSaved outputs:')
for p in [
    ROOT_OUT / 'internal_test_seed_metrics.csv',
    ROOT_OUT / 'internal_test_scaling_summary.csv',
    ROOT_OUT / 'paired_comparisons_vs_zero.csv',
    ROOT_OUT / 'best_point_by_metric.csv',
    ROOT_OUT / 'internal_test_scaling_report.md',
]:
    print(' -', p)


# Refresh the two report-facing tables after a successful evaluation.
CURATED_ROOT = PROJECT_ROOT / "result" / "scaling_internal_test"
CURATED_ROOT.mkdir(parents=True, exist_ok=True)
for source_name, target_name in [
    ("internal_test_scaling_summary.csv", "summary.csv"),
    ("internal_test_seed_metrics.csv", "per_model_seed.csv"),
]:
    source_path = ROOT_OUT / source_name
    if source_path.exists():
        shutil.copy2(source_path, CURATED_ROOT / target_name)
