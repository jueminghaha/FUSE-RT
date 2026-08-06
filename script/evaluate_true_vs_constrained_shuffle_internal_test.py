#!/usr/bin/env python3
"""Evaluate E2 and the correct/shuffled E8 scaling curves on internal test."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

try:
    from scipy.stats import t as student_t
    from scipy.stats import ttest_rel
except ImportError:
    student_t = None
    ttest_rel = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import engine


SEEDS = [2004, 2006, 2011, 2012, 2016, 2020, 2022, 2027, 2032, 2034]
PERCENTAGES = np.power(10.0, np.linspace(np.log10(0.05), 2.0, 5))
PERCENT_LABELS = ["0.05%", "0.334370%", "2.236068%", "14.953488%", "100%"]

RUN_ROOT = (
    PROJECT_ROOT
    / "result"
    / "_runs"
    / "constrained_shuffle_scaling_internal_test"
)
CURATED_ROOT = (
    PROJECT_ROOT / "result" / "constrained_shuffle_scaling_internal_test"
)
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

LOWER_IS_BETTER = {"mape_pct", "mae_sec", "rmse_sec"}
METRICS = [
    "mape_pct",
    "mae_sec",
    "rmse_sec",
    "r2",
    "spearman",
    "macro_mape_pct",
    "macro_mae_sec",
    "macro_rmse_sec",
    "macro_r2",
    "macro_spearman",
]


def build_point_configs() -> list[dict]:
    points = [
        {
            "point_key": "E2_zero",
            "series_key": "baseline",
            "series_label": "E2 no auxiliary",
            "series_order": 0,
            "axis_step": 0,
            "radonpy_percent": 0.0,
            "display_label": "No aux",
            "model_label": "E2 (M0-R1)",
            "source_dir": CHECKPOINT_ROOT / "experiments" / "E2",
        }
    ]
    for index, (percentage, label) in enumerate(
        zip(PERCENTAGES, PERCENT_LABELS),
        start=1,
    ):
        true_dir = (
            CHECKPOINT_ROOT / "experiments" / "E8"
            if index == 5
            else CHECKPOINT_ROOT / "scaling" / f"E8_p{index:02d}"
        )
        points.append(
            {
                "point_key": f"true_p{index:02d}",
                "series_key": "true",
                "series_label": "Correct PolyOmics labels",
                "series_order": 1,
                "axis_step": index,
                "radonpy_percent": float(percentage),
                "display_label": label,
                "model_label": f"E8 true labels P{index}",
                "source_dir": true_dir,
            }
        )
    for index, (percentage, label) in enumerate(
        zip(PERCENTAGES, PERCENT_LABELS),
        start=1,
    ):
        points.append(
            {
                "point_key": f"shuffle_p{index:02d}",
                "series_key": "shuffle",
                "series_label": "Constrained shuffled labels",
                "series_order": 2,
                "axis_step": index,
                "radonpy_percent": float(percentage),
                "display_label": label,
                "model_label": f"E8 constrained shuffled labels P{index}",
                "source_dir": (
                    CHECKPOINT_ROOT
                    / "constrained_shuffle"
                    / f"E8_p{index:02d}"
                ),
            }
        )
    return points


def find_checkpoint(source_dir: Path, seed: int) -> Optional[Path]:
    run_dir = source_dir / f"seed_{seed}"
    for name in ("best_model.pt", "best_joint.pt", "best_rt.pt", "final_model.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_metadata(checkpoint: dict) -> dict:
    return {
        "checkpoint_EXP_ID": checkpoint.get("EXP_ID", ""),
        "checkpoint_EXP_NAME": checkpoint.get("EXP_NAME", ""),
        "checkpoint_MOLECULE_MODE": checkpoint.get(
            "MOLECULE_MODE",
            checkpoint.get("training_mode", ""),
        ),
        "checkpoint_RT_ARCHITECTURE": checkpoint.get("RT_ARCHITECTURE", ""),
        "checkpoint_use_device_metadata": bool(
            checkpoint.get("USE_DEVICE_METADATA", True)
        ),
        "checkpoint_rt_head_type": checkpoint.get("rt_head_type", "multi"),
        "checkpoint_joint_multitask": bool(
            checkpoint.get("JOINT_MULTITASK", False)
        ),
        "checkpoint_radonpy_percent": float(
            checkpoint.get("radonpy_percent", np.nan)
        ),
        "checkpoint_radonpy_rows_used": int(
            checkpoint.get("radonpy_rows_used", 0) or 0
        ),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_score": float(
            checkpoint.get("best_score", np.nan)
        ),
    }


def method_macro_metrics(predictions: pd.DataFrame) -> dict:
    rows = []
    for dataset_id, group in predictions.groupby("dataset_id", sort=True):
        metrics = engine.compute_metrics_from_arrays(
            group["true_min"].to_numpy(float),
            group["pred_min"].to_numpy(float),
        )
        rows.append(
            {
                "dataset_id": str(dataset_id).zfill(4),
                "n_rows": int(len(group)),
                **metrics,
            }
        )
    per_method = pd.DataFrame(rows)
    macro = {}
    for metric in ("mae_sec", "rmse_sec", "mape_pct", "r2", "spearman"):
        values = (
            pd.to_numeric(per_method.get(metric), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        macro[f"macro_{metric}"] = (
            float(values.mean()) if len(values) else np.nan
        )
    macro["macro_n_methods"] = (
        int(per_method["dataset_id"].nunique()) if len(per_method) else 0
    )
    return macro


def relative_checkpoint_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def evaluate_point_seed(
    point: dict,
    seed: int,
    graph_cache,
    save_predictions: bool,
) -> dict:
    checkpoint_path = find_checkpoint(point["source_dir"], seed)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint for {point['point_key']}, seed={seed}, "
            f"under {point['source_dir']}"
        )
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_seed = int(
        checkpoint.get("split_seed", checkpoint.get("seed", seed))
    )
    if checkpoint_seed != int(seed):
        raise RuntimeError(
            f"Checkpoint seed mismatch: requested {seed}, "
            f"checkpoint contains {checkpoint_seed}."
        )
    expected_exp_id = {
        "baseline": "E2",
        "true": "E8",
        "shuffle": "E8_shuffle",
    }[point["series_key"]]
    checkpoint_exp_id = str(checkpoint.get("EXP_ID", ""))
    if checkpoint_exp_id != expected_exp_id:
        raise RuntimeError(
            f"Checkpoint experiment mismatch for {point['point_key']}: "
            f"expected {expected_exp_id}, found {checkpoint_exp_id!r}."
        )
    checkpoint_percent = float(checkpoint.get("radonpy_percent", np.nan))
    if not np.isclose(
        checkpoint_percent,
        point["radonpy_percent"],
        rtol=0,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"Checkpoint auxiliary percentage mismatch for {point['point_key']}: "
            f"expected {point['radonpy_percent']}, found {checkpoint_percent}."
        )
    checkpoint_architecture = str(checkpoint.get("RT_ARCHITECTURE", ""))
    if checkpoint_architecture != "R1_device_multi":
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {point['point_key']}: "
            f"expected R1_device_multi, found {checkpoint_architecture!r}."
        )

    engine.EXP_ID = str(checkpoint.get("EXP_ID", point["point_key"]))
    engine.EXP_NAME = str(checkpoint.get("EXP_NAME", point["model_label"]))
    engine.MOLECULE_MODE = str(
        checkpoint.get(
            "MOLECULE_MODE",
            checkpoint.get("training_mode", "unknown"),
        )
    )
    engine.RT_ARCHITECTURE = str(
        checkpoint.get("RT_ARCHITECTURE", "unknown")
    )
    engine.USE_DEVICE_METADATA = bool(
        checkpoint.get("USE_DEVICE_METADATA", True)
    )
    engine.RT_HEAD_TYPE = str(checkpoint.get("rt_head_type", "multi"))
    engine.JOINT_MULTITASK = bool(
        checkpoint.get("JOINT_MULTITASK", False)
    )

    evaluation_dir = (
        RUN_ROOT
        / point["series_key"]
        / point["point_key"]
        / f"seed_{seed}"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    data = engine.prepare_rt_split(seed, evaluation_dir, graph_cache)

    checkpoint_y_mean = float(checkpoint["y_mean"])
    checkpoint_y_std = float(checkpoint["y_std"])
    if not np.isclose(engine.y_mean, checkpoint_y_mean, rtol=0, atol=1e-8):
        raise RuntimeError(
            f"y_mean mismatch for {point['point_key']}, seed={seed}."
        )
    if not np.isclose(engine.y_std, checkpoint_y_std, rtol=0, atol=1e-8):
        raise RuntimeError(
            f"y_std mismatch for {point['point_key']}, seed={seed}."
        )

    engine.y_mean = checkpoint_y_mean
    engine.y_std = checkpoint_y_std
    engine.cat_vocabs = checkpoint["cat_vocabs"]
    engine.method_vocab = {
        str(key): int(value)
        for key, value in checkpoint["method_vocab"].items()
    }
    engine.CFG = dict(checkpoint.get("cfg", engine.CFG))

    radon_targets = checkpoint.get("radon_targets", []) or []
    model = engine.GraphEnvRTModel(
        cat_vocabs=engine.cat_vocabs,
        radon_targets=len(radon_targets),
        cfg=engine.CFG,
        head_type=engine.RT_HEAD_TYPE,
        num_methods=len(engine.method_vocab),
        use_device_metadata=engine.USE_DEVICE_METADATA,
    ).to(engine.DEVICE)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load reported incompatibility: {incompatible}"
        )

    metrics, predictions = engine.evaluate_rt(
        model,
        data["test_loader"],
        "internal_test",
        save_predictions=True,
    )
    # The published comparison uses overall and method-macro metrics. The
    # notebook's connectivity-overlap flags mixed connectivity and exact keys;
    # exclude those derived subgroup fields rather than publishing mislabels.
    metrics = {
        key: value
        for key, value in metrics.items()
        if not key.startswith(("clean_overlap_", "nonoverlap_"))
    }
    if predictions is None or predictions.empty:
        raise RuntimeError(
            f"No internal-test predictions for {point['point_key']}, seed={seed}."
        )
    macro = method_macro_metrics(predictions)
    if save_predictions:
        predictions = predictions.drop(
            columns=["clean_radonpy_overlap", "non_radonpy_overlap"],
            errors="ignore",
        )
        predictions.insert(0, "point_key", point["point_key"])
        predictions.insert(1, "axis_step", point["axis_step"])
        predictions.insert(
            2,
            "radonpy_percent_requested",
            point["radonpy_percent"],
        )
        predictions.to_csv(
            evaluation_dir / "predictions_internal_test.csv",
            index=False,
        )

    return {
        "point_key": point["point_key"],
        "axis_step": point["axis_step"],
        "display_label": point["display_label"],
        "model_label": point["model_label"],
        "radonpy_percent_requested": point["radonpy_percent"],
        "seed": int(seed),
        "checkpoint": relative_checkpoint_path(checkpoint_path),
        **checkpoint_metadata(checkpoint),
        **metrics,
        **macro,
        "series_key": point["series_key"],
        "series_label": point["series_label"],
        "series_order": point["series_order"],
    }


def summarize(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "series_key",
        "series_label",
        "series_order",
        "point_key",
        "axis_step",
        "display_label",
        "model_label",
        "radonpy_percent_requested",
        "checkpoint_EXP_ID",
        "checkpoint_RT_ARCHITECTURE",
        "checkpoint_use_device_metadata",
    ]
    rows = []
    for keys, group in seed_metrics.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            standard_deviation = (
                float(values.std(ddof=1)) if count > 1 else np.nan
            )
            standard_error = (
                standard_deviation / math.sqrt(count)
                if count > 1
                else np.nan
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_sem"] = standard_error
            row[f"{metric}_ci95"] = (
                float(student_t.ppf(0.975, count - 1)) * standard_error
                if student_t is not None and count > 1
                else 1.96 * standard_error
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["series_order", "axis_step"])
        .reset_index(drop=True)
    )


def paired_p_value(first: pd.Series, second: pd.Series) -> float:
    if ttest_rel is None or len(first) < 2:
        return np.nan
    return float(ttest_rel(first, second).pvalue)


def paired_vs_e2(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        seed_metrics.loc[seed_metrics["point_key"].eq("E2_zero")]
        .set_index("seed")
        .sort_index()
    )
    if baseline.empty:
        return pd.DataFrame()
    rows = []
    for series_key in ("true", "shuffle"):
        series = seed_metrics.loc[seed_metrics["series_key"].eq(series_key)]
        for point_key, group in series.groupby("point_key", sort=False):
            comparison = group.set_index("seed").sort_index()
            common = baseline.index.intersection(comparison.index)
            for metric in ("mape_pct", "mae_sec", "rmse_sec", "r2", "spearman"):
                base_values = pd.to_numeric(
                    baseline.loc[common, metric],
                    errors="coerce",
                )
                comparison_values = pd.to_numeric(
                    comparison.loc[common, metric],
                    errors="coerce",
                )
                valid = base_values.notna() & comparison_values.notna()
                base_values = base_values[valid]
                comparison_values = comparison_values[valid]
                improvement = (
                    base_values - comparison_values
                    if metric in LOWER_IS_BETTER
                    else comparison_values - base_values
                )
                first = comparison.iloc[0]
                rows.append(
                    {
                        "series_key": series_key,
                        "series_label": first["series_label"],
                        "baseline_point": "E2_zero",
                        "comparison_point": point_key,
                        "axis_step": int(first["axis_step"]),
                        "radonpy_percent": float(
                            first["radonpy_percent_requested"]
                        ),
                        "metric": metric,
                        "n_pairs": int(len(improvement)),
                        "mean_signed_improvement_vs_E2": (
                            float(improvement.mean())
                            if len(improvement)
                            else np.nan
                        ),
                        "sd_signed_improvement_vs_E2": (
                            float(improvement.std(ddof=1))
                            if len(improvement) > 1
                            else np.nan
                        ),
                        "paired_t_p_value": paired_p_value(
                            comparison_values,
                            base_values,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_true_vs_shuffle(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_step in range(1, 6):
        true_frame = (
            seed_metrics.loc[
                seed_metrics["point_key"].eq(f"true_p{axis_step:02d}")
            ]
            .set_index("seed")
            .sort_index()
        )
        shuffle_frame = (
            seed_metrics.loc[
                seed_metrics["point_key"].eq(f"shuffle_p{axis_step:02d}")
            ]
            .set_index("seed")
            .sort_index()
        )
        if true_frame.empty or shuffle_frame.empty:
            continue
        common = true_frame.index.intersection(shuffle_frame.index)
        percentage = float(
            true_frame["radonpy_percent_requested"].iloc[0]
        )
        for metric in ("mape_pct", "mae_sec", "rmse_sec", "r2", "spearman"):
            true_values = pd.to_numeric(
                true_frame.loc[common, metric],
                errors="coerce",
            )
            shuffle_values = pd.to_numeric(
                shuffle_frame.loc[common, metric],
                errors="coerce",
            )
            valid = true_values.notna() & shuffle_values.notna()
            true_values = true_values[valid]
            shuffle_values = shuffle_values[valid]
            improvement = (
                shuffle_values - true_values
                if metric in LOWER_IS_BETTER
                else true_values - shuffle_values
            )
            rows.append(
                {
                    "axis_step": axis_step,
                    "radonpy_percent": percentage,
                    "metric": metric,
                    "n_pairs": int(len(improvement)),
                    "true_minus_shuffle_improvement": (
                        float(improvement.mean()) if len(improvement) else np.nan
                    ),
                    "sd_paired_improvement": (
                        float(improvement.std(ddof=1))
                        if len(improvement) > 1
                        else np.nan
                    ),
                    "paired_t_p_value": paired_p_value(
                        true_values,
                        shuffle_values,
                    ),
                }
            )
    return pd.DataFrame(rows)


def checkpoint_audit(points: list[dict]) -> pd.DataFrame:
    rows = []
    for point in points:
        for seed in SEEDS:
            checkpoint = find_checkpoint(point["source_dir"], seed)
            rows.append(
                {
                    "series_key": point["series_key"],
                    "point_key": point["point_key"],
                    "seed": seed,
                    "checkpoint": (
                        relative_checkpoint_path(checkpoint)
                        if checkpoint is not None
                        else ""
                    ),
                    "checkpoint_exists": checkpoint is not None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate E2, correct-label E8 scaling, and constrained-shuffle "
            "E8 scaling on the 179-method internal test."
        )
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Evaluate available checkpoints instead of failing the audit.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Retain per-row predictions under result/_runs.",
    )
    args = parser.parse_args()

    engine.configure_experiment("E8")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CURATED_ROOT.mkdir(parents=True, exist_ok=True)
    points = build_point_configs()

    audit = checkpoint_audit(points)
    audit.to_csv(RUN_ROOT / "checkpoint_audit.csv", index=False)
    missing = audit.loc[~audit["checkpoint_exists"]]
    if len(missing) and not args.skip_missing:
        raise FileNotFoundError(
            f"{len(missing)} checkpoints are missing; see "
            f"{RUN_ROOT / 'checkpoint_audit.csv'}."
        )

    graph_cache = engine.PrecomputedPyGGraphCache(
        RUN_ROOT / "_shared" / "pyg_graph_cache.pt"
    )
    rows, failures = [], []
    for point in points:
        for seed in SEEDS:
            if find_checkpoint(point["source_dir"], seed) is None:
                continue
            try:
                rows.append(
                    evaluate_point_seed(
                        point,
                        seed,
                        graph_cache,
                        save_predictions=args.save_predictions,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "point_key": point["point_key"],
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if not args.skip_missing:
                    raise

    seed_metrics = pd.DataFrame(rows)
    if seed_metrics.empty:
        raise RuntimeError("No checkpoints were evaluated.")

    summary = summarize(seed_metrics)
    paired_e2 = paired_vs_e2(seed_metrics)
    paired_curves = paired_true_vs_shuffle(seed_metrics)

    seed_metrics.to_csv(CURATED_ROOT / "per_model_seed.csv", index=False)
    summary.to_csv(CURATED_ROOT / "summary.csv", index=False)
    paired_e2.to_csv(CURATED_ROOT / "paired_vs_e2.csv", index=False)
    paired_curves.to_csv(
        CURATED_ROOT / "paired_true_vs_shuffle.csv",
        index=False,
    )
    if failures:
        pd.DataFrame(failures).to_csv(RUN_ROOT / "failed_runs.csv", index=False)
    print(
        f"Evaluated {len(seed_metrics)} point-seed runs; "
        f"curated results: {CURATED_ROOT}"
    )


if __name__ == "__main__":
    main()
