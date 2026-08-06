#!/usr/bin/env python3
"""Evaluate saved E1-E9 checkpoints on validation and original internal-test splits.

This is the script form of ``11_Evaluate_E1_E9_weight_checkpoints.ipynb``.
Model and data definitions are imported from :mod:`model.engine` so that the
training and evaluation paths cannot silently diverge.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import engine


EXPERIMENT_KEYS = [f"E{i}" for i in range(1, 10)]
PREFERRED_CHECKPOINTS = ["best_model.pt", "best_joint.pt", "best_rt.pt", "final_model.pt"]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "experiments"
RUN_ROOT = PROJECT_ROOT / "result" / "_runs" / "internal_test_original"
RESULT_ROOT = PROJECT_ROOT / "result" / "internal_test_original"
RESULT_COLUMNS = [
    "experiment_key",
    "EXP_ID",
    "EXP_NAME",
    "MOLECULE_MODE",
    "RT_ARCHITECTURE",
    "USE_DEVICE_METADATA",
    "RT_HEAD_TYPE",
    "JOINT_MULTITASK",
    "seed",
    "split",
    "checkpoint_path",
    "checkpoint_name",
    "checkpoint_epoch",
    "checkpoint_best_score",
    "model_load_status",
    "missing_keys",
    "unexpected_keys",
    "prediction_path",
    "per_task_metrics_path",
    "mae_sec",
    "median_ae_sec",
    "rmse_sec",
    "mape_pct",
    "mape_n_rows",
    "r2",
    "spearman",
    "n_rows",
    "clean_overlap_mae_sec",
    "clean_overlap_mape_pct",
    "clean_overlap_r2",
    "clean_overlap_spearman",
    "clean_overlap_n_rows",
    "nonoverlap_mae_sec",
    "nonoverlap_mape_pct",
    "nonoverlap_r2",
    "nonoverlap_spearman",
    "nonoverlap_n_rows",
]


def experiment_metadata(experiment_key: str) -> dict[str, Any]:
    cfg = engine.configure_experiment(experiment_key)
    return {
        "experiment_key": experiment_key,
        "EXP_ID": cfg["exp_id"],
        "EXP_NAME": cfg["name"],
        "MOLECULE_MODE": cfg["molecule_mode"],
        "RT_ARCHITECTURE": cfg["rt_architecture"],
        "USE_DEVICE_METADATA": bool(cfg["use_device_metadata"]),
        "RT_HEAD_TYPE": cfg["rt_head_type"],
        "JOINT_MULTITASK": bool(cfg["joint_multitask"]),
    }


def find_checkpoint(experiment_id: str, seed: int) -> Path | None:
    run_dir = CHECKPOINT_ROOT / experiment_id / f"seed_{seed}"
    for name in PREFERRED_CHECKPOINTS:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return next(iter(sorted(run_dir.glob("best*.pt"))), None)


def inferred_radon_targets(checkpoint: dict[str, Any]) -> int:
    targets = checkpoint.get("radon_targets") or []
    count = len(targets) if isinstance(targets, (list, tuple)) else 0
    for key, value in checkpoint.get("model", {}).items():
        if key.endswith("radon_heads.4.weight") and hasattr(value, "shape"):
            count = max(count, int(value.shape[0]))
    return count


def restore_checkpoint_metadata(checkpoint: dict[str, Any], train_df: pd.DataFrame | None = None) -> None:
    if "y_mean" in checkpoint and "y_std" in checkpoint:
        engine.y_mean = float(checkpoint["y_mean"])
        engine.y_std = float(checkpoint["y_std"])
    elif train_df is not None and len(train_df):
        engine.y_mean = float(train_df["rt"].mean())
        engine.y_std = float(train_df["rt"].std(ddof=0))
    else:
        raise RuntimeError("Cannot recover RT normalization statistics from checkpoint or train split.")
    if not np.isfinite(engine.y_std) or engine.y_std < 1e-8:
        engine.y_std = 1.0

    checkpoint_vocabs = checkpoint.get("cat_vocabs")
    if checkpoint_vocabs:
        engine.cat_vocabs = checkpoint_vocabs
    elif train_df is not None:
        engine.cat_vocabs = {
            "column_cat0": engine.collect_vocab(train_df, "column_cat", 0),
            "brand_cat0": engine.collect_vocab(train_df, "brand_cat", 0),
            "solvent_cat0": engine.collect_vocab(train_df, "solvent_cat", 0),
            "solvent_cat1": engine.collect_vocab(train_df, "solvent_cat", 1),
        }
    else:
        raise RuntimeError("Cannot recover categorical vocabularies.")

    checkpoint_methods = checkpoint.get("method_vocab")
    if checkpoint_methods:
        engine.method_vocab = {str(key).zfill(4): int(value) for key, value in checkpoint_methods.items()}
    elif train_df is not None:
        methods = sorted(train_df["dir"].astype(str).unique())
        engine.method_vocab = {method: index for index, method in enumerate(methods)}
    else:
        raise RuntimeError("Cannot recover method vocabulary.")


def add_overlap_flags(frame: pd.DataFrame, train_keys: set[str], radon_keys: set[str]) -> pd.DataFrame:
    frame = frame.copy()
    molecule_keys = frame["mol_key"].astype(str)
    frame["in_radonpy"] = molecule_keys.isin(radon_keys) if radon_keys else False
    frame["appears_in_train"] = molecule_keys.isin(train_keys)
    frame["clean_radonpy_overlap"] = frame["in_radonpy"] & (~frame["appears_in_train"])
    frame["non_radonpy_overlap"] = ~frame["in_radonpy"]
    return frame


def make_loader(frame: pd.DataFrame, graph_cache: Any, batch_size: int) -> Any:
    dataset = engine.RTGraphDataset(frame, graph_cache)
    return engine.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=engine.collate_rt,
        num_workers=int(engine.CFG["num_workers"]),
    )


def prepare_eval_data(
    seed: int,
    checkpoint: dict[str, Any],
    graph_cache: Any,
    run_dir: Path,
) -> dict[str, Any]:
    split_dir = engine.resolve_split_dir(seed)
    frames: dict[str, pd.DataFrame] = {}
    for split_name, filename in (
        ("train", "train.csv"),
        ("valid", "valid.csv"),
        ("internal_test", "internal_test.csv"),
    ):
        frame = pd.read_csv(split_dir / filename)
        frame = engine.normalize_split_df(frame, split_name)
        frames[split_name] = engine.attach_env_features(frame)

    restore_checkpoint_metadata(checkpoint, frames["train"])
    train_keys = set(frames["train"]["mol_key"].astype(str))
    radon_keys = set(engine.load_radon_keys_for_flags())
    for split_name in frames:
        frames[split_name] = add_overlap_flags(frames[split_name], train_keys, radon_keys)

    molecule_sets = {name: set(frame["mol_key"].astype(str)) for name, frame in frames.items()}
    leakage = pd.DataFrame(
        [
            {"check": "train_valid_mol_overlap", "value": len(molecule_sets["train"] & molecule_sets["valid"])},
            {
                "check": "train_internal_test_mol_overlap",
                "value": len(molecule_sets["train"] & molecule_sets["internal_test"]),
            },
            {
                "check": "valid_internal_test_mol_overlap",
                "value": len(molecule_sets["valid"] & molecule_sets["internal_test"]),
            },
        ]
    )
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    leakage.to_csv(audit_dir / "leakage.csv", index=False)
    if (leakage["value"] != 0).any():
        raise RuntimeError(f"Molecule leakage audit failed for seed={seed}:\n{leakage}")

    graph_cache.build(
        pd.concat([frames["valid"]["smiles"], frames["internal_test"]["smiles"]], ignore_index=True),
        save=True,
    )
    batch_size = int(checkpoint.get("cfg", engine.CFG).get("batch_size", engine.CFG["batch_size"]))
    return {
        "split_dir": split_dir,
        "valid": make_loader(frames["valid"], graph_cache, batch_size),
        "internal_test": make_loader(frames["internal_test"], graph_cache, batch_size),
    }


def build_model(checkpoint: dict[str, Any], metadata: dict[str, Any]) -> Any:
    cfg = checkpoint.get("cfg", engine.CFG)
    method_vocab = checkpoint.get("method_vocab") or engine.method_vocab
    model = engine.GraphEnvRTModel(
        cat_vocabs=checkpoint.get("cat_vocabs") or engine.cat_vocabs,
        radon_targets=inferred_radon_targets(checkpoint),
        cfg=cfg,
        head_type=str(checkpoint.get("rt_head_type", metadata["RT_HEAD_TYPE"])),
        num_methods=max(1, len(method_vocab)),
        use_device_metadata=bool(
            checkpoint.get("USE_DEVICE_METADATA", metadata["USE_DEVICE_METADATA"])
        ),
    ).to(engine.DEVICE)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.load_status = "strict"
        model.missing_keys = []
        model.unexpected_keys = []
    except RuntimeError:
        result = model.load_state_dict(checkpoint["model"], strict=False)
        model.load_status = "non_strict"
        model.missing_keys = list(result.missing_keys)
        model.unexpected_keys = list(result.unexpected_keys)
    model.eval()
    return model


def evaluate_checkpoint(
    metadata: dict[str, Any],
    seed: int,
    checkpoint_path: Path,
    graph_cache: Any,
    splits: list[str],
    resume: bool,
) -> list[dict[str, Any]]:
    engine.configure_experiment(metadata["experiment_key"])
    engine.set_all_seeds(seed)
    run_dir = RUN_ROOT / metadata["EXP_ID"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "evaluation_summary.csv"
    complete_path = run_dir / "evaluation_complete.json"
    if resume and summary_path.exists() and complete_path.exists():
        existing = pd.read_csv(summary_path)
        existing_splits = set(existing.get("split", pd.Series(dtype=str)).astype(str))
        if set(splits).issubset(existing_splits):
            return existing.loc[existing["split"].isin(splits)].to_dict("records")

    checkpoint = engine.torch_load_compat(checkpoint_path, map_location="cpu")
    loaders = prepare_eval_data(seed, checkpoint, graph_cache, run_dir)
    model = build_model(checkpoint, metadata)
    rows: list[dict[str, Any]] = []
    for split_name in splits:
        metrics, predictions = engine.evaluate_rt(
            model,
            loaders[split_name],
            split_name=split_name,
            save_predictions=True,
        )
        prediction_path = run_dir / f"predictions_{split_name}.csv"
        predictions.to_csv(prediction_path, index=False)
        engine.save_per_task_metrics(predictions, run_dir, split_name)
        rows.append(
            {
                **metadata,
                "seed": int(seed),
                "split": split_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_epoch": checkpoint.get("epoch", np.nan),
                "checkpoint_best_score": checkpoint.get("best_score", np.nan),
                "model_load_status": model.load_status,
                "missing_keys": ";".join(model.missing_keys),
                "unexpected_keys": ";".join(model.unexpected_keys),
                "prediction_path": str(prediction_path),
                "per_task_metrics_path": str(run_dir / f"per_task_metrics_{split_name}.csv"),
                **metrics,
            }
        )

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    complete_path.write_text(
        json.dumps(
            {
                "complete": True,
                "experiment": metadata["EXP_ID"],
                "seed": int(seed),
                "splits": splits,
                "checkpoint": str(checkpoint_path),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, checkpoint, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "EXP_ID",
        "EXP_NAME",
        "MOLECULE_MODE",
        "RT_ARCHITECTURE",
        "split",
    ]
    metric_columns = ["mae_sec", "median_ae_sec", "rmse_sec", "mape_pct", "r2", "spearman"]
    summary_columns = group_columns + ["n_completed_seeds", "seeds_present", "n_rows_total_mean"]
    for metric in metric_columns:
        summary_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_sem",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
                f"{metric}_n",
            ]
        )
    if metrics.empty:
        return pd.DataFrame(columns=summary_columns)

    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["n_completed_seeds"] = int(group["seed"].nunique())
        row["seeds_present"] = ",".join(map(str, sorted(group["seed"].astype(int).unique())))
        row["n_rows_total_mean"] = float(pd.to_numeric(group["n_rows"], errors="coerce").mean())
        for metric in metric_columns:
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            count = len(values)
            mean = float(values.mean()) if count else np.nan
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            sem = std / math.sqrt(count) if count > 1 and np.isfinite(std) else np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci95_low"] = mean - 1.96 * sem if np.isfinite(sem) else np.nan
            row[f"{metric}_ci95_high"] = mean + 1.96 * sem if np.isfinite(sem) else np.nan
            row[f"{metric}_n"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    experiment_keys: list[str],
    seeds: list[int],
    splits: list[str],
    resume: bool,
    fail_fast: bool,
) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    graph_cache_path = RUN_ROOT / "_shared" / "pyg_graph_cache_eval.pt"
    graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
    graph_cache = engine.PrecomputedPyGGraphCache(graph_cache_path)

    completed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for experiment_key in experiment_keys:
        metadata = experiment_metadata(experiment_key)
        for seed in seeds:
            checkpoint_path = find_checkpoint(metadata["EXP_ID"], seed)
            if checkpoint_path is None:
                missing.append({"EXP_ID": metadata["EXP_ID"], "seed": seed, "reason": "checkpoint_not_found"})
                continue
            try:
                completed.extend(
                    evaluate_checkpoint(metadata, seed, checkpoint_path, graph_cache, splits, resume)
                )
            except Exception as exc:
                error = traceback.format_exc()
                failed.append(
                    {
                        "EXP_ID": metadata["EXP_ID"],
                        "seed": seed,
                        "checkpoint_path": str(checkpoint_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                failure_path = RUN_ROOT / metadata["EXP_ID"] / f"seed_{seed}" / "evaluation_failed.txt"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(error, encoding="utf-8")
                if fail_fast:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    metrics = pd.DataFrame(completed).reindex(columns=RESULT_COLUMNS)
    metrics.to_csv(RESULT_ROOT / "per_model_seed.csv", index=False)
    summarize(metrics).to_csv(RESULT_ROOT / "summary.csv", index=False)
    pd.DataFrame(missing).to_csv(RUN_ROOT / "missing_checkpoints.csv", index=False)
    pd.DataFrame(failed).to_csv(RUN_ROOT / "failed_runs.csv", index=False)
    print(f"Completed rows: {len(metrics)}")
    print(f"Missing checkpoints: {len(missing)}")
    print(f"Failed runs: {len(failed)}")
    print(f"Curated results: {RESULT_ROOT}")


def parse_args() -> argparse.Namespace:
    engine.configure_experiment("E1")
    parser = argparse.ArgumentParser(
        description="Evaluate E1-E9 saved checkpoints on valid and original internal-test splits."
    )
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENT_KEYS, choices=EXPERIMENT_KEYS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(engine.SPLIT_SEEDS))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "internal_test"],
        choices=["valid", "internal_test"],
    )
    parser.add_argument("--force", action="store_true", help="Re-evaluate completed experiment/seed pairs.")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.experiments, args.seeds, args.splits, not args.force, args.fail_fast)


if __name__ == "__main__":
    main()
