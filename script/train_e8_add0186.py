#!/usr/bin/env python3
"""Train FUSE-RT E8 on the frozen 180-method split containing method 0186.

This is the train-only counterpart of the original 0186 notebook. It reuses the
shared FUSE-RT model implementation and adds only the protocol specific to the
180-method experiment:

- E8 M2-joint molecular training with the R1 device-aware multihead RT model;
- the frozen 179-method splits extended with method 0186;
- capped-temperature cyclic RT sampling with complete row coverage;
- checkpoint selection and early stopping only after complete RT coverage.

The script writes checkpoints, validation predictions, logs, and data audits
under ``checkpoints/``. It never evaluates the internal-test or external-OOD
splits and never writes curated files under ``result/``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import engine


EXPERIMENT_KEY = "E8_180_add0186"
RUN_VARIANT = "180dataset_add0186_T2_cap20000_cyclic_fullcoverage"
EXPERIMENT_NAME = (
    "M2 joint multitask x R1 device encoder with method-specific heads | "
    "180 methods | T=2 | cap=20000 | cyclic full coverage"
)

EXPECTED_TRAIN_METHODS = 180
ADDED_METHOD_ID = "0186"
DEFAULT_SPLIT_SEEDS = (
    2004,
    2006,
    2011,
    2012,
    2016,
    2020,
    2022,
    2027,
    2032,
    2034,
)

RT_SAMPLING_CONFIG = {
    "rt_sampling_mode": "capped_temperature_cyclic",
    "rt_sampling_temperature": 2.0,
    "rt_sampling_alpha": 0.5,
    "rt_sampling_cap": 20_000,
    "rt_sampler_epoch_budget_mode": "base_179_rows",
    "rt_sampler_num_samples_factor": 1.0,
    "rt_sampler_cycle_without_replacement": True,
    "rt_require_full_train_coverage_before_best_selection": True,
}

_BASE_CHECKPOINT_BUILDER = engine._build_ckpt_payload


def _project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _default_split_root() -> Path:
    paths = json.loads(
        (PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8")
    )
    try:
        configured = Path(paths["split_root_180"])
    except KeyError as exc:
        raise KeyError("config/paths.json must define split_root_180.") from exc
    return _project_path(configured)


def _write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class CappedTemperatureCyclicSampler(Sampler):
    """Sample fixed per-method quotas without repeating rows within a cycle."""

    def __init__(
        self,
        task_series: pd.Series,
        task_quotas: pd.Series,
        seed: int,
    ) -> None:
        self.task_series = task_series.astype(str).reset_index(drop=True)
        self.task_ids = [str(value) for value in task_quotas.index]
        self.task_quotas = {
            str(method_id): int(quota)
            for method_id, quota in task_quotas.items()
        }
        self.num_samples = int(sum(self.task_quotas.values()))
        self.base_seed = int(seed) + 230_000
        self.epoch = 0
        self.last_epoch_audit = pd.DataFrame()

        task_array = self.task_series.to_numpy(dtype=object)
        self.indices_by_task = {
            method_id: np.flatnonzero(task_array == method_id).astype(np.int64)
            for method_id in self.task_ids
        }
        missing = [
            method_id
            for method_id, indices in self.indices_by_task.items()
            if len(indices) == 0
        ]
        if missing:
            raise RuntimeError(
                f"No training rows found for cyclic-sampler methods: {missing}"
            )

        self._rng_by_task = {}
        self._queue_by_task = {}
        self._cursor_by_task = {}
        self._completed_cycles_by_task = {}
        self._seen_by_task = {}
        for method_id in self.task_ids:
            digest = hashlib.sha256(
                f"{self.base_seed}|{method_id}".encode("utf-8")
            ).digest()
            method_seed = int.from_bytes(
                digest[:8],
                byteorder="little",
                signed=False,
            )
            rng = np.random.default_rng(method_seed)
            self._rng_by_task[method_id] = rng
            self._queue_by_task[method_id] = rng.permutation(
                self.indices_by_task[method_id]
            )
            self._cursor_by_task[method_id] = 0
            self._completed_cycles_by_task[method_id] = 0
            self._seen_by_task[method_id] = set()

    def __len__(self) -> int:
        return self.num_samples

    def _draw_from_task(self, method_id: str, quota: int) -> np.ndarray:
        pieces = []
        remaining = int(quota)
        while remaining > 0:
            queue = self._queue_by_task[method_id]
            cursor = int(self._cursor_by_task[method_id])
            available = int(len(queue) - cursor)
            if available == 0:
                self._completed_cycles_by_task[method_id] += 1
                queue = self._rng_by_task[method_id].permutation(
                    self.indices_by_task[method_id]
                )
                self._queue_by_task[method_id] = queue
                cursor = 0
                self._cursor_by_task[method_id] = 0
                available = len(queue)

            take = min(remaining, available)
            pieces.append(queue[cursor : cursor + take])
            self._cursor_by_task[method_id] = cursor + take
            remaining -= take

        return np.concatenate(pieces).astype(np.int64, copy=False)

    def __iter__(self):
        epoch_number = self.epoch + 1
        epoch_indices = []
        audit_rows = []

        for method_id in self.task_ids:
            quota = int(self.task_quotas[method_id])
            drawn = self._draw_from_task(method_id, quota)
            epoch_indices.extend(drawn.tolist())

            unique_drawn = set(int(index) for index in np.unique(drawn))
            self._seen_by_task[method_id].update(unique_drawn)
            n_rows = int(len(self.indices_by_task[method_id]))
            cumulative_unique = int(len(self._seen_by_task[method_id]))
            audit_rows.append(
                {
                    "epoch": int(epoch_number),
                    "method_id": method_id,
                    "quota_rows": quota,
                    "unique_rows_this_epoch": int(len(unique_drawn)),
                    "n_train_rows": n_rows,
                    "cumulative_unique_rows_seen": cumulative_unique,
                    "cumulative_coverage_fraction": (
                        cumulative_unique / max(n_rows, 1)
                    ),
                    "completed_full_cycles": int(
                        self._completed_cycles_by_task[method_id]
                    ),
                    "is_added_0186": method_id == ADDED_METHOD_ID,
                }
            )

        if len(epoch_indices) != self.num_samples:
            raise RuntimeError(
                "Cyclic sampler produced "
                f"{len(epoch_indices)} indices; expected {self.num_samples}."
            )

        mixing_rng = np.random.default_rng(
            self.base_seed + 9_999_991 * epoch_number
        )
        mixing_rng.shuffle(epoch_indices)
        self.epoch = epoch_number
        self.last_epoch_audit = pd.DataFrame(audit_rows)
        return iter(int(index) for index in epoch_indices)


def _allocate_integer_task_quotas(
    task_probabilities: pd.Series,
    num_samples: int,
) -> pd.Series:
    """Allocate integer quotas by largest remainder with one draw per method."""

    if num_samples < len(task_probabilities):
        raise ValueError(
            f"num_samples={num_samples} is smaller than "
            f"n_methods={len(task_probabilities)}."
        )

    raw_quotas = task_probabilities.astype(np.float64) * float(num_samples)
    quotas = np.floor(raw_quotas).astype(np.int64)
    quotas[quotas < 1] = 1
    difference = int(num_samples - quotas.sum())
    fractional = raw_quotas - np.floor(raw_quotas)

    if difference > 0:
        order = fractional.sort_values(
            ascending=False,
            kind="mergesort",
        ).index.tolist()
        cursor = 0
        while difference > 0:
            method_id = order[cursor % len(order)]
            quotas.loc[method_id] += 1
            cursor += 1
            difference -= 1
    elif difference < 0:
        order = fractional.sort_values(
            ascending=True,
            kind="mergesort",
        ).index.tolist()
        cursor = 0
        safety = 0
        while difference < 0:
            method_id = order[cursor % len(order)]
            if quotas.loc[method_id] > 1:
                quotas.loc[method_id] -= 1
                difference += 1
            cursor += 1
            safety += 1
            if safety > 10 * num_samples:
                raise RuntimeError("Failed to reconcile integer method quotas.")

    if int(quotas.sum()) != int(num_samples) or bool((quotas <= 0).any()):
        raise RuntimeError("Invalid integer method-quota allocation.")
    return quotas.astype(np.int64)


def build_capped_temperature_sampler(
    dataset,
    output_dir: Path,
    seed: int,
):
    """Build the deterministic T=2 cyclic sampler and its audit artifacts."""

    if not hasattr(dataset, "df"):
        raise TypeError("The capped-temperature sampler requires dataset.df.")

    cfg = engine.CFG
    mode = str(cfg["rt_sampling_mode"]).strip().lower()
    if mode != "capped_temperature_cyclic":
        raise ValueError(f"Unexpected RT sampling mode: {mode!r}")

    temperature = float(cfg["rt_sampling_temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"rt_sampling_temperature must be positive, got {temperature}."
        )
    alpha = 1.0 / temperature
    configured_alpha = float(cfg["rt_sampling_alpha"])
    if not np.isclose(configured_alpha, alpha, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"T={temperature} implies alpha={alpha}, "
            f"but the configuration contains {configured_alpha}."
        )

    cap = int(cfg["rt_sampling_cap"])
    if cap <= 0:
        raise ValueError(f"rt_sampling_cap must be positive, got {cap}.")
    if not bool(cfg["rt_sampler_cycle_without_replacement"]):
        raise ValueError(
            "The 0186 protocol requires cyclic sampling without replacement."
        )

    task_series = (
        dataset.df["dir"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str.zfill(4)
    )
    if bool(task_series.isna().any()):
        raise RuntimeError("Missing method IDs while building the RT sampler.")

    counts = task_series.value_counts().sort_index().astype(np.int64)
    if len(counts) != EXPECTED_TRAIN_METHODS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_METHODS} RT methods, found {len(counts)}."
        )
    if ADDED_METHOD_ID not in counts:
        raise RuntimeError(
            f"Added method {ADDED_METHOD_ID} is absent from the training split."
        )

    effective_counts = counts.clip(upper=cap).astype(np.float64)
    task_scores = np.power(effective_counts, alpha)
    target_probabilities = task_scores / float(task_scores.sum())

    epoch_budget_mode = str(
        cfg["rt_sampler_epoch_budget_mode"]
    ).strip().lower()
    if epoch_budget_mode == "base_179_rows":
        reference_rows = int(counts.drop(index=ADDED_METHOD_ID).sum())
    elif epoch_budget_mode == "full_180_rows":
        reference_rows = int(len(dataset))
    elif epoch_budget_mode == "sum_capped_rows":
        reference_rows = int(effective_counts.sum())
    else:
        raise ValueError(
            f"Unsupported rt_sampler_epoch_budget_mode={epoch_budget_mode!r}."
        )

    factor = float(cfg["rt_sampler_num_samples_factor"])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(
            f"rt_sampler_num_samples_factor must be positive, got {factor}."
        )
    num_samples = max(1, int(round(reference_rows * factor)))
    task_quotas = _allocate_integer_task_quotas(
        target_probabilities,
        num_samples,
    )
    actual_probabilities = task_quotas.astype(np.float64) / float(num_samples)
    coverage_bounds = np.ceil(
        counts.astype(np.float64) / task_quotas.astype(np.float64)
    ).astype(np.int64)
    all_methods_coverage_epoch = int(coverage_bounds.max())
    added_method_coverage_epoch = int(coverage_bounds.loc[ADDED_METHOD_ID])
    if all_methods_coverage_epoch > engine.RT_EPOCHS:
        raise RuntimeError(
            "The cyclic sampler requires "
            f"{all_methods_coverage_epoch} epochs for complete coverage, "
            f"but RT_EPOCHS={engine.RT_EPOCHS}."
        )

    sampler = CappedTemperatureCyclicSampler(
        task_series=task_series,
        task_quotas=task_quotas,
        seed=seed,
    )

    natural_probabilities = counts.astype(np.float64) / float(counts.sum())
    audit = pd.DataFrame(
        {
            "method_id": counts.index.astype(str),
            "n_train_rows": counts.to_numpy(dtype=np.int64),
            "natural_row_probability": natural_probabilities.to_numpy(
                dtype=np.float64
            ),
            "effective_n_after_cap": effective_counts.to_numpy(
                dtype=np.float64
            ),
            "temperature": temperature,
            "alpha_1_over_temperature": alpha,
            "task_score": task_scores.to_numpy(dtype=np.float64),
            "target_task_probability": target_probabilities.to_numpy(
                dtype=np.float64
            ),
            "quota_rows_per_epoch": task_quotas.to_numpy(dtype=np.int64),
            "actual_task_probability": actual_probabilities.to_numpy(
                dtype=np.float64
            ),
            "expected_exposure_per_original_row_per_epoch": (
                task_quotas / counts.astype(np.float64)
            ).to_numpy(dtype=np.float64),
            "epochs_to_cover_all_rows_once_bound": coverage_bounds.to_numpy(
                dtype=np.int64
            ),
            "sampling_probability_vs_row_uniform": (
                actual_probabilities / natural_probabilities
            ).to_numpy(dtype=np.float64),
        }
    )
    audit["is_added_0186"] = audit["method_id"].eq(ADDED_METHOD_ID)
    audit = audit.sort_values(
        ["n_train_rows", "method_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "rt_capped_temperature_cyclic_distribution.csv"
    config_path = output_dir / "rt_capped_temperature_cyclic_config.json"
    audit.to_csv(audit_path, index=False)
    _write_json(
        {
            "definition": (
                "q_t proportional to min(n_t, cap) ** (1 / temperature); "
                "integer per-method quotas; seed-shuffled cyclic traversal "
                "without row repetition before a complete method cycle"
            ),
            "temperature": temperature,
            "alpha_1_over_temperature": alpha,
            "cap": cap,
            "cycle_without_replacement": True,
            "epoch_budget_mode": epoch_budget_mode,
            "epoch_budget_reference_rows": reference_rows,
            "num_samples_per_epoch": num_samples,
            "num_samples_factor": factor,
            "n_train_rows": int(len(dataset)),
            "n_train_methods": int(len(counts)),
            "added_method": ADDED_METHOD_ID,
            "added_method_train_rows": int(counts.loc[ADDED_METHOD_ID]),
            "added_method_quota_rows_per_epoch": int(
                task_quotas.loc[ADDED_METHOD_ID]
            ),
            "added_method_full_coverage_epoch_bound": (
                added_method_coverage_epoch
            ),
            "all_methods_full_coverage_epoch_bound": (
                all_methods_coverage_epoch
            ),
            "require_full_train_coverage_before_best_selection": bool(
                cfg[
                    "rt_require_full_train_coverage_before_best_selection"
                ]
            ),
            "seed": int(seed),
            "audit_csv": str(audit_path),
        },
        config_path,
    )

    added_row = audit.loc[audit["is_added_0186"]].iloc[0]
    print(
        "RT cyclic sampler:",
        {
            "temperature": temperature,
            "alpha": alpha,
            "cap": cap,
            "epoch_budget_mode": epoch_budget_mode,
            "num_samples_per_epoch": num_samples,
            "batches_per_epoch": int(
                math.ceil(num_samples / engine.CFG["batch_size"])
            ),
            "all_methods_full_coverage_epoch": all_methods_coverage_epoch,
            "method_0186_rows": int(added_row["n_train_rows"]),
            "method_0186_quota": int(added_row["quota_rows_per_epoch"]),
            "method_0186_full_coverage_epoch": added_method_coverage_epoch,
        },
    )
    return {
        "sampler": sampler,
        "audit": audit,
        "audit_path": audit_path,
        "config_path": config_path,
        "all_methods_full_coverage_epoch": all_methods_coverage_epoch,
        "added_0186_full_coverage_epoch": added_method_coverage_epoch,
    }


def _resolve_split_dir(seed: int) -> Path:
    split_dir = Path(engine.SPLIT_ROOT) / f"seed_{seed}"
    required = (
        "train.csv",
        "valid.csv",
        "internal_test.csv",
        "external_ood.csv",
    )
    missing = [name for name in required if not (split_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Frozen 180-method split seed {seed} is incomplete at "
            f"{split_dir}; missing {missing}."
        )
    return split_dir


def _validate_exact_molecular_keys(
    split_name: str,
    frame: pd.DataFrame,
) -> None:
    if "mol_key_exact" not in frame:
        raise KeyError(
            f"{split_name} is missing the required mol_key_exact column."
        )
    missing = frame["mol_key_exact"].isna() | frame[
        "mol_key_exact"
    ].astype(str).str.strip().isin({"", "nan", "None"})
    if bool(missing.any()):
        raise RuntimeError(
            f"{split_name} contains {int(missing.sum())} rows without a "
            "saved exact full InChIKey."
        )
    mismatch = (
        frame["mol_key"].astype(str).str.strip()
        != frame["mol_key_exact"].astype(str).str.strip()
    )
    if bool(mismatch.any()):
        raise RuntimeError(
            f"{split_name} contains {int(mismatch.sum())} rows where mol_key "
            "does not equal the saved exact full InChIKey."
        )


def _attach_overlap_flags(
    frame: pd.DataFrame,
    radon_keys: set[str],
    train_keys: set[str],
) -> pd.DataFrame:
    frame = frame.copy()
    frame["in_radonpy"] = (
        frame["mol_key"].astype(str).isin(radon_keys)
        if radon_keys
        else False
    )
    frame["appears_in_train"] = (
        frame["mol_key"].astype(str).isin(train_keys)
    )
    frame["clean_radonpy_overlap"] = (
        frame["in_radonpy"] & ~frame["appears_in_train"]
    )
    frame["non_radonpy_overlap"] = ~frame["in_radonpy"]
    return frame


def prepare_rt_split(seed: int, run_dir: Path, graph_cache):
    """Load and audit the frozen 180-method split for one seed."""

    split_dir = _resolve_split_dir(seed)
    frames = {
        split_name: engine.normalize_split_df(
            pd.read_csv(split_dir / filename),
            split_name,
        )
        for split_name, filename in {
            "train": "train.csv",
            "valid": "valid.csv",
            "internal_test": "internal_test.csv",
            "external_ood": "external_ood.csv",
        }.items()
    }
    for split_name, frame in frames.items():
        _validate_exact_molecular_keys(split_name, frame)
        frames[split_name] = engine.attach_env_features(frame)

    train_frame = frames["train"]
    engine.y_mean = float(train_frame["rt"].mean())
    engine.y_std = float(train_frame["rt"].std(ddof=0))
    if not np.isfinite(engine.y_std) or engine.y_std < 1e-8:
        engine.y_std = 1.0

    engine.cat_vocabs = {
        "column_cat0": engine.collect_vocab(
            train_frame,
            "column_cat",
            0,
        ),
        "brand_cat0": engine.collect_vocab(
            train_frame,
            "brand_cat",
            0,
        ),
        "solvent_cat0": engine.collect_vocab(
            train_frame,
            "solvent_cat",
            0,
        ),
        "solvent_cat1": engine.collect_vocab(
            train_frame,
            "solvent_cat",
            1,
        ),
    }
    train_methods = sorted(train_frame["dir"].astype(str).unique())
    engine.method_vocab = {
        method_id: index
        for index, method_id in enumerate(train_methods)
    }

    radon_keys = set(engine.load_radon_keys_for_flags())
    train_keys = set(train_frame["mol_key"].astype(str))
    frames = {
        split_name: _attach_overlap_flags(
            frame,
            radon_keys,
            train_keys,
        )
        for split_name, frame in frames.items()
    }
    train_frame = frames["train"]

    train_keys = set(train_frame["mol_key"].astype(str))
    valid_keys = set(frames["valid"]["mol_key"].astype(str))
    internal_test_keys = set(
        frames["internal_test"]["mol_key"].astype(str)
    )
    external_keys = set(frames["external_ood"]["mol_key"].astype(str))
    internal_methods = set().union(
        *(
            set(frames[name]["dir"].astype(str))
            for name in ("train", "valid", "internal_test")
        )
    )
    external_methods = set(frames["external_ood"]["dir"].astype(str))

    leakage_audit = pd.DataFrame(
        [
            {
                "check": "train_valid_exact_mol_overlap",
                "value": len(train_keys & valid_keys),
            },
            {
                "check": "train_test_exact_mol_overlap",
                "value": len(train_keys & internal_test_keys),
            },
            {
                "check": "valid_test_exact_mol_overlap",
                "value": len(valid_keys & internal_test_keys),
            },
            {
                "check": "ood_internal_method_overlap",
                "value": len(external_methods & internal_methods),
            },
            {
                "check": "vocabulary_uses_test_or_ood",
                "value": 0,
            },
        ]
    )
    if bool((leakage_audit["value"] != 0).any()):
        raise RuntimeError(
            f"Internal leakage audit failed for seed={seed}:\n"
            f"{leakage_audit.to_string(index=False)}"
        )

    external_overlap_audit = pd.DataFrame(
        [
            {
                "check": "train_external_exact_mol_overlap",
                "value": len(train_keys & external_keys),
            },
            {
                "check": "valid_external_exact_mol_overlap",
                "value": len(valid_keys & external_keys),
            },
            {
                "check": "internal_test_external_exact_mol_overlap",
                "value": len(internal_test_keys & external_keys),
            },
        ]
    )
    train_external_overlap = int(
        external_overlap_audit.loc[
            external_overlap_audit["check"].eq(
                "train_external_exact_mol_overlap"
            ),
            "value",
        ].iloc[0]
    )
    if train_external_overlap:
        print(
            f"[DATA AUDIT] seed={seed}: external_ood.csv contains "
            f"{train_external_overlap} exact molecules also present in train. "
            "The file is audited but is not evaluated by this training script."
        )

    internal_method_counts = {
        split_name: int(frames[split_name]["dir"].nunique())
        for split_name in ("train", "valid", "internal_test")
    }
    incorrect_counts = {
        split_name: count
        for split_name, count in internal_method_counts.items()
        if count != EXPECTED_TRAIN_METHODS
    }
    if incorrect_counts:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_METHODS} methods in every internal "
            f"split for seed={seed}; found {incorrect_counts}."
        )
    for split_name in ("train", "valid", "internal_test"):
        if ADDED_METHOD_ID not in set(
            frames[split_name]["dir"].astype(str)
        ):
            raise RuntimeError(
                f"Method {ADDED_METHOD_ID} is absent from "
                f"{split_name} for seed={seed}."
            )

    stage0_dir = run_dir / "stage0_data"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    leakage_audit.to_csv(
        stage0_dir / "leakage_audit.csv",
        index=False,
    )
    external_overlap_audit.to_csv(
        stage0_dir / "external_exact_overlap_audit.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": len(frame),
                "methods": frame["dir"].nunique(),
                "mol_keys": frame["mol_key"].nunique(),
            }
            for split_name, frame in frames.items()
        ]
    ).to_csv(stage0_dir / "split_summary.csv", index=False)
    _write_json(engine.cat_vocabs, stage0_dir / "cat_vocabs.json")
    _write_json(
        engine.cat_vocabs.get("brand_cat0", {}),
        stage0_dir / "brand_vocab.json",
    )
    _write_json(engine.method_vocab, stage0_dir / "method_vocab.json")
    _write_json(
        {
            "experiment_key": EXPERIMENT_KEY,
            "EXP_ID": engine.EXP_ID,
            "EXP_NAME": engine.EXP_NAME,
            "RUN_VARIANT": RUN_VARIANT,
            "EXPECTED_TRAIN_METHODS": EXPECTED_TRAIN_METHODS,
            "ADDED_METHOD_ID": ADDED_METHOD_ID,
            "MOLECULE_MODE": engine.MOLECULE_MODE,
            "RT_ARCHITECTURE": engine.RT_ARCHITECTURE,
            "USE_DEVICE_METADATA": engine.USE_DEVICE_METADATA,
            "RT_HEAD_TYPE": engine.RT_HEAD_TYPE,
            "JOINT_MULTITASK": engine.JOINT_MULTITASK,
            "WEIGHTS_ONLY": True,
            "EVALUATE_INTERNAL_TEST": False,
            "EVALUATE_EXTERNAL_OOD": False,
            "CFG": engine.CFG,
            "split_dir": str(split_dir),
            "y_mean": engine.y_mean,
            "y_std": engine.y_std,
        },
        stage0_dir / "experiment_config.json",
    )

    all_smiles = pd.concat(
        [frame["smiles"] for frame in frames.values()],
        ignore_index=True,
    )
    graph_cache.build(all_smiles, save=True)

    datasets = {
        split_name: engine.RTGraphDataset(frame, graph_cache)
        for split_name, frame in frames.items()
    }
    sampler_info = build_capped_temperature_sampler(
        datasets["train"],
        stage0_dir,
        seed,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 240_000)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=engine.CFG["batch_size"],
            sampler=sampler_info["sampler"],
            shuffle=False,
            collate_fn=engine.collate_rt,
            num_workers=engine.CFG["num_workers"],
            generator=loader_generator,
        )
    }
    for split_name in ("valid", "internal_test", "external_ood"):
        loaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=engine.CFG["batch_size"],
            shuffle=False,
            collate_fn=engine.collate_rt,
            num_workers=engine.CFG["num_workers"],
        )

    validation_batch = next(iter(loaders["valid"]))
    if (
        validation_batch["graph"].x_cat.size(0)
        != validation_batch["graph"].batch.size(0)
    ):
        raise RuntimeError("PyG validation batch node alignment failed.")

    return {
        "split_dir": split_dir,
        "train_df": frames["train"],
        "valid_df": frames["valid"],
        "test_df": frames["internal_test"],
        "external_df": frames["external_ood"],
        "train_ds": datasets["train"],
        "valid_ds": datasets["valid"],
        "test_ds": datasets["internal_test"],
        "external_ds": datasets["external_ood"],
        "train_loader": loaders["train"],
        "valid_loader": loaders["valid"],
        "test_loader": loaders["internal_test"],
        "external_loader": loaders["external_ood"],
        "rt_sampler": sampler_info["sampler"],
        "rt_sampling_table": sampler_info["audit"],
        "rt_sampling_audit_path": str(sampler_info["audit_path"]),
        "rt_sampling_config_path": str(sampler_info["config_path"]),
        "all_methods_full_coverage_epoch": int(
            sampler_info["all_methods_full_coverage_epoch"]
        ),
        "added_0186_full_coverage_epoch": int(
            sampler_info["added_0186_full_coverage_epoch"]
        ),
        "external_train_exact_overlap": train_external_overlap,
    }


def _build_checkpoint_payload(
    model,
    epoch: int,
    best_score: float,
    seed: int,
    data: dict,
    radon: dict,
) -> dict:
    payload = _BASE_CHECKPOINT_BUILDER(
        model,
        epoch,
        best_score,
        seed,
        data,
        radon,
    )
    payload.update(
        {
            "experiment_key": EXPERIMENT_KEY,
            "RUN_VARIANT": RUN_VARIANT,
            "EXPECTED_TRAIN_METHODS": EXPECTED_TRAIN_METHODS,
            "ADDED_METHOD_ID": ADDED_METHOD_ID,
            "rt_sampling_audit_path": data["rt_sampling_audit_path"],
            "rt_sampling_config_path": data[
                "rt_sampling_config_path"
            ],
            "internal_test_evaluation": False,
            "ood_evaluation": False,
        }
    )
    return payload


def _save_epoch_coverage(
    audit: pd.DataFrame,
    path: Path,
) -> None:
    audit.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def _completed_run_is_compatible(run_dir: Path, seed: int) -> bool:
    required_paths = (
        run_dir / "run_complete.json",
        run_dir / "run_summary.csv",
        run_dir / engine.BEST_CKPT_NAME,
        run_dir / "best_model.pt",
        run_dir / "final_model.pt",
        run_dir / "best_valid_predictions.csv",
        run_dir / "training_log.csv",
        run_dir / "weights_manifest.json",
        run_dir / "run_manifest.json",
    )
    if not all(path.exists() for path in required_paths):
        return False

    try:
        complete = json.loads(
            (run_dir / "run_complete.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (run_dir / "weights_manifest.json").read_text(encoding="utf-8")
        )
        expected_split_dir = _resolve_split_dir(seed).resolve()
        recorded_split_dir = Path(manifest["split_dir"]).expanduser().resolve()
        cfg = manifest["cfg"]
        checks = (
            complete.get("complete") is True,
            int(manifest["seed"]) == int(seed),
            manifest.get("experiment_key") == EXPERIMENT_KEY,
            manifest.get("run_variant") == RUN_VARIANT,
            int(manifest["dataset_method_count"]) == EXPECTED_TRAIN_METHODS,
            str(manifest["added_method"]).zfill(4) == ADDED_METHOD_ID,
            recorded_split_dir == expected_split_dir,
            manifest.get("weight_only") is True,
            manifest.get("internal_test_evaluation") is False,
            manifest.get("ood_evaluation") is False,
            manifest.get("rt_sampling_mode")
            == RT_SAMPLING_CONFIG["rt_sampling_mode"],
            float(manifest["rt_sampling_temperature"])
            == RT_SAMPLING_CONFIG["rt_sampling_temperature"],
            int(manifest["rt_sampling_cap"])
            == RT_SAMPLING_CONFIG["rt_sampling_cap"],
            int(cfg["rt_sampling_cap"])
            == RT_SAMPLING_CONFIG["rt_sampling_cap"],
            bool(
                cfg[
                    "rt_require_full_train_coverage_before_best_selection"
                ]
            )
            is True,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return all(checks)


def _clear_seed_run_artifacts(run_dir: Path) -> None:
    """Invalidate stale completion/checkpoint files before a fresh run."""

    for filename in (
        "run_complete.json",
        "run_summary.csv",
        engine.BEST_CKPT_NAME,
        "best_model.pt",
        "final_model.pt",
        "best_valid_predictions.csv",
        "training_log.csv",
        "weights_manifest.json",
        "run_manifest.json",
        "rt_sampling_epoch_coverage.csv",
        "run_failed.txt",
    ):
        (run_dir / filename).unlink(missing_ok=True)


def run_one_seed(seed: int, graph_cache) -> dict:
    """Train and checkpoint one split seed without test/OOD evaluation."""

    engine.set_all_seeds(seed)
    run_dir = engine.FRACTION_OUT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "run_complete.json"
    summary_path = run_dir / "run_summary.csv"
    best_path = run_dir / engine.BEST_CKPT_NAME

    if engine.RESUME_COMPLETED and _completed_run_is_compatible(run_dir, seed):
        print(f"[SKIP complete weights] E8 seed={seed}")
        return pd.read_csv(summary_path).iloc[0].to_dict()
    if complete_path.exists():
        print(
            f"[RETRAIN incompatible completion] E8 seed={seed}; "
            "the saved artifacts do not match the active 0186 protocol."
        )
    _clear_seed_run_artifacts(run_dir)

    started_at = time.time()
    print("\n" + "=" * 100)
    print(f"WEIGHT-ONLY RUN E8: {engine.EXP_NAME} | seed={seed}")
    print("=" * 100)

    data = prepare_rt_split(seed, run_dir, graph_cache)
    radon = engine.prepare_radonpy_loaders(
        engine.RADONPY_FRACTION,
        seed,
        graph_cache,
        run_dir,
    )
    engine.radon_train_loader = radon["train_loader"]
    engine.radon_valid_loader = radon["valid_loader"]
    engine.radon_targets = radon["targets"]
    engine.radon_target_transforms = radon["transforms"]
    engine.radon_target_mean = radon["target_mean"]
    engine.radon_target_std = radon["target_std"]

    if engine.radon_train_loader is None:
        raise RuntimeError("E8 joint training requires the RadonPy loader.")

    model = engine.GraphEnvRTModel(
        cat_vocabs=engine.cat_vocabs,
        radon_targets=len(engine.radon_targets),
        cfg=engine.CFG,
        head_type="multi",
        num_methods=len(engine.method_vocab),
        use_device_metadata=True,
    ).to(engine.DEVICE)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = engine.make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=engine.CFG["scheduler_factor"],
        patience=engine.CFG["scheduler_patience"],
        min_lr=engine.CFG["min_lr"],
    )

    require_coverage = bool(
        engine.CFG[
            "rt_require_full_train_coverage_before_best_selection"
        ]
    )
    coverage_guard_epoch = (
        int(data["all_methods_full_coverage_epoch"])
        if require_coverage
        else 1
    )
    print(
        "Best-checkpoint coverage guard:",
        {
            "enabled": require_coverage,
            "all_methods_full_coverage_epoch": data[
                "all_methods_full_coverage_epoch"
            ],
            "added_0186_full_coverage_epoch": data[
                "added_0186_full_coverage_epoch"
            ],
            "selection_starts_at_epoch": coverage_guard_epoch,
        },
    )

    best_score = np.inf
    best_epoch = 0
    no_improve = 0
    log_rows = []
    epoch_coverage_path = run_dir / "rt_sampling_epoch_coverage.csv"
    if epoch_coverage_path.exists():
        epoch_coverage_path.unlink()

    for epoch in range(1, engine.RT_EPOCHS + 1):
        train_stats = engine.train_one_joint_epoch(
            model,
            optimizer,
            data["train_loader"],
            engine.radon_train_loader,
            aux_weight=engine.CFG.get("aux_loss_weight", 1.0),
        )
        train_loss = train_stats["total_loss"]

        sampler_audit = data["rt_sampler"].last_epoch_audit.copy()
        if sampler_audit.empty:
            raise RuntimeError(
                f"Cyclic sampler produced no audit at epoch={epoch}."
            )
        _save_epoch_coverage(sampler_audit, epoch_coverage_path)
        added_rows = sampler_audit.loc[sampler_audit["is_added_0186"]]
        if len(added_rows) != 1:
            raise RuntimeError(
                f"Expected one 0186 coverage row at epoch={epoch}; "
                f"found {len(added_rows)}."
            )
        added_coverage = float(
            added_rows["cumulative_coverage_fraction"].iloc[0]
        )
        minimum_method_coverage = float(
            sampler_audit["cumulative_coverage_fraction"].min()
        )
        all_rows_seen_once = bool(
            (
                sampler_audit["cumulative_coverage_fraction"]
                >= 1.0 - 1e-12
            ).all()
        )

        auxiliary_valid_loss = engine.evaluate_radon(
            model,
            engine.radon_valid_loader,
        )
        valid_metrics, valid_predictions = engine.evaluate_rt(
            model,
            data["valid_loader"],
            "valid",
            save_predictions=True,
        )
        score = engine.valid_score_from_metrics(valid_metrics)
        scheduler.step(score)

        selection_eligible = bool(
            epoch >= coverage_guard_epoch and all_rows_seen_once
        )
        improved = bool(
            selection_eligible and score < best_score - 1e-6
        )
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            checkpoint = _build_checkpoint_payload(
                model,
                epoch,
                best_score,
                seed,
                data,
                radon,
            )
            checkpoint["all_rt_train_rows_seen_once"] = True
            checkpoint["rt_coverage_guard_epoch"] = coverage_guard_epoch
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, run_dir / "best_model.pt")
            valid_predictions.to_csv(
                run_dir / "best_valid_predictions.csv",
                index=False,
            )
        elif selection_eligible:
            no_improve += 1
        else:
            no_improve = 0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "rt_train_loss": train_stats["rt_loss"],
            "aux_train_loss": train_stats["aux_loss"],
            "aux_valid_loss": auxiliary_valid_loss,
            "rt_batches": train_stats["rt_batches"],
            "aux_batches": train_stats["aux_batches"],
            "rt_rows_exposed": train_stats["rt_rows"],
            "aux_rows_exposed": train_stats["aux_rows"],
            "aux_loss_weight": train_stats["aux_loss_weight"],
            "rt_0186_cumulative_coverage": added_coverage,
            "rt_min_task_cumulative_coverage": minimum_method_coverage,
            "rt_all_train_rows_seen_once": all_rows_seen_once,
            "selection_eligible": selection_eligible,
            "coverage_guard_epoch": coverage_guard_epoch,
            "valid_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{
                f"valid_{key}": value
                for key, value in valid_metrics.items()
            },
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(
            run_dir / "training_log.csv",
            index=False,
        )
        print(
            f"[RT {epoch:03d}/{engine.RT_EPOCHS}] "
            f"total={train_loss:.5f} "
            f"rt={row['rt_train_loss']:.5f} "
            f"aux={row['aux_train_loss']:.5f} "
            f"0186_cov={added_coverage:.3f} "
            f"min_cov={minimum_method_coverage:.3f} "
            f"eligible={selection_eligible} "
            f"valid_MAE={valid_metrics['mae_sec']:.2f}s "
            f"score={score:.2f} "
            f"best={best_score:.2f}@{best_epoch}"
        )
        if (
            selection_eligible
            and no_improve >= engine.CFG["early_stop_patience"]
        ):
            print("Early stopping triggered after complete RT coverage.")
            break

    if best_epoch == 0 or not best_path.exists():
        raise RuntimeError(
            "No official best checkpoint was created after complete RT "
            "training-row coverage."
        )

    final_checkpoint = _build_checkpoint_payload(
        model,
        epoch,
        best_score,
        seed,
        data,
        radon,
    )
    final_checkpoint["checkpoint_type"] = "final_model"
    final_checkpoint["all_rt_train_rows_seen_once"] = bool(
        log_rows[-1]["rt_all_train_rows_seen_once"]
    )
    final_checkpoint["rt_coverage_guard_epoch"] = coverage_guard_epoch
    torch.save(final_checkpoint, run_dir / "final_model.pt")

    elapsed_minutes = (time.time() - started_at) / 60.0
    summary = {
        "experiment_key": EXPERIMENT_KEY,
        "EXP_ID": engine.EXP_ID,
        "EXP_NAME": engine.EXP_NAME,
        "MOLECULE_MODE": engine.MOLECULE_MODE,
        "RT_ARCHITECTURE": engine.RT_ARCHITECTURE,
        "use_device_metadata": True,
        "rt_head_type": "multi",
        "joint_multitask": True,
        "weight_only": True,
        "ood_evaluation": False,
        "internal_test_evaluation": False,
        "run_variant": RUN_VARIANT,
        "dataset_method_count": EXPECTED_TRAIN_METHODS,
        "added_method": ADDED_METHOD_ID,
        "rt_sampling_mode": engine.CFG["rt_sampling_mode"],
        "rt_sampling_temperature": engine.CFG[
            "rt_sampling_temperature"
        ],
        "rt_sampling_alpha": engine.CFG["rt_sampling_alpha"],
        "rt_sampling_cap": engine.CFG["rt_sampling_cap"],
        "rt_sampler_epoch_budget_mode": engine.CFG[
            "rt_sampler_epoch_budget_mode"
        ],
        "rt_sampler_num_samples_factor": engine.CFG[
            "rt_sampler_num_samples_factor"
        ],
        "rt_sampling_audit_path": data["rt_sampling_audit_path"],
        "rt_sampling_config_path": data["rt_sampling_config_path"],
        "rt_sampling_epoch_coverage_path": str(epoch_coverage_path),
        "all_methods_full_coverage_epoch": int(
            data["all_methods_full_coverage_epoch"]
        ),
        "added_0186_full_coverage_epoch": int(
            data["added_0186_full_coverage_epoch"]
        ),
        "best_checkpoint_requires_full_rt_coverage": require_coverage,
        "external_train_exact_overlap": int(
            data["external_train_exact_overlap"]
        ),
        "radonpy_percent": engine.RADONPY_PERCENT,
        "radonpy_fraction": engine.RADONPY_FRACTION,
        "seed": seed,
        "split_dir": str(data["split_dir"]),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_score),
        "radonpy_rows_used": int(radon["n_train_used"]),
        "radonpy_rows_full_train": int(radon["n_train_full"]),
        "radonpy_valid_rows": int(radon["n_valid"]),
        "elapsed_min": elapsed_minutes,
        "best_checkpoint": str(best_path),
        "best_model_alias": str(run_dir / "best_model.pt"),
        "final_checkpoint": str(run_dir / "final_model.pt"),
        "training_log": str(run_dir / "training_log.csv"),
        "best_valid_predictions": str(
            run_dir / "best_valid_predictions.csv"
        ),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    cap = int(engine.CFG["rt_sampling_cap"])
    temperature = float(engine.CFG["rt_sampling_temperature"])
    manifest = {
        **summary,
        "cfg": engine.CFG,
        "split_seeds": list(engine.SPLIT_SEEDS),
        "radon_targets": engine.radon_targets,
        "radon_target_transforms": dict(
            zip(
                engine.radon_targets,
                engine.radon_target_transforms,
            )
        ),
        "radonpy_row_split": (
            "90/10 by row; nested training fraction prefix"
        ),
        "brand_vocab_policy": (
            "training split only; unseen validation/test brands map to <UNK>"
        ),
        "rt_sampling_policy": (
            f"q_t proportional to min(n_t,{cap})^(1/{temperature:g}); "
            "exact integer per-method quotas; seed-shuffled cyclic traversal "
            "without repetition before a complete method cycle; "
            "num_samples equals the base-179 train rows; official checkpoint "
            "selection starts only after every RT training row is observed"
        ),
    }
    _write_json(manifest, run_dir / "weights_manifest.json")
    _write_json(manifest, run_dir / "run_manifest.json")
    _write_json(
        {"complete": True, **summary},
        complete_path,
    )

    del model, optimizer, scheduler, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _aggregate_summary(completed: pd.DataFrame) -> pd.DataFrame:
    row = {
        "experiment_key": EXPERIMENT_KEY,
        "EXP_ID": engine.EXP_ID,
        "EXP_NAME": engine.EXP_NAME,
        "RUN_VARIANT": RUN_VARIANT,
        "dataset_method_count": EXPECTED_TRAIN_METHODS,
        "added_method": ADDED_METHOD_ID,
        "rt_sampling_temperature": engine.CFG[
            "rt_sampling_temperature"
        ],
        "rt_sampling_alpha": engine.CFG["rt_sampling_alpha"],
        "rt_sampling_cap": engine.CFG["rt_sampling_cap"],
        "rt_sampler_epoch_budget_mode": engine.CFG[
            "rt_sampler_epoch_budget_mode"
        ],
        "rt_sampler_cycle_without_replacement": engine.CFG[
            "rt_sampler_cycle_without_replacement"
        ],
        "rt_require_full_train_coverage_before_best_selection": (
            engine.CFG[
                "rt_require_full_train_coverage_before_best_selection"
            ]
        ),
        "n_completed_seeds": (
            int(completed["seed"].nunique()) if len(completed) else 0
        ),
        "weight_only": True,
        "internal_test_evaluation": False,
        "ood_evaluation": False,
    }
    for column in ("best_valid_score", "best_epoch", "elapsed_min"):
        values = (
            pd.to_numeric(completed[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            if column in completed
            else pd.Series(dtype=float)
        )
        row[f"{column}_mean"] = (
            float(values.mean()) if len(values) else np.nan
        )
        row[f"{column}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else np.nan
        )
        row[f"{column}_min"] = (
            float(values.min()) if len(values) else np.nan
        )
        row[f"{column}_max"] = (
            float(values.max()) if len(values) else np.nan
        )
    return pd.DataFrame([row])


def _completion_audit(seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        run_dir = engine.FRACTION_OUT / f"seed_{seed}"
        rows.append(
            {
                "EXP_ID": engine.EXP_ID,
                "seed": seed,
                "best_checkpoint_exists": (
                    run_dir / engine.BEST_CKPT_NAME
                ).exists(),
                "best_model_alias_exists": (
                    run_dir / "best_model.pt"
                ).exists(),
                "final_model_exists": (
                    run_dir / "final_model.pt"
                ).exists(),
                "training_log_exists": (
                    run_dir / "training_log.csv"
                ).exists(),
                "weights_manifest_exists": (
                    run_dir / "weights_manifest.json"
                ).exists(),
                "run_complete_exists": (
                    run_dir / "run_complete.json"
                ).exists(),
            }
        )
    return pd.DataFrame(rows)


def run_all_seeds(seeds: Sequence[int]) -> bool:
    """Train selected seeds and write checkpoint-local completion metadata."""

    graph_cache = engine.PrecomputedPyGGraphCache(
        engine.OUT_DIR / "pyg_graph_cache.pt"
    )
    completed_rows = []
    failed_rows = []
    for seed in seeds:
        try:
            completed_rows.append(run_one_seed(seed, graph_cache))
        except Exception as exc:
            error = traceback.format_exc()
            print(f"[FAILED] E8 seed={seed}: {exc}\n{error}")
            failure_dir = engine.FRACTION_OUT / f"seed_{seed}"
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "run_failed.txt").write_text(
                error,
                encoding="utf-8",
            )
            failed_rows.append(
                {
                    "EXP_ID": engine.EXP_ID,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if engine.FAIL_FAST:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed = pd.DataFrame(completed_rows)
    failed = pd.DataFrame(
        failed_rows,
        columns=("EXP_ID", "seed", "error_type", "error"),
    )
    completed.to_csv(
        engine.FRACTION_OUT / "weights_run_summary_all_seeds.csv",
        index=False,
    )
    completed.to_csv(
        engine.FRACTION_OUT / "experiment_run_summary_all_seeds.csv",
        index=False,
    )
    failed.to_csv(
        engine.FRACTION_OUT / "experiment_failed_runs.csv",
        index=False,
    )

    aggregate = _aggregate_summary(completed)
    aggregate.to_csv(
        engine.FRACTION_OUT / "weights_summary.csv",
        index=False,
    )
    aggregate.to_csv(
        engine.FRACTION_OUT / "summary_metrics.csv",
        index=False,
    )

    checkpoint_audit = _completion_audit(seeds)
    checkpoint_audit.to_csv(
        engine.FRACTION_OUT / "weights_completion_audit.csv",
        index=False,
    )
    checkpoint_audit.to_csv(
        engine.FRACTION_OUT / "experiment_completion_audit.csv",
        index=False,
    )
    required_columns = [
        "best_checkpoint_exists",
        "best_model_alias_exists",
        "final_model_exists",
        "training_log_exists",
        "weights_manifest_exists",
        "run_complete_exists",
    ]
    all_complete = bool(
        len(checkpoint_audit) == len(seeds)
        and checkpoint_audit[required_columns].all().all()
        and len(failed) == 0
    )
    _write_json(
        {
            "experiment_key": EXPERIMENT_KEY,
            "EXP_ID": engine.EXP_ID,
            "EXP_NAME": engine.EXP_NAME,
            "RUN_VARIANT": RUN_VARIANT,
            "dataset_method_count": EXPECTED_TRAIN_METHODS,
            "added_method": ADDED_METHOD_ID,
            "rt_sampling_temperature": engine.CFG[
                "rt_sampling_temperature"
            ],
            "rt_sampling_alpha": engine.CFG["rt_sampling_alpha"],
            "rt_sampling_cap": engine.CFG["rt_sampling_cap"],
            "rt_sampler_epoch_budget_mode": engine.CFG[
                "rt_sampler_epoch_budget_mode"
            ],
            "rt_sampler_cycle_without_replacement": engine.CFG[
                "rt_sampler_cycle_without_replacement"
            ],
            "rt_require_full_train_coverage_before_best_selection": (
                engine.CFG[
                    "rt_require_full_train_coverage_before_best_selection"
                ]
            ),
            "expected_seeds": list(seeds),
            "n_completed": int(
                checkpoint_audit["run_complete_exists"].sum()
            ),
            "all_complete": all_complete,
            "weight_only": True,
            "internal_test_evaluation": False,
            "ood_evaluation": False,
        },
        engine.FRACTION_OUT / "weights_complete.json",
    )
    print(
        f"Completed={len(completed)}, failed={len(failed)}, "
        f"all_complete={all_complete}, output={engine.FRACTION_OUT}"
    )
    return all_complete


def configure_engine(
    *,
    split_root: Path,
    output_dir: Path,
    seeds: Sequence[int],
    cuda_device_index: int,
    fail_fast: bool,
    force: bool,
) -> None:
    """Apply the fixed 0186 training protocol to the shared E8 engine."""

    if not split_root.is_dir():
        raise FileNotFoundError(
            f"The frozen 180-method split root does not exist: {split_root}\n"
            "Generate it with data/extend_split_0186.py or pass --split-root."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate split seeds are not allowed.")

    engine.configure_experiment("E8")
    engine.EXP_KEY = EXPERIMENT_KEY
    engine.EXP_ID = "E8"
    engine.EXP_NAME = EXPERIMENT_NAME
    engine.MOLECULE_MODE = "M2_joint"
    engine.RT_ARCHITECTURE = "R1_device_multi"
    engine.USE_DEVICE_METADATA = True
    engine.RT_HEAD_TYPE = "multi"
    engine.JOINT_MULTITASK = True
    engine.WEIGHTS_ONLY = True
    engine.EVALUATE_INTERNAL_TEST = False
    engine.EVALUATE_EXTERNAL_OOD = False
    engine.RADONPY_PERCENT = 100.0
    engine.RADONPY_FRACTION = 1.0
    engine.RADONPY_PRETRAIN_EPOCHS = 0
    engine.SPLIT_ROOT = split_root
    engine.SPLIT_SEEDS = list(seeds)
    engine.RESUME_COMPLETED = not force
    engine.FAIL_FAST = bool(fail_fast)
    engine.BEST_CKPT_NAME = "best_joint.pt"
    engine.ROOT_OUT = output_dir
    engine.FRACTION_OUT = output_dir
    engine.OUT_DIR = output_dir / "_shared"
    engine.OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine.CFG.update(RT_SAMPLING_CONFIG)
    engine.CFG["aux_loss_weight"] = 1.0

    if cuda_device_index < 0:
        raise ValueError("--cuda-device-index must be nonnegative.")
    engine.CUDA_DEVICE_INDEX = int(cuda_device_index)
    if torch.cuda.is_available():
        if cuda_device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {cuda_device_index} is unavailable; "
                f"detected {torch.cuda.device_count()} devices."
            )
        engine.DEVICE = torch.device(f"cuda:{cuda_device_index}")
    else:
        engine.DEVICE = torch.device("cpu")

    print(
        {
            "experiment": EXPERIMENT_KEY,
            "device": str(engine.DEVICE),
            "split_root": str(engine.SPLIT_ROOT),
            "output_dir": str(engine.FRACTION_OUT),
            "seeds": list(engine.SPLIT_SEEDS),
            "expected_methods": EXPECTED_TRAIN_METHODS,
            "added_method": ADDED_METHOD_ID,
            "weight_only": True,
            "internal_test_evaluation": False,
            "ood_evaluation": False,
            **RT_SAMPLING_CONFIG,
        }
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train E8 on the frozen 180-method split containing method 0186. "
            "The script saves weights and validation artifacts only."
        )
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help=(
            "Frozen 180-method split root. Defaults to split_root_180 in "
            "config/paths.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "experiments"
            / f"E8_{RUN_VARIANT}"
        ),
        help="Checkpoint-local output directory.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPLIT_SEEDS),
        help="Split seeds to train.",
    )
    parser.add_argument(
        "--cuda-device-index",
        type=int,
        default=engine.CUDA_DEVICE_INDEX,
        help="CUDA device index; CPU is used when CUDA is unavailable.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain seeds even when complete checkpoint artifacts exist.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    split_root = (
        _default_split_root()
        if args.split_root is None
        else _project_path(args.split_root)
    )
    output_dir = _project_path(args.output_dir)
    configure_engine(
        split_root=split_root,
        output_dir=output_dir,
        seeds=args.seeds,
        cuda_device_index=args.cuda_device_index,
        fail_fast=args.fail_fast,
        force=args.force,
    )
    return 0 if run_all_seeds(args.seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
