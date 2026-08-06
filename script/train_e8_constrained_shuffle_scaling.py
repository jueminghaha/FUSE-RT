#!/usr/bin/env python3
"""Train the five-point E8 constrained shuffled-label scaling control.

Only the selected auxiliary-training labels are permuted. Molecular inputs,
RepoRT splits, auxiliary validation labels, and all optimization settings are
identical to the corresponding E8 correct-label scaling experiments.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from tqdm.auto import tqdm

try:
    from rdkit.Chem import rdFingerprintGenerator

    _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except (ImportError, AttributeError):
    from rdkit.Chem import AllChem

    _MORGAN_GENERATOR = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import engine


POINTS = {
    "p01": {
        "base_experiment": "E8_p01",
        "percent": 0.05,
        "name": "E8 constrained shuffled-label control at 0.05% auxiliary data",
    },
    "p02": {
        "base_experiment": "E8_p02",
        "percent": 0.334370152488211,
        "name": "E8 constrained shuffled-label control at 0.334370152488% auxiliary data",
    },
    "p03": {
        "base_experiment": "E8_p03",
        "percent": 2.23606797749979,
        "name": "E8 constrained shuffled-label control at 2.2360679775% auxiliary data",
    },
    "p04": {
        "base_experiment": "E8_p04",
        "percent": 14.9534878122122,
        "name": "E8 constrained shuffled-label control at 14.9534878122% auxiliary data",
    },
    "p05": {
        "base_experiment": "E8",
        "percent": 100.0,
        "name": "E8 constrained shuffled-label control at 100% auxiliary data",
    },
}

AUX_LABEL_CONTROL = "constrained_rowwise_derangement"
AUX_SHUFFLE_MAX_TANIMOTO = 0.50
AUX_SHUFFLE_FP_RADIUS = 2
AUX_SHUFFLE_FP_BITS = 2048
AUX_SHUFFLE_EXACT_MATCH_MAX = 600
AUX_COUNT_ROUNDING_RULE = (
    "round-half-up for nonnegative counts: floor(fraction * N_aux_train_full + 0.5)"
)
SCALING_PROTOCOL = (
    "E8_constrained_rowwise_shuffle_log10_percent_0p05_to_100_5points"
)
SCALING_ZERO_BASELINE_EXP_ID = "E2"
AUX_SHUFFLE_PERCENT_GRID = np.power(10.0, np.linspace(np.log10(0.05), 2.0, 5))
AUX_SHUFFLE_FRACTION_GRID = AUX_SHUFFLE_PERCENT_GRID / 100.0

_TRUE_LABEL_LOADER = engine.prepare_radonpy_loaders
_TRUE_CHECKPOINT_BUILDER = engine._build_ckpt_payload
_CHEM_CACHE: dict[str, tuple[str, str, Any]] = {}


def _round_half_up_nonnegative(value: float) -> int:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Expected a finite nonnegative value, got {value!r}")
    return int(math.floor(float(value) + 0.5))


def _chem_record(smiles: str):
    smiles = str(smiles).strip()
    if smiles in _CHEM_CACHE:
        return _CHEM_CACHE[smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse RadonPy SMILES: {smiles!r}")
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    connectivity_key = (
        inchikey.split("-")[0]
        if inchikey
        else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    )
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if _MORGAN_GENERATOR is not None:
        fingerprint = _MORGAN_GENERATOR.GetFingerprint(mol)
    else:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=AUX_SHUFFLE_FP_RADIUS,
            nBits=AUX_SHUFFLE_FP_BITS,
        )
    record = (connectivity_key, canonical_smiles, fingerprint)
    _CHEM_CACHE[smiles] = record
    return record


def _build_chem_arrays(smiles_values):
    keys, canonical_smiles, fingerprints = [], [], []
    for smiles in tqdm(smiles_values, desc="Auxiliary shuffle fingerprints", leave=False):
        key, canonical, fingerprint = _chem_record(smiles)
        keys.append(key)
        canonical_smiles.append(canonical)
        fingerprints.append(fingerprint)
    return (
        np.asarray(keys, dtype=object),
        np.asarray(canonical_smiles, dtype=object),
        fingerprints,
    )


def _pair_is_valid(recipient: int, donor: int, keys, fingerprints) -> bool:
    if recipient == donor or keys[recipient] == keys[donor]:
        return False
    similarity = float(
        DataStructs.TanimotoSimilarity(fingerprints[recipient], fingerprints[donor])
    )
    return similarity < AUX_SHUFFLE_MAX_TANIMOTO


def _random_bipartite_assignment(
    recipient_indices,
    donor_indices,
    keys,
    fingerprints,
    rng,
):
    """Find an exact randomized matching without similarity-based ranking."""
    recipient_indices = np.asarray(recipient_indices, dtype=np.int64)
    donor_indices = np.asarray(donor_indices, dtype=np.int64)
    if len(recipient_indices) != len(donor_indices):
        raise ValueError("Recipient and donor pools must have the same size.")

    n_rows = len(recipient_indices)
    donor_fingerprints = [fingerprints[int(index)] for index in donor_indices]
    candidates = []
    for recipient in recipient_indices:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[int(recipient)],
                donor_fingerprints,
            ),
            dtype=float,
        )
        valid_local = np.flatnonzero(
            (donor_indices != int(recipient))
            & (keys[donor_indices] != keys[int(recipient)])
            & (similarities < AUX_SHUFFLE_MAX_TANIMOTO)
        ).astype(np.int64)
        rng.shuffle(valid_local)
        candidates.append(valid_local.tolist())
    if any(not row_candidates for row_candidates in candidates):
        return None

    order = np.arange(n_rows, dtype=np.int64)
    rng.shuffle(order)
    order = np.asarray(
        sorted(order.tolist(), key=lambda index: len(candidates[int(index)])),
        dtype=np.int64,
    )
    donor_to_recipient = np.full(n_rows, -1, dtype=np.int64)

    def augment(recipient_local: int, seen: np.ndarray) -> bool:
        for donor_local in candidates[recipient_local]:
            if seen[donor_local]:
                continue
            seen[donor_local] = True
            previous_recipient = int(donor_to_recipient[donor_local])
            if previous_recipient < 0 or augment(previous_recipient, seen):
                donor_to_recipient[donor_local] = recipient_local
                return True
        return False

    for recipient_local in order:
        if not augment(int(recipient_local), np.zeros(n_rows, dtype=bool)):
            return None

    assignment = np.full(n_rows, -1, dtype=np.int64)
    for donor_local, recipient_local in enumerate(donor_to_recipient):
        assignment[int(recipient_local)] = int(donor_indices[donor_local])
    return assignment if np.all(assignment >= 0) else None


def _random_repair_permutation(keys, fingerprints, rng):
    """Construct a constrained derangement by randomized matching and repair."""
    n_rows = len(keys)
    if n_rows < 2:
        raise RuntimeError(
            "A constrained auxiliary-label derangement requires at least two rows."
        )
    if n_rows <= AUX_SHUFFLE_EXACT_MATCH_MAX:
        exact = _random_bipartite_assignment(
            np.arange(n_rows, dtype=np.int64),
            np.arange(n_rows, dtype=np.int64),
            keys,
            fingerprints,
            rng,
        )
        if exact is None:
            raise RuntimeError(
                f"No feasible derangement exists for {n_rows} rows under "
                f"Tanimoto < {AUX_SHUFFLE_MAX_TANIMOTO} and different connectivity."
            )
        return exact

    for _ in range(64):
        permutation = rng.permutation(n_rows).astype(np.int64)
        for _ in range(24):
            invalid = np.asarray(
                [
                    index
                    for index in range(n_rows)
                    if not _pair_is_valid(
                        index,
                        int(permutation[index]),
                        keys,
                        fingerprints,
                    )
                ],
                dtype=np.int64,
            )
            if len(invalid) == 0:
                return permutation
            rng.shuffle(invalid)
            progress = 0
            for first in invalid:
                first = int(first)
                if _pair_is_valid(
                    first,
                    int(permutation[first]),
                    keys,
                    fingerprints,
                ):
                    continue
                for _ in range(4096):
                    second = int(rng.integers(0, n_rows))
                    if second == first:
                        continue
                    first_donor = int(permutation[first])
                    second_donor = int(permutation[second])
                    if _pair_is_valid(
                        first,
                        second_donor,
                        keys,
                        fingerprints,
                    ) and _pair_is_valid(
                        second,
                        first_donor,
                        keys,
                        fingerprints,
                    ):
                        permutation[first], permutation[second] = (
                            second_donor,
                            first_donor,
                        )
                        progress += 1
                        break
            if progress == 0:
                break

        invalid = np.asarray(
            [
                index
                for index in range(n_rows)
                if not _pair_is_valid(
                    index,
                    int(permutation[index]),
                    keys,
                    fingerprints,
                )
            ],
            dtype=np.int64,
        )
        if len(invalid) == 0:
            return permutation

        if len(invalid) < AUX_SHUFFLE_EXACT_MATCH_MAX:
            max_buffer = AUX_SHUFFLE_EXACT_MATCH_MAX - len(invalid)
            valid_pool = np.setdiff1d(
                np.arange(n_rows, dtype=np.int64),
                invalid,
                assume_unique=False,
            )
            extra_count = min(
                len(valid_pool),
                max_buffer,
                max(32, 8 * len(invalid)),
            )
            extras = (
                rng.choice(valid_pool, size=extra_count, replace=False)
                if extra_count
                else np.empty(0, dtype=np.int64)
            )
            recipients = np.unique(np.concatenate([invalid, extras])).astype(np.int64)
            donors = permutation[recipients].copy()
            assignment = _random_bipartite_assignment(
                recipients,
                donors,
                keys,
                fingerprints,
                rng,
            )
            if assignment is not None:
                permutation[recipients] = assignment
                if all(
                    _pair_is_valid(
                        index,
                        int(permutation[index]),
                        keys,
                        fingerprints,
                    )
                    for index in range(n_rows)
                ):
                    return permutation

    raise RuntimeError(
        f"Failed to construct a constrained derangement for {n_rows} rows."
    )


def _nested_boundaries(n_full_train: int) -> list[int]:
    boundaries = []
    for fraction in AUX_SHUFFLE_FRACTION_GRID:
        count = max(
            1,
            min(
                int(n_full_train),
                _round_half_up_nonnegative(float(fraction) * n_full_train),
            ),
        )
        if not boundaries or count > boundaries[-1]:
            boundaries.append(count)
    if boundaries[-1] != int(n_full_train):
        boundaries.append(int(n_full_train))
    return boundaries


def _build_nested_mapping(smiles_values, seed: int, n_full_train: int):
    smiles_values = np.asarray(smiles_values, dtype=object)
    n_selected = len(smiles_values)
    keys, canonical_smiles, fingerprints = _build_chem_arrays(smiles_values)

    all_boundaries = _nested_boundaries(int(n_full_train))
    active_boundaries = [boundary for boundary in all_boundaries if boundary <= n_selected]
    if not active_boundaries or active_boundaries[-1] != n_selected:
        active_boundaries.append(n_selected)

    donors = np.full(n_selected, -1, dtype=np.int64)
    block_ids = np.full(n_selected, -1, dtype=np.int64)
    start = 0
    for block_id, end in enumerate(active_boundaries, start=1):
        block_size = int(end - start)
        if block_size < 2:
            raise RuntimeError(
                f"Nested shuffle block {block_id} has {block_size} row(s); "
                "a derangement is impossible."
            )
        block_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(block_id)])
        )
        local_permutation = _random_repair_permutation(
            keys[start:end],
            fingerprints[start:end],
            block_rng,
        )
        donors[start:end] = start + local_permutation
        block_ids[start:end] = block_id
        start = end

    row_positions = np.arange(n_selected, dtype=np.int64)
    if not np.array_equal(np.sort(donors), row_positions):
        raise RuntimeError("The constrained shuffle is not a one-to-one permutation.")

    similarities = np.asarray(
        [
            float(
                DataStructs.TanimotoSimilarity(
                    fingerprints[index],
                    fingerprints[int(donors[index])],
                )
            )
            for index in range(n_selected)
        ],
        dtype=float,
    )
    self_assignment = donors == row_positions
    same_connectivity = keys == keys[donors]
    threshold_violation = similarities >= AUX_SHUFFLE_MAX_TANIMOTO
    if self_assignment.any() or same_connectivity.any() or threshold_violation.any():
        raise RuntimeError(
            "Constrained shuffle audit failed: "
            f"self={int(self_assignment.sum())}, "
            f"same_connectivity={int(same_connectivity.sum())}, "
            f"tanimoto_violations={int(threshold_violation.sum())}."
        )
    return donors, {
        "keys": keys,
        "canonical_smiles": canonical_smiles,
        "similarities": similarities,
        "block_ids": block_ids,
        "active_boundaries": active_boundaries,
        "all_boundaries": all_boundaries,
    }


def _update_json(path: Path, updates: dict) -> None:
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_constrained_radonpy_loaders(
    fraction: float,
    seed: int,
    graph_cache,
    out_dir: Path,
):
    """Build the standard loaders, then permute selected training targets only."""
    result = _TRUE_LABEL_LOADER(fraction, seed, graph_cache, out_dir)
    if result.get("train_loader") is None:
        return result

    out_dir = Path(out_dir)
    train_dataset = result["train_loader"].dataset
    expected_count = max(
        1,
        min(
            int(result["n_train_full"]),
            _round_half_up_nonnegative(float(fraction) * result["n_train_full"]),
        ),
    )
    n_selected = len(train_dataset)
    if n_selected != expected_count:
        # The shared engine uses Python's ties-to-even round. Rebuild with an
        # exact integer ratio when it differs from the notebook's half-up rule.
        loader_fraction = expected_count / int(result["n_train_full"])
        result = _TRUE_LABEL_LOADER(
            loader_fraction,
            seed,
            graph_cache,
            out_dir,
        )
        train_dataset = result["train_loader"].dataset
        n_selected = len(train_dataset)
    if n_selected != expected_count:
        raise RuntimeError(
            f"Auxiliary row-count mismatch: loader={n_selected}, "
            f"protocol={expected_count}."
        )

    scaling_selection = {
        "EXP_ID": engine.EXP_ID,
        "SCALING_EXPERIMENT_ID": engine.CFG["scaling_experiment_id"],
        "scaling_protocol": SCALING_PROTOCOL,
        "scaling_point_number": int(engine.CFG["scaling_point_number"]),
        "scaling_log10_percent": float(engine.CFG["scaling_log10_percent"]),
        "requested_auxiliary_percent": float(engine.RADONPY_PERCENT),
        "requested_auxiliary_fraction": float(fraction),
        "n_auxiliary_rows_total_after_cleaning": int(
            result["n_train_full"] + result["n_valid"]
        ),
        "n_auxiliary_train_full_after_10pct_valid_holdout": int(
            result["n_train_full"]
        ),
        "requested_auxiliary_count_float": float(
            fraction * result["n_train_full"]
        ),
        "requested_auxiliary_count_rounded_half_up": int(expected_count),
        "n_auxiliary_train_used_after_bounds": int(n_selected),
        "n_auxiliary_valid_fixed": int(result["n_valid"]),
        "rounding_rule": AUX_COUNT_ROUNDING_RULE,
        "subset_rule": (
            "same seed-specific permutation; nested prefix across scaling points"
        ),
        "zero_percent_baseline": SCALING_ZERO_BASELINE_EXP_ID,
        "seed": int(seed),
    }
    _update_json(
        out_dir / "radonpy_scaling_selection.json",
        scaling_selection,
    )

    smiles_values = train_dataset.df["smiles"].astype(str).to_numpy()
    donor_positions, metadata = _build_nested_mapping(
        smiles_values,
        seed=int(seed),
        n_full_train=int(result["n_train_full"]),
    )

    true_targets = np.asarray(train_dataset.y, dtype=np.float32).copy()
    true_mask = np.asarray(train_dataset.mask, dtype=np.float32).copy()
    train_dataset.y = true_targets[donor_positions].copy()
    train_dataset.mask = true_mask[donor_positions].copy()

    source_ids = (
        train_dataset.df["source_row_id"].astype(str).to_numpy()
        if "source_row_id" in train_dataset.df
        else np.arange(n_selected).astype(str)
    )
    recipient_positions = np.arange(n_selected, dtype=np.int64)
    mapping = pd.DataFrame(
        {
            "seed": int(seed),
            "nested_block_id": metadata["block_ids"],
            "recipient_local_position": recipient_positions,
            "donor_local_position": donor_positions,
            "recipient_source_row_id": source_ids,
            "donor_source_row_id": source_ids[donor_positions],
            "recipient_smiles": smiles_values,
            "donor_smiles": smiles_values[donor_positions],
            "recipient_canonical_smiles": metadata["canonical_smiles"],
            "donor_canonical_smiles": metadata["canonical_smiles"][donor_positions],
            "recipient_connectivity_key": metadata["keys"],
            "donor_connectivity_key": metadata["keys"][donor_positions],
            "morgan_tanimoto": metadata["similarities"],
            "self_assignment": donor_positions == recipient_positions,
            "same_connectivity": metadata["keys"]
            == metadata["keys"][donor_positions],
            "tanimoto_threshold_violation": metadata["similarities"]
            >= AUX_SHUFFLE_MAX_TANIMOTO,
        }
    )
    mapping_path = out_dir / "radonpy_aux_label_shuffle_mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    audit = {
        "aux_label_control": AUX_LABEL_CONTROL,
        "seed": int(seed),
        "mapping_scope": "selected auxiliary training rows only",
        "mapping_fixed_for_entire_training_run": True,
        "whole_target_vector_and_missing_mask_moved_together": True,
        "auxiliary_validation_labels_shuffled": False,
        "candidate_selection": (
            "randomized eligible-candidate matching; no similarity ranking"
        ),
        "eligibility_rules": {
            "no_self_assignment": True,
            "different_connectivity_key": True,
            "morgan_tanimoto_strictly_below": AUX_SHUFFLE_MAX_TANIMOTO,
            "morgan_radius": AUX_SHUFFLE_FP_RADIUS,
            "morgan_n_bits": AUX_SHUFFLE_FP_BITS,
        },
        "nested_across_five_scaling_points": True,
        "nested_full_train_boundaries": [
            int(value) for value in metadata["all_boundaries"]
        ],
        "active_boundaries": [
            int(value) for value in metadata["active_boundaries"]
        ],
        "n_selected_training_rows": int(n_selected),
        "permutation_is_bijective": bool(
            np.array_equal(np.sort(donor_positions), recipient_positions)
        ),
        "n_self_assignments": int(mapping["self_assignment"].sum()),
        "n_same_connectivity_pairs": int(mapping["same_connectivity"].sum()),
        "n_tanimoto_threshold_violations": int(
            mapping["tanimoto_threshold_violation"].sum()
        ),
        "tanimoto_mean": float(metadata["similarities"].mean()),
        "tanimoto_median": float(np.median(metadata["similarities"])),
        "tanimoto_p95": float(np.percentile(metadata["similarities"], 95)),
        "tanimoto_max": float(metadata["similarities"].max()),
        "mapping_csv": str(mapping_path),
    }
    audit_path = out_dir / "radonpy_aux_label_shuffle_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config_updates = {
        "aux_label_control": AUX_LABEL_CONTROL,
        "aux_label_shuffle_seed": int(seed),
        "aux_label_shuffle_mapping": str(mapping_path),
        "aux_label_shuffle_audit": str(audit_path),
        "aux_label_shuffle_max_tanimoto": AUX_SHUFFLE_MAX_TANIMOTO,
        "aux_label_shuffle_different_connectivity_required": True,
        "aux_label_shuffle_whole_row_and_mask_together": True,
        "aux_label_shuffle_validation_labels_unchanged": True,
        "aux_label_shuffle_nested_across_scaling_points": True,
    }
    _update_json(
        out_dir / "radonpy_target_config.json",
        {**scaling_selection, **config_updates},
    )
    result.update(config_updates)
    result.update(
        {
            "requested_aux_count_float": scaling_selection[
                "requested_auxiliary_count_float"
            ],
            "requested_aux_count_rounded": expected_count,
        }
    )

    print(
        f"[AUX SHUFFLE] seed={seed} rows={n_selected} "
        f"blocks={metadata['active_boundaries']} "
        f"Tanimoto mean/median/max={audit['tanimoto_mean']:.3f}/"
        f"{audit['tanimoto_median']:.3f}/{audit['tanimoto_max']:.3f}"
    )
    return result


def build_constrained_checkpoint(model, epoch, best_score, seed, data, radon):
    payload = _TRUE_CHECKPOINT_BUILDER(
        model,
        epoch,
        best_score,
        seed,
        data,
        radon,
    )
    payload.update(
        {
            "aux_label_control": AUX_LABEL_CONTROL,
            "scaling_experiment_id": engine.CFG["scaling_experiment_id"],
            "scaling_protocol": SCALING_PROTOCOL,
            "scaling_point_number": int(engine.CFG["scaling_point_number"]),
            "scaling_log10_percent": float(
                engine.CFG["scaling_log10_percent"]
            ),
            "aux_count_rounding_rule": AUX_COUNT_ROUNDING_RULE,
            "zero_percent_baseline_exp_id": SCALING_ZERO_BASELINE_EXP_ID,
        }
    )
    return payload


def configure_point(point_key: str, fail_fast: bool) -> dict:
    point = POINTS[point_key]
    point_number = int(point_key.removeprefix("p"))
    scaling_experiment_id = f"E8_shuffle_scale_log5_p{point_number:02d}"
    engine.configure_experiment(point["base_experiment"])
    engine.EXP_KEY = scaling_experiment_id
    engine.EXP_ID = "E8_shuffle"
    engine.EXP_NAME = point["name"]
    engine.RADONPY_PERCENT = float(point["percent"])
    engine.RADONPY_FRACTION = engine.RADONPY_PERCENT / 100.0
    engine.FAIL_FAST = bool(fail_fast)
    engine.ROOT_OUT = (
        PROJECT_ROOT / "checkpoints" / "constrained_shuffle" / f"E8_{point_key}"
    )
    engine.FRACTION_OUT = engine.ROOT_OUT
    engine.OUT_DIR = engine.FRACTION_OUT / "_shared"
    engine.OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine.CFG.update(
        {
            "aux_label_control": AUX_LABEL_CONTROL,
            "aux_shuffle_max_tanimoto": AUX_SHUFFLE_MAX_TANIMOTO,
            "aux_shuffle_fp_radius": AUX_SHUFFLE_FP_RADIUS,
            "aux_shuffle_fp_bits": AUX_SHUFFLE_FP_BITS,
            "aux_shuffle_nested_across_scaling_points": True,
            "aux_validation_labels_shuffled": False,
            "scaling_protocol": SCALING_PROTOCOL,
            "scaling_experiment_id": scaling_experiment_id,
            "scaling_point_number": point_number,
            "scaling_log10_percent": float(
                np.log10(float(point["percent"]))
            ),
            "aux_count_rounding_rule": AUX_COUNT_ROUNDING_RULE,
            "zero_percent_baseline_exp_id": SCALING_ZERO_BASELINE_EXP_ID,
        }
    )
    engine.prepare_radonpy_loaders = prepare_constrained_radonpy_loaders
    engine._build_ckpt_payload = build_constrained_checkpoint
    point_protocol = {
        "EXP_ID": engine.EXP_ID,
        "SCALING_EXPERIMENT_ID": scaling_experiment_id,
        "scaling_protocol": SCALING_PROTOCOL,
        "scaling_point_number": point_number,
        "scaling_log10_percent": float(np.log10(float(point["percent"]))),
        "radonpy_percent": float(point["percent"]),
        "radonpy_fraction": float(point["percent"]) / 100.0,
        "rounding_rule": AUX_COUNT_ROUNDING_RULE,
        "zero_percent_baseline_exp_id": SCALING_ZERO_BASELINE_EXP_ID,
        "split_seeds": list(engine.SPLIT_SEEDS),
        "fraction_output": str(engine.FRACTION_OUT),
        "aux_label_control": AUX_LABEL_CONTROL,
        "aux_shuffle_max_tanimoto": AUX_SHUFFLE_MAX_TANIMOTO,
        "aux_shuffle_nested_across_scaling_points": True,
        "aux_validation_labels_shuffled": False,
    }
    (engine.FRACTION_OUT / "scaling_point_protocol.json").write_text(
        json.dumps(point_protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return point


def _summary_table(frame: pd.DataFrame, point_key: str) -> pd.DataFrame:
    row = {
        "experiment_key": f"E8_shuffle_scale_log5_{point_key}",
        "EXP_ID": "E8_shuffle",
        "EXP_NAME": POINTS[point_key]["name"],
        "n_completed_seeds": int(frame["seed"].nunique()) if len(frame) else 0,
        "radonpy_percent": float(POINTS[point_key]["percent"]),
        "aux_label_control": AUX_LABEL_CONTROL,
    }
    for column in ("best_valid_score", "best_epoch", "elapsed_min"):
        values = (
            pd.to_numeric(frame[column], errors="coerce").dropna()
            if column in frame
            else pd.Series(dtype=float)
        )
        row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
        row[f"{column}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else np.nan
        )
    return pd.DataFrame([row])


def run_point(point_key: str, fail_fast: bool = False) -> None:
    configure_point(point_key, fail_fast=fail_fast)
    graph_cache = engine.PrecomputedPyGGraphCache(
        engine.OUT_DIR / "pyg_graph_cache.pt"
    )
    completed, failed = [], []
    for split_seed in engine.SPLIT_SEEDS:
        try:
            row = engine.run_one_seed(split_seed, graph_cache)
            row["aux_label_control"] = AUX_LABEL_CONTROL
            row["scaling_point"] = point_key
            completed.append(row)
        except Exception as exc:
            failed.append(
                {
                    "EXP_ID": engine.EXP_ID,
                    "scaling_point": point_key,
                    "seed": split_seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            failure_dir = engine.FRACTION_OUT / f"seed_{split_seed}"
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "run_failed.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            if fail_fast:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed_frame = pd.DataFrame(completed)
    failed_frame = pd.DataFrame(failed)
    result_dir = (
        PROJECT_ROOT / "result" / "constrained_shuffle_training" / f"E8_{point_key}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    completed_frame.to_csv(result_dir / "per_model_seed.csv", index=False)
    _summary_table(completed_frame, point_key).to_csv(
        result_dir / "summary.csv",
        index=False,
    )
    if len(failed_frame):
        failed_frame.to_csv(engine.FRACTION_OUT / "failed_runs.csv", index=False)
    print(
        f"{point_key}: completed={len(completed_frame)}, "
        f"failed={len(failed_frame)}, checkpoints={engine.FRACTION_OUT}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train one or all E8 constrained shuffled-label auxiliary-scaling points."
        )
    )
    parser.add_argument(
        "point",
        choices=[*POINTS, "all"],
        help="Scaling point to train.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed seed.",
    )
    args = parser.parse_args()

    selected_points = list(POINTS) if args.point == "all" else [args.point]
    for point_key in selected_points:
        run_point(point_key, fail_fast=args.fail_fast)


if __name__ == "__main__":
    main()
