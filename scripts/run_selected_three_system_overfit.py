#!/usr/bin/env python3
"""Real CUDA overfitting test on the three selected ATLAS systems."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from module.bond_sources import build_canonical_reference_index
from trainer.codec_losses import CodecLossWeights
from trainer.codec_trainer import (
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "outputs/atlas_selected_trajectories/clip_store/train"
BASE_OUTPUT = ROOT / "outputs/atlas_selected_trajectories/overfit_three_systems"
DEVICE = torch.device("cuda:0")
SEED = 20260826
STEPS = 500
LOG_EVERY = 10
MAX_TOKENS = 80000
SAMPLE_IDS = (
    "atlas_5e3e_A_R1_w000000",
    "atlas_1v7r_A_R1_w000000",
    "atlas_2wlt_A_R1_w000000",
)
CONTROLS: dict[str, dict[str, int]] = {
    "ratio1_no_temporal": {"temporal_layers": 0, "temporal_ratio": 1},
    "ratio1_temporal": {"temporal_layers": 1, "temporal_ratio": 1},
    "ratio4_temporal": {"temporal_layers": 1, "temporal_ratio": 4},
}


def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "OVERFIT_HARD_FAIL: torch.cuda.is_available() is False; "
            "real-data overfitting requires CUDA and has no CPU fallback"
        )
    try:
        import torch_cluster
        from torch_cluster import radius_graph
    except Exception as exc:
        raise RuntimeError(
            "OVERFIT_HARD_FAIL: CUDA torch_cluster.radius_graph is unavailable"
        ) from exc
    if not callable(radius_graph):
        raise RuntimeError(
            "OVERFIT_HARD_FAIL: torch_cluster.radius_graph is not callable"
        )
    props = torch.cuda.get_device_properties(DEVICE)
    return {
        "device": str(DEVICE),
        "gpu": props.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_cluster": getattr(torch_cluster, "__version__", "unknown"),
        "total_memory_bytes": int(props.total_memory),
    }


def select_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not TRAIN_ROOT.is_dir():
        raise FileNotFoundError(f"selected train clip store is missing: {TRAIN_ROOT}")
    dataset = ClipMMapDataset(TRAIN_ROOT)
    by_sample_id = {str(item[0]): index for index, item in enumerate(dataset._index)}
    missing = [sample_id for sample_id in SAMPLE_IDS if sample_id not in by_sample_id]
    if missing:
        raise RuntimeError(f"selected overfit samples are missing: {missing}")
    records = [dataset[by_sample_id[sample_id]] for sample_id in SAMPLE_IDS]
    dataset.close()
    actual = tuple(str(record["sample_id"]) for record in records)
    if actual != SAMPLE_IDS:
        raise RuntimeError(f"selected sample order changed: {actual!r}")
    batch = collate_clip_records(records)
    tokens = int(batch.frames * sum(
        int(batch.atom_ptr[index + 1] - batch.atom_ptr[index])
        for index in range(batch.batch_size)
    ))
    if tokens > MAX_TOKENS:
        raise RuntimeError(
            f"fixed overfit batch is oversized: {tokens} > max_tokens={MAX_TOKENS}"
        )
    return records, {
        "sample_ids": list(actual),
        "systems": [str(record["sample_id"]).rsplit("_R", 1)[0] for record in records],
        "frames": int(batch.frames),
        "atoms_per_sample": [
            int(batch.atom_ptr[index + 1] - batch.atom_ptr[index])
            for index in range(batch.batch_size)
        ],
        "packed_atoms": int(batch.atom_count),
        "effective_tokens": tokens,
        "time_bucket_ids": list(batch.time_bucket_id),
        "source_split": "train",
        "validation_samples_used": False,
    }



def make_distance_references(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    references = build_canonical_reference_index(records, source_split="train")
    if len(references) != len(records):
        raise RuntimeError("distance_only canonical reference count mismatch")
    return references


def make_model(
    control: Mapping[str, int],
    *,
    bond_mode: str = "topology",
) -> PVBCodecModel:
    return PVBCodecModel(
        hidden_channels=128,
        spatial_layers=2,
        temporal_layers=int(control["temporal_layers"]),
        temporal_ratio=int(control["temporal_ratio"]),
        num_rbf=50,
        num_heads=8,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_neighbors=32,
        neighbor_backend="cuda_radius",
        bond_construction={"mode": bond_mode},
        spatial_execution={"mode": "full"},
    )


def make_config(steps: int) -> CodecTrainConfig:
    return CodecTrainConfig(
        lr=1.0e-4,
        weight_decay=1.0e-6,
        max_steps=steps,
        grad_clip=1.0,
        warmup_steps=20,
        device="cuda",
        precision="fp32",
        bucket_specs=(
            TimeBucketSpec("dt_100ps", 100.0, 0.5, 1.0),
        ),
        loss_schedule=(
            (0, CodecLossWeights(coordinate=1.0)),
            (
                100,
                CodecLossWeights(
                    coordinate=1.0,
                    local=0.1,
                    bond=0.1,
                    velocity=0.1,
                    acceleration=0.05,
                ),
            ),
        ),
        normalization_min_count=32,
        normalization_epsilon=1.0e-6,
    )


def make_loader(records: list[dict[str, Any]]) -> Any:
    loader = make_clip_dataloader(
        records,
        max_tokens=MAX_TOKENS,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
        oversize_policy="error",
        seed=SEED,
        shuffle=False,
        replacement=True,
    )
    if len(loader) != 1:
        raise RuntimeError(
            f"the three selected systems must form one repeated overfit batch, got {len(loader)}"
        )
    return loader


def evaluate_records(trainer: CodecTrainer, records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = trainer.evaluate_batch(collate_clip_records(records))
    per_system = {}
    for record in records:
        metrics = trainer.evaluate_batch(collate_clip_records([record]))
        per_system[str(record["sample_id"])] = metrics
    return {"aggregate": aggregate, "per_system": per_system}


def parse_log(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_steps = list(range(LOG_EVERY, STEPS + 1, LOG_EVERY))
    actual_steps = [int(row["step"]) for row in rows]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"overfit log is incomplete: expected {expected_steps[-3:]}, got {actual_steps[-3:]}"
        )
    totals = [float(row["metrics"]["total"]) for row in rows]
    return {
        "records": len(rows),
        "first_logged_step": actual_steps[0],
        "last_logged_step": actual_steps[-1],
        "first_logged_total": totals[0],
        "last_logged_total": totals[-1],
        "minimum_logged_total": min(totals),
        "last_logged_epoch": int(rows[-1]["epoch"]),
    }


def run_control(
    name: str,
    control: Mapping[str, int],
    records: list[dict[str, Any]],
    run_dir: Path,
    *,
    bond_mode: str,
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    control_dir = run_dir / name
    control_dir.mkdir(parents=True, exist_ok=False)
    loader = make_loader(records)
    model = make_model(control, bond_mode=bond_mode)
    trainer = CodecTrainer(
        model,
        loader,
        config=make_config(STEPS),
        device=DEVICE,
        non_blocking_transfer=False,
    )
    if bond_mode == "distance_only":
        trainer.model.prepare_distance_bonds(references, device=trainer.device)
        torch.cuda.synchronize()
    normalization_start = time.perf_counter()
    trainer.fit_normalization(loader)
    torch.cuda.synchronize()
    normalization_elapsed = time.perf_counter() - normalization_start

    initial_start = time.perf_counter()
    initial = evaluate_records(trainer, records)
    torch.cuda.synchronize()
    initial_elapsed = time.perf_counter() - initial_start

    log_path = control_dir / "train_metrics.jsonl"
    torch.cuda.reset_peak_memory_stats(DEVICE)
    training_start = time.perf_counter()
    trainer.run(max_steps=STEPS, log_path=log_path, log_every=LOG_EVERY)
    torch.cuda.synchronize()
    training_elapsed = time.perf_counter() - training_start

    final_start = time.perf_counter()
    final = evaluate_records(trainer, records)
    torch.cuda.synchronize()
    final_elapsed = time.perf_counter() - final_start

    checkpoint = control_dir / f"codec_step_{trainer.step:08d}.pt"
    trainer.save_checkpoint(checkpoint)
    torch.cuda.synchronize()
    log_summary = parse_log(log_path)
    initial_total = float(initial["aggregate"]["total"])
    final_total = float(final["aggregate"]["total"])
    result = {
        "control": name,
        "temporal_layers": int(control["temporal_layers"]),
        "temporal_ratio": int(control["temporal_ratio"]),
        "steps": int(trainer.step),
        "batch_count": len(loader),
        "batch_sample_ids": list(SAMPLE_IDS),
        "initial_metrics": initial,
        "final_metrics": final,
        "loss_log": log_summary,
        "aggregate_initial_total": initial_total,
        "aggregate_final_total": final_total,
        "aggregate_relative_reduction": (initial_total - final_total) / max(abs(initial_total), 1.0e-8),
        "fit_status": "loss_decreased" if final_total < initial_total else "loss_not_decreased",
        "normalization_elapsed_s": normalization_elapsed,
        "initial_evaluation_elapsed_s": initial_elapsed,
        "training_elapsed_s": training_elapsed,
        "final_evaluation_elapsed_s": final_elapsed,
        "total_elapsed_s": normalization_elapsed + initial_elapsed + training_elapsed + final_elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "checkpoint": str(checkpoint),
        "log": str(log_path),
        "precision": "fp32",
        "bond_construction_mode": bond_mode,
        "distance_reference_contract": trainer.model.distance_reference_contract(),
    }
    (control_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del trainer, model, loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bond-mode",
        choices=("topology", "distance_only"),
        default="topology",
    )
    args = parser.parse_args(argv)
    bond_mode = str(args.bond_mode)
    runtime = require_cuda()
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    records, data_contract = select_records()
    references = (
        make_distance_references(records) if bond_mode == "distance_only" else {}
    )
    output_root = (
        BASE_OUTPUT
        if bond_mode == "topology"
        else ROOT / "outputs/atlas_selected_trajectories/overfit_three_systems_distance_only"
    )
    run_dir = output_root / (
        "run_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing overfit run: {run_dir}")
    run_dir.mkdir(parents=True)
    protocol = {
        "runtime": runtime,
        "seed": SEED,
        "steps": STEPS,
        "log_every": LOG_EVERY,
        "max_tokens": MAX_TOKENS,
        "data_root": str(TRAIN_ROOT),
        "data_contract": data_contract,
        "selection_policy": "one fixed R1/w000000 train clip from each selected system",
        "model_policy": f"three full-width modified-code {bond_mode} controls from fresh initialization",
        "bond_construction_mode": bond_mode,
        "distance_reference_policy": (
            "first frame of each selected train clip; supplied bond/atom topology "
            "is not used for inference"
            if bond_mode == "distance_only"
            else None
        ),
        "distance_reference_sample_ids": (
            sorted(item["sample_id"] for item in references.values())
            if references
            else []
        ),
        "distance_reference_manifest": (
            [
                {
                    key: value
                    for key, value in item.items()
                    if key != "coordinates"
                }
                for _topology_id, item in sorted(references.items())
            ]
            if references
            else []
        ),
        "optimizer": "AdamW",
        "precision": "fp32",
        "cuda_only": True,
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "status": "running",
        "protocol": protocol,
        "controls": {},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, control in CONTROLS.items():
        result = run_control(
            name,
            control,
            records,
            run_dir,
            bond_mode=bond_mode,
            references=references,
        )
        summary["controls"][name] = result
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    torch.cuda.synchronize()
    summary["status"] = "completed"
    summary["elapsed_s"] = time.perf_counter() - started
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
