#!/usr/bin/env python
"""Run one bounded, validation-only T0 shape smoke for one R2/R4 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.dit_evaluation import _coordinates
from module.latent_rectified_flow import generate_state_detail_latent
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    StateDetailLatentAdapter,
    LatentStatistics,
    build_observation_condition,
)
from trainer.codec_trainer import PVBCodecModel
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("ratio2_state_detail", "ratio4_state_detail"), required=True)
    parser.add_argument(
        "--data-root",
        default="outputs/state_detail_codec_v2/t0_data/clip_store/valid",
        help="bounded T0 clip store only; paths containing test are rejected",
    )
    parser.add_argument("--output-root", default="outputs/dit_state_detail_probe_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--records", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--history-frames", type=int, default=4)
    return parser


def _require_safe_inputs(data_root: Path, output_root: Path, max_steps: int) -> None:
    if "test" in str(data_root).lower():
        raise RuntimeError("the DiT smoke refuses any path containing the test split")
    if str(output_root).startswith("outputs/state_detail_codec_v2"):
        raise RuntimeError("the DiT smoke refuses the active T1 output root")
    if not 2 <= int(max_steps) <= 100:
        raise ValueError("smoke max_steps must be between 2 and 100, including resume")


def _finite_gradients(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = name.split(".", 1)[0]
        groups.setdefault(group, []).append(parameter.grad.detach())
    result = {}
    for name, values in groups.items():
        result[name] = {
            "finite": all(bool(torch.isfinite(value).all()) for value in values),
            "nonzero": any(bool(torch.any(value != 0)) for value in values),
            "parameters": len(values),
        }
    return result


def _load_clip(data_root: Path, records: int):
    dataset = ClipMMapDataset(data_root)
    if len(dataset) < int(records):
        raise ValueError(f"T0 store has only {len(dataset)} records, requested {records}")
    selected = [dataset[index] for index in range(int(records))]
    batches = collate_clip_records(selected)
    return dataset, batches, selected


def run_smoke(
    *,
    candidate: str,
    data_root: str | Path,
    output_root: str | Path,
    device: str,
    max_steps: int,
    records: int,
    seed: int,
    history_frames: int,
) -> dict[str, Any]:
    ratio = 2 if candidate.startswith("ratio2") else 4
    data_root = Path(data_root)
    output_root = Path(output_root)
    _require_safe_inputs(data_root, output_root, max_steps)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    smoke_start = time.perf_counter()
    target_device = torch.device(device)
    dataset, clip_batch_cpu, selected = _load_clip(data_root, records)
    clip_batch = clip_batch_cpu.to(target_device)

    codec = PVBCodecModel(
        hidden_channels=128,
        spatial_layers=1,
        spatial_backbone="torchmd_et",
        temporal_layers=1,
        temporal_ratio=ratio,
        temporal_codec_mode=candidate,
        use_spatial_refiner=False,
        freeze_frame_encoder=False,
        coordinate_stem="centered_vector",
    ).to(target_device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.prepare_batch(clip_batch_cpu)
    codec_hash_before = module_state_hash(codec)
    frame_hash_before = module_state_hash(codec.frame_encoder)
    with torch.no_grad():
        oracle_latent = codec.encode(clip_batch)

    adapter = StateDetailLatentAdapter(
        codec_width=128, scalar_width=256, vector_width=128, ratio=ratio
    ).to(target_device)
    latent_batch = adapter.pack(
        oracle_latent,
        codec_hash=codec_hash_before,
        data_hash="t0_valid_shape_smoke",
    )
    condition = build_observation_condition(
        latent_batch,
        history_frames=history_frames,
        coordinates=clip_batch.x,
        frame_mask=clip_batch.frame_mask,
    )
    latent_batch = latent_batch.with_observation(
        condition.latent_observation_mask,
        sample_origin=condition.sample_origin,
    )
    stats = LatentStatistics.fit(
        [latent_batch],
        ratio=ratio,
        provenance={
            "source_split": "T0_valid",
            "data_root": str(data_root),
            "statistics_status": "synthetic_or_explicitly_labeled_smoke_only",
            "codec_hash": codec_hash_before,
        },
    )
    config = DiTTrainConfig(
        ratio=ratio,
        mode=candidate,
        max_steps=max_steps,
        seed=seed,
        output_root=str(output_root),
        data_hash="t0_valid_shape_smoke",
        codec_hash=codec_hash_before,
        stats_hash=stats.hash,
    )
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=256,
        vector_width=128,
        depth=4,
        heads=8,
        ffn_multiplier=4,
        dropout=0.0,
    ).to(target_device)
    trainer = DiTTrainer(
        model,
        adapter,
        config=config,
        statistics=stats,
        codec=codec,
        frame_encoder=codec.frame_encoder,
    )
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    initial_log = trainer.train_step(latent_batch)
    logs = [initial_log]
    while trainer.step < max_steps - 1 and time.perf_counter() - smoke_start < 15 * 60:
        logs.append(trainer.train_step(latent_batch))
    if trainer.step > 100:
        raise RuntimeError("smoke exceeded the authorized optimizer-step bound")
    checkpoint = output_root / candidate / "smoke_resume.pt"
    trainer.save_checkpoint(checkpoint)
    resume_step = trainer.step
    trainer.load_checkpoint(checkpoint, map_location=target_device)
    logs.append(trainer.train_step(latent_batch))
    resumed_step = trainer.step
    if resumed_step > 100:
        raise RuntimeError("checkpoint resume exceeded the authorized optimizer-step bound")
    model.eval()
    normalized = stats.normalize(latent_batch)
    trunk_start = time.perf_counter()
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    with torch.no_grad():
        generated8, meta8 = generate_state_detail_latent(
            model, adapter, latent_batch, stats, steps=8, seed=seed
        )
        generated16, meta16 = generate_state_detail_latent(
            model, adapter, latent_batch, stats, steps=16, seed=seed
        )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    trunk_seconds = max(time.perf_counter() - trunk_start, 1e-9)
    with torch.no_grad():
        decoded8 = codec.decode(generated8)
        decoded16 = codec.decode(generated16)
    if not bool(torch.isfinite(_coordinates(decoded8)).all()) or not bool(torch.isfinite(_coordinates(decoded16)).all()):
        raise FloatingPointError("8-step or 16-step generated decode is non-finite")
    codec_hash_after = module_state_hash(codec)
    frame_hash_after = module_state_hash(codec.frame_encoder)
    elapsed = time.perf_counter() - smoke_start
    if elapsed > 15 * 60:
        raise RuntimeError("smoke exceeded the authorized 15-minute wall-time bound")
    result = {
        "schema_version": "pvb.dit.state_detail.smoke.v1",
        "candidate": candidate,
        "ratio": ratio,
        "status": "PASS",
        "execution_evidence_only": True,
        "scientific_claim": False,
        "data": {
            "root": str(data_root),
            "split": "T0_valid",
            "sample_ids": [str(item.get("sample_id", "")) for item in selected],
            "topology_ids": [str(value) for value in clip_batch_cpu.topology_id],
            "provenance": "existing T0 clip store; no T1 test access",
        },
        "contracts": {
            "codec_hash": codec_hash_before,
            "statistics_hash": stats.hash,
            "adapter_hash": latent_batch.hash,
            "model_hash": model.model_hash,
            "codec_unchanged": codec_hash_before == codec_hash_after,
            "frame_encoder_unchanged": frame_hash_before == frame_hash_after,
        },
        "losses": {
            "initial": initial_log,
            "final": logs[-1],
        },
        "gradients": _finite_gradients(model),
        "resume": {"checkpoint": str(checkpoint), "before": resume_step, "after": resumed_step},
        "generation": {
            "steps8": meta8,
            "steps16": meta16,
            "finite_decode": True,
            "raw_detail_h": None,
            "raw_detail_v": None,
        },
        "runtime": {
            "wall_time_s": elapsed,
            "trunk_tokens_per_s": (latent_batch.tokens * latent_batch.num_atoms * 16) / trunk_seconds,
            "end_to_end_samples_per_s": latent_batch.batch_size / max(elapsed, 1e-9),
            "parameters": model.parameter_count,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)) if target_device.type == "cuda" else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)) if target_device.type == "cuda" else 0,
        },
        "warnings": [
            "Randomly initialized T0 shape smoke; not a trained codec or scientific DiT pilot.",
            "The smoke does not rank R2 versus R4.",
        ],
    }
    output_path = output_root / candidate / "smoke_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset.close()
    return result


def main() -> None:
    args = _parser().parse_args()
    result = run_smoke(
        candidate=args.candidate,
        data_root=args.data_root,
        output_root=args.output_root,
        device=args.device,
        max_steps=args.max_steps,
        records=args.records,
        seed=args.seed,
        history_frames=args.history_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
