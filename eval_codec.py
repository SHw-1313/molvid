#!/usr/bin/env python3
"""Run the three T08 codec controls and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from evaluation.codec_evaluation import evaluate_controls, model_control, write_report
from train_codec import _fractional_subset
from trainer.codec_losses import compute_codec_losses
from trainer.codec_trainer import CodecTrainConfig, PVBCodecModel, _to_device


def _dataset(roots: Sequence[str]) -> Any:
    stores = [ClipMMapDataset(root) for root in roots]
    if not stores:
        raise ValueError("evaluation requires data.valid_roots in the codec YAML")
    return stores[0] if len(stores) == 1 else ConcatDataset(stores)


def _predictor(model: PVBCodecModel, device: torch.device):
    def predict(batch):
        moved = _to_device(batch, device)
        with torch.no_grad():
            output = model(moved)
        return output.x_hat.detach().cpu()
    return predict


def _loss_evaluator(
    device: torch.device,
    config: CodecTrainConfig,
    normalization: Mapping[str, Any],
    step: int,
):
    def evaluate(prediction: torch.Tensor, batch: Any) -> dict[str, float]:
        bucket_ids = tuple(str(item) for item in batch.time_bucket_id)
        if not bucket_ids or len(set(bucket_ids)) != 1:
            raise ValueError("validation loss requires homogeneous time buckets")
        moved = _to_device(batch, device)
        coordinates = prediction.to(device)
        losses = compute_codec_losses(
            coordinates,
            moved,
            weights=config.weights_at(step),
            normalization=normalization,
        )
        losses["total"] = losses["total"] * config.bucket(bucket_ids[0]).weight
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        }
    return evaluate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/codec.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--json", type=Path, default=Path("eval/codec_eval.json"))
    parser.add_argument("--markdown", type=Path, default=Path("eval/codec_eval.md"))
    parser.add_argument("--ratio1-no-temporal-checkpoint", type=Path, default=None)
    parser.add_argument("--ratio1-temporal-checkpoint", type=Path, default=None)
    parser.add_argument("--ratio4-temporal-checkpoint", type=Path, default=None)
    parser.add_argument("--valid-root", action="append", default=None)
    parser.add_argument("--subset-fraction", type=float, default=None, help="deterministically evaluate on this fraction of the validation dataset")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("codec YAML root must be a mapping")
    runtime = dict(raw)
    if args.device is not None:
        runtime.setdefault("training", {})["device"] = args.device
    if args.valid_root is not None:
        raw.setdefault("data", {})["valid_roots"] = args.valid_root
    if args.seed is not None:
        raw.setdefault("data", {})["seed"] = args.seed
    seed = int(raw.get("data", {}).get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_config = CodecTrainConfig.from_mapping(runtime)
    selected = args.device or train_config.device
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected)
    data_config = raw.get("data", {})
    dataset = _dataset(data_config.get("valid_roots", []))
    dataset, subset_info = _fractional_subset(
        dataset, args.subset_fraction, seed + 1
    )
    loader = make_clip_dataloader(
        dataset,
        max_tokens=int(data_config.get("max_tokens", 4096)),
        collate_fn=collate_clip_records,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        seed=int(data_config.get("seed", 0)),
        shuffle=False,
        replacement=False,
    )
    model_config = raw.get("model", {})
    common = {key: value for key, value in model_config.items() if key in {
        "hidden_channels", "spatial_layers", "num_rbf", "num_heads", "cutoff_lower",
        "cutoff_upper", "max_num_neighbors", "neighbor_backend", "use_spatial_refiner",
        "time_scale_ps",
    }}
    controls = []
    checkpoint_paths = {
        "ratio1_no_temporal": args.ratio1_no_temporal_checkpoint,
        "ratio1_temporal": args.ratio1_temporal_checkpoint,
        "ratio4_temporal": args.ratio4_temporal_checkpoint,
    }
    for name, ratio, layers, temporal in (
        ("ratio1_no_temporal", 1, 0, False),
        ("ratio1_temporal", 1, 1, True),
        ("ratio4_temporal", 4, 1, True),
    ):
        model = PVBCodecModel(
            **common,
            temporal_layers=layers,
            temporal_ratio=ratio,
        ).to(device).eval()
        checkpoint = checkpoint_paths[name]
        loss_evaluator = None
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if not isinstance(payload, dict) or payload.get("schema_version") != "pvb.codec.checkpoint.v1":
                raise ValueError(f"{name} checkpoint is not a pvb.codec.checkpoint.v1 file")
            model.load_state_dict(payload["model_state"])
            checkpoint_config = CodecTrainConfig.from_mapping(payload["config"])
            loss_evaluator = _loss_evaluator(
                device,
                checkpoint_config,
                payload.get("normalization_stats", {}),
                int(payload.get("step", checkpoint_config.max_steps)),
            )
        controls.append(
            model_control(
                name,
                _predictor(model, device),
                ratio=ratio,
                temporal=temporal,
                loss_evaluator=loss_evaluator,
            )
        )
    report = evaluate_controls(controls, loader, max_batches=args.max_batches, device=device)
    report["evaluation_data"] = {
        "subset": subset_info,
        "max_batches": args.max_batches,
    }
    write_report(report, args.json, args.markdown)
    print(f"wrote {args.json} and {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
