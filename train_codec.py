#!/usr/bin/env python3
"""Launch the isolated multi-frame codec trainer.

The script deliberately consumes only the versioned clip stores.  Legacy
``train.py`` and its pair-record model path are not routed through this entry
point.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset

from data.clip_batching import make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from trainer.codec_trainer import CodecTrainConfig, CodecTrainer, PVBCodecModel


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("codec YAML must contain a mapping at its root")
    return config


def _dataset(roots: Sequence[str]) -> Any:
    stores = [ClipMMapDataset(root) for root in roots]
    if not stores:
        raise ValueError("codec data roots are empty; set data.train_roots in the YAML")
    return stores[0] if len(stores) == 1 else ConcatDataset(stores)


def _loader(dataset: Any, data_config: dict[str, Any], *, training: bool) -> Any:
    return make_clip_dataloader(
        dataset,
        max_tokens=int(data_config["max_tokens"]),
        collate_fn=collate_clip_records,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=bool(data_config.get("pin_memory", False)),
        seed=int(data_config.get("seed", 0)) + (0 if training else 1),
        shuffle=bool(training),
        replacement=bool(training),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/codec.yaml"))
    parser.add_argument("--device", default=None, help="override training.device")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--fit-normalization", action="store_true")
    parser.add_argument("--normalization-batches", type=int, default=None, help="bound normalization fitting to a deterministic number of train batches")
    parser.add_argument("--temporal-ratio", type=int, default=None)
    parser.add_argument("--temporal-layers", type=int, default=None)
    parser.add_argument("--train-root", action="append", default=None)
    parser.add_argument("--valid-root", action="append", default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None, help="append per-step training metrics as JSONL")
    parser.add_argument("--log-every", type=int, default=1, help="write one training record every N optimizer steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate one batch without optimizing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw = _load_config(args.config)
    if args.device is not None:
        raw.setdefault("training", {})["device"] = args.device
    if args.temporal_ratio is not None:
        raw.setdefault("model", {})["temporal_ratio"] = args.temporal_ratio
    if args.temporal_layers is not None:
        raw.setdefault("model", {})["temporal_layers"] = args.temporal_layers
    if args.train_root is not None:
        raw.setdefault("data", {})["train_roots"] = args.train_root
    if args.valid_root is not None:
        raw.setdefault("data", {})["valid_roots"] = args.valid_root
    if args.save_dir is not None:
        raw.setdefault("training", {})["save_dir"] = str(args.save_dir)
    if args.seed is not None:
        raw.setdefault("data", {})["seed"] = args.seed
    data_seed = int(raw.get("data", {}).get("seed", 0))
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    train_config = CodecTrainConfig.from_mapping(raw)
    data_config = raw.get("data", {})
    if not isinstance(data_config, dict):
        raise ValueError("data config must be a mapping")
    train_dataset = _dataset(data_config.get("train_roots", []))
    train_loader = _loader(train_dataset, data_config, training=True)
    valid_loader = None
    valid_roots = data_config.get("valid_roots", [])
    if valid_roots:
        valid_loader = _loader(_dataset(valid_roots), data_config, training=False)

    model_config = raw.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a mapping")
    model = PVBCodecModel(
        **{key: value for key, value in model_config.items() if key in {
            "hidden_channels", "spatial_layers", "temporal_layers", "temporal_ratio",
            "num_rbf", "num_heads", "cutoff_lower", "cutoff_upper",
            "max_num_neighbors", "neighbor_backend", "use_spatial_refiner", "time_scale_ps",
        }}
    )
    trainer = CodecTrainer(model, train_loader, valid_loader, train_config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    if args.fit_normalization:
        if args.normalization_batches is not None:
            if args.normalization_batches < 1:
                raise ValueError("--normalization-batches must be positive")
            normalization_source = islice(train_loader, args.normalization_batches)
        else:
            normalization_source = train_loader
        stats = trainer.fit_normalization(normalization_source)
        print(json.dumps({key: value.as_dict() for key, value in stats.items()}, indent=2))
    if args.dry_run:
        batch = next(iter(train_loader))
        print(json.dumps(trainer.evaluate_batch(batch), sort_keys=True))
        return 0
    save_dir = Path(raw.get("training", {}).get("save_dir", "ckpt/codec"))
    log_path = args.log_path if args.log_path is not None else save_dir / "train_metrics.jsonl"
    metrics = trainer.run(
        max_steps=args.max_steps,
        log_path=log_path,
        log_every=args.log_every,
    )
    checkpoint = save_dir / f"codec_step_{trainer.step:08d}.pt"
    trainer.save_checkpoint(checkpoint)
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "log": str(log_path),
        "step": trainer.step,
        "metrics": metrics,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
