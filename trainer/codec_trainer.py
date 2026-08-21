"""Isolated trainer and end-to-end model wrapper for the PVB clip codec."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from data.clip_dataset import ClipBatch, TASK_NAMES
from module.coordinate_decoder import CodecLatent, JointMultiFrameDecoder, LatentConditionedSpatialRefiner
from module.multiframe_codec import PVBFrameEncoder
from module.temporal_codec import CausalTemporalEncoder

from .codec_losses import (
    BucketNormalization,
    CodecLossWeights,
    compute_codec_losses,
    fit_time_bucket_normalization,
)


CODEC_CONFIG_SCHEMA = "pvb.codec.config.v1"
CODEC_CHECKPOINT_SCHEMA = "pvb.codec.checkpoint.v1"


@dataclass(frozen=True)
class TimeBucketSpec:
    bucket_id: str
    center_ps: float
    tolerance_ps: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.bucket_id):
            raise ValueError("time bucket id must be non-empty")
        if not math.isfinite(float(self.center_ps)) or float(self.center_ps) < 0:
            raise ValueError("time bucket center must be finite and non-negative")
        if not math.isfinite(float(self.tolerance_ps)) or float(self.tolerance_ps) < 0:
            raise ValueError("time bucket tolerance must be finite and non-negative")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0:
            raise ValueError("time bucket weight must be finite and non-negative")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "id": str(self.bucket_id),
            "center_ps": float(self.center_ps),
            "tolerance_ps": float(self.tolerance_ps),
            "weight": float(self.weight),
        }


def validate_codec_config(config: Mapping[str, Any]) -> tuple[TimeBucketSpec, ...]:
    """Validate the explicit physical-time and normalization configuration."""

    if str(config.get("schema_version", "")) != CODEC_CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported codec config schema; expected {CODEC_CONFIG_SCHEMA!r}"
        )
    time_config = config.get("time")
    if not isinstance(time_config, Mapping):
        raise ValueError("codec config requires a time mapping")
    if str(time_config.get("unit", "")) != "ps":
        raise ValueError("codec time.unit must be the canonical 'ps'")
    continuous_scale = float(time_config.get("continuous_scale_ps", 0.0))
    if not math.isfinite(continuous_scale) or continuous_scale <= 0:
        raise ValueError("time.continuous_scale_ps must be finite and positive")
    buckets = time_config.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("codec config requires at least one time bucket")
    specs = tuple(
        TimeBucketSpec(
            bucket_id=str(item.get("id", "")),
            center_ps=float(item.get("center_ps", 0.0)),
            tolerance_ps=float(item.get("tolerance_ps", 0.0)),
            weight=float(item.get("weight", 1.0)),
        )
        for item in buckets
        if isinstance(item, Mapping)
    )
    if len(specs) != len(buckets) or len({item.bucket_id for item in specs}) != len(specs):
        raise ValueError("time bucket entries must be mappings with unique ids")
    normalization = time_config.get("normalization", {})
    if not isinstance(normalization, Mapping):
        raise ValueError("time.normalization must be a mapping")
    min_count = int(normalization.get("min_count", 1))
    epsilon = float(normalization.get("epsilon", 0.0))
    if min_count < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization min_count must be positive and epsilon must be positive")
    return specs


@dataclass
class CodecTrainConfig:
    """Validated runtime subset of ``config/codec.yaml``."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1
    grad_clip: Optional[float] = 1.0
    warmup_steps: int = 0
    device: str = "cpu"
    loss_schedule: tuple[tuple[int, CodecLossWeights], ...] = ((0, CodecLossWeights()),)
    bucket_specs: tuple[TimeBucketSpec, ...] = (
        TimeBucketSpec("static", 0.0, 0.0, 1.0),
        TimeBucketSpec("dt_100ps", 100.0, 1.0, 1.0),
    )
    continuous_scale_ps: float = 100.0
    normalization_min_count: int = 32
    normalization_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        self.loss_schedule = tuple(
            (int(start), weights if isinstance(weights, CodecLossWeights) else CodecLossWeights.from_mapping(weights))
            for start, weights in self.loss_schedule
        )
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0:
            raise ValueError("weight decay must be finite and non-negative")
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if self.grad_clip is not None and (not math.isfinite(float(self.grad_clip)) or float(self.grad_clip) <= 0):
            raise ValueError("grad_clip must be positive when supplied")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not self.loss_schedule or self.loss_schedule[0][0] != 0:
            raise ValueError("loss schedule must start at step zero")
        if any(start < 0 for start, _ in self.loss_schedule):
            raise ValueError("loss schedule steps must be non-negative")
        if any(self.loss_schedule[i][0] >= self.loss_schedule[i + 1][0] for i in range(len(self.loss_schedule) - 1)):
            raise ValueError("loss schedule steps must be strictly increasing")
        if not math.isfinite(float(self.continuous_scale_ps)) or float(self.continuous_scale_ps) <= 0:
            raise ValueError("continuous_scale_ps must be finite and positive")
        if int(self.normalization_min_count) < 1 or float(self.normalization_epsilon) <= 0:
            raise ValueError("normalization guards must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CodecTrainConfig":
        specs = validate_codec_config(config)
        time_config = config["time"]
        training = config.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("training config must be a mapping")
        base_weights = CodecLossWeights.from_mapping(training.get("loss_weights"))
        stages = training.get("loss_schedule")
        if stages is None:
            schedule = ((0, base_weights),)
        else:
            if not isinstance(stages, list) or not stages:
                raise ValueError("training.loss_schedule must be a non-empty list")
            parsed = []
            for stage in stages:
                if not isinstance(stage, Mapping):
                    raise ValueError("each loss schedule stage must be a mapping")
                stage_weights = stage.get("weights", stage)
                parsed.append((int(stage.get("start_step", 0)), CodecLossWeights.from_mapping(stage_weights)))
            schedule = tuple(sorted(parsed, key=lambda item: item[0]))
        normalization = time_config.get("normalization", {})
        grad_clip = training.get("grad_clip", 1.0)
        return cls(
            lr=float(training.get("lr", 1e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_steps=int(training.get("max_steps", 1)),
            grad_clip=None if grad_clip is None else float(grad_clip),
            warmup_steps=int(training.get("warmup_steps", 0)),
            device=str(training.get("device", "cpu")),
            loss_schedule=schedule,
            bucket_specs=specs,
            continuous_scale_ps=float(time_config["continuous_scale_ps"]),
            normalization_min_count=int(normalization.get("min_count", 32)),
            normalization_epsilon=float(normalization.get("epsilon", 1e-6)),
        )

    def weights_at(self, step: int) -> CodecLossWeights:
        selected = self.loss_schedule[0][1]
        for start, weights in self.loss_schedule:
            if int(step) >= start:
                selected = weights
            else:
                break
        return selected

    def bucket(self, bucket_id: str) -> TimeBucketSpec:
        for spec in self.bucket_specs:
            if spec.bucket_id == str(bucket_id):
                return spec
        raise ValueError(f"unconfigured time bucket: {bucket_id!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CODEC_CONFIG_SCHEMA,
            "time": {
                "unit": "ps",
                "continuous_scale_ps": float(self.continuous_scale_ps),
                "buckets": [spec.as_dict() for spec in self.bucket_specs],
                "normalization": {
                    "min_count": int(self.normalization_min_count),
                    "epsilon": float(self.normalization_epsilon),
                },
            },
            "training": {
                "lr": float(self.lr),
                "weight_decay": float(self.weight_decay),
                "max_steps": int(self.max_steps),
                "grad_clip": self.grad_clip,
                "warmup_steps": int(self.warmup_steps),
                "device": self.device,
                "loss_schedule": [
                    {"start_step": start, "weights": weights.as_dict()}
                    for start, weights in self.loss_schedule
                ],
            },
        }


class PVBCodecModel(nn.Module):
    """Compose the existing PVB spatial backbone with T05/T06 codec modules."""

    def __init__(
        self,
        hidden_channels: int = 128,
        spatial_layers: int = 2,
        temporal_layers: int = 1,
        temporal_ratio: int = 4,
        num_rbf: int = 50,
        num_heads: int = 8,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        neighbor_backend: str = "auto",
        use_spatial_refiner: bool = False,
        time_scale_ps: float = 100.0,
    ) -> None:
        super().__init__()
        self.frame_encoder = PVBFrameEncoder(
            hidden_channels=hidden_channels,
            num_layers=spatial_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            neighbor_backend=neighbor_backend,
        )
        self.temporal_encoder = CausalTemporalEncoder(
            hidden_channels,
            hidden_channels,
            ratio=temporal_ratio,
            num_layers=temporal_layers,
            num_heads=num_heads,
            time_scale_ps=time_scale_ps,
        )
        refiner = None
        if use_spatial_refiner:
            refiner = LatentConditionedSpatialRefiner(
                hidden_channels,
                hidden_channels,
                hidden_channels=hidden_channels,
                num_layers=spatial_layers,
                num_rbf=num_rbf,
                num_heads=num_heads,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_num_neighbors=max_num_neighbors,
            )
        self.decoder = JointMultiFrameDecoder(
            hidden_channels,
            hidden_channels,
            temporal_layers=temporal_layers,
            num_heads=num_heads,
            time_scale_ps=time_scale_ps,
            spatial_refiner=refiner,
        )

    def forward(self, batch: ClipBatch):
        encoded = self.frame_encoder(batch)
        state = self.temporal_encoder(
            encoded.h,
            encoded.v,
            batch.time_ps,
            frame_mask=batch.frame_mask,
            abid=batch.abid,
        )
        graph = encoded.graph
        topology = {
            "z": graph.z,
            "b": graph.b,
            "batch": graph.batch,
            "edge_index": graph.edge_index,
            "bond_type": graph.bond_type,
        }
        latent = CodecLatent.from_temporal_state(
            state,
            x_anchor=batch.x[0],
            topology=topology,
        )
        return self.decoder(
            latent,
            target_time_ps=batch.time_ps,
            target_mask=batch.frame_mask,
        )


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _to_device(getattr(value, field.name), device) for field in fields(value)})
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


class CodecTrainer:
    """Small self-contained optimizer loop with versioned checkpoint state."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[Any],
        valid_loader: Optional[Iterable[Any]] = None,
        config: CodecTrainConfig | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
        normalization_stats: Mapping[str, BucketNormalization | Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = (
            config if isinstance(config, CodecTrainConfig)
            else CodecTrainConfig.from_mapping(config) if config is not None
            else CodecTrainConfig()
        )
        selected_device = device or self.config.device
        if str(selected_device) == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(selected_device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.normalization_stats: dict[str, BucketNormalization] = {}
        for key, value in (normalization_stats or {}).items():
            self.normalization_stats[str(key)] = value if isinstance(value, BucketNormalization) else BucketNormalization.from_dict(value)
        self.step = 0
        self.epoch = 0
        self._train_iterator: Optional[Iterable[Any]] = None

    def fit_normalization(self, batches: Optional[Iterable[Any]] = None) -> dict[str, BucketNormalization]:
        source = self.train_loader if batches is None else batches
        self.normalization_stats = fit_time_bucket_normalization(
            source,
            min_count=self.config.normalization_min_count,
            epsilon=self.config.normalization_epsilon,
        )
        return self.normalization_stats

    def _validate_batch_clock(self, batch: Any) -> float:
        bucket_ids = tuple(str(item) for item in batch.time_bucket_id)
        if not bucket_ids:
            raise ValueError("codec batch must contain at least one time bucket")
        weights = []
        for sample_index, bucket_id in enumerate(bucket_ids):
            spec = self.config.bucket(bucket_id)
            weights.append(spec.weight)
            if getattr(batch, "frames", int(batch.x.shape[0])) > 1:
                valid_delta = batch.delta_time_ps[sample_index][batch.frame_mask[sample_index, 1:]]
                if valid_delta.numel() and torch.any((valid_delta - spec.center_ps).abs() > spec.tolerance_ps):
                    raise ValueError(
                        f"batch delta_time_ps does not match bucket {bucket_id!r} center/tolerance"
                    )
        return float(sum(weights) / len(weights))

    def _loss_for_batch(self, batch: Any) -> tuple[dict[str, Tensor], float, Any]:
        batch = _to_device(batch, self.device)
        bucket_weight = self._validate_batch_clock(batch)
        output = self.model(batch)
        losses = compute_codec_losses(
            output,
            batch,
            weights=self.config.weights_at(self.step),
            normalization=self.normalization_stats,
        )
        losses["total"] = losses["total"] * bucket_weight
        return losses, bucket_weight, batch

    @staticmethod
    def _metrics(losses: Mapping[str, Tensor], batch: Any) -> dict[str, float]:
        metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
        task = torch.as_tensor(batch.task, dtype=torch.long).flatten()
        for task_id in torch.unique(task).tolist():
            task_name = TASK_NAMES[int(task_id)]
            metrics[f"task_{task_name}_total"] = metrics["total"]
        for bucket_id in dict.fromkeys(str(item) for item in batch.time_bucket_id):
            safe_bucket = bucket_id.replace("/", "_")
            metrics[f"bucket_{safe_bucket}_velocity"] = metrics["velocity"]
            metrics[f"bucket_{safe_bucket}_velocity_raw"] = metrics["velocity_raw"]
            metrics[f"bucket_{safe_bucket}_acceleration"] = metrics["acceleration"]
            metrics[f"bucket_{safe_bucket}_acceleration_raw"] = metrics["acceleration_raw"]
        return metrics

    def optimizer_step(self, batch: Any) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses, _, moved_batch = self._loss_for_batch(batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("codec loss is NaN or Inf before backward")
        total.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("codec gradient is NaN or Inf")
        self.optimizer.step()
        self.step += 1
        if self.config.warmup_steps:
            scale = min(1.0, self.step / float(self.config.warmup_steps))
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr * scale
        return self._metrics(losses, moved_batch)

    @torch.no_grad()
    def evaluate_batch(self, batch: Any) -> dict[str, float]:
        self.model.eval()
        losses, _, moved_batch = self._loss_for_batch(batch)
        return self._metrics(losses, moved_batch)

    def run(
        self,
        max_steps: Optional[int] = None,
        *,
        log_path: str | Path | None = None,
        log_every: int = 1,
    ) -> dict[str, float]:
        """Optimize until max_steps and optionally append step metrics as JSONL."""
        target = int(max_steps if max_steps is not None else self.config.max_steps)
        if target < self.step:
            raise ValueError("max_steps cannot be less than the current checkpoint step")
        if int(log_every) < 1:
            raise ValueError("log_every must be positive")
        log_handle = None
        if log_path is not None:
            destination = Path(log_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            log_handle = destination.open("a", encoding="utf-8")
        iterator = iter(self.train_loader)
        last: dict[str, float] = {}
        try:
            while self.step < target:
                try:
                    batch = next(iterator)
                except StopIteration:
                    self.epoch += 1
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                last = self.optimizer_step(batch)
                if log_handle is not None and (self.step % int(log_every) == 0 or self.step == target):
                    record = {
                        "schema_version": "pvb.codec.train.v1",
                        "split": "train",
                        "step": int(self.step),
                        "epoch": int(self.epoch),
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "metrics": last,
                    }
                    log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    log_handle.flush()
        finally:
            if log_handle is not None:
                log_handle.close()
        return last

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": CODEC_CHECKPOINT_SCHEMA,
            "step": int(self.step),
            "epoch": int(self.epoch),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "normalization_stats": {
                key: value.as_dict() for key, value in self.normalization_stats.items()
            },
            "config": self.config.as_dict(),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), destination)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != CODEC_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"unsupported codec checkpoint schema; expected {CODEC_CHECKPOINT_SCHEMA!r}"
            )
        required = {"step", "epoch", "model_state", "optimizer_state", "normalization_stats", "config"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"codec checkpoint is missing fields: {sorted(missing)}")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.step = int(payload["step"])
        self.epoch = int(payload["epoch"])
        self.normalization_stats = {
            str(key): BucketNormalization.from_dict(value)
            for key, value in payload["normalization_stats"].items()
        }


__all__ = [
    "CODEC_CHECKPOINT_SCHEMA",
    "CODEC_CONFIG_SCHEMA",
    "CodecTrainConfig",
    "CodecTrainer",
    "PVBCodecModel",
    "TimeBucketSpec",
    "validate_codec_config",
]
