#!/usr/bin/env python3
"""Run the independent Session B data/capacity comparison.

The runner owns four experiment directories and never reuses a trained DiT
weight as another experiment's starting point.  Frozen codec/statistics and
the read-only selected clip views are shared; trainable adapters, models,
optimizers, scalers, RNGs, cursors, and checkpoints are not.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dit_capacity_data import BlockedDataError, CapacityData, load_capacity_data
from evaluation.dit_diagnostics import (
    aggregate_rows,
    parse_sample_id,
    stable_seed,
    trajectory_metric_record,
)
from evaluation.dit_evaluation import evaluate_oracle_vs_generated
from evaluation.codec_evaluation import (
    _candidate_nonbond_pairs,
    _nonbond_pairs,
    aligned_rmsf_metrics,
    dynamic_acf_metrics,
)
from module.dit_latent_cache import LatentFieldCache, cache_key
from module.latent_flow_source import build_observed_center, future_field_masks
from module.latent_rectified_flow import (
    LatentFieldSet,
    generate_state_detail_latent,
)
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    contract_hash,
)
from scripts.run_dit_source_ab import (
    _bond_rmse,
    _coordinate_rms,
    _merge_center_with_observed,
    _source_metric_row,
)
from scripts.run_state_detail_dit_pilot import (
    FrozenCodec,
    _encode_batch,
    _load_approved_codec,
    _make_sampler,
    _make_validation_plan,
    _observed_batch,
    _sha256,
)
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


EXPERIMENT_IDS = ("G48", "C48", "C192", "C48D8")
HISTORY_SCHEDULE = (4, 8)
CHECKPOINT_STEPS = (4500, 10000, 20000)
CAPACITY_SCHEMA = "pvb.dit.capacity_data.runner.v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _git_commit() -> str:
    override = os.environ.get("DIT_CODE_COMMIT", "").strip().lower()
    if override:
        if len(override) != 40 or any(character not in "0123456789abcdef" for character in override):
            raise ValueError("DIT_CODE_COMMIT must be a full 40-character hexadecimal commit")
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Tensor):
        if value.numel() == 1:
            number = float(value.detach().float().cpu())
            return number if math.isfinite(number) else None
        return value.detach().float().cpu().tolist()
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe(value), sort_keys=True) + "\n")


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(value), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("capacity configuration must be a mapping")
    if int(value.get("schedule", {}).get("checkpoint_interval", 0)) <= 0:
        raise ValueError("capacity checkpoint_interval must be a positive integer")
    return json.loads(json.dumps(value))


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capacity numerical stages require an actual CUDA device")
    torch.cuda.set_device(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_info(device: torch.device) -> dict[str, Any]:
    _require_cuda(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "requested_device": str(device),
        "logical_index": torch.cuda.current_device(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": str(getattr(properties, "uuid", "")),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tf32": False,
        "amp": "bfloat16",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }


def _tensor_bytes_hash(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_tensor_hashes(state: Mapping[str, Tensor]) -> dict[str, str]:
    return {str(name): _tensor_bytes_hash(value) for name, value in sorted(state.items())}


def _recursive_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, Tensor):
            digest.update(b"tensor")
            digest.update(_tensor_bytes_hash(item).encode())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _copy_state_cpu(model: torch.nn.Module) -> tuple[dict[str, Tensor], str, dict[str, str]]:
    state = {name: value.detach().to(device="cpu").clone() for name, value in model.state_dict().items()}
    return state, module_state_hash(model), _state_tensor_hashes(state)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    source_mode: str
    depth: int
    data_scale: str


@dataclass
class CapacityContext:
    cfg: dict[str, Any]
    output_dir: Path
    data: CapacityData
    codec: FrozenCodec
    statistics: LatentStatistics
    statistics_device: LatentStatistics
    data_adapter: StateDetailLatentAdapter
    device: torch.device

    @property
    def data_hash(self) -> str:
        return str(self.data.data_hash)

    def close(self) -> None:
        self.data.close()


def _experiment_specs(cfg: Mapping[str, Any]) -> dict[str, ExperimentSpec]:
    values: dict[str, ExperimentSpec] = {}
    configured = cfg.get("experiments", {})
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id not in configured:
            raise RuntimeError(f"capacity config is missing experiment {experiment_id}")
        item = configured[experiment_id]
        values[experiment_id] = ExperimentSpec(
            experiment_id=experiment_id,
            source_mode=str(item["source_mode"]),
            depth=int(item["depth"]),
            data_scale=str(item["data_scale"]),
        )
        if values[experiment_id].source_mode not in ("gaussian", "conditional"):
            raise ValueError(f"unsupported source mode for {experiment_id}")
        if values[experiment_id].data_scale not in ("base48", "expanded192"):
            raise ValueError(f"unsupported data scale for {experiment_id}")
    if values["G48"].depth != 4 or values["C48"].depth != 4 or values["C192"].depth != 4:
        raise ValueError("G48/C48/C192 must use depth 4")
    if values["C48D8"].depth != 8:
        raise ValueError("C48D8 must use depth 8")
    return values


def _blocked_experiments(
    ctx: CapacityContext,
    specs: Mapping[str, ExperimentSpec],
) -> dict[str, str]:
    return {
        experiment_id: str(ctx.data.blocked_data[spec.data_scale])
        for experiment_id, spec in specs.items()
        if spec.data_scale in ctx.data.blocked_data
    }


def _manifest_metadata(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = _resolve(cfg["data_manifest_root"])
    manifest_path = root / "manifest.json"
    materialization_path = root / "materialization.json"
    if not manifest_path.is_file() or not materialization_path.is_file():
        raise FileNotFoundError(f"capacity data manifest/materialization missing under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if manifest.get("test_sampling", {}).get("opened") is not False or materialization.get("test_opened") is not False:
        raise RuntimeError("capacity preflight detected opened test data")
    content = dict(manifest)
    if content.pop("manifest_content_sha256", None) != _canonical_hash(content):
        raise RuntimeError("capacity manifest content hash is invalid")
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("capacity data manifest is not FROZEN")
    candidate = cfg["candidate"]
    data_overrides = dict(cfg.get("data_source_overrides", {}))
    paths = {
        "codec_result": _resolve(candidate["codec_result"]),
        "codec_checkpoint": _resolve(candidate["codec_checkpoint"]),
        "statistics": _resolve(candidate["statistics"]),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"capacity preflight missing {label}: {path}")
    statistics_hash = _sha256(paths["statistics"])
    if statistics_hash != str(candidate["statistics_file_sha256"]):
        raise RuntimeError("frozen statistics file hash differs from capacity config")
    codec_hash = _sha256(paths["codec_checkpoint"])
    if codec_hash != str(candidate["codec_checkpoint_sha256"]):
        raise RuntimeError("frozen codec checkpoint hash differs from capacity config")
    expected_source_hash = str(manifest["source_manifest_sha256"])
    source_manifest = _resolve(
        data_overrides.get("source_manifest", str(manifest["source_manifest"]))
    )
    if source_manifest.is_file():
        if _sha256(source_manifest) != expected_source_hash:
            raise RuntimeError("approved source manifest hash differs from capacity manifest")
        source_proof = {
            "kind": "direct_source_manifest",
            "path": str(source_manifest),
            "sha256": expected_source_hash,
        }
    else:
        provenance_value = data_overrides.get("provenance_manifest")
        if not provenance_value:
            raise FileNotFoundError(
                f"source manifest is missing and no frozen provenance manifest was configured: {source_manifest}"
            )
        provenance_manifest = _resolve(provenance_value)
        if not provenance_manifest.is_file():
            raise FileNotFoundError(f"frozen provenance manifest is missing: {provenance_manifest}")
        provenance = json.loads(provenance_manifest.read_text(encoding="utf-8"))
        if provenance.get("status") != "FROZEN":
            raise RuntimeError("source provenance manifest is not FROZEN")
        if provenance.get("source_manifest_sha256") != expected_source_hash:
            raise RuntimeError("source provenance manifest records a different source hash")
        if provenance.get("test_sampling", {}).get("opened") is not False:
            raise RuntimeError("source provenance manifest does not certify unopened test sampling")
        source_proof = {
            "kind": "frozen_nested_manifest",
            "path": str(provenance_manifest),
            "sha256": _sha256(provenance_manifest),
            "recorded_source_manifest_sha256": expected_source_hash,
        }
    source_root_overrides = {}
    for label in ("train", "valid"):
        if label not in data_overrides:
            continue
        source_root = _resolve(data_overrides[label])
        if "test" in {part.lower() for part in source_root.parts}:
            raise RuntimeError(f"source override unexpectedly references test data: {source_root}")
        if not (source_root / "data.bin").is_file() or not (source_root / "index.txt").is_file():
            raise FileNotFoundError(f"source override is incomplete: {source_root}")
        source_root_overrides[label] = str(source_root)
    return {
        "schema": "pvb.dit.capacity_data.preflight.v1",
        "manifest_root": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "materialization_sha256": _sha256(materialization_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "data_counts": manifest.get("counts", {}),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": expected_source_hash,
        "source_manifest_proof": source_proof,
        "source_root_overrides": source_root_overrides,
        "allow_missing_expanded": bool(data_overrides.get("allow_missing_expanded", False)),
        "codec_result": str(paths["codec_result"]),
        "codec_result_sha256": _sha256(paths["codec_result"]),
        "codec_checkpoint": str(paths["codec_checkpoint"]),
        "codec_checkpoint_sha256": codec_hash,
        "statistics": str(paths["statistics"]),
        "statistics_file_sha256": statistics_hash,
        "test_payload_opened": False,
        "code_commit": _git_commit(),
    }


def _load_context(cfg: dict[str, Any], output_dir: Path, device: torch.device) -> CapacityContext:
    _require_cuda(device)
    data_overrides = dict(cfg.get("data_source_overrides", {}))
    source_root_overrides = {
        label: _resolve(data_overrides[label])
        for label in ("train", "valid")
        if label in data_overrides
    }
    data = load_capacity_data(
        _resolve(cfg["data_manifest_root"]),
        source_root_overrides=source_root_overrides,
        allow_missing_expanded=bool(data_overrides.get("allow_missing_expanded", False)),
    )
    candidate = cfg["candidate"]
    codec = _load_approved_codec(
        candidate=str(candidate["mode"]),
        result_path=_resolve(candidate["codec_result"]),
        checkpoint_path=_resolve(candidate["codec_checkpoint"]),
        device=device,
    )
    statistics_path = _resolve(candidate["statistics"])
    statistics = LatentStatistics.from_state_dict(
        torch.load(statistics_path, map_location="cpu", weights_only=False)
    )
    if statistics.ratio != int(candidate["ratio"]) or statistics.mode != str(candidate["mode"]):
        raise RuntimeError("capacity statistics ratio/mode mismatch")
    if _sha256(statistics_path) != str(candidate["statistics_file_sha256"]):
        raise RuntimeError("capacity statistics file changed")
    model_cfg = cfg["model"]
    data_adapter = StateDetailLatentAdapter(
        codec_width=int(model_cfg["codec_width"]),
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        ratio=int(candidate["ratio"]),
    ).to(device)
    return CapacityContext(
        cfg=cfg,
        output_dir=output_dir,
        data=data,
        codec=codec,
        statistics=statistics,
        statistics_device=statistics.to(device=device),
        data_adapter=data_adapter,
        device=device,
    )


def _new_adapter(ctx: CapacityContext, device: torch.device) -> StateDetailLatentAdapter:
    model_cfg = ctx.cfg["model"]
    return StateDetailLatentAdapter(
        codec_width=int(model_cfg["codec_width"]),
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        ratio=int(ctx.cfg["candidate"]["ratio"]),
    ).to(device)


def _new_model(ctx: CapacityContext, spec: ExperimentSpec, adapter: StateDetailLatentAdapter) -> MolecularDiT:
    model_cfg = ctx.cfg["model"]
    return MolecularDiT(
        adapter=adapter,
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        depth=int(spec.depth),
        heads=int(model_cfg["heads"]),
        ffn_multiplier=int(model_cfg["ffn_multiplier"]),
        dropout=float(model_cfg["dropout"]),
        execution_backend=str(model_cfg["execution_backend"]),
    ).to(ctx.device)


def _initialization(ctx: CapacityContext, depth: int) -> tuple[dict[str, Tensor], str, dict[str, str], int]:
    torch.manual_seed(int(ctx.cfg["seed"]["init"]))
    adapter = _new_adapter(ctx, ctx.device)
    spec = ExperimentSpec("init", "gaussian", int(depth), "base48")
    model = _new_model(ctx, spec, adapter)
    state, init_hash, tensor_hashes = _copy_state_cpu(model)
    return state, init_hash, tensor_hashes, int(model.parameter_count)


def _make_trainer(
    ctx: CapacityContext,
    spec: ExperimentSpec,
    target_steps: int,
    init_state: Mapping[str, Tensor],
    init_hash: str,
) -> DiTTrainer:
    torch.manual_seed(int(ctx.cfg["seed"]["init"]))
    adapter = _new_adapter(ctx, ctx.device)
    model = _new_model(ctx, spec, adapter)
    model.load_state_dict(init_state, strict=True)
    if module_state_hash(model) != init_hash:
        raise RuntimeError(f"initialization state did not round-trip for {spec.experiment_id}")
    model_cfg = ctx.cfg["model"]
    protocol = ctx.cfg["protocol"]
    train_cfg = DiTTrainConfig(
        ratio=int(ctx.cfg["candidate"]["ratio"]),
        mode=str(ctx.cfg["candidate"]["mode"]),
        codec_width=int(model_cfg["codec_width"]),
        scalar_width=int(model_cfg["scalar_width"]),
        vector_width=int(model_cfg["vector_width"]),
        depth=int(spec.depth),
        heads=int(model_cfg["heads"]),
        ffn_multiplier=int(model_cfg["ffn_multiplier"]),
        dropout=float(model_cfg["dropout"]),
        learning_rate=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        grad_clip=float(protocol["grad_clip"]),
        max_steps=int(target_steps),
        seed=int(ctx.cfg["seed"]["training"]),
        amp=True,
        output_root=str(ctx.output_dir / spec.experiment_id),
        data_hash=ctx.data_hash,
        codec_hash=ctx.codec.codec_state_hash,
        stats_hash=ctx.statistics.hash,
        source_mode=spec.source_mode,
        center_kind=str(ctx.cfg["source"]["center_kind"]),
        source_sigma=float(ctx.cfg["source"]["sigma"]),
        normalization_hash=ctx.statistics.hash,
        init_hash=init_hash,
        observation_mixture=tuple(int(value) for value in ctx.cfg["schedule"]["observation_history"]),
        metadata={
            "phase": "dit_capacity_data_v1",
            "experiment_id": spec.experiment_id,
            "source_mode": spec.source_mode,
            "data_scale": spec.data_scale,
            "depth": int(spec.depth),
            "execution_backend": "factorized_v2",
            "capacity_schema": CAPACITY_SCHEMA,
        },
    )
    return DiTTrainer(
        model,
        adapter,
        config=train_cfg,
        statistics=ctx.statistics_device,
        codec=ctx.codec.model,
        frame_encoder=ctx.codec.model.frame_encoder,
    )


def _prepare_encoded(
    ctx: CapacityContext,
    dataset: Any,
    indices: Sequence[int],
) -> tuple[Any, Any, Any]:
    latent, batch_cpu, batch = _encode_batch(
        dataset,
        indices,
        codec=ctx.codec,
        adapter=ctx.data_adapter,
        data_hash=ctx.data_hash,
        device=ctx.device,
    )
    target_batch = ctx.data_adapter.pack(
        latent,
        codec_hash=ctx.codec.codec_state_hash,
        data_hash=ctx.data_hash,
        origin_from_latent=True,
        loss_mask=batch.loss_mask,
    )
    return batch_cpu, batch, target_batch


def _observed(ctx: CapacityContext, target_batch: Any, batch: Any, history: int) -> Any:
    return _observed_batch(
        target_batch,
        batch,
        adapter=ctx.data_adapter,
        history_frames=int(history),
    )


def _decode_capacity_fields(
    ctx: CapacityContext,
    target_batch: Any,
    fields: LatentFieldSet,
) -> Tensor:
    raw = ctx.statistics_device.inverse_fields(fields)
    latent = ctx.data_adapter.make_generated_latent(target_batch, raw)
    with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
        return ctx.codec.model.decode(latent).x_hat.float()


def _center(
    ctx: CapacityContext,
    batch: Any,
    target_batch: Any,
    history: int,
    cache: LatentFieldCache,
) -> tuple[LatentFieldSet, dict[str, Any]]:
    key = cache_key(
        {
            "schema": "pvb.dit.capacity.center.v1",
            "sample_ids": list(batch.sample_id),
            "history": int(history),
            "center_kind": str(ctx.cfg["source"]["center_kind"]),
            "codec_hash": ctx.codec.codec_state_hash,
            "statistics_hash": ctx.statistics.hash,
            "data_hash": ctx.data_hash,
        }
    )
    cached = cache.get(key, device=ctx.device, dtype=target_batch.state_h.dtype)
    if cached is not None:
        return cached, {"cache": "hit", "cache_key": key, "uses_future_coordinates": False}
    center, metadata = build_observed_center(
        str(ctx.cfg["source"]["center_kind"]),
        codec_model=ctx.codec.model,
        coordinate_batch=batch,
        target_batch=target_batch,
        adapter=ctx.data_adapter,
        statistics=ctx.statistics_device,
        history_frames=int(history),
        codec_hash=ctx.codec.codec_state_hash,
        data_hash=ctx.data_hash,
    )
    cache.put(key, center)
    return center, {
        "cache": "miss",
        "cache_key": key,
        "uses_future_coordinates": False,
        "template_coordinates": metadata.get("template_coordinates"),
        "template_latent": metadata.get("template_latent"),
    }


def _optimizer_hash(optimizer: torch.optim.Optimizer) -> str:
    return _recursive_hash(optimizer.state_dict())


def _trainable_storage(model: torch.nn.Module) -> set[int]:
    return {
        int(parameter.detach().data_ptr())
        for parameter in model.parameters()
        if parameter.requires_grad
    }


def _source_check(ctx: CapacityContext) -> dict[str, Any]:
    selected = []
    for index, row in enumerate(ctx.data.valid._index):
        parsed = parse_sample_id(str(row[0]))
        if parsed["replica"] == "R1" and parsed["window"] == 30:
            selected.append((index, str(row[0])))
    selected.sort(key=lambda item: item[1])
    if len(selected) != 8:
        raise RuntimeError(f"source check requires eight fixed validation clips, got {len(selected)}")
    rows: list[dict[str, Any]] = []
    center_kind = str(ctx.cfg["source"]["center_kind"])
    for index, sample_id in selected:
        _batch_cpu, batch, target_batch = _prepare_encoded(ctx, ctx.data.valid, (index,))
        oracle_latent = ctx.codec.model.encode(batch)
        oracle_coordinates = ctx.codec.model.decode(oracle_latent).x_hat.float()
        for history in HISTORY_SCHEDULE:
            observed = _observed(ctx, target_batch, batch, history)
            center, metadata = _center(ctx, batch, observed, history, LatentFieldCache(mode="disabled"))
            merged = _merge_center_with_observed(ctx, observed, center)
            decoded = _decode_capacity_fields(ctx, observed, merged)
            template = metadata.get("template_coordinates")
            if template is None:
                template = batch.x.clone()
                template[history:] = batch.x[history - 1].unsqueeze(0)
            metrics = _source_metric_row(decoded, template, batch, history, oracle_coordinates)
            metrics.update({
                "sample_id": sample_id,
                "history_frames": history,
                "center_kind": center_kind,
                "uses_future_coordinates": False,
                "implementation_pass": bool(
                    metrics["template_raw_rmsd"] <= metrics["threshold_template_raw_rmsd"]
                    and metrics["template_bond_rmse"] <= 0.10
                    and metrics["decoded_future_internal_displacement_rms"] <= 1.0e-4
                ),
            })
            rows.append(metrics)
    passed = bool(rows) and all(bool(row["implementation_pass"]) for row in rows)
    result = {
        "schema": "pvb.dit.capacity_data.source_check.v1",
        "status": "PASS" if passed else "FAIL",
        "selected_center_kind": center_kind if passed else None,
        "fixed_clip_count": 8,
        "rows": rows,
        "thresholds": {
            "template_raw_rmsd": "max(0.10 A, 5 * clip codec oracle raw RMSD)",
            "template_bond_rmse": 0.10,
            "decoded_future_internal_displacement_rms": 1.0e-4,
        },
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "source_decision.json", result)
    return result


def _first_train_batch(dataset: Any, seed: int) -> tuple[int, ...]:
    sampler = _make_sampler(dataset, seed=int(seed), clips_per_trajectory=24)
    batches = sampler.global_batches
    if not batches:
        raise RuntimeError("capacity train sampler produced no batches")
    return tuple(int(value) for value in batches[0])


def _isolation_check(ctx: CapacityContext, specs: Mapping[str, ExperimentSpec]) -> dict[str, Any]:
    init4, hash4, hashes4, _count4 = _initialization(ctx, 4)
    init8, hash8, hashes8, _count8 = _initialization(ctx, 8)
    g = _make_trainer(ctx, specs["G48"], 2, init4, hash4)
    c = _make_trainer(ctx, specs["C48"], 2, init4, hash4)
    d8 = _make_trainer(ctx, specs["C48D8"], 2, init8, hash8)
    storage_g = _trainable_storage(g.model)
    storage_c = _trainable_storage(c.model)
    storage_d8 = _trainable_storage(d8.model)
    overlap_gc = sorted(storage_g.intersection(storage_c))
    overlap_gd8 = sorted(storage_g.intersection(storage_d8))
    if module_state_hash(g.model) != hash4 or module_state_hash(c.model) != hash4:
        raise RuntimeError("same-shape initialization hash mismatch")
    if module_state_hash(d8.model) != hash8 or hash8 == hash4:
        raise RuntimeError("depth-8 initialization hash is not independently recorded")
    indices = _first_train_batch(ctx.data.train48, int(ctx.cfg["seed"]["training"]))
    _batch_cpu, batch, target_batch = _prepare_encoded(ctx, ctx.data.train48, indices)
    observed = _observed(ctx, target_batch, batch, 4)
    center, _ = _center(ctx, batch, observed, 4, LatentFieldCache(mode="disabled"))
    generator_g = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    generator_c = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    generator_d8 = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    c_before = module_state_hash(c.model)
    c_opt_before = _optimizer_hash(c.optimizer)
    g.train_step(observed, generator=generator_g)
    c_after_g = module_state_hash(c.model)
    c_opt_after_g = _optimizer_hash(c.optimizer)
    if c_before != c_after_g or c_opt_before != c_opt_after_g:
        raise RuntimeError("model 1 changed model 2 parameters or optimizer state")
    g_before_c = module_state_hash(g.model)
    g_opt_before_c = _optimizer_hash(g.optimizer)
    c.train_step(observed, generator=generator_c, source_center=center)
    if g_before_c != module_state_hash(g.model) or g_opt_before_c != _optimizer_hash(g.optimizer):
        raise RuntimeError("model 2 changed model 1 parameters or optimizer state")
    d8.train_step(observed, generator=generator_d8, source_center=center)
    resume_dir = ctx.output_dir / "verification" / "resume"
    resume_path = resume_dir / "continuous_step1.pt"
    continuous = _make_trainer(ctx, specs["G48"], 2, init4, hash4)
    resumed = _make_trainer(ctx, specs["G48"], 2, init4, hash4)
    gen_cont = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    gen_resume = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    continuous.train_step(observed, generator=gen_cont)
    payload = continuous.checkpoint_payload()
    payload["capacity"] = {
        "schema": CAPACITY_SCHEMA + ".checkpoint.v1",
        "experiment_id": "G48",
        "init_hash": hash4,
        "generator_state": gen_cont.get_state(),
        "cursor": {"epoch": 0, "batch_index": 1},
        "successful_optimizer_updates": continuous.successful_updates,
    }
    _atomic_torch_save(resume_path, payload)
    continuous_row = continuous.train_step(observed, generator=gen_cont)
    loaded = resumed.load_checkpoint(resume_path, map_location=ctx.device)
    gen_resume.set_state(loaded["capacity"]["generator_state"].detach().to(device="cpu"))
    resumed_row = resumed.train_step(observed, generator=gen_resume)
    resume_components = {
        "model_state": _recursive_hash(continuous.model.state_dict())
        == _recursive_hash(resumed.model.state_dict()),
        "optimizer_state": _optimizer_hash(continuous.optimizer)
        == _optimizer_hash(resumed.optimizer),
        "tau_mean": continuous_row["tau_mean"] == resumed_row["tau_mean"],
        "tau_min": continuous_row["tau_min"] == resumed_row["tau_min"],
        "tau_max": continuous_row["tau_max"] == resumed_row["tau_max"],
    }
    resume_match = all(resume_components.values())
    if not resume_match:
        raise RuntimeError(
            "continuous and save/resume next updates differ: "
            + json.dumps(resume_components, sort_keys=True)
        )
    result = {
        "schema": "pvb.dit.capacity_data.isolation_verification.v1",
        "status": "PASS",
        "device": _cuda_info(ctx.device),
        "same_shape_initialization_hash": hash4,
        "same_shape_tensor_hashes": hashes4,
        "depth8_initialization_hash": hash8,
        "depth8_tensor_hashes": hashes8,
        "parameter_counts": {
            "G48": sum(parameter.numel() for parameter in g.model.parameters()),
            "C48": sum(parameter.numel() for parameter in c.model.parameters()),
            "C48D8": sum(parameter.numel() for parameter in d8.model.parameters()),
        },
        "trainable_storage_overlap": {
            "G48_C48": overlap_gc,
            "G48_C48D8": overlap_gd8,
        },
        "cross_mutation": {
            "G48_does_not_change_C48": c_before == c_after_g,
            "C48_does_not_change_G48": g_before_c == module_state_hash(g.model),
            "optimizer_state_isolated": c_opt_before == c_opt_after_g and g_opt_before_c == _optimizer_hash(g.optimizer),
        },
        "resume": {
            "continuous_updates": continuous.successful_updates,
            "restored_updates": resumed.successful_updates,
            "batch_cursor": {"epoch": 0, "batch_index": 1},
            "next_update_match": resume_match,
            "next_update_components": resume_components,
        },
        "test_payload_opened": False,
    }
    del g, c, d8, continuous, resumed, batch, target_batch, observed, center
    torch.cuda.empty_cache()
    return result


def _real_smoke(ctx: CapacityContext, specs: Mapping[str, ExperimentSpec]) -> dict[str, Any]:
    states: dict[int, tuple[dict[str, Tensor], str]] = {}
    for depth in (4, 8):
        state, init_hash, _hashes, _count = _initialization(ctx, depth)
        states[depth] = (state, init_hash)
    clip_specs = ctx.data.train48.clip_spec_table()
    ordered_indices = sorted(
        range(len(clip_specs)),
        key=lambda index: (
            clip_specs[index].effective_tokens,
            clip_specs[index].sample_id,
        ),
    )
    indices = (int(ordered_indices[len(ordered_indices) // 2]),)
    _batch_cpu, batch, target_batch = _prepare_encoded(ctx, ctx.data.train48, indices)
    smoke_sample_ids = list(batch.sample_id)
    smoke_batch_tokens = _batch_tokens(ctx.data.train48, indices)
    rows = []
    blocked = _blocked_experiments(ctx, specs)
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id in blocked:
            continue
        torch.cuda.empty_cache()
        spec = specs[experiment_id]
        init_state, init_hash = states[spec.depth]
        trainer = _make_trainer(ctx, spec, 1, init_state, init_hash)
        train_observed = _observed(ctx, target_batch, batch, 4)
        train_center = None
        if spec.source_mode == "conditional":
            train_center, _ = _center(ctx, batch, train_observed, 4, LatentFieldCache(mode="disabled"))
        train_row = trainer.train_step(
            train_observed,
            generator=torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"])),
            source_center=train_center,
        )
        for history in HISTORY_SCHEDULE:
            observed = _observed(ctx, target_batch, batch, history)
            center = None
            if spec.source_mode == "conditional":
                center, _ = _center(ctx, batch, observed, history, LatentFieldCache(mode="disabled"))
            generated, metadata = generate_state_detail_latent(
                trainer.model,
                trainer.adapter,
                observed,
                ctx.statistics_device,
                steps=16,
                seed=stable_seed(int(ctx.cfg["seed"]["validation"]), "smoke", history, 0, experiment_id),
                source_center=center,
                source_mode=spec.source_mode,
            )
            decoded = ctx.codec.model.decode(generated).x_hat.float()
            latent_fields = (
                generated.state_h,
                generated.detail_h,
                generated.state_v,
                generated.detail_v,
            )
            finite = bool(
                torch.isfinite(decoded).all()
                and all(value is not None and torch.isfinite(value).all() for value in latent_fields)
            )
            row = {
                "experiment_id": experiment_id,
                "history_frames": history,
                "train_step": train_row if history == HISTORY_SCHEDULE[0] else None,
                "generated_shape": list(decoded.shape),
                "finite": finite,
                "observed_clamp_exact": bool(metadata["observed_clamp_exact"]),
                "device": str(ctx.device),
            }
            rows.append(row)
            if not finite or not row["observed_clamp_exact"]:
                raise RuntimeError(f"real clip smoke failed for {experiment_id}, H{history}")
            del observed, center, generated, decoded, metadata
        del trainer, train_observed, train_center
        torch.cuda.empty_cache()
    return {
        "schema": "pvb.dit.capacity_data.real_smoke.v1",
        "status": "PASS",
        "rows": rows,
        "sample_ids": smoke_sample_ids,
        "batch_tokens": smoke_batch_tokens,
        "selection": "deterministic median effective-token train48 clip",
        "blocked_experiments": {
            experiment_id: {"status": "BLOCKED_DATA", "reason": reason}
            for experiment_id, reason in blocked.items()
        },
        "test_payload_opened": False,
    }


def _verify(ctx: CapacityContext, specs: Mapping[str, ExperimentSpec]) -> dict[str, Any]:
    isolation = _isolation_check(ctx, specs)
    smoke = _real_smoke(ctx, specs)
    blocked = _blocked_experiments(ctx, specs)
    result = {
        "schema": "pvb.dit.capacity_data.verification.v1",
        "status": "PASS",
        "isolation": isolation,
        "real_smoke": smoke,
        "blocked_experiments": {
            experiment_id: {"status": "BLOCKED_DATA", "reason": reason}
            for experiment_id, reason in blocked.items()
        },
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "verification.json", result)
    return result


def _dataset_sampler(dataset: Any, cfg: Mapping[str, Any]) -> Any:
    return _make_sampler(
        dataset,
        seed=int(cfg["seed"]["training"]),
        clips_per_trajectory=int(cfg["schedule"]["clips_per_trajectory"]),
    )


def _next_batch(
    sampler: Any,
    cursor: dict[str, int],
    plan_cache: dict[str, Any],
) -> tuple[tuple[int, ...], str, int]:
    epoch = int(cursor["epoch"])
    if plan_cache.get("epoch") != epoch:
        sampler.set_epoch(epoch)
        batches = tuple(tuple(int(value) for value in batch) for batch in sampler.global_batches)
        if not batches:
            raise RuntimeError("capacity sampler produced no batches")
        plan_cache.clear()
        plan_cache.update({
            "epoch": epoch,
            "batches": batches,
            "schedule_hash": _canonical_hash({
                "schema": "pvb.dit.capacity_data.schedule.v1",
                "epoch": epoch,
                "batches": [list(value) for value in batches],
                "selected_sample_ids": list(sampler.selected_sample_ids),
            }),
        })
    batches = plan_cache["batches"]
    if int(cursor["batch_index"]) >= len(batches):
        epoch += 1
        cursor["epoch"] = epoch
        cursor["batch_index"] = 0
        sampler.set_epoch(epoch)
        batches = tuple(tuple(int(value) for value in batch) for batch in sampler.global_batches)
        if not batches:
            raise RuntimeError("capacity sampler produced no batches")
        plan_cache.clear()
        plan_cache.update({
            "epoch": epoch,
            "batches": batches,
            "schedule_hash": _canonical_hash({
                "schema": "pvb.dit.capacity_data.schedule.v1",
                "epoch": epoch,
                "batches": [list(value) for value in batches],
                "selected_sample_ids": list(sampler.selected_sample_ids),
            }),
        })
    batch = batches[int(cursor["batch_index"])]
    cursor["batch_index"] += 1
    return batch, str(plan_cache["schedule_hash"]), epoch


def _batch_tokens(specs: Sequence[Any], indices: Sequence[int]) -> int:
    return int(sum(specs[int(index)].effective_tokens for index in indices))


def _estimate_tokens(dataset: Any, cfg: Mapping[str, Any], steps: int) -> tuple[int, float]:
    sampler = _dataset_sampler(dataset, cfg)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = dataset.clip_spec_table()
    total = 0
    values = []
    for _step in range(int(steps)):
        indices, _schedule_hash, _epoch = _next_batch(sampler, cursor, plan_cache)
        amount = _batch_tokens(specs, indices)
        values.append(amount)
        total += amount
    return total, (sum(values) / max(len(values), 1))


def _profile_one(
    ctx: CapacityContext,
    spec: ExperimentSpec,
    init_state: Mapping[str, Tensor],
    init_hash: str,
) -> dict[str, Any]:
    warmup = int(ctx.cfg["schedule"]["profiling_warmup_updates"])
    measured_count = int(ctx.cfg["schedule"]["profiling_measured_updates"])
    total_updates = warmup + measured_count
    dataset = ctx.data.train_for_scale(spec.data_scale)
    sampler = _dataset_sampler(dataset, ctx.cfg)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = dataset.clip_spec_table()
    trainer = _make_trainer(ctx, spec, 20000, init_state, init_hash)
    generator = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    timings = []
    tokens = []
    cache = LatentFieldCache(mode="disabled")
    for update in range(total_updates):
        indices, _schedule_hash, _epoch = _next_batch(sampler, cursor, plan_cache)
        history = HISTORY_SCHEDULE[update % len(HISTORY_SCHEDULE)]
        started = time.perf_counter()
        _batch_cpu, batch, target_batch = _prepare_encoded(ctx, dataset, indices)
        observed = _observed(ctx, target_batch, batch, history)
        center = None
        if spec.source_mode == "conditional":
            center, _ = _center(ctx, batch, observed, history, cache)
        trainer.train_step(observed, generator=generator, source_center=center)
        _sync(ctx.device)
        timings.append(time.perf_counter() - started)
        tokens.append(_batch_tokens(specs, indices))
    measured = timings[warmup:]
    ordered = sorted(measured)
    p90 = ordered[max(0, int(math.ceil(0.9 * len(ordered))) - 1)]
    p50 = ordered[len(ordered) // 2]
    result = {
        "experiment_id": spec.experiment_id,
        "source_mode": spec.source_mode,
        "data_scale": spec.data_scale,
        "depth": spec.depth,
        "warmup_updates": warmup,
        "measured_updates": measured_count,
        "wall_seconds": measured,
        "p50_seconds": p50,
        "p90_seconds": p90,
        "mean_batch_tokens": sum(tokens[warmup:]) / max(len(tokens[warmup:]), 1),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(ctx.device)),
        "includes": ["clip_read", "metadata", "codec_encode", "source_center", "H2D", "forward", "backward", "optimizer"],
        "test_payload_opened": False,
    }
    del trainer
    torch.cuda.empty_cache()
    return result


def _prepare(ctx: CapacityContext, specs: Mapping[str, ExperimentSpec]) -> dict[str, Any]:
    if not (ctx.output_dir / "verification.json").is_file():
        raise RuntimeError("targeted verification must pass before prepare")
    verification = json.loads((ctx.output_dir / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("verification is not PASS")
    init_by_depth: dict[int, tuple[dict[str, Tensor], str, dict[str, str], int]] = {}
    for depth in sorted({spec.depth for spec in specs.values()}):
        init_by_depth[depth] = _initialization(ctx, depth)
    init_payload = {
        "schema": "pvb.dit.capacity_data.shared_initialization.v1",
        "init_seed": int(ctx.cfg["seed"]["init"]),
        "by_depth": {
            str(depth): {
                "init_hash": payload[1],
                "tensor_hashes": payload[2],
                "parameter_count": payload[3],
                "state_dict": payload[0],
            }
            for depth, payload in init_by_depth.items()
        },
        "test_payload_opened": False,
    }
    _atomic_torch_save(ctx.output_dir / "shared_initialization.pt", init_payload)
    _write_json(ctx.output_dir / "shared_initialization.json", {
        "schema": init_payload["schema"],
        "init_seed": init_payload["init_seed"],
        "by_depth": {
            depth: {key: value for key, value in payload.items() if key != "state_dict"}
            for depth, payload in init_payload["by_depth"].items()
        },
    })
    profile = {
        "schema": "pvb.dit.capacity_data.runner_profile.v1",
        "device": _cuda_info(ctx.device),
        "experiments": {},
        "test_payload_opened": False,
    }
    blocked = _blocked_experiments(ctx, specs)
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id in blocked:
            continue
        spec = specs[experiment_id]
        init_state, init_hash = init_by_depth[spec.depth][:2]
        profile["experiments"][experiment_id] = _profile_one(ctx, spec, init_state, init_hash)
    if blocked:
        measured_p90 = [
            float(value["p90_seconds"])
            for value in profile["experiments"].values()
            if value.get("p90_seconds") is not None
        ]
        if not measured_p90:
            raise RuntimeError("no available experiment remained for conservative blocked-data costing")
        conservative_p90 = max(measured_p90) * 1.25
        for experiment_id, reason in blocked.items():
            spec = specs[experiment_id]
            profile["experiments"][experiment_id] = {
                "experiment_id": experiment_id,
                "source_mode": spec.source_mode,
                "data_scale": spec.data_scale,
                "depth": spec.depth,
                "status": "BLOCKED_DATA",
                "reason": reason,
                "p90_seconds": conservative_p90,
                "cost_estimate": "1.25 * maximum measured p90 across available experiments",
                "test_payload_opened": False,
            }
    profile["blocked_experiments"] = {
        experiment_id: {"status": "BLOCKED_DATA", "reason": reason}
        for experiment_id, reason in blocked.items()
    }
    _write_json(ctx.output_dir / "runner_profile.json", profile)
    target_steps = int(ctx.cfg["schedule"]["target_steps"])
    target_tokens, mean_base_tokens = _estimate_tokens(ctx.data.train48, ctx.cfg, target_steps)
    token_estimates = {}
    for experiment_id, spec in specs.items():
        if experiment_id in blocked:
            token_estimates[experiment_id] = {
                "status": "BLOCKED_DATA",
                "reason": blocked[experiment_id],
                "target_tokens_reserved": target_tokens,
            }
        else:
            token_estimates[experiment_id] = _estimate_tokens(
                ctx.data.train_for_scale(spec.data_scale), ctx.cfg, target_steps
            )
    total_gpu_seconds = float(sum(
        profile["experiments"][experiment_id]["p90_seconds"] * target_steps
        for experiment_id in EXPERIMENT_IDS
    ))
    total_budget_seconds = float(ctx.cfg["budget"]["gpu_hours_total"]) * 3600.0
    total_wall_seconds = total_gpu_seconds
    total_wall_budget_seconds = float(ctx.cfg["budget"]["wall_hours_total"]) * 3600.0
    reserve = float(ctx.cfg["budget"]["reserve_fraction"])
    usable = total_budget_seconds * (1.0 - reserve)
    usable_wall = total_wall_budget_seconds * (1.0 - reserve)
    selected_steps = 0
    selected_tokens = 0
    candidate_records = []
    for candidate_steps in sorted((int(value) for value in ctx.cfg["budget"]["candidate_steps"]), reverse=True):
        candidate_tokens, _ = _estimate_tokens(ctx.data.train48, ctx.cfg, candidate_steps)
        candidate_cost = float(sum(
            profile["experiments"][experiment_id]["p90_seconds"] * candidate_steps
            for experiment_id in EXPERIMENT_IDS
        ))
        fits_gpu = candidate_cost <= usable
        fits_wall = candidate_cost <= usable_wall
        fits = fits_gpu and fits_wall
        candidate_records.append({
            "steps": candidate_steps,
            "target_tokens": candidate_tokens,
            "estimated_training_gpu_seconds": candidate_cost,
            "estimated_training_wall_seconds": candidate_cost,
            "fits_reserved_gpu_budget": fits_gpu,
            "fits_reserved_wall_budget": fits_wall,
            "fits_reserved_budget": fits,
        })
        if fits and selected_steps == 0:
            selected_steps = candidate_steps
            selected_tokens = candidate_tokens
    if selected_steps == 0:
        status = "BLOCKED_BUDGET"
    elif selected_steps < max(int(value) for value in ctx.cfg["budget"]["candidate_steps"]):
        status = "PARTIAL_BUDGET"
    else:
        status = "PASS"
    per_experiment_target_steps = {}
    for experiment_id, spec in specs.items():
        # C192 is stopped by the same effective token target, not by an
        # artificial equal-step claim.  Its actual step count is recorded by
        # the training loop after complete batches.
        per_experiment_target_steps[experiment_id] = selected_steps
    budget = {
        "schema": "pvb.dit.capacity_data.budget.v1",
        "status": status,
        "gpu_hours_total": float(ctx.cfg["budget"]["gpu_hours_total"]),
        "wall_hours_total": float(ctx.cfg["budget"]["wall_hours_total"]),
        "reserved_fraction": reserve,
        "reserved_gpu_seconds": total_budget_seconds * reserve,
        "usable_training_gpu_seconds": usable,
        "wall_hours_total": float(ctx.cfg["budget"]["wall_hours_total"]),
        "reserved_wall_seconds": total_wall_budget_seconds * reserve,
        "usable_training_wall_seconds": usable_wall,
        "selected_base_steps": selected_steps,
        "selected_target_tokens": selected_tokens,
        "per_experiment_target_step_cap": per_experiment_target_steps,
        "candidate_records": candidate_records,
        "profile_p90_seconds": {key: value["p90_seconds"] for key, value in profile["experiments"].items()},
        "token_estimates_at_cap": token_estimates,
        "mean_base_batch_tokens": mean_base_tokens,
        "estimated_training_gpu_seconds_at_config_target": total_gpu_seconds,
        "estimated_training_wall_seconds_at_config_target": total_wall_seconds,
        "budget_frozen_before_results": True,
        "evaluation_reserve_included": True,
        "blocked_experiments": {
            experiment_id: {"status": "BLOCKED_DATA", "reason": reason}
            for experiment_id, reason in blocked.items()
        },
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "budget.json", budget)
    if status == "BLOCKED_BUDGET":
        raise RuntimeError("no candidate training budget fits after the required evaluation reserve")
    contract = {
        "schema": "pvb.dit.capacity_data.experiment_contract.v1",
        "code_commit": _git_commit(),
        "data_hash": ctx.data_hash,
        "codec_state_hash": ctx.codec.codec_state_hash,
        "codec_checkpoint_sha256": ctx.codec.checkpoint_sha256,
        "statistics_hash": ctx.statistics.hash,
        "statistics_file_sha256": str(ctx.cfg["candidate"]["statistics_file_sha256"]),
        "source_contract": {
            "center_kind": str(ctx.cfg["source"]["center_kind"]),
            "sigma": float(ctx.cfg["source"]["sigma"]),
            "histories": list(HISTORY_SCHEDULE),
        },
        "model_contract": {
            "codec_width": int(ctx.cfg["model"]["codec_width"]),
            "scalar_width": int(ctx.cfg["model"]["scalar_width"]),
            "vector_width": int(ctx.cfg["model"]["vector_width"]),
            "heads": int(ctx.cfg["model"]["heads"]),
            "ffn_multiplier": int(ctx.cfg["model"]["ffn_multiplier"]),
            "dropout": float(ctx.cfg["model"]["dropout"]),
            "execution_backend": "factorized_v2",
        },
        "seeds": dict(ctx.cfg["seed"]),
        "experiments": {
            experiment_id: {
                "status": "BLOCKED_DATA" if experiment_id in blocked else "READY",
                "blocked_reason": blocked.get(experiment_id),
                "source_mode": spec.source_mode,
                "depth": spec.depth,
                "data_scale": spec.data_scale,
                "init_hash": init_by_depth[spec.depth][1],
                "parameter_count": init_by_depth[spec.depth][3],
            }
            for experiment_id, spec in specs.items()
        },
        "schedule": {
            "frames": 16,
            "histories": list(HISTORY_SCHEDULE),
            "max_atom_frame_tokens": int(ctx.cfg["schedule"]["max_atom_frame_tokens"]),
            "clips_per_trajectory": int(ctx.cfg["schedule"]["clips_per_trajectory"]),
            "checkpoint_interval": int(ctx.cfg["schedule"]["checkpoint_interval"]),
            "tau": "one uniform per sample, independent generator",
            "eps": "isotropic unit normal per valid coefficient, independent generator",
        },
        "budget": budget,
        "test_payload_opened": False,
    }
    contract["contract_hash"] = _canonical_hash(contract)
    _write_json(ctx.output_dir / "experiment_contract.json", contract)
    return {
        "status": status,
        "budget": budget,
        "contract_hash": contract["contract_hash"],
        "profile": profile,
        "blocked_experiments": profile["blocked_experiments"],
    }


def _load_contract(ctx: CapacityContext) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    budget_path = ctx.output_dir / "budget.json"
    contract_path = ctx.output_dir / "experiment_contract.json"
    init_path = ctx.output_dir / "shared_initialization.pt"
    for path in (budget_path, contract_path, init_path):
        if not path.is_file():
            raise FileNotFoundError(f"capacity prepare artifact is missing: {path}")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    if budget.get("status") not in ("PASS", "PARTIAL_BUDGET") or not int(budget.get("selected_base_steps", 0)):
        raise RuntimeError("capacity budget is not usable")
    if contract.get("data_hash") != ctx.data_hash or contract.get("codec_state_hash") != ctx.codec.codec_state_hash:
        raise RuntimeError("capacity contract frozen input hash mismatch")
    if init.get("schema") != "pvb.dit.capacity_data.shared_initialization.v1":
        raise RuntimeError("capacity shared initialization schema mismatch")
    return budget, contract, init


def _checkpoint_payload(
    ctx: CapacityContext,
    trainer: DiTTrainer,
    spec: ExperimentSpec,
    init_hash: str,
    schedule_hash: str,
    generator: torch.Generator,
    cursor: Mapping[str, int],
    tokens_seen: int,
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    payload["capacity"] = {
        "schema": CAPACITY_SCHEMA + ".checkpoint.v1",
        "experiment_id": spec.experiment_id,
        "source_mode": spec.source_mode,
        "data_scale": spec.data_scale,
        "depth": int(spec.depth),
        "init_hash": init_hash,
        "schedule_hash": schedule_hash,
        "cursor": {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
        "generator_state": generator.get_state(),
        "tokens_seen": int(tokens_seen),
        "successful_optimizer_updates": int(trainer.successful_updates),
        "data_hash": ctx.data_hash,
        "test_payload_opened": False,
    }
    return payload


def _save_checkpoint(
    ctx: CapacityContext,
    trainer: DiTTrainer,
    spec: ExperimentSpec,
    init_hash: str,
    schedule_hash: str,
    generator: torch.Generator,
    cursor: Mapping[str, int],
    tokens_seen: int,
    *,
    label: str,
) -> Path:
    payload = _checkpoint_payload(ctx, trainer, spec, init_hash, schedule_hash, generator, cursor, tokens_seen)
    directory = ctx.output_dir / spec.experiment_id / "checkpoints"
    path = directory / (f"checkpoint_step{trainer.step:06d}.pt" if label == "step" else f"checkpoint_{label}.pt")
    _atomic_torch_save(path, payload)
    _atomic_torch_save(ctx.output_dir / spec.experiment_id / "latest.pt", payload)
    return path


def _restore_checkpoint(
    ctx: CapacityContext,
    trainer: DiTTrainer,
    spec: ExperimentSpec,
    path: Path,
    init_hash: str,
    expected_data_scale: str,
) -> tuple[dict[str, int], torch.Tensor, int, str]:
    payload = trainer.load_checkpoint(path, map_location=ctx.device)
    capacity = payload.get("capacity", {})
    if capacity.get("schema") != CAPACITY_SCHEMA + ".checkpoint.v1":
        raise RuntimeError(f"{spec.experiment_id} checkpoint lacks capacity contract")
    for key, expected in (
        ("experiment_id", spec.experiment_id),
        ("source_mode", spec.source_mode),
        ("data_scale", expected_data_scale),
        ("depth", int(spec.depth)),
        ("init_hash", init_hash),
        ("data_hash", ctx.data_hash),
    ):
        if capacity.get(key) != expected:
            raise RuntimeError(f"{spec.experiment_id} resume contract mismatch at {key}")
    cursor = capacity.get("cursor")
    if not isinstance(cursor, Mapping) or int(cursor.get("epoch", -1)) < 0 or int(cursor.get("batch_index", -1)) < 0:
        raise RuntimeError(f"{spec.experiment_id} checkpoint cursor is invalid")
    if int(capacity.get("successful_optimizer_updates", -1)) != trainer.successful_updates:
        raise RuntimeError(f"{spec.experiment_id} successful update count mismatch")
    return (
        {"epoch": int(cursor["epoch"]), "batch_index": int(cursor["batch_index"])},
        capacity["generator_state"].detach().to(device="cpu"),
        int(capacity.get("tokens_seen", 0)),
        str(capacity.get("schedule_hash", "")),
    )


def _validation_rf(
    ctx: CapacityContext,
    trainer: DiTTrainer,
    plan: Any,
    spec: ExperimentSpec,
    step: int,
) -> dict[str, Any]:
    trainer.model.eval()
    numerators = {name: 0.0 for name in ("state_h", "detail_h", "state_v", "detail_v")}
    counts = {name: 0 for name in numerators}
    cache = LatentFieldCache(mode="ram", max_bytes=512 * 1024**2)
    for history in HISTORY_SCHEDULE:
        for local_indices in plan.batches:
            _batch_cpu, batch, target_batch = _prepare_encoded(ctx, plan.subset, local_indices)
            observed = _observed(ctx, target_batch, batch, history)
            center = None
            if spec.source_mode == "conditional":
                center, _ = _center(ctx, batch, observed, history, cache)
            seed = stable_seed(
                int(ctx.cfg["seed"]["validation"]),
                "|".join(batch.sample_id),
                history,
                0,
                "rf_validation",
            )
            generator = torch.Generator(device=ctx.device).manual_seed(seed)
            normalized = trainer._normalise_batch(observed)
            with torch.no_grad(), trainer.autocast_context():
                sample = trainer.flow.sample(
                    normalized,
                    generator=generator,
                    source_center=center,
                    source_mode=spec.source_mode,
                )
                prediction = trainer.model(normalized.with_fields(sample.interpolated), sample.tau)
                loss = trainer.flow.loss(prediction, sample.target, normalized)
            for name, value in loss.fields.items():
                amount = int(loss.valid_elements[name])
                counts[name] += amount
                numerators[name] += float(value.detach().float().cpu()) * amount
    fields = {name: numerators[name] / max(counts[name], 1) for name in numerators}
    return {
        "step": int(step),
        "experiment_id": spec.experiment_id,
        "source_mode": spec.source_mode,
        "history": list(HISTORY_SCHEDULE),
        "total": sum(fields.values()) / 4.0,
        "fields": fields,
        "valid_elements": counts,
        "sample_count": len(plan.selected_sample_ids),
        "seed_contract": "stable_seed(validation_seed,sample_ids,H,draw0,rf_validation); no step/arm/batch position",
    }


def _block_displacements(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    if history <= 0 or history >= prediction.shape[0]:
        return {"available": False, "reason": "history must leave an observed boundary"}
    block_id = batch.block_id.to(device=prediction.device)
    valid = batch.loss_mask.to(device=prediction.device, dtype=torch.bool)
    future = torch.arange(prediction.shape[0], device=prediction.device) >= int(history)
    pred_within: list[Tensor] = []
    target_within: list[Tensor] = []
    pred_inter: list[Tensor] = []
    target_inter: list[Tensor] = []
    unique_blocks = torch.unique(block_id, sorted=True)
    for frame in torch.nonzero(future, as_tuple=False).flatten().tolist():
        pred_disp = prediction[frame] - prediction[history - 1]
        target_disp = target[frame] - target[history - 1]
        pred_centers = []
        target_centers = []
        for block in unique_blocks.tolist():
            nodes = torch.nonzero((block_id == int(block)) & valid, as_tuple=False).flatten()
            if nodes.numel() == 0:
                continue
            pred_mean = pred_disp[nodes].mean(dim=0)
            target_mean = target_disp[nodes].mean(dim=0)
            pred_centers.append(pred_mean)
            target_centers.append(target_mean)
            pred_within.append((pred_disp[nodes] - pred_mean).square().sum(dim=-1).mean().sqrt())
            target_within.append((target_disp[nodes] - target_mean).square().sum(dim=-1).mean().sqrt())
        if len(pred_centers) > 1:
            pred_stack = torch.stack(pred_centers)
            target_stack = torch.stack(target_centers)
            pred_center = pred_stack.mean(dim=0)
            target_center = target_stack.mean(dim=0)
            pred_inter.append((pred_stack - pred_center).square().sum(dim=-1).mean().sqrt())
            target_inter.append((target_stack - target_center).square().sum(dim=-1).mean().sqrt())
    return {
        "available": bool(pred_within),
        "prediction_within_block_rms": None if not pred_within else float(torch.stack(pred_within).mean().cpu()),
        "target_within_block_rms": None if not target_within else float(torch.stack(target_within).mean().cpu()),
        "prediction_between_block_centroid_rms": None if not pred_inter else float(torch.stack(pred_inter).mean().cpu()),
        "target_between_block_centroid_rms": None if not target_inter else float(torch.stack(target_inter).mean().cpu()),
        "reference": "displacement from last observed frame, future-only; block IDs from frozen topology",
    }


def _true_occupancy_mae(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    atoms = int(prediction.shape[1])
    if atoms > 2048:
        return {"value": None, "reason": "not_computed_for_atoms_over_2048"}
    pairs = _nonbond_pairs(batch, atoms, prediction.device)
    if pairs[0].numel() == 0:
        return {"value": None, "reason": "no_nonbond_pairs"}
    values = []
    for source, destination in zip(pairs[0].tolist(), pairs[1].tolist()):
        valid = batch.loss_mask.to(device=prediction.device, dtype=torch.bool)[source] & batch.loss_mask.to(device=prediction.device, dtype=torch.bool)[destination]
        if not bool(valid):
            continue
        pred_distance = torch.linalg.vector_norm(prediction[history:, source] - prediction[history:, destination], dim=-1)
        target_distance = torch.linalg.vector_norm(target[history:, source] - target[history:, destination], dim=-1)
        values.append((pred_distance < 4.5).float().mean() - (target_distance < 4.5).float().mean())
    if not values:
        return {"value": None, "reason": "no_valid_future_nonbond_pairs"}
    return {"value": float(torch.stack(values).abs().mean().cpu()), "pair_count": len(values), "cutoff_angstrom": 4.5}


def _corrected_dynamics(prediction: Tensor, target: Tensor, batch: Any, history: int) -> dict[str, Any]:
    future = tuple(range(history, int(prediction.shape[0])))
    rmsf = aligned_rmsf_metrics(prediction, target, batch, frames=future)
    dynamic = dynamic_acf_metrics(prediction, target, batch, frames=future)
    # The legacy evaluator's zero-variance convention is not an applicable
    # correlation.  Keep RMSF=0 valid, but report ACF/Pearson as null.
    if dynamic.get("dynamic_correlation") in (0.0, 1.0) and (
        rmsf.get("prediction", 0.0) <= 1.0e-5 or rmsf.get("target", 0.0) <= 1.0e-5
    ):
        dynamic = {
            "prediction": None,
            "target": None,
            "dynamic_correlation": None,
            "reason": "near_zero_future_velocity_variance",
            "lag": 1,
            "units": "angstrom_per_ps",
        }
    return {"rmsf": rmsf, "velocity_lag1_acf": dynamic}


def _sanitize_degenerate_correlations(metrics: dict[str, Any]) -> None:
    """Turn legacy zero-variance correlation sentinels into JSON nulls."""

    def sanitize_temporal(temporal: Any) -> None:
        if not isinstance(temporal, dict):
            return
        rmsf = temporal.get("rmsf")
        if isinstance(rmsf, dict):
            pred = rmsf.get("prediction")
            target = rmsf.get("target")
            if (
                isinstance(pred, (int, float))
                and isinstance(target, (int, float))
                and (float(pred) <= 1.0e-5 or float(target) <= 1.0e-5)
            ):
                rmsf["correlation"] = None
                rmsf["correlation_reason"] = "zero_variance_rmsf_series"
        dynamic = temporal.get("dynamic")
        if isinstance(dynamic, dict):
            pred = rmsf.get("prediction") if isinstance(rmsf, dict) else None
            target = rmsf.get("target") if isinstance(rmsf, dict) else None
            if (
                isinstance(pred, (int, float))
                and isinstance(target, (int, float))
                and (float(pred) <= 1.0e-5 or float(target) <= 1.0e-5)
            ):
                for key in ("prediction", "target", "dynamic_correlation", "dynamic_correlation_absolute_error"):
                    if key in dynamic:
                        dynamic[key] = None
                dynamic["reason"] = "zero_variance_future_velocity_series"

    sanitize_temporal(metrics.get("temporal", {}).get("future"))
    for horizon in metrics.get("horizons", {}).values():
        if isinstance(horizon, dict):
            sanitize_temporal(horizon.get("temporal"))


def _save_prediction(path: Path, coordinates: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, coordinates=coordinates.detach().cpu().to(torch.float16).numpy())
    os.replace(temporary, path)


def _clip_prediction_row(
    ctx: CapacityContext,
    trainer: DiTTrainer,
    spec: ExperimentSpec,
    dataset: Any,
    index: int,
    sample_id: str,
    history: int,
    steps: int,
    draw_id: int,
    center_cache: LatentFieldCache,
    prediction_root: Path,
    checkpoint_step: int,
) -> dict[str, Any]:
    _sync(ctx.device)
    torch.cuda.empty_cache()
    _batch_cpu, batch, target_batch = _prepare_encoded(ctx, dataset, (index,))
    observed = _observed(ctx, target_batch, batch, history)
    center = None
    center_metadata = None
    if spec.source_mode == "conditional":
        center, center_metadata = _center(ctx, batch, observed, history, center_cache)
    seed = stable_seed(
        int(ctx.cfg["seed"]["validation"]),
        sample_id,
        history,
        draw_id,
        f"generation_{steps}",
    )
    generated, generation = generate_state_detail_latent(
        trainer.model,
        trainer.adapter,
        observed,
        ctx.statistics_device,
        steps=int(steps),
        seed=seed,
        source_center=center,
        source_mode=spec.source_mode,
    )
    decoded = ctx.codec.model.decode(generated).x_hat.float()
    _sync(ctx.device)
    torch.cuda.empty_cache()
    metrics = trajectory_metric_record(decoded, batch.x.float(), batch, history)
    _sanitize_degenerate_correlations(metrics)
    corrected = _corrected_dynamics(decoded, batch.x.float(), batch, history)
    future = metrics.get("future", {})
    output_name = f"{spec.experiment_id}__{sample_id.replace('/', '_')}__H{history}__L{steps}__draw{draw_id}.npz"
    output_path = prediction_root / output_name
    _save_prediction(output_path, decoded)
    return {
        "schema": "pvb.dit.capacity_data.generation_row.v1",
        "experiment_id": spec.experiment_id,
        "source_mode": spec.source_mode,
        "data_scale": spec.data_scale,
        "checkpoint_step": int(checkpoint_step),
        "sample_id": sample_id,
        "system": parse_sample_id(sample_id)["system"],
        "replica": parse_sample_id(sample_id)["replica"],
        "window": parse_sample_id(sample_id)["window"],
        "history_frames": int(history),
        "steps": int(steps),
        "draw_id": int(draw_id),
        "seed": int(seed),
        "generation": generation,
        "metrics": metrics,
        "corrected_metrics": corrected,
        "true_future_contact_occupancy_mae": _true_occupancy_mae(decoded, batch.x.float(), batch, history),
        "block_displacements": _block_displacements(decoded, batch.x.float(), batch, history),
        "prediction_coordinates": str(output_path),
        "center_cache": None if center_metadata is None else {key: value for key, value in center_metadata.items() if key not in ("template_coordinates", "template_latent")},
        "test_payload_opened": False,
    }


def _monitor(
    ctx: CapacityContext,
    trainer: DiTTrainer,
    spec: ExperimentSpec,
    step: int,
    monitor_indices: Sequence[tuple[str, Any, int]],
) -> dict[str, Any]:
    rows = []
    root = ctx.output_dir / spec.experiment_id / "predictions" / "monitor" / f"step{step:06d}"
    cache = LatentFieldCache(mode="ram", max_bytes=512 * 1024**2)
    was_training = trainer.model.training
    trainer.model.eval()
    with torch.no_grad():
        for sample_id, dataset, index in monitor_indices:
            for history in HISTORY_SCHEDULE:
                _sync(ctx.device)
                torch.cuda.empty_cache()
                rows.append(_clip_prediction_row(
                    ctx, trainer, spec, dataset, index, sample_id, history, 16, 0, cache, root, step
                ))
    _sync(ctx.device)
    torch.cuda.empty_cache()
    if was_training:
        trainer.model.train()
    path = ctx.output_dir / spec.experiment_id / "monitor" / f"step{step:06d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {
        "step": int(step),
        "row_count": len(rows),
        "rows_path": str(path),
        "prediction_root": str(root),
        "actual_generation": True,
        "test_payload_opened": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break
    return rows


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _truncate_history_to_checkpoint(path: Path, checkpoint_step: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    retained = [row for row in rows if int(row.get("step", -1)) <= int(checkpoint_step)]
    _write_jsonl_atomic(path, retained)
    return retained


def _train_one(
    ctx: CapacityContext,
    spec: ExperimentSpec,
    init_payload: Mapping[str, Any],
    budget: Mapping[str, Any],
    plan: Any,
    monitor_indices: Sequence[tuple[str, Any, int]],
    resume: bool,
) -> dict[str, Any]:
    run_dir = ctx.output_dir / spec.experiment_id
    train_history_path = run_dir / "train_history.jsonl"
    validation_history_path = run_dir / "validation_history.jsonl"
    monitor_history_path = run_dir / "monitor_history.json"
    latest_checkpoint = run_dir / "latest.pt"
    if not resume:
        existing = [
            path
            for path in (
                train_history_path,
                validation_history_path,
                monitor_history_path,
                latest_checkpoint,
                run_dir / "train_summary.json",
            )
            if path.exists()
        ]
        if existing:
            raise RuntimeError(
                f"fresh {spec.experiment_id} run refuses existing artifacts: "
                + ", ".join(str(path) for path in existing)
            )
    depth_payload = init_payload["by_depth"][str(spec.depth)]
    init_state = depth_payload["state_dict"]
    init_hash = str(depth_payload["init_hash"])
    target_step_cap = int(budget["selected_base_steps"])
    trainer = _make_trainer(ctx, spec, max(target_step_cap, 1), init_state, init_hash)
    dataset = ctx.data.train_for_scale(spec.data_scale)
    sampler = _dataset_sampler(dataset, ctx.cfg)
    cursor = {"epoch": 0, "batch_index": 0}
    plan_cache: dict[str, Any] = {}
    specs = dataset.clip_spec_table()
    generator = torch.Generator(device=ctx.device).manual_seed(int(ctx.cfg["seed"]["training"]))
    tokens_seen = 0
    schedule_hash = ""
    if resume:
        if not latest_checkpoint.is_file():
            raise FileNotFoundError(
                f"resume checkpoint missing for {spec.experiment_id}: {latest_checkpoint}"
            )
        cursor, generator_state, tokens_seen, schedule_hash = _restore_checkpoint(
            ctx, trainer, spec, latest_checkpoint, init_hash, spec.data_scale
        )
        generator.set_state(generator_state)
    target_tokens = int(budget["selected_target_tokens"])
    started = time.perf_counter()
    if resume:
        train_rows = _truncate_history_to_checkpoint(train_history_path, trainer.step)
        validation_rows = _truncate_history_to_checkpoint(validation_history_path, trainer.step)
        if monitor_history_path.is_file():
            loaded_monitor_rows = json.loads(monitor_history_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_monitor_rows, list):
                raise RuntimeError(f"invalid monitor history for {spec.experiment_id}")
            monitor_rows = [
                row
                for row in loaded_monitor_rows
                if int(row.get("step", -1)) <= trainer.step
            ]
            _write_json(monitor_history_path, monitor_rows)
        else:
            monitor_rows = []
    else:
        train_rows = []
        validation_rows = []
        monitor_rows = []
    center_cache = LatentFieldCache(
        mode=str(ctx.cfg["cache"]["mode"]),
        max_bytes=int(float(ctx.cfg["cache"]["ram_gib"]) * 1024**3),
    )
    while trainer.successful_updates < target_step_cap and tokens_seen < target_tokens:
        update_index = trainer.successful_updates
        indices, schedule_hash, epoch = _next_batch(sampler, cursor, plan_cache)
        history = HISTORY_SCHEDULE[update_index % len(HISTORY_SCHEDULE)]
        _batch_cpu, batch, target_batch = _prepare_encoded(ctx, dataset, indices)
        observed = _observed(ctx, target_batch, batch, history)
        center = None
        if spec.source_mode == "conditional":
            center, _ = _center(ctx, batch, observed, history, center_cache)
        row = trainer.train_step(observed, generator=generator, source_center=center)
        batch_tokens = _batch_tokens(specs, indices)
        tokens_seen += batch_tokens
        row.update({
            "schema": "pvb.dit.capacity_data.train_row.v1",
            "experiment_id": spec.experiment_id,
            "source_mode": spec.source_mode,
            "data_scale": spec.data_scale,
            "history_frames": history,
            "batch_tokens": batch_tokens,
            "tokens_seen": tokens_seen,
            "successful_optimizer_updates": trainer.successful_updates,
            "epoch": epoch,
            "batch_index": int(cursor["batch_index"]),
        })
        train_rows.append(row)
        _append_jsonl(train_history_path, row)
        step = trainer.successful_updates
        if (
            step % int(ctx.cfg["schedule"]["checkpoint_interval"]) == 0
            or step in CHECKPOINT_STEPS
        ):
            _save_checkpoint(
                ctx,
                trainer,
                spec,
                init_hash,
                schedule_hash,
                generator,
                cursor,
                tokens_seen,
                label="step",
            )
        if step % int(ctx.cfg["schedule"]["validation_interval"]) == 0 or step in CHECKPOINT_STEPS:
            validation = _validation_rf(ctx, trainer, plan, spec, step)
            validation["tokens_seen"] = tokens_seen
            validation_rows.append(validation)
            _append_jsonl(validation_history_path, validation)
        if step % int(ctx.cfg["schedule"]["generation_interval"]) == 0:
            monitor_rows.append(_monitor(ctx, trainer, spec, step, monitor_indices))
            _write_json(monitor_history_path, monitor_rows)
    final_path = _save_checkpoint(ctx, trainer, spec, init_hash, schedule_hash, generator, cursor, tokens_seen, label="final")
    elapsed = time.perf_counter() - started
    summary = {
        "schema": "pvb.dit.capacity_data.train_summary.v1",
        "status": "PASS" if tokens_seen >= target_tokens else "PARTIAL_TOKEN_BUDGET",
        "experiment_id": spec.experiment_id,
        "source_mode": spec.source_mode,
        "data_scale": spec.data_scale,
        "depth": spec.depth,
        "target_token_budget": target_tokens,
        "tokens_seen": tokens_seen,
        "actual_optimizer_updates": trainer.successful_updates,
        "target_step_cap": target_step_cap,
        "last_batch_token_overshoot": max(tokens_seen - target_tokens, 0),
        "elapsed_wall_seconds": elapsed,
        "estimated_gpu_hours": elapsed / 3600.0,
        "initialization_hash": init_hash,
        "schedule_hash": schedule_hash,
        "final_checkpoint": str(final_path),
        "cache": center_cache.stats().as_dict(),
        "validation_rows": len(validation_rows),
        "monitor_rows": len(monitor_rows),
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / spec.experiment_id / "train_summary.json", summary)
    return summary


def _monitor_indices(ctx: CapacityContext) -> list[tuple[str, Any, int]]:
    result = []
    train_candidates = []
    for index, row in enumerate(ctx.data.train48._index):
        parsed = parse_sample_id(str(row[0]))
        if parsed["replica"] in ("R1", "R2") and parsed["window"] in (0, 30, 61):
            train_candidates.append((str(row[0]), ctx.data.train48, index))
    train_candidates.sort(key=lambda item: item[0])
    valid_candidates = []
    for index, row in enumerate(ctx.data.valid._index):
        parsed = parse_sample_id(str(row[0]))
        if parsed["replica"] == "R1" and parsed["window"] in (0, 30, 61):
            valid_candidates.append((str(row[0]), ctx.data.valid, index))
    valid_candidates.sort(key=lambda item: item[0])
    result.extend(train_candidates[: int(ctx.cfg["evaluation"]["monitor_train_clips"])])
    result.extend(valid_candidates[: int(ctx.cfg["evaluation"]["monitor_valid_clips"])])
    if len(result) != 8:
        raise RuntimeError("monitor plan must contain four train and four validation clips")
    return result


def _train_evaluation_indices(ctx: CapacityContext) -> list[tuple[str, Any, int]]:
    """Choose eight deterministic train clips for the four-draw audit."""

    candidates: dict[str, tuple[str, Any, int]] = {}
    for index, row in enumerate(ctx.data.train48._index):
        sample_id = str(row[0])
        parsed = parse_sample_id(sample_id)
        if parsed["replica"] == "R1" and parsed["window"] == 30:
            candidates.setdefault(parsed["system"], (sample_id, ctx.data.train48, index))
    selected = [candidates[system] for system in sorted(candidates)]
    expected = int(ctx.cfg["evaluation"].get("train_subset_clips", 8))
    if len(selected) < expected:
        raise RuntimeError(f"train evaluation plan has only {len(selected)} systems; need {expected}")
    return selected[:expected]


def _train(ctx: CapacityContext, specs: Mapping[str, ExperimentSpec], resume: bool, selected: Sequence[str]) -> dict[str, Any]:
    budget, contract, init_payload = _load_contract(ctx)
    if not (ctx.output_dir / "source_decision.json").is_file():
        raise RuntimeError("source decision is missing")
    source_decision = json.loads((ctx.output_dir / "source_decision.json").read_text(encoding="utf-8"))
    if source_decision.get("status") != "PASS":
        raise RuntimeError("source decision is not PASS")
    plan = _make_validation_plan(ctx.data.valid)
    monitors = _monitor_indices(ctx)
    selected_specs = [specs[key] for key in selected]
    blocked = _blocked_experiments(ctx, specs)
    results = {}
    for spec in selected_specs:
        if spec.experiment_id in blocked:
            results[spec.experiment_id] = {
                "status": "BLOCKED_DATA",
                "experiment_id": spec.experiment_id,
                "data_scale": spec.data_scale,
                "reason": blocked[spec.experiment_id],
                "test_payload_opened": False,
            }
            _write_json(
                ctx.output_dir / spec.experiment_id / "train_summary.json",
                results[spec.experiment_id],
            )
        else:
            results[spec.experiment_id] = _train_one(
                ctx, spec, init_payload, budget, plan, monitors, resume
            )
    statuses = {value["status"] for value in results.values()}
    if statuses == {"BLOCKED_DATA"}:
        status = "BLOCKED_DATA"
    elif statuses.issubset({"PASS", "BLOCKED_DATA"}):
        status = "PASS_WITH_BLOCKED_DATA" if "BLOCKED_DATA" in statuses else "PASS"
    else:
        status = "PARTIAL_TOKEN_BUDGET"
    return {
        "schema": "pvb.dit.capacity_data.training.v1",
        "status": status,
        "experiments": results,
        "selected": list(selected),
        "contract_hash": contract["contract_hash"],
        "test_payload_opened": False,
    }


def _evaluate_one(ctx: CapacityContext, spec: ExperimentSpec, init_payload: Mapping[str, Any], train_summary: Mapping[str, Any]) -> dict[str, Any]:
    init_payload_depth = init_payload["by_depth"][str(spec.depth)]
    trainer = _make_trainer(ctx, spec, max(int(train_summary["actual_optimizer_updates"]), 1), init_payload_depth["state_dict"], str(init_payload_depth["init_hash"]))
    checkpoint = Path(str(train_summary["final_checkpoint"]))
    trainer.load_checkpoint(checkpoint, map_location=ctx.device)
    trainer.model.eval()
    plan = _make_validation_plan(ctx.data.valid)
    rows = []
    prediction_root = ctx.output_dir / spec.experiment_id / "predictions" / "final"
    cache = LatentFieldCache(mode="ram", max_bytes=512 * 1024**2)
    for index, sample_id in zip(plan.selected_dataset_indices, plan.selected_sample_ids):
        for history in HISTORY_SCHEDULE:
            rows.append(_clip_prediction_row(
                ctx, trainer, spec, ctx.data.valid, int(index), str(sample_id), history,
                int(ctx.cfg["evaluation"]["final_steps"]), 0, cache, prediction_root,
                int(train_summary["actual_optimizer_updates"]),
            ))
    subset_rows = []
    for index, row in enumerate(ctx.data.valid._index):
        parsed = parse_sample_id(str(row[0]))
        if parsed["replica"] == "R1" and parsed["window"] == 30:
            sample_id = str(row[0])
            for history in HISTORY_SCHEDULE:
                for steps in (int(ctx.cfg["evaluation"]["short_steps"]), int(ctx.cfg["evaluation"]["final_steps"])):
                    for draw_id in (int(value) for value in ctx.cfg["evaluation"]["subset_draws"]):
                        subset_rows.append(_clip_prediction_row(
                            ctx, trainer, spec, ctx.data.valid, index, sample_id, history, steps, draw_id,
                            cache, ctx.output_dir / spec.experiment_id / "predictions" / "subset",
                            int(train_summary["actual_optimizer_updates"]),
                        ))
    train_subset_rows = []
    for sample_id, dataset, index in _train_evaluation_indices(ctx):
        for history in HISTORY_SCHEDULE:
            for steps in (int(ctx.cfg["evaluation"]["short_steps"]), int(ctx.cfg["evaluation"]["final_steps"])):
                for draw_id in (int(value) for value in ctx.cfg["evaluation"]["subset_draws"]):
                    train_subset_rows.append(_clip_prediction_row(
                        ctx, trainer, spec, dataset, index, sample_id, history, steps, draw_id,
                        cache, ctx.output_dir / spec.experiment_id / "predictions" / "train_subset",
                        int(train_summary["actual_optimizer_updates"]),
                    ))
    rows_path = ctx.output_dir / spec.experiment_id / "generation_metrics.jsonl"
    rows_path.write_text(
        "".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in rows + subset_rows + train_subset_rows),
        encoding="utf-8",
    )
    return {
        "schema": "pvb.dit.capacity_data.evaluation.v1",
        "experiment_id": spec.experiment_id,
        "source_mode": spec.source_mode,
        "data_scale": spec.data_scale,
        "depth": spec.depth,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(train_summary["actual_optimizer_updates"]),
        "final_rows": len(rows),
        "subset_rows": len(subset_rows),
        "train_subset_rows": len(train_subset_rows),
        "final_aggregate": _aggregate_generation_rows(rows),
        "subset_aggregate": _aggregate_generation_rows(subset_rows),
        "train_subset_aggregate": _aggregate_generation_rows(train_subset_rows),
        "rows_path": str(rows_path),
        "test_payload_opened": False,
    }


def _numeric_future(row: Mapping[str, Any], key: str) -> float | None:
    value: Any = row.get("metrics", {}).get("future", {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _aggregate_generation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["history_frames"]), int(row["steps"])), []).append(row)
    output: dict[str, Any] = {}
    for (history, steps), group in sorted(groups.items()):
        by_sample: dict[str, list[Mapping[str, Any]]] = {}
        for row in group:
            by_sample.setdefault(str(row["sample_id"]), []).append(row)
        sample_rows = []
        for sample_id, sample_group in by_sample.items():
            values = {}
            for key in ("aligned_rmsd", "drmsd", "bond_rmse", "contact_f1", "contact_occupancy_mae"):
                numbers = [_numeric_future(row, key) for row in sample_group]
                numbers = [value for value in numbers if value is not None]
                if numbers:
                    values[key] = sum(numbers) / len(numbers)
            occ = [row.get("true_future_contact_occupancy_mae", {}).get("value") for row in sample_group]
            occ = [float(value) for value in occ if isinstance(value, (int, float)) and math.isfinite(float(value))]
            if occ:
                values["true_future_contact_occupancy_mae"] = sum(occ) / len(occ)
            rmsf = [row.get("corrected_metrics", {}).get("rmsf", {}) for row in sample_group]
            rmsf_pred = [item.get("prediction") for item in rmsf if isinstance(item.get("prediction"), (int, float))]
            rmsf_target = [item.get("target") for item in rmsf if isinstance(item.get("target"), (int, float))]
            if rmsf_pred:
                values["rmsf_prediction"] = sum(rmsf_pred) / len(rmsf_pred)
            if rmsf_target:
                values["rmsf_target"] = sum(rmsf_target) / len(rmsf_target)
            if values.get("rmsf_target") not in (None, 0.0) and values.get("rmsf_prediction") is not None:
                values["rmsf_ratio"] = values["rmsf_prediction"] / values["rmsf_target"]
            acf = [row.get("corrected_metrics", {}).get("velocity_lag1_acf", {}).get("prediction") for row in sample_group]
            acf = [float(value) for value in acf if isinstance(value, (int, float)) and math.isfinite(float(value))]
            if acf:
                values["velocity_lag1_acf_prediction"] = sum(acf) / len(acf)
            values["sample_id"] = sample_id
            values["system"] = str(sample_group[0]["system"])
            sample_rows.append(values)
        by_system: dict[str, list[Mapping[str, Any]]] = {}
        for row in sample_rows:
            by_system.setdefault(str(row["system"]), []).append(row)
        keys = sorted({key for row in sample_rows for key in row if key not in ("sample_id", "system")})
        system_equal = {}
        for key in keys:
            system_values = []
            for system_rows in by_system.values():
                values = [row[key] for row in system_rows if isinstance(row.get(key), (int, float))]
                if values:
                    system_values.append(sum(values) / len(values))
            if system_values:
                system_equal[key] = sum(system_values) / len(system_values)
        system_rows = {}
        for system, system_samples in sorted(by_system.items()):
            values = {"sample_count": len(system_samples)}
            for key in keys:
                numbers = [row[key] for row in system_samples if isinstance(row.get(key), (int, float))]
                if numbers:
                    values[key] = sum(numbers) / len(numbers)
            system_rows[system] = values
        output[f"H{history}_L{steps}"] = {
            "row_count": len(group),
            "sample_count": len(sample_rows),
            "system_count": len(by_system),
            "system_equal": system_equal,
            "system_rows": system_rows,
            "aggregation": "draw_to_clip_sample_to_system; no best_of_n",
        }
    return output


def _evaluate(ctx: CapacityContext, specs: Mapping[str, ExperimentSpec]) -> dict[str, Any]:
    _budget, contract, init_payload = _load_contract(ctx)
    evaluations = {}
    blocked = _blocked_experiments(ctx, specs)
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id in blocked:
            evaluations[experiment_id] = {
                "schema": "pvb.dit.capacity_data.evaluation.v1",
                "status": "BLOCKED_DATA",
                "experiment_id": experiment_id,
                "source_mode": specs[experiment_id].source_mode,
                "data_scale": specs[experiment_id].data_scale,
                "depth": specs[experiment_id].depth,
                "reason": blocked[experiment_id],
                "test_payload_opened": False,
            }
            continue
        summary_path = ctx.output_dir / experiment_id / "train_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"training summary missing for {experiment_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evaluations[experiment_id] = _evaluate_one(ctx, specs[experiment_id], init_payload, summary)
    comparison = {
        "schema": "pvb.dit.capacity_data.comparison.v1",
        "status": "PARTIAL_BLOCKED_DATA" if blocked else "PASS",
        "contract_hash": contract["contract_hash"],
        "experiments": evaluations,
        "main_protocol": {
            "validation_clips": 72,
            "histories": list(HISTORY_SCHEDULE),
            "steps": 16,
            "draws": [0],
            "subset_clips": 8,
            "subset_draws": 4,
            "train_subset_clips": 8,
            "train_subset_draws": 4,
            "test_payload_opened": False,
        },
        "selection": "compare equal effective token budget and fixed final protocol; no cross-source RF-loss ranking",
    }
    _write_json(ctx.output_dir / "comparison.json", comparison)
    _write_report(ctx, comparison)
    return comparison


def _write_report(ctx: CapacityContext, comparison: Mapping[str, Any]) -> None:
    lines = [
        "# DiT capacity/data comparison — Session B",
        "",
        "This report uses frozen R4 state/detail codec/statistics, factorized_v2, 16-frame dt=100 ps clips, and real per-clip generation. Test coordinates were not opened.",
        "",
        "## Main final comparison",
        "",
        "| ID | source | depth | data | H | aligned RMSD | dRMSD | bond RMSE | contact F1 | RMSF ratio |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment_id in EXPERIMENT_IDS:
        item = comparison["experiments"][experiment_id]
        if item.get("status") == "BLOCKED_DATA":
            for history in HISTORY_SCHEDULE:
                lines.append(
                    f"| {experiment_id} | {item['source_mode']} | {item['depth']} | "
                    f"{item['data_scale']} | {history} | BLOCKED_DATA | n/a | n/a | n/a | n/a |"
                )
            continue
        for history in HISTORY_SCHEDULE:
            aggregate = item["final_aggregate"].get(f"H{history}_L16", {}).get("system_equal", {})
            lines.append(
                f"| {experiment_id} | {item['source_mode']} | {item['depth']} | {item['data_scale']} | {history} | "
                f"{aggregate.get('aligned_rmsd', 'n/a')} | {aggregate.get('drmsd', 'n/a')} | "
                f"{aggregate.get('bond_rmse', 'n/a')} | {aggregate.get('contact_f1', 'n/a')} | "
                f"{aggregate.get('rmsf_ratio', 'n/a')} |"
            )
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- G48 vs C48 isolates source mode at the original 48-system capacity.",
        "- C48 vs C192 isolates nested training-system diversity at depth 4; the data manifest proves system disjointness, but no sequence/homology audit was available, so no family-generalization claim is made.",
        "- C48 vs C48D8 isolates DiT depth at the original 48-system data scale.",
        "- RF validation is reported within source/configuration and is not used to rank Gaussian against Conditional.",
        "- ACF/Pearson-like values are null when the future velocity variance is numerically degenerate; true occupancy MAE is null when the all-pair implementation is not applicable.",
        "",
        f"Code commit: `{_git_commit()}`",
        f"Output root: `{ctx.output_dir}`",
        "",
    ])
    (ctx.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_curves(ctx: CapacityContext) -> dict[str, Any]:
    curve_path = ctx.output_dir / "learning_curves.json"
    curves = {
        experiment_id: {
            "train": _read_jsonl(ctx.output_dir / experiment_id / "train_history.jsonl"),
            "validation": _read_jsonl(ctx.output_dir / experiment_id / "validation_history.jsonl"),
        }
        for experiment_id in EXPERIMENT_IDS
    }
    _write_json(curve_path, {"schema": "pvb.dit.capacity_data.learning_curves.v1", "experiments": curves})
    plot_path = ctx.output_dir / "learning_curves.png"
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(13, 4))
        for experiment_id, values in curves.items():
            train = values["train"]
            validation = values["validation"]
            if train:
                axes[0].plot([row["step"] for row in train], [row["loss"] for row in train], label=experiment_id)
            if validation:
                axes[1].plot([row["step"] for row in validation], [row["total"] for row in validation], label=experiment_id)
        axes[0].set_title("Training RF loss")
        axes[1].set_title("Validation RF loss")
        for axis in axes:
            axis.set_xlabel("successful optimizer update")
            axis.set_yscale("log")
            axis.legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=140)
        plt.close(figure)
        plot_status = "written"
    except Exception as exc:  # plotting is reporting-only; curves JSON remains authoritative
        plot_status = f"unavailable: {type(exc).__name__}: {exc}"
    return {"curve_json": str(curve_path), "plot": str(plot_path), "plot_status": plot_status}


def _summarize(ctx: CapacityContext) -> dict[str, Any]:
    comparison_path = ctx.output_dir / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("capacity comparison.json is missing")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    curve = _plot_curves(ctx)
    train_summaries = {}
    for experiment_id in EXPERIMENT_IDS:
        path = ctx.output_dir / experiment_id / "train_summary.json"
        if path.is_file():
            train_summaries[experiment_id] = json.loads(path.read_text(encoding="utf-8"))
    if comparison.get("status") == "PARTIAL_BLOCKED_DATA":
        status = "DIT_CAPACITY_DATA_V1_PARTIAL_BLOCKED_DATA"
    else:
        status = "DIT_CAPACITY_DATA_V1_COMPLETE_FOR_REVIEW"
    if any(
        value.get("status") not in ("PASS", "BLOCKED_DATA")
        for value in train_summaries.values()
    ):
        status = "DIT_CAPACITY_DATA_V1_PARTIAL_BUDGET"
    result = {
        "schema": "pvb.dit.capacity_data.summary.v1",
        "status": status,
        "comparison": comparison,
        "train_summaries": train_summaries,
        "curves": curve,
        "hashes": {
            "code_commit": _git_commit(),
            "data_hash": ctx.data_hash,
            "codec_state_hash": ctx.codec.codec_state_hash,
            "statistics_hash": ctx.statistics.hash,
            "statistics_file_sha256": str(ctx.cfg["candidate"]["statistics_file_sha256"]),
        },
        "test_payload_opened": False,
    }
    _write_json(ctx.output_dir / "summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/dit_capacity_data_v1.yaml")
    parser.add_argument("--stage", choices=("preflight", "verify", "source_check", "prepare", "train", "evaluate", "summarize", "all"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-manifest-root", type=Path, default=None)
    parser.add_argument("--experiment", choices=("all",) + EXPERIMENT_IDS, default="all")
    parser.add_argument("--resume", action="store_true")
    return parser


def _resolved_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cfg = _load_config(args.config.resolve())
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.data_manifest_root is not None:
        cfg["data_manifest_root"] = str(args.data_manifest_root)
    if args.output_root is not None:
        cfg["output_root"] = str(args.output_root)
    output_dir = _resolve(cfg["output_root"]) / str(cfg["run_id"])
    return cfg, output_dir


def main() -> None:
    args = _parser().parse_args()
    cfg, output_dir = _resolved_args(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        result = _manifest_metadata(cfg)
        result.update({"run_id": cfg["run_id"], "code_commit": _git_commit()})
        _write_json(output_dir / "preflight.json", result)
        print(json.dumps(_safe(result), indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stages = ("preflight", "verify", "source_check", "prepare", "train", "evaluate", "summarize") if args.stage == "all" else (args.stage,)
    selected = EXPERIMENT_IDS if args.experiment == "all" else (args.experiment,)
    result: Any = None
    for stage in stages:
        if stage == "preflight":
            result = _manifest_metadata(cfg)
            result.update({"run_id": cfg["run_id"], "code_commit": _git_commit()})
            _write_json(output_dir / "preflight.json", result)
        else:
            ctx = _load_context(cfg, output_dir, device)
            try:
                specs = _experiment_specs(cfg)
                if stage == "verify":
                    result = _verify(ctx, specs)
                elif stage == "source_check":
                    result = _source_check(ctx)
                elif stage == "prepare":
                    result = _prepare(ctx, specs)
                elif stage == "train":
                    result = _train(ctx, specs, args.resume, selected)
                elif stage == "evaluate":
                    result = _evaluate(ctx, specs)
                elif stage == "summarize":
                    result = _summarize(ctx)
                else:
                    raise ValueError(f"unsupported capacity stage: {stage}")
            finally:
                ctx.close()
        print(json.dumps(_safe({"stage": stage, "result": result}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
