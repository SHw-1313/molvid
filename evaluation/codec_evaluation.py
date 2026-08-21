"""Round-trip metrics and control evaluation for the multi-frame codec."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor

from data.clip_dataset import STATIC_TASK, ClipBatch
from trainer.codec_losses import acceleration_loss, velocity_loss

try:
    from torch_cluster import radius_graph
except (ImportError, OSError):
    radius_graph = None


EVALUATION_SCHEMA = "pvb.codec.eval.v1"


def _field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _prediction(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Mapping) and "x_hat" in value:
        return value["x_hat"]
    if hasattr(value, "x_hat"):
        return value.x_hat
    raise TypeError("control predictor must return coordinates or expose x_hat")


def _mask(batch: Any, frames: int, device: torch.device) -> Tensor:
    frame_mask = torch.as_tensor(_field(batch, "frame_mask"), device=device, dtype=torch.bool)
    abid = torch.as_tensor(_field(batch, "abid"), device=device, dtype=torch.long)
    loss_mask = torch.as_tensor(_field(batch, "loss_mask"), device=device, dtype=torch.bool)
    return frame_mask.index_select(0, abid).transpose(0, 1) & loss_mask.unsqueeze(0)


def _pairs(batch: Any, device: torch.device) -> Optional[Tensor]:
    value = _field(batch, "bond_index")
    if value is None:
        return None
    value = torch.as_tensor(value, device=device, dtype=torch.long)
    if value.numel() == 0:
        return None
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError("bond_index must have shape [2, E]")
    return value


def _safe_mean(value: Tensor) -> float:
    return float(value.mean().detach().cpu()) if value.numel() else 0.0


def _rmsd(prediction: Tensor, target: Tensor, mask: Tensor, frames: Sequence[int]) -> float:
    values = []
    for frame in frames:
        valid = mask[frame]
        if torch.any(valid):
            values.append((prediction[frame, valid] - target[frame, valid]).square().sum(dim=-1).mean())
    return math.sqrt(max(_safe_mean(torch.stack(values)) if values else 0.0, 0.0))


def _drmsd(prediction: Tensor, target: Tensor, mask: Tensor, frames: Sequence[int]) -> float:
    values = []
    for frame in frames:
        valid_indices = torch.nonzero(mask[frame], as_tuple=False).flatten()
        if valid_indices.numel() < 2:
            continue
        pred_dist = torch.pdist(prediction[frame].index_select(0, valid_indices))
        target_dist = torch.pdist(target[frame].index_select(0, valid_indices))
        values.append((pred_dist - target_dist).square().mean())
    return math.sqrt(max(_safe_mean(torch.stack(values)) if values else 0.0, 0.0))


def _bond_rmse(prediction: Tensor, target: Tensor, batch: Any, mask: Tensor, frames: Sequence[int]) -> float:
    pairs = _pairs(batch, prediction.device)
    if pairs is None:
        return 0.0
    values = []
    source, destination = pairs
    for frame in frames:
        pair_mask = mask[frame, source] & mask[frame, destination]
        if torch.any(pair_mask):
            pred_dist = torch.linalg.vector_norm(prediction[frame, source] - prediction[frame, destination], dim=-1)
            target_dist = torch.linalg.vector_norm(target[frame, source] - target[frame, destination], dim=-1)
            values.append((pred_dist[pair_mask] - target_dist[pair_mask]).square())
    return math.sqrt(max(_safe_mean(torch.cat(values)) if values else 0.0, 0.0))


def _nonbond_pairs(batch: Any, atoms: int, device: torch.device) -> tuple[Tensor, Tensor]:
    abid_value = _field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(atoms, device=device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=device, dtype=torch.long)
    if abid.numel() != atoms:
        raise ValueError("abid must have one entry per atom")
    source_parts: list[Tensor] = []
    destination_parts: list[Tensor] = []
    for sample in torch.unique(abid, sorted=True).tolist():
        nodes = torch.nonzero(abid == int(sample), as_tuple=False).flatten()
        if nodes.numel() < 2:
            continue
        local = torch.triu_indices(nodes.numel(), nodes.numel(), offset=1, device=device)
        source_parts.append(nodes.index_select(0, local[0]))
        destination_parts.append(nodes.index_select(0, local[1]))
    if not source_parts:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    source = torch.cat(source_parts)
    destination = torch.cat(destination_parts)
    pairs = _pairs(batch, device)
    if pairs is not None:
        bond_codes = torch.minimum(pairs[0], pairs[1]) * atoms + torch.maximum(pairs[0], pairs[1])
        codes = source * atoms + destination
        keep = ~torch.isin(codes, bond_codes)
        source, destination = source[keep], destination[keep]
    return source, destination


def _candidate_nonbond_pairs(
    batch: Any, positions: Sequence[Tensor], radius: float
) -> tuple[Tensor, Tensor]:
    atoms = int(positions[0].shape[0])
    if atoms <= 2048 or radius_graph is None:
        return _nonbond_pairs(batch, atoms, positions[0].device)
    abid_value = _field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(atoms, device=positions[0].device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=positions[0].device, dtype=torch.long)
    code_parts: list[Tensor] = []
    for position in positions:
        edges = radius_graph(
            position,
            r=float(radius),
            batch=abid,
            loop=False,
            max_num_neighbors=128,
        )
        if edges.numel():
            low = torch.minimum(edges[0], edges[1])
            high = torch.maximum(edges[0], edges[1])
            code_parts.append(torch.unique(low * atoms + high))
    if not code_parts:
        empty = torch.empty(0, device=positions[0].device, dtype=torch.long)
        return empty, empty
    codes = torch.unique(torch.cat(code_parts))
    source = torch.div(codes, atoms, rounding_mode="floor")
    destination = codes.remainder(atoms)
    pairs = _pairs(batch, positions[0].device)
    if pairs is not None:
        bond_codes = torch.minimum(pairs[0], pairs[1]) * atoms + torch.maximum(pairs[0], pairs[1])
        keep = ~torch.isin(codes, bond_codes)
        source, destination = source[keep], destination[keep]
    return source, destination


def _nonbond_count(batch: Any, valid_mask: Tensor) -> int:
    abid_value = _field(batch, "abid")
    if abid_value is None:
        abid = torch.zeros(valid_mask.shape[0], device=valid_mask.device, dtype=torch.long)
    else:
        abid = torch.as_tensor(abid_value, device=valid_mask.device, dtype=torch.long)
    total = 0
    for sample in torch.unique(abid, sorted=True).tolist():
        count = int((valid_mask & (abid == int(sample))).sum().item())
        total += count * (count - 1) // 2
    pairs = _pairs(batch, valid_mask.device)
    if pairs is not None:
        same_sample = abid.index_select(0, pairs[0]) == abid.index_select(0, pairs[1])
        total -= int((same_sample & valid_mask.index_select(0, pairs[0]) & valid_mask.index_select(0, pairs[1])).sum().item())
    return max(total, 1)


def _contact_error(prediction: Tensor, target: Tensor, batch: Any, mask: Tensor, frames: Sequence[int], cutoff: float = 4.5) -> float:
    values = []
    for frame in frames:
        source, destination = _candidate_nonbond_pairs(
            batch, (prediction[frame], target[frame]), cutoff
        )
        pair_mask = mask[frame, source] & mask[frame, destination]
        total_pairs = _nonbond_count(batch, mask[frame])
        if torch.any(pair_mask):
            pred_dist = torch.linalg.vector_norm(prediction[frame, source] - prediction[frame, destination], dim=-1)
            target_dist = torch.linalg.vector_norm(target[frame, source] - target[frame, destination], dim=-1)
            pred_contacts = (pred_dist[pair_mask] < cutoff).sum()
            target_contacts = (target_dist[pair_mask] < cutoff).sum()
            values.append((pred_contacts - target_contacts).abs().float() / float(total_pairs))
        else:
            values.append(torch.zeros((), device=prediction.device))
    return _safe_mean(torch.stack(values)) if values else 0.0


def _clash_rate(prediction: Tensor, batch: Any, mask: Tensor, frames: Sequence[int], cutoff: float = 0.7) -> float:
    values = []
    for frame in frames:
        source, destination = _candidate_nonbond_pairs(batch, (prediction[frame],), cutoff)
        pair_mask = mask[frame, source] & mask[frame, destination]
        total_pairs = _nonbond_count(batch, mask[frame])
        if torch.any(pair_mask):
            distance = torch.linalg.vector_norm(prediction[frame, source] - prediction[frame, destination], dim=-1)
            values.append((distance[pair_mask] < cutoff).sum().float() / float(total_pairs))
        else:
            values.append(torch.zeros((), device=prediction.device))
    return _safe_mean(torch.stack(values)) if values else 0.0


def _wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _dihedral(points: Tensor) -> Tensor:
    b0 = points[..., 1, :] - points[..., 0, :]
    b1 = points[..., 2, :] - points[..., 1, :]
    b2 = points[..., 3, :] - points[..., 2, :]
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    return torch.atan2((torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def _torsion_change(prediction: Tensor, target: Tensor, batch: Any, mask: Tensor, frames: Sequence[int]) -> float:
    torsions = _field(batch, "torsion_index")
    if torsions is None:
        return 0.0
    torsions = torch.as_tensor(torsions, device=prediction.device, dtype=torch.long)
    if torsions.numel() == 0:
        return 0.0
    if torsions.ndim != 2 or torsions.shape[0] != 4:
        raise ValueError("torsion_index must have shape [4, Q]")
    values = []
    for frame in frames:
        valid = mask[frame].index_select(0, torsions.reshape(-1)).reshape(4, -1).all(0)
        if torch.any(valid):
            pred_angle = _dihedral(prediction[frame, torsions].transpose(0, 1))[valid]
            target_angle = _dihedral(target[frame, torsions].transpose(0, 1))[valid]
            values.append(_wrap_angle(pred_angle - target_angle).abs())
    return _safe_mean(torch.cat(values)) if values else 0.0


def _frequency_retention(prediction: Tensor, target: Tensor, mask: Tensor) -> float:
    if prediction.shape[0] < 3:
        return 1.0 if torch.allclose(prediction, target) else 0.0
    valid = mask.all(0)
    if not torch.any(valid):
        return 0.0
    pred_signal = prediction[:, valid] - prediction[:, valid].mean(0, keepdim=True)
    target_signal = target[:, valid] - target[:, valid].mean(0, keepdim=True)
    pred_power = torch.fft.rfft(pred_signal, dim=0).abs().square()[1:].sum()
    target_power = torch.fft.rfft(target_signal, dim=0).abs().square()[1:].sum()
    if target_power <= 1e-12:
        return 1.0 if pred_power <= 1e-12 else 0.0
    return float((pred_power / target_power).detach().cpu())


def _metrics(prediction: Tensor, target: Tensor, batch: Any) -> dict[str, dict[str, float]]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("control output and target must both have shape [T, N, 3]")
    mask = _mask(batch, prediction.shape[0], prediction.device)
    future = list(range(1, prediction.shape[0]))
    all_frames = list(range(prediction.shape[0]))
    def one(frames: Sequence[int]) -> dict[str, float]:
        return {
            "rmsd": _rmsd(prediction, target, mask, frames),
            "drmsd": _drmsd(prediction, target, mask, frames),
            "bond_rmse": _bond_rmse(prediction, target, batch, mask, frames),
            "contact_error": _contact_error(prediction, target, batch, mask, frames),
            "clash_rate": _clash_rate(prediction, batch, mask, frames),
            "torsion_change": _torsion_change(prediction, target, batch, mask, frames),
        }
    future_metrics = one(future) if future else one([])
    result = {
        "frame0": one([0]),
        "future": future_metrics,
        "all_frames": one(all_frames),
        "velocity_rmse": math.sqrt(max(float(velocity_loss(prediction, target, batch).detach().cpu()), 0.0)),
        "acceleration_rmse": math.sqrt(max(float(acceleration_loss(prediction, target, batch).detach().cpu()), 0.0)),
        "frequency_retention": _frequency_retention(prediction, target, mask),
    }
    return result


@dataclass(frozen=True)
class EvaluationControl:
    name: str
    predictor: Callable[[Any], Any]
    ratio: int
    temporal: bool
    loss_evaluator: Optional[Callable[[Tensor, Any], Mapping[str, float]]] = None


def anchor_control(name: str = "ratio1_no_temporal") -> EvaluationControl:
    def predict(batch: Any) -> Tensor:
        x = torch.as_tensor(_field(batch, "x"))
        return x[0].unsqueeze(0).expand_as(x)
    return EvaluationControl(name=name, predictor=predict, ratio=1, temporal=False)


def model_control(
    name: str,
    model: Callable[[Any], Any],
    *,
    ratio: int,
    temporal: bool,
    loss_evaluator: Optional[Callable[[Tensor, Any], Mapping[str, float]]] = None,
) -> EvaluationControl:
    return EvaluationControl(
        name=name,
        predictor=model,
        ratio=int(ratio),
        temporal=bool(temporal),
        loss_evaluator=loss_evaluator,
    )


def _mean_records(records: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    if not records:
        return {"frame0": {}, "future": {}, "all_frames": {}}
    result: dict[str, Any] = {}
    for section in ("frame0", "future", "all_frames"):
        keys = records[0][section]
        result[section] = {
            key: sum(record[section][key] for record in records) / len(records)
            for key in keys
        }
    for key in ("velocity_rmse", "acceleration_rmse", "frequency_retention"):
        result[key] = sum(record[key] for record in records) / len(records)
    return result


def _mean_loss_records(records: list[Mapping[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = tuple(records[0])
    return {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in keys
    }


def _limited_batches(batches: Iterable[Any], max_batches: Optional[int]) -> Iterable[Any]:
    if max_batches is not None and int(max_batches) < 0:
        raise ValueError("max_batches must be non-negative")
    for index, batch in enumerate(batches):
        if max_batches is not None and index >= int(max_batches):
            break
        yield batch


def evaluate_controls(
    controls: Sequence[EvaluationControl],
    batches: Iterable[Any],
    *,
    max_batches: Optional[int] = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate controls without an unqualified cross-time-bucket mean."""

    output: dict[str, Any] = {"schema_version": EVALUATION_SCHEMA, "controls": {}}
    memory_device = torch.device(device) if device is not None else None
    for control in controls:
        per_bucket: dict[str, dict[str, Any]] = {}
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(memory_device)
        sample_count = 0
        for batch in _limited_batches(batches, max_batches):
            prediction = _prediction(control.predictor(batch))
            target = torch.as_tensor(_field(batch, "x"), device=prediction.device, dtype=prediction.dtype)
            bucket_ids = tuple(str(item) for item in _field(batch, "time_bucket_id"))
            if not bucket_ids or len(set(bucket_ids)) != 1:
                raise ValueError("evaluation requires homogeneous time buckets")
            bucket = bucket_ids[0]
            batch_metrics = _metrics(prediction, target, batch)
            delta = _field(batch, "delta_time_ps")
            frame_count = int(target.shape[0])
            if frame_count > 1 and torch.as_tensor(delta).numel():
                native_delta = float(torch.as_tensor(delta).mean())
                latent_interval = native_delta * control.ratio
            else:
                native_delta = None
                latent_interval = None
            spans = torch.as_tensor(_field(batch, "time_ps"))[:, -1] - torch.as_tensor(_field(batch, "time_ps"))[:, 0]
            atom_count = int(target.shape[1])
            bucket_state = per_bucket.setdefault(
                bucket,
                {
                    "records": [],
                    "loss_records": [],
                    "sample_count": 0,
                    "clip_spans": [],
                    "native_deltas": [],
                    "latent_intervals": [],
                    "latent_tokens": [],
                },
            )
            bucket_state["records"].append(batch_metrics)
            if control.loss_evaluator is not None:
                bucket_state["loss_records"].append(
                    dict(control.loss_evaluator(prediction, batch))
                )
            batch_size = int(getattr(batch, "batch_size", len(bucket_ids)))
            bucket_state["sample_count"] += batch_size
            bucket_state["clip_spans"].extend(float(value) for value in spans.tolist())
            if native_delta is not None:
                bucket_state["native_deltas"].append(native_delta)
                bucket_state["latent_intervals"].append(latent_interval)
            bucket_state["latent_tokens"].append(int(math.ceil(frame_count / control.ratio) * atom_count))
            sample_count += batch_size
        elapsed = time.perf_counter() - start
        by_bucket = {}
        for bucket, state in per_bucket.items():
            by_bucket[bucket] = {
                "sample_count": state["sample_count"],
                "native_delta_time_ps": (sum(state["native_deltas"]) / len(state["native_deltas"]) if state["native_deltas"] else None),
                "physical_clip_span_ps": (sum(state["clip_spans"]) / len(state["clip_spans"]) if state["clip_spans"] else 0.0),
                "latent_interval_ps": (sum(state["latent_intervals"]) / len(state["latent_intervals"]) if state["latent_intervals"] else None),
                "latent_tokens": (sum(state["latent_tokens"]) / len(state["latent_tokens"]) if state["latent_tokens"] else 0.0),
                "metrics": _mean_records(state["records"]),
            }
            if state["loss_records"]:
                by_bucket[bucket]["loss"] = _mean_loss_records(state["loss_records"])
        output["controls"][control.name] = {
            "ratio": control.ratio,
            "temporal": control.temporal,
            "sample_count": sample_count,
            "runtime": {
                "wall_time_s": elapsed,
                "samples_per_s": sample_count / elapsed if elapsed > 0 else 0.0,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(memory_device)) if torch.cuda.is_available() else 0,
            },
            "by_time_bucket": by_bucket,
        }
    return output


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# PVB codec round-trip evaluation", "", "Metrics are stratified by native time bucket; no cross-bucket mean is reported.", ""]
    for name, control in report.get("controls", {}).items():
        lines.extend([f"## {name}", "", "| Bucket | Δt (ps) | Span (ps) | Latent interval (ps) | Validation total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Velocity RMSE | Acceleration RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for bucket, values in control.get("by_time_bucket", {}).items():
            metrics = values["metrics"]
            loss = values.get("loss", {})
            total_loss = loss.get("total")
            loss_text = f"{total_loss:.6g}" if total_loss is not None else "—"
            native_delta = values["native_delta_time_ps"]
            latent_interval = values["latent_interval_ps"]
            lines.append(
                f"| {bucket} | {native_delta if native_delta is not None else '—'} | "
                f"{values['physical_clip_span_ps']:.4g} | "
                f"{latent_interval if latent_interval is not None else '—'} | "
                f"{loss_text} | {metrics['frame0'].get('rmsd', 0.0):.6g} | "
                f"{metrics['future'].get('rmsd', 0.0):.6g} | "
                f"{metrics['future'].get('drmsd', 0.0):.6g} | "
                f"{metrics.get('velocity_rmse', 0.0):.6g} | "
                f"{metrics.get('acceleration_rmse', 0.0):.6g} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    import json
    Path(json_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(markdown_path).write_text(report_markdown(report), encoding="utf-8")


__all__ = [
    "EVALUATION_SCHEMA",
    "EvaluationControl",
    "anchor_control",
    "evaluate_controls",
    "model_control",
    "report_markdown",
    "write_report",
]
