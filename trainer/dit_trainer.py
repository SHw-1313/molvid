"""Training and checkpoint contracts for the bounded state/detail DiT probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from contextlib import nullcontext
import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn

from module.latent_rectified_flow import RectifiedFlowObjective
from module.state_detail_latent_adapter import (
    DIT_MODEL_SCHEMA,
    DiTLatentBatch,
    LatentStatistics,
    StateDetailLatentAdapter,
    contract_hash,
)


DIT_CHECKPOINT_SCHEMA = "pvb.dit.state_detail.checkpoint.v2"


def module_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class DiTTrainConfig:
    ratio: int
    mode: str
    codec_width: int = 128
    scalar_width: int = 256
    vector_width: int = 128
    depth: int = 4
    heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 100
    seed: int = 0
    amp: bool = False
    output_root: str = "outputs/dit_state_detail_probe_v1"
    data_hash: str = ""
    codec_hash: str = ""
    stats_hash: str = ""
    observation_mixture: tuple[int, ...] = (0, 4, 8)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.ratio not in (2, 4) or self.mode != f"ratio{self.ratio}_state_detail":
            raise ValueError("DiT config must select ratio2_state_detail or ratio4_state_detail")
        if self.codec_width < 1 or self.scalar_width < 1 or self.vector_width < 1:
            raise ValueError("all model widths must be positive")
        if self.depth < 1 or self.heads < 1 or self.ffn_multiplier < 1:
            raise ValueError("depth, heads, and ffn_multiplier must be positive")
        if self.scalar_width % self.heads or self.vector_width % self.heads:
            raise ValueError("model widths must be divisible by heads")
        max_steps_limit = 5000 if self.metadata.get("phase") == "t1_pilot" else 100
        if self.max_steps < 1 or self.max_steps > max_steps_limit:
            raise ValueError(
                f"the configured DiT phase allows at most {max_steps_limit} optimizer steps"
            )
        if not self.observation_mixture or any(value not in (0, 4, 8) for value in self.observation_mixture):
            raise ValueError("observation mixture must contain only H=0,4,8")
        if self.output_root.startswith("outputs/state_detail_codec_v2"):
            raise ValueError("DiT outputs must not be written under the active T1 root")

    def contract(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["observation_mixture"] = list(self.observation_mixture)
        return value


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class DiTTrainer:
    """Minimal trainer with explicit frozen-module and checkpoint contracts."""

    def __init__(
        self,
        model: nn.Module,
        adapter: StateDetailLatentAdapter,
        *,
        config: DiTTrainConfig,
        statistics: Optional[LatentStatistics] = None,
        codec: Optional[nn.Module] = None,
        frame_encoder: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        config.validate()
        if getattr(model, "adapter", adapter) is not adapter:
            raise ValueError("model and trainer must share the same latent adapter")
        if config.ratio != adapter.ratio or config.mode != adapter.mode:
            raise ValueError("trainer config and adapter ratio/mode disagree")
        self.model = model
        self.adapter = adapter
        self.config = config
        self.statistics = statistics
        model_parameter = next(iter(model.parameters()), None)
        if model_parameter is None:
            raise ValueError("the DiT must expose parameters")
        self.device = model_parameter.device
        if config.amp and self.device.type != "cuda":
            raise RuntimeError("DiT AMP requires a CUDA model; CPU fallback is not AMP")
        self.amp_enabled = bool(config.amp)
        if statistics is not None:
            if statistics.ratio != config.ratio or statistics.mode != config.mode:
                raise ValueError("statistics ratio/mode disagrees with trainer config")
            if config.stats_hash and config.stats_hash != statistics.hash:
                raise ValueError("configured statistics hash does not match statistics")
            config.stats_hash = statistics.hash
        self.codec = codec
        self.frame_encoder = frame_encoder
        self._freeze_module(codec, "codec")
        self._freeze_module(frame_encoder, "frame_encoder")
        self.flow = RectifiedFlowObjective()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("the DiT has no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = scheduler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.step = 0
        self.last_activation_dtypes: dict[str, str] = {}
        self._rng_seed = int(config.seed)
        _set_seed(self._rng_seed)
        self.frozen_hashes = self.frozen_state_hashes()
        self.codec_contract = (
            self.codec.model_contract() if self.codec is not None and hasattr(self.codec, "model_contract") else {}
        )

    def autocast_context(self):
        """Use the repository's explicit CUDA BF16 policy when AMP is enabled."""

        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _freeze_module(module: Optional[nn.Module], label: str) -> None:
        if module is None:
            return
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{label} contains a trainable parameter; freeze it before DiT training")
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def frozen_state_hashes(self) -> dict[str, str]:
        result = {}
        if self.codec is not None:
            result["codec"] = module_state_hash(self.codec)
        if self.frame_encoder is not None:
            result["frame_encoder"] = module_state_hash(self.frame_encoder)
        return result

    def _normalise_batch(self, batch: DiTLatentBatch) -> DiTLatentBatch:
        if batch.ratio != self.config.ratio or batch.mode != self.config.mode:
            raise ValueError("batch ratio/mode disagrees with trainer")
        if self.statistics is None:
            return batch.zero_invalid()
        return self.statistics.normalize(batch)

    def train_step(
        self,
        batch: DiTLatentBatch,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, Any]:
        self.model.train()
        if self.codec is not None:
            self.codec.eval()
        if self.frame_encoder is not None:
            self.frame_encoder.eval()
        batch = self._normalise_batch(batch)
        flow_sample = self.flow.sample(batch, generator=generator)
        noisy_batch = batch.with_fields(flow_sample.interpolated)
        with self.autocast_context():
            prediction = self.model(noisy_batch, flow_sample.tau)
            self.last_activation_dtypes = {
                name: str(getattr(prediction, name).dtype)
                for name in prediction.names()
            }
            loss = self.flow.loss(prediction, flow_sample.target, batch)
        if not torch.isfinite(loss.total):
            raise FloatingPointError("non-finite DiT loss")
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss.total).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.grad_clip,
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite DiT gradient norm")
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        valid = batch.field_masks()
        observation = batch.observed_mask
        return {
            "step": self.step,
            "loss": float(loss.total.detach().cpu()),
            "state_h_loss": float(loss.fields["state_h"].detach().cpu()),
            "detail_h_loss": float(loss.fields["detail_h"].detach().cpu()),
            "state_v_loss": float(loss.fields["state_v"].detach().cpu()),
            "detail_v_loss": float(loss.fields["detail_v"].detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "tau_mean": float(flow_sample.tau.mean().detach().cpu()),
            "tau_min": float(flow_sample.tau.min().detach().cpu()),
            "tau_max": float(flow_sample.tau.max().detach().cpu()),
            "observation_fraction": float(observation.float().mean().detach().cpu()),
            "valid_elements": {name: int(valid[name].sum().item()) for name in valid},
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        stats_contract = None if self.statistics is None else self.statistics.contract()
        return {
            "schema": DIT_CHECKPOINT_SCHEMA,
            "step": int(self.step),
            "config": self.config.contract(),
            "ratio": self.config.ratio,
            "mode": self.config.mode,
            "model_schema": DIT_MODEL_SCHEMA,
            "model_contract": self.model.contract() if hasattr(self.model, "contract") else {},
            "model_contract_hash": contract_hash(self.model.contract()) if hasattr(self.model, "contract") else "",
            "adapter_contract": self.adapter.contract(),
            "adapter_contract_hash": contract_hash(self.adapter.contract()),
            "statistics_contract": stats_contract,
            "statistics_hash": "" if self.statistics is None else self.statistics.hash,
            "statistics_state": None if self.statistics is None else self.statistics.state_dict(),
            "codec_contract": self.codec_contract,
            "codec_hash": self.config.codec_hash,
            "data_hash": self.config.data_hash,
            "frozen_hashes": self.frozen_state_hashes(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), target)
        return target

    def _validate_checkpoint(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != DIT_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported DiT checkpoint schema")
        for key, expected in (
            ("ratio", self.config.ratio),
            ("mode", self.config.mode),
            ("codec_hash", self.config.codec_hash),
            ("data_hash", self.config.data_hash),
        ):
            actual = payload.get(key, "")
            if expected and actual != expected:
                raise ValueError(f"checkpoint {key} mismatch: {actual!r} != {expected!r}")
        expected_stats = "" if self.statistics is None else self.statistics.hash
        if payload.get("statistics_hash", "") != expected_stats:
            raise ValueError("checkpoint statistics hash mismatch")
        statistics_state = payload.get("statistics_state")
        if self.statistics is None:
            if statistics_state is not None:
                raise ValueError("checkpoint contains statistics but trainer has no matching statistics")
        else:
            if not isinstance(statistics_state, Mapping):
                raise ValueError("checkpoint is missing serialized statistics tensors")
            restored = LatentStatistics.from_state_dict(statistics_state)
            if restored.hash != self.statistics.hash:
                raise ValueError("checkpoint serialized statistics mismatch")
            if statistics_state.get("statistics_hash") != payload.get("statistics_hash"):
                raise ValueError("checkpoint statistics artifact hash mismatch")
        if payload.get("codec_contract", {}) != self.codec_contract:
            raise ValueError("checkpoint codec contract mismatch")
        expected_model = contract_hash(self.model.contract()) if hasattr(self.model, "contract") else ""
        if payload.get("model_contract_hash", "") != expected_model:
            raise ValueError("checkpoint model contract mismatch")
        if payload.get("adapter_contract_hash") != contract_hash(self.adapter.contract()):
            raise ValueError("checkpoint adapter contract mismatch")
        if payload.get("frozen_hashes", {}) != self.frozen_state_hashes():
            raise ValueError("checkpoint frozen codec/frame-encoder hash mismatch")

    def load_checkpoint(self, path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if self.statistics is None and payload.get("statistics_state") is not None:
            self.statistics = LatentStatistics.from_state_dict(payload["statistics_state"])
            if self.config.stats_hash and self.config.stats_hash != self.statistics.hash:
                raise ValueError("checkpoint statistics hash disagrees with trainer config")
            self.config.stats_hash = self.statistics.hash
        self._validate_checkpoint(payload)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and payload.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            self.scaler.load_state_dict(payload["scaler_state"])
        self.step = int(payload["step"])
        rng = payload.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].to(device="cpu"))
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                for device_index, state in enumerate(rng["cuda"]):
                    torch.cuda.set_rng_state(
                        state.detach().to(device="cpu", dtype=torch.uint8),
                        device=device_index,
                    )
        if self.frozen_state_hashes() != self.frozen_hashes:
            raise RuntimeError("frozen codec/frame-encoder state changed during checkpoint load")
        return dict(payload)

    @staticmethod
    def load_statistics_from_checkpoint(
        path: str | Path, *, map_location: Any = "cpu"
    ) -> LatentStatistics:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state = payload.get("statistics_state")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint does not contain serialized latent statistics")
        return LatentStatistics.from_state_dict(state)


__all__ = [
    "DIT_CHECKPOINT_SCHEMA",
    "DiTTrainConfig",
    "DiTTrainer",
    "module_state_hash",
]
