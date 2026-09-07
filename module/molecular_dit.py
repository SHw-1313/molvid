"""Factorized scalar/vector molecular DiT for state/detail latents.

The spatial backend is intentionally explicit: dense attention is applied only
over blocks within one sample and latent time. Temporal attention is applied
only over K for one atom. There is no dense attention over K*N atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
from torch import Tensor, nn

from .state_detail_latent_adapter import (
    AxisPreservingLinear,
    DIT_MODEL_SCHEMA,
    DiTLatentBatch,
    LatentFieldSet,
    StateDetailLatentAdapter,
    contract_hash,
)


def _safe_ids(value: Tensor, size: int) -> Tensor:
    return value.to(dtype=torch.long).remainder(int(size))


def _sample_token_values(value: Tensor, abid: Tensor) -> Tensor:
    return value.index_select(0, abid).transpose(0, 1)


class ScalarVectorAttention(nn.Module):
    """Multi-head attention with scalar q/k and shared scalar weights."""

    def __init__(self, scalar_width: int, vector_width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if scalar_width % heads or vector_width % heads:
            raise ValueError("scalar and vector widths must be divisible by heads")
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.heads = int(heads)
        self.scalar_head = scalar_width // heads
        self.vector_head = vector_width // heads
        self.q = nn.Linear(scalar_width, scalar_width)
        self.k = nn.Linear(scalar_width, scalar_width)
        self.scalar_value = nn.Linear(scalar_width, scalar_width)
        self.scalar_out = nn.Linear(scalar_width, scalar_width)
        self.vector_value = AxisPreservingLinear(vector_width, vector_width)
        self.vector_out = AxisPreservingLinear(vector_width, vector_width)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        key_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("attention expects [B,L,Dh] and [B,L,3,Dv]")
        batch, length = h.shape[:2]
        q = self.q(h).reshape(batch, length, self.heads, self.scalar_head).transpose(1, 2)
        k = self.k(h).reshape(batch, length, self.heads, self.scalar_head).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.scalar_head)
        valid = key_mask.to(dtype=torch.bool).reshape(batch, 1, 1, length)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        scalar_value = self.scalar_value(h).reshape(
            batch, length, self.heads, self.scalar_head
        )
        scalar_mixed = torch.einsum("bhij,bjhd->bihd", weights, scalar_value).reshape(
            batch, length, self.scalar_width
        )
        vector_value = self.vector_value(v).reshape(
            batch, length, 3, self.heads, self.vector_head
        ).permute(0, 3, 1, 2, 4)
        vector_mixed = torch.einsum("bhij,bhjrc->bhirc", weights, vector_value)
        vector_mixed = vector_mixed.permute(0, 2, 3, 1, 4).reshape(
            batch, length, 3, self.vector_width
        )
        return self.scalar_out(scalar_mixed), self.vector_out(vector_mixed)


class AdaLNZero(nn.Module):
    """Scalar shifts are allowed; vector modulation has scale/gate only."""

    def __init__(self, scalar_width: int, vector_width: int) -> None:
        super().__init__()
        self.scalar_norm = nn.LayerNorm(scalar_width, elementwise_affine=False)
        self.vector_norm = nn.LayerNorm(vector_width, elementwise_affine=False)
        self.modulation = nn.Linear(scalar_width, 3 * scalar_width + 2 * vector_width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h: Tensor, v: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = self.modulation(condition)
        dh = h.shape[-1]
        dv = v.shape[-1]
        h_shift, h_scale, h_gate, v_scale, v_gate = torch.split(
            values, (dh, dh, dh, dv, dv), dim=-1
        )
        h_norm = self.scalar_norm(h) * (1.0 + h_scale) + h_shift
        v_norm = self.vector_norm(v) * (1.0 + v_scale).unsqueeze(-2)
        return h_norm, v_norm, h_gate, v_gate


class ScalarVectorFFN(nn.Module):
    def __init__(self, scalar_width: int, vector_width: int, multiplier: int) -> None:
        super().__init__()
        scalar_hidden = int(scalar_width * multiplier)
        vector_hidden = int(vector_width * multiplier)
        self.scalar = nn.Sequential(
            nn.Linear(scalar_width, scalar_hidden),
            nn.SiLU(),
            nn.Linear(scalar_hidden, scalar_width),
        )
        self.vector_in = AxisPreservingLinear(vector_width, vector_hidden)
        self.vector_out = AxisPreservingLinear(vector_hidden, vector_width)
        self.vector_gate = nn.Linear(scalar_width, vector_width)

    def forward(self, h: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        scalar = self.scalar(h)
        vector = self.vector_out(torch.nn.functional.silu(self.vector_in(v)))
        vector = vector * torch.sigmoid(self.vector_gate(h)).unsqueeze(-2)
        return scalar, vector


class FactorizedDiTBlock(nn.Module):
    def __init__(
        self,
        scalar_width: int,
        vector_width: int,
        heads: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spatial = ScalarVectorAttention(scalar_width, vector_width, heads, dropout)
        self.temporal = ScalarVectorAttention(scalar_width, vector_width, heads, dropout)
        self.ffn = ScalarVectorFFN(scalar_width, vector_width, ffn_multiplier)
        self.spatial_adaln = AdaLNZero(scalar_width, vector_width)
        self.temporal_adaln = AdaLNZero(scalar_width, vector_width)
        self.ffn_adaln = AdaLNZero(scalar_width, vector_width)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        condition: Tensor,
        batch: DiTLatentBatch,
        block_groups: list[Tensor],
        block_samples: list[int],
    ) -> tuple[Tensor, Tensor]:
        h_norm, v_norm, h_gate, v_gate = self.spatial_adaln(h, v, condition)
        spatial_h, spatial_v = _block_spatial(
            h_norm, v_norm, batch, self.spatial, block_groups, block_samples
        )
        h = h + torch.tanh(h_gate) * spatial_h
        v = v + torch.tanh(v_gate).unsqueeze(-2) * spatial_v

        h_norm, v_norm, h_gate, v_gate = self.temporal_adaln(h, v, condition)
        temporal_h, temporal_v = _atom_temporal(h_norm, v_norm, batch, self.temporal)
        h = h + torch.tanh(h_gate) * temporal_h
        v = v + torch.tanh(v_gate).unsqueeze(-2) * temporal_v

        h_norm, v_norm, h_gate, v_gate = self.ffn_adaln(h, v, condition)
        ffn_h, ffn_v = self.ffn(h_norm, v_norm)
        h = h + torch.tanh(h_gate) * ffn_h
        v = v + torch.tanh(v_gate).unsqueeze(-2) * ffn_v
        return h, v


def _block_spatial(
    h: Tensor,
    v: Tensor,
    batch: DiTLatentBatch,
    attention: ScalarVectorAttention,
    groups: list[Tensor],
    group_samples: list[int],
) -> tuple[Tensor, Tensor]:
    """Pool/broadcast one latent time using sample-local block attention."""
    k_count, n_atoms = h.shape[:2]
    h_out = torch.zeros_like(h)
    v_out = torch.zeros_like(v)
    for time_index in range(k_count):
        block_h: list[Tensor] = []
        block_v: list[Tensor] = []
        for atom_indices in groups:
            block_h.append(h[time_index].index_select(0, atom_indices).mean(dim=0))
            block_v.append(v[time_index].index_select(0, atom_indices).mean(dim=0))
        if not block_h:
            continue
        all_h = torch.stack(block_h)
        all_v = torch.stack(block_v)
        for sample in range(batch.batch_size):
            block_indices = [
                index for index, owner in enumerate(group_samples) if owner == sample
            ]
            if not block_indices or not bool(batch.token_mask[sample, time_index]):
                continue
            index = torch.tensor(block_indices, device=h.device, dtype=torch.long)
            local_h = all_h.index_select(0, index).unsqueeze(0)
            local_v = all_v.index_select(0, index).unsqueeze(0)
            local_mask = torch.ones((1, index.numel()), device=h.device, dtype=torch.bool)
            context_h, context_v = attention(local_h, local_v, local_mask)
            context_h, context_v = context_h[0], context_v[0]
            for local, global_index in enumerate(block_indices):
                atom_indices = groups[global_index]
                h_out[time_index, atom_indices] = context_h[local]
                v_out[time_index, atom_indices] = context_v[local]
    return h_out, v_out


def _atom_temporal(
    h: Tensor, v: Tensor, batch: DiTLatentBatch, attention: ScalarVectorAttention
) -> tuple[Tensor, Tensor]:
    h_atoms = h.permute(1, 0, 2)
    v_atoms = v.permute(1, 0, 2, 3)
    valid = batch.token_mask.index_select(0, batch.abid)
    h_out, v_out = attention(h_atoms, v_atoms, valid)
    return h_out.permute(1, 0, 2), v_out.permute(1, 0, 2, 3)


def _build_block_groups(batch: DiTLatentBatch) -> tuple[list[Tensor], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for atom, (sample, block) in enumerate(zip(batch.abid.tolist(), batch.block_id.tolist())):
        groups.setdefault((int(sample), int(block)), []).append(atom)
    ordered = sorted(groups.items())
    return [
        torch.tensor(indices, device=batch.state_h.device, dtype=torch.long)
        for _, indices in ordered
    ], [sample for (sample, _), _ in ordered]


class MolecularDiT(nn.Module):
    """One shared R2/R4 factorized scalar/vector DiT implementation."""

    def __init__(
        self,
        *,
        adapter: StateDetailLatentAdapter,
        scalar_width: int = 256,
        vector_width: int = 128,
        depth: int = 4,
        heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        max_atom_type: int = 128,
        max_block_type: int = 256,
        max_component_type: int = 128,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.ffn_multiplier = int(ffn_multiplier)
        self.dropout = float(dropout)
        if min(self.scalar_width, self.vector_width, self.depth, self.heads) < 1:
            raise ValueError("model widths, depth, and heads must be positive")
        if self.scalar_width % self.heads or self.vector_width % self.heads:
            raise ValueError("both model widths must divide heads")
        self.flow_time = nn.Sequential(
            nn.Linear(3, self.scalar_width),
            nn.SiLU(),
            nn.Linear(self.scalar_width, self.scalar_width),
        )
        self.ratio_embedding = nn.Embedding(2, self.scalar_width)
        self.observed_embedding = nn.Embedding(2, self.scalar_width)
        self.atom_embedding = nn.Embedding(int(max_atom_type), self.scalar_width)
        self.block_embedding = nn.Embedding(int(max_block_type), self.scalar_width)
        self.component_embedding = nn.Embedding(int(max_component_type), self.scalar_width)
        self.blocks = nn.ModuleList(
            [
                FactorizedDiTBlock(
                    self.scalar_width,
                    self.vector_width,
                    self.heads,
                    self.ffn_multiplier,
                    self.dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.backend = "dense_block_attention_factorized_atom_temporal_v1"

    def _condition(self, batch: DiTLatentBatch, tau: Tensor) -> Tensor:
        valid_frames = batch.block_frame_mask
        count = valid_frames.sum(dim=-1).clamp_min(1).to(batch.block_time_ps.dtype)
        block_time = (batch.block_time_ps * valid_frames.to(batch.block_time_ps.dtype)).sum(dim=-1) / count
        min_time = batch.block_time_ps.masked_fill(~valid_frames, float("inf")).amin(dim=-1)
        max_time = batch.block_time_ps.masked_fill(~valid_frames, float("-inf")).amax(dim=-1)
        span = torch.where(valid_frames.any(dim=-1), max_time - min_time, torch.zeros_like(max_time))
        time = _sample_token_values(block_time, batch.abid)
        span = _sample_token_values(span, batch.abid)
        flow = _sample_token_values(
            torch.as_tensor(tau, device=batch.state_h.device, dtype=batch.state_h.dtype).reshape(-1, 1).expand(
                batch.batch_size, batch.tokens
            ),
            batch.abid,
        )
        numeric = torch.stack((flow, time * 0.01, span * 0.01), dim=-1)
        condition = self.flow_time(numeric)
        condition = condition + self.ratio_embedding(
            torch.full_like(batch.abid, 0 if batch.ratio == 2 else 1)
        ).unsqueeze(0)
        observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1).long()
        condition = condition + self.observed_embedding(observed)
        condition = condition + self.atom_embedding(_safe_ids(batch.atom_type, self.atom_embedding.num_embeddings)).unsqueeze(0)
        condition = condition + self.block_embedding(_safe_ids(batch.block_type, self.block_embedding.num_embeddings)).unsqueeze(0)
        condition = condition + self.component_embedding(_safe_ids(batch.component_id, self.component_embedding.num_embeddings)).unsqueeze(0)
        return condition

    def forward(self, batch: DiTLatentBatch, tau: Tensor | float) -> LatentFieldSet:
        if batch.ratio != self.adapter.ratio or batch.mode != self.adapter.mode:
            raise ValueError("model and batch ratio/mode disagree")
        h, v = self.adapter.project_inputs(batch)
        condition = self._condition(batch, torch.as_tensor(tau, device=h.device, dtype=h.dtype))
        h = h + condition
        groups, group_samples = _build_block_groups(batch)
        for block in self.blocks:
            h, v = block(h, v, condition, batch, groups, group_samples)
        output = self.adapter.project_outputs(h, v)
        return _masked_output(output, batch)

    def contract(self) -> dict[str, object]:
        return {
            "schema_version": DIT_MODEL_SCHEMA,
            "adapter_schema": self.adapter_contract_hash,
            "ratio": self.adapter.ratio,
            "mode": self.adapter.mode,
            "scalar_width": self.scalar_width,
            "vector_width": self.vector_width,
            "depth": self.depth,
            "heads": self.heads,
            "ffn_multiplier": self.ffn_multiplier,
            "dropout": self.dropout,
            "backend": self.backend,
            "attention_complexity": "sum_s K*M_s^2 + sum_n K^2",
            "dense_all_atom_time_attention": False,
            "temporal_attention": "bidirectional",
            "vector_maps": "bias_free_channel_only",
            "vector_adaln": "scale_and_gate_only",
        }

    @property
    def adapter_contract_hash(self) -> str:
        return contract_hash(self.adapter.contract())

    @property
    def model_hash(self) -> str:
        return contract_hash(self.contract())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _masked_output(fields: LatentFieldSet, batch: DiTLatentBatch) -> LatentFieldSet:
    masks = batch.field_masks()
    return LatentFieldSet(
        fields.state_h * masks["state_h"].to(fields.state_h.dtype).unsqueeze(-1),
        fields.detail_h * masks["detail_h"].to(fields.detail_h.dtype).unsqueeze(-1),
        fields.state_v * masks["state_v"].to(fields.state_v.dtype).unsqueeze(-1).unsqueeze(-1),
        fields.detail_v * masks["detail_v"].to(fields.detail_v.dtype).unsqueeze(-1).unsqueeze(-1),
    )


__all__ = [
    "AdaLNZero",
    "AxisPreservingLinear",
    "FactorizedDiTBlock",
    "MolecularDiT",
    "ScalarVectorAttention",
    "ScalarVectorFFN",
]
