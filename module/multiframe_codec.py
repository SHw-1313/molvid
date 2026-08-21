"""Batched spatial encoding for the multi-frame codec path.

The legacy PVB models consume one packed structure at a time.  This module
keeps that path untouched and provides the small adapter needed by the clip
contract: a time-major ``[T, N_total, 3]`` tensor is flattened into one graph
batch with one graph id per ``(sample, frame)`` pair.  The spatial backbone is
called exactly once and its scalar/vector outputs are restored to the
time-major layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from data.clip_dataset import ClipBatch
from utils.bio_utils import NUM_ATOM_TYPE, NUM_BLOCK_TYPE

from .torchmd_et import TorchMD_VQ_ET


def _get_field(batch: ClipBatch | Mapping[str, Any], name: str) -> Any:
    if isinstance(batch, Mapping):
        if name not in batch:
            raise ValueError(f"clip batch is missing required field: {name}")
        return batch[name]
    if not hasattr(batch, name):
        raise ValueError(f"clip batch is missing required field: {name}")
    return getattr(batch, name)


def _as_tensor(value: Any, *, name: str, dtype: torch.dtype | None = None) -> Tensor:
    result = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if result.numel() == 0 and name not in {"bond_index", "delta_time_ps"}:
        raise ValueError(f"clip field {name} must not be empty")
    return result


def _expand_atom_field(value: Tensor, *, frames: int, atoms: int, name: str) -> Tensor:
    """Expand a per-atom field without materializing an unnecessary copy."""

    if value.ndim == 1:
        if value.shape[0] != atoms:
            raise ValueError(f"{name} must have shape [N_total]")
        return value.unsqueeze(0).expand(frames, -1).reshape(-1)
    if value.ndim >= 2 and value.shape[0] == frames and value.shape[1] == atoms:
        return value.reshape(frames * atoms, *value.shape[2:])
    raise ValueError(
        f"{name} must have shape [N_total, ...] or [T, N_total, ...], "
        f"got {tuple(value.shape)}"
    )


def _normalise_bonds(bond_index: Tensor, *, atoms: int) -> Tensor:
    if bond_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=bond_index.device)
    if bond_index.ndim != 2:
        raise ValueError("bond_index must have shape [2, E]")
    if bond_index.shape[0] != 2 and bond_index.shape[1] == 2:
        bond_index = bond_index.transpose(0, 1)
    if bond_index.shape[0] != 2:
        raise ValueError("bond_index must have shape [2, E]")
    bond_index = bond_index.to(dtype=torch.long)
    if torch.any(bond_index < 0) or torch.any(bond_index >= atoms):
        raise ValueError("bond_index contains an out-of-range atom")
    return bond_index


@dataclass
class FrameGraphBatch:
    """Flattened graph inputs and topology used by one spatial encoder call."""

    pos: Tensor  # [T*N_total, 3]
    z: Tensor  # [T*N_total]
    b: Tensor  # [T*N_total]
    batch: Tensor  # [T*N_total], graph id = frame_id * B + sample_id
    graph_id: Tensor  # alias with an explicit name for callers/tests
    frame_id: Tensor  # [T*N_total]
    sample_id: Tensor  # [T*N_total]
    edge_index: Tensor  # [2, E], union of distance and covalent edges
    edge_weight: Tensor  # [E]
    edge_vec: Tensor  # [E, 3]
    bond_type: Tensor  # [E], 1 for a replicated covalent edge
    bond_index: Tensor  # [2, T*E_bond], replicated and frame-offset
    distance_edge_index: Tensor  # [2, E_distance]
    distance_edge_weight: Tensor  # [E_distance]
    distance_edge_vec: Tensor  # [E_distance, 3]
    frame_mask: Tensor  # [B, T]

    @property
    def num_nodes(self) -> int:
        return int(self.pos.shape[0])

    @property
    def num_graphs(self) -> int:
        if self.batch.numel() == 0:
            return 0
        return int(self.batch.max().item()) + 1


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Auditable result of optional spatial-backbone checkpoint loading."""

    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...] = ()

    @property
    def matched_keys(self) -> tuple[str, ...]:
        return self.matched

    @property
    def missing_keys(self) -> tuple[str, ...]:
        return self.missing

    @property
    def unexpected_keys(self) -> tuple[str, ...]:
        return self.unexpected

    def __getitem__(self, key: str) -> Any:
        if key in {"matched", "matched_keys"}:
            return self.matched
        if key in {"missing", "missing_keys"}:
            return self.missing
        if key in {"unexpected", "unexpected_keys"}:
            return self.unexpected
        if key == "shape_mismatch":
            return self.shape_mismatch
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": list(self.matched),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "shape_mismatch": [
                {
                    "key": key,
                    "checkpoint_shape": list(checkpoint_shape),
                    "model_shape": list(model_shape),
                }
                for key, checkpoint_shape, model_shape in self.shape_mismatch
            ],
        }


@dataclass
class FrameEncoderOutput:
    """Time-major scalar/vector features plus the auditable graph batch."""

    h: Tensor  # [T, N_total, C], invariant scalar features
    v: Tensor  # [T, N_total, 3, C], equivariant vector features
    graph: FrameGraphBatch

    @property
    def scalar(self) -> Tensor:
        return self.h

    @property
    def vector(self) -> Tensor:
        return self.v

    def __iter__(self):
        # Keep the common ``h, v = encoder(batch)`` spelling convenient while
        # retaining graph metadata for tests and downstream codec modules.
        yield self.h
        yield self.v


class PVBFrameEncoder(nn.Module):
    """Batch all frames through one shared :class:`TorchMD_VQ_ET` call.

    ``ClipBatch.x`` is time-major and atoms are packed by sample.  The graph
    id formula is deliberately explicit and stable:

    ``graph_id[t, atom] = t * batch_size + abid[atom]``.

    The default backbone is the existing PVB TorchMD implementation.  Tests
    and small downstream adapters may inject an equivalent module through
    ``spatial_encoder``; it must return scalar and vector features as its
    first two outputs.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_rbf: int = 50,
        num_heads: int = 8,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        *,
        spatial_encoder: nn.Module | None = None,
        encoder: nn.Module | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint: str | Path | None = None,
        neighbor_backend: str = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if spatial_encoder is not None and encoder is not None:
            raise ValueError("pass only one of spatial_encoder and encoder")
        if checkpoint_path is not None and checkpoint is not None:
            raise ValueError("pass only one of checkpoint_path and checkpoint")
        if neighbor_backend not in {"auto", "optimized", "dense"}:
            raise ValueError("neighbor_backend must be 'auto', 'optimized', or 'dense'")
        if max_num_neighbors < 1:
            raise ValueError("max_num_neighbors must be positive")

        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_num_neighbors = int(max_num_neighbors)
        self.neighbor_backend = neighbor_backend
        self.spatial_encoder = (
            spatial_encoder if spatial_encoder is not None else encoder
        )
        if self.spatial_encoder is None:
            self.spatial_encoder = TorchMD_VQ_ET(
                hidden_channels=hidden_channels,
                extra_channels=0,
                num_layers=num_layers,
                num_rbf=num_rbf,
                num_heads=num_heads,
                cutoff_lower=cutoff_lower,
                cutoff_upper=cutoff_upper,
                max_z=NUM_ATOM_TYPE,
                max_b=NUM_BLOCK_TYPE,
                max_num_neighbors=max_num_neighbors,
                cross_attn=False,
                dtype=dtype,
            )
        self.checkpoint_report: CheckpointLoadReport | None = None
        selected_checkpoint = checkpoint_path if checkpoint_path is not None else checkpoint
        if selected_checkpoint is not None:
            self.checkpoint_report = self.load_checkpoint(selected_checkpoint)

    @property
    def encoder_module(self) -> nn.Module:
        """Alias useful to callers that call the backbone an encoder."""

        return self.spatial_encoder

    def _dense_distance_edges(self, pos: Tensor, graph_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Small, dependency-free neighbor construction used by CPU tests."""

        edge_parts: list[Tensor] = []
        weight_parts: list[Tensor] = []
        for graph in torch.unique(graph_id, sorted=True).tolist():
            nodes = torch.nonzero(graph_id == graph, as_tuple=False).flatten()
            local_pos = pos.index_select(0, nodes)
            distances = torch.cdist(local_pos, local_pos)
            valid = (distances >= self.cutoff_lower) & (
                distances < self.cutoff_upper
            )
            # A self edge keeps a singleton/sparse synthetic graph valid even
            # when a caller configures a positive lower cutoff.
            valid.fill_diagonal_(True)
            k = min(self.max_num_neighbors, int(nodes.numel()))
            scores = distances.masked_fill(~valid, float("inf"))
            values, neighbors = torch.topk(scores, k=k, dim=1, largest=False)
            keep = torch.isfinite(values)
            target = nodes[:, None].expand_as(neighbors)[keep]
            source = nodes[neighbors][keep]
            edge_parts.append(torch.stack([source, target], dim=0))
            weight_parts.append(values[keep])
        if not edge_parts:
            empty_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            empty_weight = pos.new_empty((0,))
            empty_vec = pos.new_empty((0, 3))
            return empty_index, empty_weight, empty_vec
        edge_index = torch.cat(edge_parts, dim=1)
        edge_weight = torch.cat(weight_parts, dim=0).to(dtype=pos.dtype)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        return edge_index, edge_weight, edge_vec

    def _distance_edges(self, pos: Tensor, graph_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        backend = self.neighbor_backend
        if backend == "dense":
            return self._dense_distance_edges(pos, graph_id)
        finder = getattr(self.spatial_encoder, "distance", None)
        if finder is None:
            if backend == "optimized":
                raise ValueError(
                    "neighbor_backend='optimized' requires a spatial encoder "
                    "with a distance module"
                )
            return self._dense_distance_edges(pos, graph_id)
        try:
            edge_index, edge_weight, _ = finder(pos, batch=graph_id)
        except (ImportError, RuntimeError, NotImplementedError, NameError):
            if backend == "optimized":
                raise
            return self._dense_distance_edges(pos, graph_id)
        edge_index = edge_index.to(device=pos.device, dtype=torch.long)
        if edge_index.numel() == 0:
            return self._dense_distance_edges(pos, graph_id)
        if torch.any(edge_index < 0) or torch.any(edge_index >= pos.shape[0]):
            raise ValueError("neighbor finder returned an out-of-range edge")
        if torch.any(graph_id[edge_index[0]] != graph_id[edge_index[1]]):
            raise RuntimeError("neighbor finder created a cross-frame graph edge")
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_weight = torch.linalg.vector_norm(edge_vec, dim=-1)
        return edge_index, edge_weight, edge_vec

    @staticmethod
    def _replicate_bonds(
        bond_index: Tensor, *, frames: int, atoms: int
    ) -> Tensor:
        if bond_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=bond_index.device)
        offsets = torch.arange(frames, device=bond_index.device, dtype=torch.long)
        offsets = offsets[:, None, None] * atoms
        replicated = bond_index.unsqueeze(0) + offsets
        return replicated.permute(1, 0, 2).reshape(2, -1)

    @staticmethod
    def _union_edges(
        distance_edge_index: Tensor,
        bond_index: Tensor,
        *,
        pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if bond_index.numel() == 0:
            edge_index = distance_edge_index
        elif distance_edge_index.numel() == 0:
            edge_index = bond_index
        else:
            edge_index = torch.cat([distance_edge_index, bond_index], dim=1)
        if edge_index.numel() == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long, device=pos.device)
            return empty_index, pos.new_empty((0,)), pos.new_empty((0, 3))
        node_count = int(pos.shape[0])
        codes = edge_index[0] * node_count + edge_index[1]
        sorted_codes, order = torch.sort(codes)
        keep = torch.ones_like(sorted_codes, dtype=torch.bool)
        if keep.numel() > 1:
            keep[1:] = sorted_codes[1:] != sorted_codes[:-1]
        edge_index = edge_index[:, order[keep]]
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_weight = torch.linalg.vector_norm(edge_vec, dim=-1)
        return edge_index, edge_weight, edge_vec

    @staticmethod
    def _bond_flags(edge_index: Tensor, bond_index: Tensor, *, nodes: int) -> Tensor:
        if edge_index.numel() == 0 or bond_index.numel() == 0:
            return torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        edge_code = edge_index[0] * nodes + edge_index[1]
        bond_code = bond_index[0] * nodes + bond_index[1]
        return torch.isin(edge_code, bond_code).to(dtype=torch.long)

    def build_graph(self, batch: ClipBatch | Mapping[str, Any]) -> FrameGraphBatch:
        """Prepare one isolated graph per ``(sample, frame)`` pair."""

        x = _as_tensor(_get_field(batch, "x"), name="x")
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("x must have shape [T, N_total, 3]")
        frames, atoms = int(x.shape[0]), int(x.shape[1])
        if frames < 1 or atoms < 1:
            raise ValueError("x must contain at least one frame and one atom")

        atom_ptr = _as_tensor(
            _get_field(batch, "atom_ptr"), name="atom_ptr", dtype=torch.long
        ).flatten()
        if atom_ptr.ndim != 1 or atom_ptr.numel() < 2:
            raise ValueError("atom_ptr must have shape [B+1]")
        if int(atom_ptr[0]) != 0 or int(atom_ptr[-1]) != atoms:
            raise ValueError("atom_ptr must span the packed atom axis")
        if torch.any(atom_ptr[1:] <= atom_ptr[:-1]):
            raise ValueError("atom_ptr must contain strictly increasing sample offsets")
        batch_size = int(atom_ptr.numel() - 1)

        abid = _as_tensor(_get_field(batch, "abid"), name="abid", dtype=torch.long).flatten()
        if abid.numel() != atoms:
            raise ValueError("abid must have shape [N_total]")
        expected_abid = torch.repeat_interleave(
            torch.arange(batch_size, device=abid.device, dtype=torch.long),
            atom_ptr[1:] - atom_ptr[:-1],
        )
        if not torch.equal(abid, expected_abid):
            raise ValueError("abid must agree with atom_ptr and packed sample order")

        frame_mask = _as_tensor(
            _get_field(batch, "frame_mask"), name="frame_mask", dtype=torch.bool
        )
        if frame_mask.shape != (batch_size, frames):
            raise ValueError("frame_mask must have shape [B, T]")
        if not torch.any(frame_mask, dim=1).all():
            raise ValueError("every sample must contain at least one valid frame")

        pos = x.reshape(frames * atoms, 3)
        z = _expand_atom_field(
            _as_tensor(_get_field(batch, "atype"), name="atype", dtype=torch.long),
            frames=frames,
            atoms=atoms,
            name="atype",
        )
        b = _expand_atom_field(
            _as_tensor(_get_field(batch, "btype"), name="btype", dtype=torch.long),
            frames=frames,
            atoms=atoms,
            name="btype",
        )
        sample_id = abid.repeat(frames)
        frame_id = torch.arange(
            frames, device=pos.device, dtype=torch.long
        ).repeat_interleave(atoms)
        graph_id = frame_id * batch_size + sample_id

        # Validate topology before replication.  A bond crossing packed
        # samples would otherwise become a silent cross-sample message edge.
        bond_index = _normalise_bonds(
            _as_tensor(_get_field(batch, "bond_index"), name="bond_index", dtype=torch.long),
            atoms=atoms,
        )
        if bond_index.numel() and torch.any(
            abid[bond_index[0]] != abid[bond_index[1]]
        ):
            raise ValueError("bond_index contains a cross-sample bond")
        replicated_bonds = self._replicate_bonds(
            bond_index, frames=frames, atoms=atoms
        )

        distance_edges, _, _ = self._distance_edges(pos, graph_id)
        edge_index, edge_weight, edge_vec = self._union_edges(
            distance_edges, replicated_bonds, pos=pos
        )
        if edge_index.numel() and torch.any(
            graph_id[edge_index[0]] != graph_id[edge_index[1]]
        ):
            raise RuntimeError("constructed a cross-frame or cross-sample edge")
        bond_type = self._bond_flags(
            edge_index, replicated_bonds, nodes=int(pos.shape[0])
        )

        return FrameGraphBatch(
            pos=pos,
            z=z,
            b=b,
            batch=graph_id,
            graph_id=graph_id,
            frame_id=frame_id,
            sample_id=sample_id,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
            bond_type=bond_type,
            bond_index=replicated_bonds,
            distance_edge_index=distance_edges,
            distance_edge_weight=torch.linalg.vector_norm(
                pos[distance_edges[0]] - pos[distance_edges[1]], dim=-1
            )
            if distance_edges.numel()
            else pos.new_empty((0,)),
            distance_edge_vec=(pos[distance_edges[0]] - pos[distance_edges[1]])
            if distance_edges.numel()
            else pos.new_empty((0, 3)),
            frame_mask=frame_mask,
        )

    @staticmethod
    def _unpack_features(result: Any) -> tuple[Tensor, Tensor]:
        if isinstance(result, (tuple, list)):
            if len(result) < 2:
                raise ValueError("spatial encoder must return scalar and vector features")
            h, v = result[0], result[1]
        else:
            h = getattr(result, "h", getattr(result, "scalar", None))
            v = getattr(result, "v", getattr(result, "vector", None))
        if not isinstance(h, Tensor) or not isinstance(v, Tensor):
            raise ValueError("spatial encoder must return tensor scalar/vector features")
        return h, v

    def forward(self, batch: ClipBatch | Mapping[str, Any]) -> FrameEncoderOutput:
        graph = self.build_graph(batch)
        result = self.spatial_encoder(
            z=graph.z,
            b=graph.b,
            pos=graph.pos,
            batch=graph.batch,
            edge_index=graph.edge_index,
            edge_weight_t=graph.edge_weight,
            edge_vec_t=graph.edge_vec,
            bond_type=graph.bond_type,
        )
        h, v = self._unpack_features(result)
        expected_nodes = graph.pos.shape[0]
        if h.ndim != 2 or h.shape[0] != expected_nodes:
            raise ValueError(
                "spatial scalar output must have shape [T*N_total, C], "
                f"got {tuple(h.shape)}"
            )
        if v.ndim != 3 or v.shape[0] != expected_nodes or v.shape[1] != 3:
            raise ValueError(
                "spatial vector output must have shape [T*N_total, 3, C], "
                f"got {tuple(v.shape)}"
            )
        frames = int(_get_field(batch, "x").shape[0])
        atoms = int(_get_field(batch, "x").shape[1])
        return FrameEncoderOutput(
            h=h.reshape(frames, atoms, h.shape[-1]),
            v=v.reshape(frames, atoms, 3, v.shape[-1]),
            graph=graph,
        )

    encode = forward

    @staticmethod
    def _state_dict_from_checkpoint(payload: Any) -> Mapping[str, Tensor]:
        if isinstance(payload, Mapping):
            for key in ("state_dict", "model_state_dict", "model", "encoder"):
                candidate = payload.get(key)
                if isinstance(candidate, Mapping) and candidate:
                    payload = candidate
                    break
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint does not contain a state-dict mapping")
        state = {str(key): value for key, value in payload.items() if isinstance(value, Tensor)}
        if not state:
            raise ValueError("checkpoint state dict contains no tensor entries")
        return state

    @staticmethod
    def _candidate_keys(key: str) -> Sequence[str]:
        candidates = [key]
        prefixes = ("module.", "encoder.", "spatial_encoder.", "model.encoder.")
        changed = True
        while changed:
            changed = False
            for candidate in tuple(candidates):
                for prefix in prefixes:
                    if candidate.startswith(prefix):
                        stripped = candidate[len(prefix) :]
                        if stripped not in candidates:
                            candidates.append(stripped)
                            changed = True
        return candidates

    def load_checkpoint(self, path: str | Path) -> CheckpointLoadReport:
        """Load matching tensors and return a complete key compatibility report."""

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"encoder checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        source = self._state_dict_from_checkpoint(payload)
        target = self.spatial_encoder.state_dict()
        matched: list[str] = []
        missing = set(target)
        unexpected: list[str] = []
        shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        loadable: dict[str, Tensor] = {}
        for source_key, value in source.items():
            target_key = next(
                (candidate for candidate in self._candidate_keys(source_key) if candidate in target),
                None,
            )
            if target_key is None:
                unexpected.append(source_key)
                continue
            if tuple(value.shape) != tuple(target[target_key].shape):
                shape_mismatch.append(
                    (target_key, tuple(value.shape), tuple(target[target_key].shape))
                )
                continue
            loadable[target_key] = value
            matched.append(target_key)
            missing.discard(target_key)
        self.spatial_encoder.load_state_dict(loadable, strict=False)
        report = CheckpointLoadReport(
            matched=tuple(sorted(set(matched))),
            missing=tuple(sorted(missing)),
            unexpected=tuple(sorted(unexpected)),
            shape_mismatch=tuple(sorted(shape_mismatch)),
        )
        self.checkpoint_report = report
        return report


# Short aliases keep the adapter discoverable without duplicating the model.
FrameEncoder = PVBFrameEncoder
PVBFrameGraph = FrameGraphBatch

