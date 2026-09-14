"""Read-only selected views for the Session B capacity/data manifest.

The materializer publishes a small index view and symlinks the approved source
``data.bin``.  This module owns the local index remapping; it never opens the
sealed test store and never copies trajectory payloads into the worktree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.clip_batching import ClipItemSpec, ClipSpecTable
from data.clip_dataset import ClipMMapDataset, ClipBatch, collate_clip_records


SCHEMA = "pvb.dit.capacity_data.v1"


class BlockedDataError(RuntimeError):
    """Raised when a requested frozen data scale is unavailable on this machine."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SelectedClipDataset(torch.utils.data.Dataset):
    """A local-index view over one approved ``ClipMMapDataset``.

    The source mmap remains read-only.  ``_index`` and ``clip_spec_table`` are
    remapped so existing deterministic samplers can treat the view as a normal
    packed clip store.
    """

    def __init__(self, source_root: str | Path, sample_ids: Sequence[str]) -> None:
        self.source_root = Path(source_root).resolve()
        self.source = ClipMMapDataset(self.source_root)
        positions = {str(row[0]): index for index, row in enumerate(self.source._index)}
        missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in positions]
        if missing:
            self.source.close()
            raise RuntimeError(f"selected sample IDs are missing from {self.source_root}: {missing[:8]}")
        self.sample_ids = tuple(str(value) for value in sample_ids)
        self._source_indices = tuple(positions[value] for value in self.sample_ids)
        self._index = [self.source._index[index] for index in self._source_indices]

    def __len__(self) -> int:
        return len(self._source_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self.source[self._source_indices[index]]
        if str(record.get("sample_id")) != self.sample_ids[index]:
            raise RuntimeError("selected view source record identity changed")
        return record

    def clip_spec_table(self) -> ClipSpecTable:
        source_specs = self.source.clip_spec_table()
        specs = []
        for local_index, source_index in enumerate(self._source_indices):
            item = source_specs[source_index]
            specs.append(
                ClipItemSpec(
                    index=local_index,
                    atoms=item.atoms,
                    frames=item.frames,
                    task=item.task,
                    time_bucket_id=item.time_bucket_id,
                    native_delta_time_ps=item.native_delta_time_ps,
                    physical_clip_span_ps=item.physical_clip_span_ps,
                    sample_id=self.sample_ids[local_index],
                )
            )
        return ClipSpecTable.from_specs(specs)

    @staticmethod
    def collate_fn(records: Sequence[Mapping[str, Any]]) -> ClipBatch:
        return collate_clip_records(records)

    def close(self) -> None:
        self.source.close()


@dataclass(frozen=True)
class CapacityData:
    manifest_root: Path
    manifest: Mapping[str, Any]
    materialization: Mapping[str, Any]
    train48: SelectedClipDataset
    train192: SelectedClipDataset | None
    valid: SelectedClipDataset
    data_hash: str
    source_index_hashes: Mapping[str, str]
    source_roots: Mapping[str, str]
    blocked_data: Mapping[str, str]

    def train_for_scale(self, scale: str) -> SelectedClipDataset:
        if str(scale) == "base48":
            return self.train48
        if str(scale) == "expanded192":
            if self.train192 is None:
                raise BlockedDataError(
                    self.blocked_data.get("expanded192", "expanded192 is unavailable")
                )
            return self.train192
        raise ValueError(f"unsupported capacity data scale: {scale!r}")

    def close(self) -> None:
        seen: set[int] = set()
        for dataset in (self.train48, self.train192, self.valid):
            if dataset is not None and id(dataset) not in seen:
                seen.add(id(dataset))
                dataset.close()


def _validate_manifest(
    manifest_root: Path,
    *,
    allow_missing_expanded: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    manifest_root = manifest_root.resolve()
    manifest_path = manifest_root / "manifest.json"
    materialization_path = manifest_root / "materialization.json"
    if not manifest_path.is_file() or not materialization_path.is_file():
        raise FileNotFoundError(f"capacity manifest/materialization missing under {manifest_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = dict(manifest)
    recorded = content.pop("manifest_content_sha256", None)
    if recorded != _canonical_hash(content):
        raise RuntimeError("capacity manifest content hash is invalid")
    if manifest.get("schema_version") != "pvb.dit.capacity_data.manifest.v1":
        raise RuntimeError("unsupported capacity manifest schema")
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("capacity manifest is not FROZEN")
    if manifest.get("test_sampling", {}).get("opened") is not False:
        raise RuntimeError("capacity manifest does not certify unopened test sampling")
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if materialization.get("schema_version") != "pvb.dit.capacity_data.materialization.v1":
        raise RuntimeError("unsupported capacity materialization schema")
    if materialization.get("test_opened") is not False:
        raise RuntimeError("capacity materialization opened test data")
    blocked_data: dict[str, str] = {}
    for view in ("train48", "train192", "valid"):
        record = materialization.get("views", {}).get(view, {})
        root = manifest_root / str(record.get("relative_root", ""))
        if not root.is_dir() or not (root / "index.txt").is_file():
            raise FileNotFoundError(f"capacity materialized view is missing: {root}")
        if not (root / "data.bin").is_file():
            if view == "train192" and allow_missing_expanded:
                blocked_data["expanded192"] = (
                    f"expanded192 payload is unavailable on this machine: {root / 'data.bin'}"
                )
            else:
                raise FileNotFoundError(f"capacity materialized view payload is missing: {root}")
        if int(record.get("count", -1)) != len(manifest["views"][view]["sample_ids"]):
            raise RuntimeError(f"materialization count differs for {view}")
        if _sha256(root / "index.txt") != str(record.get("index_sha256")):
            raise RuntimeError(f"materialized index hash differs for {view}")
    return manifest, materialization, blocked_data


def load_capacity_data(
    manifest_root: str | Path,
    *,
    source_root_overrides: Mapping[str, str | Path] | None = None,
    allow_missing_expanded: bool = False,
) -> CapacityData:
    root = Path(manifest_root).resolve()
    manifest, materialization, blocked_data = _validate_manifest(
        root,
        allow_missing_expanded=bool(allow_missing_expanded),
    )
    source_roots = {label: str(path) for label, path in manifest["source_views"].items()}
    overrides = dict(source_root_overrides or {})
    unexpected = sorted(set(overrides).difference(("train", "valid")))
    if unexpected:
        raise ValueError(f"unsupported source-root overrides: {unexpected}")
    source_roots.update({label: str(Path(path).resolve()) for label, path in overrides.items()})
    for label in ("train", "valid"):
        source_root = Path(source_roots[label]).resolve()
        if "test" in {part.lower() for part in source_root.parts}:
            raise RuntimeError(f"source {label} path unexpectedly references test: {source_root}")
        if not source_root.is_dir():
            raise FileNotFoundError(f"source {label} store is missing: {source_root}")
    train48 = SelectedClipDataset(source_roots["train"], manifest["views"]["train48"]["sample_ids"])
    valid = SelectedClipDataset(source_roots["valid"], manifest["views"]["valid"]["sample_ids"])
    train_positions = {str(row[0]) for row in train48.source._index}
    missing_expanded = [
        str(sample_id)
        for sample_id in manifest["views"]["train192"]["sample_ids"]
        if str(sample_id) not in train_positions
    ]
    train192: SelectedClipDataset | None
    if blocked_data.get("expanded192") or missing_expanded:
        if not allow_missing_expanded:
            train48.close()
            valid.close()
            raise RuntimeError(
                f"expanded192 source is incomplete: {len(missing_expanded)} selected clips are missing"
            )
        train192 = None
        if missing_expanded:
            blocked_data["expanded192"] = (
                f"expanded192 source is incomplete on this machine: "
                f"{len(missing_expanded)} selected clips are missing"
            )
    else:
        train192 = SelectedClipDataset(
            source_roots["train"], manifest["views"]["train192"]["sample_ids"]
        )
    source_index_hashes = {
        label: _sha256(Path(source_roots[label]) / "index.txt") for label in ("train", "valid")
    }
    data_contract = {
        "schema": SCHEMA,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "materialization_sha256": materialization["materialization_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "views": {
            name: {
                "sample_ids_sha256": _canonical_hash(manifest["views"][name]["sample_ids"]),
                "count": len(manifest["views"][name]["sample_ids"]),
            }
            for name in ("train48", "train192", "valid")
        },
        "test_opened": False,
    }
    return CapacityData(
        manifest_root=root,
        manifest=manifest,
        materialization=materialization,
        train48=train48,
        train192=train192,
        valid=valid,
        data_hash=_canonical_hash(data_contract),
        source_index_hashes=source_index_hashes,
        source_roots={label: str(Path(path).resolve()) for label, path in source_roots.items()},
        blocked_data=blocked_data,
    )


__all__ = [
    "BlockedDataError",
    "CapacityData",
    "SCHEMA",
    "SelectedClipDataset",
    "load_capacity_data",
]
