#!/usr/bin/env python3
"""Freeze nested 48->192 Atlas train views without copying trajectory payloads.

Only train/validation indexes and source metadata are read.  The sealed test
store is never opened or hashed.  Each output view owns its deterministic
index and symlinks the approved source ``data.bin`` read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps")
DEFAULT_BASE_MANIFEST = Path(
    "/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/"
    "manifest_20260904_token80000"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/dit_capacity_data_v1/data_manifest_20260914"
MAX_TOKENS = 80000
REPLICAS = ("R1", "R2", "R3")
WINDOWS = tuple(range(62))
BASE_TRAIN_COUNT = 48
ADDED_TRAIN_COUNT = 144
SELECTION_SEED = 20260914
SELECTION_NAMESPACE = "pvb-dit-capacity-data-v1-system-selection"


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


def _split_sample_id(sample_id: str) -> tuple[str, str, int]:
    prefix, window_text = str(sample_id).rsplit("_w", 1)
    system, replica = prefix.rsplit("_", 1)
    if replica not in REPLICAS or not window_text.isdigit():
        raise ValueError(f"unsupported sample id: {sample_id!r}")
    return system, replica, int(window_text)


def _read_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                raise ValueError(f"invalid index row {path}:{line_number}")
            sample_id, start, end, atoms, frames, bucket = fields
            if int(frames) != 16 or bucket != "dt_100ps":
                raise ValueError(f"non-100-ps/non-16-frame row in {path}:{line_number}")
            system, replica, window = _split_sample_id(sample_id)
            rows.append({
                "sample_id": sample_id,
                "start": int(start),
                "end": int(end),
                "atoms": int(atoms),
                "frames": int(frames),
                "time_bucket_id": bucket,
                "system": system,
                "replica": replica,
                "window": window,
            })
    return rows


def _inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[int, Mapping[str, Any]]]]:
    result: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for row in rows:
        system = str(row["system"])
        replica = str(row["replica"])
        window = int(row["window"])
        result.setdefault(system, {}).setdefault(replica, {})[window] = row
    for system, replicas in result.items():
        if set(replicas) != set(REPLICAS):
            raise RuntimeError(f"{system} does not have R1/R2/R3")
        for replica, windows in replicas.items():
            if set(windows) != set(WINDOWS):
                raise RuntimeError(f"{system}_{replica} does not have windows 0..61")
    return result


def _rank(system: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_NAMESPACE}|seed={SELECTION_SEED}|split=train|system={system}".encode()
    ).hexdigest()


def _ordered_ids(systems: Sequence[str]) -> list[str]:
    return [
        f"{system}_{replica}_w{window:06d}"
        for system in sorted(systems)
        for replica in REPLICAS
        for window in WINDOWS
    ]


def _link_view(
    output_root: Path,
    name: str,
    source_root: Path,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    view_root = output_root / "clip_store" / name
    view_root.mkdir(parents=True, exist_ok=True)
    data_link = view_root / "data.bin"
    if data_link.exists() or data_link.is_symlink():
        if data_link.is_symlink() and data_link.resolve() == (source_root / "data.bin").resolve():
            pass
        else:
            raise FileExistsError(f"refusing to replace existing data view: {data_link}")
    else:
        data_link.symlink_to((source_root / "data.bin").resolve())
    index_path = view_root / "index.txt"
    index_text = "".join(
        "{sample_id}\t{start}\t{end}\t{atoms}\t{frames}\t{time_bucket_id}\n".format(**rows_by_id[sample_id])
        for sample_id in sample_ids
    )
    if index_path.exists():
        if index_path.read_text(encoding="utf-8") != index_text:
            raise FileExistsError(f"existing capacity index differs: {index_path}")
    else:
        index_path.write_text(index_text, encoding="utf-8")
    stats_path = view_root / "stats.json"
    stats = {
        "count": len(sample_ids),
        "compressed_bytes": int((source_root / "data.bin").stat().st_size),
        "storage_format": "npz-v1",
        "payload_mode": "read_only_symlink_to_approved_source_data_bin",
        "source_root": str(source_root.resolve()),
    }
    if stats_path.exists():
        if json.loads(stats_path.read_text(encoding="utf-8")) != stats:
            raise FileExistsError(f"existing capacity stats differ: {stats_path}")
    else:
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "relative_root": str(view_root.relative_to(output_root)),
        "source_root": str(source_root.resolve()),
        "payload_data_bin": str(data_link.resolve()),
        "count": len(sample_ids),
        "index_sha256": _sha256(index_path),
        "stats_sha256": _sha256(stats_path),
        "data_bin_size_bytes": int(data_link.stat().st_size),
        "payload_copied": False,
    }


def build_manifest(source_root: Path, base_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    base_root = base_root.resolve()
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    base_manifest = json.loads((base_root / "manifest.json").read_text(encoding="utf-8"))
    source_index_path = source_root / "train" / "index.txt"
    valid_index_path = source_root / "valid" / "index.txt"
    train_rows = _read_index(source_index_path)
    valid_rows = _read_index(valid_index_path)
    train_inventory = _inventory(train_rows)
    valid_inventory = _inventory(valid_rows)
    base_systems = tuple(sorted(str(value) for value in base_manifest["source_splits"]["train"]["selected_systems"]))
    valid_systems = tuple(sorted(str(value) for value in base_manifest["source_splits"]["valid"]["selected_systems"]))
    holdout_test_systems = tuple(sorted(str(value) for value in base_manifest["source_splits"]["test"]["selected_systems"]))
    if len(base_systems) != BASE_TRAIN_COUNT or len(valid_systems) != 8 or len(holdout_test_systems) != 8:
        raise RuntimeError("frozen base manifest does not have the expected 48/8/8 systems")
    eligible = {
        system
        for system, replicas in train_inventory.items()
        if max(int(row["atoms"]) * 16 for windows in replicas.values() for row in windows.values()) <= MAX_TOKENS
    }
    if not set(base_systems).issubset(eligible):
        raise RuntimeError("one or more original train systems are not eligible in the approved source pool")
    candidates = sorted(
        eligible.difference(base_systems).difference(valid_systems).difference(holdout_test_systems),
        key=lambda system: (_rank(system), system),
    )
    if len(candidates) < ADDED_TRAIN_COUNT:
        raise RuntimeError(f"only {len(candidates)} eligible additional systems; need {ADDED_TRAIN_COUNT}")
    added_systems = tuple(candidates[:ADDED_TRAIN_COUNT])
    expanded_systems = tuple(sorted(set(base_systems).union(added_systems)))
    if len(expanded_systems) != BASE_TRAIN_COUNT + ADDED_TRAIN_COUNT:
        raise RuntimeError("expanded train systems are not a 48+144 nested set")
    train_ids = {
        "train48": _ordered_ids(base_systems),
        "train192": _ordered_ids(expanded_systems),
    }
    valid_ids = _ordered_ids(valid_systems)
    train_by_id = {str(row["sample_id"]): row for row in train_rows}
    valid_by_id = {str(row["sample_id"]): row for row in valid_rows}
    for name, ids in train_ids.items():
        if any(sample_id not in train_by_id for sample_id in ids):
            raise RuntimeError(f"{name} contains a sample not present in source train index")
    if any(sample_id not in valid_by_id for sample_id in valid_ids):
        raise RuntimeError("valid view contains a sample not present in source valid index")
    source_timing = {
        "source_native_delta_time_ps": source_manifest.get("native_delta_time_ps"),
        "source_stride": source_manifest.get("source_stride"),
        "effective_delta_time_ps": float(source_manifest.get("native_delta_time_ps", 0.0)) * int(source_manifest.get("source_stride", 0)),
        "window_stride_frames": source_manifest.get("window_stride"),
        "window_count_per_replica": len(WINDOWS),
        "clip_frames": 16,
        "nominal_raw_duration_ps": 100000.0,
        "nominal_raw_frame_count": 10001,
    }
    all_selected_ids = train_ids["train192"] + valid_ids
    distributions = {}
    for name, systems in (("train48", base_systems), ("added144", added_systems), ("train192", expanded_systems), ("valid8", valid_systems)):
        inventory = train_inventory if name != "valid8" else valid_inventory
        atom_counts = [
            int(row["atoms"])
            for system in systems
            for replicas in inventory[system].values()
            for row in replicas.values()
        ]
        distributions[name] = {
            "system_count": len(systems),
            "min_atoms_per_clip": min(atom_counts),
            "mean_atoms_per_clip": sum(atom_counts) / len(atom_counts),
            "max_atoms_per_clip": max(atom_counts),
            "clip_count": len(systems) * len(REPLICAS) * len(WINDOWS),
            "clip_frame_observations": len(systems) * len(REPLICAS) * len(WINDOWS) * 16,
            "nonpadding_atom_frame_tokens": sum(atom_counts) * 16,
            "overlapping_windows_are_not_new_physical_frames": True,
        }
    manifest = {
        "schema_version": "pvb.dit.capacity_data.manifest.v1",
        "status": "FROZEN",
        "phase": "dit_capacity_data_v1",
        "selection_seed": SELECTION_SEED,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_algorithm": "sha256-ranked eligible source-train systems excluding frozen train/valid/test",
        "source_root": str(source_root),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_manifest_system_count": source_manifest.get("counts", {}).get("systems_selected"),
        "source_splits": {"train": str(source_root / "train"), "valid": str(source_root / "valid")},
        "source_views": {"train": str(source_root / "train"), "valid": str(source_root / "valid")},
        "base_manifest_root": str(base_root),
        "base_manifest_sha256": _sha256(base_root / "manifest.json"),
        "base_materialization_sha256": _sha256(base_root / "materialization.json"),
        "base_train_systems": list(base_systems),
        "added_train_systems": list(added_systems),
        "valid_systems": list(valid_systems),
        "test_systems": list(holdout_test_systems),
        "views": {
            "train48": {"systems": list(base_systems), "sample_ids": train_ids["train48"]},
            "train192": {"systems": list(expanded_systems), "sample_ids": train_ids["train192"]},
            "valid": {"systems": list(valid_systems), "sample_ids": valid_ids},
        },
        "frames_per_clip": 16,
        "time_bucket_id": "dt_100ps",
        "native_delta_time_ps": 100.0,
        "timing": source_timing,
        "temperature": {"status": "not_recorded_in_approved_clip_manifest"},
        "homology_audit": {
            "status": "not_available",
            "claim": "system_disjoint_only",
            "sequence_chain_grouping_checked": False,
            "standard": "existing source system split only; no relaxed deduplication",
        },
        "distributions": distributions,
        "test_sampling": {
            "opened": False,
            "coordinates_read": False,
            "payload_hashed": False,
            "holdout_system_count_recorded": len(holdout_test_systems),
        },
        "counts": {
            "base_train_systems": len(base_systems),
            "added_train_systems": len(added_systems),
            "expanded_train_systems": len(expanded_systems),
            "valid_systems": len(valid_systems),
            "test_systems": len(holdout_test_systems),
            "train48_clips": len(train_ids["train48"]),
            "train192_clips": len(train_ids["train192"]),
            "valid_clips": len(valid_ids),
        },
        "physics_contract": {
            "accepted_source": "approved Atlas dynamic pool",
            "accepted_effective_dt_ps": 100.0,
            "interpolation": False,
            "time_label_rewrite": False,
            "static_mixing": False,
        },
        "materialization_notes": "Selected index views symlink source data.bin read-only; no payload copied.",
    }
    manifest["manifest_content_sha256"] = _canonical_hash(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    base_root = args.base_manifest.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir() or not base_root.is_dir():
        raise FileNotFoundError("source or frozen base manifest is missing")
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = dict(manifest)
        recorded = content.pop("manifest_content_sha256", None)
        if recorded != _canonical_hash(content):
            raise RuntimeError("existing capacity manifest hash is invalid")
        if Path(manifest["source_root"]).resolve() != source_root:
            raise RuntimeError("existing capacity manifest source root differs")
    else:
        manifest = build_manifest(source_root, base_root, output_root)
        output_root.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_rows = {str(row["sample_id"]): row for row in _read_index(source_root / "train" / "index.txt")}
    valid_rows = {str(row["sample_id"]): row for row in _read_index(source_root / "valid" / "index.txt")}
    views = {
        "train48": _link_view(output_root, "train48", source_root / "train", train_rows, manifest["views"]["train48"]["sample_ids"]),
        "train192": _link_view(output_root, "train192", source_root / "train", train_rows, manifest["views"]["train192"]["sample_ids"]),
        "valid": _link_view(output_root, "valid", source_root / "valid", valid_rows, manifest["views"]["valid"]["sample_ids"]),
    }
    materialization = {
        "schema_version": "pvb.dit.capacity_data.materialization.v1",
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "materialized": True,
        "payload_copied": False,
        "test_opened": False,
        "views": views,
    }
    materialization["materialization_sha256"] = _canonical_hash(materialization)
    materialization_path = output_root / "materialization.json"
    if materialization_path.exists():
        existing = json.loads(materialization_path.read_text(encoding="utf-8"))
        if existing != materialization:
            raise FileExistsError(f"existing capacity materialization differs: {materialization_path}")
    else:
        materialization_path.write_text(json.dumps(materialization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "materialization": materialization}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
