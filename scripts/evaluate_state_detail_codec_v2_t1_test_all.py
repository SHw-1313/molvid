#!/usr/bin/env python3
"""Evaluate all four completed T1 controls on the frozen test split.

Validation-only selection must already be frozen.  This command opens the test
store only after that packet exists, evaluates every control from its own best
validation checkpoint, and never changes the validation ranking.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from data.clip_batching import TaskAwareClipBatchSampler, make_clip_dataloader
from data.clip_dataset import ClipMMapDataset, collate_clip_records
from scripts.run_state_detail_codec_v2_t0 import (
    RATIOS,
    _evaluate_loader,
    _make_config,
    _make_model,
    _scheduled_ids,
    _sha256,
    _write_json,
)
from scripts.run_state_detail_codec_v2_t1 import _load_manifest, _make_loaders
from scripts.select_state_detail_codec_v2_t1 import MODES
from trainer.codec_trainer import CodecTrainer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/manifest_20260904_token80000"
DEFAULT_FULL_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/full_20260904_seed20260903"
DEFAULT_SELECTION_ROOT = ROOT / "outputs/state_detail_codec_v2/t1/selection_20260904"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_frozen_selection(selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    content = dict(selection)
    recorded = content.pop("selection_content_sha256", None)
    if selection.get("status") != "FROZEN":
        raise RuntimeError("validation selection is not frozen")
    if recorded != _canonical_hash(content):
        raise RuntimeError("validation selection content hash is invalid")
    rule = selection.get("selection_rule", {})
    if rule.get("no_test_metrics_used") is not True:
        raise RuntimeError("selection rule does not prove validation-only ranking")
    ranked_modes = [str(row.get("mode")) for row in selection.get("ranking", [])]
    if sorted(ranked_modes) != sorted(MODES):
        raise RuntimeError("selection ranking does not contain exactly all four controls")
    return selection


def _best_checkpoint(result: Mapping[str, Any], full_root: Path, mode: str) -> Path:
    recorded = Path(str(result["best_checkpoint"]))
    candidates = (
        recorded,
        full_root / mode / recorded.name,
        full_root / mode / "codec_best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"best checkpoint for {mode} is not present; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _evaluate_all(
    *,
    manifest_root: Path,
    full_root: Path,
    selection_root: Path,
    modes: Sequence[str],
) -> dict[str, Any]:
    modes = tuple(modes)
    if not modes:
        raise ValueError("at least one test control is required")
    if len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError(f"invalid test controls: {modes}")
    selection_path = selection_root / "selection_rule.json"
    selection = _load_frozen_selection(selection_path)
    results = {
        mode: json.loads((full_root / mode / "result.json").read_text(encoding="utf-8"))
        for mode in modes
    }
    if any(result.get("status") != "passed" for result in results.values()):
        raise RuntimeError("all four T1 controls must have passed results")

    manifest, materialization = _load_manifest(manifest_root)
    seed = int(results[modes[0]]["seed"])
    train_dataset, valid_dataset, train_loader, valid_loader, epoch_batches = _make_loaders(
        manifest_root, seed=seed
    )
    test_dataset = ClipMMapDataset(manifest_root / "clip_store" / "test")
    expected_ids = tuple(manifest["source_splits"]["test"]["sample_ids"])
    actual_ids = tuple(str(row[0]) for row in test_dataset._index)
    if actual_ids != expected_ids:
        raise RuntimeError("test index differs from frozen manifest")
    test_sampler = TaskAwareClipBatchSampler(
        test_dataset,
        max_tokens=80000,
        seed=seed + 2,
        shuffle=False,
        replacement=False,
        oversize_policy="error",
    )
    test_loader = make_clip_dataloader(
        test_dataset,
        sampler=test_sampler,
        collate_fn=collate_clip_records,
        num_workers=0,
        pin_memory=False,
    )
    if _scheduled_ids(test_loader, test_dataset, 0) != list(expected_ids):
        raise RuntimeError("test sampler is not an exact no-replacement pass")

    evaluations: dict[str, Any] = {}
    try:
        for mode in modes:
            result = results[mode]
            checkpoint = _best_checkpoint(result, full_root, mode)
            model = _make_model(mode, freeze_frame_encoder=True)
            trainer = CodecTrainer(
                model,
                train_loader,
                valid_loader,
                config=_make_config(
                    int(result["total_steps"]),
                    train_batches=int(epoch_batches[0]),
                    schedule_steps=int(result["total_steps"]),
                ),
                device="cuda:0",
                non_blocking_transfer=False,
            )
            trainer.load_checkpoint(checkpoint)
            evaluation = _evaluate_loader(
                trainer,
                test_loader,
                test_dataset,
                epoch=0,
                ratio=RATIOS[mode],
                detailed=True,
            )
            evaluations[mode] = {
                "mode": mode,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "sample_count": evaluation["sample_count"],
                "exact_coverage": evaluation["exact_coverage"],
                "evaluation": evaluation,
            }
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        train_dataset.close()
        valid_dataset.close()
        test_dataset.close()

    packet = {
        "schema_version": "pvb.codec.state_detail.t1_test_evaluations.v1",
        "status": "OPENED_AFTER_VALIDATION_SELECTION",
        "selection_rule_sha256": _sha256(selection_path),
        "selection_content_sha256": selection["selection_content_sha256"],
        "selection_test_opened_before_this_packet": bool(selection.get("test_opened")),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "materialization_sha256": materialization["materialization_sha256"],
        "controls": list(modes),
        "control_count": len(evaluations),
        "sample_count": len(expected_ids),
        "exact_coverage": all(
            bool(item["exact_coverage"]) and int(item["sample_count"]) == len(expected_ids)
            for item in evaluations.values()
        ),
        "evaluations": evaluations,
    }
    output_name = "test_evaluations_all.json" if set(modes) == set(MODES) else "test_evaluations_partial.json"
    output_path = selection_root / output_name
    _write_json(output_path, packet)
    packet["output_path"] = str(output_path)
    packet["output_sha256"] = _sha256(output_path)
    print(json.dumps(packet, indent=2, sort_keys=True))
    return packet


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args(argv)
    manifest_root = args.manifest_root if args.manifest_root.is_absolute() else ROOT / args.manifest_root
    full_root = args.full_root if args.full_root.is_absolute() else ROOT / args.full_root
    selection_root = args.selection_root if args.selection_root.is_absolute() else ROOT / args.selection_root
    if not torch.cuda.is_available():
        raise RuntimeError("T1 test evaluation requires CUDA")
    _evaluate_all(
        manifest_root=manifest_root,
        full_root=full_root,
        selection_root=selection_root,
        modes=args.modes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
