#!/usr/bin/env python3
"""Evaluate selected ATLAS overfit checkpoints with PVB-style metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.clip_dataset import collate_clip_records
from eval_codec import _loss_evaluator
from evaluation.codec_evaluation import evaluate_controls, model_control, write_report
from scripts.run_selected_three_system_overfit import (
    CONTROLS,
    DEVICE,
    ROOT,
    make_distance_references,
    make_model,
    require_cuda,
    select_records,
)
from trainer.codec_contract import require_contract_equal
from trainer.codec_trainer import (
    CODEC_CHECKPOINT_SCHEMA,
    LEGACY_CODEC_CHECKPOINT_SCHEMA,
    CodecTrainConfig,
    PVBCodecModel,
    prepare_batch_then_to_device,
)


DEFAULT_RUN_DIRS = {
    "topology": (
        ROOT
        / "outputs/atlas_selected_trajectories/overfit_three_systems"
        / "run_20260826T094444"
    ),
    "distance_only": (
        ROOT
        / "outputs/atlas_selected_trajectories/overfit_three_systems_distance_only"
        / "run_20260826T105636"
    ),
}


def _checkpoint_payload(
    path: Path,
    *,
    allow_legacy: bool,
    legacy_bond_mode: str | None,
    bond_mode: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    schema = str(payload.get("schema_version", ""))
    if schema == CODEC_CHECKPOINT_SCHEMA:
        required = {
            "step",
            "model_state",
            "normalization_stats",
            "config",
            "model_contract",
            "distance_reference_contract",
            "optimizer_contract",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"checkpoint is missing fields {sorted(missing)}: {path}")
    elif schema == LEGACY_CODEC_CHECKPOINT_SCHEMA:
        if not allow_legacy or legacy_bond_mode is None:
            raise ValueError(
                f"legacy checkpoint v1 requires --allow-legacy-checkpoint and "
                f"--legacy-bond-mode={bond_mode!r}: {path}"
            )
        if legacy_bond_mode != bond_mode:
            raise ValueError("legacy graph mode conflicts with --bond-mode")
        required = {"step", "model_state", "normalization_stats", "config"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"checkpoint is missing fields {sorted(missing)}: {path}")
    else:
        raise ValueError(
            f"unsupported checkpoint schema {schema!r}; expected {CODEC_CHECKPOINT_SCHEMA!r}"
        )
    return payload


def _predictor(model: Any, device: torch.device):
    def predict(batch):
        moved = prepare_batch_then_to_device(model, batch, device)
        with torch.no_grad():
            output = model(moved)
        return output.x_hat.detach().cpu()

    return predict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bond-mode",
        choices=("topology", "distance_only"),
        default="topology",
        help="graph bond construction mode used by the checkpoint",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--legacy-bond-mode", choices=("topology", "distance_only"), default=None)
    args = parser.parse_args()

    runtime = require_cuda()
    bond_mode = str(args.bond_mode)
    if args.legacy_bond_mode is not None and args.legacy_bond_mode != bond_mode:
        raise ValueError("--legacy-bond-mode conflicts with --bond-mode")
    run_dir = args.run_dir or DEFAULT_RUN_DIRS[bond_mode]
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"overfit run directory is missing: {run_dir}")
    json_path = args.json or (run_dir / "codec_eval.json")
    markdown_path = args.markdown or (run_dir / "codec_eval.md")

    records, data_contract = select_records()
    references = make_distance_references(records) if bond_mode == "distance_only" else None
    batches = [collate_clip_records([record]) for record in records]
    if len(batches) != len(CONTROLS):
        raise RuntimeError("evaluation expects one batch per selected system")

    controls = []
    for name, control in CONTROLS.items():
        checkpoint = run_dir / name / "codec_step_00000500.pt"
        payload = _checkpoint_payload(
            checkpoint,
            allow_legacy=args.allow_legacy_checkpoint,
            legacy_bond_mode=args.legacy_bond_mode,
            bond_mode=bond_mode,
        )
        expected = make_model(control, bond_mode=bond_mode)
        if payload["schema_version"] == CODEC_CHECKPOINT_SCHEMA:
            require_contract_equal(
                payload["model_contract"],
                expected.model_contract(),
                label=f"{name} overfit model contract",
            )
            model = PVBCodecModel.from_model_contract(payload["model_contract"])
        else:
            model = expected
        model = model.to(DEVICE).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        if references is not None:
            model.prepare_distance_bonds(references, device=DEVICE)
            if payload["schema_version"] == CODEC_CHECKPOINT_SCHEMA:
                require_contract_equal(
                    payload["distance_reference_contract"],
                    model.distance_reference_contract(),
                    label=f"{name} overfit distance reference contract",
                )
        config = CodecTrainConfig.from_mapping(payload["config"])
        loss_evaluator = _loss_evaluator(
            DEVICE,
            config,
            payload["normalization_stats"],
            int(payload["step"]),
        )
        controls.append(
            model_control(
                name,
                _predictor(model, DEVICE),
                ratio=int(control["temporal_ratio"]),
                temporal=bool(control["temporal_layers"]),
                loss_evaluator=loss_evaluator,
            )
        )

    report = evaluate_controls(controls, batches, device=DEVICE)
    report["evaluation_data"] = {
        "runtime": runtime,
        "source_split": data_contract["source_split"],
        "sample_ids": data_contract["sample_ids"],
        "systems": data_contract["systems"],
        "frames": data_contract["frames"],
        "packed_atoms": data_contract["packed_atoms"],
        "effective_tokens": data_contract["effective_tokens"],
        "batch_policy": "three one-clip batches; metrics are the unweighted mean over the three selected systems",
        "validation_samples_used": False,
        "bond_construction_mode": bond_mode,
        "distance_reference_policy": (
            "first frame of each selected train clip; supplied topology is not used "
            "for inference graph construction"
            if bond_mode == "distance_only" else None
        ),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, json_path, markdown_path)
    torch.cuda.synchronize()
    print(json.dumps({
        "bond_mode": bond_mode,
        "json": str(json_path),
        "markdown": str(markdown_path),
        "controls": list(report["controls"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
