#!/usr/bin/env python3
"""Merge the selected-control and missing-control T1 test packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "ratio1_state_detail",
    "ratio2_state_detail",
    "ratio4_state_detail",
    "ratio4_matched_pooling",
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=ROOT / "outputs/state_detail_codec_v2/t1/selection_20260904",
    )
    args = parser.parse_args(argv)
    root = args.selection_root if args.selection_root.is_absolute() else ROOT / args.selection_root
    selection = json.loads((root / "selection_rule.json").read_text(encoding="utf-8"))
    selected = json.loads((root / "test_evaluation.json").read_text(encoding="utf-8"))
    partial = json.loads((root / "test_evaluations_partial.json").read_text(encoding="utf-8"))
    if selected.get("status") != "OPENED_AFTER_VALIDATION_SELECTION":
        raise RuntimeError("selected-control test packet is not post-selection")
    if partial.get("status") != "OPENED_AFTER_VALIDATION_SELECTION":
        raise RuntimeError("partial test packet is not post-selection")
    if int(selected["sample_count"]) != 1488 or not selected["exact_coverage"]:
        raise RuntimeError("R1 test packet does not cover the frozen 8-system test split")
    if int(partial["sample_count"]) != 1488 or not partial["exact_coverage"]:
        raise RuntimeError("partial test packet does not cover the frozen 8-system test split")
    if sorted(partial.get("controls", [])) != sorted(MODES[1:]):
        raise RuntimeError("partial packet does not contain exactly R2, R4-SD, and matched")
    evaluations = {
        MODES[0]: {
            "mode": MODES[0],
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "sample_count": selected["sample_count"],
            "exact_coverage": selected["exact_coverage"],
            "evaluation": selected["evaluation"],
        }
    }
    evaluations.update(partial["evaluations"])
    if sorted(evaluations) != sorted(MODES):
        raise RuntimeError("merged test packet does not contain exactly four controls")
    manifest_hashes = {
        selected["manifest_content_sha256"],
        partial["manifest_content_sha256"],
    }
    materialization_hashes = {
        selected["materialization_sha256"],
        partial["materialization_sha256"],
    }
    if len(manifest_hashes) != 1 or len(materialization_hashes) != 1:
        raise RuntimeError("selected and partial packets use different frozen stores")
    packet = {
        "schema_version": "pvb.codec.state_detail.t1_test_evaluations_all.v1",
        "status": "OPENED_AFTER_VALIDATION_SELECTION",
        "selection_rule_sha256": selected["selection_rule_sha256"],
        "selection_content_sha256": selection["selection_content_sha256"],
        "manifest_content_sha256": selected["manifest_content_sha256"],
        "materialization_sha256": selected["materialization_sha256"],
        "controls": list(MODES),
        "control_count": len(MODES),
        "sample_count": 1488,
        "exact_coverage": all(
            int(value["sample_count"]) == 1488 and bool(value["exact_coverage"])
            for value in evaluations.values()
        ),
        "evaluations": evaluations,
        "source_packets": {
            "selected_control": str(root / "test_evaluation.json"),
            "missing_controls": str(root / "test_evaluations_partial.json"),
        },
    }
    output = root / "test_evaluations_all.json"
    _write_json(output, packet)
    packet["output_path"] = str(output)
    packet["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    selection["test_evaluations_all"] = str(output)
    selection["test_evaluations_all_sha256"] = packet["output_sha256"]
    selection["selection_content_sha256"] = _canonical_hash(
        {key: value for key, value in selection.items() if key != "selection_content_sha256"}
    )
    _write_json(root / "selection_rule.json", selection)
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
