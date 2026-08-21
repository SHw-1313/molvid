from __future__ import annotations

import json
from pathlib import Path

import torch

from data.clip_dataset import collate_clip_records
from evaluation.codec_evaluation import (
    anchor_control,
    evaluate_controls,
    model_control,
    report_markdown,
    write_report,
)
from tests.test_codec_training import _record


def _trajectory(bucket: str, delta: float):
    record = _record(times=torch.arange(4, dtype=torch.float32) * delta)
    record["time_bucket_id"] = bucket
    return collate_clip_records([record])


def test_controls_are_separated_by_bucket_and_frame0():
    batch_100 = _trajectory("dt_100ps", 100.0)
    batch_80 = _trajectory("dt_80ps", 80.0)
    static = collate_clip_records([_record(task="static", times=torch.tensor([0.0]))])
    perfect = model_control(
        "ratio4_temporal",
        lambda batch: batch.x.clone(),
        ratio=4,
        temporal=True,
    )
    report = evaluate_controls(
        [anchor_control(), perfect], [batch_100, batch_80, static]
    )
    assert set(report["controls"]) == {"ratio1_no_temporal", "ratio4_temporal"}
    assert "overall" not in report["controls"]["ratio4_temporal"]
    ratio4 = report["controls"]["ratio4_temporal"]
    assert set(ratio4["by_time_bucket"]) == {"dt_100ps", "dt_80ps", "static"}
    assert ratio4["by_time_bucket"]["dt_80ps"]["latent_interval_ps"] == 320.0
    assert ratio4["by_time_bucket"]["static"]["latent_interval_ps"] is None
    assert ratio4["by_time_bucket"]["dt_100ps"]["metrics"]["future"]["rmsd"] == 0.0
    assert ratio4["by_time_bucket"]["dt_100ps"]["metrics"]["frame0"]["rmsd"] == 0.0
    anchor_future = report["controls"]["ratio1_no_temporal"]["by_time_bucket"]["dt_100ps"]["metrics"]["future"]["rmsd"]
    assert anchor_future > 0.0

    metrics = ratio4["by_time_bucket"]["dt_100ps"]["metrics"]
    for key in ("rmsd", "drmsd", "bond_rmse", "contact_error", "clash_rate", "torsion_change"):
        assert key in metrics["frame0"] and key in metrics["future"]
    for key in ("velocity_rmse", "acceleration_rmse", "frequency_retention"):
        assert key in metrics


def test_json_and_markdown_report_roundtrip(tmp_path: Path):
    batch = _trajectory("dt_100ps", 100.0)
    report = evaluate_controls([anchor_control()], [batch])
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["schema_version"] == "pvb.codec.eval.v1"
    markdown = markdown_path.read_text()
    assert "dt_100ps" in markdown
    assert "no cross-bucket mean" in markdown
    assert report_markdown(report) == markdown
