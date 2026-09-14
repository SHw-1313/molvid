from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import os
from types import SimpleNamespace

import pytest
import torch

from module.latent_flow_source import LatentFieldSet
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import LatentStatistics, StateDetailLatentAdapter
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash
from dit_test_utils import make_batch


torch.set_num_threads(1)


def _value_hash(value) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _require_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_CAPACITY_ISOLATION") != "1":
        pytest.skip("set DIT_RUN_CAPACITY_ISOLATION=1 for the required CUDA isolation check")
    if not torch.cuda.is_available():
        pytest.fail("capacity isolation was requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _trainer(tmp_path, device: torch.device, state, *, source_mode: str, depth: int) -> DiTTrainer:
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2).to(device)
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=depth,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    stats = LatentStatistics.fit([make_batch(2, width=4)], ratio=2, provenance={"split": "capacity-test"}).to(device)
    config = DiTTrainConfig(
        ratio=2,
        mode="ratio2_state_detail",
        scalar_width=8,
        vector_width=4,
        depth=depth,
        heads=2,
        ffn_multiplier=2,
        max_steps=2,
        seed=20260914,
        amp=True,
        output_root=str(tmp_path / source_mode / f"depth{depth}"),
        data_hash="capacity-test-data",
        source_mode=source_mode,
        init_hash=module_state_hash(model),
        metadata={"phase": "dit_capacity_data_v1"},
    )
    return DiTTrainer(model, adapter, config=config, statistics=stats)


def test_capacity_config_has_four_fixed_experiments() -> None:
    from pathlib import Path

    from scripts.run_dit_capacity_data_v1 import EXPERIMENT_IDS, _experiment_specs, _load_config

    root = Path(__file__).resolve().parents[1]
    cfg = _load_config(root / "config/dit_capacity_data_v1.yaml")
    specs = _experiment_specs(cfg)
    assert cfg["protocol"]["deterministic_cuda"] is True
    assert cfg["schedule"]["checkpoint_interval"] == 500
    assert tuple(specs) == EXPERIMENT_IDS
    assert (specs["G48"].source_mode, specs["G48"].depth, specs["G48"].data_scale) == ("gaussian", 4, "base48")
    assert (specs["C48"].source_mode, specs["C48"].depth, specs["C48"].data_scale) == ("conditional", 4, "base48")
    assert (specs["C192"].source_mode, specs["C192"].depth, specs["C192"].data_scale) == ("conditional", 4, "expanded192")
    assert (specs["C48D8"].source_mode, specs["C48D8"].depth, specs["C48D8"].data_scale) == ("conditional", 8, "base48")


def test_frozen_capacity_views_are_nested_and_test_free() -> None:
    from pathlib import Path

    from data.dit_capacity_data import load_capacity_data

    root = Path(__file__).resolve().parents[1] / "outputs/dit_capacity_data_v1/data_manifest_20260914"
    data = load_capacity_data(root)
    try:
        assert len(data.train48) == 48 * 3 * 62
        assert len(data.train192) == 192 * 3 * 62
        assert len(data.valid) == 8 * 3 * 62
        assert set(data.manifest["views"]["train48"]["sample_ids"]).issubset(
            set(data.manifest["views"]["train192"]["sample_ids"])
        )
        assert data.manifest["test_sampling"]["opened"] is False
        assert data.materialization["test_opened"] is False
        assert data.blocked_data == {}
    finally:
        data.close()


def test_expanded_scale_raises_explicit_blocked_data() -> None:
    from pathlib import Path

    from data.dit_capacity_data import BlockedDataError, CapacityData

    base = object()
    valid = object()
    data = CapacityData(
        manifest_root=Path("."),
        manifest={},
        materialization={},
        train48=base,
        train192=None,
        valid=valid,
        data_hash="semantic-data-hash",
        source_index_hashes={},
        source_roots={},
        blocked_data={"expanded192": "expanded source missing"},
    )
    assert data.train_for_scale("base48") is base
    with pytest.raises(BlockedDataError, match="expanded source missing"):
        data.train_for_scale("expanded192")


def test_git_commit_override_requires_full_sha(monkeypatch) -> None:
    from scripts.run_dit_capacity_data_v1 import _git_commit

    monkeypatch.setenv("DIT_CODE_COMMIT", "a" * 40)
    assert _git_commit() == "a" * 40
    monkeypatch.setenv("DIT_CODE_COMMIT", "not-a-full-sha")
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        _git_commit()


def test_batch_plan_is_built_once_per_epoch() -> None:
    from scripts.run_dit_capacity_data_v1 import _next_batch

    class Sampler:
        selected_sample_ids = ("sample0", "sample1")

        def __init__(self) -> None:
            self.calls = []
            self.global_batches = ()

        def set_epoch(self, epoch: int) -> None:
            self.calls.append(epoch)
            self.global_batches = ((epoch * 2,), (epoch * 2 + 1,))

    sampler = Sampler()
    cursor = {"epoch": 0, "batch_index": 0}
    cache = {}
    first, first_hash, first_epoch = _next_batch(sampler, cursor, cache)
    second, second_hash, second_epoch = _next_batch(sampler, cursor, cache)
    third, third_hash, third_epoch = _next_batch(sampler, cursor, cache)

    assert (first, second, third) == ((0,), (1,), (2,))
    assert (first_epoch, second_epoch, third_epoch) == (0, 0, 1)
    assert first_hash == second_hash
    assert third_hash != first_hash
    assert sampler.calls == [0, 1]


def test_resume_history_is_atomically_truncated_to_checkpoint(tmp_path) -> None:
    from scripts.run_dit_capacity_data_v1 import _read_jsonl, _truncate_history_to_checkpoint

    path = tmp_path / "train_history.jsonl"
    path.write_text(
        '{"step": 1, "loss": 3.0}\n'
        '{"step": 2, "loss": 2.0}\n'
        '{"step": 3, "loss": 1.0}\n'
        '{"step":',
        encoding="utf-8",
    )
    retained = _truncate_history_to_checkpoint(path, 2)

    assert [row["step"] for row in retained] == [1, 2]
    assert _read_jsonl(path) == retained
    assert not (tmp_path / "train_history.jsonl.tmp").exists()


def test_generation_aggregation_is_draw_clip_system_and_retains_regions() -> None:
    from scripts.run_dit_capacity_data_v1 import _aggregate_generation_rows

    def row(sample_id: str, system: str, draw: int, value: float) -> dict:
        region = {
            "aligned_rmsd": value,
            "drmsd": value + 0.1,
            "bond_rmse": value + 0.2,
            "contact_f1": value + 0.3,
        }
        rmsf = {
            "prediction": value,
            "target": 2.0,
            "correlation": value / 10.0,
        }
        return {
            "sample_id": sample_id,
            "system": system,
            "draw": draw,
            "history_frames": 4,
            "steps": 16,
            "metrics": {
                "future": region,
                "horizons": {
                    "L4": {
                        "available": True,
                        "metrics": region,
                        "temporal": {"rmsf": rmsf},
                    },
                    "L8": {
                        "available": True,
                        "metrics": region,
                        "temporal": {"rmsf": rmsf},
                    },
                },
            },
            "corrected_metrics": {
                "rmsf": rmsf,
                "velocity_lag1_acf": {
                    "prediction": value / 20.0,
                    "target": 0.5,
                    "dynamic_correlation": value / 30.0,
                },
            },
            "true_future_contact_occupancy_mae": {"value": value + 0.4},
            "block_displacements": {
                "prediction_within_block_rms": value,
                "target_within_block_rms": 2.0,
                "prediction_between_block_centroid_rms": value * 2.0,
                "target_between_block_centroid_rms": 4.0,
            },
        }

    aggregate = _aggregate_generation_rows(
        [
            row("sample-a", "system-a", 0, 1.0),
            row("sample-a", "system-a", 1, 3.0),
            row("sample-b", "system-a", 0, 5.0),
            row("sample-c", "system-b", 0, 9.0),
        ]
    )["H4_L16"]

    # sample-a averages its draws to 2; system-a then averages samples a/b to 3.5.
    assert aggregate["system_rows"]["system-a"]["aligned_rmsd"] == 3.5
    assert aggregate["system_rows"]["system-b"]["aligned_rmsd"] == 9.0
    assert aggregate["system_equal"]["aligned_rmsd"] == 6.25
    assert aggregate["regions"]["L4"]["system_equal"]["aligned_rmsd"] == 6.25
    assert aggregate["regions"]["L8"]["system_equal"]["rmsf_ratio"] == 3.125
    assert aggregate["regions"]["future"]["system_equal"][
        "within_block_displacement_ratio"
    ] == 3.125
    assert aggregate["regions"]["future"]["system_equal"][
        "between_block_displacement_ratio"
    ] == 3.125
    assert "contact_occupancy_mae" not in aggregate["system_equal"]
    assert "contact_occupancy_mae" in aggregate["excluded_metrics"]
    assert "diversity" in aggregate["excluded_metrics"]
    assert aggregate["sampling_steps"] == 16


def test_true_future_occupancy_matches_pairwise_definition() -> None:
    from scripts.run_dit_capacity_data_v1 import _true_occupancy_mae

    prediction = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    target = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ]
    )
    batch = SimpleNamespace(
        loss_mask=torch.ones(3, dtype=torch.bool),
        abid=torch.zeros(3, dtype=torch.long),
        bond_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    result = _true_occupancy_mae(prediction, target, batch, history=1)

    # The bond (0,1) is excluded. Pair (0,2) differs for one of two future
    # frames, while pair (1,2) has identical occupancy: mean MAE = (0.5+0)/2.
    assert result["value"] == 0.25
    assert result["pair_count"] == 2
    assert result["implementation"] == "bounded_chunk_vectorized"


def test_evaluation_rows_resume_from_atomic_contract_shards(tmp_path) -> None:
    from scripts.run_dit_capacity_data_v1 import _materialize_evaluation_row

    coordinates = tmp_path / "prediction.npz"
    coordinates.write_bytes(b"compact coordinates")
    row_path = tmp_path / "rows" / "sample.json"
    calls = []
    expected = {
        "experiment_id": "C48",
        "sample_id": "sample-a",
        "history_frames": 4,
        "steps": 16,
        "draw_id": 0,
    }

    def build() -> dict:
        calls.append("called")
        return {
            **expected,
            "schema": "pvb.dit.capacity_data.generation_row.v1",
            "prediction_coordinates": str(coordinates),
            "test_payload_opened": False,
        }

    first = _materialize_evaluation_row(
        row_path,
        split="final",
        contract_hash="contract-a",
        expected=expected,
        build_row=build,
    )
    second = _materialize_evaluation_row(
        row_path,
        split="final",
        contract_hash="contract-a",
        expected=expected,
        build_row=lambda: pytest.fail("completed row should not be regenerated"),
    )

    assert first == second
    assert calls == ["called"]
    assert not row_path.with_name(row_path.name + ".tmp").exists()
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _materialize_evaluation_row(
            row_path,
            split="final",
            contract_hash="contract-b",
            expected=expected,
            build_row=build,
        )


def test_report_retains_horizons_motion_cost_and_blocked_data(tmp_path, monkeypatch) -> None:
    from scripts.run_dit_capacity_data_v1 import _write_report

    def aggregate(value: float) -> dict:
        metrics = {
            "aligned_rmsd": value,
            "drmsd": value + 0.1,
            "bond_rmse": value + 0.2,
            "contact_f1": 1.0 - value / 10.0,
            "rmsf_prediction": value,
            "rmsf_target": 1.0,
            "rmsf_ratio": value,
            "rmsf_atom_correlation": 0.5,
            "prediction_within_block_rms": value,
            "target_within_block_rms": 1.0,
            "within_block_displacement_ratio": value,
            "prediction_between_block_centroid_rms": value,
            "target_between_block_centroid_rms": 1.0,
            "between_block_displacement_ratio": value,
        }
        return {
            f"H{history}_L16": {
                "regions": {
                    region: {"system_equal": metrics}
                    for region in ("L4", "L8", "future")
                }
            }
            for history in (4, 8)
        }

    experiments = {}
    for experiment_id, value in (("G48", 3.0), ("C48", 2.0), ("C48D8", 1.5)):
        experiments[experiment_id] = {
            "status": "PASS",
            "source_mode": "gaussian" if experiment_id == "G48" else "conditional",
            "data_scale": "base48",
            "depth": 8 if experiment_id == "C48D8" else 4,
            "final_aggregate": aggregate(value),
            "train_subset_aggregate": aggregate(value + 0.5),
            "generated_row_gpu_hours": 0.25,
        }
        run_dir = tmp_path / experiment_id
        run_dir.mkdir()
        (run_dir / "train_summary.json").write_text(
            '{"actual_optimizer_updates": 4500, "tokens_seen": 100, '
            '"estimated_gpu_hours": 1.5}\n',
            encoding="utf-8",
        )
    experiments["C192"] = {
        "status": "BLOCKED_DATA",
        "source_mode": "conditional",
        "data_scale": "expanded192",
        "depth": 4,
        "reason": "expanded payload missing",
    }
    (tmp_path / "source_decision.json").write_text(
        '{"rows": [{"template_bond_rmse": 0.01}]}\n', encoding="utf-8"
    )
    monkeypatch.setenv("DIT_CODE_COMMIT", "a" * 40)
    _write_report(SimpleNamespace(output_dir=tmp_path), {"experiments": experiments})
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "| G48 | gaussian | 4 | base48 | 4 | L4 |" in report
    assert "## Future motion decomposition" in report
    assert "Conditional source gain retained (C48 vs G48): YES" in report
    assert "More training systems (C192 vs C48): BLOCKED_DATA" in report
    assert "training GPU-hours" in report
    assert "explicit geometry supervision or local atom interactions" in report


def test_cuda_legacy_both_path_builds_independent_trainable_models(tmp_path) -> None:
    from scripts.run_dit_source_ab import ExperimentContext, _make_trainer, _shared_initialization

    cfg = {
        "seed": {"init": 20260914, "training": 20260914},
        "candidate": {"ratio": 2, "mode": "ratio2_state_detail"},
        "model": {
            "codec_width": 4,
            "scalar_width": 8,
            "vector_width": 4,
            "depth": 1,
            "heads": 2,
            "ffn_multiplier": 2,
            "dropout": 0.0,
            "execution_backend": "factorized_v2",
        },
        "protocol": {"learning_rate": 2e-4, "weight_decay": 0.01, "grad_clip": 1.0},
        "source": {"sigma": 1.0},
        "schedule": {"observation_history": [4, 8]},
    }
    device = _require_cuda()
    statistics = LatentStatistics.fit(
        [make_batch(2, width=4)], ratio=2, provenance={"split": "legacy-both-test"}
    )
    codec_model = torch.nn.Identity().to(device)
    codec_model.frame_encoder = torch.nn.Identity().to(device)
    context = ExperimentContext(
        cfg=cfg,
        output_dir=tmp_path,
        data=SimpleNamespace(data_hash="legacy-both-data"),
        codec=SimpleNamespace(model=codec_model, codec_state_hash="legacy-both-codec"),
        statistics=statistics,
        adapter=StateDetailLatentAdapter(
            codec_width=4, scalar_width=8, vector_width=4, ratio=2
        ).to(device),
        device=device,
        statistics_path=tmp_path / "statistics.pt",
        statistics_file_sha256="legacy-both-statistics-file",
    )
    init_state, init_hash = _shared_initialization(cfg, device)
    gaussian = _make_trainer(
        context,
        source_mode="gaussian",
        center_kind="repeat_last_coordinate_encode",
        target_steps=2,
        init_state=init_state,
        init_hash=init_hash,
    )
    conditional = _make_trainer(
        context,
        source_mode="conditional",
        center_kind="repeat_last_coordinate_encode",
        target_steps=2,
        init_state=init_state,
        init_hash=init_hash,
    )

    gaussian_storage = {parameter.data_ptr() for parameter in gaussian.model.parameters()}
    conditional_storage = {parameter.data_ptr() for parameter in conditional.model.parameters()}
    context_storage = {parameter.data_ptr() for parameter in context.adapter.parameters()}
    assert module_state_hash(gaussian.model) == module_state_hash(conditional.model) == init_hash
    assert gaussian_storage.isdisjoint(conditional_storage)
    assert gaussian_storage.isdisjoint(context_storage)
    assert conditional_storage.isdisjoint(context_storage)


def test_cuda_capacity_parameter_isolation_and_resume(tmp_path) -> None:
    device = _require_cuda()
    torch.manual_seed(20260914)
    base4 = MolecularDiT(
        adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2).to(device),
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    ).to(device)
    init4 = {name: value.detach().cpu().clone() for name, value in base4.state_dict().items()}
    base8 = MolecularDiT(
        adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2).to(device),
        scalar_width=8,
        vector_width=4,
        depth=2,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    ).to(device)
    init8 = {name: value.detach().cpu().clone() for name, value in base8.state_dict().items()}
    g = _trainer(tmp_path, device, init4, source_mode="gaussian", depth=1)
    c = _trainer(tmp_path, device, init4, source_mode="conditional", depth=1)
    d8 = _trainer(tmp_path, device, init8, source_mode="conditional", depth=2)
    assert module_state_hash(g.model) == module_state_hash(c.model)
    assert module_state_hash(d8.model) != module_state_hash(g.model)
    assert {
        parameter.data_ptr() for parameter in g.model.parameters() if parameter.requires_grad
    }.isdisjoint({parameter.data_ptr() for parameter in c.model.parameters() if parameter.requires_grad})
    assert {
        parameter.data_ptr() for parameter in g.model.parameters() if parameter.requires_grad
    }.isdisjoint({parameter.data_ptr() for parameter in d8.model.parameters() if parameter.requires_grad})

    batch = make_batch(2, width=4).to(device)
    zero_center = LatentFieldSet.zeros_like(batch.fields)
    generator_g = torch.Generator(device=device).manual_seed(20260914)
    generator_c = torch.Generator(device=device).manual_seed(20260914)
    c_before = module_state_hash(c.model)
    c_optimizer_before = _value_hash(c.optimizer.state_dict())
    g.train_step(batch, generator=generator_g)
    assert module_state_hash(c.model) == c_before
    assert _value_hash(c.optimizer.state_dict()) == c_optimizer_before
    g_before = module_state_hash(g.model)
    g_optimizer_before = _value_hash(g.optimizer.state_dict())
    c.train_step(batch, generator=generator_c, source_center=zero_center)
    assert module_state_hash(g.model) == g_before
    assert _value_hash(g.optimizer.state_dict()) == g_optimizer_before

    continuous = _trainer(tmp_path, device, init4, source_mode="gaussian", depth=1)
    resumed = _trainer(tmp_path, device, init4, source_mode="gaussian", depth=1)
    generator_cont = torch.Generator(device=device).manual_seed(20260914)
    generator_resume = torch.Generator(device=device).manual_seed(20260914)
    continuous.train_step(batch, generator=generator_cont)
    payload = continuous.checkpoint_payload()
    payload["explicit_generator_state"] = generator_cont.get_state()
    checkpoint = tmp_path / "resume.pt"
    torch.save(payload, checkpoint)
    continuous_row = continuous.train_step(batch, generator=generator_cont)
    loaded = resumed.load_checkpoint(checkpoint, map_location=device)
    generator_resume.set_state(loaded["explicit_generator_state"].detach().to(device="cpu"))
    resumed_row = resumed.train_step(batch, generator=generator_resume)
    assert continuous_row["tau_mean"] == resumed_row["tau_mean"]
    assert continuous_row["tau_min"] == resumed_row["tau_min"]
    assert continuous_row["tau_max"] == resumed_row["tau_max"]
    assert module_state_hash(continuous.model) == module_state_hash(resumed.model)
    assert _value_hash(continuous.optimizer.state_dict()) == _value_hash(resumed.optimizer.state_dict())
    assert continuous.successful_updates == resumed.successful_updates == 2
