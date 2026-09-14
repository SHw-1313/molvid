from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import os

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
