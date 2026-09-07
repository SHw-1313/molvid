from __future__ import annotations

import pytest
import torch
from torch import nn

from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import LatentStatistics, StateDetailLatentAdapter
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer
from dit_test_utils import make_batch


torch.set_num_threads(1)


def _trainer(tmp_path, *, data_hash: str = "fixture"):
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    model = MolecularDiT(adapter=adapter, scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2)
    batch = make_batch(2, width=4)
    stats = LatentStatistics.fit([batch], ratio=2, provenance={"split": "train"})
    codec = nn.Linear(2, 2)
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    config = DiTTrainConfig(
        ratio=2,
        mode="ratio2_state_detail",
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        max_steps=2,
        output_root=str(tmp_path / "dit"),
        data_hash=data_hash,
    )
    return DiTTrainer(model, adapter, config=config, statistics=stats, codec=codec), batch, stats, config


def test_frozen_codec_logging_and_field_diagnostics(tmp_path) -> None:
    trainer, batch, _, _ = _trainer(tmp_path)
    assert not any(parameter.requires_grad for parameter in trainer.codec.parameters())
    assert not trainer.codec.training
    log = trainer.train_step(batch)
    assert trainer.step == 1
    assert log["loss"] >= 0
    assert set(log["valid_elements"]) == {"state_h", "detail_h", "state_v", "detail_v"}
    assert log["tau_min"] >= 0 and log["tau_max"] <= 1
    assert 0 <= log["observation_fraction"] <= 1


def test_checkpoint_roundtrip_and_contract_refusal(tmp_path) -> None:
    trainer, batch, stats, config = _trainer(tmp_path)
    trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    adapter2 = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    model2 = MolecularDiT(adapter=adapter2, scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2)
    codec2 = nn.Linear(2, 2)
    codec2.load_state_dict(trainer.codec.state_dict())
    for parameter in codec2.parameters():
        parameter.requires_grad_(False)
    trainer2 = DiTTrainer(model2, adapter2, config=config, statistics=stats, codec=codec2)
    payload = trainer2.load_checkpoint(path)
    assert trainer2.step == 1
    assert payload["schema"].endswith("checkpoint.v1")
    assert trainer2.train_step(batch)["step"] == 2
    trainer3, _, _, _ = _trainer(tmp_path, data_hash="different")
    with pytest.raises(ValueError, match="data_hash"):
        trainer3.load_checkpoint(path)
