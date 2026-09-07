from __future__ import annotations

import torch

from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import StateDetailLatentAdapter
from dit_test_utils import make_batch


def _model(ratio: int, width: int = 4) -> tuple[MolecularDiT, object]:
    adapter = StateDetailLatentAdapter(
        codec_width=width, scalar_width=8, vector_width=4, ratio=ratio
    )
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
    )
    return model, adapter


def test_shared_size_policy_and_factorized_contract() -> None:
    model2, _ = _model(2)
    model4, _ = _model(4)
    assert model2.parameter_count == model4.parameter_count
    assert "O((K*N)^2)" not in model2.contract()["attention_complexity"]
    assert model2.contract()["backend"].startswith("dense_block_attention")
    assert model2.contract()["temporal_attention"] == "bidirectional"
    assert model2.contract()["vector_maps"] == "bias_free_channel_only"


def test_scalar_invariance_vector_equivariance_and_time_conditioning() -> None:
    torch.manual_seed(41)
    model, _ = _model(2)
    model.eval()
    batch = make_batch(2, width=4)
    tau = torch.tensor([0.25, 0.75])
    first = model(batch, tau)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    rotated_fields = batch.fields.map(
        lambda value: value
        if value.ndim == 3
        else torch.einsum("ab,knbc->knac", q, value)
    )
    rotated = model(batch.with_fields(rotated_fields), tau)
    assert torch.allclose(first.state_h, rotated.state_h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(first.detail_h, rotated.detail_h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        rotated.state_v,
        torch.einsum("ab,knbc->knac", q, first.state_v),
        atol=2e-5,
        rtol=2e-5,
    )
    later_time = batch.with_fields(batch.fields.clone())
    later_time.block_time_ps = later_time.block_time_ps + 5.0
    later = model(later_time, tau)
    assert not torch.allclose(first.state_h, later.state_h)


def test_ragged_sample_isolation_and_all_vector_maps_are_bias_free() -> None:
    model, _ = _model(4)
    batch = make_batch(4, width=4, invalid_last_token=True)
    first = model(batch, torch.tensor([0.3, 0.6]))
    changed = batch.fields.clone()
    mask = batch.abid == 1
    changed.state_h[:, mask] += 10.0
    changed.detail_h[:, mask] -= 7.0
    changed.state_v[:, mask] *= 3.0
    changed.detail_v[:, mask] *= -2.0
    second = model(batch.with_fields(changed), torch.tensor([0.3, 0.6]))
    assert torch.allclose(first.state_h[:, ~mask], second.state_h[:, ~mask], atol=2e-5, rtol=2e-5)
    assert torch.equal(first.state_h[-1, -1], torch.zeros_like(first.state_h[-1, -1]))
    assert all(module.bias is None for module in model.modules() if module.__class__.__name__ == "AxisPreservingLinear")
    assert all(
        parameter.ndim != 2 or parameter.shape[0] != 3
        for name, parameter in model.named_parameters()
        if "vector" in name.lower()
    )


def test_factorized_trunk_and_output_receive_gradients_after_adaln_warmup() -> None:
    model, _ = _model(2)
    batch = make_batch(2, width=4)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        output = model(batch, torch.tensor([0.2, 0.8]))
        loss = sum(value.square().mean() for value in output.as_dict().values())
        loss.backward()
    intended = (
        "adapter.scalar_in",
        "adapter.vector_in",
        "blocks.0.spatial",
        "blocks.0.temporal",
        "blocks.0.ffn",
        "adapter.scalar_out",
        "adapter.vector_out",
    )
    for prefix in intended:
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        assert gradients, prefix
        assert all(torch.isfinite(value).all() for value in gradients)
        assert any(torch.any(value != 0) for value in gradients), prefix
