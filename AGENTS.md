# molvid project instructions

## Active implementation phase

The active phase is **zero-preserving state/detail temporal codec v2**.

Before editing code, read the preceding phase documents as immutable historical context, then
the current prompt and new phase documents in this order:

1. v1 files: `agents/visnet_spatial_v1/PLAN.md`, `DECISIONS.md`, `ACCEPTANCE.md`, `TASKS.md`,
   `HANDOFF.md`, and `LUNA_CODEX_PROMPT.md`;
2. v2 files: `agents/visnet_spatial_v2/PLAN.md`, `DECISIONS.md`, `ACCEPTANCE.md`,
   `REFERENCE_PARITY.md`, `TASKS.md`, and `HANDOFF.md`;
3. `LUNA_REVIEW_FIX_PROMPT_260903.md`;
4. `agents/state_detail_codec_v2/PLAN.md`;
5. `agents/state_detail_codec_v2/DECISIONS.md`;
6. `agents/state_detail_codec_v2/ACCEPTANCE.md`;
7. `agents/state_detail_codec_v2/TASKS.md`;
8. `agents/state_detail_codec_v2/OPERATOR_REVIEW.md`;
9. `agents/state_detail_codec_v2/HANDOFF.md`.

The state/detail phase files are authoritative for this branch. All v1/v2 phase files, source
implementations, checkpoints, reports, and result artifacts are read-only historical evidence
and must remain reproducible. The older cumulative files under `agents/PLAN.md`,
`agents/TASKS.md`, `agents/DECISIONS.md`, and `agents/HANDOFF.md` are historical context and must
not be overwritten or repurposed.

## Repository and environment rules

- Work on `feat/state-detail-codec-v2`, based on `feat/visnet-spatial-v2` commit
  `045dbb8809e9d7aee418eb355b6477e05088d4fd`.
- Run all Python, tests, fixture generation, training, evaluation, and plotting only through
  `enter-container` with the `torch-ito` conda environment.
- New experiments use `spatial_backbone=torchmd_et`; ViSNet v1/v2 backends remain compatible,
  reproducible, and out of this phase's matrix.
- Do not install packages, upgrade dependencies, use arbitrary network access, download data,
  perform destructive Git operations, or commit generated checkpoints.
- Preserve unrelated user changes.
- Serialize GPU-heavy commands.
- Do not commit generated checkpoints, large datasets, or binary plots unless explicitly requested.

## Scope discipline

This phase implements and evaluates only the deterministic zero-preserving state/detail temporal
codec, its contracts/tests, and the bounded three-system/nine-trajectory T0. Every new run uses
the existing `torchmd_et` frame encoder. Keep ViSNet v1/v2 backends, contracts, checkpoints, and
result files unchanged and reproducible.

Do not create a T1 manifest or run T1, static/dynamic large-data training, DiT, observation
adapter, forecasting, rollout, AF3/MSA conditioning, VAE/KL/VQ, scaling-law, full-data, or any
later architecture work. After the T0 operator review packet is complete, set phase status to
`WAITING_FOR_OPERATOR_REVIEW` and stop; no later task may start without explicit operator approval.

## Documentation ownership

- `AGENTS.md`: this required S201 transition is authorized; otherwise preserve it.
- v1/v2 phase files and artifacts: read-only historical evidence; never rewrite them.
- `agents/state_detail_codec_v2/PLAN.md` and `ACCEPTANCE.md`: read-only.
- `agents/state_detail_codec_v2/DECISIONS.md`: append only when an implementation fact forces a
  real new decision; never rewrite prior decisions.
- `agents/state_detail_codec_v2/TASKS.md`: update task status and attach concise evidence.
- `agents/state_detail_codec_v2/HANDOFF.md`: append dated commands, tests, outputs, blockers,
  metrics, and next task.
- `agents/state_detail_codec_v2/OPERATOR_REVIEW.md`: fill only with observed evidence after T0.

Do not begin coding by rewriting the plan. Execute the prepared state/detail plan, and do not change
acceptance thresholds to make a failing result pass.
