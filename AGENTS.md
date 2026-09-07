# molvid project instructions

## Active implementation phase

The active phase is **state/detail codec v2 authorized T1 experiment**.

Before editing code, read the preceding phase documents as immutable historical context, then
the current prompt and new phase documents in this order:

1. v1 files: `agents/visnet_spatial_v1/PLAN.md`, `DECISIONS.md`, `ACCEPTANCE.md`, `TASKS.md`,
   `HANDOFF.md`, and `LUNA_CODEX_PROMPT.md`;
2. v2 files: `agents/visnet_spatial_v2/PLAN.md`, `DECISIONS.md`, `ACCEPTANCE.md`,
   `REFERENCE_PARITY.md`, `TASKS.md`, and `HANDOFF.md`;
3. `LUNA_REVIEW_FIX_PROMPT_260903.md`;
4. `agents/state_detail_codec_v2/PLAN.md` and its appended T1-gates repair addendum;
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

- Work on `fix/state-detail-codec-v2-t1-gates`, based on source commit
  `48bbff992e66cc5f351911e23f31750325ef3726` from `feat/state-detail-codec-v2`.
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

The operator explicitly authorized the T1 gates after the repaired T0 packet on 2026-09-04.
Execute only this frozen sequence: materialize and verify an independent-system 48/8/8 split
over 64 systems; run four 200-step profiles; launch the four complete one-seed controls only if
all profiles pass; select a ratio using validation only; freeze the selection rule before opening
test; then run an additional seed only for the top two controls. Every new run uses the existing
`torchmd_et` frame encoder, the fixed state/detail contracts, and the unchanged data/loss/
optimizer/decoder protocol unless the operator explicitly changes it.

Do not start any later architecture work: DiT, observation adapter, forecasting, rollout,
AF3/MSA conditioning, VAE/KL/VQ, scaling-law, or unrelated full-data work. Do not broaden the
T1 matrix, use test results for selection, or start the complete runs before the profile gate.
After the authorized T1 sequence and its review packet are complete, set phase status to
`WAITING_FOR_OPERATOR_REVIEW` and stop.

## Documentation ownership

- `AGENTS.md`: this transition records the explicit operator T1 authorization; otherwise preserve
  it.
- v1/v2 phase files and artifacts: read-only historical evidence; never rewrite them.
- `agents/state_detail_codec_v2/PLAN.md` and `ACCEPTANCE.md`: the approved original text is
  immutable; append only the repair addendum required by the active repair prompt to `PLAN.md`.
- `agents/state_detail_codec_v2/DECISIONS.md`: append only when an implementation fact forces a
  real new decision; never rewrite prior decisions.
- `agents/state_detail_codec_v2/TASKS.md`: update task status and attach concise evidence.
- `agents/state_detail_codec_v2/HANDOFF.md`: append dated commands, tests, outputs, blockers,
  metrics, and next task.
- `agents/state_detail_codec_v2/OPERATOR_REVIEW.md`: preserve prior reviews as history and append
  only observed T1 profile/training/evaluation evidence.

Do not begin coding by rewriting the plan. Execute the prepared state/detail plan, and do not change
acceptance thresholds to make a failing result pass.
