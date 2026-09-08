# molvid project instructions

## Active implementation phase

The active phase is **R2/R4 state-detail latent DiT probe v1**.

Before editing code, read the immutable historical context and then the current prompt and phase
documents in this exact order:

1. root `AGENTS.md`;
2. the v1, v2, repair-prompt, and state/detail codec files named by the preceding root phase
   instructions, as read-only historical context;
3. `LUNA_DIT_STATE_DETAIL_PROBE_V1_PROMPT_260907.md`;
4. `agents/dit_state_detail_probe_v1/PLAN.md`;
5. `agents/dit_state_detail_probe_v1/DECISIONS.md`;
6. `agents/dit_state_detail_probe_v1/ACCEPTANCE.md`;
7. `agents/dit_state_detail_probe_v1/TASKS.md`;
8. `agents/dit_state_detail_probe_v1/HANDOFF.md`;
9. `agents/dit_state_detail_probe_v1/OPERATOR_REVIEW.md`.

The DiT phase PLAN, prepared DECISIONS, and ACCEPTANCE are binding. Preceding v1/v2/state-detail
phase files, source implementations, active T1 processes, T1 branches, manifests, checkpoints,
reports, and result artifacts are read-only historical or active-experiment evidence and must
remain reproducible. The active codec T1 checkout is `/data4/users/sihao/workspace/PVB`; do not
edit or run DiT work there. The dedicated target is
`/data4/users/sihao/workspace/molvid-dit-state-detail-probe-v1` on
`feat/dit-state-detail-probe-v1`, based on audited source commit
`23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541`.

Older cumulative files under `agents/PLAN.md`, `agents/TASKS.md`, `agents/DECISIONS.md`, and
`agents/HANDOFF.md` are historical context and must not be overwritten or repurposed.

## Repository and environment rules

- Keep all DiT work in the dedicated target branch/worktree. Preserve unrelated user changes.
- Run every Python command, test, fixture generation, smoke, evaluation, and plotting command
  only through `enter-container` with `conda activate torch-ito`.
- Do not install packages, upgrade dependencies, use arbitrary network access, download data,
  perform destructive Git operations, or commit generated checkpoints, large datasets, or binary
  plots unless explicitly requested.
- Serialize GPU-heavy commands and use only an audited idle GPU. Never kill, suspend, migrate,
  or preempt another process.
- Do not silently rebase onto changed codec semantics. If the audited source branch advances,
  record and inspect the difference before proceeding.

## Scope discipline

Only `ratio2_state_detail` and `ratio4_state_detail` are DiT candidates. `ratio1_state_detail`
and `ratio4_matched_pooling` are historical codec controls and are not DiT candidates. The codec,
TorchMD frame encoder, centered-vector stem, coordinate decoder, codec trainer/evaluator semantics,
and all T1 scripts are frozen and must not be rewritten.

This phase is limited to the focused latent adapter, rectified-flow objective, shared molecular
DiT, observation-mask plumbing, trainer/checkpoint contracts, evaluation/report plumbing,
correctness tests, and one bounded smoke per candidate. Use a separate output root under
`outputs/dit_state_detail_probe_v1/`; never write under active T1 outputs.

Do not start the real R2/R4 scientific pilot, full-T1-data DiT training, T1 test access, scaling
experiments, static mixing, AF3/MSA conditioning, VAE/KL/VQ, long rollout, H=2, energy guidance,
ensemble control, or any later architecture work. Do not rank R2 versus R4 from smoke results.
The final status must be `WAITING_FOR_T1_AND_OPERATOR_REVIEW`, followed by a stop.

## Documentation ownership

- This root transition is authoritative only on the dedicated DiT branch; do not edit root
  `AGENTS.md` in the active T1 checkout.
- Preceding phase documents and artifacts remain immutable historical evidence.
- `agents/dit_state_detail_probe_v1/PLAN.md`, `DECISIONS.md`, and `ACCEPTANCE.md` are prepared
  authority; do not rewrite them. Append DECISIONS only for real implementation-forced facts.
- Update `agents/dit_state_detail_probe_v1/TASKS.md` only when evidence exists.
- Append dated commands, tests, hashes, metrics, blockers, and next tasks to `HANDOFF.md`.
- Fill `OPERATOR_REVIEW.md` with observed implementation/CPU/smoke evidence at the end.

Do not weaken acceptance criteria, add compatibility fallbacks, or continue after the required
operator-review stop.

## Active pilot transition — R2/R4 state-detail latent DiT T1 pilot v1

This separate branch/worktree is explicitly authorized for the matched 64-system pilot described
in `agents/dit_state_detail_pilot_v1/PILOT_PROTOCOL.md`. It is based on repair commit
`28a499f4c03deb4647a6468a5c477bd8eda8d62f` from `feat/dit-state-detail-probe-v1`; the repair
branch remains independently reviewable and must not be modified from this worktree.

- The frozen manifest is `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`.
- Only the manifest's train and validation materializations may be opened. Never construct,
  read, hash, or evaluate `clip_store/test`; no test split access is authorized tonight.
- Only the completed, SHA256-verified T1 `ratio2_state_detail` and `ratio4_state_detail`
  checkpoints named in the pilot protocol may be loaded. Random, T0, incomplete, or guessed
  codec checkpoints are prohibited.
- The frozen codec and frame encoder stay in eval mode and frozen. The shared DiT width/depth/
  heads, optimizer policy, ordered schedule, H=4/H=8 schedule, and validation protocol are
  identical for R2 and R4. No H=2, H=0 training, DDP, static mixing, AF3/MSA, architecture
  changes, or test evaluation.
- Real-data pilot outputs belong under `outputs/dit_state_detail_pilot_v1/`; do not write under
  active T1 output roots or commit checkpoints, datasets, caches, or binary plots.
- Every Python, test, profile, training, evaluation, and plotting command still runs through
  `enter-container` with `conda activate torch-ito`. Use only audited idle GPUs and never
  preempt another process.
- Freeze one common step budget in `pilot_budget.json` only after both 200-step profiles pass;
  require the common budget to be at least 1000 steps. Validation RF loss selects checkpoints;
  do not declare an R2/R4 winner.
- The pilot branch's final status is `WAITING_FOR_OPERATOR_REVIEW`; test remains sealed.
