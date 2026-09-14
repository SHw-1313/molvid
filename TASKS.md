# Session B tasks

## Completed

- Created worktree `molvid-dit-capacity-data-v1` from fixed commit `5c2754fcce44ed77dad77db09db709408fee7634` on branch `exp/dit-capacity-data-v1`.
- Fixed legacy `both` and capacity-runner parameter isolation: every experiment owns its adapter, model, optimizer, scaler, and generators; only frozen codec/statistics are shared.
- Added successful-update accounting and exact checkpoint restoration for sampler cursor, generators, optimizer/scaler, and source/data/model contracts.
- Materialized the frozen nested 48/192 train and 8-system validation views without opening test coordinates.
- Verified host/neibu fixed-record equivalence, frozen artifact hashes, deterministic CUDA isolation/mutation/resume, and real H4/H8 sampler/decode smoke.
- Added a direct CUDA regression for the legacy source runner's two trainer factories.
- Cached batch plans and clip specs per epoch; exact 20,000-step token estimation dropped from an impractical repeated rebuild to 3.30 seconds.
- Froze the measured budget on neibu: 4,500 updates and 267,988,032 effective atom-frame tokens, with 25% reserved for evaluation.

## Running on neibu

- tmux session `dit-capacity-train` on GPU0 runs independent Python processes in order: G48, C48, C48D8, final evaluation, summarize.
- Canonical numerical code commit: `b68dafa968b72519e976f795b7611673c1154a0f`.
- Atomic checkpoints are written every 500 successful updates; resume truncates trailing JSONL rows to the checkpoint step before continuing.
- The first formal G48 checkpoint at step 500 was written and inspected successfully; training has continued past it.
- Run root: `/workspace/molvid-dit-capacity-data-v1/outputs/dit_capacity_data_v1/20260914_capacity_data_v1`.

## Pending

- Inspect training/evaluation outputs and failure status when the tmux queue exits.
- Sync compact evidence/checkpoints needed for handoff back to the B host worktree.
- Finish `report.md`, learning curves, per-system results, and final `HANDOFF.md`.
- Train C192 only if a local GPU becomes genuinely idle while the full expanded view remains available; do not copy the large payload to neibu.

The first 89-step G48 launch was intentionally stopped before any checkpoint while the recovery
audit was tightened. Its files are preserved separately as
`G48_interrupted_precheckpoint_step000089` and are excluded from the formal run.

## Explicit blocker

- neibu has the exact base48/validation records but lacks 26,784 selected expanded clips, so C192 is `BLOCKED_DATA` there. The complete C192 manifest/payload remains available only on the original host, whose GPUs were all occupied at launch.
