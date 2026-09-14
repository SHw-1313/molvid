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
- Moved periodic checkpoints ahead of expensive generation monitors and released the CUDA
  allocator around each sampler/decode metric pass. G48's step-2000 monitor completed all 16
  real clips after resuming from the step-1500 checkpoint, then training continued.
- Added draw-to-clip-to-system aggregation for H4/H8 and L4/L8/future metrics. Constant RMSF,
  ACF, and Pearson inputs now produce explicit null values instead of numeric sentinels.

## Running on neibu

- tmux session `dit-capacity-train` on GPU0 runs independent Python processes in order: G48, C48, C48D8. Evaluation is not part of this queue.
- Canonical training/numerical code commit: `c070e3c2aeecb6dc33d338e2429bfb426876faac`.
- Evaluation/reporting code is committed separately as `60f056b` on the host and is intentionally not synced while the training queue can still launch later arms.
- Atomic checkpoints are written every 500 successful updates; resume truncates trailing JSONL rows to the checkpoint step before continuing.
- G48 has valid checkpoints through step 2000. Its fixed step-2000 monitor has 16/16 coordinate files and 16/16 metric rows, and training has continued past step 2000.
- Run root: `/workspace/molvid-dit-capacity-data-v1/outputs/dit_capacity_data_v1/20260914_capacity_data_v1`.

## Pending

- Inspect training/evaluation outputs and failure status when the tmux queue exits.
- After all three training processes exit, sync `60f056b`, run its CUDA metric regressions, and then start final evaluation on an actually idle neibu GPU.
- Sync compact evidence/checkpoints needed for handoff back to the B host worktree.
- Finish `report.md`, learning curves, per-system results, and final `HANDOFF.md`.
- Train C192 only if a local GPU becomes genuinely idle while the full expanded view remains available; do not copy the large payload to neibu.

The first 89-step G48 launch was intentionally stopped before any checkpoint while the recovery
audit was tightened. Its files are preserved separately as
`G48_interrupted_precheckpoint_step000089` and are excluded from the formal run.

The first attempt to pass step 2000 stopped during generation monitoring with
`CUSOLVER_STATUS_INTERNAL_ERROR`: training had retained about 81 GiB in the CUDA allocator before
the Kabsch SVD. It was an evaluation resource failure, not a non-finite update. The step-2000
checkpoint was already made durable by the corrected ordering, and the formal history was safely
restored from step 1500 for the code change before completing the monitor under `c070e3c`.

## Explicit blocker

- neibu has the exact base48/validation records but lacks 26,784 selected expanded clips, so C192 is `BLOCKED_DATA` there. The complete C192 manifest/payload remains available only on the original host, whose GPUs were all occupied at launch.
