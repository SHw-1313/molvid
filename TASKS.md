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

## Training and evaluation status on neibu (2026-09-15)

- The `dit-capacity-train` and `dit-capacity-eval` queues have exited cleanly; neibu GPU0 is idle.
- Canonical training/numerical code commit: `c070e3c2aeecb6dc33d338e2429bfb426876faac`.
- Evaluation/reporting code is committed on the host through `9960a479071ddd50df575790a2fb6388fa399e93` and passed the targeted CUDA gate: 4 passed, 1 skipped, 17 deselected. The synced source used `DIT_CODE_COMMIT` set to that full commit.
- Atomic checkpoints are written every 500 successful updates; resume truncates trailing JSONL rows to the checkpoint step before continuing.
- G48, C48, and C48D8 each completed PASS at 4,500 successful updates and exactly 267,988,032 effective atom-frame tokens. Their resumed summaries report 2.357, 4.941, and 4.998 GPU-hours; including G48's initial formal failed attempt, formal training attempts total about 13.161 GPU-hours.
- All arms have 9 RF validation rows, complete 16-row real-generation monitors at steps 2,000 and 4,000, and 400 final-evaluation row files (144 final validation, 128 validation subset, 128 train subset). G48/C48 shared the exact initial 4-layer tensor hash and the first 16 batch/H/tau records; C48D8 has its distinct 8-layer initialization hash. Formal final checkpoints are in each arm's `checkpoints/checkpoint_final.pt`.
- Run root: `/workspace/molvid-dit-capacity-data-v1/outputs/dit_capacity_data_v1/20260914_capacity_data_v1`.

## Final status

- C192 remains `BLOCKED_DATA`: neibu lacks 26,784 selected expanded clips, and the data must not be copied or the frozen selection/homology criteria weakened for this run.
- Final generation evaluation ran on neibu GPU0 with actual sampler/decode/metrics, then resumable row shards were aggregated into `comparison.json`, `per_system_results.json`, `learning_curves.json`, `learning_curves.png`, `report.md`, and `summary.json`.
- The completed outputs and compact prediction coordinates were synchronized back to the B host worktree; final hashes and checkpoint paths are recorded in `HANDOFF.md`.
- Training plus real-generation evaluation is estimated at about 15.239 GPU-hours, excluding CPU-only preparation and targeted-test overhead, within the frozen 32 GPU-hour budget.

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
