# Session B tasks

## Completed

- Created worktree `molvid-dit-capacity-data-v1` at fixed commit `5c2754fcce44ed77dad77db09db709408fee7634`, branch `exp/dit-capacity-data-v1`.
- Patched trainable adapter/model construction in the legacy source runner and added successful-update checkpoint accounting in `trainer/dit_trainer.py`.
- Added the independent capacity/data runner, frozen experiment config, nested data materializer/loader, focused tests, and B-specific root `AGENTS.md` scope.
- Materialized `outputs/dit_capacity_data_v1/data_manifest_20260914/` with train48/train192/valid index views and read-only symlinks to the approved train/valid payloads.
- Verified CPU contracts and related tests; test payload remains unopened.

## Pending after a writable CUDA tmux is available

- Run the CUDA isolation/resume test and the runner's `verify` stage.
- Run `source_check`, then `prepare` to freeze the measured end-to-end budget before training.
- Train G48/C48/C48D8 first; add C192 when the same frozen data manifest is ready (the manifest is already ready).
- Run final per-clip generation/evaluation, plot curves, write `report.md`, and complete `HANDOFF.md` with results/costs/blockers.

## Current blockers

- No confirmed idle GPU: all eight GPUs were occupied at the last check. Do not preempt those jobs.
- Inside `enter-container`, use `/workspace/molvid-dit-capacity-data-v1`: this is the writable mapping of the host B worktree. The host-absolute workspace path under `/data4/users/sihao/workspace` is read-only inside the container. The approved source data under `/data4/users/sihao/data/.../dt_100ps` remains read-only by design.
