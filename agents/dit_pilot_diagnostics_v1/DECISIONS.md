# A decisions

## Prepared decisions — 2026-09-09

- 不另建分支；A使用现有pilot checkout。
- 不重训R1/R2/R4，不再比较no-temporal或pooling。
- R2/R4只加载已完成pilot；当前R4为工程主线，不代表已证明通用生成优于R2。
- 当前codec的detail是Haar-like块内差分投影，不再沿用早期“匀速残差”口头定义。
- 复制最后block state/zero-detail只是一项待验证prior，不是保证合理的最后原子构象。
- raw detail的0经scalar normalization后未必为0。
- 诊断与B加速独立；只通过固定输入/输出协议交汇。
- 不用pathwise相关接近0或未超persistence RMSD单独否定stochastic dynamics。

## Implementation-forced additions

待worker追加有证据的事实。不得删除/改写prepared decisions以迁就结果。

## Observed implementation facts — 2026-09-09

- Session A stayed on `exp/dit-state-detail-t1-pilot-v1` at the audited `a22f60c4ffd1f502ab300352a00eafe841b1f8d2` snapshot; B's worktree and files were not modified.
- Preflight loaded only the frozen manifest `manifest_20260904_token80000`, the approved R2/R4 codec checkpoints, and the existing best-validation DiT checkpoints/statistics. The manifest input hash was `b478d9fbe6721847f62767940dc0481753ea3c4f1e26573e49a78a76379e3948`; test remained unopened.
- The first R4 deep run exposed a real diagnostics bug: vector norm quantiles used a `[K,N]` token mask directly against `[K,N,C]` norms. The fixed path expands the mask at the reporting boundary; the regression is covered by `test_latent_summary_expands_vector_token_masks_for_norm_quantiles`.
- Corrected R4 and R2 deep runs completed on CUDA with no optimizer steps. Main evaluation covered 72 clips, 8 systems, R1/R2/R3, windows 0/30/61, H4/H8 and five methods; deep evaluation covered eight R1/window-30 clips with H4/H8, 8/16-step sampling, four draws, fixed tau, perturbations and field swaps.
- The observed-only `latent_block_state_persistence` control is retained as a reference-only diagnostic, not a deployable prior/source center: both candidates' generated future geometry/contact results were worse than this control in the main evidence. No R2/R4 winner is declared.
