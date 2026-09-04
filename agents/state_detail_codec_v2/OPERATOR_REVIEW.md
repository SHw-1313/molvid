# OPERATOR REVIEW — state/detail codec v2 T0

Status: WAITING_FOR_OPERATOR_REVIEW

The implementation gates and the bounded four-control T0 are complete. This packet is awaiting
operator review; no T1 work is authorized by this document.

## 1. Exact code state

```text
branch: feat/state-detail-codec-v2
base commit: 045dbb8809e9d7aee418eb355b6477e05088d4fd
review commit or working-tree hash: HEAD=045dbb8809e9d7aee418eb355b6477e05088d4fd; no review commit; dirty worktree
git status: tracked edits in AGENTS.md, config/codec.yaml, module/__init__.py, trainer/codec_trainer.py;
  untracked prompt, state/detail phase files, new module/tests/scripts, and T0 outputs
diff summary: tracked diff is 334 insertions / 98 deletions across 4 files; generated artifacts are untracked
```

Changed source files:

- `AGENTS.md`
- `config/codec.yaml`
- `module/__init__.py`
- `module/state_detail_codec_v2.py`
- `trainer/codec_trainer.py`

Changed tests/scripts/configs and phase records:

- `tests/test_state_detail_codec_v2.py`
- `scripts/pack_t0_clip_store.py`
- `scripts/run_state_detail_codec_v2_t0.py`
- `agents/state_detail_codec_v2/TASKS.md`
- `agents/state_detail_codec_v2/HANDOFF.md`
- `agents/state_detail_codec_v2/OPERATOR_REVIEW.md`

## 2. Commands and environment

```text
test/compile command: enter-container; conda activate torch-ito; cd /workspace/PVB
conda environment: torch-ito
Python: 3.11.15
torch/cuda/torch_cluster: torch 2.5.1+cu121 / CUDA 12.1 / torch_cluster 1.6.3+pt25cu121
GPU mapping: `neibu`, CUDA_VISIBLE_DEVICES=0, NVIDIA A100-SXM4-80GB (80 GiB)
```

Final T0 command, run in the isolated `neibu` checkout and copied back to this checkout:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_state_detail_codec_v2_t0.py --mode all \
  --store-root outputs/state_detail_codec_v2/t0_data/clip_store \
  --output-root outputs/state_detail_codec_v2/t0_remote_final
```

Validation commands/results:

```text
PYTHONPATH=. python -m pytest -q -> 107 passed, 1 pre-existing FutureWarning, 12.81 s
python -m py_compile module/state_detail_codec_v2.py module/__init__.py trainer/codec_trainer.py
  scripts/pack_t0_clip_store.py scripts/run_state_detail_codec_v2_t0.py
  tests/test_state_detail_codec_v2.py -> passed
git diff --check -> passed
```

## 3. Architecture checks

| Check | Evidence path | Result |
|---|---|---|
| R1/R2/R4 lifting and inverse mapping | `tests/test_state_detail_codec_v2.py`; S210/S240 | Passed in strict FP32 at `rtol=1e-6`, `atol=1e-7`; R4 order is `Dmid,D01,D23`. |
| latent shapes and 16C/16C/8C accounting | `module/state_detail_codec_v2.py`; S211/S220-S225 | Passed: C=128 gives R1=2048 active elements, R2=2048, R4-SD=1024. |
| R4 matched-pooling 8C accounting | `module/state_detail_codec_v2.py`; S230-S231 | Passed: two independent C-wide banks, 1024 active elements, natural parameter count retained. |
| no per-atom x0/reference bypass | `tests/test_state_detail_codec_v2.py`; source audit; S212/S244 | Passed; new decoder has no target-coordinate argument and `state_detail_codec_v2.py` has no `x_anchor`. |
| only per-sample origin bypasses latent | `module/state_detail_codec_v2.py`; model contract; S212/S224 | Passed; origin is one `[B,3]` `frame0_loss_masked_centroid` vector. |
| repeated-static zero detail | `tests/test_state_detail_codec_v2.py`; S241 | Passed exactly before and after the learned detail bottleneck for R2/R4. |
| repeated-static zero coordinate motion | `tests/test_state_detail_codec_v2.py`; S241/S224 | Passed; decoded within-block feature/coordinate motion is zero under zero detail. |
| dynamic detail nonzero and receives gradients | `tests/test_state_detail_codec_v2.py`; final `micro_summary.json`; S242/S252 | Passed; all four micro runs have finite nonzero principal gradients; R2/R4 detail is utilized. |
| T=1 versus repeated-T16 semantics | `tests/test_state_detail_codec_v2.py`; S225/S240 | Passed; T=1 has `detail_valid=false`, repeated T=16 retains valid zero detail. |
| SE(3), masks, irregular clocks, isolation | `tests/test_state_detail_codec_v2.py`; S240/S243 | Passed for rotation/translation, partial blocks, invalid-frame isolation, clocks, and samples. |
| old checkpoint/legacy regression | `tests/test_state_detail_codec_v2.py`; `outputs/state_detail_codec_v2/preflight/legacy_audit.json`; S213/S245 | Passed; legacy loading retains old schema and old per-atom-anchor behavior. |

## 4. T0 data identity

```text
systems: atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A
replicates: R1, R2, R3 for each system (nine trajectories)
train clips: 441, windows w000000 through w000048
holdout clips: 117 late holdout, windows w000049 through w000061
T / dt: T=16 / native dt_100ps (100 ps)
manifest contract hash: 2dc3790c22ddb2b9606d3e4087a7204e9b91be20252dd6dc6dc7a3126550bca4
source manifest hash: 700f01e40f4e0fda697191cd161bb8161b97493a994415cee286354a9809450e
train-index hash: 04806744073a7df56af2fd5671849e1147a7c6492e956d389b6830aa136d9b0a
holdout-index hash: 5d247fcb5f21ba2780d6b93880afbc01675ae1a42a935292f04161b7ff1ea0d1
common TorchMD encoder hash: e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a
```

The manifest independently reports `intersection_count=0`, lazy loading, `replacement=false`,
`max_tokens=80000`, FP32, 30 complete epochs, and a 6000-step safety cap. There is no semantic
protocol discrepancy. The original source root is not present inside the portable copied
container (`source_roots_exist_at_audit=false`); the exact-byte compact clip store and its
provenance are present at `outputs/state_detail_codec_v2/t0_data/clip_store/`.

## 5. Training completion

| Control | 30 epochs | Best/final checkpoints | Resume | Finite | Detail/bank utilized |
|---|---|---|---|---|---|
| R1-SD (`ratio1_state_detail`) | yes; 4,976 steps | `ratio1_state_detail/codec_best.pt` (`7e98e51b...`); `codec_step_00004976.pt` (`26cfc054...`) | `4976 -> 4977`, passed | yes | state only; detail absent by contract |
| R2-SD (`ratio2_state_detail`) | yes; 4,976 steps | `ratio2_state_detail/codec_best.pt` (`0425288a...`); `codec_step_00004976.pt` (`32466181...`) | `4976 -> 4977`, passed | yes | detail valid fraction 1.0; encoded h norm 0.191008 |
| R4-SD (`ratio4_state_detail`) | yes; 4,976 steps | `ratio4_state_detail/codec_best.pt` (`51bb7f5a...`); `codec_step_00004976.pt` (`d8c381e2...`) | `4976 -> 4977`, passed | yes | detail valid fraction 1.0; encoded h norm 0.238345 |
| R4-Matched-Pooling (`ratio4_matched_pooling`) | yes; 4,976 steps | `ratio4_matched_pooling/codec_best.pt` (`ba807ff9...`); `codec_step_00004976.pt` (`4fd3b29f...`) | `4976 -> 4977`, passed | yes | bank A h norm 1.618426; bank B h norm 1.584718 |

All four runs used common frozen encoder source hash above; before/after frozen-state hash was
`c8e3fe219b4fc100774a03c74150dc8d2b8d12299d334f59102d44be2e2c377f` in the micro evidence.
The separate one-step unfreeze smoke passed with 41 nonzero spatial gradients and changed
frame-encoder state; it was not a fifth comparison run.

Loss-curve report:

```text
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curves.png/.pdf
per-control loss_curve_ratio1_state_detail.{png,pdf}, loss_curve_ratio2_state_detail.{png,pdf},
loss_curve_ratio4_state_detail.{png,pdf}, loss_curve_ratio4_matched_pooling.{png,pdf}
```

The operator should inspect the complete curves and plateau behavior; fixed step count alone is
not evidence of convergence.

## 6. Core evaluation

Aggregate report paths:

```text
JSON:     outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.json
CSV:      outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.csv
Markdown: outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.md
plots:    loss_curves, evaluation_curves, block_detail_metrics, performance_summary, and all four
          per-control loss curves, each in PNG and PDF form in the same directory
```

The following are final late-holdout future metrics. ACF is prediction followed by absolute error;
boundary is predicted jump followed by absolute error. Peak memory is allocated bytes.

| Metric | R1-SD | R2-SD | R4-SD | R4-Matched-Pooling |
|---|---:|---:|---:|---:|
| Holdout RMSD | 11.424228 | 11.241862 | 11.138619 | 10.544774 |
| Holdout dRMSD | 10.154870 | 9.900528 | 9.767301 | 9.306050 |
| Velocity RMSE | 0.0209102 | 0.0128179 | 0.00817435 | 0.0106722 |
| Acceleration RMSE | 0.000358905 | 0.000192065 | 0.000126111 | 0.000170632 |
| RMSF correlation | 0.766342 | 0.784969 | 0.782398 | 0.714334 |
| Lagged/ACF metric | 0.958019 / 0.039499 | 0.984965 / 0.012553 | 0.994066 / 0.003452 | 0.990999 / 0.006518 |
| Frequency retention | 9.382636 | 6.367882 | 4.089264 | 4.967844 |
| Boundary jump | 3.297411 / 2.378538 | 2.793389 / 1.875074 | 2.323409 / 1.404246 | 2.670770 / 1.751607 |
| Peak memory (bytes) | 25,781,457,408 | 26,321,821,696 | 26,322,873,344 | 25,788,809,728 |
| End-to-end samples/s | 15.6238 | 15.3303 | 15.1350 | 15.6286 |

Capacity and parameter accounting for C=128: R1 is 16C active elements with 643,016 total /
82,432 trainable parameters; R2 is 16C with 724,936 / 164,352; R4-SD is 8C with 856,008 /
295,424; matched pooling is 8C with 1,102,536 / 541,952. R2 does not halve total active
feature volume; it preserves 16C using state plus detail banks.

Per-system and block-offset reports:

```text
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio1_state_detail/evaluation.json
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio2_state_detail/evaluation.json
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio4_state_detail/evaluation.json
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio4_matched_pooling/evaluation.json
```

These files contain per-frame, per-block-offset, per-system, bond/contact/clash, velocity/
acceleration, RMSF, ACF, frequency, and boundary metrics. Runtime JSON files contain spatial-only,
temporal-only, end-to-end, samples/s, token/s, wall time, and peak allocated/reserved memory.

## 7. State/detail diagnostics

```text
repeated-static detail norm: exactly zero before and after the learned bottleneck (unit tests)
repeated-static maximum coordinate motion: exactly zero under zero-detail decode (unit tests)
dynamic R2 detail norm/distribution: holdout raw h=0.160897, encoded h=0.191008, decoded h=0.680709;
  valid fraction=1.0 and zero fraction=0.0 (norm summaries, not a fitted distribution)
dynamic R4 detail norm/distribution: holdout raw h=0.163351, encoded h=0.238345, decoded h=0.623712;
  valid fraction=1.0 and zero fraction=2.96e-08 (norm summaries, not a fitted distribution)
R2 full versus detail-zero decoded metrics: future RMSD 12.228289 / 12.275951 and dRMSD
  11.139406 / 11.230306 (full / detail-zero)
R4 full versus detail-zero decoded metrics: future RMSD 12.113563 / 12.166277 and dRMSD
  10.978540 / 11.085510 (full / detail-zero)
matched-pooling bank utilization: bank A h=1.618426, bank B h=1.584718; no detail semantics
```

## 8. Failures, warnings, and interpretation limits

- Two early remote attempts were not accepted as T0 evidence: the first exposed a pre-training
  one-step unfreeze schedule-boundary bug; the second completed R1 but exposed resume mismatch
  caused by dynamic per-epoch batch counts. The runner was corrected to precompute all 30 epoch
  batch counts (total 4,976), then the final run above completed all controls and resume checks.
- The full suite retains one pre-existing `torch.load(weights_only=False)` FutureWarning in the
  legacy multiframe test; no new test failure occurred.
- T0 uses one seed and three selected systems/nine trajectories. The apparent matched-pooling
  advantage in future RMSD/dRMSD is not a production recommendation and is confounded by its
  natural larger parameter count.
- The source data root is absent in the portable copied container, but exact clip bytes,
  provenance, indexes, and audit hashes are present; this is a path portability note, not a
  semantic split mismatch.

Do not use these three systems to choose the production ratio. The review decision concerns code
correctness and whether T0 behavior is sane enough to justify T1.

## 9. Operator decision

Leave exactly one status after review:

```text
APPROVE_T1
REQUEST_FIX_AND_REPEAT_T0
REJECT_CURRENT_CODEC_DESIGN
```

Operator status: PENDING

Operator notes:

```text
pending operator review
```
