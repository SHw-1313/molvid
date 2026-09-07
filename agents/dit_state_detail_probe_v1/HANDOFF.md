# HANDOFF — R2/R4 state-detail latent DiT probe v1

Append evidence; do not delete prior entries. Do not mark a task complete without a path, command,
or observed result.

## Initial state

```text
date:
repository:
worktree:
branch:
HEAD:
remote:
git status:
source commit comparison:
active T1 checkout/processes:
Python/PyTorch/CUDA:
visible and occupied GPUs:
```

## Required implementation record

```text
completed task IDs:
changed source files:
changed tests/configs/scripts:
model contract version/hash:
R2 codec checkpoint/hash used for smoke:
R4 codec checkpoint/hash used for smoke:
latent-stat source/hash:
data manifest/hash:
```

## Required commands

Record exact commands, including entry into `enter-container`, activation of `torch-ito`, working
directory, CUDA mapping, configuration, seed, output root, and exit status.

```text
preflight:
focused tests:
full tests:
compile/source checks:
R2 smoke:
R4 smoke:
checkpoint resume:
report generation:
```

## Required evidence table

| Evidence | R2 | R4 |
|---|---|---|
| K and packed shapes | | |
| total/trainable parameters | | |
| initial/final smoke loss | | |
| per-field RF losses | | |
| finite/nonzero gradients | | |
| frozen codec hash before/after | | |
| resume step | | |
| trunk tokens/s | | |
| end-to-end samples/s | | |
| peak allocated/reserved GiB | | |
| 8/16-step decode finite | | |
| observed clamp exact | | |

## Deviations and failures

For every deviation, failed attempt, warning, or unavailable dependency, record:

```text
task ID:
observed fact:
impact:
decision:
artifact/log path:
whether acceptance changed: no/yes (yes requires operator approval)
```

Do not weaken thresholds or silently replace a failed backend.

## Final handoff

```text
implementation gate: PASS / FAIL / BLOCKED
CPU correctness gate: PASS / FAIL / BLOCKED
bounded CUDA gate: PASS / FAIL / NOT_RUN
scientific R2/R4 comparison: NOT_RUN
T1 test accessed: NO
real DiT pilot started: NO
phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
operator decision requested: APPROVE_PILOT / REQUEST_FIX / REJECT_DESIGN
```

## Observed execution record — 2026-09-07

Repository isolation:

- Target: /data4/users/sihao/workspace/molvid-dit-state-detail-probe-v1, branch
  feat/dit-state-detail-probe-v1, HEAD 23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541.
- Audited source: /data4/users/sihao/workspace/PVB, branch fix/state-detail-codec-v2-t1-gates,
  same HEAD. Its pre-existing dirty files remained untouched:
  scripts/report_state_detail_codec_v2_t1.py and the two untracked T1 packet scripts.
- Remote: https://github.com/SHw-1313/molvid.git. The source and target worktrees stayed
  separate. No Git reset, checkout-over, clean, rebase, push, or commit was used.
- GPU 7 was idle before each successful CUDA launch; it was not killed, suspended, migrated, or
  preempted. Final audit still showed GPU 7 at 0 MiB and 0 percent utilization.

Completed evidence-backed tasks: D000-D004, D010-D064. D100-D104 remain blocked pending later
operator authorization. Source files added or changed:

- module/state_detail_latent_adapter.py
- module/latent_rectified_flow.py
- module/molecular_dit.py
- trainer/dit_trainer.py
- evaluation/dit_evaluation.py
- config/dit_state_detail_probe.yaml
- scripts/run_state_detail_dit_smoke.py
- package exports in module/__init__.py, trainer/__init__.py, evaluation/__init__.py
- focused tests in tests/test_state_detail_latent_adapter.py, tests/test_latent_rectified_flow.py,
  tests/test_molecular_dit.py, tests/test_dit_trainer.py, and tests/dit_test_utils.py

Contract record:

- R2: K=8, state/detail scalar [8,N,128], vector [8,N,3,128], packed scalar [8,N,256],
  packed vector [8,N,3,256].
- R4: K=4 with the same C=128 and packed channel policy.
- Smoke model policy was identical for both ratios: Dh=256, Dv=128, depth=4, heads=8,
  FFN multiplier=4, dropout=0. Parameter count was 8,679,680 for both.
- Model backend contract: dense block-level spatial attention plus bidirectional per-atom
  temporal attention; reported complexity is sum_s K*M_s^2 + sum_n K^2. No all-atom/time dense
  attention, coordinate metadata, radius graph, distance, contact, or raw Haar input enters the
  DiT batch.

Commands and results, all successful Python commands run after entering enter-container and
activating torch-ito from the target directory:

- Baseline before implementation: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
  -> 115 passed, 1 pre-existing FutureWarning.
- Focused final DiT suite: the four new test files -> 25 passed.
- Legacy/state-detail/evaluator regression selection:
  tests/test_state_detail_codec_v2.py tests/test_codec_training.py
  tests/test_codec_evaluation.py tests/test_luna_review_contracts.py -> 45 passed.
- Final full suite: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
  -> 140 passed, 1 pre-existing torch.load FutureWarning.
- Final compile: python -m py_compile module/state_detail_latent_adapter.py
  module/latent_rectified_flow.py module/molecular_dit.py trainer/dit_trainer.py
  evaluation/dit_evaluation.py scripts/run_state_detail_dit_smoke.py tests/dit_test_utils.py
  tests/test_state_detail_latent_adapter.py tests/test_latent_rectified_flow.py
  tests/test_molecular_dit.py tests/test_dit_trainer.py -> passed.
- Source audit covered forbidden raw-detail/target-coordinate/radius-graph paths and dense
  attention; git diff --check -> passed.

Smoke commands:

    CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 python scripts/run_state_detail_dit_smoke.py --candidate ratio2_state_detail --device cuda --max-steps 1 --records 2 --data-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t0_data/clip_store/valid --output-root outputs/dit_state_detail_probe_v1

    CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 python scripts/run_state_detail_dit_smoke.py --candidate ratio4_state_detail --device cuda --max-steps 2 --records 2 --data-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t0_data/clip_store/valid --output-root outputs/dit_state_detail_probe_v1

The target worktree had the T0 manifest/index but not data.bin. Per D017, both runs read the
immutable T0-valid data.bin in the audited source worktree through target code; no T1 test split
was opened. The two sample IDs were atlas_5e3e_A_R1_w000049 and atlas_5e3e_A_R1_w000050, with
topology ID 5e3e_A::75184f3b9f0f379e56ff670b2f368df8ac39ee225ec62a1d8731a70f5dec1b80.

Smoke evidence:

| Evidence | R2 | R4 |
|---|---:|---:|
| K / packed shapes | 8 / [8,N,256], [8,N,3,256] | 4 / [4,N,256], [4,N,3,256] |
| steps / report wall time | 1 logged + resume to 2 / 21.660 s | 2 total / 10.407 s |
| initial total loss | 3.351095 | 3.480752 |
| final total loss | 3.351095 (report generated before bookkeeping fix; step-2 resume was finite) | 3.151710 |
| state_h / detail_h / state_v / detail_v initial | 3.964561 / 3.873134 / 2.842202 / 2.724482 | 4.173808 / 4.475360 / 2.626466 / 2.647372 |
| state_h / detail_h / state_v / detail_v final | same logged step-1 values; see caveat above | 3.633723 / 3.836304 / 2.461705 / 2.675109 |
| parameters | 8,679,680 | 8,679,680 |
| trunk tokens/s | 13,775.39 | 14,339.10 |
| end-to-end samples/s | 0.09234 | 0.19218 |
| peak allocated / reserved GiB | 4.544 / 10.117 | 2.368 / 10.117 |
| gradients | finite and nonzero in adapter, flow time, all embeddings, and blocks | finite and nonzero in adapter, flow time, all embeddings, and blocks |
| frozen codec / frame encoder | unchanged / unchanged | unchanged / unchanged |
| checkpoint resume | step 1 to step 2 | step 1 to step 2 |
| 8/16-step decode | finite / finite; raw detail absent | finite / finite; raw detail absent |
| observed clamp | exact | exact |

Artifacts are under outputs/dit_state_detail_probe_v1/ratio2_state_detail and
outputs/dit_state_detail_probe_v1/ratio4_state_detail. Generated checkpoints are intentionally
uncommitted. Codec hashes, statistics hashes, adapter hashes, model hashes, exact losses, and
runtime values are in each smoke_report.json.

Deviations and repairs:

1. The first target-launched smoke stopped before data access because direct script execution did
   not put the repository root on sys.path. The runner now resolves its own root.
2. The next attempt reached the frozen encoder and correctly failed because the topology cache had
   not been registered. The runner now calls the existing CPU prepare_batch API before CUDA.
3. Two checkpoint-resume attempts exposed CPU and CUDA RNG-device restoration issues. The trainer
   now restores CPU RNG as CPU bytes and each CUDA state explicitly; the successful R2/R4 runs
   verified resume.
4. The successful R2 report predates the final smoke bookkeeping change, so its final loss field
   is explicitly classified as the logged step-1 value, while its step-2 resume and finite decode
   evidence remain recorded. R4 uses the corrected max-step accounting.
5. One diagnostic host-shell Python invocation was attempted outside enter-container and failed
   immediately with ModuleNotFoundError before importing torch or touching project data. All
   successful Python, test, compile, smoke, and evaluation commands ran inside enter-container
   with torch-ito. Acceptance criteria were not changed.

Final gate classification:

- implementation gate: PASS
- CPU correctness gate: PASS
- bounded CUDA gate: PASS WITH THE RECORDED R2 REPORT CAVEAT
- scientific R2/R4 comparison: NOT RUN
- real DiT pilot: NOT STARTED
- T1 test accessed: NO
- phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
```
