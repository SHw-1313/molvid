# OPERATOR REVIEW — R2/R4 state-detail latent DiT probe v1

Status: DRAFT

This packet is for implementation and bounded-smoke review. It must not claim a scientific codec
winner because the real R2/R4 DiT pilot is outside the current authorization.

## 1. Exact source state

```text
branch:
base commit:
review commit or worktree hash:
git status:
diff summary:
```

## 2. Scope audit

```text
active T1 checkout modified: NO/YES
T1 outputs modified: NO/YES
T1 test opened: NO/YES
real DiT pilot started: NO/YES
longest CUDA smoke steps/time:
```

## 3. Architecture contract

Report:

- R2/R4 exact input/output shapes;
- packed state/detail and scalar/vector widths;
- spatial and temporal interaction paths;
- flow-time and physical-time conditioning;
- observation-mask/clamping semantics;
- static topology fields and prohibited fields;
- model, codec, stats, and data contract hashes.

## 4. Correctness results

Include exact test commands and results for:

- normalization and RF equations;
- observation/future isolation;
- SE(3) and isotropic vector noise;
- ragged sample/block isolation;
- checkpoint/config/statistics round-trip;
- existing codec/T1 regression suite.

## 5. Bounded smoke results

| Metric | R2 | R4 |
|---|---:|---:|
| Steps / wall time | | |
| Initial/final total RF loss | | |
| state_h/detail_h/state_v/detail_v loss | | |
| Parameters | | |
| Trunk tokens/s | | |
| End-to-end samples/s | | |
| Peak allocated/reserved GiB | | |
| Frozen codec unchanged | | |
| Resume | | |
| 8/16-step decode finite | | |

## 6. Known limitations

At minimum state whether:

- T1 codec checkpoints were unavailable or provisional;
- latent statistics came from synthetic/T0 rather than approved T1 train data;
- smoke loss is not a generated-trajectory quality comparison;
- block-level dense attention remains a future scaling constraint;
- H=2 and partial-block observation remain unsupported.

## 7. Decision

Leave exactly one proposed status:

```text
APPROVE_PILOT
REQUEST_FIX
REJECT_DESIGN
```

Proposed status: PENDING

Operator notes:

```text
pending review
```

## Observed review packet — 2026-09-07

Exact source: feat/dit-state-detail-probe-v1 at HEAD 23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541.
The audited T1 source checkout stayed read-only and its pre-existing dirty files were preserved.

### Observed implementation state

The target branch is uncommitted and contains the focused adapter, rectified-flow objective,
factorized molecular DiT, trainer/checkpoint protocol, evaluation report plumbing, configuration,
smoke runner, and focused tests. No generated checkpoint is intended for commit. The source T1
checkout, T1 scripts, codec implementation, codec checkpoints, and T1 output root were not
modified.

The shared model contract is:

- R2-SD: K=8, state_h/detail_h [8,N,128], state_v/detail_v [8,N,3,128], packed scalar
  [8,N,256] and vector [8,N,3,256].
- R4-SD: K=4, state_h/detail_h [4,N,128], state_v/detail_v [4,N,3,128], packed scalar
  [4,N,256] and vector [4,N,3,256].
- State and detail share one sequence location; neither candidate doubles the sequence length.
- The smoke model uses scalar width 256, vector width 128, depth 4, 8 heads, FFN multiplier 4,
  and dropout 0. The backend is factorized block spatial attention plus bidirectional per-atom
  temporal attention, with complexity sum_s K*M_s^2 + sum_n K^2.
- Conditions include flow time, physical block time/span, ratio, observation flag, atom/block/
  component IDs, and coordinate-derived frame-0 origin only for H>0. Raw detail, target
  coordinates, future-derived metadata, dense K*N attention, and learned xyz mixing are excluded.
- H=0, H=4, and H=8 are implemented. Partial blocks and H=2 are rejected.

### Verification evidence

Commands were run inside enter-container with the torch-ito environment:

- Baseline before implementation: 115 passed, one pre-existing torch.load FutureWarning.
- Focused DiT tests: 25 passed.
- Legacy/state-detail/evaluator regression selection: 45 passed.
- Final full suite: 140 passed, one pre-existing torch.load FutureWarning.
- py_compile over all new Python files passed.
- git diff --check passed.

The focused tests cover R2/R4 packing and exact K values, no token doubling, raw-detail exclusion,
mask-aware normalization and inverse normalization, arbitrary-rank flow equations, equal four-field
loss weighting, ragged inputs, physical-time conditioning, H-prefix conversion and partial-block
rejection, future-mutation isolation, observed clamping, scalar invariance, vector equivariance
including a same-noise rotation fixture, axis-preserving projections, deterministic 8/16-step
sampling, generated-latent decoding with raw detail absent, and checkpoint contract rejection.

### Bounded CUDA smoke

GPU 7 was idle before both launches; no running process was killed, suspended, migrated, or
preempted. Both runs used the immutable audited-source T0-valid payload, records 2, H=4, and the
same model-size policy. The payload is not a T1 test split.

| candidate | K | steps | initial total | final total | parameters | 8/16 decode | report |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ratio2_state_detail | 8 | 1 plus resume to 2 | 3.351095 | 3.351095 | 8,679,680 | finite/finite | outputs/dit_state_detail_probe_v1/ratio2_state_detail/smoke_report.json |
| ratio4_state_detail | 4 | 2 plus resume check | 3.480752 | 3.151710 | 8,679,680 | finite/finite | outputs/dit_state_detail_probe_v1/ratio4_state_detail/smoke_report.json |

The R2 report records the first one-step training result and a successful one-step checkpoint
resume to step 2; its final-loss entry is therefore the recorded step-1 value, not an unrecorded
post-resume loss. The R4 report records two training steps and a successful resume check. This is
an execution-record caveat, not a changed acceptance threshold.

R2 field losses (state_h, detail_h, state_v, detail_v) were
3.964561, 3.873134, 2.842202, 2.724482. R4 field losses were initially
4.173809, 4.475360, 2.626467, 2.647372 and finally
3.633723, 3.836304, 2.461705, 2.675109. Gradients were finite and nonzero in every intended
trainable group; frozen codec and frame-encoder hashes were unchanged; checkpoint resume and
finite 8/16-step decoding succeeded; generated latents had raw detail absent and observed values
were clamped.

R2 measured trunk throughput was 13,775.39 tokens/s and end-to-end throughput 0.09234 samples/s;
peak allocated/reserved memory was 4.54/10.12 GiB. R4 measured 14,339.10 tokens/s and
0.19218 samples/s; peak allocated/reserved memory was 2.37/10.12 GiB. These are bounded smoke
measurements only and do not rank R2 versus R4.

The reports contain exact sample IDs, topology IDs, codec/statistics/adapter/model hashes, losses,
gradients, resume evidence, memory, throughput, decode finiteness, and provenance. The R2 report
topology IDs were corrected in the report to the exact source-batch IDs after the first successful
run; the runner was then corrected before the R4 run so it records them directly.

### Scope audit and limitations

No approved trained T1 codec checkpoint was available in the target worktree, so the smoke used
randomly initialized torchmd_et codec instances only to exercise interfaces and decoding. Latent
statistics are explicitly labeled T0 smoke statistics and are not production train-only
statistics. No scientific generated-trajectory comparison, codec selection, ratio selection,
test access, real T1 pilot, best-of-N result, long rollout, H=2, energy, ensemble, static mixing,
AF3/MSA conditioning, VAE/KL/VQ, scaling study, or later architecture work was run. Evaluation
plumbing separates codec oracle, generated result, and generation gap and includes latent loss and
runtime fields, but no scientific evaluation claim is made from this smoke.

The initial smoke attempts exposed and repaired only execution issues: direct-script import path,
missing target-worktree T0 payload, topology-cache preparation, and CPU/CUDA RNG restoration.
One host-shell Python inspection was attempted and failed immediately because host Python had no
torch; it read no project data and caused no mutation. All substantive Python work remained inside
enter-container.

Proposed status: PENDING

Implementation gate: PASS.
CPU correctness gate: PASS.
Bounded CUDA execution gate: PASS WITH THE RECORDED R2 REPORT CAVEAT.
Scientific pilot gate: NOT RUN; operator authorization is still required.

phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
scientific DiT pilot: NOT_STARTED
T1 test accessed: NO
