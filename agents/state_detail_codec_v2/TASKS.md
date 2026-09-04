# TASKS — zero-preserving state/detail temporal codec v2

Status legend: `[ ]` pending, `[-]` active, `[x]` complete, `[!]` blocked.

## Preparation

- [x] S200 inspect repository, branch, HEAD, remote, worktree, environment, current codec code,
  and prior three-system artifacts without discarding user changes — source/status, torch-ito
  environment, legacy source contracts, and immutable v1 outputs were audited.
- [x] S201 create/enter `feat/state-detail-codec-v2` from approved commit and update root
  `AGENTS.md` for this phase — target branch created from `045dbb8809e9d7aee418eb355b6477e05088d4fd`;
  root rules now name this phase and the mandatory operator stop.
- [x] S202 record exact old temporal encoder, decoder, `x_anchor`, model/checkpoint contracts,
  tests, and stored R1/R4 behavior before editing — see `outputs/state_detail_codec_v2/preflight/legacy_audit.json`.
- [x] S203 verify the exact TorchMD frame-encoder checkpoint/hash to be shared and frozen across
  all T0 controls — extracted 42-key 128-channel state from immutable v1 TorchMD step 4979;
  hash is recorded in the preflight audit.
- [x] S204 verify the three systems, all nine trajectory replicas, 441/117 clip split, indexes,
  timestamps, and artifact hashes; stop on mismatch — independent torch-ito audit confirmed
  exact IDs/windows/T=16/dt_100ps and zero overlap.

## Algebra and latent contracts

- [x] S210 implement/test reusable orthonormal R1/R2/R4 lifting and exact inverse lifting for h/v
  — strict FP32 identity fixtures pass at rtol=1e-6, atol=1e-7.
- [x] S211 implement structured state/detail latent schema, masks, clocks, coefficient order,
  capacity accounting, and serialization — 18 codec tests cover shapes, masks, clocks, order,
  and contract fields.
- [x] S212 implement explicit per-sample centroid/origin handling and remove every per-atom x0
  bypass from new codec controls — new latents expose only [B,3] origin and no x_anchor.
- [x] S213 version model/checkpoint contracts while preserving explicit legacy loading — v3 new
  contract round-trip and legacy checkpoint/output regression pass.

## State/detail encoder and decoder

- [x] S220 implement R1 C-wide state path and absent/masked detail semantics — shape/capacity and
  static tests pass.
- [x] S221 implement R2 C-wide state plus C-wide zero-preserving detail path — zero and dynamic
  detail tests pass.
- [x] S222 implement R4 C-wide state and three-detail-to-C zero-preserving encoder — exact
  Dmid,D01,D23 ordering and 8C capacity are tested.
- [x] S223 implement R2/R4 zero-preserving detail decoders and inverse lifting — identity and
  masked inverse fixtures pass.
- [x] S224 implement shared framewise equivariant centered-coordinate decoder without additive
  time motion or spatial refiner — SE(3), clock-invariance, and no-anchor tests pass.
- [x] S225 implement static T=1 full-state path and distinguish it from repeated-static T=16
  — T=1 invalid-detail and repeated-static-zero tests pass.

## Matched control

- [x] S230 implement R4 two-head matched pooling with two unstructured C-wide banks — 8C
  capacity and independent-bank gradient tests pass.
- [x] S231 implement its no-anchor four-frame expansion decoder and report natural parameters
  — contract and integration tests pass.
- [x] S232 expose exactly four new control names/configurations without changing legacy names
  — explicit mode validation and legacy default regression pass.

## Correctness tests

- [x] S240 pass lifting roundtrip, ordering, mask, capacity, T=1/T=16, partial/padded block, and
  irregular-clock tests — covered by the 18-test phase suite.
- [x] S241 pass exact repeated-static zero coefficient, zero latent-detail, zero decoded-feature
  motion, and zero coordinate-motion tests — all three state/detail ratios pass.
- [x] S242 pass moving-input detail utilization and finite/nonzero detail/decoder/coordinate
  gradient tests — scalar/vector/detail/coordinate principal parameters are finite and nonzero.
- [x] S243 pass scalar invariance, vector/coordinate SE(3), translation-origin, frame isolation,
  and sample isolation tests — phase suite passes.
- [x] S244 prove decoder has no target-coordinate argument and no new per-atom x0/reference path
  — API, contract, latent, and post-encode target-mutation tests pass.
- [x] S245 pass new checkpoint save/resume/config roundtrip and legacy output/checkpoint regression
  — phase and legacy tests pass.
- [x] S246 pass the full relevant suite, `py_compile`, source audit, and `git diff --check`
  — 107 passed; py_compile and diff check clean.

## T0-A bounded training smoke

- [x] S250 freeze the four-control micro protocol before viewing outcomes — runner fixes seed,
  FP32, 500 steps, common frozen frame checkpoint, and immutable three-clip selection.
- [x] S251 run one-batch/single-clip overfit for R1-SD, R2-SD, R4-SD, and R4-Matched-Pooling
  — remote CUDA run completed for all four.
- [x] S252 verify loss decrease, finite/nonzero new-module gradients, no hidden spatial updates,
  checkpoint step/resume, runtime, memory, and detail utilization — all four remote micro result
  records report pass, >30% loss reduction, unchanged frozen encoder, and resume pass.
- [x] S253 run one explicit unfrozen-encoder forward/backward smoke without starting a fifth
  comparison experiment — `outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/unfreeze_smoke.json`
  reports `status=passed`, 41 nonzero spatial gradients, and changed frame-encoder state.

## T0-B three-system/nine-trajectory comparison

- [x] S260 freeze manifest, common encoder hash, loader coverage, optimizer, schedule, evaluator,
  seed, output paths, and device assignment before launch — final protocol is recorded in
  `outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/protocol.json`; manifest
  contract is `.../manifest_contract.json`, with common frame-encoder SHA256
  `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a` and CUDA A100 device data.
- [x] S261 train R1-SD for exactly 30 complete epochs — final result is in
  `.../ratio1_state_detail/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S262 train R2-SD for exactly 30 complete epochs — final result is in
  `.../ratio2_state_detail/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S263 train R4-SD for exactly 30 complete epochs — final result is in
  `.../ratio4_state_detail/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S264 train R4-Matched-Pooling for exactly 30 complete epochs — final result is in
  `.../ratio4_matched_pooling/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S265 verify every run's exact coverage, complete holdout evaluations, logs, best/final
  checkpoints, and checkpoint resume — all four have 30 epoch rows, exact 441/117 protocol
  coverage, complete holdout evaluation, final/best checkpoints, and resume `4976 -> 4977`.
- [x] S266 generate rate/capacity, reconstruction, dynamics, block-offset/boundary, latent-bank,
  counterfactual, system-level, runtime, and memory reports/plots — aggregate JSON/CSV/Markdown
  plus loss, evaluation, block-detail, and performance plots are in the final run directory.

## Operator packet and mandatory stop

- [x] S270 fill `OPERATOR_REVIEW.md` with exact evidence paths and observed results — packet is
  complete and leaves the operator decision pending.
- [x] S271 append complete changed files, commands, hashes, tests, metrics, deviations, warnings,
  and limitations to `HANDOFF.md` — final dated handoff appended.
- [x] S272 set phase status to `WAITING_FOR_OPERATOR_REVIEW` — recorded in the operator packet,
  aggregate report, and final handoff.
- [x] S273 stop without preparing or running T1 or any DiT/static/full-data work — no T1 manifest
  or later architecture task was started.

## T1 — not authorized until explicit later approval

- [!] S300 materialize the 64-system 48/8/8 split and manifest
- [!] S301 benchmark 200 steps and freeze the 18–20 GPU-hour per-control budget
- [!] S302 run the four one-seed T1 controls
- [!] S303 evaluate validation, freeze selection rules, then evaluate test

S300–S303 remain blocked even if S200–S273 pass. Only a later explicit operator approval may
change their status.
