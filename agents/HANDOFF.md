# Multi-frame codec v1 handoff

## Prepared state

- Planning repository: official PVB clone.
- Inspected upstream commit: `c08e5e3cd49d45c6d748387e78224843bd356f50` (`submit version`).
- Remote inspected: `https://github.com/yaledeus/PVB.git`.
- No implementation code has been changed by the planning pass.
- Added only the agent guide, project-scoped Luna worker configuration, and the four planning files.
- Updated all six files so physical time is a first-class codec condition: continuous timestamps, relative-time attention bias, time-bucket batching, timestamped latents/decoder queries, and bucket-aware losses/evaluation are now binding.
- The planning environment did not expose a `codex` executable, so the command below is based on current official Codex CLI syntax and must be run in the user's Codex-enabled shell.

## What the next worker should do

1. Read `AGENTS.md`, `PLAN.md`, `TASKS.md`, and `DECISIONS.md` completely.
2. Run T00 and append the real environment/baseline evidence to this file.
3. Implement T01 through T08 sequentially in small commits or reviewable diffs.
4. Run only synthetic/CPU tests until the local container and GPU state are confirmed.
5. Stop before full training. T09 is a smoke gate; T10 prepares the operator-run pilot.

The worker must not collapse the task back to pair prediction. The implementation is wrong if the learned path accepts only `x0/x1`, loops over frames through `realization()`, lets the decoder see later target coordinates, treats frame index as physical time, equates 100 ps with 1 ns, or fabricates fine-timescale labels by interpolating coarse trajectories.

## Exact initial worker prompt

Use this in an interactive Codex session, or as the prompt to `codex exec`:

> Act as the sole implementation worker for Scheme 1A multi-frame codec v1. Read `AGENTS.md` and all four files in `docs/agent/multiframe_codec_v1/` before editing. Start with T00, then execute the first unblocked tasks in dependency order. Preserve all existing PVB behavior and implement an additive packed-clip path with batched PVB spatial encoding, causal SE(3)-safe temporal compression, and joint multi-frame decoding. Treat `time_ps`/`delta_time_ps` as first-class conditions: implement continuous relative-time attention bias, timestamped latents, target-time decoder queries, homogeneous sampling-interval buckets, correct physical finite differences, and bucket-stratified metrics. Never equate 100 ps with 1 ns or interpolate coarse trajectories into fine-timescale labels. Do not implement a generative trunk, rollout, KL/VQ, residue pooling, or AF3/MSA conditioning. Use `enter-container` and the `torch-ito` environment for Python/tests, make no dependency installs or network calls, and serialize GPU work. After every completed task, update `TASKS.md` with evidence and append commands, tests, changed files, and blockers to `HANDOFF.md`. Complete all CPU/synthetic implementation gates that are safe; stop and report precisely if real data, timestamps, checkpoint, container syntax, or GPU access blocks the next gate. Do not claim scientific success without the G5 pilot.

## Recommended non-interactive launch

From a shell where `codex` is installed, set `PVB_REPO` to the writable repository root and run:

```bash
PVB_REPO=/absolute/path/to/PVB
codex exec \
  -C "$PVB_REPO" \
  --model gpt-5.6-luna \
  -c 'model_reasoning_effort="max"' \
  --sandbox workspace-write \
  --ask-for-approval never \
  "Act as the sole implementation worker for Scheme 1A multi-frame codec v1. Read AGENTS.md and all files in docs/agent/multiframe_codec_v1/ completely, then execute the HANDOFF.md instructions and the first unblocked TASKS.md items in dependency order. Treat physical time as a first-class model condition: carry time_ps/delta_time_ps through temporal attention, compressed latents, target-time decoding, finite-difference losses, time-bucket batching, checkpoints, and stratified evaluation; never equate 100 ps and 1 ns or invent fine labels from coarse interpolation. Update TASKS.md, DECISIONS.md when needed, and HANDOFF.md with evidence. Preserve existing PVB behavior; do not install dependencies, use network access, or run full training."
```

Why these flags:

- `-C` sets the repository before Codex reads project instructions.
- the model and reasoning override pin the requested Luna Max worker;
- `workspace-write` permits repository edits while retaining a sandbox;
- `never` makes unapproved installs/escapes fail rather than block an unattended run.

Do not use `--dangerously-bypass-approvals-and-sandbox` for this task.

## Interactive-session instruction

The project contains `.codex/agents/worker.toml`, which overrides the built-in `worker` role with `gpt-5.6-luna` and max reasoning for this repository. In a Codex session opened at the repository root, say:

> Spawn/use the project-scoped `worker` agent as the sole writer. Give it the exact initial worker prompt in `docs/agent/multiframe_codec_v1/HANDOFF.md`. Wait for it to finish, then review its diff and report completed task IDs, tests, and blockers. Do not spawn additional write agents.

If the client does not expose custom agents, run the non-interactive command above; do not silently substitute another model.

## Inputs the operator may need to provide later

Do not block T01, T04, T05, T06, or synthetic T07 on these values. They are needed for later real-data/GPU gates:

```text
WRITABLE_PVB_REPO=/data4/users/sihao/workspace/PVB
PVB_ENCODER_CKPT=/data1/repo/PVB/ckpt
ATLAS_ROOT=/data1/repo/BioKinema/data_atlas
MISATO_ROOT=/data1/repo/PVB/download_data/MISATO
PROCESSED_DATASET_ROOT=/data4/users/sihao/data/pvb_cross_dataset_20260810
CODEC_GPU_IDS=0,1,2,3
ATLAS_NATIVE_DT_PS=10
MISATO_NATIVE_DT_PS=80
```

The worker must not commit these local paths.

## Evidence log template

The worker should append entries, not replace this prepared state.

```text
### YYYY-MM-DD — Txx

- Status:
- Files changed:
- Commands:
- Tests/results:
- Decisions added or changed:
- Blockers/next task:
```

## Pilot result template


| Time bucket | Clip span | Model             | Temporal ratio | Physical latent interval | Future dRMSD | Velocity RMSE | Acceleration RMSE | Bond RMSE | Clash rate | HF power retention | Peak GPU | Samples/s |
| ----------: | --------: | ----------------- | -------------: | -----------------------: | -----------: | ------------: | ----------------: | --------: | ---------: | -----------------: | -------: | --------: |
|      100 ps |    1.5 ns | framewise control |              1 |                   100 ps |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|      100 ps |    1.5 ns | temporal codec    |              1 |                   100 ps |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|      100 ps |    1.5 ns | temporal codec    |              4 |                  ~400 ps |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|        1 ns |     15 ns | framewise control |              1 |                     1 ns |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|        1 ns |     15 ns | temporal codec    |              1 |                     1 ns |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|        1 ns |     15 ns | temporal codec    |              4 |                    ~4 ns |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |

Final go/no-go decision: **TBD after G5; no result has been fabricated in this planning pass.**

### 2026-08-18 — T00/T01/T02 data worker

- Status: T00 complete; T01 core contract and T02 ATLAS/MISATO path implemented; full half-data materialization running next. mdCATH is blocked because no raw root or timestamp metadata was supplied in this task.
- Files changed: `utils/bio_utils.py` (NumPy 2.x dtype fix); `data/clip_dataset.py`; `data/trajectory_clips.py`; `scripts/preprocess_trajectory_clips.py`; `tests/test_clip_data.py`; this ledger.
- Baseline: branch `main`, upstream commit `c08e5e3cd49d45c6d748387e78224843bd356f50`, dirty state initially only untracked `agents/`. Valid invocation is `enter-container`, then `source /home/sihao/miniforge3/bin/activate torch-ito`. Environment: Python 3.11.15, NumPy 2.4.6, PyTorch 2.5.1+cu121, CUDA 12.1, 8 visible NVIDIA A100-SXM4-80GB GPUs. `python -c 'import torch; import data; from module.model import dyVAE; print(...)'` passed.
- Commands/tests: `python -m pytest -q tests/test_clip_data.py` — 6 passed; `python -m py_compile data/clip_dataset.py data/trajectory_clips.py scripts/preprocess_trajectory_clips.py utils/bio_utils.py tests/test_clip_data.py` — passed.
- Real-data evidence: one ATLAS system with 3 replicas and `clip_len=4`, `source_stride=1000` produced 6 clips; one MISATO system with `clip_len=4`, `source_stride=1` produced 25 clips. Both stores round-tripped through `ClipMMapDataset` and collated to packed `[4, N_total, 3]`; ATLAS native XTC time was verified as 10 ps, and MISATO uses the explicit 80 ps value from this handoff because the HDF5 itself has no timestamp field. A default-style ATLAS one-system run (`clip_len=16`, `window_stride=16`, `source_stride=10`) produced 186 clips, `dt_100ps`, `[16, 4508, 3]`, and 185,411,817 compressed data bytes.
- Decisions added or changed: no architectural decision changed. The new path is additive and does not alter the legacy `x0/x1` preprocessors. ATLAS 10 ps frames are decimated to an explicitly labeled 100 ps bucket; no coarse-to-fine interpolation is used. MISATO remains an explicitly labeled 80 ps bucket.
- Blockers/next task: write the requested half-system stores to `PROCESSED_DATASET_ROOT=/data4/users/sihao/data/pvb_cross_dataset_20260810` using `--fraction 0.5`, then validate manifests/counts and update this file. `/data4/scratch/sihao` is read-only inside the container; it was used only for disposable smoke output via `/tmp` instead.

### 2026-08-18 — T01/T02 full ATLAS and MISATO materialization

- Status: complete for the requested ATLAS and MISATO half-data; mdCATH remains blocked because this task supplied neither a raw root nor timestamp metadata. The static T=1 loading line remains pending as a separate integration task; the clip schema itself validates and serializes static T=1 records without frame replication.
- Output: `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps` and `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/misato/dt_80ps`. Both use `schema_version=pvb.clip.v1`, `storage_format=npz-v1`, three non-overlapping system splits, and compressed per-record NumPy payloads with JSON metadata.
- ATLAS result: fraction `0.5`, seed `20260810`, selected `546/94/76` train/valid/test systems, `101556/17484/14136` records, `133176` total, `systems_skipped=0`. Native XTC timestamps were checked at 10 ps; this run uses source stride 10, producing an explicit 100 ps bucket with no interpolation.
- MISATO result: official split preserved, fraction `0.5`, seed `20260810`, selected `6985/790/822` train/valid/test systems, `41910/4740/4932` records, `51582` total, `systems_skipped=0`. MISATO HDF5 has no timestamp field, so the handoff-supplied verified native interval of 80 ps is recorded explicitly; source stride is 1 and no interpolation is used.
- Commands: full runs used `python scripts/preprocess_trajectory_clips.py atlas --root /data1/repo/BioKinema/data_atlas --output-root <staging> --fraction 0.5 --seed 20260810 --clip-len 16 --window-stride 16 --source-stride 10` and the corresponding MISATO command with `/data1/repo/PVB/download_data/MISATO` and `--source-stride 1`. After staging, every split was read from the final target with `ClipMMapDataset`; first/middle/last records were validated and two-record batches were collated.
- Validation: target-side readback passed for all six stores, including index lengths, `stats.json` storage format, `T=16`, time buckets, physical deltas, NumPy array fields, packed coordinates, and bond offsets. Final clip tree is approximately `84G`; no temporary staging directory remains.
- Files changed: `data/clip_dataset.py` (versioned npz-v1 writer/reader, canonical validation/collation; reader returns NumPy arrays), `data/trajectory_clips.py`, `scripts/preprocess_trajectory_clips.py`, `utils/bio_utils.py`, `tests/test_clip_data.py`, `agents/TASKS.md`, and this handoff.
- Tests: `python -m pytest -q tests/test_clip_data.py` — `9 passed`; `python -m py_compile data/clip_dataset.py data/trajectory_clips.py scripts/preprocess_trajectory_clips.py utils/bio_utils.py tests/test_clip_data.py` — passed. Existing legacy pair preprocessors were not changed except for the NumPy 2.x `np.compat.long` dtype fix.
- Next task: T03 task-aware dynamic batching, followed by the model codec gates. Do not treat the generated clip stores as evidence of model quality; no training or G5 result was run.

### 2026-08-18 — T02 static T=1 integration

- Status: complete for the existing ANI1x, PCQM4Mv2, and PDBBind static block stores. mdCATH was intentionally skipped per operator instruction and remains outside this handoff.
- Implementation: `legacy_static_record_to_clip()` and lazy `StaticClipDataset` in `data/clip_dataset.py` adapt legacy gzip-JSON `x0`/`b0` records to canonical static clips with `x=[x0]`, `bpos=[b0]`, `time_ps=[0]`, empty `delta_time_ps`, `task=STATIC`, and `time_bucket_id=static`. The adapter does not rewrite the old stores and uses NumPy view expansion rather than replicating frames to `T=16`.
- Metadata: existing atom order, coordinates, block positions, bond indices, masks, and PDBBind `edge_mask` are preserved. Legacy stores do not contain original atom names/indices, so the adapter records deterministic stored-order identities and labels their provenance as `legacy_block_order`; block ids are stably recovered from block-center position/type pairs.
- Files changed: `data/clip_dataset.py`, `data/__init__.py`, `tests/test_clip_data.py`, `agents/TASKS.md`, and this handoff.
- Unit tests: `python -m pytest -q tests/test_clip_data.py` — `11 passed`; `python -m py_compile data/clip_dataset.py data/__init__.py tests/test_clip_data.py` — passed; `import data` plus `StaticClipDataset` import — passed.
- Real-data validation against `/data4/users/sihao/data/pvb_cross_dataset_20260810/blocks`: all train/valid/test stores for all three sources loaded successfully. Counts/shapes were ANI1x `2050623/285072/161913` with first-record shapes `(1,10,3)/(1,13,3)/(1,12,3)`, PCQM4Mv2 `1351433/168929/168930` with `(1,18,3)/(1,17,3)/(1,17,3)`, and PDBBind `6413/367/167` with `(1,2169,3)/(1,2241,3)/(1,1333,3)`. Every checked record validated as `T=1`; representative batches had shape `[1, N_total, 3]` and static task id `0`.
- Next task: T03 task-aware dynamic batching. No model training or quality claim was made.

### 2026-08-18 — T03 task-aware dynamic batching

- Status: complete. The new clip path now forms deterministic homogeneous dynamic batches with an initial `T*N` bound, independent task/bucket mixture weights, DDP batch sharding, and batch diagnostics. No model training or quality claim was run.
- Files changed: `data/clip_batching.py`, `data/clip_dataset.py`, `data/__init__.py`, `tests/test_clip_batching.py`, `agents/TASKS.md`, and this handoff.
- Sampler: `TaskAwareClipBatchSampler` groups by `(task, frames, time_bucket_id)`, packs records while the sum of `frames * atoms` stays below `max_tokens`, and supports `task_weights`, `time_bucket_weights`, `batches_per_epoch`, `replacement`, `seed`, and `set_epoch`.
- DDP: one global deterministic batch schedule is generated and assigned by `global_batch_id % num_replicas`; ranks therefore do not receive the same logical batch and no padding duplicate is introduced.
- Metadata/logging: indexed trajectory stores avoid payload decompression for atom/frame/bucket specs; PDBBind static specs read actual atom counts because its legacy property zero is an adjacency budget. `summarize_clip_batch` and `ClipBatchLogger` report atoms, frames, effective tokens, task type, native delta time, and physical clip span.
- Commands/tests: inside `enter-container` with `torch-ito`, `python -m py_compile data/clip_batching.py data/clip_dataset.py data/__init__.py tests/test_clip_batching.py` passed; `python -m pytest -q` passed with `16 passed`. The T03 suite covers deterministic packing/bounds, zero-weight task and bucket exclusion, two explicit DDP ranks with disjoint batch sets, logs, indexed store readback, and DataLoader collation.
- Decisions added or changed: none. This is additive; legacy pair training and the existing `DynamicBatchWrapper` remain unchanged.
- Blockers/next task: no T03 blocker. Proceed to T04 batched PVB frame encoder; do not start full training before the later codec gates.

### 2026-08-18 — T04 batched PVB frame encoder

- Status: complete for the CPU/synthetic T04 gate. This is an additive spatial adapter; the legacy `dyVAE` path was not modified.
- Files changed: `module/multiframe_codec.py`, `module/__init__.py`, `tests/test_multiframe_codec.py`, `tests/test_multiframe_codec_equivariance.py`, `tests/test_multiframe_codec_optimized.py`, `agents/TASKS.md`, and this handoff.
- Implementation: `PVBFrameEncoder` expands per-atom metadata across time, assigns `graph_id = frame_id * B + abid`, vectorizes frame-offset covalent bond replication, builds distance edges in the same graph-id space, unions missing covalent edges, calls one shared `TorchMD_VQ_ET`, and restores `[T,N_total,C]`/`[T,N_total,3,C]` outputs. `FrameGraphBatch` exposes all graph/topology evidence. Checkpoint initialization returns matched/missing/unexpected/shape-mismatch keys.
- Commands/results: `python -m py_compile module/multiframe_codec.py module/__init__.py tests/test_multiframe_codec.py tests/test_multiframe_codec_equivariance.py tests/test_multiframe_codec_optimized.py` passed. Final `python -m pytest -q` passed with `21 passed` after the default-backend fallback fix.
- Coverage: T=1 and T=16 shapes, one-call behavior, frame/sample isolation, covalent offsets, scalar translation invariance, vector rotation equivariance, checkpoint key reporting, and default neighbor-backend behavior all pass in CPU synthetic tests.
- Environment/blocker: the current `torch-ito` installation exposes `TorchMD_VQ_ET.distance` but lacks `get_neighbor_pairs_kernel`, producing `NameError`. `neighbor_backend=auto` catches this and uses the dependency-free dense CPU finder; explicit `neighbor_backend=optimized` remains strict. No GPU smoke, real-data encoder run, or training was started.
- Decisions/next: no new architectural decision. Proceed to T05 causal temporal codec blocks; retain the optimized-kernel issue as an environment note for later GPU profiling.

### 2026-08-18 — neighbor backend repair and T05 temporal gate

- Status: the environment blocker found after T04 was repaired; T05 causal temporal feature gate is complete. T06 coordinate decoder and T07 trainer remain pending.
- Neighbor repair: `utils/torchmd_utils.py` now defines the missing `get_neighbor_pairs_kernel` through the installed non-periodic `torch_cluster.radius_graph`. `OptimizedDistance` returns the expected edge index/vector/weight/num-pairs tuple; periodic box requests fail explicitly. Explicit optimized neighbor construction passed a batched isolation check, and the full CPU suite passed.
- Original backbone: PVB's existing `TorchMD_VQ_ET` remains the spatial backbone; no replacement with an unrelated model was made. The original pair path still uses `module.graph.construct_edges`, while the new adapter uses the repaired optimized distance path.
- T05 implementation: `module/temporal_codec.py` adds physical-time RBF/MLP attention bias, local causal scalar/vector attention, SO(3)-safe vector channel normalization/FFN, ratio 1/2/4 causal compression, right-edge time/mask propagation, target-time causal upsampling, and refinement blocks.
- Tests: `python -m pytest -q tests/test_temporal_codec.py` — `7 passed`; full `python -m pytest -q` — `27 passed` with the existing `torch.load` FutureWarning only.
- Real GPU smoke: on A100 GPU 4, real ATLAS `atlas_5e3e_A_R1_w000000` (`T=16,N=887`) passed spatial optimized-neighbor encoding, ratio-4 temporal encoding, target-time decoding, backward, and AdamW update; latent `(4,887,16)`, decoded `(16,887,16)`, `0.756 s`, peak allocated `803.2 MiB`. This is a smoke result, not a quality or full-training claim.
- Next task: implement T06 joint coordinate decoder/refiner, then T07 losses and the isolated codec trainer before launching a real experiment.

### 2026-08-19 — T06 joint coordinate decoder and refiner

- Status: complete for the T06 CPU/synthetic acceptance gate; T07 is next. The decoder is additive and leaves the existing PVB `TorchMD_VQ_ET` class and legacy model/checkpoint path unchanged.
- Implementation: `module/coordinate_decoder.py` defines `CodecLatent`, `CoordinateDecoderOutput`, `JointMultiFrameDecoder`, and `LatentConditionedSpatialRefiner`. The decoder queries all target timestamps through `CausalTemporalDecoder`, predicts equivariant coordinate displacements from vector latents, and anchors them at explicit frame-0 `x_anchor`. It has no target-coordinate argument.
- Refiner: one existing `TorchMD_VQ_ET` call receives a flattened `[T*N_total]` graph. Static `z/b/batch/edge_index/bond_type` topology is replicated and frame-offset; `edge_weight_0/edge_vec_0` use repeated `x_anchor`, while `edge_weight_t/edge_vec_t` use `x_coarse`. The wrapper returns refined scalar/vector features and an equivariant displacement.
- Validation: `python -m pytest -q tests/test_coordinate_decoder.py` — 5 passed; full `python -m pytest -q` — 33 passed, with only the existing `torch.load(weights_only=False)` warning. Tests cover `[T,N,3]` shapes, target-time sensitivity, signature-level information-leak prevention, SE(3), gradients, one-call batching, a real small-graph TorchMD refiner, and two-clip tiny overfit.
- GPU note: a T06 real-ATLAS GPU attempt was made after the CPU gate, but this container currently reports `torch.cuda.is_available() == False` and NVML initialization failure. No T06 GPU result or coordinate-quality claim is made. The earlier T05 GPU smoke remains recorded above.
- Next action: implement T07 masked physical-unit losses, codec trainer/config/checkpoint round-trip, then continue through the remaining task ledger without pausing at a stage boundary.

### 2026-08-19 — T07 losses, trainer, and checkpoint gate

- Status: complete for the CPU/synthetic acceptance gate; T08 is next. The new path is isolated in `trainer/codec_losses.py`, `trainer/codec_trainer.py`, `train_codec.py`, and `config/codec.yaml`; existing `train.py` and legacy model types remain untouched.
- Losses: coordinate, local/contact pair distance, covalent bond length, explicit-`delta_time_ps` velocity, and centered nonuniform-grid acceleration. Temporal masks require consecutive valid trajectory frames and task id `TRAJECTORY`; static temporal losses return exact zero.
- Normalization/config: train-batch fitting produces per-bucket coordinate/motion scales with minimum-count fallback and epsilon guards. Canonical `time.unit=ps`, continuous time scale, bucket center/tolerance/weight, staged weights, and normalization settings are validated before training. Raw `velocity_raw`/`acceleration_raw` and normalized optimization terms are emitted per bucket, with task metrics.
- Checkpoint: schema `pvb.codec.checkpoint.v1` stores model/optimizer state, step/epoch, validated config, and normalization statistics; schema/missing-field failures are explicit. `PVBCodecModel` composes the existing PVB `TorchMD_VQ_ET` spatial path with T05/T06 modules.
- Tests: `python -m pytest -q tests/test_codec_training.py` — 5 passed; full suite — 38 passed with the existing `torch.load(weights_only=False)` warning. The PVB wrapper completed one CPU optimizer step on a distinct-atom synthetic clip, and checkpoint resume restored parameters and statistics.
- Next action: add T08 evaluation metrics and the ratio-1/no-temporal, ratio-1 temporal, and ratio-4 temporal controls, then proceed to GPU/DDP smoke gates.

### 2026-08-19 — T08 round-trip evaluation and controls

- Status: complete for the tiny-fixture evaluation gate; T09 GPU/DDP smoke is next.
- `evaluation/codec_evaluation.py` reports frame-0/future/all-frame RMSD and dRMSD, bond RMSE, contact error, clash rate, optional torsion change, velocity/acceleration RMSE, FFT frequency retention, latent token count, throughput, wall time, and peak GPU bytes. `eval_codec.py` constructs ratio-1/no-temporal, ratio-1 temporal, and ratio-4 temporal PVB controls from the codec config.
- Every control result is nested under native `time_bucket_id` with native delta, physical span, and ratio-derived latent interval. No cross-bucket mean is emitted; Markdown repeats that constraint.
- Tests: `python -m pytest -q tests/test_codec_evaluation.py` — 2 passed; full suite — 40 passed. A fixture report separates 80 ps, 100 ps, and static buckets and distinguishes anchor/no-temporal from a perfect temporal predictor.
- Next action: run reproducible one-device `T=16` ratio-4 forward/backward/optimizer/validation/checkpoint smoke, then attempt the two-rank DDP smoke if two idle GPUs are actually available.

### 2026-08-19 — T09 smoke gate

- Initial status before container device-cgroup repair: CPU/synthetic smoke was complete; GPU/NCCL acceptance was environment-blocked.
- `python smoke_codec.py --cpu-fallback --output /tmp/pvb_codec_smoke.json` passed on the actual `PVBCodecModel` with `T=16`, temporal ratio 4, `dt_100ps` and `dt_1ns` synthetic clips, optimizer step, validation, checkpoint save/load, `resumed_step=2`, `wall_time_s=0.1221`, and `peak_memory_bytes=0` (CPU).
- `python smoke_codec.py --ddp` passed with two CPU Gloo ranks. The script performs rank-specific input, one DDP backward/update, and an all-reduce. The smoke code uses one flattened PVB spatial call per batch; no per-frame spatial model loop is introduced.
- Initial GPU attempt before container repair: `python smoke_codec.py --device cuda:4` failed before model construction because `torch.cuda.is_available() == False`; NVML returned `Unknown Error`.
- Root-cause check: host-side `nvidia-smi -L` and `nvidia-container-cli info` both see all eight A100s, while inside `sihao-dev` opening `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia4` returns `EPERM` despite mode `0666`; direct `cuInit` returns `CUDA_ERROR_NO_DEVICE`. The container device-cgroup allow-list is missing.
- Pilot launch/config templates are in `agents/PILOT_COMMANDS.md`; after repair, the commands were executed on GPU 5 and the NCCL smoke used GPUs 0 and 5.

### 2026-08-19 — T10 pilot handoff

- Status: fixed-seed pilot execution complete after the container GPU repair; no final go/no-go claim is made from the pilot sample alone.
- `agents/PILOT_COMMANDS.md` contains fixed-seed commands for the three G5 controls, per-bucket runs (dt100/dt80/1ns), evaluation checkpoints, smoke checks, and the result table.
- `train_codec.py` supports dataset-root, seed, control, and checkpoint overrides; `eval_codec.py` accepts one checkpoint per control. Paths are rooted in `PROCESSED_DATASET_ROOT` from this handoff.
- Three checkpoints were trained for 1000 steps with 256-batch normalization fitting, and `outputs/g5/codec_eval.json`/`.md` contain the 32-batch native-bucket report (48 clips per control across dt100/dt80). Results are observations for threshold review, not a final go/no-go claim.
- Next action: review the pilot metrics against project quality thresholds and extend validation or launch the next experiment only after that decision.

### 2026-08-19 — T10 pilot execution

- `outputs/g5/ratio1_no_temporal/codec_step_00001000.pt` and `outputs/g5/ratio1_temporal/codec_step_00001000.pt` completed 1000 fixed-seed steps; `ratio4_temporal/codec_step_00001000.pt` completed likewise.
- `outputs/g5/codec_eval.json` and `outputs/g5/codec_eval.md` were generated on GPU 5 with 32 validation batches, native `dt_100ps`/`dt_80ps` stratification, 48 clips per control, and nonzero GPU peak accounting.
- A full quality/go-no-go decision is intentionally deferred until the report is compared with the project thresholds.

### 2026-08-20 - Training logs and visualization

- Status: complete for the requested training diagnostics; the final quality/go/no-go decision is still intentionally deferred to project thresholds.
- Files changed: trainer/codec_trainer.py, train_codec.py, tests/test_codec_training.py, scripts/plot_codec_results.py, agents/PILOT_COMMANDS.md, agents/TASKS.md, and this handoff.
- Logging: CodecTrainer.run(..., log_path=..., log_every=...) appends pvb.codec.train.v1 JSONL records. train_codec.py defaults to <save-dir>/train_metrics.jsonl and exposes --log-path/--log-every.
- Execution: the three fixed-seed controls were rerun on GPU 5 for 1,000 steps. Original outputs/g5/ checkpoints were preserved; logged rerun checkpoints are outputs/g5_logged/ratio*/codec_step_00001000.pt, with 1,000 log records in each outputs/g5/ratio*/train_metrics.jsonl.
- Evaluation/plots: outputs/g5_logged/codec_eval.json and .md were generated with 32 validation batches and 48 clips per control across native dt_100ps/dt_80ps. outputs/g5_logged/plots/ contains loss curves, reconstruction/structure plots, dynamics/PVB-style metric plots, and eval_metrics.csv.
- Metrics visualized: frame-0/future RMSD and dRMSD, future bond RMSE, contact error, clash rate, velocity/acceleration RMSE, and FFT frequency retention; the JSON retains the full bucket-stratified report.
- Tests: python -m pytest -q tests/test_codec_training.py - 6 passed; full python -m pytest -q - 41 passed with the existing torch.load warning; changed Python files compile successfully.
- Limitations/next action: loss curves are from same-seed/config diagnostic reruns rather than the original pilot checkpoints; there is no real 1 ns bucket. Compare the report against project quality thresholds before a go/no-go decision.


### 2026-08-20 — Unmodified PVB_origin four-condition baseline comparison

- Status: complete. The untouched reference repository at /data4/users/sihao/workspace/PVB_origin, commit c08e5e3cd49d45c6d748387e78224843bd356f50, was used for all four conditions and remains clean. PDB was excluded.
- Conditions: A static checkpoint direct; B dynamic checkpoint direct; C static checkpoint retrained; D dynamic checkpoint retrained. Checkpoints are the requested top entries under /data1/repo/PVB/ckpt/pdbbind_pretrain/version_1/checkpoint/epoch191_step113472.ckpt and /data1/repo/PVB/ckpt/misato_from_author_pretrain/version_0/checkpoint/epoch1_step4.ckpt, with C/D final checkpoints recorded in their output directories.
- Shared evaluation policy: seed 20260810, max_tokens 80000, max_batches 32, 48 selected validation clips (ATLAS dt_100ps: 20; MISATO dt_80ps: 28), rollout T=16 with 15 sequential transitions, and 10 SDE steps. Training policy uses ubound_per_batch=5000, max_batches=500 per source, max_epoch=1, and no PDB.
- C completed 1000 train batches with 8 original trainer OOM skips and global_step=992; its 1000-batch valid loss mean is 1.815241908967495. D completed 1000 train batches with global_step=1000; its 1000-batch valid loss mean is 1.3128217451274395.
- A/B/C/D structure and motion results are aggregated under outputs/pvb_origin_baselines/summary/summary.json and summary.md. loss_curves.{png,pdf}, eval_reconstruction.{png,pdf}, eval_dynamics.{png,pdf}, eval_metrics.csv, training_loss.csv, and validation_loss.csv are in the same directory.
- The training logs are static_retrain/train_pilot500.log and dynamic_retrain/baseline_d_train_gpu4_main.log. Raw direct/retrained reports, selected clip manifests, checkpoints, and valid-step reports remain next to them in outputs/pvb_origin_baselines/.
- Compatibility boundary: npz-v1 clips are not the original gzip-JSON MMAPDataset format, so the only adaptation is output-local streaming pair/clip reading. Original PVB dyVAE, DynamicTrainer, collator, and checkpoint serialization were not edited. The plotted values are PVB-style clip structure/motion metrics, not exact original eval_prot.py raw-trajectory TICA/MSM values.
- C and D retain the original checkpoint-specific fine-tuning settings from their respective configs: C lr=5e-5/warmup=1000, D lr=1e-4/warmup=100. The data, seed, split, batch bound, step budget, and evaluation policy are shared; this hyperparameter difference is explicitly retained rather than hidden.
- Validation: python -m py_compile scripts/aggregate_pvb_origin_baselines.py; python -m pytest -q -> 41 passed, 1 existing torch.load FutureWarning; aggregate finite-metric audit passed with 96 metric rows and 2,000 loss records.

### 2026-08-21 — Modified codec ATLAS-only retraining and evaluation

- Status: complete. This run uses the current modified multi-frame codec (`train_codec.py`/`eval_codec.py`), not the untouched `PVB_origin` baseline. PDB, MISATO, and all other datasets were excluded; only ATLAS `dt_100ps` train/valid stores were used.
- Fixed policy: seed `20260810`, `max_tokens=80000`, 256 train batches for normalization fitting, 1,000 optimizer steps per control, `log_every=1`, GPU 7, and 32 validation batches per control (64 ATLAS clips/control). The three controls were serialized to avoid GPU contention.
- Checkpoints/logs: `outputs/atlas_only_modified/ratio1_no_temporal/codec_step_00001000.pt` + `train_metrics.jsonl`; `ratio1_temporal/codec_step_00001000.pt` + log; `ratio4_temporal/codec_step_00001000.pt` + log. Each log contains exactly 1,000 records through step 1,000.
- Training totals decreased from step 1 to step 1,000 as follows: ratio1/no-temporal `1.1371 -> 0.8248` (last-100 mean `1.2241`), ratio1/temporal `1.1702 -> 0.5446` (last-100 mean `0.8728`), and ratio4/temporal `1.3259 -> 0.7067` (last-100 mean `1.0626`). The curves improve but remain minibatch-noisy rather than fully flat/converged.
- Validation reports: `outputs/atlas_only_modified/atlas_only_codec_eval.json` and `.md` include checkpoint-matched normalized weighted loss and all PVB-style metrics, stratified by native bucket. Validation total loss is ratio1/temporal `2.52509`, ratio4/temporal `2.83812`, and ratio1/no-temporal `3.16047`.
- ATLAS `dt_100ps` future metrics (ratio1/no-temporal; ratio1/temporal; ratio4/temporal): future RMSD `2.98163; 2.65981; 2.78176`, future dRMSD `2.01296; 1.78700; 1.87184`, velocity RMSE `0.0102119; 0.00880295; 0.0108576`, acceleration RMSE `0.000164202; 0.000140411; 0.000174638`, future bond RMSE `0.189274; 0.253299; 0.211263`, future contact error `0.000241982; 0.000209706; 0.000145346`, future clash rate `3.2407e-7; 7.0342e-7; 2.8288e-6`, and FFT frequency retention `0.016559; 0.250068; 0.067732`. Full frame-0/all-frame and loss components remain in the JSON.
- Visualization: `outputs/atlas_only_modified/plots/` contains `loss_curves`, `eval_loss`, `eval_reconstruction`, `eval_dynamics` in PNG/PDF plus `eval_metrics.csv`.
- Evaluation implementation now reports validation loss using each checkpoint's saved config, loss schedule, and normalization statistics; the plot/CSV path includes validation total and component losses.
- Validation: `python -m py_compile eval_codec.py evaluation/codec_evaluation.py scripts/plot_codec_results.py` passed; full `python -m pytest -q` passed (`41 passed`, one existing `torch.load(weights_only=False)` warning); output audit passed for checkpoints, 3,000 train records, 192 eval samples, loss fields, and plots.
- Boundary: the two mistakenly started `PVB_origin` ATLAS-only jobs were interrupted before this run and are not used as evidence; the reference repository was not modified.
