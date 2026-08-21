# Multi-frame codec v1 decision log

Decisions here are binding until superseded by a new numbered decision with evidence. Do not silently edit an old decision after implementation begins.

## D001 — Additive codec path, not a `dyVAE` rewrite

**Decision:** Keep current PVB training/inference intact and introduce an isolated codec entrypoint and modules.

**Why:** Existing checkpoints and one-step experiments must remain reproducible. The new clip contract is semantically incompatible with silently reusing `x0/x1`.

## D002 — Packed time-major coordinate contract

**Decision:** Use `x: [T, N_total, 3]` with packed atoms, `abid`, and `atom_ptr`. Batches are homogeneous in task and temporal length.

**Why:** Padding proteins to a global `N_max` is wasteful; PVB already packs atoms. Time-major layout makes temporal shape explicit and lets the spatial adapter flatten frames into graph batches.

## D003 — True clips replace frame pairs

**Decision:** Add ordered clip preprocessing/loaders. Do not reconstruct clips by chaining independently sampled `(x0, x1)` records.

**Why:** Pair records lose temporal ordering, timestamps, and higher-order dynamics—the exact information the codec is intended to learn.

## D004 — Static and dynamics share weights, not fake time

**Decision:** Static data uses `T=1` corrupted-coordinate denoising; trajectory data uses ordered `T=16` clips. A task-aware loader alternates homogeneous batches.

**Why:** Repeating a static structure across time teaches zero motion as if it were a real trajectory and biases the temporal module.

## D005 — PVB scalar/vector features are the spatial latent basis

**Decision:** Reuse `TorchMD_VQ_ET` outputs `h` and `v`; do not replace the encoder with ViSNet or introduce CG-heavy irreps in v1.

**Why:** This preserves PVB's atom/residue type and local geometry strengths while providing an invariant/equivariant interface suitable for temporal mixing.

## D006 — SE(3)-safe temporal mixing without CG coefficients

**Decision:** Temporal attention/gates are computed from invariant scalar features. The resulting scalar weights mix scalar channels and vector channels identically across xyz; learned maps act on channels only.

**Why:** This retains equivariance while avoiding arbitrary coordinate-axis mixing and the implementation burden of CG tensor products.

## D007 — Causal encoder and decoder

**Decision:** All temporal attention/convolution/downsampling uses only the current and earlier frames. Upsampling uses repeat/interpolation plus causal refinement.

**Why:** The codec should later support streaming/chunk continuation. A bidirectional v1 would inflate reconstruction quality but create an incompatible latent interface for the planned world-model trunk.

## D008 — Deterministic codec before VAE/VQ

**Decision:** v1 has no KL term, posterior sampling, or vector quantization. Add such a bottleneck only after deterministic ratio-4 reconstruction and causality/equivariance gates pass.

**Why:** Otherwise posterior collapse or quantization error would be confounded with temporal architecture failure.

## D009 — Dual scalar/vector temporal latent

**Decision:** Preserve both invariant `z_h` and equivariant `z_v` through temporal compression.

**Why:** Scalar-only latents are easy to condition but force the decoder to recover directional motion from scratch. Vector latents provide a direct equivariant basis for displacements.

## D010 — First frame is an explicit anchor

**Decision:** Decode displacements relative to `x_anchor=x[0]` (or its corrupted form for static denoising). Report future-frame metrics separately and do not count the anchor as a learned temporal token.

**Why:** Molecular dynamics is naturally conditioned on an initial structure, and an anchor fixes the translation gauge. Static corruption prevents a trivial identity solution.

## D011 — Protein-based clip alignment

**Decision:** Kabsch-align every frame to frame 0 using a protein backbone/CA mask and apply the same transform to the whole system.

**Why:** It removes irrelevant global motion while preserving ligand motion relative to the protein. Independent ligand alignment is forbidden.

## D012 — Atom tokens only in v1

**Decision:** Do not add residue pooling or spatial patch tokens yet. Use `T*N` dynamic batching and measure memory first.

**Why:** The current data has per-atom `btype` and CA/block positions but no robust unique residue id in every dataset. Adding pooling now would couple data migration to the codec proof.

## D013 — Factorized video-style codec, no generative trunk

**Decision:** Spatial graph encoding/refinement and causal temporal blocks are factorized. The v1 output is a reconstructed observed clip, not a predicted future clip.

**Why:** This imports the useful video-codec principle—joint multi-frame tokens and temporal compression—without conflating it with diffusion/flow generation or rollout quality.

## D014 — Ratio 1 is a first-class control

**Decision:** Temporal ratios 1, 2, and 4 share the same API; ratio 1 supports both temporal-on and temporal-off controls.

**Why:** The pilot needs to separate gains from temporal mixing from losses caused by compression.

## D015 — No unapproved dependency changes

**Decision:** Implement with the current repository environment. Do not add FlashAttention, xFormers, PyTorch Lightning, Hydra, or another storage framework in v1.

**Why:** The immediate goal is a runnable patch in the user's existing PVB environment, not ecosystem migration.

## D016 — Physical time is a first-class model condition

**Decision:** Every trajectory clip carries `time_ps` and `delta_time_ps`. Temporal attention uses a continuous invariant relative-time bias derived from physical `|t_i-t_j|`; compressed latents carry `latent_time_ps`; the decoder receives `target_time_ps`.

**Why:** Frame index is not a physical clock. A 16-frame clip sampled every 100 ps spans 1.5 ns, while one sampled every 1 ns spans 15 ns. Treating them identically would make the shared codec learn contradictory motion semantics.

## D017 — Homogeneous sampling-interval buckets with shared weights

**Decision:** A minibatch is homogeneous in task, temporal length, and sampling-interval bucket. Static, 100 ps, 1 ns, and other configured buckets alternate under explicit sampling weights while sharing codec parameters. Continuous timestamps, not bucket id or dataset id, are the authoritative model input.

**Why:** Homogeneous buckets simplify masking, normalization, and profiling without splitting the model by dataset. Continuous conditioning preserves interpolation to unseen intervals better than a categorical dataset token alone.

## D018 — No coarse-to-fine label fabrication

**Decision:** Never interpolate a coarse trajectory into fine-timescale training targets. Fine trajectories may be decimated into explicitly labeled coarse clips; the reverse is forbidden.

**Why:** Dividing displacement by `dt` changes units but cannot reconstruct fast events already removed by 1 ns sampling. Interpolated 100 ps labels would teach visually smooth but physically unsupported motion.

## D019 — Temporal losses and metrics are time-bucket aware

**Decision:** Compute velocity and irregular-grid acceleration from physical `delta_time_ps`. Optimize with train-split normalization statistics per time bucket, checkpoint those statistics, log raw physical units, and report metrics separately by bucket. Do not use an unqualified cross-bucket mean for model selection.

**Why:** Fine and coarse sampling observe different frequency bands and produce different derivative distributions. A single aggregate can be dominated by dataset size or hide failure at one physical scale.

## D020 — Versioned binary clip storage after JSON measurement

**Decision:** New clip stores use `npz-v1`: numeric arrays are compressed NumPy members and scalar/list metadata is one JSON member. The reader accepts only this versioned payload; obsolete gzip+JSON compatibility is removed.

**Why:** A representative ATLAS record serialized as JSON was 1,726,089 bytes and gzip level 6 reduced it to 403,058 bytes while spending about 0.04 s on a single record in the local environment. `np.savez_compressed` produced 220,131 bytes in about 0.04 s and avoids Python float serialization; uncompressed NumPy storage was 844,984 bytes in under 1 ms. The versioned binary format is therefore both smaller and operationally safer for the half-data materialization. This is a storage implementation choice, not a change to the ClipBatch contract.

## Open questions requiring operator evidence

These are parameters, not permission to redesign the architecture:

1. Exact writable PVB repository path and branch convention.
2. Exact local `enter-container` command syntax and whether `torch-ito` already contains the PVB dependencies.
3. Local ATLAS/mdCATH raw paths, native frame timestamps/spacing, and desired 100 ps/1 ns (or other) bucket definitions.
4. PVB checkpoint path for encoder initialization.
5. Whether the first pilot is protein-only or includes a protein-ligand trajectory dataset.
6. Available A100 memory (40 GB or 80 GB), which sets the initial `T*N` bound.

The worker should implement synthetic/CPU-safe portions without these values, use explicit config placeholders, and stop before a data/GPU gate that genuinely needs them.
