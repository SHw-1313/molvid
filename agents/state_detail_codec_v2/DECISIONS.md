# DECISIONS — zero-preserving state/detail temporal codec v2

## D1 — TorchMD is fixed

Use `torchmd_et` for every new codec control. ViSNet v1/v2 code and artifacts remain compatible
history but are not retrained or compared in this phase.

## D2 — codec and generation task remain separate

The codec encodes and reconstructs the complete 16-frame clip. Observation masks, two-frame
conditioning, history/future splitting, DiT denoising, and rollout belong to later phases.

## D3 — deterministic AE

The codec has no KL, stochastic posterior, VQ, MMD, adversarial prior, or latent-distribution
loss. Latent statistics are logged for diagnosis only.

## D4 — dual-bank capacity is intentional

R1 uses 16 C-wide state tokens. R2 uses eight tokens with C-wide state and C-wide detail banks,
preserving 16C active feature volume while halving the future DiT sequence length. R4 uses four
tokens with the same two banks, yielding 8C active feature volume and four temporal tokens.

Ratio denotes temporal-token reduction. Reports must not misstate R2 as a twofold reduction in
all latent elements.

## D5 — hierarchical Haar, not constant-velocity residual

Use fixed orthonormal Haar lifting for R2 and hierarchical R4 state/detail coefficients. The
three R4 detail coefficients are `Dmid`, `D01`, and `D23` in that fixed order. They are temporal
resolutions inside one four-frame block, not molecular slow/fast modes.

## D6 — block-local tokenizer

No cross-block temporal attention is added to the codec. The future DiT is responsible for
long-range temporal interaction. R4's first token may encode frames 0–3 because it is a clean
target token, not an observed forecasting condition.

## D7 — zero motion is enforced by construction

The state/detail detail encoder and decoder satisfy `f(0)=0`; state/time cannot add motion.
Repeated-static zero detail and zero decoded motion are correctness gates, not loss terms or
optional ablations.

## D8 — static and repeated-static have different semantics

Static `T=1` has no valid detail and uses a full C-wide state token. Synthetic repeated-static
`T=16` has valid detail equal to zero. Static structures are never repeated to create training
trajectories.

## D9 — remove the complete x0 bypass in the new path

No per-atom first-frame coordinates are stored in a new latent or given to its decoder. Retain
only one masked centroid/origin vector per sample to restore translation. Geometry must be
reconstructed from latent h/v. Old checkpoint schemas retain their original `x_anchor` behavior.

## D10 — matched pooling is new-framework and capacity matched

The only extra T0 control is an R4 two-head unstructured pooling codec with four 2C-wide tokens
and no per-atom x0 bypass. The old single-bank `4 x C` R4 results are historical and are not
retrained or included in the locked four-way ranking.

## D11 — fixed losses, no semantic regularizer

Use coordinate/local/bond/velocity/acceleration losses and the frozen staged schedule. Do not add
detail shrinkage, orthogonality, feature reconstruction, or zero-motion consistency loss.

## D12 — no spatial refiner

The optional spatial refiner is disabled for all controls so temporal-codec and basic decoder
behavior remain visible.

## D13 — T0 is not architecture selection

The prior three-system/nine-trajectory dataset is an implementation and training-sanity test. It
cannot decide the production ratio or support a scaling claim.

## D14 — frozen TorchMD during the locked T0 comparison

All four controls load identical recorded TorchMD frame-encoder weights and freeze them during
the 30-epoch comparison. Separate smoke evidence must show that gradients can flow when the
encoder is unfrozen. This isolates the temporal codec without forbidding later co-adaptation.

## D15 — hard operator gate before T1

After code gates and T0 reports, the phase stops in `WAITING_FOR_OPERATOR_REVIEW`. T1 data
materialization and execution require a later explicit user approval; passing automated tests
does not authorize continuation.

## D16 — planned T1 scale

After approval, T1 uses 64 independent systems split 48/8/8 by system, up to three trajectories
per system, capped resampled clips, one native time bucket, and a per-control target of 18–20 GPU
hours. This decision records the intended design but grants no current execution authority.
