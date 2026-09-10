# Session A DiT pilot diagnostics

Status: `A_DIAGNOSTICS_READY_FOR_REVIEW`

This report is inference-only evidence from the existing best-validation checkpoints.
No optimizer step and no test clip payload access were permitted.

## Candidate outputs

| Candidate | Main | Deep | Smoke |
|---|---|---|---|
| ratio2_state_detail | present | present | present |
| ratio4_state_detail | present | present | present |

## Protocol

Observed frames are `[0,H)`, future frames `[H,16)`, boundary is `[H-1,H)`, and the full clip is diagnostic-only.
L4/L8 rows are separate cropped horizons with physical timestamps and masks rebuilt.
RMSD diversity uses `sqrt(sum_xyz_squared / valid_atom_count)`; no best-of-N metric is used.

## Prior decision

`latent_block_state_persistence` is retained as a reference-only control, not a deployable prior/source center: for both candidates, main `dit_generated` future geometry/contact results are worse than this observed-only control. Deep field, boundary, and dynamic outputs are attribution evidence only; no R2/R4 winner is declared.

## Limitations

- no optimizer steps or test payload access
- stochastic trajectories are not expected to reproduce reference MD frame by frame
- R2/R4 diagnostics use separate latent dimensions and identical seed protocol, not identical elementwise noise
- one validation run and four deep draws do not establish long-time kinetics or a scientific winner
