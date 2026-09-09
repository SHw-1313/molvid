# Session A compact review evidence

- Status: `A_DIAGNOSTICS_READY_FOR_REVIEW`
- Branch: `exp/dit-state-detail-t1-pilot-v1`
- Source/config/test commit: `a25e6bf66214a82adf9a56d492a85a28207ef248`
- Full report: `outputs/dit_state_detail_diagnostics_v1/session_a_260909/report.md`
- Device: GPU0, NVIDIA A100-SXM4-80GB; DiT inference BF16 autocast; metric accumulation FP32.

## Scope and protocol

This was inference-only evaluation of the existing R2/R4 best-validation checkpoints. It used
H=4/H=8, observed `[0,H)`, future `[H,16)`, boundary `[H-1,H)`, 72 validation clips from all
8 validation systems per candidate, and 8-clip deep diagnostics with four draws at 8/16 Euler
steps. There were zero optimizer steps and no test payload access.

## Main result

| candidate | H | codec oracle aligned RMSD | latent-state persistence aligned RMSD | DiT generated aligned RMSD | latent persistence contact F1 | DiT generated contact F1 |
|---|---:|---:|---:|---:|---:|---:|
| R2-SD | 4 | 0.0223 | 1.8391 | 3.1437 | 0.8835 | 0.4767 |
| R2-SD | 8 | — | 1.7131 | 2.9999 | — | 0.4888 |
| R4-SD | 4 | 0.0212 | 1.8067 | 3.1363 | 0.8878 | 0.4911 |
| R4-SD | 8 | — | 1.6858 | 2.9855 | — | 0.5011 |

The latent-state persistence control is retained as a conservative reference-only diagnostic, not
as a deployable prior or source center. Generated future geometry/contact metrics were worse for
both candidates, so no R2/R4 winner is declared.

The first R4 deep attempt was excluded after a vector-mask broadcasting defect in latent-summary
quantiles. The defect was fixed, covered by a regression test, and both deep runs were rerun.

No DiT training, backend change, frozen-codec change, push, merge, or conditional-prior run was
performed. Large JSONL outputs remain local and uncommitted.
