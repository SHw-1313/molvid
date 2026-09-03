# molvid project instructions

## Active implementation phase

The active phase is **ViSNet spatial backbone v2 reference-faithful repair**.

Before editing code, read the v1 phase files below, in order, as immutable historical context,
then read the v2 phase files, in order:

1. `agents/visnet_spatial_v1/PLAN.md`
2. `agents/visnet_spatial_v1/DECISIONS.md`
3. `agents/visnet_spatial_v1/ACCEPTANCE.md`
4. `agents/visnet_spatial_v1/TASKS.md`
5. `agents/visnet_spatial_v1/HANDOFF.md`
6. `agents/visnet_spatial_v1/LUNA_CODEX_PROMPT.md`
7. `LUNA_REVIEW_FIX_PROMPT_260902.md`
8. `agents/visnet_spatial_v2/PLAN.md`
9. `agents/visnet_spatial_v2/DECISIONS.md`
10. `agents/visnet_spatial_v2/ACCEPTANCE.md`
11. `agents/visnet_spatial_v2/REFERENCE_PARITY.md`
12. `agents/visnet_spatial_v2/TASKS.md`
13. `agents/visnet_spatial_v2/HANDOFF.md`

The v2 phase files are authoritative for this branch. The v1 phase files and artifacts are
read-only historical evidence and must remain reproducible. The older cumulative files under
`agents/PLAN.md`, `agents/TASKS.md`, `agents/DECISIONS.md`, and `agents/HANDOFF.md` are historical
context and must not be overwritten or repurposed.

## Repository and environment rules

- Work on `feat/visnet-spatial-v2`, based on `feat/visnet-spatial-v1` commit
  `bca9ea98df643f4965b009688f2967bd669e7fe2`.
- Run Python, tests, and training only through `enter-container` with the `torch-ito` conda environment.
- Run fixture generation through `enter-container` with `torch-ito` as well.
- Do not install or upgrade dependencies.
- The only network exception is Git clone/fetch of the two pinned reference repositories below,
  into `/data4/users/sihao/workspace`:
  - `https://github.com/microsoft/AI2BMD.git` at commit
    `497efaa190ee6f6cbc6030710c44208a01ece52d`
  - `https://github.com/pyg-team/pytorch_geometric.git` at commit
    `79d33965a40b7fa83616a9f598a0f8619f25d939`
- Those reference checkouts are read-only scientific specifications and are not production
  runtime dependencies.
- Preserve unrelated user changes.
- Serialize GPU-heavy commands.
- Do not commit generated checkpoints, large datasets, or binary plots unless explicitly requested.

## Scope discipline

This phase implements and audits reference-faithful selectable spatial backbones only:

- `torchmd_et`
- `visnet_radius`
- `visnet_bonded`
- `visnet_v2_radius`
- `visnet_v2_bonded`

Keep the existing ratio-1 temporal codec, decoder, data schema, optimizer, losses, evaluator, and
graph-runtime repairs unchanged. Preserve v1 backends, contracts, checkpoints, and result files.
Do not implement EPT hierarchy, state-detail ratio-4 packing, decoder redesign, residue-token
temporal modeling, AF3/MSA conditioning, diffusion, flow matching, rollout, scaling-law work,
VideoDiT, or full-data training.

## Documentation ownership

- `AGENTS.md`: the required v2 phase-transition update is authorized; otherwise preserve it.
- v1 phase files: read-only historical evidence; never rewrite them.
- `agents/visnet_spatial_v2/PLAN.md`: read-only.
- `agents/visnet_spatial_v2/DECISIONS.md`: append only when an implementation fact forces a real
  new decision; never rewrite prior decisions.
- `agents/visnet_spatial_v2/TASKS.md`: update task status and attach concise evidence.
- `agents/visnet_spatial_v2/HANDOFF.md`: append dated commands, tests, outputs, blockers, and
  next task.
- `agents/visnet_spatial_v2/REFERENCE_PARITY.md`: update the required audit table and fixture
  evidence as parity is proven.
- `agents/visnet_spatial_v2/ACCEPTANCE.md`: read-only unless the operator explicitly approves a
  changed gate.

Do not begin coding by rewriting the plan. Execute the prepared v2 plan, and do not change
acceptance thresholds to make a failing result pass.
