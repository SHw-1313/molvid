# molvid project instructions

## Active implementation phase

The active phase is **ViSNet spatial backbone v1**.

Before editing code, read these files in order:

1. `agents/visnet_spatial_v1/PLAN.md`
2. `agents/visnet_spatial_v1/DECISIONS.md`
3. `agents/visnet_spatial_v1/ACCEPTANCE.md`
4. `agents/visnet_spatial_v1/TASKS.md`
5. `agents/visnet_spatial_v1/HANDOFF.md`
6. `agents/visnet_spatial_v1/LUNA_CODEX_PROMPT.md`

The phase files above are authoritative for this branch. The older cumulative files under
`agents/PLAN.md`, `agents/TASKS.md`, `agents/DECISIONS.md`, and `agents/HANDOFF.md`
are historical context and must not be overwritten or repurposed.

## Repository and environment rules

- Work on `feat/visnet-spatial-v1`, based on `fix/graph-runtime-v1`.
- Run Python, tests, and training only through `enter-container` with the `torch-ito` conda environment.
- Do not install or upgrade dependencies and do not use network access.
- Preserve unrelated user changes.
- Serialize GPU-heavy commands.
- Do not commit generated checkpoints, large datasets, or binary plots unless explicitly requested.

## Scope discipline

This phase implements selectable spatial backbones only:

- `torchmd_et`
- `visnet_radius`
- `visnet_bonded`

Keep the existing ratio-1 temporal codec, decoder, data schema, optimizer, and losses unchanged.
Do not implement state-detail ratio-4 packing, decoder redesign, residue-token temporal modeling,
AF3 conditioning, diffusion, flow matching, or VideoDiT.

## Documentation ownership

- `AGENTS.md`: read-only for the worker.
- `PLAN.md`: read-only for the worker.
- `DECISIONS.md`: append only when an implementation fact forces a new decision; never rewrite prior decisions.
- `TASKS.md`: update task status and attach concise evidence.
- `HANDOFF.md`: append dated commands, tests, outputs, blockers, and next task.
- `ACCEPTANCE.md`: read-only unless the operator explicitly approves a changed gate.

Do not begin coding by rewriting the plan. Execute the prepared plan.
