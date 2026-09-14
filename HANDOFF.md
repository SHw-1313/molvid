# Session B handoff

Worktree: `/data4/users/sihao/workspace/molvid-dit-capacity-data-v1`  
Branch: `exp/dit-capacity-data-v1`  
Base: `5c2754fcce44ed77dad77db09db709408fee7634`

Inside `enter-container`, use `/workspace/molvid-dit-capacity-data-v1`; it is the writable
mapping of the host worktree. Do not use the host absolute workspace path inside the container:
`/data4` is a read-only container mount. The approved dynamic source remains read-only at
`/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps`.

The B-only implementation is in the worktree. The frozen data manifest is at [manifest.json](outputs/dit_capacity_data_v1/data_manifest_20260914/manifest.json); it contains 48 nested base systems, 144 added systems, 192 expanded systems, and the original 8-system validation view. The source manifest hash is `36c67ad7c1d0575b3f55805ad63be70c0449dfa440dc26b1e3bfd37fa0c3b0f4`. Test payload was not opened.

The source manifest has native dt=10 ps, source stride=10 (effective dt=100 ps), window stride=16,
and 16-frame clips. The materializer reads only approved train/valid indexes, requires R1/R2/R3
and windows 0..61, preserves the frozen 48/8/8 system selections, excludes frozen valid/test
systems, and chooses 144 additional eligible train systems by deterministic SHA256 ranking under
the 80,000 atom-frame-token cap. It records `system_disjoint_only`: no sequence/chain homology
audit was available. Final validation uses the fixed 8 systems × 3 replicas × windows 0/30/61.

CPU checks passed:

- `python -m pytest -q tests/test_dit_capacity_data_v1.py -k 'not cuda'`
- `python -m pytest -q tests/test_dit_trainer.py tests/test_dit_pilot_runner.py tests/test_dit_source_ab.py -k 'not cuda'`

The preflight was rerun from the correct container cwd `/workspace/molvid-dit-capacity-data-v1`
and passed; its host-visible artifact is
`outputs/dit_capacity_data_v1/20260914_capacity_data_v1/preflight.json`.

The required CUDA isolation test is `DIT_RUN_CAPACITY_ISOLATION=1 python -m pytest -q tests/test_dit_capacity_data_v1.py -k cuda`. The runner sequence is `preflight → source_check/verify → prepare → train → evaluate → summarize`; use `scripts/run_dit_capacity_data_v1.py` and keep output under `outputs/dit_capacity_data_v1/20260914_capacity_data_v1/`. Do not start training until `verify` and `source_check` are PASS and `budget.json` freezes the measured budget.

At the last pause all GPUs were occupied. No training or generation was started. With the correct
container path, output/checkpoint writes should go under the relative B output root, which resolves
under `/workspace/molvid-dit-capacity-data-v1/outputs/`.
