# Session B handoff

Worktree: `/data4/users/sihao/workspace/molvid-dit-capacity-data-v1`  
Branch: `exp/dit-capacity-data-v1`  
Base: `5c2754fcce44ed77dad77db09db709408fee7634`

The B-only implementation is in the worktree. The frozen data manifest is at [manifest.json](outputs/dit_capacity_data_v1/data_manifest_20260914/manifest.json); it contains 48 nested base systems, 144 added systems, 192 expanded systems, and the original 8-system validation view. The source manifest hash is `36c67ad7c1d0575b3f55805ad63be70c0449dfa440dc26b1e3bfd37fa0c3b0f4`. Test payload was not opened.

CPU checks passed:

- `python -m pytest -q tests/test_dit_capacity_data_v1.py -k 'not cuda'`
- `python -m pytest -q tests/test_dit_trainer.py tests/test_dit_pilot_runner.py tests/test_dit_source_ab.py -k 'not cuda'`

The required CUDA isolation test is `DIT_RUN_CAPACITY_ISOLATION=1 python -m pytest -q tests/test_dit_capacity_data_v1.py -k cuda`. The runner sequence is `preflight → source_check/verify → prepare → train → evaluate → summarize`; use `scripts/run_dit_capacity_data_v1.py` and keep output under `outputs/dit_capacity_data_v1/20260914_capacity_data_v1/`. Do not start training until `verify` and `source_check` are PASS and `budget.json` freezes the measured budget.

At the last pause all GPUs were occupied. The current container exposes `/data4` read-only, so the tmux environment must provide writable output/checkpoint storage while importing Python modules from this B worktree. No training or generation was started.
