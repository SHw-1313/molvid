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

Latest counts are 5/5 capacity CPU tests and 10/10 directly related trainer/pilot/source tests.

The preflight was rerun from the correct container cwd `/workspace/molvid-dit-capacity-data-v1`
and passed; its host-visible artifact is
`outputs/dit_capacity_data_v1/20260914_capacity_data_v1/preflight.json`.

The required CUDA isolation test is `DIT_RUN_CAPACITY_ISOLATION=1 python -m pytest -q tests/test_dit_capacity_data_v1.py -k cuda`. The runner sequence is `preflight → source_check/verify → prepare → train → evaluate → summarize`; use `scripts/run_dit_capacity_data_v1.py` and keep output under `outputs/dit_capacity_data_v1/20260914_capacity_data_v1/`. Do not start training until `verify` and `source_check` are PASS and `budget.json` freezes the measured budget.

neibu execution uses host `neibu:/data/users/yuansihao/workspace/molvid-dit-capacity-data-v1`,
mapped inside its container as `/workspace/molvid-dit-capacity-data-v1`. The matching frozen codec
and statistics are read from `/data`. Fixed evaluation sample IDs and full record tensor bytes match
the original host (`0a21407c...` and `4b6f63c2...`). neibu lacks 26,784 expanded train clips, so
C192 is explicitly `BLOCKED_DATA`; base48 and validation are complete and test remains unopened.

CUDA evidence on neibu GPU0 passed:

- 2/2 gated CUDA tests, including direct legacy `both` trainer construction.
- Full-size same-shape init hash `973dd528cc11fdba339c1ec062932687dcfd7248d4e834d881ad877eef7c5f6f`.
- Depth-8 init hash `8ef5f46f8489aed211d297519a50adb2d6dfa080b0120a235b76381e9510f606`.
- No trainable storage overlap; bidirectional mutation isolation and exact deterministic next-step resume passed.
- Real 1,525-atom clip H4/H8 sampler/decode smoke passed for G48/C48/C48D8.
- Source geometry check passed for all 8 validation systems at H4/H8; template raw RMSD is about 0.021--0.024 A and template bond RMSE about 0.012--0.013 A.

The frozen prepare contract uses numerical code commit
`cbe9394a6c46e6fc35b02addf7e5b1061662abf8`, contract hash
`99393586bef7c6a851d3b11945be6b0c90255c196d3a923fbbb6732788566375`, 4,500 successful
updates, and 267,988,032 effective atom-frame tokens. Measured p90 end-to-end update times are
1.574 s (G48), 2.997 s (C48), and 3.097 s (C48D8); C192 reserves a conservative 3.871 s.

Current asynchronous work is the neibu tmux session `dit-capacity-train` on GPU0. It runs separate
Python processes in order G48, C48, C48D8, evaluate, summarize. Logs and progress are under
`outputs/dit_capacity_data_v1/20260914_capacity_data_v1/`; inspect the active arm's
`train_history.jsonl` and do not launch another GPU0 job concurrently.
