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

Latest host CPU run for the directly related codec/capacity/source suite passed 17 tests with 5 CUDA-only tests deselected. The final neibu CUDA gate passed 4 tests, skipped 1 optional test, and deselected 17 tests.

The preflight was rerun from the correct container cwd `/workspace/molvid-dit-capacity-data-v1`
and passed; its host-visible artifact is
`outputs/dit_capacity_data_v1/20260914_capacity_data_v1/preflight.json`.

The targeted CUDA command is `DIT_RUN_CAPACITY_ISOLATION=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_dit_capacity_data_v1.py tests/test_dit_source_ab.py tests/test_codec_evaluation.py -k cuda`; it passed 4 tests, skipped 1 optional test, and deselected 17 tests. The runner sequence `preflight → source_check/verify → prepare → train → evaluate → summarize` is complete for G48/C48/C48D8; C192 is blocked by missing data.

neibu execution uses host `neibu:/data/users/yuansihao/workspace/molvid-dit-capacity-data-v1`,
mapped inside its container as `/workspace/molvid-dit-capacity-data-v1`. The matching frozen codec
and statistics are read from `/data`. Fixed evaluation sample IDs and full record tensor bytes match
the original host (`0a21407c...` and `4b6f63c2...`). neibu lacks 26,784 expanded train clips, so
C192 is explicitly `BLOCKED_DATA`; base48 and validation are complete and test remains unopened.

CUDA evidence on neibu GPU0 passed:

- 4 gated CUDA tests passed, 1 optional test skipped, and 17 unrelated CUDA tests were deselected, including direct legacy `both` trainer construction.
- Full-size same-shape init hash `973dd528cc11fdba339c1ec062932687dcfd7248d4e834d881ad877eef7c5f6f`.
- Depth-8 init hash `8ef5f46f8489aed211d297519a50adb2d6dfa080b0120a235b76381e9510f606`.
- No trainable storage overlap; bidirectional mutation isolation and exact deterministic next-step resume passed.
- Real 1,525-atom clip H4/H8 sampler/decode smoke passed for G48/C48/C48D8.
- Source geometry check passed for all 8 validation systems at H4/H8; template raw RMSD is about 0.021--0.024 A and template bond RMSE about 0.012--0.013 A.

The frozen prepare contract uses training/numerical code commit
`c070e3c2aeecb6dc33d338e2429bfb426876faac`, contract hash
`92752fb091ed8486f6ffba13fb6f31582986cba2526695aea174fd71a7814faf`, 4,500 successful
updates, and 267,988,032 effective atom-frame tokens. Measured p90 end-to-end update times are
1.574 s (G48), 2.998 s (C48), and 3.095 s (C48D8); C192 reserves a conservative 3.869 s.
Atomic checkpoints are written every 500 successful updates. Resume restores the checkpointed
cursor/generator/optimizer/scaler/token count and atomically truncates any trailing JSONL rows,
while a fresh launch refuses existing formal-run artifacts.

G48, C48, and C48D8 completed PASS at 4,500 successful updates and 267,988,032 effective
atom-frame tokens each. Each has 9 RF validation rows, complete 16-row real-generation monitors
at steps 2,000 and 4,000, and a complete final evaluation of 144 + 128 + 128 rows. Their final
checkpoints are `G48/checkpoints/checkpoint_final.pt`, `C48/checkpoints/checkpoint_final.pt`, and
`C48D8/checkpoints/checkpoint_final.pt` under the run root. Evaluation generated 384 rows and
reused 16 deterministic rows per arm, costing 0.675119, 0.690703, and 0.712130 GPU-hours.
G48's training summary covers only the resumed segment; including the estimated 0.864438-hour
initial formal attempt, all formal training attempts are about 13.16110 GPU-hours. Training plus real-generation evaluation is about 15.23906 GPU-hours; the precheckpoint step-89 smoke is excluded.

The first step-2000 monitor attempt failed after training updates had completed because the
training allocator retained about 81 GiB and cuSOLVER could not create the Kabsch SVD handle. This
was not an optimization or checkpoint failure. Commit `c070e3c2aeecb6dc33d338e2429bfb426876faac`
makes the checkpoint durable before monitoring and synchronizes/releases the allocator around
each sampler/decode metric pass. G48 was restored from checkpoint 1500; the existing JSONL tail was
atomically truncated, step 2000 was reproduced, all 16 fixed H4/H8 monitor clips and metric rows
completed, and training continued past step 2000.

Evaluation/reporting code is committed as `9960a479071ddd50df575790a2fb6388fa399e93` and was
synced to neibu by per-file SHA256. It adds resumable row shards, H4/H8 and L4/L8/future
draw→clip→system aggregates, per-system outputs, block-displacement decomposition, bounded-chunk
true future occupancy, and nullable constant RMSF/ACF/Pearson handling. Missing torsions remain
null; legacy occupancy and unimplemented diversity are excluded. Final artifacts are
`comparison.json`, `per_system_results.json`, `learning_curves.json`, `learning_curves.png`,
`report.md`, and `summary.json` under the run root. The summary status is
`DIT_CAPACITY_DATA_V1_PARTIAL_BLOCKED_DATA` solely because C192 lacks 26,784 selected clips.

The first formal checkpoint, `G48/checkpoints/checkpoint_step000500.pt`, was written successfully
(134,315,877 bytes). Inspection found step/successful_updates/capacity_successful_updates all 500,
cursor `{epoch: 0, batch_index: 500}`, 28,951,104 tokens, 166 optimizer state entries, all five AMP
scaler fields, and generator-state SHA256
`2813e1e20d7dba2bbe87db48d148130a06b3d06a86e5a380e061d375a674c940`.

G48 completed PASS at 4,500 successful updates and exactly 267,988,032 effective atom-frame tokens.
Its final checkpoint is durable; it recorded 9 RF validation rows, two 16-row real-generation
monitors at steps 2000/4000, and 2.357 estimated GPU-hours. C48 then started as a fresh process.
The first 16 C48 rows matched G48 exactly for epoch, batch index/tokens, H, and tau min/mean/max;
losses differ as expected because only the source center changes.

An initial G48 process was stopped at step 89 before it had a checkpoint, specifically to add the
periodic checkpoint and resume-history guarantees. That evidence is preserved under
`G48_interrupted_precheckpoint_step000089` and is not part of the formal learning curve.

## Final artifact record

- Run root: `/workspace/molvid-dit-capacity-data-v1/outputs/dit_capacity_data_v1/20260914_capacity_data_v1/`.
- Contract: `92752fb091ed8486f6ffba13fb6f31582986cba2526695aea174fd71a7814faf`; code: `9960a479071ddd50df575790a2fb6388fa399e93`.
- Source manifest: `36c67ad7c1d0575b3f55805ad63be70c0449dfa440dc26b1e3bfd37fa0c3b0f4`; materialized data: `a9bdf4d8cfd0d4ea612bb5bc9a5aad29fdba1aaec800ce272c5fc5401217e1f6`.
- Codec state: `9a30e3838403cbd9f2cfa7344276cbdc7a8176d75ce5d4da028a3ea39ec390b9`; statistics hash: `863c1894bac27c6fa121eaaa635961e44c37cc95c9f55b3fb371b4fb0bab778d`; statistics file SHA256: `4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`.
- Final checkpoint SHA256: G48 `8062a49124a64ec872a125d7f883eacbe545343dda4c0aec68d7a73b625acf71`; C48 `d17285697a09c6f26bf82f0627fa88911c8490ff535b6ee77a8164a2f12bfd94`; C48D8 `fbbca747f18674cf844930a4df34b9a7ed5e3ed4d3006ee3837bb18c3dd8fc81`.
- C192 is `BLOCKED_DATA` because 26,784 selected expanded clips are absent on neibu; no test coordinates were opened.
