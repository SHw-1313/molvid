# Pilot tasks

- [x] P001 verify the pushed repair commit, isolated branch/worktree, frozen manifest counts,
  system disjointness, unopened test sampling, and approved T1 codec checkpoints
- [x] P002 create the pilot protocol and preserve the branch-level safety transition in `AGENTS.md`
- [x] P003 implement the focused resumable real-T1 pilot runner with train/validation-only data
  access, frozen codec/frame encoder, train-only statistics, deterministic H=4/H=8 schedule,
  atomic checkpoints, and hash/contract refusal; verified by runner tests and T1 preflight
- [x] P004 run one 200-step profile for each candidate with validation RF loss and checkpoint resume;
  repaired schedule-hash contract and completed both candidates under `profiles_20260908_fix2`
- [x] P005 freeze and record one common `pilot_budget.json`; two idle GPUs were available and the
  common budget is `S=4500` (>=1000)
- [ ] P006 run the matched R2/R4 pilot with the common seed, schedule, and step budget without
  opening test
- [ ] P007 select checkpoints by validation RF loss and evaluate the fixed all-eight-system
  validation subset for H=4/H=8 future metrics and codec-oracle generation gaps
- [ ] P008 append exact commands, hashes, profiles, losses, checkpoints, warnings, and final
  operator-review status; do not declare a ratio winner

## 2026-09-08 — P003/P004 execution evidence and repair

The first two profile attempts exposed a real runner defect before any valid training result:
`_batch_schedule_hash()` hashed the full sampler contract, while `_next_train_batch()` returned a
hash of only the batch lists. Both candidates therefore stopped at the first training batch with
`RuntimeError: training schedule hash changed`. The runner now hashes the same materialized
epoch-specific sampler contract consumed by `_next_train_batch()`, and
`tests/test_dit_pilot_runner.py::test_consumed_training_schedule_hash_matches_contract_hash`
guards the behavior. The failed outputs are retained under `profiles_20260908` and
`profiles_20260908_fix`; they are not valid pilot evidence.

The repaired run completed both candidates in `outputs/dit_state_detail_pilot_v1/profiles_20260908_fix2`:

| candidate | status | steps | wall seconds | train seconds/step | optimizer seconds/step | peak allocated/reserved bytes | resume |
|---|---:|---:|---:|---:|---:|---:|---|
| `ratio2_state_detail` | PASS | 200 | 2328.952 | 9.969244 | 9.651530 | 26601327104 / 84477476864 | 200 → 201 |
| `ratio4_state_detail` | PASS | 200 | 1198.269 | 5.008414 | 4.691752 | 26356072960 / 84477476864 | 200 → 201 |

Both used NVIDIA A100-SXM4-80GB, BF16 autocast, seed `20260907`, GPU 5/7, train-only statistics,
the same H=4/H=8 schedule, and the same validation plan (all 8 validation systems, 72 clips,
windows 0/30/61). Data hash is
`9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304ebfe184`; both report
`test_opened=false`. Statistics hashes are R2
`2dc3541483afb823fd264f48276b7a7e39cba253d7db36cac566cd38729c0e1c` and R4
`34733304618c1ffdd009bc6d78d3a8d83ea11b4d0625919d79dbfee8a1aee993`.

Step-200 validation totals were R2 `1.2945222069915887` and R4 `1.3085407876826318`.
These are execution/profile observations only and are not a scientific ranking.

## 2026-09-08 — P005 frozen budget

At the post-profile audit, GPU 5 and GPU 7 were idle; no other GPU process was touched. The common
budget is frozen in `outputs/dit_state_detail_pilot_v1/pilot_budget.json` before matched training:
`S=4500`, seed `20260907`, validation interval 100, identical H=4/H=8 schedule, and no test
access. The declared local overnight cutoff is 2026-09-09 08:00, with 57,434.4 seconds available
at the required 80% fraction. The conservative R2 estimate including 30 minutes train-statistics
reserve and 90 minutes generated-validation reserve is 57,199.85 seconds; the next 100-step
budget estimates 58,308.32 seconds and therefore exceeds the usable window. This makes 4,500 the
largest checked 100-step common budget under the recorded assumptions. The R4 estimate is
32,752.18 seconds. The budget is frozen before any matched pilot step.
