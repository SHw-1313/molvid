# A handoff

Status: A_DIAGNOSTICS_READY_FOR_REVIEW

此文件当前是模板，未运行任何测试或评测。

## 实际工作信息

- branch / start HEAD / final HEAD:
- preserved user changes:
- container entry and mapped worktree:
- python / torch / CUDA / GPU UUID / dtype:
- checkpoint / codec / statistics / manifest hashes:
- source-output ownership check:

## 实际命令与测试

按执行顺序追加命令、退出码、耗时、CUDA数值测试数量与skip原因。不要只写pytest通过而不写设备。

## 主结果

填写sample/system counts、H/L、aggregation、基线与生成结果、checkpoint和sampling seed。区分pathwise/分布/物理合法性。

## 深诊断与下一步

- last-block-state/zero-detail的几何与边界是否可用:
- raw-zero/normalized-zero语义测试:
- 哪个field/tau区间主要出问题:
- oracle扰动/field-swap能否定位decoder敏感性:
- 8/16steps、多draw与diversity:
- 最支持的下一轮假设/仍未知:

## 交付

- code commit:
- small evidence files:
- full local output path:
- exact reproduction command:
- missing files / blockers:
- final status:

停在operator review，不启动两臂训练。

## Session A execution record — 2026-09-09

Status: `A_DIAGNOSTICS_READY_FOR_REVIEW`

### Repository and ownership

- Worktree: `/data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1`
- Branch: `exp/dit-state-detail-t1-pilot-v1`
- Starting HEAD: `a22f60c4ffd1f502ab300352a00eafe841b1f8d2`
- B worktree/files were not edited. No push, merge, conditional-prior training, DiT training, backend change, codec change, or test-split access occurred.
- Root `AGENTS.md` contains the Session A transition and `Local session: A`.

### Environment and frozen inputs

- All Python/test/evaluation commands ran through `enter-container` with `torch-ito`.
- CUDA device: logical GPU0, `NVIDIA A100-SXM4-80GB`, UUID `b6b9107a-2a73-2d74-1dc0-1cbe16ae86d0`, CUDA runtime 12.1, PyTorch `2.5.1+cu121`.
- Inference policy recorded by reports: FP32 science metrics and BF16 DiT autocast.
- Manifest: `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`; preflight passed 48/8/8 system counts, 8928/1488/1488 clip counts, system disjointness, and `test_sampling.opened=false`.
- Preflight input hash: `b478d9fbe6721847f62767940dc0481753ea3c4f1e26573e49a78a76379e3948`.
- Approved codec checkpoint hashes: R2 `b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9`; R4 `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.
- Best-validation DiT checkpoint hashes captured by preflight: R2 `2d48363d272d0768281948df13e20d068b9da3a86fd3ece1540cdd8e6d173771`; R4 `bbd9c3710addecc05c0ecf41cd779b0ceaebcd4ed5f2e1bbd80e482e0b523992`.
- Statistics artifact hashes: R2 `5c167678596cd2169b926604b71db8481deea120f185cb1ed8749ab0217d256c`; R4 `4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`.

### Commands and results

- Focused diagnostics CUDA test: `CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_dit_diagnostics.py` — 10 passed in 7.92s; the CUDA test was executed, not skipped.
- Final affected regression subset: `... tests/test_dit_diagnostics.py tests/test_dit_evaluation.py tests/test_state_detail_latent_adapter.py tests/test_latent_rectified_flow.py tests/test_molecular_dit.py tests/test_dit_trainer.py` — 49 passed in 10.56s.
- `python -m py_compile evaluation/dit_diagnostics.py scripts/evaluate_dit_pilot_diagnostics.py tests/test_dit_diagnostics.py` — passed inside `torch-ito`; `git diff --check` — zero output.
- Smoke stages: R2 and R4 real validation clips completed with finite outputs, exact normalized observed clamping, raw inverse-normalization tolerance checks, no optimizer steps, and no test access.
- Main commands used the repaired runner with `--stage main --candidate ratio2_state_detail` and `--candidate ratio4_state_detail`, `--device cuda:0`, `--run-id session_a_260909`; each produced 720 JSONL rows, `sample_count=72`, `system_count=8`, windows 0/30/61, H4/H8, and `test_opened=false`. Elapsed seconds: R2 6229.1, R4 5477.6.
- Deep commands used `--stage deep` for both candidates with the same run id. Each produced `sample_count=8`, H4/H8, 8/16 steps, four draws, fixed tau, perturbation and field-swap outputs; `optimizer_steps=0`, `test_opened=false`. Elapsed seconds: R2 3952.8, R4 3404.7.
- Summary command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/evaluate_dit_pilot_diagnostics.py --config config/dit_pilot_diagnostics_v1.yaml --stage summarize --run-id session_a_260909`.

### Defect and conclusion

- The first R4 deep attempt failed at `latent_summary`: a `[K,N]` field mask was applied directly to `[K,N,C]` vector norms. The fix expands the token mask before quantiles and adds a regression test. The failed attempt is explicitly excluded; R4 and R2 deep were rerun successfully after the fix.
- Main diagnostic pattern: codec oracle remains near 0.02 Å, while generated future aligned RMSD is about 3.14 Å (R2 H4) / 3.14 Å (R4 H4); observed-only latent state persistence is about 1.84 Å (R2 H4) / 1.81 Å (R4 H4). Generated contact F1 is about 0.48–0.50 versus about 0.88–0.89 for latent persistence. These are execution diagnostics, not a winner claim.
- `latent_block_state_persistence` is therefore useful as a conservative reference-only control, not a deployable generative prior/source center. The report does not select R2 or R4.

### Files and outputs

- A-owned source/config/test: `evaluation/dit_diagnostics.py`, `scripts/evaluate_dit_pilot_diagnostics.py`, `config/dit_pilot_diagnostics_v1.yaml`, `tests/test_dit_diagnostics.py`.
- A-owned records: this file, `DECISIONS.md`, `TASKS.md`, and the root Session A transition in `AGENTS.md`.
- Full local outputs: `outputs/dit_state_detail_diagnostics_v1/session_a_260909/` (large JSONL/deep summaries remain untracked/local).
- Small report/evidence files are staged separately under `agents/dit_pilot_diagnostics_v1/evidence/`.
- Local source/config/test commit: `a25e6bf66214a82adf9a56d492a85a28207ef248` (`diagnostics: add Session A checkpoint evaluation`).
- Compact evidence files: `agents/dit_pilot_diagnostics_v1/evidence/session_a_260909_summary.json` and `agents/dit_pilot_diagnostics_v1/evidence/session_a_260909_report.md`.
- Generated report was rerun after the source commit and records `actual_code_commit=a25e6bf66214a82adf9a56d492a85a28207ef248`.
- The follow-up local documentation/evidence commit is the next commit in this branch history; no push is authorized.
