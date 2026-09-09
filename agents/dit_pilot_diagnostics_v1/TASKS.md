# A tasks

按依赖执行，完成一项记录实际证据。不重跑历史pilot。

- [x] A001 保持当前分支、检查用户改动、追加根phase说明；核验环境/输入权重/statistics/validation schedule。Evidence: root transition appended; preflight passed with frozen manifest/checkpoint/statistics hashes.
- [x] A002 实现新runner、per-sample evaluator、四种observed-only基线与horizon/aggregation。Evidence: `evaluation/dit_diagnostics.py`, runner, 720-row main outputs per candidate.
- [x] A003 实现tau/latent/perturbation/field-swap/8-vs16/multi-draw诊断。Evidence: both deep summaries completed after the vector-mask fix.
- [x] A004 targeted CUDA数值测试与R4/R2真实clip smoke；只跑受影响回归。Evidence: CUDA diagnostics test ran (not skipped), both real-candidate smoke reports passed, affected regression subset passed 49 tests.
- [x] A005 执行72-clip main + 8-clip deep，保留逐项状态与输入输出hash。Evidence: R2/R4 main each 720 rows/72 samples/8 systems; R2/R4 deep each sample_count=8, test_opened=false, optimizer_steps=0.
- [x] A006 小型报告、DECISIONS/HANDOFF更新、diff检查、本地明确范围commit，停止。Evidence: source/config/test commit `a25e6bf66214a82adf9a56d492a85a28207ef248`; compact JSON/Markdown evidence is recorded under `agents/dit_pilot_diagnostics_v1/evidence/`; no push, merge, B, or training.

Current status: A_DIAGNOSTICS_READY_FOR_REVIEW

## Review-fix evidence — 2026-09-09

- The first R4 deep attempt failed only in diagnostics latent-summary mask broadcasting; it was not counted as a result. The focused regression and corrected CUDA deep run passed afterward.
- Final affected regression command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_dit_diagnostics.py tests/test_dit_evaluation.py tests/test_state_detail_latent_adapter.py tests/test_latent_rectified_flow.py tests/test_molecular_dit.py tests/test_dit_trainer.py` — 49 passed.
