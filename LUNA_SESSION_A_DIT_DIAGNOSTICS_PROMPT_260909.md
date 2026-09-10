# Session A — 现有 pilot 分支上的诊断与评测

用户已经授权实施本任务。不要只复述计划；先完成必要检查、实现，再执行本轮 CUDA 诊断。不要从旧 TASKS 恢复已经结束的 pilot 或直接启动整套 pytest。

## 工作树、阅读顺序与阶段切换

- 目标工作树：`/data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1`。
- 必须保持当前分支 `exp/dit-state-detail-t1-pilot-v1`，不创建分支、不 checkout、不新建 worktree。
- 已审阅快照：`a22f60c4ffd1f502ab300352a00eafe841b1f8d2`。记录实际 HEAD；若是其后继，检查增量再继续，不能把用户的新结果覆盖成旧快照。若分支不同或有另一写入者，报告冲突而不是切换。
- `git status --short` 的已有变化属于用户。未涉及本任务的变化保留，不得 `git add -A`。

完整读取：

1. 当前根 `AGENTS.md` 和 `agents/AGENTS.md`；
2. 本 prompt；
3. `agents/dit_parallel_v2/CONTRACT.md`、`ROOT_AGENTS_APPEND.md`、`NEXT_STAGE.md`；
4. `agents/dit_pilot_diagnostics_v1/AGENTS.md`、`PLAN.md`、`ACCEPTANCE.md`、`DECISIONS.md`、`TASKS.md`、`HANDOFF.md`；
5. 旧 `agents/dit_state_detail_pilot_v1/PILOT_PROTOCOL.md` 和 `HANDOFF.md`，作为结果与路径来源，只读。

第一次编辑是在本工作树根 `AGENTS.md` 末尾追加共享 ROOT_AGENTS_APPEND 中的内容，并记录本 session 为 A。保留旧章节；追加内容明确替代旧 phase 的“停止/目录/禁止当前评测”规则，但不解除 CUDA、数据、凭证与用户变更保护。无需改父级 `agents/AGENTS.md`；新的任务目录有自己的 `AGENTS.md` 覆盖陈旧任务说明。

## 执行顺序

1. 查证容器入口、实际挂载路径、CUDA、数据/codec/DiT checkpoint/statistics；生成 input manifest。缺 checkpoint 时先报告准确缺件，同时可继续不依赖权重的实现与测试；不得用随机权重冒充 pilot。
2. 实现 PLAN 里的 per-sample evaluator、observed-only 基线、固定噪声的 RF 分桶、扰动敏感性及 sampler 对照。
3. 新数值单测必须实际在 CUDA 上执行；先一个真实 clip 的 end-to-end 冒烟，再跑受影响的测试。不要先跑全仓回归。
4. 冒烟通过后自动执行本轮限定的 validation 评测，使用现有 best-validation checkpoint，不启动 optimizer。
5. 汇总诊断与局限，更新 TASKS/HANDOFF，执行 `git diff --check`，只提交自己文件及小型报告到当前本地分支。
6. 状态设为 `A_DIAGNOSTICS_READY_FOR_REVIEW`；如权重/环境缺失则为 `A_BLOCKED_WITH_PARTIAL_DELIVERY`。停止，不 push、不合并、不训练。

允许范围、CLI、指标和输出规范均在 PLAN/ACCEPTANCE。不要把 B 的加速或尚未授权的 conditional-prior 训练混进本 session。

最终回复只包含：完成的主要诊断、最影响下一步选择的结果、代码 commit、报告路径、是否需要用户补文件。不宣称随机轨迹必须逐帧复现参考 MD。
