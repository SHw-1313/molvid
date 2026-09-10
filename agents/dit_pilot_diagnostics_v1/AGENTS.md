# Session A instructions

本目录是 pilot 后诊断的新阶段，覆盖父级 agents/AGENTS.md 中旧 codec-v1 的任务、CPU-first、因果分块和旁路要求；保留仓库安全/环境规则。本轮保持 unified 16-frame R2/R4 block-local codec 与 H4/H8。

先读根 Session A prompt，再读共享 CONTRACT、NEXT_STAGE 和本目录 PLAN、ACCEPTANCE、DECISIONS、TASKS、HANDOFF。PLAN/ACCEPTANCE 不随结果改目标；真实实现事实追加到 DECISIONS，进度填 TASKS，证据填 HANDOFF。

只在当前 `exp/dit-state-detail-t1-pilot-v1` 分支工作，不建分支、不切换、不碰 B 工作树。不改 module/trainer/旧 RF/codec/旧 evaluator；新增 evaluator/runner/config/test 由 A 独占。使用现有已训练 checkpoint，不训练、不调 loss，不打开 test。

所有 Python 在 enter-container + torch-ito；模型数值检查与实数据运算用 CUDA，不能以 CPU fallback 或 skipped test 交付。优先实现，再 targeted CUDA / real-clip smoke，再完整的本轮诊断；无需在实现前全仓 pytest。

报告按样本、体系和 draw 层级汇总。清楚区分 reference path 对齐误差与 stochastic dynamics 分布。不能把 generated-Oracle 的差值当成独立解码敏感性证据。

完成后本地提交自己文件，状态 A_DIAGNOSTICS_READY_FOR_REVIEW 并停止。不自动 push、合并或开始第三阶段。
