# A — 验收

本文件列出必须可验证的行为，不要求模型结果变好。

## 实现正确性

- A保持原分支；B/旧codec/旧pilot输出无修改；没有optimizer训练或test payload访问。
- 现有 best-validation checkpoint 与正确codec/statistics成套加载；记录实际哈希；原stat文件前后不变。
- CUDA上跑真实推理、解码与关键指标；包括一个真实clip smoke，不以skip/CPU代替。
- Future target mutation 不改变 observation condition、persistence/CV/latent-prior 构造；同噪声的生成不随隐藏目标变化。oracle/sensitivity例外必须显式标注。
- H4/H8、L4/L8索引、时间、掩码与边界定义正确；所有metadata随裁剪正确重建。
- raw detail=0 的构造与标准化/反标准化往返一致；无scalar-mean误用，无raw_detail旁路。
- 每样本 seed不依赖batch顺序；重新分batch的sample/system汇总一致。
- ACF/相关性零方差、过短窗口、零频谱分母有null与原因，无NaN静默平均。
- 四field perturbation/swap只改future、保留observed/masks；含oracle信息的输出不能列为真实baseline。
- diversity定义的xyz求和正确，不继承名称是RMSD却差sqrt(3)的实现。

## 测试顺序与最小集

实现后先为新tests显式指定cuda device/skip策略，使无CUDA时报BLOCKED，而非验收PASS。覆盖：

1. observed-only source构造 + target mutation；
2. 标准化zero-detail往返；
3. H/L裁剪 + 非均匀dt finite difference；
4. per-sample/system aggregation及batch重排；
5. rotation/translation及mask一致；
6. 真实 R4 一个clip完整链路，随后R2加载/采样smoke。

纯metadata测试允许CPU，单独列出。通过后执行完整main/deep，不需要全仓回归先行。若受影响旧测试需要跑，仅选择相关项，命令写入HANDOFF。

## 交付

- 72 clips × H4/H8 的R2/R4/基线比较，horizon表和8体系分组结果；
- 8-clip深诊断：tau loss、latent统计、decoder扰动/field-swap、8/16步与4 draws；
- 数据缺失或运行未完时逐项标注not_run，不宣称完成；
- 新evaluator/CLI可复现；提交小summary/report及实际命令；
- 结论明确 prior baseline 是否可用，以及下一轮优先验证的一个假设，不自动训练。

完成状态 `A_DIAGNOSTICS_READY_FOR_REVIEW`，不足则 `A_BLOCKED_WITH_PARTIAL_DELIVERY`。旧pilot的低dynamic_correlation不单独作为拒绝随机模型的标准；不对源分布不同的RF loss硬排名。
