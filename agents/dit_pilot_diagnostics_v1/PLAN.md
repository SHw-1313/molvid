# A — 现有 checkpoint 的生成诊断

## 目标与可复现输入

回答三个问题：生成是否损坏几何/运动分布；差距主要来自哪一个 latent field/flow time/decoder 敏感性；block-state persistence 是否值得作为下一次 conditional source 的中心。

读取 shared CONTRACT 指定的 pilot checkpoint/codec/train statistics，记录哈希，不能替换成另外一个更优 codec。无需用户重新上传整个输出目录；缺件时报告 candidate、文件名、期望路径。主输入配置应有 manifest_root、pilot_root 和可选显式 checkpoint/statistics 路径，不依赖相对 cwd 寻找二进制。

实现四个新文件：

- `evaluation/dit_diagnostics.py`：观测基线、per-sample/cropped-horizon metrics、聚合、latent/perturbation 分析。
- `scripts/evaluate_dit_pilot_diagnostics.py`：只读加载、CUDA 检查、分阶段执行与断点恢复。
- `config/dit_pilot_diagnostics_v1.yaml`：本文件默认值。
- `tests/test_dit_diagnostics.py`：元数据/CUDA 分类测试。

复用现有 codec encode/decode 与指标公式，但不修改旧 evaluator。需要裁剪 batch 时使用新 helper 正确重建 frame_mask、time_ps、abid、loss/align masks；不能只切坐标而保留错误时间或原子索引。绘图可选，Markdown/JSON 表必须有。

## 数据规模与随机性

主评测沿用8个 validation 体系 × 3 replicas × windows 0/30/61 = 72 clips；H=4/8。R2/R4都用 best_validation.pt。完整主比较16步 Euler，每个 sample/H 一个固定 draw；不训练、不访问 test。

深入诊断固定每个 validation 体系的 R1 replica、window30，共8 clips；用原 manifest 名称生成，不手选好看的体系。8 vs16步、multi-draw、tau 和 perturbation 在这个子集执行。可按配置切阶段，已有分阶段结果哈希吻合时复用，避免每次重跑全部。

新评测 seed 从稳定的 (master_seed, sample_id, H, draw_id, diagnostic_kind) 哈希得到，不使用 Python randomized hash，不依赖 batch offset 或训练 step。8/16步对照采用同一初始噪声。R2/R4维度不同，不能声称逐元素噪声完全相同，只能保证同样 seed 协议和采样数量。

## A1. 非学习基线

所有基线构造只读 observed [0,H)，不能从完整 clip 的 future state 或均值取信息。真实未来仅供目标/评分，latent oracle 单独命名。frame0 origin 与现有 codec gauge 一致，保留 original loss_mask。

1. `coordinate_persistence`：复制 x[H-1]。
2. `coordinate_constant_velocity`：用 x[H-2],x[H-1] 和真实 dt 估速度，按 query time 外推；注明 gauge/未对齐或观测内对齐约定，不用 future 对齐来生成。100ps 位置差不是瞬时物理速度，故仅为基线。
3. `latent_block_state_persistence`：原始 codec 空间把最后完整 observed token 的 state_h/state_v 复制到 future；detail_h/detail_v 设原始零。observed token 保持原样。不能复制 raw_detail 辅助旁路。
4. `coordinate_repeat_reencoded`（仅8-clip diagnostic）：observed history 保持，未来用最后观测坐标重复，作为合成基线经过同一冻结 codec；不是训练数据、不赋予伪 MD 动力学标签。与第3项分开命名，不能偷偷替代它。

第3项不是最后原子构象：R2/R4 state 是 Haar-like 聚合后投影，零 detail 表示块内无运动，未必有正确 bond 或 H-1→H 边界。报告其 bond/contact、与坐标 persistence 差异、首未来帧跳变。zero detail 在 raw 与 standardized space 分开处理：normalized scalar detail 可能为 -mean/std，而不是0。

把上面基线与 `codec_oracle`、现有 `dit_generated` 放同一表。oracle 使用完整目标编码只作为 reconstruction reference，不能作为真实可部署 baseline。

## A2. 评测与聚合修正

保留完整 clip 16帧采样；评分提供：

- H4，L4：[4,8)；H4，L8：[4,12)。
- H8，L4：[8,12)；H8，L8：[8,16)。
- 各自完整 future：H4 [4,16)、H8 [8,16)，明确是不同 horizon。
- boundary 是 H-1→H 的转移，不是一个孤立区间内的逐帧误差。

上述同 L 是相对 forecast lag 可比，但 target 起点不同；不能只凭此推断多 history 的因果收益。要研究纯 history effect，下一阶段另做相同未来目标/相同锚点的 observation ablation，本轮不改 tokenizer。

主指标：aligned RMSD、dRMSD、bond RMSE、contact F1/occupancy、boundary displacement-vector error、RMSF预测与目标及其相关、lagged ACF、位移/频谱统计。写明单位、align_mask、loss_mask、contact排除规则和物理 dt。

沿用历史指标名称的同时说明：`dynamic_correlation` 是生成与目标速度逐点相关还是自相关，不混为一谈。随机采样逐点相关可低，ACF形态与RMSF/占比才是另一类证据。频谱/RMSF 在短窗口是短时诊断，不推论长期 kinetics。

必要时用 per-sample 逐条评分（A无需追求最快 evaluator），得出结构化行。sample-equal 先在同一样本的 draws 内平均，再等权样本；system-equal 先体系内平均，再8体系等权；生成 draw 不作为独立蛋白样本。保存每体系 paired difference。bootstrap 若做，以体系为单位。F1等非线性量明确是宏平均还是汇总 counts 计算。

旧26 batch均值不能通过随意乘 batch_size 精确还原；如缺 per-sample统计必须重算。聚合重分 batch 应得到相同结果。全局量与分体系量一起保存。

零运动基线的 ACF、Pearson、频率比若分母为0，输出 null + applicability_reason，不能用0/1填充。所有有限差分按实际 dt。主结果不使用 best-of-N。

## A3. Latent 与 flow-time 诊断

8-clip子集，固定2个 noise draws，tau midpoints [0.05,0.25,0.50,0.75,0.95]，分别报告四 field masked RF MSE，按 H 分开。与当前 objective 的训练 path 完全一致，不改变模型，不使用另一个 loss 的尺度来下结论。

报告 raw/standardized 两套 state_h/detail_h/state_v/detail_v channel summaries：mean、std/RMS、norm分位数、generated-to-reference范数比、state/detail关系。scalar可用对角标准化距离，vector用旋转不变量范数和跨channel Gram摘要。训练参考分布来自 train only；优先复用原 statistics，额外分位数用固定、均匀覆盖48体系的 train reference subset（例如每体系R1/w30）单独统计，不能写回原 statistics，也不能称其是全训练集精确分布。

标出不支持的全协方差/密度估计。单通道z-score大只表示异常候选，不是严格流形距离。保存生成端分布与原始 encoder latent 的对应图/表。

## A4. Decoder sensitivity 与 field 替换

在 normalized oracle future latent 注入 isotropic 扰动，scale=[0,0.01,0.05,0.10]，各2个draw；state-only、detail-only 两类（h/v按对应field处理），observed 完全不动。decode前 inverse-normalize 一次，记录噪声实际RMS与几何/边界变化。vectors xyz 同方差，不对 xyz 单独统计缩放。

额外做三种 diagnostic 混合：oracle state + generated detail；generated state + oracle detail；generated state + raw-zero detail。使用未标准化field或明确一次反标准化，保持schema/mask。含真实 future 的混合只用于归因，标题标注 `oracle_field_swap_not_deployable`，不能算模型性能。

只对 generated detail 清零不是可部署 persistence baseline，因为 generated state 自身随时间变化；单独命名。以上能判断“哪类误差被 decoder 放大”，不能证明通过 noise robust decoder 训练一定解决。

## A5. Sampling 与多样性

8 clips，H4/H8，每个4个固定 draws，比较8步/16步 Euler。报告每个draw与均值/离散度，不能挑最好结果。若现有接口只允许8/16，就不扩展solver，不跑32/Heun。

样本间 diversity 使用明确的 RMSD 定义（先sum xyz再平均原子），区分对齐与raw，配合RMSF/contact分布解释“变化”是真运动还是几何噪声。仅噪声产生的高diversity不能被视为优点。

## 实现后提供的 CLI

以下是需要实现的接口，不是现有脚本：

```text
python scripts/evaluate_dit_pilot_diagnostics.py --config config/dit_pilot_diagnostics_v1.yaml --stage preflight
python scripts/evaluate_dit_pilot_diagnostics.py --config config/dit_pilot_diagnostics_v1.yaml --stage smoke --device cuda:0
python scripts/evaluate_dit_pilot_diagnostics.py --config config/dit_pilot_diagnostics_v1.yaml --stage main --device cuda:0
python scripts/evaluate_dit_pilot_diagnostics.py --config config/dit_pilot_diagnostics_v1.yaml --stage deep --device cuda:0
python scripts/evaluate_dit_pilot_diagnostics.py --config config/dit_pilot_diagnostics_v1.yaml --stage summarize
```

设备由已审计GPU的CUDA_VISIBLE_DEVICES映射而来。stage preflight/summarize允许只有元数据CPU工作；不得伪造CUDA inference。若需要覆盖路径，提供manifest-root/pilot-root/output-root配置项并在 --help 中说明。所有命令仍在容器内执行。先完成实现，再用 --help 检查接口。

输出 `outputs/dit_state_detail_diagnostics_v1/<run_id>/`：inputs.json、config_resolved.yaml、sample_metrics.jsonl、summary.json、report.md，以及deep各阶段小汇总。run_id不复用旧pilot路径。恢复时比较代码/协议/input/seed hash，不盲目跳过已有文件。

summary包含schema、actual_code_commit、inputs、sample/system counts、seeds、device/dtype、horizons、aggregation、baseline/generation metrics、latent findings、prior_baseline_usable（支持/不支持/待定及证据）、limitations。仅将小型 summary/report 放入本目录 evidence/ 提交；大的逐样本输出留在新output目录。
