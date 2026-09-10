# B handoff

Status: NOT_STARTED

当前只有指令模板；没有运行数值测试或profile。

## 环境与输入

- worktree / branch / base / final commit:
- user changes preserved:
- enter-container invocation / mapped cwd / python / torch:
- GPU UUID / CUDA / BF16 / TF32:
- codec / DiT checkpoint / statistics hashes:
- reference source commit:

## 实现与兼容

- vectorized pooling/broadcast layout:
- actual attention backend/kernel:
- checkpoint contract/load treatment:
- duplicate encoder call test:
- cache validity / invalidation:

## 数值证据

逐项记录case、dtype、四field error、gradient/update error、容差、命令、退出码。注明非零attention gate、实际CUDA与skip情况。

## 性能表

填写R4 small/median/large、R2 median的N/K/M、weights/input hash、reference与optimized forward/backward/optimizer/end-to-end median/p90、tokens/s、warmup后peak、kernel与trace瓶颈。不要引用历史all-process peak代替新测量。

## 交付与建议

- semantic parity:
- speed/memory conclusion:
- unresolved synchronization/padding cost:
- exact commands / small evidence paths / full output paths:
- local commit:
- ready for integration or missing evidence:
- final status:

停在review，不合并A，不训练conditional prior。
