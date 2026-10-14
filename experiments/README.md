# 补充实验

2026 年 10 月在 2025 年 6 月那三个 `batch512_her4` 检查点之后补做的评测，以及 SpatialDR / Fixed Large 的受控重训。数字以各目录里的报告为准。

没有放进来的内容：冒烟运行、中断后废弃的 Fixed Large 种子 0，以及编译出来的 `panda_push.mjb` 缓存。正式模型仍在 `spatial_training_controlled_2026-10-02/runs/`。

| 目录 | 内容 |
|---|---|
| `physical_element/`、`spatial_generalization/`、`hyperparameter_analysis/` | 论文初稿用过的汇总表 |
| `corrective_2026-10-01/` | 延迟和四项物理量的四格。无延迟联合物理 1578/1800（87.67%），只有延迟 454/1800（25.22%） |
| `corrective_round3_2026-10-01/` | 清洁初始化和写法 C 之后重跑的四格。无延迟联合物理 1715/1800（95.28%），只有延迟 443/1800（24.61%） |
| `single_factor_and_training_audit_2026-10-02/` | 单参数复测。质量和滑动摩擦仍高；整段阻尼有六档低于 98% |
| `spatial_clean_2026-10-02/` | 边长 0.10–0.80 m 的空间曲线。0.80 m 为 1020/1800（56.67%） |
| `spatial_training_controlled_2026-10-02/` | 六个模型各训练 1,000,002 步。L = 0.80 m 上 SpatialDR 1666/1800（92.56%），Fixed Large 1660/1800（92.22%） |

种子都是 0、42、123。成功判据是物体到目标的距离小于 0.05 m。
