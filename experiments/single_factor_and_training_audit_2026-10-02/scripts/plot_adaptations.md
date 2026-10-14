# 单参数图的适配

原脚本副本是 `scripts/original_physical_plot.py`。它与 `data_and_analysis/physical_element/data-map-single-phy-element-disturbance.py` 以及 `paper_revision_2026-10-02_style_v2/scripts/original_physical_plot.py` 的 SHA-256 相同，都是 `a7849750016ffb7b7c326a992899fde977481fa14745ecbfaabd40c781db0756`。副本没有改。

绘图脚本是 `scripts/plot_single_factor_clean.py`。相对原脚本的改动：

1. 读 `single_factor_rates.csv`。不打开六月的 `eval_phy_*.csv`，也不打开旧的 summary 表。
2. 质量 1 kg、滑动摩擦 1、增益比例 1 用的是已经写进这张表的 P0-v3 成功率。横轴注明这一点。阻尼比例 1 是 P0，不画进阻尼图。
3. 横轴写实际被缩放的量。阻尼是全部 15 个 `dof_damping`。增益是 `actuator_gainprm` 的第 0 列。摩擦是几何体摩擦的第 0 分量。
4. 误差棒仍是三个策略成功率的样本标准差，`ddof=1`，单位是百分点。原脚本的 `DataFrame.std` 就是这个定义。它不是置信区间。
5. 每张图写 300 dpi PNG 和 SVG。不调用 `plt.show()`。
6. 种子保持整数，图例写作 Seed 0。
7. 纵轴保持原 Figure 1 的 0 到 110。
8. 画完后把每个点的均值和样本 SD 与 `single_factor_summary.csv` 比较，容差 1e-9，不一致就退出。

布局仍是四张分开的图：各策略一条折线，黑色虚线为均值，误差棒为样本 SD。
