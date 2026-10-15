# 更早的脚本

这些目录是放进 `train/`、`test/` 和 `eval/` 之前的训练和评测草稿。每个目录留着当时 `import` 的那份 `domain_rand.py`。

当前入口仍是仓库里的 `train/`、`test/` 和 `eval/`。这些草稿要在各自目录里运行，并把仓库根目录放进 `PYTHONPATH`，这样能找到 `panda_mujoco_gym`。`domain_rand.py` 用的是该目录里的那一份。

| 目录 | 内容 |
|---|---|
| `dr_extreme/` | 极端范围的域随机化训练和扫描 |
| `dr_project/` | 定稿前的域随机化训练 |
| `multi_element/` | 多参数联合扰动的早期评测 |
| `ubuntu_eval/` | 另一套域随机化扫描 |
| `xml_generalization/` | 直接改 XML 的早期泛化测试，配套 `worker.py` |
