# 外骨骼 1.7 秒长跳纠偏微调

最终检查点为 `model_159.pt`。训练从已完成的 1.7 秒长跳策略初始化，启用背包与腰部的持续侧倾代价。左右支杆仍由策略分别给出推力指令。

MuJoCo Warp CUDA 训练使用 8192 个环境，每次更新采样 32 步，共 160 次更新、41,943,040 个控制步，耗时 2 分 39 秒。训练地形为 12、3、24、47，种子为 42。

| 训练全程统计 | 纠偏前长跳训练 | 本轮微调 |
|---|---:|---:|
| 人体估算能耗率 | 477.6 W | 470.5 W |
| 人体腿部力矩 RMS | 24.40 N·m | 23.88 N·m |

统计涵盖各自训练全程的探索采样。同速回放待验证，关注背包侧倾、落点横向漂移、落地间隔、单跳距离与人体 J/m。

`config.json` 和 `terrain_manifest.json` 保存训练配置与地形校验值。`training.log`、TensorBoard 事件和 `training_efficiency.json` 保存原始训练记录，`summary.json` 保存结果摘要，`archive.json` 记录归档来源和 SHA-256。

## 从项目根目录回放

```bash
./run_exo_force.sh bests/exo_jump_force_long_hop_roll_2026-11-19/model_159.pt \
  --terrain-idx 16 --jump-speed 1.0
```
 