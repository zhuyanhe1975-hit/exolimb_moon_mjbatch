# Wangyu HUT sagittal-hinge strut v1 (`hut_hinge_aft_v7` + 100m + `fit3d_18v`) · `model_399`

- **任务 / 机构**：`exo_hinge_jump` / `hut_hinge_aft_v7`（上端 HUT 挂点矢状面双向力矩驱动 `±100 N·m/侧` + 同轴直线伸缩推力 `500 N/侧`）
- **几何**：`fit3d_18v`（18 顶点 / 32 三角面贴合靴底凸包，`assets/boot_hull_18v/fit.json`）
- **100m 150s（5 seeds × 30s）**：
  - `1.5 m/s` 全开（`500 N + 100 N·m`）：**0/5 摔倒**，巡航 `1.350 m/s`，外骨骼总做功占比 **`21.1%`**，离地 `13.5 cm`（峰值 `30.3 cm`）
  - `1.5 m/s` 仅关铰链（`500 N + 0 N·m`）：`4/5` 摔倒，人体代谢 `+8.6%`（`686.0 -> 745.3 J/m`），人体正功 `+14.1%`，外骨骼占比跌至 `5.9%`
  - `1.5 m/s` 全关（`0 N + 0 N·m`）：`12/5` 摔倒，人体代谢 `+16.2%`（`797.3 J/m`）
  - `1.0 / 0.5 m/s` 全开：均 **0/5 摔倒**，巡航 `0.997 / 0.547 m/s`，外骨骼占比 `20.7% / 21.3%`
- **16m 标准 6 地形**：`0.5 / 1.0 / 1.5 m/s` 全开 **0/18 摔倒**，`1.5 m/s` 降耗 **`ΔJ/m = -23.5%` / `ΔhPos = -20.8%`**（关助力摔倒 `5/6`）
- **已知局限**：`v=0.0 m/s` 原地站立在 16m 六地形上出现 `4/6` 后仰摔倒（`move_gate` 在静止时将铰链力矩归零，后端新增 `0.70 kg` 电机质量与阻尼引起微偏后），留待 `v2` 优化。

在仓库根目录运行以下命令回放：

```bash
# 100m 地形长距离连续跳跃（推荐，1.5 m/s，500 N 伸缩 + 100 N·m 矢状面铰链）
./run_exo_hinge.sh bests/exo_hinge_jump_wangyu_hut_hinge_aft_v1_100m_boot18v_2026-11-24/model_399.pt --terrain-size 100 --jump-speed 1.5

# 或使用 --best 快捷入口
./run.sh --best exo_hinge_jump --viewer studio --viz skin --terrain-size 100 --jump-speed 1.5

# 消融对照：仅关闭矢状面铰链（0 N·m）
./run_exo_hinge.sh bests/exo_hinge_jump_wangyu_hut_hinge_aft_v1_100m_boot18v_2026-11-24/model_399.pt --terrain-size 100 --jump-speed 1.5 --hinge-torque 0

# 查看 physx 碰撞与铰链几何
./run_exo_hinge.sh bests/exo_hinge_jump_wangyu_hut_hinge_aft_v1_100m_boot18v_2026-11-24/model_399.pt --terrain-size 100 --jump-speed 1.5 --viz physx
```
