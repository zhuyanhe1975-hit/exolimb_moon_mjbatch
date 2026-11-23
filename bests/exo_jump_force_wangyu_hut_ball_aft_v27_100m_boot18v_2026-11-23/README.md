# Wangyu HUT ball aft v27 (100m + `fit3d_18v` boot hull) · `model_399`

- **几何**：`fit3d_18v`（18 顶点 / 32 三角面，8 顶点贴合外底 + 10 顶点 3D 靴面凸包，`assets/boot_hull_18v/fit.json`）
- **训练地形**：100m × 100m 月面（Warp 训练采用精确 36×20m 切片，回放自动使用完整连续 100m 地形）
- **100m 150s（5 seeds × 30s）**：`1.5 m/s` 助力 500 N **0 摔倒（5/5）**，巡航 `1.437 m/s`，质心起跳 `+9.9 cm`，脚底离地 `13.9 cm`（峰值 `24.5 cm`），腾空 `0.68 s`；无助力对照 3 摔倒
- **16m 标准 6 地形**：`0.0 / 0.5 / 1.0 / 1.5 m/s` 助力全过 **0/24 摔倒**（`1.5 m/s` 无助力 4/6 摔倒，助力降耗 `-23.2% J/m`）

在仓库根目录运行以下命令回放：

```bash
# 100m 地形长距离连续跳跃（推荐，1.5 m/s，助力 500 N）
./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --terrain-size 100 --jump-speed 1.5

# 16m 标准月面地形（1.5 m/s，助力 500 N）
./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --jump-speed 1.5

# 查看 fit3d_18v 碰撞网格（physx 视图）
./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --terrain-size 100 --jump-speed 1.5 --viz physx

# 无助力对照（0 N）
./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --terrain-size 100 --jump-speed 1.5 --assist-force 0
```
