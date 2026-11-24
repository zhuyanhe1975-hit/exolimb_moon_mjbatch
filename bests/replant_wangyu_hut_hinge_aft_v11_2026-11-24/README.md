# Wangyu 跌倒脱困站立 v11 (`hut_hinge_aft_v7_detached` + `fit3d_18v` + DeepMimic 动捕跟踪残差) · `model_1999`

- **任务 / 机构**：`Wangyu-Replant-v11` / `hut_hinge_aft_v7_detached`（继承 `hut_hinge_aft_v7` HUT 挂点与几何，下端球铰脱开为自由月面触地末端，每侧 3 自由度双向位置 PD：矢状面俯仰 `[-125°, +35°]` ±150 N·m、单向外摆侧倾 `[-25°, +2°]`/`[-2°, +25°]` ±80 N·m、伸缩 `0.56–1.18 m` ±450 N，全机 29 驱动）
- **参考轨迹**：`wangyu_replant_ref_v7`（[motion.npz](../../assets/wangyu_replant_ref/motion.npz)，527 帧 / 8.77 s，含起始贴地双手、全帧第一阶段双胫着地、0.40 s 跪稳缓冲、0.62 m 支杆换撑与终态双脚平行站立）
- **训练配置**：MuJoCo Warp CUDA 8192 并行世界 × 2000 轮（60 Hz 控制 / 120 Hz 物理，月球重力 `1.635 m/s²`），`[512, 256]` ELU，干净 5 项严格正值 DeepMimic 跟踪奖励（关节位置/速度 + 有符号 3D 根部位置/四元数姿态/速度 + 8 关键末端 + 接触一致）+ `+0.20 s` 前瞻观测 + 60 Hz 直驱小幅残差 PD + 真实支杆-人体/手臂-胸甲非对称碰撞防穿透

## 独立 CPU 完整趴姿（`k=0`）物理验收结果（[final_verification.json](final_verification.json)）

| 指标 | 实测结果 |
| --- | --- |
| 完整四阶段物理推进 | **通过**（Phase 1 跪稳 `3.17 s` → Phase 2 抬杆离地 `3.65 s` → Phase 3 换撑落点 `4.07 s` → 站稳 1 s 验收 `9.52 s`） |
| 换撑实际位移 / 抬杆净空 | `60.6 cm` / 球心最高 `6.69 cm`（离地净空 `3.49 cm`） |
| 全程跟踪误差（人体关节 / 根部姿态 / 支杆末端） | `4.62°` / `3.41°` / `2.66 cm` |
| 支杆-人体全程最小净距 | **`+1.51 cm`**（零穿透） |
| 整机质心角动量 / 人体均方根出力 | `5.84 N·m·s` / `6.58%` |
| 末 1 s 稳态站立高度 / 直立度 | `0.8094 m` / `min upright = 0.9991` |
| 末 1 s 双脚承重 / 双胫载荷 / 支杆护稳 | 每脚最低 `92.6 N`（双脚合计 `0.816 BW`）/ 胫骨 `0.0 N` / 每杆最低 `23.4 N` |
| 末 1 s 双脚平行度 / COM-支杆距 / 速度 | `max sole_dx = 3.95 cm` / `2.93 cm` / `0.031 m/s` |

> 注：本次为平地固定趴卧初态下的独立 CPU MuJoCo 闭环物理验收，不构成起伏地形或大外力扰动下的鲁棒性结论。

## 回放与参考预览命令

在仓库根目录执行：

```bash
# 回放归档的最佳检查点（Studio + 宇航服蒙皮）
./replant.sh play --best

# 或显式指定归档路径
./replant.sh play bests/replant_wangyu_hut_hinge_aft_v11_2026-11-24/model_1999.pt

# 查看物理代理几何与外骨骼碰撞细节
./replant.sh play --best --viz physx

# 纯运动学预览参考轨迹
./replant.sh preview
```
