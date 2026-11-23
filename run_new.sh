# 历史归档参考：
# ./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v19d_crouch_launch_2026-11-22/model_250.pt --jump-speed 1.5
# ./run_exo_force.sh logs/cpu_exo_train/v22_calibrated_sym_cruise/model_349.pt --terrain-size 100 --jump-speed 1.5

# 最新归档 v27（100m 地形 + fit3d_18v 贴合靴底凸包，1.5 m/s，助力 500 N）
./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --terrain-size 100 --jump-speed 1.5 "$@"

# 16m 标准月面地形（1.5 m/s，助力 500 N）
# ./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --jump-speed 1.5

# 查看 fit3d_18v 碰撞几何（physx 视图）
# ./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --terrain-size 100 --jump-speed 1.5 --viz physx

# 无助力对照（0 N）
# ./run_exo_force.sh bests/exo_jump_force_wangyu_hut_ball_aft_v27_100m_boot18v_2026-11-23/model_399.pt --terrain-size 100 --jump-speed 1.5 --assist-force 0
