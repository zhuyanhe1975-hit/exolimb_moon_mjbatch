# Wangyu-Replant-v12 Archive (`hut_hinge_hoof_v8_detached`, Magnetic Damped Ball-Joint Hoof)

- **Checkpoint**: `model_999.pt` (1000-iter warm-start fine-tune from archived `Wangyu-Replant-v11` `model_1999.pt`)
- **Mechanism**: `hut_hinge_hoof_v8_detached` (3-DOF damped ball-joint magnetic hoof, single rubber-box physics collision, zero-overconstraint `mjEQ_WELD` boot side magnetic dock)
- **Geometry / Suit**: Wangyu EVA (`fit3d_18v`), lunar gravity `1.635 m/s^2`, EVA knee hard cap `108 deg`
- **Reference**: Self-contained `reference/motion.npz` (`wangyu_replant_ref_v8`, SHA `53a99ea07409`)

## Independent CPU Verification (Full Docked-Prone Magnetic Release -> Forward Deploy -> 4-Phase Stand-Up)
- **Deploy prelude (`2.35 s`)**: magnetically welded to boots (cyan LED) -> releases (`eq_active=0`, amber LED) -> outward anti-penetration sweep (`min clearance = +1.82 cm`) -> front anchor handoff
- **RL recovery**: `success = True` at `8.917 s` (`failed = False`), phases `{'1': 3.483, '2': 3.5, '3': 4.15}`
- **Tracking & anti-slip**: `human q_rmse = 3.45 deg`, `tip_rmse = 1.88 cm`, loaded hoof slip = `0.0107 m/s`, `min strut/hoof-human clearance = +1.17 cm`
- **Final 1 s standing**: `z = 0.8088 m`, `min upright = 0.9995`, `sole >= 86.8 N`, `shin = 0.0 N`, `tip >= 25.3 N`, `sole_dx <= 1.52 cm`

## Replay
```bash
./replant.sh play --best
# or explicitly:
./replant.sh play bests/replant_wangyu_hut_hinge_hoof_v12_2026-11-24/model_999.pt
```
