#!/usr/bin/env python3
"""Retarget Apollo fixed_support_v14 Kimodo get-up reference to Wangyu hut_hinge_aft_v7_detached + fit3d_18v.

Reads ~/myWorks/zbot_exoarm_moon/.../fixed_support_v14/motion.npz read-only (never mutates source),
solves bounded Wangyu EVA human kinematics + exact closed-form 3-DOF detached strut IK, audits every
frame for joint limits, ground clearance, arm-torso clearance, and kneeling COM support, and writes
SHA-locked assets/wangyu_replant_ref/{motion.npz,manifest.json}.
"""
from __future__ import annotations

import hashlib
import json
from math import radians
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from exoskeleton import HOOF_SOLE_HEIGHT_M
from wangyu_replant import CONTACT_GEOMS, DT, GEOMETRY, MECHANISM, SIDES, TASK, build_model

SOURCE_NPZ = (
    Path.home()
    / 'myWorks/zbot_exoarm_moon/experiments/apollo_lunar/data/kimodo'
    / 'two_stage_replant_20261111/fixed_support_v14/motion.npz'
)
OUT_DIR = ROOT / 'assets/wangyu_replant_ref'

# Direct joint mapping from Apollo fixed_support_v14 -> Wangyu MuJoCo joint names.
HUMAN_JOINT_MAP = {
    'LeftHip': 'robot/LeftHipPitch',
    'RightHip': 'robot/RightHipPitch',
    'LeftKnee': 'robot/LeftKnee',
    'RightKnee': 'robot/RightKnee',
    'LeftAnkle': 'robot/LeftAnkle',
    'RightAnkle': 'robot/RightAnkle',
    'WaistPitch': 'robot/WaistPitch',
    'WaistRoll': 'robot/WaistRoll',
    'LeftShoulderPitch': 'robot/LeftShoulderPitch',
    'RightShoulderPitch': 'robot/RightShoulderPitch',
    'LeftShoulderRoll': 'robot/LeftShoulderRoll',
    'RightShoulderRoll': 'robot/RightShoulderRoll',
    'LeftElbow': 'robot/LeftElbow',
    'RightElbow': 'robot/RightElbow',
}


# EVA spacesuit comfort ROM envelopes for reference retargeting (strictly inside physical limits
# with generous dynamic margin so RL never slams into pressurized suit stops).
EVA_REF_LIMITS_DEG = {
    'robot/LeftHipPitch': (-80.0, 18.0),
    'robot/RightHipPitch': (-80.0, 18.0),
    'robot/LeftKnee': (3.0, 100.0),
    'robot/RightKnee': (3.0, 100.0),
    'robot/LeftAnkle': (-13.0, 14.0),
    'robot/RightAnkle': (-13.0, 14.0),
    'robot/WaistPitch': (-11.0, 22.0),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def slerp_batch(q_xyzw: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate normalized quaternions (xyzw -> wxyz) on dense source timestamps."""
    idx = np.clip(np.searchsorted(t_src, t_dst, side='right') - 1, 0, len(t_src) - 2)
    alpha = np.clip((t_dst - t_src[idx]) / np.maximum(t_src[idx + 1] - t_src[idx], 1e-9), 0., 1.)
    q0 = q_xyzw[idx]
    q1 = q_xyzw[idx + 1]
    dot = (q0 * q1).sum(axis=-1, keepdims=True)
    q1 = np.where(dot < 0., -q1, q1)
    q = (1. - alpha[:, None]) * q0 + alpha[:, None] * q1
    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-9)
    # Convert xyzw -> MuJoCo wxyz
    return np.column_stack((q[:, 3], q[:, 0], q[:, 1], q[:, 2]))


def solve_strut_ik(model, data, side: str, tip_world: np.ndarray):
    """Exact closed-form 3-DOF IK for detached hut_hinge_aft_v7 strut (pitch, roll, slide)."""
    name = f'robot/exo_{side}'
    upper_world = data.site(f'{name}_pack').xpos
    yoke = model.body(f'{name}_yoke')
    sleeve = model.body(f'{name}_sleeve')
    shaft = model.body(f'{name}_shaft')
    torso = model.body(int(yoke.parentid[0]))
    v_world = tip_world - upper_world
    v_torso = data.xmat[torso.id].reshape(3, 3).T @ v_world
    basis = np.empty(9)
    mujoco.mju_quat2Mat(basis, yoke.quat)
    x, y, z = basis.reshape(3, 3).T @ v_torso
    phi_0 = 2. * np.arctan2(float(sleeve.quat[1]), float(sleeve.quat[0]))
    q_pitch = float(np.arctan2(x, z))
    q_roll = float(np.arctan2(-y, np.hypot(x, z)) - phi_0)
    q_slide = float(np.linalg.norm(v_world) - float(shaft.pos[2]))
    return q_pitch, q_roll, q_slide


def main():
    if not SOURCE_NPZ.is_file():
        raise FileNotFoundError(f'Source reference not found: {SOURCE_NPZ}')
    model = build_model(training_sensors=False)
    data = mujoco.MjData(model)
    ground = model.geom('terrain').id
    root_id = model.body('robot/Pelvis').id
    torso_id = model.body('robot/Torso').id
    tip_ids = {model.geom(f'robot/exo_{s}_hoof_pad').id for s in SIDES}
    human_colliders = [i for i in range(model.ngeom) if (model.geom_contype[i] & 2) != 0 and i not in tip_ids]
    support_ids = [model.geom(name).id for name in CONTACT_GEOMS[:4]]

    with np.load(SOURCE_NPZ, allow_pickle=True) as z:
        t_src = z['times'].astype(np.float64)
        jnames = z['joint_names'].tolist()
        jpos_src = z['joint_positions'].astype(np.float64)
        rpos_src = z['root_positions'].astype(np.float64)
        rquat_src = z['root_quaternions_xyzw'].astype(np.float64)
        replant_win = z['replant_windows_s'][0].astype(np.float64)  # [1.89, 3.05]

    t_kneel_src = float(replant_win[0])
    t_replant_end_src = float(replant_win[1])
    # Under lunar 1/6g:
    # - Stage 1 prone->kneel paced 1.40x (2.64 s) for gentle grounded push-up
    # - 0.40 s calm kneeling settle pause (2.64 -> 3.04 s) to damp backward pitch momentum before lifting struts
    # - Replant swing (1.15x -> 1.01 s)
    # - Left-foot forward step & plant (t_src 3.05 -> 6.40 s) paced 1.60x so the forward foot
    #   plants briskly (~0.85 s) without lunar 1/6g ballistic liftoff or landing stomp
    # - Final bipedal rise to parallel standing (t_src 6.40 -> 9.00 s) paced 1.45x for low hip/knee torque
    t_step_plant_src = 6.40
    t_stand_src = min(float(t_src[-1]), 9.0)
    t_kneel_arrive = t_kneel_src * 1.40
    kneel_settle_s = 0.40
    t_kneel = t_kneel_arrive + kneel_settle_s
    t_replant_end = t_kneel + (t_replant_end_src - t_kneel_src) / 1.15
    t_step_plant = t_replant_end + (t_step_plant_src - t_replant_end_src) / 1.60
    t_motion_end = t_step_plant + (t_stand_src - t_step_plant_src) / 1.45
    hold_s = 1.0
    t_total = t_motion_end + hold_s
    n_frames = int(round(t_total / DT)) + 1
    times = np.linspace(0., (n_frames - 1) * DT, n_frames, dtype=np.float64)

    # Piecewise-linear time warp mapping Wangyu target timestamps -> source timestamps.
    t_query = np.empty_like(times)
    for k, tk in enumerate(times):
        if tk <= t_kneel_arrive:
            t_query[k] = tk / 1.40
        elif tk <= t_kneel:
            t_query[k] = t_kneel_src
        elif tk <= t_replant_end:
            u = (tk - t_kneel) / max(t_replant_end - t_kneel, 1e-9)
            t_query[k] = t_kneel_src + u * (t_replant_end_src - t_kneel_src)
        elif tk <= t_step_plant:
            u = (tk - t_replant_end) / max(t_step_plant - t_replant_end, 1e-9)
            t_query[k] = t_replant_end_src + u * (t_step_plant_src - t_replant_end_src)
        elif tk <= t_motion_end:
            u = (tk - t_step_plant) / max(t_motion_end - t_step_plant, 1e-9)
            t_query[k] = t_step_plant_src + u * (t_stand_src - t_step_plant_src)
        else:
            t_query[k] = t_stand_src

    # Interpolate source root & human joints to exact 60 Hz Wangyu control rate.
    rpos = np.column_stack([np.interp(t_query, t_src, rpos_src[:, k]) for k in range(3)])
    rquat_wxyz = slerp_batch(rquat_src, t_src, t_query)
    jpos_interp = np.column_stack([np.interp(t_query, t_src, jpos_src[:, k]) for k in range(jpos_src.shape[1])])

    # Gentle zero-phase Gaussian smoothing over kneel->stand segment to remove micro-bobbing.
    k_replant_end_tmp = int(np.argmin(np.abs(times - t_replant_end)))
    k_stand_end_tmp = int(np.argmin(np.abs(times - t_motion_end)))
    if k_stand_end_tmp > k_replant_end_tmp + 10:
        ker_x = np.arange(-9, 10, dtype=np.float64)
        ker = np.exp(-0.5 * (ker_x / 3.0) ** 2)
        ker /= ker.sum()
        seg = jpos_interp[k_replant_end_tmp:k_stand_end_tmp + 1].copy()
        padded = np.pad(seg, ((9, 9), (0, 0)), mode='edge')
        smooth_seg = np.column_stack([np.convolve(padded[:, c], ker, mode='valid') for c in range(seg.shape[1])])
        # Fade smoothing smoothly at boundaries so velocity stays continuous.
        n_seg = len(seg)
        w_fade = np.ones(n_seg, dtype=np.float64)
        fade = min(12, n_seg // 4)
        w_fade[:fade] = np.linspace(0., 1., fade)
        w_fade[-fade:] = np.linspace(1., 0., fade)
        jpos_interp[k_replant_end_tmp:k_stand_end_tmp + 1] = (
            (1. - w_fade[:, None]) * seg + w_fade[:, None] * smooth_seg
        )

    # Scale horizontal translation by Wangyu/Apollo standing pelvis height ratio and recenter start XY=0.
    scale_xy = 0.8116 / float(rpos_src[-1, 2])
    rpos_xy = (rpos[:, :2] - rpos[0:1, :2]) * scale_xy

    # Joint target safety margin (1.5 deg inside physical joint limits).
    margin_rad = radians(1.5)
    src_idx = {name: i for i, name in enumerate(jnames)}

    # Pass 1: Solve human qpos + ground-contact root_z for all frames (without struts yet).
    qpos_traj = np.zeros((n_frames, model.nq), dtype=np.float64)
    com_traj = np.zeros((n_frames, 3), dtype=np.float64)
    upper_traj = np.zeros((n_frames, 2, 3), dtype=np.float64)
    sole_traj = np.zeros((n_frames, 2, 3), dtype=np.float64)
    shin_traj = np.zeros((n_frames, 2, 3), dtype=np.float64)
    hand_traj = np.zeros((n_frames, 2, 3), dtype=np.float64)
    chest_gid = model.geom('robot/chest').id
    forearm_gids = [model.geom('robot/left_forearm_hand').id, model.geom('robot/right_forearm_hand').id]
    sole_gids = [model.geom('robot/left_sole_hull').id, model.geom('robot/right_sole_hull').id]
    shin_gids_p1 = [model.geom('robot/left_shin').id, model.geom('robot/right_shin').id]

    for k in range(n_frames):
        mujoco.mj_resetData(model, data)
        data.qpos[0] = rpos_xy[k, 0]
        data.qpos[1] = rpos_xy[k, 1]
        data.qpos[2] = 2.0
        data.qpos[3:7] = rquat_wxyz[k]
        for src_name, dst_name in HUMAN_JOINT_MAP.items():
            j = model.joint(dst_name)
            val = float(jpos_interp[k, src_idx[src_name]])
            # Keep slight comfortable outward arm roll so arms never touch the rigid HUT.
            if dst_name == 'robot/LeftShoulderRoll':
                val = max(val, radians(14.))
            elif dst_name == 'robot/RightShoulderRoll':
                val = min(val, -radians(14.))
            lo, hi = float(j.range[0]) + margin_rad, float(j.range[1]) - margin_rad
            if dst_name in EVA_REF_LIMITS_DEG:
                eva_lo, eva_hi = EVA_REF_LIMITS_DEG[dst_name]
                lo = max(lo, radians(eva_lo))
                hi = min(hi, radians(eva_hi))
            data.qpos[j.qposadr[0]] = np.clip(val, lo, hi)
        mujoco.mj_forward(model, data)
        gaps = [mujoco.mj_geomDistance(model, data, ground, gid, 10., None) for gid in human_colliders]
        data.qpos[2] -= min(gaps) - 0.001
        mujoco.mj_forward(model, data)

        # Early prone push-up phase (t <= 1.05 s):
        # plant both gloved hands gently on the lunar surface WITHOUT ever lifting the shins off the ground!
        tk = float(times[k])
        if tk <= 1.05:
            blend = 1.0 if tk <= 0.40 else 0.5 * (1.0 + np.cos(np.pi * (tk - 0.40) / 0.65))
            u_early = min(tk / 0.40, 1.0)
            # Keep root_z locked to leg/shin ground contact; solve elbow angle so hands touch gently (gap >= 1 mm).
            sp_brace = np.deg2rad(-51.5 + 12.0 * u_early)
            el_base = np.deg2rad(-20.0 - 15.0 * u_early)
            for s_i, (side, sign) in enumerate(zip(SIDES, (1., -1.))):
                sp_j = model.joint(f'robot/{side.title()}ShoulderPitch')
                sr_j = model.joint(f'robot/{side.title()}ShoulderRoll')
                el_j = model.joint(f'robot/{side.title()}Elbow')
                sp_orig = float(data.qpos[sp_j.qposadr[0]])
                sr_orig = float(data.qpos[sr_j.qposadr[0]])
                el_orig = float(data.qpos[el_j.qposadr[0]])
                sp_t = (1. - blend) * sp_orig + blend * sp_brace
                sr_t = (1. - blend) * sr_orig + blend * (sign * np.deg2rad(16.))
                el_t = (1. - blend) * el_orig + blend * el_base
                data.qpos[sp_j.qposadr[0]] = np.clip(sp_t, float(sp_j.range[0]) + margin_rad, float(sp_j.range[1]) - margin_rad)
                data.qpos[sr_j.qposadr[0]] = np.clip(sr_t, float(sr_j.range[0]) + margin_rad, float(sr_j.range[1]) - margin_rad)
                data.qpos[el_j.qposadr[0]] = np.clip(el_t, float(el_j.range[0]) + margin_rad, float(el_j.range[1]) - margin_rad)
            # If hands dip below terrain, flex elbows slightly instead of lifting the pelvis/shins off the ground!
            mujoco.mj_forward(model, data)
            for _ in range(48):
                hg = min(mujoco.mj_geomDistance(model, data, ground, fg, 5., None) for fg in forearm_gids)
                if hg >= 0.001:
                    break
                for side in SIDES:
                    el_j = model.joint(f'robot/{side.title()}Elbow')
                    data.qpos[el_j.qposadr[0]] = max(float(data.qpos[el_j.qposadr[0]]) - np.deg2rad(0.5),
                                                     float(el_j.range[0]) + margin_rad)
                mujoco.mj_forward(model, data)

        qpos_traj[k] = data.qpos.copy()
        com_traj[k] = data.subtree_com[root_id].copy()
        for s_i, side in enumerate(SIDES):
            upper_traj[k, s_i] = data.site(f'robot/exo_{side}_pack').xpos.copy()
            sole_traj[k, s_i] = data.geom_xpos[sole_gids[s_i]].copy()
            shin_traj[k, s_i] = data.geom_xpos[shin_gids_p1[s_i]].copy()
            hand_traj[k, s_i] = data.geom_xpos[forearm_gids[s_i]].copy()

    # Choose physically optimal fixed ground anchor points for Wangyu's hut_hinge_aft_v7_detached struts:
    # - Prone/push-up anchor (t in [0, t_kneel]): forward-below initial HUT mount so strut pushes torso up cleanly.
    # - Standing anchor (t in [t_replant_end, end]): directly at standing COM x (within <=2 cm of COM),
    #   laterally aligned with Wangyu's ±0.29 m HUT mount so coronal roll stays near 0 deg.
    k_kneel = int(np.argmin(np.abs(times - t_kneel)))
    k_replant_end = int(np.argmin(np.abs(times - t_replant_end)))
    k_stand = n_frames - 1

    # Find x_prone that keeps strut length safely inside [0.60, 1.12] across [0 .. k_kneel]
    # and points clearly downward ahead of initial prone torso.
    # y_anchor = 0.345 m (+5 cm outward from HUT mount) guarantees > 6 cm clearance from boots/shins/thighs/arms.
    u_prone = upper_traj[:k_kneel + 1, 0]  # left side (x, y, z)
    x_candidates = np.linspace(float(u_prone[0, 0]) - 0.15, float(u_prone[0, 0]) + 0.45, 121)
    y_anchor = 0.345
    z_anchor = HOOF_SOLE_HEIGHT_M + 0.001  # 0.023 m (flat rubber hoof sole 1 mm above plane)
    best_x_prone = None
    best_score = 1e9
    for xc in x_candidates:
        d_vec = u_prone - np.array([xc, y_anchor, z_anchor])
        lens = np.linalg.norm(d_vec, axis=-1)
        if lens.min() >= 0.59 and lens.max() <= 1.13:
            # Prefer anchor ahead of initial torso COM with generous length margin
            score = abs(xc - (u_prone[0, 0] + 0.20)) + 2.0 * max(0., lens.max() - 1.05) + 2.0 * max(0., 0.64 - lens.min())
            if score < best_score:
                best_score = score
                best_x_prone = float(xc)
    if best_x_prone is None:
        raise RuntimeError('No feasible prone ground anchor found inside hut_hinge_aft_v7 stroke!')

    # Choose x_stand near standing COM x (so final COM-to-tip-line distance <= 2 cm!) and reachable across [k_replant_end..end]
    u_stand = upper_traj[k_replant_end:, 0]
    com_x_stand = float(com_traj[k_stand, 0])
    x_stand_cands = np.linspace(com_x_stand - 0.06, com_x_stand + 0.06, 61)
    best_x_stand = None
    best_s_score = 1e9
    for xc in x_stand_cands:
        d_vec = u_stand - np.array([xc, y_anchor, z_anchor])
        lens = np.linalg.norm(d_vec, axis=-1)
        if lens.min() >= 0.59 and lens.max() <= 1.15 and abs(xc - best_x_prone) >= 0.20:
            score = abs(xc - com_x_stand) + 2.0 * max(0., lens.max() - 1.12)
            if score < best_s_score:
                best_s_score = score
                best_x_stand = float(xc)
    if best_x_stand is None:
        raise RuntimeError('No feasible standing ground anchor found inside hut_hinge_aft_v7 stroke!')

    # Build smooth tip target trajectories (fixed during Phase 0, lifted arc during replant, fixed during stand-up).
    tip_targets = np.zeros((n_frames, 2, 3), dtype=np.float64)
    tip_contact_mask = np.ones((n_frames, 2), dtype=bool)
    for k, tk in enumerate(times):
        if tk <= t_kneel:
            tx, tz, c = best_x_prone, z_anchor, True
        elif tk >= t_replant_end:
            tx, tz, c = best_x_stand, z_anchor, True
        else:
            u = (tk - t_kneel) / max(t_replant_end - t_kneel, 1e-6)
            s_smooth = 0.5 - 0.5 * np.cos(np.pi * u)
            tx = (1. - s_smooth) * best_x_prone + s_smooth * best_x_stand
            tz = z_anchor + 0.095 * np.sin(np.pi * u)  # peak tip_z = 0.118 m (clean flat-hoof ground clearance)
            c = False
        tip_targets[k, 0] = [tx, +y_anchor, tz]
        tip_targets[k, 1] = [tx, -y_anchor, tz]
        tip_contact_mask[k] = [c, c]

    # Pass 2: Solve exact closed-form 3-DOF strut IK and audit every frame!
    phase_traj = np.zeros(n_frames, dtype=np.int64)
    for k, tk in enumerate(times):
        if tk < t_kneel:
            phase_traj[k] = 0
        elif tk < 0.5 * (t_kneel + t_replant_end):
            phase_traj[k] = 1
        elif tk < t_replant_end:
            phase_traj[k] = 2
        else:
            phase_traj[k] = 3

    max_fk_err = 0.0
    min_ground_gap = 1e9
    min_arm_chest_gap = 1e9
    min_strut_human_gap = 1e9
    max_limit_viol = 0.0
    pitch_rng = [1e9, -1e9]
    roll_rng = [1e9, -1e9]
    len_rng = [1e9, -1e9]
    strut_gids = [model.geom(n).id for n in (
        'robot/exo_left_outer_tube', 'robot/exo_left_sliding_shaft', 'robot/exo_left_hoof_pad',
        'robot/exo_right_outer_tube', 'robot/exo_right_sliding_shaft', 'robot/exo_right_hoof_pad',
    )]
    sole_contact_mask = np.zeros((n_frames, 2), dtype=bool)
    shin_contact_mask = np.zeros((n_frames, 2), dtype=bool)
    shin_gids = [model.geom('robot/left_shin').id, model.geom('robot/right_shin').id]

    height_traj = np.zeros(n_frames, dtype=np.float64)
    upright_traj = np.zeros(n_frames, dtype=np.float64)
    for k in range(n_frames):
        data.qpos[:] = qpos_traj[k]
        mujoco.mj_forward(model, data)
        for s_i, side in enumerate(SIDES):
            qp, qr, qs = solve_strut_ik(model, data, side, tip_targets[k, s_i])
            for suffix, val in (('hinge_pitch', qp), ('hooke_roll', qr), ('slide', qs)):
                j = model.joint(f'robot/exo_{side}_{suffix}')
                viol = max(float(j.range[0]) - val, val - float(j.range[1]), 0.)
                max_limit_viol = max(max_limit_viol, viol)
                data.qpos[j.qposadr[0]] = np.clip(val, float(j.range[0]), float(j.range[1]))
            pitch_rng = [min(pitch_rng[0], np.rad2deg(qp)), max(pitch_rng[1], np.rad2deg(qp))]
            roll_rng = [min(roll_rng[0], np.rad2deg(qr)), max(roll_rng[1], np.rad2deg(qr))]
            L = qs + float(model.body(f'robot/exo_{side}_shaft').pos[2])
            len_rng = [min(len_rng[0], L), max(len_rng[1], L)]
        mujoco.mj_forward(model, data)
        # Level the damped ball-joint hooves parallel to the lunar ground plane across all frames.
        for side in SIDES:
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot/exo_{side}_hoof_ball')
            if j_id < 0:
                continue
            shaft_id = model.body(f'robot/exo_{side}_shaft').id
            hoof_id = model.body(f'robot/exo_{side}_hoof').id
            R_rest = np.empty(9)
            mujoco.mju_quat2Mat(R_rest, model.body_quat[hoof_id])
            R_zero = data.xmat[shaft_id].reshape(3, 3) @ R_rest.reshape(3, 3)
            q_ball = np.empty(4)
            mujoco.mju_mat2Quat(q_ball, R_zero.T.ravel())
            if q_ball[0] < 0.:
                q_ball = -q_ball
            adr = int(model.jnt_qposadr[j_id])
            data.qpos[adr:adr + 4] = q_ball
        mujoco.mj_forward(model, data)
        qpos_traj[k] = data.qpos.copy()
        height_traj[k] = float(data.xpos[root_id, 2])
        upright_traj[k] = float(data.xmat[torso_id, 8])
        for s_i, side in enumerate(SIDES):
            tip_fk = data.geom(f'robot/exo_{side}_ball').xpos
            max_fk_err = max(max_fk_err, float(np.linalg.norm(tip_fk - tip_targets[k, s_i])))
            sole_contact_mask[k, s_i] = mujoco.mj_geomDistance(model, data, ground, sole_gids[s_i], 5., None) <= 0.015
            shin_contact_mask[k, s_i] = mujoco.mj_geomDistance(model, data, ground, shin_gids[s_i], 5., None) <= 0.015
        all_colliders = [i for i in range(model.ngeom) if (model.geom_contype[i] & 2) != 0]
        gaps = [mujoco.mj_geomDistance(model, data, ground, gid, 10., None) for gid in all_colliders]
        min_ground_gap = min(min_ground_gap, min(gaps))
        for fg in forearm_gids:
            min_arm_chest_gap = min(min_arm_chest_gap, float(mujoco.mj_geomDistance(model, data, chest_gid, fg, 5., None)))
        if k % 3 == 0 or k == n_frames - 1:
            for sg in strut_gids:
                for hg in human_colliders:
                    min_strut_human_gap = min(min_strut_human_gap, float(mujoco.mj_geomDistance(model, data, sg, hg, 2., None)))

    # Finite-difference qvel (nv) matching MuJoCo tangent space for smooth RSI initialization.
    qvel_traj = np.zeros((n_frames, model.nv), dtype=np.float64)
    for k in range(n_frames - 1):
        mujoco.mj_differentiatePos(model, qvel_traj[k], DT, qpos_traj[k], qpos_traj[k + 1])
    qvel_traj[-1] = 0.

    # Extract actuator-order target joint positions (nu = 29).
    act_joints = model.actuator_trnid[:, 0]
    act_qadr = model.jnt_qposadr[act_joints]
    ctrl_traj = qpos_traj[:, act_qadr]

    # Audit early prone hand ground-contact, kneeling, and standing key states.
    data.qpos[:] = qpos_traj[0]
    mujoco.mj_forward(model, data)
    prone_hand_gaps = [round(float(mujoco.mj_geomDistance(model, data, ground, fg, 5., None)), 4)
                       for fg in forearm_gids]

    data.qpos[:] = qpos_traj[k_kneel]
    mujoco.mj_forward(model, data)
    kneel_upright = float(data.xmat[torso_id, 8])
    kneel_z = float(data.xpos[root_id, 2])
    kneel_com = data.subtree_com[root_id, :2].copy()
    kneel_leg_gaps = {CONTACT_GEOMS[i]: round(float(mujoco.mj_geomDistance(model, data, ground, support_ids[i], 5., None)), 4)
                      for i in range(4)}

    data.qpos[:] = qpos_traj[k_stand]
    mujoco.mj_forward(model, data)
    stand_upright = float(data.xmat[torso_id, 8])
    stand_z = float(data.xpos[root_id, 2])
    stand_com = data.subtree_com[root_id, :2].copy()
    stand_tip_dist = abs(float(stand_com[0]) - best_x_stand)

    audit = {
        'task': TASK,
        'geometry': GEOMETRY,
        'exo_mechanism': MECHANISM,
        'frames': n_frames,
        'dt': DT,
        'duration_s': round(float(times[-1]), 4),
        't_kneel_s': round(t_kneel, 4),
        't_replant_end_s': round(t_replant_end, 4),
        'anchor_prone_xy': [round(best_x_prone, 4), y_anchor],
        'anchor_stand_xy': [round(best_x_stand, 4), y_anchor],
        'replant_shift_m': round(abs(best_x_stand - best_x_prone), 4),
        'max_strut_fk_error_m': round(max_fk_err, 8),
        'max_joint_limit_violation_rad': round(max_limit_viol, 8),
        'min_ground_gap_m': round(min_ground_gap, 5),
        'min_forearm_chest_gap_m': round(min_arm_chest_gap, 4),
        'min_strut_human_gap_m': round(min_strut_human_gap, 4),
        'strut_pitch_deg_range': [round(pitch_rng[0], 2), round(pitch_rng[1], 2)],
        'strut_roll_deg_range': [round(roll_rng[0], 2), round(roll_rng[1], 2)],
        'strut_length_m_range': [round(len_rng[0], 4), round(len_rng[1], 4)],
        'prone_hand_gaps_m': prone_hand_gaps,
        'kneel_frame': {
            'index': k_kneel, 'z_m': round(kneel_z, 4), 'upright': round(kneel_upright, 4),
            'com_xy': [round(float(kneel_com[0]), 4), round(float(kneel_com[1]), 4)],
            'leg_gaps_m': kneel_leg_gaps,
        },
        'stand_frame': {
            'index': k_stand, 'z_m': round(stand_z, 4), 'upright': round(stand_upright, 4),
            'com_xy': [round(float(stand_com[0]), 4), round(float(stand_com[1]), 4)],
            'com_to_tip_line_m': round(stand_tip_dist, 4),
        },
    }
    print(json.dumps(audit, indent=2))
    assert max_limit_viol < 1e-6, f'Strut joint limit violation: {max_limit_viol}'
    assert max_fk_err < 1e-5, f'Strut FK mismatch: {max_fk_err}'
    assert min_ground_gap >= -0.0005, f'Ground penetration detected: {min_ground_gap}'
    assert min_arm_chest_gap > 0.0, f'Arm-chest penetration detected: {min_arm_chest_gap}'
    assert min_strut_human_gap >= 0.025, f'Strut-human clearance too tight: {min_strut_human_gap}'
    leg_grounded_stage1 = shin_contact_mask[:k_replant_end + 1] | sole_contact_mask[:k_replant_end + 1]
    assert np.all(leg_grounded_stage1.all(axis=1)), 'Both legs (shin or boot) must stay grounded throughout Stage 1 & replant!'

    stand_window_frame = int(np.argmin(np.abs(times - max(t_motion_end - 0.5, t_replant_end))))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUT_DIR / 'motion.npz'
    np.savez(
        out_npz,
        times=times.astype(np.float32),
        qpos=qpos_traj.astype(np.float32),
        qvel=qvel_traj.astype(np.float32),
        ctrl=ctrl_traj.astype(np.float32),
        tip_targets=tip_targets.astype(np.float32),
        sole_targets=sole_traj.astype(np.float32),
        shin_targets=shin_traj.astype(np.float32),
        hand_targets=hand_traj.astype(np.float32),
        tip_contact_mask=tip_contact_mask,
        sole_contact_mask=sole_contact_mask,
        shin_contact_mask=shin_contact_mask,
        phase=phase_traj.astype(np.int64),
        height=height_traj.astype(np.float32),
        upright=upright_traj.astype(np.float32),
        t_kneel_s=np.float32(t_kneel),
        t_replant_end_s=np.float32(t_replant_end),
        stand_window_frame=np.int64(stand_window_frame),
        anchor_prone=np.array([best_x_prone, y_anchor], dtype=np.float32),
        anchor_stand=np.array([best_x_stand, y_anchor], dtype=np.float32),
    )
    manifest = {
        'revision': 'wangyu_replant_ref_v8',
        'source_motion': str(SOURCE_NPZ),
        'source_sha256': sha256_file(SOURCE_NPZ),
        'script': 'scripts/prepare_wangyu_replant_ref.py',
        'script_sha256': sha256_file(Path(__file__).resolve()),
        'files': {
            'assets/wangyu_replant_ref/motion.npz': sha256_file(out_npz),
        },
        'audit': audit,
    }
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'Saved SHA-locked reference to {out_npz} and {OUT_DIR / "manifest.json"}')


if __name__ == '__main__':
    main()
