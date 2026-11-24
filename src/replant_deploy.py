"""Deterministic physical prelude: prone Exoskeleton (magnetically docked) -> undock -> forward deploy -> RL handoff.

Not part of RL training. Runs purely via closed-form 3-DOF strut IK + smooth minimum-jerk
abduction sweep with guaranteed non-penetration against the astronaut body and lunar ground,
then hands the live MjData seamlessly over to the trained replant policy at the front anchor.
"""
from __future__ import annotations

import mujoco
import numpy as np

from exo_mode_switch import solve_strut_ik_to_world_point
from exoskeleton import (MAG_LED_DETACHED_COLOR, MAG_LED_DOCKED_COLOR,
                         STRUTS, fit_struts_to_pose)
from wangyu_replant import DT, SIDES

DOCK_HOLD_S = 0.40       # Stage A0: lie prone with hooves magnetically welded to boots (cyan LED)
UNDOCK_CLEAR_S = 0.35    # Stage A1: release magnetic weld (amber LED) & abduct clear of boot sides
SWEEP_FORWARD_S = 1.15   # Stage A2: elevated outward sweep past hips/arms above ground to front hover
PLANT_DESCENT_S = 0.45   # Stage A3: vertical descent onto front push-up anchor & settle load
DEPLOY_TOTAL_S = DOCK_HOLD_S + UNDOCK_CLEAR_S + SWEEP_FORWARD_S + PLANT_DESCENT_S
DEPLOY_STEPS = int(round(DEPLOY_TOTAL_S / DT))


def _min_jerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _quad_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, s: float) -> np.ndarray:
    a = 1.0 - s
    return (a * a) * p0 + (2.0 * a * s) * p1 + (s * s) * p2


def build_deploy_waypoints(model: mujoco.MjModel, ref: dict) -> dict[str, dict[str, np.ndarray]]:
    """Compute world-space non-penetrating deploy waypoints from docked prone boot to front anchor."""
    d = mujoco.MjData(model)
    d.qpos[:] = ref['qpos'][0]
    fit_struts_to_pose(model, d)
    mujoco.mj_forward(model, d)
    waypoints: dict[str, dict[str, np.ndarray]] = {}
    for s_i, side in enumerate(SIDES):
        sign = 1.0 if side == 'left' else -1.0
        p_pack = d.site(f'robot/exo_{side}_pack').xpos.copy()
        p_boot = d.site(f'robot/exo_{side}_boot').xpos.copy()
        p_front = ref['tip_targets'][0, s_i].astype(np.float64).copy()
        # 1) Clear outward from boot magnetic dock plate (+14 cm lateral, +3 cm upward).
        p_clear = p_boot + np.array([0.04, sign * 0.14, 0.03], dtype=np.float64)
        # 2) Elevated outward apex passing alongside pelvis/torso at roll ~35 deg, L ~0.58 m:
        #    y ~ ±0.61 m (>20 cm from torso/arms) and z_ball ~0.095 m (>+6 cm above lunar plane).
        p_apex = np.array([
            p_pack[0] + 0.04,
            p_pack[1] + sign * 0.34,
            max(0.095, p_pack[2] - 0.455),
        ], dtype=np.float64)
        # 3) Hover cleanly above front push-up anchor before vertical touchdown.
        p_hover = p_front + np.array([0.0, sign * 0.02, 0.075], dtype=np.float64)
        waypoints[side] = {
            'pack': p_pack,
            'boot': p_boot,
            'clear': p_clear,
            'apex': p_apex,
            'hover': p_hover,
            'front': p_front,
        }
    return waypoints


def evaluate_deploy_target(wp: dict[str, np.ndarray], t: float) -> tuple[np.ndarray, bool]:
    """Return (target_ball_world_xyz, is_welded) at prelude time t in [0, DEPLOY_TOTAL_S]."""
    if t < DOCK_HOLD_S:
        return wp['boot'].copy(), True
    t1 = t - DOCK_HOLD_S
    if t1 < UNDOCK_CLEAR_S:
        s = _min_jerk(t1 / UNDOCK_CLEAR_S)
        return (1.0 - s) * wp['boot'] + s * wp['clear'], False
    t2 = t1 - UNDOCK_CLEAR_S
    if t2 < SWEEP_FORWARD_S:
        s = _min_jerk(t2 / SWEEP_FORWARD_S)
        return _quad_bezier(wp['clear'], wp['apex'], wp['hover'], s), False
    t3 = t2 - SWEEP_FORWARD_S
    s = _min_jerk(t3 / max(PLANT_DESCENT_S, 1e-6))
    return (1.0 - s) * wp['hover'] + s * wp['front'], False


class ProneDeployPrelude:
    """Runs deterministic prone-docked -> magnetic release -> forward deploy directly in MuJoCo physics."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ref: dict):
        self.model = model
        self.data = data
        self.ref = ref
        self.waypoints = build_deploy_waypoints(model, ref)
        self.human_aids = np.array([
            i for i in range(model.nu) if not model.actuator(i).name.startswith('robot/exo_')
        ], dtype=np.int64)
        self.human_ctrl0 = ref['ctrl'][0, self.human_aids].astype(np.float64).copy()
        self.strut_handles = {}
        for side in STRUTS:
            name = f'robot/exo_{side}'
            self.strut_handles[side] = {
                'eq': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f'{name}_mag_dock'),
                'led': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{name}_mag_led'),
                'act_hinge': model.actuator(f'{name}_hinge').id,
                'act_roll': model.actuator(f'{name}_roll').id,
                'act_slide': model.actuator(name).id,
                'jnt_hinge': model.joint(f'{name}_hinge_pitch').id,
                'jnt_roll': model.joint(f'{name}_hooke_roll').id,
                'jnt_slide': model.joint(f'{name}_slide').id,
            }
        self.step_idx = 0
        self.done = False
        self.stage = 'docked'
        self.reset()

    def _set_led(self, welded: bool):
        color = MAG_LED_DOCKED_COLOR if welded else MAG_LED_DETACHED_COLOR
        for side in STRUTS:
            gid = self.strut_handles[side]['led']
            if gid >= 0:
                self.model.geom_rgba[gid] = color

    def reset(self):
        """Initialize astronaut prone in Exoskeleton Mode (hooves magnetically welded to boots)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.ref['qpos'][0]
        d.qvel[:] = 0.0
        # Fit struts + damped ball-joint hooves flush onto both boot magnetic dock plates.
        fit_struts_to_pose(m, d)
        for side in STRUTS:
            eq_id = self.strut_handles[side]['eq']
            if eq_id >= 0:
                d.eq_active[eq_id] = 1
        d.ctrl[self.human_aids] = self.human_ctrl0
        for side in STRUTS:
            h = self.strut_handles[side]
            d.ctrl[h['act_hinge']] = float(d.qpos[m.jnt_qposadr[h['jnt_hinge']]])
            d.ctrl[h['act_roll']] = float(d.qpos[m.jnt_qposadr[h['jnt_roll']]])
            d.ctrl[h['act_slide']] = float(d.qpos[m.jnt_qposadr[h['jnt_slide']]])
        self._set_led(True)
        mujoco.mj_forward(m, d)
        self.step_idx = 0
        self.done = False
        self.stage = 'docked'

    def step(self) -> bool:
        """Advance one 60 Hz physics control step (2 substeps). Returns True when ready for RL handoff."""
        if self.done:
            return True
        m, d = self.model, self.data
        t = (self.step_idx + 1) * DT
        d.ctrl[self.human_aids] = self.human_ctrl0

        any_welded = False
        for side in STRUTS:
            h = self.strut_handles[side]
            target_xyz, welded = evaluate_deploy_target(self.waypoints[side], t)
            any_welded = any_welded or welded
            if h['eq'] >= 0:
                d.eq_active[h['eq']] = 1 if welded else 0
            if welded:
                d.ctrl[h['act_hinge']] = float(d.qpos[m.jnt_qposadr[h['jnt_hinge']]])
                d.ctrl[h['act_roll']] = float(d.qpos[m.jnt_qposadr[h['jnt_roll']]])
                d.ctrl[h['act_slide']] = float(d.qpos[m.jnt_qposadr[h['jnt_slide']]])
            else:
                qp, qr, qs = solve_strut_ik_to_world_point(m, d, side, target_xyz)
                d.ctrl[h['act_hinge']] = qp
                d.ctrl[h['act_roll']] = qr
                d.ctrl[h['act_slide']] = qs
        self._set_led(any_welded)

        mujoco.mj_step(m, d, nstep=2)
        mujoco.mj_forward(m, d)
        self.step_idx += 1
        if t < DOCK_HOLD_S:
            self.stage = 'docked'
        elif t < DOCK_HOLD_S + UNDOCK_CLEAR_S:
            self.stage = 'undock'
        elif t < DOCK_HOLD_S + UNDOCK_CLEAR_S + SWEEP_FORWARD_S:
            self.stage = 'sweep'
        else:
            self.stage = 'plant'
        if self.step_idx >= DEPLOY_STEPS:
            self.done = True
            self.stage = 'ready'
        return self.done

    def kinematic_frame(self, step_idx: int) -> bool:
        """Set pure kinematic pose for preview mode (returns welded flag)."""
        m, d = self.model, self.data
        t = step_idx * DT
        d.qpos[:] = self.ref['qpos'][0]
        d.qvel[:] = 0.0
        d.time = t
        if t < DOCK_HOLD_S:
            fit_struts_to_pose(m, d)
            self._set_led(True)
            mujoco.mj_forward(m, d)
            return True
        mujoco.mj_forward(m, d)
        for side in STRUTS:
            h = self.strut_handles[side]
            target_xyz, _ = evaluate_deploy_target(self.waypoints[side], t)
            qp, qr, qs = solve_strut_ik_to_world_point(m, d, side, target_xyz)
            d.qpos[m.jnt_qposadr[h['jnt_hinge']]] = qp
            d.qpos[m.jnt_qposadr[h['jnt_roll']]] = qr
            d.qpos[m.jnt_qposadr[h['jnt_slide']]] = qs
        mujoco.mj_forward(m, d)
        # Level hooves parallel to ground during forward sweep/plant in preview.
        for side in STRUTS:
            j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'robot/exo_{side}_hoof_ball')
            if j_id >= 0:
                shaft_id = m.body(f'robot/exo_{side}_shaft').id
                hoof_id = m.body(f'robot/exo_{side}_hoof').id
                R_rest = np.empty(9)
                mujoco.mju_quat2Mat(R_rest, m.body_quat[hoof_id])
                R_zero = d.xmat[shaft_id].reshape(3, 3) @ R_rest.reshape(3, 3)
                q_ball = np.empty(4)
                mujoco.mju_mat2Quat(q_ball, R_zero.T.ravel())
                if q_ball[0] < 0.:
                    q_ball = -q_ball
                adr = int(m.jnt_qposadr[j_id])
                d.qpos[adr:adr + 4] = q_ball
        self._set_led(False)
        mujoco.mj_forward(m, d)
        return False


def handoff_to_recovery(world):
    """Align Recovery bookkeeping to live post-deploy MjData without resetting physics state."""
    d = world.d
    world.start_frame = 0
    world.ref_idx = 0
    world.from_prone = True
    world.phase = 0
    world.hold = 0
    world.steps = 0
    world.anchor = d.geom_xpos[world.tips, :2].copy()
    world.prev_hoof_xy = d.geom_xpos[world.tip_contacts, :2].copy()
    world.prev_action[:] = 0.0
    world.hoof_slip_mps = 0.0
    # Populate current contact loads from the landed hooves & hands/shins so obs[0] is exact.
    world.load.fill(0.0)
    wrench = np.empty(6)
    for k in range(int(d.ncon)):
        c = d.contact[k]
        if world.ground not in c.geom:
            continue
        other = c.geom[1] if c.geom[0] == world.ground else c.geom[0]
        if other in world.contacts:
            i = world.contacts.index(other)
            mujoco.mj_contactForce(world.m, d, k, wrench)
            world.load[i] += wrench[0]
