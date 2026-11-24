"""Runtime magnetic-dock mode switching between Exoskeleton (assist) and Exolimb (ground-bracing) modes.

Operates on a single compiled ``hut_hinge_hoof_v8`` MuJoCo model without recompiling:
  - EXO_ASSIST     : electromagnetic mjEQ_WELD latched flush onto the boot outer sole;
                     damped spherical ball joint articulates freely with ankle/foot motion;
                     coronal roll passive, slide & sagittal hinge in torque/assist mode.
  - UNDOCKING      : releases mjEQ_WELD, activates 3-DOF strut PD, abducts hoof cleanly
                     outward (~6 deg) to clear the boot magnetic dock plate.
  - EXOLIMB_BRACE  : detached 3-DOF strut positions the damped ball-joint cloven hoof on the
                     lunar surface; high-friction flat-bottom hoof self-aligns to terrain.
  - DOCKING        : real-time 3-DOF IK steers hoof back to the live boot magnetic dock;
                     inside the capture zone (< 4.5 cm), short-range electromagnetic dipole
                     wrench pulls hoof flush and snaps mjEQ_WELD back into EXO_ASSIST.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import radians

import mujoco
import numpy as np

from exoskeleton import (DETACHED_HINGE_MAX_RAD, DETACHED_HINGE_MIN_RAD,
                         DETACHED_HINGE_TORQUE_NM, DETACHED_ROLL_LIMIT_RAD,
                         DETACHED_ROLL_TORQUE_NM, MAG_LED_DETACHED_COLOR,
                         MAG_LED_DOCKED_COLOR, MAX_FORCE_N, MAX_HINGE_TORQUE_NM,
                         STRUTS, mount_names)

MAG_CAPTURE_DIST_M = 0.045       # magnetic pull-in capture distance (4.5 cm)
MAG_LATCH_DIST_M = 0.016         # hard weld snap distance (1.6 cm)
MAG_LATCH_ANGLE_RAD = radians(28.)
MAG_PULL_FORCE_N = 90.0          # peak short-range electromagnetic pull force (N)
MAG_PULL_TORQUE_NM = 6.0         # short-range magnetic alignment torque (N·m)
UNDOCK_CLEAR_DIST_M = 0.055      # lateral clearance before transitioning to EXOLIMB_BRACE
UNDOCK_ABDUCT_RAD = radians(7.5) # outward roll clearance during undocking


class ExoMode(str, Enum):
    EXO_ASSIST = 'exo_assist'
    UNDOCKING = 'undocking'
    EXOLIMB_BRACE = 'exolimb_brace'
    DOCKING = 'docking'


@dataclass(frozen=True)
class DockStatus:
    side: str
    mode: ExoMode
    welded: bool
    dock_distance_m: float
    dock_angle_deg: float
    hoof_ground_gap_m: float


def _rot_angle_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    tr = float(np.trace(R1.T @ R2))
    return float(np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0)))


def _rot_err_vec(R_cur: np.ndarray, R_dst: np.ndarray) -> np.ndarray:
    """World-frame angular rotation vector carrying R_cur toward R_dst."""
    R_rel = R_dst @ R_cur.T
    skew = 0.5 * np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ], dtype=np.float64)
    sin_a = float(np.linalg.norm(skew))
    cos_a = float(np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0))
    angle = float(np.arctan2(sin_a, cos_a))
    if sin_a < 1e-8:
        return skew
    return (angle / sin_a) * skew


def solve_strut_ik_to_world_point(model: mujoco.MjModel, data: mujoco.MjData,
                                  side: str, target_world: np.ndarray) -> tuple[float, float, float]:
    """Exact closed-form 3-DOF IK (hinge_pitch, hooke_roll, slide) placing shaft ball center at target_world."""
    name = f'robot/exo_{side}'
    upper_name = mount_names(side)[0]
    yoke = model.body(f'{name}_yoke')
    torso = model.body(int(yoke.parentid[0]))
    vec_world = np.asarray(target_world, dtype=np.float64) - data.site(upper_name).xpos
    v_torso = data.xmat[torso.id].reshape(3, 3).T @ vec_world
    basis = np.empty(9)
    mujoco.mju_quat2Mat(basis, yoke.quat)
    x, y, z = basis.reshape(3, 3).T @ v_torso
    sleeve = model.body(f'{name}_sleeve')
    phi_0 = 2.0 * float(np.arctan2(float(sleeve.quat[1]), float(sleeve.quat[0])))
    pitch = float(np.arctan2(x, z))
    roll = float(np.arctan2(-y, np.hypot(x, z)) - phi_0)
    slide = float(np.linalg.norm(v_torso) - float(model.body(f'{name}_shaft').pos[2]))
    p_lo, p_hi = model.joint(f'{name}_hinge_pitch').range
    r_lo, r_hi = model.joint(f'{name}_hooke_roll').range
    s_lo, s_hi = model.joint(f'{name}_slide').range
    return (
        float(np.clip(pitch, p_lo, p_hi)),
        float(np.clip(roll, r_lo, r_hi)),
        float(np.clip(slide, s_lo, s_hi)),
    )


class MagneticHoofController:
    """Dual-mode runtime controller for ``hut_hinge_hoof_v8`` (Exoskeleton <-> Exolimb)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._handles: dict[str, dict[str, int]] = {}
        self._modes: dict[str, ExoMode] = {}
        self._transition_steps: dict[str, int] = {}
        for side in STRUTS:
            name = f'robot/exo_{side}'
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f'{name}_mag_dock')
            if eq_id < 0:
                raise ValueError(f'Model does not contain hut_hinge_hoof_v8 magnetic dock {name}_mag_dock')
            self._handles[side] = {
                'sign': 1 if side == 'left' else -1,
                'eq': eq_id,
                'hoof_body': model.body(f'{name}_hoof').id,
                'foot_body': model.body(f'robot/{side.title()}Foot').id,
                'hoof_dock': model.site(f'{name}_hoof_dock').id,
                'boot_dock': model.site(f'{name}_boot_dock').id,
                'boot_ball': model.site(mount_names(side)[1]).id,
                'hoof_sole': model.site(f'{name}_hoof_sole_site').id,
                'hoof_pad': model.geom(f'{name}_hoof_pad').id,
                'led_geom': model.geom(f'{name}_mag_led').id,
                'act_slide': model.actuator(name).id,
                'act_hinge': model.actuator(f'{name}_hinge').id,
                'act_roll': model.actuator(f'{name}_roll').id,
                'jnt_slide': model.joint(f'{name}_slide').id,
                'jnt_hinge': model.joint(f'{name}_hinge_pitch').id,
                'jnt_roll': model.joint(f'{name}_hooke_roll').id,
                'slide_range_lo': float(model.joint(f'{name}_slide').range[0]),
                'slide_range_hi': float(model.joint(f'{name}_slide').range[1]),
            }
            welded = bool(data.eq_active[eq_id])
            self._modes[side] = ExoMode.EXO_ASSIST if welded else ExoMode.EXOLIMB_BRACE
            self._transition_steps[side] = 0
            self._apply_actuator_profile(side, exolimb=not welded)
            self._update_led(side)

    def mode(self, side: str) -> ExoMode:
        return self._modes[side]

    @property
    def overall_mode(self) -> ExoMode:
        modes = {self._modes[s] for s in STRUTS}
        if len(modes) == 1:
            return next(iter(modes))
        if ExoMode.UNDOCKING in modes:
            return ExoMode.UNDOCKING
        if ExoMode.DOCKING in modes:
            return ExoMode.DOCKING
        return ExoMode.EXOLIMB_BRACE

    def _apply_actuator_profile(self, side: str, *, exolimb: bool):
        """Switch strut actuators & joint damping between Exo (torque/passive roll) and Exolimb (3-DOF PD)."""
        m = self.model
        h = self._handles[side]
        a_s, a_h, a_r = h['act_slide'], h['act_hinge'], h['act_roll']
        d_s = int(m.jnt_dofadr[h['jnt_slide']])
        d_h = int(m.jnt_dofadr[h['jnt_hinge']])
        d_r = int(m.jnt_dofadr[h['jnt_roll']])
        if exolimb:
            # 3-DOF critically-damped position-PD bracing control
            for aid, kp, kv, cr, fr in (
                (a_s, 800., 90., (h['slide_range_lo'], h['slide_range_hi']), (-450., 450.)),
                (a_h, 140., 22., (DETACHED_HINGE_MIN_RAD, DETACHED_HINGE_MAX_RAD),
                 (-DETACHED_HINGE_TORQUE_NM, DETACHED_HINGE_TORQUE_NM)),
                (a_r, 100., 16., (-DETACHED_ROLL_LIMIT_RAD, DETACHED_ROLL_LIMIT_RAD),
                 (-DETACHED_ROLL_TORQUE_NM, DETACHED_ROLL_TORQUE_NM)),
            ):
                m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
                m.actuator_gainprm[aid, :] = 0.
                m.actuator_gainprm[aid, 0] = kp
                m.actuator_biasprm[aid, :] = 0.
                m.actuator_biasprm[aid, 1] = -kp
                m.actuator_biasprm[aid, 2] = -kv
                m.actuator_ctrlrange[aid] = cr
                m.actuator_forcerange[aid] = fr
            m.dof_damping[d_s] = 2.0
            m.dof_damping[d_h] = 2.0
            m.dof_damping[d_r] = 1.5
            m.geom_contype[h['hoof_pad']] = 2
            m.geom_conaffinity[h['hoof_pad']] = 1
        else:
            # Exoskeleton assist mode: direct force/torque on slide & sagittal hinge, passive roll
            for aid, cr, fr in (
                (a_s, (0., MAX_FORCE_N), (0., MAX_FORCE_N)),
                (a_h, (-MAX_HINGE_TORQUE_NM, MAX_HINGE_TORQUE_NM), (-MAX_HINGE_TORQUE_NM, MAX_HINGE_TORQUE_NM)),
            ):
                m.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
                m.actuator_gainprm[aid, :] = 0.
                m.actuator_gainprm[aid, 0] = 1.0
                m.actuator_biasprm[aid, :] = 0.
                m.actuator_ctrlrange[aid] = cr
                m.actuator_forcerange[aid] = fr
            m.actuator_biastype[a_r] = mujoco.mjtBias.mjBIAS_NONE
            m.actuator_gainprm[a_r, :] = 0.
            m.actuator_biasprm[a_r, :] = 0.
            m.dof_damping[d_s] = 0.20
            m.dof_damping[d_h] = 0.30
            m.dof_damping[d_r] = 0.15
            m.geom_contype[h['hoof_pad']] = 0
            m.geom_conaffinity[h['hoof_pad']] = 0
            self.data.ctrl[[a_s, a_h, a_r]] = 0.

    def _update_led(self, side: str):
        h = self._handles[side]
        welded = bool(self.data.eq_active[h['eq']])
        self.model.geom_rgba[h['led_geom']] = (
            MAG_LED_DOCKED_COLOR if welded else MAG_LED_DETACHED_COLOR
        )

    def request_exolimb(self, sides: tuple[str, ...] = STRUTS):
        """Release magnetic boot lock and transition smoothly into Exolimb ground-bracing mode."""
        for side in sides:
            h = self._handles[side]
            self.data.eq_active[h['eq']] = 0
            self._apply_actuator_profile(side, exolimb=True)
            self._modes[side] = ExoMode.UNDOCKING
            self._transition_steps[side] = 0
            # Hold current angles as initial PD target so release has zero torque jerk.
            self.data.ctrl[h['act_hinge']] = float(self.data.qpos[self.model.jnt_qposadr[h['jnt_hinge']]])
            self.data.ctrl[h['act_roll']] = float(self.data.qpos[self.model.jnt_qposadr[h['jnt_roll']]])
            self.data.ctrl[h['act_slide']] = float(self.data.qpos[self.model.jnt_qposadr[h['jnt_slide']]])
            self._update_led(side)

    def request_exoskeleton(self, sides: tuple[str, ...] = STRUTS):
        """Steer detached hoof back to boot outer sole and magnetically dock into Exoskeleton mode."""
        for side in sides:
            h = self._handles[side]
            if bool(self.data.eq_active[h['eq']]):
                self._modes[side] = ExoMode.EXO_ASSIST
                continue
            self._apply_actuator_profile(side, exolimb=True)
            self._modes[side] = ExoMode.DOCKING
            self._transition_steps[side] = 0
            self._update_led(side)

    def toggle_mode(self):
        """Toggle between Exoskeleton (docked) and Exolimb (detached) modes."""
        if self.overall_mode in (ExoMode.EXO_ASSIST, ExoMode.DOCKING):
            self.request_exolimb()
        else:
            self.request_exoskeleton()
        return self.overall_mode

    def status(self, side: str) -> DockStatus:
        h = self._handles[side]
        d = self.data
        p_h = d.site_xpos[h['hoof_dock']]
        p_b = d.site_xpos[h['boot_dock']]
        R_h = d.site_xmat[h['hoof_dock']].reshape(3, 3)
        R_b = d.site_xmat[h['boot_dock']].reshape(3, 3)
        dist = float(np.linalg.norm(p_h - p_b))
        ang_deg = float(np.degrees(_rot_angle_rad(R_h, R_b)))
        sole_z = float(d.site_xpos[h['hoof_sole']][2])
        return DockStatus(
            side=side,
            mode=self._modes[side],
            welded=bool(d.eq_active[h['eq']]),
            dock_distance_m=dist,
            dock_angle_deg=ang_deg,
            hoof_ground_gap_m=sole_z,
        )

    def step_control(self, *, exo_slide_force: dict[str, float] | None = None,
                     exo_hinge_torque: dict[str, float] | None = None,
                     brace_targets_world: dict[str, np.ndarray] | None = None):
        """Advance magnetic docking / undocking state machine and write strut controls + magnetic wrench."""
        m, d = self.model, self.data
        for side in STRUTS:
            h = self._handles[side]
            sign = h['sign']
            d.xfrc_applied[h['hoof_body'], :] = 0.
            mode = self._modes[side]

            if mode == ExoMode.EXO_ASSIST:
                d.eq_active[h['eq']] = 1
                d.ctrl[h['act_slide']] = float((exo_slide_force or {}).get(side, 0.))
                d.ctrl[h['act_hinge']] = float((exo_hinge_torque or {}).get(side, 0.))
                d.ctrl[h['act_roll']] = 0.

            elif mode == ExoMode.UNDOCKING:
                self._transition_steps[side] += 1
                # Steer outward + slightly upward from boot dock so hoof clears magnetic bezel cleanly.
                boot_ball = d.site_xpos[h['boot_ball']].copy()
                R_foot = d.site_xmat[h['boot_dock']].reshape(3, 3)
                clear_target = boot_ball + R_foot @ np.array([0., sign * 0.085, 0.045])
                p_cmd, r_cmd, s_cmd = solve_strut_ik_to_world_point(m, d, side, clear_target)
                d.ctrl[h['act_hinge']] = p_cmd
                d.ctrl[h['act_roll']] = r_cmd
                d.ctrl[h['act_slide']] = s_cmd
                dist = float(np.linalg.norm(d.site_xpos[h['hoof_dock']] - d.site_xpos[h['boot_dock']]))
                if dist >= UNDOCK_CLEAR_DIST_M or self._transition_steps[side] >= 45:
                    self._modes[side] = ExoMode.EXOLIMB_BRACE

            elif mode == ExoMode.EXOLIMB_BRACE:
                if brace_targets_world is not None and side in brace_targets_world:
                    p_cmd, r_cmd, s_cmd = solve_strut_ik_to_world_point(m, d, side, brace_targets_world[side])
                    d.ctrl[h['act_hinge']] = p_cmd
                    d.ctrl[h['act_roll']] = r_cmd
                    d.ctrl[h['act_slide']] = s_cmd

            elif mode == ExoMode.DOCKING:
                self._transition_steps[side] += 1
                p_h = d.site_xpos[h['hoof_dock']]
                p_b = d.site_xpos[h['boot_dock']]
                R_h = d.site_xmat[h['hoof_dock']].reshape(3, 3)
                R_b = d.site_xmat[h['boot_dock']].reshape(3, 3)
                err_vec = p_b - p_h
                dist = float(np.linalg.norm(err_vec))
                ang = _rot_angle_rad(R_h, R_b)

                # Approach via staged lateral standoff -> direct flush contact so hoof doesn't clip sole edge.
                boot_ball = d.site_xpos[h['boot_ball']].copy()
                standoff = 0.030 if dist > 0.065 else 0.0
                target_ball = boot_ball + R_b @ np.array([0., sign * standoff, 0.])
                p_cmd, r_cmd, s_cmd = solve_strut_ik_to_world_point(m, d, side, target_ball)
                d.ctrl[h['act_hinge']] = p_cmd
                d.ctrl[h['act_roll']] = r_cmd
                d.ctrl[h['act_slide']] = s_cmd

                # Short-range inverse-distance electromagnetic pull inside capture zone.
                if dist < MAG_CAPTURE_DIST_M:
                    strength = 1.0 - dist / MAG_CAPTURE_DIST_M
                    f_pull = (MAG_PULL_FORCE_N * strength) * (err_vec / max(dist, 1e-6))
                    t_align = (MAG_PULL_TORQUE_NM * (0.4 + 0.6 * strength)) * _rot_err_vec(R_h, R_b)
                    d.xfrc_applied[h['hoof_body'], :3] = f_pull
                    d.xfrc_applied[h['hoof_body'], 3:] = t_align

                if dist <= MAG_LATCH_DIST_M and ang <= MAG_LATCH_ANGLE_RAD:
                    d.xfrc_applied[h['hoof_body'], :] = 0.
                    d.eq_active[h['eq']] = 1
                    self._apply_actuator_profile(side, exolimb=False)
                    self._modes[side] = ExoMode.EXO_ASSIST
                    self._update_led(side)
