"""Wangyu lunar fall-recovery via clean DeepMimic motion tracking + small residual PD + EVA ROM & leg-torque polish."""
import hashlib
import json
from math import radians
from pathlib import Path

import mujoco
import numpy as np
from rsl_rl.env import VecEnv
from scipy.spatial import ConvexHull, QhullError
from tensordict import TensorDict
import torch

from exoskeleton import DETACHED_ROLL_LIMIT_RAD, HUT_HINGE_HOOF_MECHANISM
from moon_scene import WANGYU_EVA_PROFILE, scene_spec
from suit_proxy_geometry import FIT3D_18V

TASK = 'Wangyu-Replant-v12'
MECHANISM = 'hut_hinge_hoof_v8_detached'
GEOMETRY = FIT3D_18V
SIDES = ('left', 'right')
DT = 1 / 60
EPISODE_STEPS = 10 * 60
LOOKAHEAD_STEPS = 12  # +0.20 s future reference target in observation

# Direct 60 Hz small residual PD bounds around the SHA-locked reference (no sluggish EMA lag).
RESIDUAL_LEG_RAD = 0.35        # ±20 deg legs/waist balance & lunar dynamics correction
RESIDUAL_ARM_RAD = 0.20        # ±11.5 deg quiet EVA arm posture
RESIDUAL_STRUT_RAD = 0.25      # ±14 deg detached strut sagittal pitch
RESIDUAL_STRUT_ROLL_RAD = 0.10 # ±5.7 deg outward coronal roll
RESIDUAL_STRUT_SLIDE_M = 0.16  # ±16 cm axial contact force & push-up lift
STRUT_INWARD_ROLL_MAX_RAD = radians(2.0)

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / 'assets' / 'wangyu_replant_ref'
REF_NPZ = REF_DIR / 'motion.npz'
REF_MANIFEST = REF_DIR / 'manifest.json'

CONTACT_GEOMS = (
    'robot/left_sole_hull',
    'robot/right_sole_hull',
    'robot/left_shin',
    'robot/right_shin',
    'robot/exo_left_hoof_pad',
    'robot/exo_right_hoof_pad',
)

# 8 key end-effectors tracked in DeepMimic key-body reward (soles, shins, hands, strut ball centers).
KEY_BODY_GEOMS = (
    'robot/left_sole_hull',
    'robot/right_sole_hull',
    'robot/left_shin',
    'robot/right_shin',
    'robot/left_forearm_hand',
    'robot/right_forearm_hand',
    'robot/exo_left_ball',
    'robot/exo_right_ball',
)

ARM_ACTUATORS = (
    'robot/LeftShoulderPitch', 'robot/LeftShoulderRoll', 'robot/LeftShoulderYaw', 'robot/LeftElbow',
    'robot/RightShoulderPitch', 'robot/RightShoulderRoll', 'robot/RightShoulderYaw', 'robot/RightElbow',
)

# Lower-limb + waist sagittal joints targeted for peak-torque reduction & EVA comfort ROM barrier.
LOWER_LIMB_ACTUATORS = (
    'robot/LeftHipPitch', 'robot/RightHipPitch',
    'robot/LeftKnee', 'robot/RightKnee',
    'robot/LeftAnkle', 'robot/RightAnkle',
    'robot/WaistPitch',
)

# Pressurized EVA suit soft comfort ROM envelopes (strictly inside hard physical limits).
EVA_SOFT_ROM_LIMITS = {
    'robot/LeftHipPitch': (-radians(85.), radians(22.)),
    'robot/RightHipPitch': (-radians(85.), radians(22.)),
    'robot/LeftKnee': (radians(1.), radians(102.)),
    'robot/RightKnee': (radians(1.), radians(102.)),
    'robot/LeftAnkle': (-radians(17.), radians(15.)),
    'robot/RightAnkle': (-radians(17.), radians(15.)),
    'robot/WaistPitch': (-radians(12.), radians(25.)),
}

# Physical EVA suit knee flexion cap (realistic pressurized knee bellows <= 108 deg).
EVA_KNEE_HARD_LIMITS = (0.0, radians(108.))

EVA_ARM_JOINT_LIMITS = {
    'robot/LeftShoulderPitch': (-radians(105.), radians(45.)),
    'robot/LeftShoulderRoll': (-radians(5.), radians(75.)),
    'robot/LeftShoulderYaw': (-radians(45.), radians(45.)),
    'robot/LeftElbow': (-radians(115.), -radians(10.)),
    'robot/RightShoulderPitch': (-radians(105.), radians(45.)),
    'robot/RightShoulderRoll': (-radians(75.), radians(5.)),
    'robot/RightShoulderYaw': (-radians(45.), radians(45.)),
    'robot/RightElbow': (-radians(115.), -radians(10.)),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_replant_reference(ref_dir: Path | None = None) -> dict:
    """Load and SHA-verify the Wangyu retargeted get-up reference trajectory."""
    base = Path(ref_dir) if ref_dir is not None else REF_DIR
    manifest_path = base / 'manifest.json'
    npz_path = base / 'motion.npz'
    if not manifest_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            f'Missing Wangyu replant reference under {base}. '
            'Run scripts/prepare_wangyu_replant_ref.py to generate it.'
        )
    manifest = json.loads(manifest_path.read_text())
    expected_sha = manifest.get('files', {}).get('assets/wangyu_replant_ref/motion.npz')
    actual_sha = _sha256_file(npz_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f'SHA-256 mismatch for {npz_path}: expected {expected_sha}, got {actual_sha}'
        )
    with np.load(npz_path) as z:
        ref = {k: z[k].copy() for k in z.files}
    ref['sha256'] = actual_sha
    ref['revision'] = manifest.get('revision', 'unknown')
    ref['source_sha256'] = manifest.get('source_sha256', '')
    ref['manifest'] = manifest
    ref['n_frames'] = int(ref['qpos'].shape[0])
    ref['stand_window_frame'] = int(ref.get('stand_window_frame', max(ref['n_frames'] - 90, 0)))
    ref['key_targets'] = np.concatenate(
        (ref['sole_targets'], ref['shin_targets'], ref['hand_targets'], ref['tip_targets']),
        axis=1,
    ).astype(np.float32)
    return ref


def actuator_residual_scales(model) -> np.ndarray:
    """Per-actuator small residual PD bounds around the preset trajectory."""
    joints = model.actuator_trnid[:, 0]
    half_span = .5 * (model.jnt_range[joints, 1] - model.jnt_range[joints, 0])
    arm_set = set(ARM_ACTUATORS)
    scales = np.empty(model.nu, dtype=np.float64)
    for i in range(model.nu):
        name = model.actuator(i).name
        if name in ('robot/exo_left', 'robot/exo_right'):
            cap = RESIDUAL_STRUT_SLIDE_M
        elif name in ('robot/exo_left_roll', 'robot/exo_right_roll'):
            cap = RESIDUAL_STRUT_ROLL_RAD
        elif name.startswith('robot/exo_'):
            cap = RESIDUAL_STRUT_RAD
        elif name in arm_set:
            cap = RESIDUAL_ARM_RAD
        else:
            cap = RESIDUAL_LEG_RAD
        scales[i] = min(float(half_span[i]), cap)
    return scales


def build_spec(*, training_sensors=False, knee_hard_limit=True):
    """Build Wangyu lunar EVA + fit3d_18v + detached hut_hinge_hoof_v8 damped ball-joint hooves with EVA ROM & anti-penetration."""
    spec = scene_spec(
        geometry=GEOMETRY,
        mass_profile=WANGYU_EVA_PROFILE,
        suit_ankle_roll=True,
        suit_ankle_pitch=True,
        suit_major_joints=True,
        relax_propulsion=True,
        exoskeleton=True,
        exo_mechanism=HUT_HINGE_HOOF_MECHANISM,
        exo_detached=True,
    )
    for jname, limits in EVA_ARM_JOINT_LIMITS.items():
        joint = spec.joint(jname)
        joint.range = list(limits)
        act = spec.actuator(jname)
        act.ctrlrange = list(limits)

    if knee_hard_limit:
        for knee_name in ('robot/LeftKnee', 'robot/RightKnee'):
            spec.joint(knee_name).range = list(EVA_KNEE_HARD_LIMITS)
            spec.actuator(knee_name).ctrlrange = list(EVA_KNEE_HARD_LIMITS)

    roll_limits = {
        'left': (-DETACHED_ROLL_LIMIT_RAD, STRUT_INWARD_ROLL_MAX_RAD),
        'right': (-STRUT_INWARD_ROLL_MAX_RAD, DETACHED_ROLL_LIMIT_RAD),
    }
    for side, lim in roll_limits.items():
        spec.joint(f'robot/exo_{side}_hooke_roll').range = list(lim)
        spec.actuator(f'robot/exo_{side}_roll').ctrlrange = list(lim)

    ground = spec.geom('terrain')
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.hfieldname = ''
    ground.pos = [0, 0, 0]
    ground.size = [8, 8, .1]
    ground.contype, ground.conaffinity = 1, 2
    for hfield in list(spec.hfields):
        spec.delete(hfield)

    # Asymmetric collision bitmasks (zero unwanted limb-limb self-contact, real strut/hoof-human & arm-torso contact).
    # On the hoof body, ONLY the bottom rubber box (exo_{side}_hoof_pad) participates in physics collision.
    skip_collide = {'robot/left_sole', 'robot/right_sole'}
    tip_geoms = {f'robot/exo_{side}_hoof_pad' for side in SIDES}
    strut_body_geoms = {
        f'robot/exo_{side}_{part}'
        for side in SIDES for part in ('outer_tube', 'sliding_shaft')
    }
    torso_core = {'robot/chest', 'robot/backpack', 'robot/pelvis', 'robot/helmet'}
    arm_collide = {'robot/left_forearm_hand', 'robot/right_forearm_hand'}
    for geom in spec.geoms:
        if not geom.name.startswith('robot/'):
            continue
        if geom.name in tip_geoms:
            geom.contype, geom.conaffinity = 10, 1
        elif geom.name in strut_body_geoms:
            geom.contype, geom.conaffinity = 8, 0
        elif geom.name.startswith('robot/exo_') or geom.name in skip_collide:
            geom.contype, geom.conaffinity = 0, 0
        elif geom.name in torso_core:
            geom.contype, geom.conaffinity = 6, 9
        elif geom.name in arm_collide:
            geom.contype, geom.conaffinity = 2, 13
        else:
            geom.contype, geom.conaffinity = 2, 9

    if training_sensors:
        for sensor in list(spec.sensors):
            spec.delete(sensor)
        for geom in CONTACT_GEOMS:
            spec.add_sensor(
                name=f'replant/{geom}',
                type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_GEOM,
                objname=geom,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,
                refname='terrain',
                intprm=[19, 2, 4],
            )
    spec.add_sensor(
        name='replant/angmom',
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname='robot/Pelvis',
    )
    return spec


def build_model(*, training_sensors=False):
    return build_spec(training_sensors=training_sensors).compile()


def quat_rot_error_np(q_cur: np.ndarray, q_ref: np.ndarray):
    """Signed 3D axis-angle rotation error from q_cur to q_ref (wxyz) and scalar angle in [0, pi]."""
    w0, x0, y0, z0 = q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]
    w1, x1, y1, z1 = q_ref[0], q_ref[1], q_ref[2], q_ref[3]
    ew = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    ex = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    ey = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    ez = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    if ew < 0.:
        ew, ex, ey, ez = -ew, -ex, -ey, -ez
    vec = np.array([ex, ey, ez], dtype=np.float64)
    sin_half = float(np.linalg.norm(vec))
    angle = 2.0 * float(np.arctan2(sin_half, max(float(ew), 1e-9)))
    rotvec = (angle / sin_half) * vec if sin_half > 1e-7 else 2.0 * vec
    return rotvec.astype(np.float32), angle


def mimic_reward_np(q_act, q_ref, qd_act, qd_ref,
                    root_pos, root_pos_ref, root_rot_angle, root_vel, root_vel_ref,
                    key_pos, key_pos_ref, load_bw, tip_contact_exp,
                    act_rate_sq, leg_tau_rms_sq, leg_tau_peak_sq, rom_excess_sq,
                    hoof_slip_sq=0.0):
    """Canonical strictly-positive DeepMimic tracking reward in [0, 1] + leg-torque, EVA-ROM & hoof-slip polish."""
    pose_err = float(np.mean((q_act - q_ref) ** 2))
    r_pose = np.exp(-pose_err / 0.04)

    vel_err = float(np.mean((qd_act - qd_ref) ** 2))
    r_vel = np.exp(-vel_err / 2.0)

    rpos_err = float(np.sum((root_pos - root_pos_ref) ** 2))
    rvel_err = float(np.sum((root_vel[:3] - root_vel_ref[:3]) ** 2)
                     + 0.25 * np.sum((root_vel[3:6] - root_vel_ref[3:6]) ** 2))
    r_root = np.exp(-(rpos_err / 0.04 + (root_rot_angle ** 2) / 0.08 + rvel_err / 1.5))

    key_err = float(np.mean((key_pos - key_pos_ref) ** 2))
    r_key = np.exp(-key_err / 0.012)

    leg_grounded = float(load_bw[:4].max() >= 0.02)
    tip_bw = load_bw[4:]
    if tip_contact_exp:
        tip_match = 0.5 * (np.clip(tip_bw[0] / 0.06, 0., 1.) + np.clip(tip_bw[1] / 0.06, 0., 1.))
    else:
        tip_match = float(np.exp(-(tip_bw[0] + tip_bw[1]) / 0.05))
    r_contact = 0.65 * leg_grounded + 0.35 * tip_match

    r_track = 0.35 * r_pose + 0.10 * r_vel + 0.25 * r_root + 0.22 * r_key + 0.08 * r_contact
    # Bounded biomechanical polish: smooth action, lower hip/knee/ankle RMS & peak torque,
    # stay inside EVA ROM, and penalize horizontal hoof skidding while loaded on ground.
    reg = min(0.03 * act_rate_sq + 0.06 * leg_tau_rms_sq + 0.06 * leg_tau_peak_sq
              + 0.40 * rom_excess_sq + 0.08 * hoof_slip_sq, 0.18)
    return float(max(r_track - reg, 0.0))


def adapt_reference_to_model(model, ref: dict) -> dict:
    """Adapt a legacy 36-nq (hut_hinge_aft_v7) reference to the 44-nq damped ball-joint hoof model without mutating archives."""
    if int(ref['qpos'].shape[1]) == int(model.nq) and int(ref['qvel'].shape[1]) == int(model.nv):
        return ref
    hoof_qadrs = sorted(
        int(model.joint(f'robot/exo_{s}_hoof_ball').qposadr[0])
        for s in SIDES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot/exo_{s}_hoof_ball') >= 0
    )
    hoof_dofadrs = sorted(
        int(model.joint(f'robot/exo_{s}_hoof_ball').dofadr[0])
        for s in SIDES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot/exo_{s}_hoof_ball') >= 0
    )
    if not hoof_qadrs or int(ref['qpos'].shape[1]) + 4 * len(hoof_qadrs) != int(model.nq):
        raise ValueError(f'Reference qpos dim {ref["qpos"].shape[1]} incompatible with model nq={model.nq}')
    keep_q = np.ones(model.nq, dtype=bool)
    for a in hoof_qadrs:
        keep_q[a:a + 4] = False
    keep_v = np.ones(model.nv, dtype=bool)
    for a in hoof_dofadrs:
        keep_v[a:a + 3] = False
    n = int(ref['qpos'].shape[0])
    qpos = np.zeros((n, model.nq), dtype=ref['qpos'].dtype)
    qvel = np.zeros((n, model.nv), dtype=ref['qvel'].dtype)
    qpos[:, keep_q] = ref['qpos']
    qvel[:, keep_v] = ref['qvel']
    # Set ground-level hoof orientation at each frame so hooves start cleanly parallel to ground plane.
    d = mujoco.MjData(model)
    for a in hoof_qadrs:
        qpos[:, a] = 1.0
    for k in range(n):
        d.qpos[:] = qpos[k]
        mujoco.mj_kinematics(model, d)
        for s in SIDES:
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'robot/exo_{s}_hoof_ball')
            if j_id < 0:
                continue
            shaft_id = model.body(f'robot/exo_{s}_shaft').id
            hoof_id = model.body(f'robot/exo_{s}_hoof').id
            R_rest = np.empty(9)
            mujoco.mju_quat2Mat(R_rest, model.body_quat[hoof_id])
            R_zero = d.xmat[shaft_id].reshape(3, 3) @ R_rest.reshape(3, 3)
            q_ball = np.empty(4)
            mujoco.mju_mat2Quat(q_ball, R_zero.T.ravel())
            if q_ball[0] < 0.:
                q_ball = -q_ball
            adr = int(model.jnt_qposadr[j_id])
            qpos[k, adr:adr + 4] = q_ball
    adapted = dict(ref)
    adapted['qpos'] = qpos
    adapted['qvel'] = qvel
    return adapted


class Recovery:
    def __init__(self, model, reference=None, *, terminate_on_ref_div=False):
        self.m = model
        self.d = mujoco.MjData(model)
        raw_ref = reference if reference is not None else load_replant_reference()
        self.ref = adapt_reference_to_model(model, raw_ref)
        self.terminate_on_ref_div = bool(terminate_on_ref_div)
        self.n_ref = int(self.ref['n_frames'])
        self.root = model.body('robot/Pelvis').id
        self.torso = model.body('robot/Torso').id
        self.ground = model.geom('terrain').id
        self.support = [model.geom(name).id for name in CONTACT_GEOMS[:4]]
        self.tip_contacts = [model.geom(name).id for name in CONTACT_GEOMS[4:]]
        self.tips = [model.geom(f'robot/exo_{side}_ball').id for side in SIDES]
        self.contacts = self.support + self.tip_contacts
        self.key_geoms = [model.geom(name).id for name in KEY_BODY_GEOMS]
        self.angmom_adr = int(model.sensor('replant/angmom').adr[0])
        joints = model.actuator_trnid[:, 0]
        self.addr = model.jnt_qposadr[joints]
        self.dof_addr = model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints].copy()
        self.residual_scale = actuator_residual_scales(model)
        self.tau_max = np.maximum(np.abs(model.actuator_forcerange[:, 1]), 1.0)
        self.lower_aids = np.array([model.actuator(n).id for n in LOWER_LIMB_ACTUATORS], dtype=np.int64)
        self.lower_qadr = np.array([int(model.joint(n).qposadr[0]) for n in LOWER_LIMB_ACTUATORS], dtype=np.int64)
        self.eva_soft_lo = np.array([EVA_SOFT_ROM_LIMITS[n][0] for n in LOWER_LIMB_ACTUATORS], dtype=np.float64)
        self.eva_soft_hi = np.array([EVA_SOFT_ROM_LIMITS[n][1] for n in LOWER_LIMB_ACTUATORS], dtype=np.float64)
        self.weight = model.body_subtreemass[self.root] * 1.635
        self.anchor_prone_pair = np.array([
            [float(self.ref['anchor_prone'][0]), +float(self.ref['anchor_prone'][1])],
            [float(self.ref['anchor_prone'][0]), -float(self.ref['anchor_prone'][1])],
        ], dtype=np.float64)
        self.anchor_stand_pair = np.array([
            [float(self.ref['anchor_stand'][0]), +float(self.ref['anchor_stand'][1])],
            [float(self.ref['anchor_stand'][0]), -float(self.ref['anchor_stand'][1])],
        ], dtype=np.float64)
        self.reset(start_frame=0)

    def reset(self, start_frame=0):
        m, d = self.m, self.d
        k = int(np.clip(start_frame, 0, self.n_ref - 1))
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.ref['qpos'][k]
        d.qvel[:] = 0. if k == 0 else self.ref['qvel'][k]
        d.ctrl[:] = self.ref['ctrl'][k]
        mujoco.mj_forward(m, d)
        self.start_frame = k
        self.ref_idx = k
        self.from_prone = (k == 0)
        self.phase = int(self.ref['phase'][k])
        self.hold = 0
        self.steps = 0
        if self.phase == 0:
            self.anchor = d.geom_xpos[self.tips, :2].copy()
        elif self.phase in (1, 2):
            self.anchor = self.anchor_prone_pair.copy()
        else:
            self.anchor = self.anchor_stand_pair.copy()
        self.load = np.zeros(6)
        self.prev_action = np.zeros(m.nu, dtype=np.float32)
        self.prev_hoof_xy = d.geom_xpos[self.tip_contacts, :2].copy()
        self.hoof_slip_mps = 0.0

    def upright(self):
        return float(self.d.xmat[self.torso, 8])

    def angmom(self):
        return self.d.sensordata[self.angmom_adr:self.angmom_adr + 3]

    def observe(self):
        d = self.d
        k_next = min(self.ref_idx + 1, self.n_ref - 1)
        k_look = min(self.ref_idx + LOOKAHEAD_STEPS, self.n_ref - 1)
        q_act = d.qpos[self.addr]
        qd_act = d.qvel[self.dof_addr]
        rpos_err = self.ref['qpos'][k_next, :3] - d.qpos[:3]
        rrot_err, _ = quat_rot_error_np(d.qpos[3:7], self.ref['qpos'][k_next, 3:7])
        rvel_err = self.ref['qvel'][k_next, :6] - d.qvel[:6]
        q_err_now = self.ref['ctrl'][k_next] - q_act
        q_err_fut = self.ref['ctrl'][k_look] - q_act
        rpos_fut = self.ref['qpos'][k_look, :3] - d.qpos[:3]
        rrot_fut, _ = quat_rot_error_np(d.qpos[3:7], self.ref['qpos'][k_look, 3:7])
        tips = d.geom_xpos[self.tips]
        tip_err = (self.ref['tip_targets'][k_next] - tips).ravel()
        ref_progress = self.ref_idx / max(self.n_ref - 1, 1)
        return np.r_[
            d.qpos[2:7],
            d.qvel[:6],
            q_act,
            qd_act,
            self.load / self.weight,
            self.prev_action,
            rpos_err, rrot_err, rvel_err,
            rpos_fut, rrot_fut,
            q_err_now,
            q_err_fut,
            tip_err,
            ref_progress,
        ].astype(np.float32)

    def step(self, action):
        m, d = self.m, self.d
        lo, hi = self.limits.T
        clipped = np.clip(action, -1., 1.).astype(np.float32)
        next_ref = min(self.ref_idx + 1, self.n_ref - 1)
        d.ctrl[:] = np.clip(self.ref['ctrl'][next_ref] + clipped * self.residual_scale, lo, hi)
        mujoco.mj_step(m, d, nstep=2)
        mujoco.mj_forward(m, d)
        self.steps += 1
        self.ref_idx = next_ref
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            return 0., False, True

        self.load.fill(0)
        points = []
        wrench = np.empty(6)
        for k, c in enumerate(d.contact):
            if self.ground not in c.geom:
                continue
            other = c.geom[1] if c.geom[0] == self.ground else c.geom[0]
            if other in self.contacts:
                i = self.contacts.index(other)
                mujoco.mj_contactForce(m, d, k, wrench)
                self.load[i] += wrench[0]
                if i < 4 and wrench[0] > 1:
                    points.append(c.pos[:2].copy())

        tips = d.geom_xpos[self.tips]
        soles = d.geom_xpos[self.support[:2]]
        key_pos = d.geom_xpos[self.key_geoms]
        com = d.subtree_com[self.root, :2]
        span = tips[1, :2] - tips[0, :2]
        nearest = tips[0, :2] + span * np.clip((com - tips[0, :2]) @ span / max(span @ span, 1e-9), 0, 1)
        tip_line_dist = float(np.linalg.norm(com - nearest))
        near = tip_line_dist <= .15
        loaded = min(self.load[4:]) >= 10.
        upright = self.upright()
        height = float(d.xpos[self.root, 2])
        speed = float(np.linalg.norm(d.qvel[:6]))
        foot_mid = .5 * (soles[0, :2] + soles[1, :2])
        shin_mid = .5 * (d.geom_xpos[self.support[2], :2] + d.geom_xpos[self.support[3], :2])
        leg_xy_dist = float(np.linalg.norm(com - foot_mid))
        shin_xy_dist = float(np.linalg.norm(com - shin_mid))
        tip_shift = float(min(np.linalg.norm(tips[:, :2] - self.anchor, axis=1)))
        both_feet = min(self.load[:2]) >= 10.
        both_shins = min(self.load[2:4]) >= 10.
        sole_dx = abs(float(soles[0, 0] - soles[1, 0]))
        phase = self.phase
        if phase == 0:
            if loaded:
                self.anchor[:] = tips[:, :2]
            leg_ready = (both_feet and leg_xy_dist <= .25) or (both_shins and shin_xy_dist <= .22)
            if not leg_ready and np.count_nonzero(self.load[:4] > 5.) >= 3 and len(points) >= 3:
                try:
                    planes = ConvexHull(points).equations
                    leg_ready = bool(np.all(planes[:, :2] @ com + planes[:, 2] <= .001))
                except QhullError:
                    leg_ready = False
            stable = (upright > .72 and height > .36
                      and self.load[:4].sum() > .55 * self.weight
                      and speed < .45 and leg_ready)
            self.hold = self.hold + 1 if stable else 0
            if self.hold >= 10:
                if loaded:
                    self.anchor[:] = tips[:, :2]
                self.phase = 1
        elif phase == 1:
            if loaded:
                self.anchor[:] = tips[:, :2]
            if max(self.load[4:]) < 1. and min(tips[:, 2]) > .06:
                self.phase = 2
        elif phase == 2:
            if loaded and near and tip_shift >= .12:
                self.phase = 3
        else:
            standing = (loaded and near
                        and min(self.load[:2]) >= .20 * self.weight
                        and self.load[:2].sum() >= .65 * self.weight
                        and abs(self.load[0] - self.load[1]) <= .30 * self.weight
                        and sole_dx <= .12
                        and max(self.load[2:4]) < 5.
                        and height > .74 and upright > .94 and speed < .30)
            self.hold = self.hold + 1 if standing else 0
        if self.phase != phase:
            self.hold = 0
        success = self.phase == 3 and self.hold >= 60

        ref_k = self.ref_idx
        load_bw = self.load / self.weight
        _, root_rot_angle = quat_rot_error_np(d.qpos[3:7], self.ref['qpos'][ref_k, 3:7])
        act_rate_sq = float(np.mean((clipped - self.prev_action) ** 2))
        self.prev_action[:] = clipped
        leg_tau_norm_sq = (d.actuator_force[self.lower_aids] / self.tau_max[self.lower_aids]) ** 2
        leg_tau_rms_sq = float(np.mean(leg_tau_norm_sq))
        leg_tau_peak_sq = float(np.max(leg_tau_norm_sq))
        q_lower = d.qpos[self.lower_qadr]
        rom_exc = np.maximum(q_lower - self.eva_soft_hi, 0.) + np.maximum(self.eva_soft_lo - q_lower, 0.)
        rom_excess_sq = float(np.mean(rom_exc ** 2))
        hoof_xy = d.geom_xpos[self.tip_contacts, :2]
        hoof_speed = np.linalg.norm(hoof_xy - self.prev_hoof_xy, axis=-1) / DT
        self.prev_hoof_xy[:] = hoof_xy
        hoof_gate = np.clip(load_bw[4:] / 0.03, 0., 1.)
        self.hoof_slip_mps = float(np.mean(hoof_gate * hoof_speed))
        hoof_slip_sq = float(np.mean(hoof_gate * (hoof_speed / 0.15) ** 2))
        reward = mimic_reward_np(
            d.qpos[self.addr], self.ref['ctrl'][ref_k],
            d.qvel[self.dof_addr], self.ref['qvel'][ref_k, self.dof_addr],
            d.qpos[:3], self.ref['qpos'][ref_k, :3],
            root_rot_angle, d.qvel[:6], self.ref['qvel'][ref_k, :6],
            key_pos, self.ref['key_targets'][ref_k],
            load_bw, bool(self.ref['tip_contact_mask'][ref_k, 0]),
            act_rate_sq, leg_tau_rms_sq, leg_tau_peak_sq, rom_excess_sq,
            hoof_slip_sq,
        )

        root_pos_err = float(np.linalg.norm(d.qpos[:3] - self.ref['qpos'][ref_k, :3]))
        key_err_mean = float(np.mean(np.linalg.norm(key_pos - self.ref['key_targets'][ref_k], axis=-1)))
        after_grace = self.steps >= 15
        if self.terminate_on_ref_div:
            failed = after_grace and (root_pos_err > .35 or root_rot_angle > .75 or key_err_mean > .30)
        else:
            failed = float(np.linalg.norm(d.qpos[:2])) > 3. or (after_grace and upright < -.20)
        return float(reward), success, failed


class ReplantEnv(VecEnv):
    device = 'cpu'
    max_episode_length = EPISODE_STEPS

    def __init__(self, model, count, *, reference=None, rsi_prob=0.0, terminate_on_ref_div=False):
        self.ref = reference if reference is not None else load_replant_reference()
        self.rsi_prob = float(rsi_prob)
        self.terminate_on_ref_div = bool(terminate_on_ref_div)
        self.num_actions, self.num_envs = model.nu, count
        self.worlds = [
            Recovery(model, reference=self.ref, terminate_on_ref_div=self.terminate_on_ref_div)
            for _ in range(count)
        ]
        self.episode_length_buf = torch.zeros(count, dtype=torch.long)
        self.cfg = dict(
            task=TASK, geometry=GEOMETRY, exo_mechanism=MECHANISM,
            reference_revision=self.ref['revision'],
            reference_sha256=self.ref['sha256'],
            rsi_prob=self.rsi_prob,
            residual_action=True,
            lookahead_steps=LOOKAHEAD_STEPS,
            residual_leg_rad=RESIDUAL_LEG_RAD,
            residual_arm_rad=RESIDUAL_ARM_RAD,
            residual_strut_rad=RESIDUAL_STRUT_RAD,
            residual_strut_roll_rad=RESIDUAL_STRUT_ROLL_RAD,
            residual_strut_slide_m=RESIDUAL_STRUT_SLIDE_M,
            terminate_on_ref_div=self.terminate_on_ref_div,
        )

    def get_observations(self):
        obs = torch.from_numpy(np.stack([w.observe() for w in self.worlds]))
        return TensorDict({'policy': obs}, batch_size=[self.num_envs])

    def step(self, actions):
        rewards, dones, timeouts, successes, prone_successes = [], [], [], [], []
        max_rsi = max(1, self.ref['n_frames'] - 30)
        for w, a in zip(self.worlds, actions.detach().cpu().numpy()):
            was_prone = w.from_prone
            r, success, failed = w.step(a)
            timeout = w.steps >= self.max_episode_length and not failed
            done = failed or timeout
            rewards.append(r)
            dones.append(done)
            timeouts.append(timeout)
            successes.append(success)
            if was_prone and done:
                prone_successes.append(float(success))
            if done:
                start_k = int(np.random.randint(1, max_rsi)) if (self.rsi_prob > 0 and np.random.random() < self.rsi_prob) else 0
                w.reset(start_frame=start_k)
        self.episode_length_buf[:] = torch.tensor([w.steps for w in self.worlds])
        log = {
            'Replant/phase': float(np.mean([w.phase for w in self.worlds])),
            'Replant/max_phase': float(np.max([w.phase for w in self.worlds])),
            'Replant/upright': float(np.mean([w.upright() for w in self.worlds])),
            'Replant/height': float(np.mean([w.d.xpos[w.root, 2] for w in self.worlds])),
            'Replant/sole_load_bw': float(np.mean([w.load[:2].sum() / w.weight for w in self.worlds])),
            'Replant/shin_load_bw': float(np.mean([w.load[2:4].sum() / w.weight for w in self.worlds])),
            'Replant/tip_load_bw': float(np.mean([w.load[4:].sum() / w.weight for w in self.worlds])),
            'Replant/hoof_slip_mps': float(np.mean([w.hoof_slip_mps for w in self.worlds])),
            'Replant/angmom_nms': float(np.mean([np.linalg.norm(w.angmom()) for w in self.worlds])),
            'Replant/success_rate': float(np.mean(successes)),
            'Replant/prone_success_rate': float(np.mean(prone_successes)) if prone_successes else 0.,
        }
        return (self.get_observations(), torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones), {'time_outs': torch.tensor(timeouts), 'log': log})
