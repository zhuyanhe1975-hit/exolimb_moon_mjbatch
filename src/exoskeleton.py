"""Two powered telescoping struts with pinned backpack and boot mounts."""
from math import expm1, isfinite, radians

import mujoco
import numpy as np
import torch

MAX_FORCE_N = 500.
DEFAULT_FORCE_N = MAX_FORCE_N
MAX_HINGE_TORQUE_NM = 200.
DEFAULT_HINGE_TORQUE_NM = 100.
HINGE_MIN_ANGLE_RAD = -radians(85.)
HINGE_MAX_ANGLE_RAD = radians(75.)
DETACHED_HINGE_MIN_RAD = -radians(125.)
DETACHED_HINGE_MAX_RAD = radians(35.)
DETACHED_ROLL_LIMIT_RAD = radians(40.)
DETACHED_HINGE_TORQUE_NM = 150.
DETACHED_ROLL_TORQUE_NM = 80.
DETACHED_TIP_RADIUS_M = .032
# Damped ball-joint magnetic cloven hoof end-effector (dual-mode exo <-> exolimb).
# Smooth elastomer centering (stiffness + damping, limited=False) avoids MuJoCo Warp's
# extra _limit_ball kernel while keeping midair hoof posture calm and self-aligning on ground.
HOOF_BALL_DAMPING = 1.2       # viscous elastomer spherical joint damping (N·m·s/rad)
HOOF_BALL_STIFFNESS = 2.0     # compliant elastomeric centering cuff (easy lunar self-alignment)
HOOF_BALL_ARMATURE = .002     # numerical stability for light hoof body (kg·m^2)
HOOF_SOLE_HEIGHT_M = .022     # distance from ball-joint center to flat cleated hoof sole (m)
HOOF_DOCK_OFFSET_M = .0245    # distance from ball-joint center to medial magnetic plate (m)
HOOF_DOCK_Z_ABOVE_SOLE_M = .026  # ball-joint center height above boot sole bottom when docked (+4 mm hoof clearance)
HOOF_FRICTION = (1.35, .03, .005)  # high-grip anti-slip lunar hoof + torsional yaw traction (condim=4)
HOOF_PAD_COLOR = [.12, .13, .15, 1.]
MAG_LED_DOCKED_COLOR = [.10, .82, .98, 1.]
MAG_LED_DETACHED_COLOR = [.96, .52, .12, 1.]
STRUTS = ('left', 'right')
STRUT_ACTIONS = len(STRUTS)
HINGE_ACTIONS = len(STRUTS)
STRUT_OBSERVATIONS = 5 * len(STRUTS)
STRUT_EXTRA_OBS = 3  # terrain_slope + left/right strut forward pitch angle
HINGE_OBSERVATIONS = 4 * len(STRUTS)  # per side: hinge_pos, hinge_vel/5, hinge_torque/100, last_hinge_request
MOTOR_POLICY_IO = 'motor_force_v1'
FORCE_POLICY_IO = 'motor_force_v2'
POLICY_TIMED_FORCE_IO = 'policy_timed_force_v1'
ACTION_FORCE_IO = 'action_force_v1'
IMPEDANCE_FORCE_IO = 'impedance_force_v1'
IMPEDANCE_HINGE_FORCE_IO = 'impedance_hinge_force_v1'
HINGE_CONTROL_REVISIONS = (IMPEDANCE_HINGE_FORCE_IO,)
DIRECT_FORCE_REVISIONS = (ACTION_FORCE_IO, IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO)
FORCE_CONTROL_REVISIONS = (FORCE_POLICY_IO, POLICY_TIMED_FORCE_IO, ACTION_FORCE_IO,
                           IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO)
PHASE_FREE_FORCE_REVISIONS = (POLICY_TIMED_FORCE_IO, ACTION_FORCE_IO,
                              IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO)
POLICY_IO = 'position_pd_v1'
INITIAL_ASSIST_N = 80.
PD_KP = 4000.  # N/m
PD_KD = 60.  # N·s/m
PD_MAX_ADVANCE_M = .15
IMPEDANCE_K_N_PER_M = 2000.  # virtual SLIP spring stiffness per strut
IMPEDANCE_C_NS_PER_M = 240.  # virtual landing compression damping per strut
IMPEDANCE_F_BASE_N = 450.    # direct grounded push thrust per strut at u=1
ACTION_FORCE_COSTS = {
    'revision': 'unloaded_force_and_ankle_moment_v1',
    'unloaded_strut_force': 2.,
    'unsupported_ankle_moment': 3.,
}
ANKLE_ROLL_MOMENT_LIMIT_NM = 8.
ANKLE_ROLL_GUARD_START = radians(3.)
ANKLE_ROLL_GUARD_STOP = radians(6.)
GUARD_CONTACTS_PER_FOOT = 12
SUPPORT_MOMENT_FRACTION = .8


def validate_assist_force(force):
    if not isfinite(force) or not 0. <= force <= MAX_FORCE_N:
        raise ValueError(f'每根支杆的助力须在 0–{MAX_FORCE_N:g} N 内')
    return float(force)


def validate_hinge_torque(torque):
    if not isfinite(torque) or not 0. <= torque <= MAX_HINGE_TORQUE_NM:
        raise ValueError(f'每侧矢状面铰链力矩须在 0–{MAX_HINGE_TORQUE_NM:g} N·m 内')
    return float(torque)


def validate_policy_io(revision):
    if revision not in (None, MOTOR_POLICY_IO, POLICY_IO, *FORCE_CONTROL_REVISIONS):
        raise ValueError(f'未知的外骨骼控制版本：{revision}')
    return revision


def exo_initial_state(source, target, *, initial_command=None, replace_strut_io=False,
                      append_hinge_io=False):
    """Warm-start a new actuator mode without interpreting old rod commands."""
    expanded = dict(source)
    for key, fill in (('mlp.0.weight', 0.), ('obs_normalizer._mean', 0.),
                      ('obs_normalizer._var', 1.), ('obs_normalizer._std', 1.)):
        old, template = source[key], target[key]
        if append_hinge_io and old.shape[:-1] == template.shape[:-1] and (
                template.shape[-1] == old.shape[-1] + HINGE_OBSERVATIONS):
            value = torch.full_like(template, fill)
            value[..., :old.shape[-1]] = old
            expanded[key] = value
            continue
        if replace_strut_io and old.shape[:-1] == template.shape[:-1] and template.shape[-1] in (
                old.shape[-1],
                old.shape[-1] + STRUT_EXTRA_OBS,
                old.shape[-1] + STRUT_EXTRA_OBS + HINGE_OBSERVATIONS,
                old.shape[-1] + HINGE_OBSERVATIONS):
            value = torch.full_like(template, fill)
            value[..., :old.shape[-1]] = old
            # The fourth feature of each rod changes command units.
            value[..., old.shape[-1] - STRUT_OBSERVATIONS + 3:old.shape[-1]:5] = fill
            expanded[key] = value
            continue
        valid_deltas = (STRUT_OBSERVATIONS,
                        STRUT_OBSERVATIONS + STRUT_EXTRA_OBS,
                        STRUT_OBSERVATIONS + STRUT_EXTRA_OBS + HINGE_OBSERVATIONS)
        if (template.shape[:-1] != old.shape[:-1]
                or (template.shape[-1] - old.shape[-1]) not in valid_deltas):
            raise ValueError(f'外骨骼观测形状不兼容：{key}')
        value = torch.full_like(template, fill)
        value[..., :old.shape[-1]] = old
        expanded[key] = value
    if initial_command is not None:
        for key, fill in (('mlp.4.weight', 0.), ('mlp.4.bias', initial_command),
                          ('distribution.std_param', .25)):
            old, template = source[key], target[key]
            if append_hinge_io:
                if (template.shape[1:] != old.shape[1:]
                        or template.shape[0] != old.shape[0] + HINGE_ACTIONS):
                    raise ValueError(f'外骨骼铰链动作形状不兼容：{key}')
                hinge_fill = 0. if key == 'mlp.4.bias' else fill
                value = torch.full_like(template, hinge_fill)
                value[:old.shape[0]] = old
                expanded[key] = value
                continue
            if replace_strut_io:
                old = old[:-STRUT_ACTIONS]
            added = template.shape[0] - old.shape[0]
            if (template.shape[1:] != old.shape[1:]
                    or added not in (STRUT_ACTIONS, STRUT_ACTIONS + HINGE_ACTIONS)):
                raise ValueError(f'外骨骼动作形状不兼容：{key}')
            value = torch.full_like(template, fill)
            value[:old.shape[0]] = old
            if added == STRUT_ACTIONS + HINGE_ACTIONS and key == 'mlp.4.bias':
                value[-HINGE_ACTIONS:] = 0.
            expanded[key] = value
    return expanded


def mount_names(side):
    return f'robot/exo_{side}_pack', f'robot/exo_{side}_boot'


MIN_LENGTH_M = .63
MAX_LENGTH_M = 1.22
TUBE_LENGTH_M = .63
WAIST_MIN_LENGTH_M = .44
WAIST_MAX_LENGTH_M = 1.02
WAIST_TUBE_LENGTH_M = .44
HUT_MIN_LENGTH_M = .52
HUT_MAX_LENGTH_M = 1.15
HUT_TUBE_LENGTH_M = .52
AFT_BALL_MIN_LENGTH_M = .56
AFT_BALL_MAX_LENGTH_M = 1.18
AFT_BALL_TUBE_LENGTH_M = .56
MECHANISM = 'hooke_slide_ball_v3'
WAIST_CLEVIS_MECHANISM = 'waist_clevis_v4'
HUT_RING_CLEVIS_MECHANISM = 'hut_ring_clevis_v5'
HUT_BALL_AFT_MECHANISM = 'hut_ball_aft_v6'
HUT_HINGE_AFT_MECHANISM = 'hut_hinge_aft_v7'
HUT_HINGE_HOOF_MECHANISM = 'hut_hinge_hoof_v8'
HINGE_MECHANISMS = (HUT_HINGE_AFT_MECHANISM, HUT_HINGE_HOOF_MECHANISM)
HUT_AFT_MECHANISMS = (HUT_BALL_AFT_MECHANISM, HUT_HINGE_AFT_MECHANISM, HUT_HINGE_HOOF_MECHANISM)
HUT_MECHANISMS = (HUT_RING_CLEVIS_MECHANISM, HUT_BALL_AFT_MECHANISM,
                  HUT_HINGE_AFT_MECHANISM, HUT_HINGE_HOOF_MECHANISM)
MECHANISMS = (MECHANISM, WAIST_CLEVIS_MECHANISM, HUT_RING_CLEVIS_MECHANISM,
              HUT_BALL_AFT_MECHANISM, HUT_HINGE_AFT_MECHANISM, HUT_HINGE_HOOF_MECHANISM)
SOCKET_RADIUS_M = .020
# Park the push-only servo below the working stroke while assistance is released.
PD_RELEASE_LENGTH_M = MIN_LENGTH_M - .50

EFFICIENCY_REWARD = {
    'revision': 'supported_work_sharing_v3',
    'human_absolute_power': 2.8,
    'human_mean_squared_torque': 1.5,
    'peak_leg_squared_torque': 1.5,
    'supported_exo_work_share': 12.,
    'supported_exo_positive_power': 18.,
}
ACTION_FORCE_EFFICIENCY_REWARD = {
    'revision': 'human_joint_metabolic_proxy_bilateral_v2',
    # Margaria/Minetti gross work-efficiency approximation for signed work.
    'positive_work_cost': 1. / .25,
    'negative_work_cost': 1. / 1.2,
    # Keep the previous reward weight per watt of positive joint work.
    'human_metabolic_proxy_power': EFFICIENCY_REWARD['human_absolute_power'] * .25,
    'human_mean_squared_torque': EFFICIENCY_REWARD['human_mean_squared_torque'],
    'peak_leg_squared_torque': EFFICIENCY_REWARD['peak_leg_squared_torque'],
    'bilateral_force_balance': 4.,
    'bilateral_force_share_deadband': .20,
    'bilateral_min_loaded_seconds': .03,
}

LOAD_SHARING_REWARD = {
    **{key: ACTION_FORCE_EFFICIENCY_REWARD[key] for key in
       ('positive_work_cost', 'negative_work_cost', 'human_mean_squared_torque', 'peak_leg_squared_torque')},
    'revision': 'human_energy_and_support_impulse_v3',
    'human_metabolic_proxy_power': 2.4,
    'strut_impulse_balance': 6.,
    'boot_support_balance': 4.,
    'bilateral_force_share_deadband': .20,
    'single_foot_overload_bw': 6.,
    'single_foot_overload_weight': 2.,
}


VERTICAL_SUPPORT_REWARD = {
    **LOAD_SHARING_REWARD,
    'revision': 'human_energy_and_vertical_support_v4',
    'single_foot_overload_bw': 4.5,
    'touchdown_window_s': .05,
    'touchdown_impulse_bw_s': .15,
    'touchdown_impulse_weight': 4.,
}


POSTURE_SUPPORT_REWARD = {
    **VERTICAL_SUPPORT_REWARD,
    'revision': 'human_energy_posture_and_push_support_v5',
    'posture_filter_seconds': 1.5,
    'pack_roll_free_deg': 1.5,
    'waist_roll_free_deg': 4.,
    'pack_roll_weight_per_rad2': 850.,
    'waist_roll_weight_per_rad2': 160.,
    'push_boot_vertical_impulse_balance': 8.,
    'push_extra_strut_impulse_balance': 6.,
}


STANCE_TRANSITION_REWARD = {
    **POSTURE_SUPPORT_REWARD,
    'revision': 'human_energy_stance_transition_v6',
    'braking_sole_weight': 22.,
    'rising_support_weight': 26.,
    'regional_load_bw': .15,
    'rise_support_bw': .40,
    'rise_extension_mps': .18,
    'vertical_speed_mps': .25,
}


def excess_side_difference(totals, deadband):
    left, right = totals.unbind(-1)
    return ((left - right).abs() - deadband * (left + right)).clamp_min(0.)


YOKE_COLOR = [0.22, .29, .36, 1.]
MOTOR_COLOR = [0.14, .18, .24, 1.]
PIN_COLOR = [.80, .45, .14, 1.]
SILVER_COLOR = [.69, .74, .77, 1.]


def _geom(body, name, kind, color, mass=0., **shape):
    return body.add_geom(name=name, type=kind, rgba=color, mass=mass,
                         contype=0, conaffinity=0, group=2, **shape)


def _cylinder(body, name, start, end, radius, color, mass=0.):
    return _geom(body, name, mujoco.mjtGeom.mjGEOM_CYLINDER, color, mass,
                 fromto=[*start, *end], size=[radius, 0., 0.])


def _site(body, name, position, quat=None):
    kw = {'quat': quat} if quat is not None else {}
    body.add_site(name=name, pos=position, size=[.012, .012, .012], group=4, rgba=[0., 0., 0., 0.], **kw)


def _boot_point(sole, offset):
    point = np.empty(3)
    mujoco.mju_rotVecQuat(point, np.asarray(offset, dtype=float), sole.quat)
    return sole.pos + point


def _hoof_shell_vertices(sign):
    """Compact sloped hoof-seat shell (m): +X front, +Y*sign outer, -Y*sign flat magnetic dock, -Z sole."""
    s = float(sign)
    # Clean truncated hoof seat flaring smoothly from the damped ball socket (z = +0.005)
    # down to the monolithic anti-slip sole pad (z = -0.014), with a flat medial docking face.
    pts = [
        # Top collar hugging the damped ball-joint socket (z = +0.005)
        (+.020, -.020 * s, +.005),
        (+.020, +.018 * s, +.005),
        (+.014, +.022 * s, +.005),
        (-.014, +.022 * s, +.005),
        (-.020, +.018 * s, +.005),
        (-.020, -.020 * s, +.005),
        # Flared hoof-seat base flush atop the rubber sole pad (z = -0.014)
        (+.034, -.021 * s, -.014),
        (+.034, +.025 * s, -.014),
        (+.028, +.031 * s, -.014),
        (-.024, +.031 * s, -.014),
        (-.030, +.025 * s, -.014),
        (-.030, -.021 * s, -.014),
    ]
    return [coord for p in pts for coord in p]


def add_exoskeleton(spec, *, pd_force=None, guard_action_force=False, contact_tangent=False,
                    mechanism=MECHANISM, detached=False, hoof_pad_collision=True):
    """Build Hooke–prismatic–ball chains that close against the boots (or open for ground bracing)."""
    if mechanism not in MECHANISMS:
        raise ValueError(f'未知的外骨骼机构版本：{mechanism}')
    if detached and mechanism not in HINGE_MECHANISMS:
        raise ValueError('支杆下端解耦撑地模式需要 hut_hinge_aft_v7 或 hut_hinge_hoof_v8 机构')
    waist_clevis = mechanism == WAIST_CLEVIS_MECHANISM
    hut_ring = mechanism == HUT_RING_CLEVIS_MECHANISM
    hut_ball_aft = mechanism == HUT_BALL_AFT_MECHANISM
    hut_hinge_hoof = mechanism == HUT_HINGE_HOOF_MECHANISM
    hut_hinge_aft = mechanism in HINGE_MECHANISMS
    hut_aft = mechanism in HUT_AFT_MECHANISMS
    min_len = (WAIST_MIN_LENGTH_M if waist_clevis else
               (AFT_BALL_MIN_LENGTH_M if hut_aft else
                (HUT_MIN_LENGTH_M if hut_ring else MIN_LENGTH_M)))
    max_len = (WAIST_MAX_LENGTH_M if waist_clevis else
               (AFT_BALL_MAX_LENGTH_M if hut_aft else
                (HUT_MAX_LENGTH_M if hut_ring else MAX_LENGTH_M)))
    tube_len = (WAIST_TUBE_LENGTH_M if waist_clevis else
                (AFT_BALL_TUBE_LENGTH_M if hut_aft else
                 (HUT_TUBE_LENGTH_M if hut_ring else TUBE_LENGTH_M)))
    upper_body = spec.body('robot/Pelvis') if waist_clevis else spec.geom('robot/backpack').parent
    backpack = spec.geom('robot/backpack')
    if waist_clevis:
        # Structural waist harness belt wrapping the pelvis brief.
        _geom(upper_body, 'robot/exo_waist_belt', mujoco.mjtGeom.mjGEOM_BOX, YOKE_COLOR,
              pos=[-.015, 0., .045], size=[.172, .222, .034])
    elif hut_aft:
        # Open-front C-shaped rigid HUT bottom ring shifted aft toward the PLSS lower
        # frame so uphill jumps receive positive forward-pitch drive under the 91 kg
        # Torso+PLSS COM, paired with the simplest possible coaxial strut + outer-sole ball.
        ring_z = .045
        ring_x0, ring_x1, ring_y = -.220, -.025, .268
        _cylinder(upper_body, 'robot/exo_hut_ring_left', [ring_x0, ring_y, ring_z], [ring_x1, ring_y, ring_z], .015, YOKE_COLOR)
        _cylinder(upper_body, 'robot/exo_hut_ring_right', [ring_x0, -ring_y, ring_z], [ring_x1, -ring_y, ring_z], .015, YOKE_COLOR)
        _cylinder(upper_body, 'robot/exo_hut_ring_aft', [ring_x0, -ring_y, ring_z], [ring_x0, ring_y, ring_z], .016, YOKE_COLOR)
        for side, sign in (('left', 1.), ('right', -1.)):
            _cylinder(upper_body, f'robot/exo_{side}_aft_plss_brace',
                      [-.150, sign * ring_y, ring_z],
                      [backpack.pos[0] + .06, sign * (backpack.size[1] - .02), ring_z + .18],
                      .012, SILVER_COLOR)
    elif hut_ring:
        # Open-front C-shaped rigid saddle/ring bolted directly to the Wangyu
        # hard upper torso (HUT) bottom rim and braced into the PLSS rigid frame
        # (zero front-belly obstruction, zero extra human-body harness needed).
        ring_z = .035
        ring_x0, ring_x1, ring_y = -.215, .010, .265
        _cylinder(upper_body, 'robot/exo_hut_ring_left', [ring_x0, ring_y, ring_z], [ring_x1, ring_y, ring_z], .015, YOKE_COLOR)
        _cylinder(upper_body, 'robot/exo_hut_ring_right', [ring_x0, -ring_y, ring_z], [ring_x1, -ring_y, ring_z], .015, YOKE_COLOR)
        _cylinder(upper_body, 'robot/exo_hut_ring_aft', [ring_x0, -ring_y, ring_z], [ring_x0, ring_y, ring_z], .016, YOKE_COLOR)
        # Diagonal structural braces tying the HUT ring directly into the PLSS rigid frame.
        for side, sign in (('left', 1.), ('right', -1.)):
            _cylinder(upper_body, f'robot/exo_{side}_plss_brace',
                      [-.045, sign * ring_y, ring_z],
                      [backpack.pos[0] + .06, sign * (backpack.size[1] - .02), ring_z + .16],
                      .012, SILVER_COLOR)
    mounts = {}
    for side, sign in (('left', 1.), ('right', -1.)):
        sole = spec.geom(f'robot/{side}_sole')
        upper_name, lower_name = mount_names(side)
        if waist_clevis:
            # Upper mount on rigid waist-harness side aligned near pelvis/COM sagittal axis.
            upper = np.array([-.025, sign * .235, .045], dtype=float)
            boot_offset = np.array([-.15 * sole.size[0], 0., -sole.size[2] + .045], dtype=float)
            lower_pos = _boot_point(sole, boot_offset)
        elif hut_hinge_hoof:
            # Aft HUT ring upper mount + outer-sole magnetic dock pad flush against the boot sole.
            # lower_name sits at the damped ball-joint center when the hoof is magnetically docked;
            # exo_{side}_boot_dock sits on the exact mating magnetic plate plane.
            upper = np.array([-.150, sign * .285, .045], dtype=float)
            dock_offset = np.array([-.42 * sole.size[0], sign * (sole.size[1] + .008),
                                    -sole.size[2] + HOOF_DOCK_Z_ABOVE_SOLE_M], dtype=float)
            boot_offset = dock_offset + np.array([0., sign * HOOF_DOCK_OFFSET_M, 0.], dtype=float)
            lower_pos = _boot_point(sole, boot_offset)
            _site(sole.parent, f'robot/exo_{side}_boot_dock', _boot_point(sole, dock_offset), quat=sole.quat)
        elif hut_aft:
            # Aft HUT ring upper mount under the 91 kg Torso+PLSS COM + simplified outer-sole
            # ball joint directly beneath the outer ankle malleolus (minimal mechanics, active
            # closed-loop force control prevents ankle inversion/eversion).
            upper = np.array([-.150, sign * .285, .045], dtype=float)
            boot_offset = np.array([-.42 * sole.size[0], sign * (sole.size[1] + .015),
                                    -sole.size[2] + SOCKET_RADIUS_M], dtype=float)
            lower_pos = _boot_point(sole, boot_offset)
        elif hut_ring:
            # Upper mount bolted directly to Wangyu HUT bottom rigid ring sides
            # (world x = -0.029 m: aligned closely with BOTH whole-body COM and WaistPitch axis).
            upper = np.array([-.045, sign * .288, .035], dtype=float)
            # Lower mount at exact boot sagittal neutral plane (y_sole = 0) directly
            # beneath the ankle axis, eliminating ankle roll torque (~1 cm moment arm).
            boot_offset = np.array([-.095, 0., -sole.size[2] + .048], dtype=float)
            lower_pos = _boot_point(sole, boot_offset)
        else:
            upper = backpack.pos + np.array([.46 * backpack.size[0],
                                             sign * (backpack.size[1] + .018), -.68 * backpack.size[2]])
            # Place the boot mount in the sole's own frame, including its toe-out yaw.
            boot_offset = np.array([-.21 * sole.size[0], sign * (sole.size[1] + .010),
                                    -sole.size[2] + SOCKET_RADIUS_M])
            lower_pos = _boot_point(sole, boot_offset)
        _site(upper_body, upper_name, upper)
        _site(sole.parent, lower_name, lower_pos)
        mounts[side] = (upper, boot_offset, sign)

    reference = spec.compile()
    rest = mujoco.MjData(reference)
    mujoco.mj_kinematics(reference, rest)
    box, sphere = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE
    for side, (upper, boot_offset, sign) in mounts.items():
        name = f'robot/exo_{side}'
        upper_name, lower_name = mount_names(side)
        sole = spec.geom(f'robot/{side}_sole')
        foot = sole.parent
        upper_id = reference.body(upper_body.name).id
        world_axis = rest.site(lower_name).xpos - rest.site(upper_name).xpos
        radius = float(np.linalg.norm(world_axis))
        local_axis = rest.xmat[upper_id].reshape(3, 3).T @ (world_axis / radius)
        rotation = np.empty(4)
        mujoco.mju_quatZ2Vec(rotation, local_axis)

        if hut_hinge_aft:
            # Actuated sagittal-plane rotary hinge mounted rigidly on the aft HUT ring
            # (axis strictly aligned with Torso +Y so positive torque pushes the boot aft
            # and propels the astronaut forward) -> passive coronal Hooke roll hinge ->
            # coaxial telescoping strut -> outer-sole ball joint.
            outward = np.array([0., sign, 0.])
            _geom(upper_body, f'{name}_pack_plate', box, YOKE_COLOR,
                  pos=upper - .026 * outward, size=[.044, .010, .032])
            _cylinder(upper_body, f'{name}_pack_boss', upper - .032 * outward,
                      upper + .006 * outward, .022, YOKE_COLOR)
            # Cylindrical QDD rotary hinge stator mounted along Torso Y axis at the HUT ring.
            _cylinder(upper_body, f'{name}_hinge_motor',
                      upper - .018 * outward, upper + .022 * outward, .036, MOTOR_COLOR, .35)
            _cylinder(upper_body, f'{name}_hinge_cap',
                      upper + .022 * outward, upper + .029 * outward, .030, PIN_COLOR)
            # Construct exact sagittal yoke frame: Y = Torso +Y, Z = sagittal projection of strut.
            y_hat = np.array([0., 1., 0.], dtype=float)
            z_sag = np.array([local_axis[0], 0., local_axis[2]], dtype=float)
            z_sag /= max(float(np.linalg.norm(z_sag)), 1e-9)
            x_sag = np.cross(y_hat, z_sag)
            yoke_rot = np.column_stack((x_sag, y_hat, z_sag))
            yoke_quat = np.empty(4)
            mujoco.mju_mat2Quat(yoke_quat, yoke_rot.ravel())
            phi_0 = float(np.arctan2(-float(local_axis @ y_hat), float(local_axis @ z_sag)))
            sleeve_quat = np.array([np.cos(.5 * phi_0), np.sin(.5 * phi_0), 0., 0.], dtype=float)

            yoke = upper_body.add_body(name=f'{name}_yoke', pos=upper, quat=yoke_quat)
            hinge_joint = f'{name}_hinge_pitch'
            yoke.add_joint(name=hinge_joint, type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=[0., 1., 0.], limited=True,
                           range=([DETACHED_HINGE_MIN_RAD, DETACHED_HINGE_MAX_RAD]
                                  if detached else [HINGE_MIN_ANGLE_RAD, HINGE_MAX_ANGLE_RAD]),
                           damping=2.0 if detached else .30,
                           armature=.015 if detached else .012)
            _cylinder(yoke, f'{name}_hinge_pin', [0., -.028, 0.], [0., .028, 0.], .013, PIN_COLOR, .10)
            _cylinder(yoke, f'{name}_hinge_yoke', [0., 0., 0.], [0., 0., .046], .022, SILVER_COLOR)
            sleeve = yoke.add_body(name=f'{name}_sleeve', quat=sleeve_quat)
            roll_joint = f'{name}_hooke_roll'
            if detached or hut_hinge_hoof:
                # Coronal abduction/adduction range supports lateral bracing and clean magnetic undocking.
                sleeve.add_joint(name=roll_joint, type=mujoco.mjtJoint.mjJNT_HINGE,
                                 axis=[1., 0., 0.], limited=True,
                                 range=[-DETACHED_ROLL_LIMIT_RAD, DETACHED_ROLL_LIMIT_RAD],
                                 damping=1.5 if detached else .15,
                                 armature=.008 if detached else .002)
            else:
                sleeve.add_joint(name=roll_joint, type=mujoco.mjtJoint.mjJNT_HINGE,
                                 axis=[1., 0., 0.], damping=.045, armature=.001)
            _cylinder(sleeve, f'{name}_inner_pin', [-.022, 0., 0.], [.022, 0., 0.], .0115, SILVER_COLOR)
            _cylinder(sleeve, f'{name}_outer_tube', [0., 0., .014], [0., 0., tube_len], .020, YOKE_COLOR, .30)
            _cylinder(sleeve, f'{name}_guide_collar', [0., 0., tube_len - .032], [0., 0., tube_len], .0235, SILVER_COLOR)
            shaft = sleeve.add_body(name=f'{name}_shaft', pos=[0., 0., radius])
            slide = f'{name}_slide'
            shaft.add_joint(name=slide, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0., 0., 1.],
                            limited=True, range=[min_len - radius, max_len - radius],
                            damping=2.0 if detached else .20,
                            armature=.010 if detached else .005)
            _cylinder(shaft, f'{name}_sliding_shaft', [0., 0., -(tube_len - .02)], [0., 0., 0.], .011, SILVER_COLOR, .25)
            if hut_hinge_hoof:
                # Damped spherical ball-joint wrist -> anti-slip bionic cloven hoof -> medial magnetic boot dock.
                # Align hoof rest frame exactly with boot sole frame so magnetic dock mates with 0 mm / 0 deg error.
                _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, pos=[0., 0., 0.], size=[.017, 0., 0.])
                _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
                R_sleeve_rel = np.empty(9)
                mujoco.mju_quat2Mat(R_sleeve_rel, sleeve_quat)
                R_shaft_world = rest.xmat[upper_id].reshape(3, 3) @ yoke_rot @ R_sleeve_rel.reshape(3, 3)
                R_sole_world = rest.geom_xmat[reference.geom(f'robot/{side}_sole').id].reshape(3, 3)
                R_hoof_rest = R_shaft_world.T @ R_sole_world
                hoof_rest_quat = np.empty(4)
                mujoco.mju_mat2Quat(hoof_rest_quat, R_hoof_rest.ravel())

                hoof = shaft.add_body(name=f'{name}_hoof', pos=[0., 0., 0.], quat=hoof_rest_quat)
                hoof.add_joint(name=f'{name}_hoof_ball', type=mujoco.mjtJoint.mjJNT_BALL,
                               limited=False,
                               damping=HOOF_BALL_DAMPING, stiffness=HOOF_BALL_STIFFNESS,
                               armature=HOOF_BALL_ARMATURE)
                # Elastomer damped ball-joint retainer collar at top of hoof.
                _cylinder(hoof, f'{name}_hoof_socket', [0., 0., -.004], [0., 0., .011], .0225, MOTOR_COLOR, .06)
                # Compact flared hoof-seat structural shell (visual convex mesh).
                mesh_name = f'{name}_hoof_shell_mesh'
                spec.add_mesh(name=mesh_name, uservert=_hoof_shell_vertices(sign))
                hoof.add_geom(name=f'{name}_hoof_shell', type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=mesh_name, rgba=YOKE_COLOR, mass=.16,
                              contype=0, conaffinity=0, group=2)
                # Monolithic flat-bottom anti-slip rubber hoof sole pad (exact bottom at z = -HOOF_SOLE_HEIGHT_M).
                # Keep collision enabled in BVH by default so runtime Exo <-> Exolimb switching works;
                # pass hoof_pad_collision=False for pure docked ExoJump Warp training (+0.2% vs v7).
                pad_hz = .004
                pad_z = -HOOF_SOLE_HEIGHT_M + pad_hz
                pad_col = bool(detached or hoof_pad_collision)
                hoof.add_geom(name=f'{name}_hoof_pad', type=box, rgba=HOOF_PAD_COLOR, mass=.20,
                              pos=[.002, sign * .005, pad_z], size=[.032, .026, pad_hz],
                              condim=4, friction=list(HOOF_FRICTION),
                              contype=2 if pad_col else 0,
                              conaffinity=1 if pad_col else 0, group=2)
                # Medial flush magnetic coupling face & electromagnetic coil ring.
                _geom(hoof, f'{name}_hoof_mag_plate', box, SILVER_COLOR,
                      pos=[0., -sign * (HOOF_DOCK_OFFSET_M - .0025), -.002], size=[.024, .0025, .014])
                _cylinder(hoof, f'{name}_hoof_mag_coil',
                          [0., -sign * (HOOF_DOCK_OFFSET_M - .0005), -.002],
                          [0., -sign * (HOOF_DOCK_OFFSET_M - .0055), -.002], .0115, PIN_COLOR)
                _site(hoof, f'{name}_hoof_dock', [0., -sign * HOOF_DOCK_OFFSET_M, 0.])
                _site(hoof, f'{name}_hoof_sole_site', [.002, sign * .005, -HOOF_SOLE_HEIGHT_M])

                # Boot outer-sole magnetic docking receiver pad + mode status indicator LED.
                dock_offset = boot_offset - np.array([0., sign * HOOF_DOCK_OFFSET_M, 0.], dtype=float)
                _geom(foot, f'{name}_boot_mag_base', box, YOKE_COLOR,
                      pos=_boot_point(sole, dock_offset + [0., -sign * .008, .004]),
                      quat=sole.quat, size=[.036, .008, .016])
                _geom(foot, f'{name}_boot_mag_plate', box, SILVER_COLOR,
                      pos=_boot_point(sole, dock_offset + [0., -sign * .002, -.002]),
                      quat=sole.quat, size=[.026, .0025, .015])
                _geom(foot, f'{name}_mag_led', box,
                      MAG_LED_DETACHED_COLOR if detached else MAG_LED_DOCKED_COLOR,
                      pos=_boot_point(sole, dock_offset + [.029, -sign * .001, .006]),
                      quat=sole.quat, size=[.004, .004, .008])
                # Runtime-switchable electromagnetic dock lock (active=True -> Exo mode; active=False -> Exolimb mode).
                spec.add_equality(name=f'{name}_mag_dock', type=mujoco.mjtEq.mjEQ_WELD,
                                  objtype=mujoco.mjtObj.mjOBJ_SITE,
                                  name1=f'{name}_hoof_dock', name2=f'{name}_boot_dock',
                                  active=not detached,
                                  solref=[.008, 1.], solimp=[.95, .99, .001, .5, 2.])
            elif detached:
                # Opened lower quick-release end-effector: rubberized spherical ground-contact tip.
                shaft.add_geom(name=f'{name}_ball', type=sphere, rgba=PIN_COLOR, mass=.35,
                               pos=[0., 0., 0.], size=[DETACHED_TIP_RADIUS_M, 0., 0.],
                               contype=2, conaffinity=1, group=2, friction=[1., .01, .001])
                _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
                _geom(foot, f'{name}_boot_plate', box, YOKE_COLOR,
                      pos=_boot_point(sole, boot_offset + [0., -.030 * sign, .020]),
                      quat=sole.quat, size=[.038, .010, .017])
                for edge, offset in (('fore', .019), ('aft', -.019)):
                    _cylinder(foot, f'{name}_boot_lug_{edge}',
                              _boot_point(sole, boot_offset + [offset, -.028 * sign, .024]),
                              _boot_point(sole, boot_offset + [offset, -.010 * sign, .002]),
                              .010, SILVER_COLOR)
                    _cylinder(foot, f'{name}_socket_{edge}',
                              _boot_point(sole, boot_offset + [offset - .005, 0., 0.]),
                              _boot_point(sole, boot_offset + [offset + .005, 0., 0.]),
                              SOCKET_RADIUS_M, YOKE_COLOR)
            else:
                _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, pos=[0., 0., 0.], size=[.017, 0., 0.])
                _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
                _geom(foot, f'{name}_boot_plate', box, YOKE_COLOR,
                      pos=_boot_point(sole, boot_offset + [0., -.030 * sign, .020]),
                      quat=sole.quat, size=[.038, .010, .017])
                for edge, offset in (('fore', .019), ('aft', -.019)):
                    _cylinder(foot, f'{name}_boot_lug_{edge}',
                              _boot_point(sole, boot_offset + [offset, -.028 * sign, .024]),
                              _boot_point(sole, boot_offset + [offset, -.010 * sign, .002]),
                              .010, SILVER_COLOR)
                    _cylinder(foot, f'{name}_socket_{edge}',
                              _boot_point(sole, boot_offset + [offset - .005, 0., 0.]),
                              _boot_point(sole, boot_offset + [offset + .005, 0., 0.]),
                              SOCKET_RADIUS_M, YOKE_COLOR)
                spec.add_equality(name=f'{name}_ball_joint', type=mujoco.mjtEq.mjEQ_CONNECT,
                                  objtype=mujoco.mjtObj.mjOBJ_SITE, name1=f'{name}_ball_pin', name2=lower_name,
                                  active=True, solref=[.008, 1.], solimp=[.95, .99, .001, .5, 2.])
        elif hut_ball_aft:
            # Ultra-simple, 100% coaxial Hooke -> Prismatic -> Outer-Sole Ball chain:
            # - Strictly coaxial outer_tube & sliding_shaft along pure [0, 0, 1] (0.0000 mm separation)
            # - Ball sphere on shaft AT [0,0,0] == ball_pin, socket on foot AT lower_name (0.0000 mm detach)
            outward = np.array([0., sign, 0.])
            _geom(upper_body, f'{name}_pack_plate', box, YOKE_COLOR,
                  pos=upper - .022 * outward, size=[.040, .010, .028])
            _cylinder(upper_body, f'{name}_pack_boss', upper - .030 * outward,
                      upper + .010 * outward, .018, YOKE_COLOR)
            yoke = upper_body.add_body(name=f'{name}_yoke', pos=upper, quat=rotation)
            yoke.add_joint(name=f'{name}_hooke_a', type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=[1., 0., 0.], damping=.045, armature=.001)
            _cylinder(yoke, f'{name}_outer_pin', [-.024, 0., 0.], [.024, 0., 0.], .011, PIN_COLOR, .08)
            sleeve = yoke.add_body(name=f'{name}_sleeve')
            sleeve.add_joint(name=f'{name}_hooke_b', type=mujoco.mjtJoint.mjJNT_HINGE,
                             axis=[0., 1., 0.], damping=.045, armature=.001)
            _cylinder(sleeve, f'{name}_inner_pin', [0., -.020, 0.], [0., .020, 0.], .0115, SILVER_COLOR)
            _cylinder(sleeve, f'{name}_outer_tube', [0., 0., .010], [0., 0., tube_len], .020, YOKE_COLOR, .30)
            _cylinder(sleeve, f'{name}_guide_collar', [0., 0., tube_len - .032], [0., 0., tube_len], .0235, SILVER_COLOR)
            shaft = sleeve.add_body(name=f'{name}_shaft', pos=[0., 0., radius])
            slide = f'{name}_slide'
            shaft.add_joint(name=slide, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0., 0., 1.],
                            limited=True, range=[min_len - radius, max_len - radius],
                            damping=.20, armature=.005)
            _cylinder(shaft, f'{name}_sliding_shaft', [0., 0., -(tube_len - .02)], [0., 0., 0.], .011, SILVER_COLOR, .25)
            _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, pos=[0., 0., 0.], size=[.017, 0., 0.])
            _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
            # Compact outer-sole socket & bracket right on the boot sole edge (exact ball center = lower_name)
            _geom(foot, f'{name}_boot_plate', box, YOKE_COLOR,
                  pos=_boot_point(sole, boot_offset + [0., -.030 * sign, .020]),
                  quat=sole.quat, size=[.038, .010, .017])
            for edge, offset in (('fore', .019), ('aft', -.019)):
                _cylinder(foot, f'{name}_boot_lug_{edge}',
                          _boot_point(sole, boot_offset + [offset, -.028 * sign, .024]),
                          _boot_point(sole, boot_offset + [offset, -.010 * sign, .002]),
                          .010, SILVER_COLOR)
                _cylinder(foot, f'{name}_socket_{edge}',
                          _boot_point(sole, boot_offset + [offset - .005, 0., 0.]),
                          _boot_point(sole, boot_offset + [offset + .005, 0., 0.]),
                          SOCKET_RADIUS_M, YOKE_COLOR)
            spec.add_equality(name=f'{name}_ball_joint', type=mujoco.mjtEq.mjEQ_CONNECT,
                              objtype=mujoco.mjtObj.mjOBJ_SITE, name1=f'{name}_ball_pin', name2=lower_name,
                              active=True, solref=[.008, 1.], solimp=[.95, .99, .001, .5, 2.])
        else:
            # Fixed upper side bracket. The two pins follow different axes.
            outward = np.array([0., sign, 0.])
            _geom(upper_body, f'{name}_pack_plate', box, YOKE_COLOR,
                  pos=upper - .022 * outward, size=[.040, .010, .028])
            _cylinder(upper_body, f'{name}_pack_boss', upper - .030 * outward,
                      upper + .010 * outward, .018, YOKE_COLOR)
            yoke = upper_body.add_body(name=f'{name}_yoke', pos=upper, quat=rotation)
            yoke.add_joint(name=f'{name}_hooke_a', type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=[1., 0., 0.], damping=.045, armature=.001)
            _cylinder(yoke, f'{name}_outer_pin', [-.026, 0., 0.], [.026, 0., 0.], .011, PIN_COLOR, .10)
            sleeve = yoke.add_body(name=f'{name}_sleeve')
            sleeve.add_joint(name=f'{name}_hooke_b', type=mujoco.mjtJoint.mjJNT_HINGE,
                             axis=[0., 1., 0.], damping=.045, armature=.001)
            _cylinder(sleeve, f'{name}_inner_pin', [0., -.022, 0.], [0., .022, 0.], .012, SILVER_COLOR)
            if hut_ring:
                rot_mat = np.empty(9)
                mujoco.mju_quat2Mat(rot_mat, rotation)
                R_sleeve = rot_mat.reshape(3, 3)
                lat_local = R_sleeve.T @ np.array([0., sign, 0.])
                lat_local[2] = 0.
                lat_local = lat_local / max(float(np.linalg.norm(lat_local)), 1e-6)
                p_top = np.array([0., 0., .012])
                p_col = np.array([0., 0., tube_len]) + .055 * lat_local
                p_col0 = p_top + ((tube_len - .032) / max(tube_len, 1e-6)) * (p_col - p_top)
                _cylinder(sleeve, f'{name}_outer_tube', p_top, p_col, .019, YOKE_COLOR, .30)
                _cylinder(sleeve, f'{name}_guide_collar', p_col0, p_col, .023, SILVER_COLOR)
            else:
                _cylinder(sleeve, f'{name}_outer_tube', [0., 0., .010], [0., 0., tube_len], .020, YOKE_COLOR, .30)
                _cylinder(sleeve, f'{name}_guide_collar', [0., 0., tube_len - .028],
                          [0., 0., tube_len], .023, SILVER_COLOR)
            shaft = sleeve.add_body(name=f'{name}_shaft', pos=[0., 0., radius])
            slide = f'{name}_slide'
            shaft.add_joint(name=slide, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0., 0., 1.],
                            limited=True, range=[min_len - radius, max_len - radius],
                            damping=.20, armature=.005)
            if hut_ring:
                half_w = .084
                rear_x = -sole.size[0] - .052
                heel_mid_foot = _boot_point(sole, [rear_x, 0., boot_offset[2] + .012])
                foot_id = reference.body(foot.name).id
                heel_mid_world = rest.xpos[foot_id] + rest.xmat[foot_id].reshape(3, 3) @ heel_mid_foot
                lower_world = rest.site(lower_name).xpos
                shaft_rot_world = rest.xmat[upper_id].reshape(3, 3) @ R_sleeve
                hub_shaft = shaft_rot_world.T @ (heel_mid_world - lower_world)
                p_knee = np.array([0., 0., -.14]) + .088 * lat_local
                tube_dir = (p_col - p_top) / max(float(np.linalg.norm(p_col - p_top)), 1e-6)
                p_start = p_knee - (tube_len - .08) * tube_dir
                _cylinder(shaft, f'{name}_sliding_shaft', p_start, p_knee, .011, SILVER_COLOR, .27)
                _geom(shaft, f'{name}_yoke_elbow', sphere, YOKE_COLOR, pos=p_knee, size=[.014, 0., 0.])
                _cylinder(shaft, f'{name}_center_yoke', p_knee, hub_shaft, .012, YOKE_COLOR)
                _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, pos=hub_shaft, size=[.018, 0., 0.])
                _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
                pin_out = _boot_point(sole, boot_offset + [0., sign * half_w, 0.])
                pin_in = _boot_point(sole, boot_offset + [0., -sign * half_w, 0.])
                heel_out = _boot_point(sole, [rear_x + .016, sign * half_w, boot_offset[2] + .004])
                heel_in = _boot_point(sole, [rear_x + .016, -sign * half_w, boot_offset[2] + .004])
                _cylinder(foot, f'{name}_stirrup_bridge_out', heel_mid_foot, heel_out, .013, YOKE_COLOR)
                _cylinder(foot, f'{name}_stirrup_bridge_in', heel_mid_foot, heel_in, .013, YOKE_COLOR)
                _cylinder(foot, f'{name}_stirrup_outer', heel_out, pin_out, .012, SILVER_COLOR)
                _cylinder(foot, f'{name}_stirrup_inner', heel_in, pin_in, .012, SILVER_COLOR)
                _geom(foot, f'{name}_stirrup_apex', sphere, YOKE_COLOR, pos=heel_mid_foot, size=[.017, 0., 0.])
                for tag, s_dir in (('outer', sign), ('inner', -sign)):
                    _geom(foot, f'{name}_boot_plate_{tag}', box, YOKE_COLOR,
                          pos=_boot_point(sole, boot_offset + [0., s_dir * (half_w - .010), .004]),
                          quat=sole.quat, size=[.036, .008, .019])
                    _cylinder(foot, f'{name}_hinge_pin_{tag}',
                              _boot_point(sole, boot_offset + [0., s_dir * (half_w - .016), 0.]),
                              _boot_point(sole, boot_offset + [0., s_dir * (half_w + .014), 0.]),
                              .013, PIN_COLOR)
            elif waist_clevis:
                _cylinder(shaft, f'{name}_sliding_shaft', [0., 0., -tube_len], [0., 0., -.085], .011, SILVER_COLOR, .27)
                _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, pos=[0., 0., -.085], size=[.016, 0., 0.])
                _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
                half_w = sole.size[1] + .012
                crown = _boot_point(sole, boot_offset + [-.015, 0., .095])
                _geom(foot, f'{name}_clevis_crown', sphere, YOKE_COLOR, pos=crown, size=[.016, 0., 0.])
                for tag, s_dir in (('outer', sign), ('inner', -sign)):
                    pin_center = _boot_point(sole, boot_offset + [0., s_dir * half_w, 0.])
                    shoulder = _boot_point(sole, boot_offset + [-.010, s_dir * (half_w + .004), .085])
                    _geom(foot, f'{name}_boot_plate_{tag}', box, YOKE_COLOR,
                          pos=_boot_point(sole, boot_offset + [0., s_dir * (sole.size[1] + .002), .004]),
                          quat=sole.quat, size=[.034, .008, .018])
                    _cylinder(foot, f'{name}_hinge_pin_{tag}',
                              _boot_point(sole, boot_offset + [0., s_dir * (sole.size[1] - .004), 0.]),
                              _boot_point(sole, boot_offset + [0., s_dir * (half_w + .010), 0.]),
                              .011, PIN_COLOR)
                    _cylinder(foot, f'{name}_clevis_bridge_{tag}', crown, shoulder, .010, SILVER_COLOR)
                    _cylinder(foot, f'{name}_clevis_arm_{tag}', shoulder, pin_center, .010, SILVER_COLOR)
            else:
                _cylinder(shaft, f'{name}_sliding_shaft', [0., 0., -tube_len], [0., 0., 0.], .010, SILVER_COLOR, .27)
                _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, size=[.015, 0., 0.])
                _site(shaft, f'{name}_ball_pin', [0., 0., 0.])
                _geom(foot, f'{name}_boot_plate', box, YOKE_COLOR,
                      pos=_boot_point(sole, boot_offset + [0., -.030 * sign, .022]),
                      quat=sole.quat, size=[.036, .010, .017])
                for edge, offset in (('fore', .018), ('aft', -.018)):
                    _cylinder(foot, f'{name}_boot_lug_{edge}',
                              _boot_point(sole, boot_offset + [offset, -.028 * sign, .026]),
                              _boot_point(sole, boot_offset + [offset, -.009 * sign, .002]),
                              .010, SILVER_COLOR)
                    _cylinder(foot, f'{name}_socket_{edge}',
                              _boot_point(sole, boot_offset + [offset - .005, 0., 0.]),
                              _boot_point(sole, boot_offset + [offset + .005, 0., 0.]),
                              SOCKET_RADIUS_M, YOKE_COLOR)
            spec.add_equality(name=f'{name}_ball_joint', type=mujoco.mjtEq.mjEQ_CONNECT,
                              objtype=mujoco.mjtObj.mjOBJ_SITE, name1=f'{name}_ball_pin', name2=lower_name,
                              active=True, solref=[.025, 1.])
        actuator = spec.add_actuator(name=name, target=slide, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                     gear=[1., 0., 0., 0., 0., 0.], ctrllimited=True,
                                     forcelimited=True)
        if detached:
            # Critically-damped bidirectional position-PD telescoping actuator for smooth lunar bracing.
            actuator.set_to_position(kp=800., kv=90.)
            actuator.ctrlrange = [min_len - radius, max_len - radius]
            actuator.forcerange = [-450., 450.]
        elif pd_force is None:
            actuator.gainprm = [1.] + [0.] * 9
            actuator.ctrlrange = [0., MAX_FORCE_N]
            actuator.forcerange = [0., MAX_FORCE_N]
        else:
            cap = validate_assist_force(pd_force)
            actuator.set_to_position(kp=PD_KP if cap else 0., kv=PD_KD if cap else 0.)
            actuator.ctrlrange = [PD_RELEASE_LENGTH_M - radius, max_len - radius]
            actuator.forcerange = [0., cap or MAX_FORCE_N]
        spec.add_sensor(name=f'{name}_stroke', type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=slide)
        if hut_hinge_aft:
            hinge_act = spec.add_actuator(name=f'{name}_hinge', target=f'{name}_hinge_pitch',
                                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                          gear=[1., 0., 0., 0., 0., 0.], ctrllimited=True,
                                          forcelimited=True)
            if detached:
                hinge_act.set_to_position(kp=140., kv=22.)
                hinge_act.ctrlrange = [DETACHED_HINGE_MIN_RAD, DETACHED_HINGE_MAX_RAD]
                hinge_act.forcerange = [-DETACHED_HINGE_TORQUE_NM, DETACHED_HINGE_TORQUE_NM]
                roll_act = spec.add_actuator(name=f'{name}_roll', target=f'{name}_hooke_roll',
                                             trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                             gear=[1., 0., 0., 0., 0., 0.], ctrllimited=True,
                                             forcelimited=True)
                roll_act.set_to_position(kp=100., kv=16.)
                roll_act.ctrlrange = [-DETACHED_ROLL_LIMIT_RAD, DETACHED_ROLL_LIMIT_RAD]
                roll_act.forcerange = [-DETACHED_ROLL_TORQUE_NM, DETACHED_ROLL_TORQUE_NM]
                spec.add_sensor(name=f'{name}_roll_pos', type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                                objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f'{name}_hooke_roll')
                spec.add_sensor(name=f'{name}_roll_vel', type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                                objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f'{name}_hooke_roll')
            else:
                hinge_act.gainprm = [1.] + [0.] * 9
                hinge_act.ctrlrange = [-MAX_HINGE_TORQUE_NM, MAX_HINGE_TORQUE_NM]
                hinge_act.forcerange = [-MAX_HINGE_TORQUE_NM, MAX_HINGE_TORQUE_NM]
                if hut_hinge_hoof:
                    # Keep coronal roll actuator declared (passive gain=0 while docked) so
                    # runtime switching to Exolimb Mode works on the same compiled model.
                    roll_act = spec.add_actuator(name=f'{name}_roll', target=f'{name}_hooke_roll',
                                                 trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                                 gear=[1., 0., 0., 0., 0., 0.], ctrllimited=True,
                                                 forcelimited=True)
                    roll_act.set_to_position(kp=0., kv=0.)
                    roll_act.ctrlrange = [-DETACHED_ROLL_LIMIT_RAD, DETACHED_ROLL_LIMIT_RAD]
                    roll_act.forcerange = [-DETACHED_ROLL_TORQUE_NM, DETACHED_ROLL_TORQUE_NM]
                    spec.add_sensor(name=f'{name}_roll_pos', type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                                    objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f'{name}_hooke_roll')
                    spec.add_sensor(name=f'{name}_roll_vel', type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                                    objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f'{name}_hooke_roll')
            spec.add_sensor(name=f'{name}_hinge_pos', type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                            objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f'{name}_hinge_pitch')
            spec.add_sensor(name=f'{name}_hinge_vel', type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                            objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f'{name}_hinge_pitch')
        if guard_action_force:
            ankle = spec.joint(f'robot/{side.title()}AnkleRoll')
            anchor_name = f'{name}_ankle_anchor'
            _site(ankle.parent, anchor_name, ankle.pos)
            for suffix, site, kind in (
                ('pack_pos', upper_name, mujoco.mjtSensor.mjSENS_FRAMEPOS),
                ('boot_pos', lower_name, mujoco.mjtSensor.mjSENS_FRAMEPOS),
                ('ankle_pos', anchor_name, mujoco.mjtSensor.mjSENS_FRAMEPOS),
                ('ankle_axis', anchor_name, mujoco.mjtSensor.mjSENS_FRAMEXAXIS),
            ):
                spec.add_sensor(name=f'{name}_{suffix}', type=kind,
                                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
            # found (1), force (3), position (3), normal (3), sorted by force.
            spec.add_sensor(name=f'{name}_support_contacts', type=mujoco.mjtSensor.mjSENS_CONTACT,
                            objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=sole.name,
                            reftype=mujoco.mjtObj.mjOBJ_GEOM, refname='terrain',
                            intprm=[115 if contact_tangent else 51, 2, GUARD_CONTACTS_PER_FOOT])


def fit_struts_to_pose(model, data):
    """Close the struts on a manually set human pose before static rendering."""
    mujoco.mj_kinematics(model, data)
    for side in STRUTS:
        name = f'robot/exo_{side}'
        upper_name, lower_name = mount_names(side)
        yoke = model.body(f'{name}_yoke')
        torso = model.body(int(yoke.parentid[0]))
        mount_vector = data.site(lower_name).xpos - data.site(upper_name).xpos
        v = data.xmat[torso.id].reshape(3, 3).T @ mount_vector
        basis = np.empty(9)
        mujoco.mju_quat2Mat(basis, yoke.quat)
        x, y, z = basis.reshape(3, 3).T @ v
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_hinge_pitch') >= 0:
            sleeve = model.body(f'{name}_sleeve')
            phi_0 = 2. * np.arctan2(float(sleeve.quat[1]), float(sleeve.quat[0]))
            data.joint(f'{name}_hinge_pitch').qpos[0] = np.arctan2(x, z)
            data.joint(f'{name}_hooke_roll').qpos[0] = np.arctan2(-y, np.hypot(x, z)) - phi_0
        else:
            data.joint(f'{name}_hooke_a').qpos[0] = np.arctan2(-y, z)
            data.joint(f'{name}_hooke_b').qpos[0] = np.arctan2(x, np.hypot(y, z))
        data.joint(f'{name}_slide').qpos[0] = np.linalg.norm(v) - model.body(f'{name}_shaft').pos[2]
    mujoco.mj_kinematics(model, data)
    # If damped ball-joint hooves are present, solve exact spherical joint quaternion so hoof_dock aligns to boot_dock.
    need_second_kin = False
    for side in STRUTS:
        name = f'robot/exo_{side}'
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{name}_hoof_ball')
        if ball_id < 0:
            continue
        shaft_id = model.body(f'{name}_shaft').id
        hoof_id = model.body(f'{name}_hoof').id
        dock_id = model.site(f'{name}_boot_dock').id
        R_shaft = data.xmat[shaft_id].reshape(3, 3)
        R_rest = np.empty(9)
        mujoco.mju_quat2Mat(R_rest, model.body_quat[hoof_id])
        R_zero = R_shaft @ R_rest.reshape(3, 3)
        R_target = data.site_xmat[dock_id].reshape(3, 3)
        q_ball = np.empty(4)
        mujoco.mju_mat2Quat(q_ball, (R_zero.T @ R_target).ravel())
        if q_ball[0] < 0.:
            q_ball = -q_ball
        data.joint(f'{name}_hoof_ball').qpos[:] = q_ball
        need_second_kin = True
    if need_second_kin:
        mujoco.mj_kinematics(model, data)


class TelescopingAssist:
    """Select rod targets from the policy and the previous measured support state."""

    def __init__(self, worlds, force, *, hinge_torque=None):
        from cpu_jump_task import lower_leg_actuators

        model = worlds.model
        self.force_n = validate_assist_force(force)
        self.worlds = worlds
        qpos = worlds.sim.data.qpos
        self.has_hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'robot/exo_left_hinge') >= 0
        self.slide_actuators = [model.actuator(f'robot/exo_{side}').id for side in STRUTS]
        self.hinge_actuators = ([model.actuator(f'robot/exo_{side}_hinge').id for side in STRUTS]
                                if self.has_hinge else [])
        self.actuators = self.slide_actuators + self.hinge_actuators
        self.hinge_torque_nm = (validate_hinge_torque(
            DEFAULT_HINGE_TORQUE_NM if hinge_torque is None else hinge_torque) if self.has_hinge else 0.)
        self.stroke = [int(model.sensor(f'robot/exo_{side}_stroke').adr[0]) for side in STRUTS]
        self.nominal_length = torch.tensor([model.body(f'robot/exo_{side}_shaft').pos[2] for side in STRUTS],
                                           dtype=qpos.dtype, device=qpos.device)
        slide_ids = [model.joint(f'robot/exo_{side}_slide').id for side in STRUTS]
        slide_ranges = torch.tensor(model.jnt_range[slide_ids], dtype=qpos.dtype, device=qpos.device)
        self.min_length_m = self.nominal_length + slide_ranges[:, 0]
        self.max_length_m = self.nominal_length + slide_ranges[:, 1]
        if self.has_hinge:
            self.hinge_pos = [int(model.sensor(f'robot/exo_{side}_hinge_pos').adr[0]) for side in STRUTS]
            self.hinge_vel = [int(model.sensor(f'robot/exo_{side}_hinge_vel').adr[0]) for side in STRUTS]
            hinge_ids = [model.joint(f'robot/exo_{side}_hinge_pitch').id for side in STRUTS]
            hinge_ranges = torch.tensor(model.jnt_range[hinge_ids], dtype=qpos.dtype, device=qpos.device)
            self.hinge_min_rad = hinge_ranges[:, 0]
            self.hinge_max_rad = hinge_ranges[:, 1]
        self.has_hoof = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/exo_left_hoof_ball') >= 0
        self.mechanism = (HUT_HINGE_HOOF_MECHANISM if self.has_hoof
                          else (HUT_HINGE_AFT_MECHANISM if self.has_hinge
                                else (HUT_BALL_AFT_MECHANISM
                                      if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'robot/exo_left_aft_plss_brace') >= 0
                                      else (HUT_RING_CLEVIS_MECHANISM
                                            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'robot/exo_hut_ring_left') >= 0
                                            else (WAIST_CLEVIS_MECHANISM
                                                  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'robot/exo_waist_belt') >= 0
                                                  else MECHANISM)))))
        self.waist_roll_qpos = int(model.joint('robot/WaistRoll').qposadr[0])
        self.ankle_roll_dof = [int(model.joint(f'robot/{side.title()}AnkleRoll').dofadr[0]) for side in STRUTS]
        self.ankle_roll_axis_sign = qpos.new_tensor([
            1. if float(model.jnt_axis[model.joint(f'robot/{side.title()}AnkleRoll').id, 0]) >= 0. else -1.
            for side in STRUTS])
        self.position_pd = bool(np.all(model.actuator_biastype[self.slide_actuators] == mujoco.mjtBias.mjBIAS_AFFINE))
        self.load = [int(model.sensor(f'robot/balance_{side}_load').adr[0]) for side in STRUTS]
        self.human_actuators = worlds.device.human_actuators
        human_ids = worlds.device.human_actuator_ids
        self.leg_columns = np.flatnonzero(np.isin(human_ids, lower_leg_actuators(model))).tolist()
        self.human_gear = torch.tensor(model.actuator_gear[human_ids, 0], dtype=qpos.dtype, device=qpos.device)
        limits = np.abs(model.actuator_forcerange[human_ids]).max(axis=1)
        self.human_torque_limits = torch.tensor(limits, dtype=qpos.dtype, device=qpos.device) * self.human_gear.abs()
        self.stance_transition = False
        self.posture_balance = False
        self.roll_balance = False
        self.vertical_support = False
        self.load_sharing = False
        self.reward_cfg = None
        self.commands = None
        self.policy_io = None
        self.assist_drop_prob = 0.
        self.force_scale = qpos.new_ones((worlds.count, 2))
        self._drop_generator = torch.Generator(device=qpos.device).manual_seed(torch.initial_seed() ^ 0x45584F44)
        self.peak_force_n = qpos.new_zeros((worlds.count, 2))
        self.last_request = torch.zeros_like(self.peak_force_n)
        self.positive_work_j = torch.zeros_like(self.peak_force_n)
        if self.has_hinge:
            self.last_hinge_request = torch.zeros_like(self.peak_force_n)
            self.peak_hinge_torque_nm = torch.zeros_like(self.peak_force_n)
            self.hinge_positive_work_j = torch.zeros_like(self.peak_force_n)
            self.hinge_absolute_work_j = torch.zeros_like(self.peak_force_n)
        self.human_positive_work_j = qpos.new_zeros(worlds.count)
        self.human_absolute_work_j = torch.zeros_like(self.human_positive_work_j)
        self.human_metabolic_proxy_j = torch.zeros_like(self.human_positive_work_j)
        self.assist_phase = qpos.new_zeros(worlds.count, dtype=torch.long)
        self.phase_force_impulse = torch.zeros_like(self.peak_force_n)
        self.phase_support_seconds = torch.zeros_like(self.peak_force_n)
        self.landing_strut_absolute_work_j = torch.zeros_like(self.peak_force_n)
        self.push_strut_absolute_work_j = torch.zeros_like(self.peak_force_n)
        self.leg_torque_squared_integral = torch.zeros_like(self.human_positive_work_j)
        self.peak_leg_torque_nm = torch.zeros_like(self.human_positive_work_j)
        self.supported_work_j = torch.zeros_like(self.human_positive_work_j)
        self.efficiency_rate = torch.zeros_like(self.human_positive_work_j)
        self.sample_seconds = 0.

    @property
    def policy_control(self):
        return self.policy_io is not None

    def bind(self, commands, *, policy_io=None):
        if self.position_pd != (policy_io == POLICY_IO):
            raise ValueError('外骨骼模型的执行器和策略控制模式不一致')
        if self.has_hinge != (policy_io in HINGE_CONTROL_REVISIONS):
            raise ValueError('外骨骼矢状面铰链模型和策略控制模式不一致')
        self.commands = commands
        self.policy_io = policy_io
        if policy_io in DIRECT_FORCE_REVISIONS:
            model = self.worlds.model
            self.guard_sensor_columns = self.nominal_length.new_tensor([
                [[int(model.sensor(f'robot/exo_{side}_{field}').adr[0]) + offset
                  for offset in range(3)]
                 for field in ('pack_pos', 'boot_pos', 'ankle_pos', 'ankle_axis')]
                for side in STRUTS], dtype=torch.long)
            self.ankle_roll_qpos = [int(model.joint(f'robot/{side.title()}AnkleRoll').qposadr[0])
                                   for side in STRUTS]
            columns = []
            for side in STRUTS:
                sensor = model.sensor(f'robot/exo_{side}_support_contacts')
                start, width = int(sensor.adr[0]), int(sensor.dim[0])
                columns.append(list(range(start, start + width)))
            self.guard_contact_columns = self.nominal_length.new_tensor(columns, dtype=torch.long)

    def enable_assist_drop(self, prob):
        if not (isfinite(prob) and 0. <= prob <= 1.):
            raise ValueError('assist_drop_prob must be in [0, 1]')
        self.assist_drop_prob = float(prob)
        self._sample_force_scale(None)

    def _sample_force_scale(self, mask):
        if self.assist_drop_prob <= 0.:
            if mask is None:
                self.force_scale.fill_(1.)
            else:
                self.force_scale[mask] = 1.
            return
        dev = self.force_scale.device
        n = self.worlds.count
        drop = torch.rand(n, device=dev, generator=self._drop_generator) < self.assist_drop_prob
        zero = torch.rand(n, device=dev, generator=self._drop_generator) < .5
        partial = .25 + .50 * torch.rand((n, 2), device=dev, generator=self._drop_generator)
        sampled = torch.where(drop[:, None], torch.where(zero[:, None], 0., partial), 1.)
        if mask is None:
            self.force_scale.copy_(sampled)
        else:
            torch.where(mask[:, None], sampled, self.force_scale, out=self.force_scale)

    def policy_actions(self, actions):
        if not self.policy_control:
            return self.commands(actions)
        width = len(self.actuators)
        human = self.commands(actions[:, :-width])
        return torch.cat((human, actions[:, -width:]), dim=-1)

    def observation(self, state):
        if not self.policy_control:
            return state
        data = self.worlds.sim.data
        length = self.nominal_length + data.sensordata[:, self.stroke]
        span = (self.max_length_m - self.min_length_m).clamp_min(1e-3)
        features = torch.stack((
            (length - self.min_length_m) / span,
            data.actuator_velocity[:, self.slide_actuators],
            data.actuator_force[:, self.slide_actuators] / MAX_FORCE_N,
            self.last_request,
            (data.sensordata[:, self.load] / self.worlds.device.weight).clamp(0., 4.),
        ), dim=-1)
        out = torch.cat((state, features.flatten(1)), dim=-1)
        if self.policy_io in (IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO):
            pack, boot, _, _ = data.sensordata[:, self.guard_sensor_columns].unbind(dim=2)
            vec = boot - pack  # (W, 2, 3)
            heading = self.commands.heading[:, None, :]  # (W, 1, 2)
            fwd = (vec[..., :2] * heading).sum(-1)
            down = (-vec[..., 2]).clamp_min(.10)
            strut_pitch = torch.atan2(fwd, down)  # (W, 2): <0 pushing forward, >0 braking
            slope = self.commands.terrain_slope()[:, None]  # (W, 1)
            out = torch.cat((out, slope, strut_pitch), dim=-1)
        if self.policy_io == IMPEDANCE_HINGE_FORCE_IO:
            hinge_features = torch.stack((
                data.sensordata[:, self.hinge_pos],
                data.sensordata[:, self.hinge_vel] / 5.,
                data.actuator_force[:, self.hinge_actuators] / DEFAULT_HINGE_TORQUE_NM,
                self.last_hinge_request,
            ), dim=-1)
            out = torch.cat((out, hinge_features.flatten(1)), dim=-1)
        return out

    def controls(self, actions=None):
        command = self.commands
        if command is None:
            raise RuntimeError('Bind the exoskeleton to the jump phase controller before stepping')
        if self.policy_control:
            expected_width = len(self.actuators)
            if actions is None or actions.shape != (self.worlds.count, expected_width):
                raise ValueError(f'每个环境必须提供 {expected_width} 维外骨骼动作')
            slide_actions = actions[:, :STRUT_ACTIONS]
            request = (slide_actions.clamp(-1., 1.) + 1.) * .5
            if not self.position_pd:
                if self.policy_io in (IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO):
                    data = self.worlds.sim.data
                    weight = self.worlds.device.weight
                    load_gate = ((data.sensordata[:, self.load] / weight - .04) / .16).clamp(0., 1.)
                    length = self.nominal_length + data.sensordata[:, self.stroke]
                    travel_end = ((self.max_length_m - length) / .04).clamp(0., 1.)
                    compression = (.97 * self.nominal_length - length).clamp_min(0.)
                    v_slide = data.actuator_velocity[:, self.slide_actuators]
                    # Automatic grounded SLIP spring-damper catch when moving + full policy
                    # thrust modulation up to 500 N per strut; zeroed smoothly when airborne or 0 N.
                    moving = (command.target_speed / .25).clamp(0., 1.)[:, None]
                    slip_comp = (compression - .02).clamp_min(0.)
                    f_passive = .45 * moving * (IMPEDANCE_K_N_PER_M * slip_comp
                                                + IMPEDANCE_C_NS_PER_M * (-v_slide).clamp_min(0.))
                    f_active = request * (IMPEDANCE_K_N_PER_M * compression
                                          + IMPEDANCE_C_NS_PER_M * (-v_slide).clamp_min(0.)
                                          + IMPEDANCE_F_BASE_N)
                    f_cmd = f_passive + f_active
                    f_cap = self.force_n * self.force_scale
                    applied = (f_cmd * (f_cap / MAX_FORCE_N) * load_gate * travel_end).clamp_min(0.).minimum(f_cap)
                    if self.mechanism in HUT_MECHANISMS:
                        # Strictly zero strut output at stationary 0.0 m/s command so the
                        # astronaut stands calmly instead of bouncing on zero-cost struts.
                        move_gate = (command.target_speed / .15).clamp(0., 1.)[:, None]
                        applied = applied * move_gate
                        # Closed-loop HUT WaistRoll safety guard: full 500 N/strut whenever
                        # |WaistRoll| <= 7 deg; smoothly tapers unilateral differential toward
                        # 280 N only if a strut pushes further into an excessive >7 deg roll.
                        w_roll = data.qpos[:, self.waist_roll_qpos]
                        worsening = torch.stack((w_roll, -w_roll), dim=-1)
                        excess = ((worsening - radians(7.)) / radians(6.)).clamp(0., 1.)
                        diff_cap = (500. - 220. * excess) * (f_cap / MAX_FORCE_N)
                        applied = applied.minimum(applied.flip(-1) + diff_cap)
                    if self.mechanism in HUT_AFT_MECHANISMS:
                        # Closed-loop Ankle Inversion/Eversion Force Governor:
                        # Uses real-time AnkleRoll angular velocity (catches active eversion
                        # immediately without false-triggering on static lunar side-slopes) +
                        # grounded boot load COP reserve + mechanical joint-stop protection.
                        pack, boot, ankle, axis = data.sensordata[:, self.guard_sensor_columns].unbind(dim=2)
                        u_boot = boot - pack
                        u_boot = u_boot / torch.linalg.vector_norm(u_boot, dim=-1, keepdim=True).clamp_min(1e-6)
                        tau_per_n = (torch.cross(boot - ankle, u_boot, dim=-1) * axis).sum(-1) * self.ankle_roll_axis_sign
                        roll = data.qpos[:, self.ankle_roll_qpos]
                        roll_vel = data.qvel[:, self.ankle_roll_dof]
                        tau_dir = tau_per_n.sign()
                        # Active eversion velocity guard: full assist when roll velocity into
                        # eversion <= 20 deg/s; smoothly throttles to 0 by 85 deg/s.
                        worsening_vel = roll_vel * tau_dir
                        vel_margin = ((radians(85.) - worsening_vel) / radians(65.)).clamp(0., 1.)
                        # Joint-stop guard: allows static terrain-matching roll on side-slopes
                        # (<= 7 deg), but cuts force smoothly if pushed past 7..8.8 deg into stop.
                        worsening_pos = (roll + .04 * roll_vel) * tau_dir
                        stop_margin = ((radians(8.8) - worsening_pos) / radians(1.8)).clamp(0., 1.)
                        margin = vel_margin.minimum(stop_margin)
                        upward_n = data.sensordata[:, self.load].clamp_min(0.)
                        # Grounded sole width COP reserve (5.5 cm margin) + pressurized boot reserve (18 Nm)
                        max_safe_nm = (18. + .055 * upward_n) * margin
                        f_safe_ankle = max_safe_nm / tau_per_n.abs().clamp_min(.02)
                        applied = applied.minimum(f_safe_ankle)
                    self.last_request.copy_(request)
                    if self.policy_io == IMPEDANCE_HINGE_FORCE_IO:
                        hinge_req = actions[:, STRUT_ACTIONS:STRUT_ACTIONS + HINGE_ACTIONS].clamp(-1., 1.)
                        tau_cap = self.hinge_torque_nm * self.force_scale
                        hinge_pos = data.sensordata[:, self.hinge_pos]
                        hinge_vel = data.sensordata[:, self.hinge_vel]
                        tau_raw = hinge_req * tau_cap
                        # Grounded-dominant bidirectional sagittal torque (100% grounded, up to 20%
                        # gentle swing-leg protraction/retraction in flight; zero at stationary 0 m/s).
                        hinge_gate = (.20 + .80 * load_gate) * move_gate
                        # Electric motor back-EMF torque-speed envelope (full up to 4 rad/s, 0 at 8 rad/s).
                        motoring_vel = (hinge_vel * tau_raw.sign()).clamp_min(0.)
                        speed_margin = ((8.0 - motoring_vel) / 4.0).clamp(0., 1.)
                        # Soft stop taper 10 deg before mechanical hinge limits.
                        lo_margin = ((hinge_pos - self.hinge_min_rad) / radians(10.)).clamp(0., 1.)
                        hi_margin = ((self.hinge_max_rad - hinge_pos) / radians(10.)).clamp(0., 1.)
                        hinge_stop = torch.where(tau_raw >= 0., hi_margin, lo_margin)
                        applied_hinge = tau_raw * hinge_gate * speed_margin * hinge_stop
                        self.last_hinge_request.copy_(hinge_req)
                        return torch.cat((applied, applied_hinge), dim=-1)
                    return applied
                requested = self.force_n * self.force_scale * request
                self.last_request.copy_(requested / MAX_FORCE_N)
                if self.policy_io == ACTION_FORCE_IO:
                    return requested
        data = self.worlds.sim.data
        weight = self.worlds.device.weight
        load = ((data.sensordata[:, self.load] / weight - .06) / .24).clamp(0., 1.)
        length = self.nominal_length + data.sensordata[:, self.stroke]
        travel = ((length - MIN_LENGTH_M) / .045).clamp(0., 1.)
        travel *= ((MAX_LENGTH_M - length) / .055).clamp(0., 1.)
        speed = data.actuator_velocity[:, self.slide_actuators]
        extension = (speed >= 0.) * ((speed + .05) / .16).clamp(0., 1.)
        if self.policy_io == POLICY_TIMED_FORCE_IO:
            gate = load * travel * extension
        else:
            supported = (command.stage == command.EXTEND) & (command.ground_frames > 0)
            phase = (command.age / .045).clamp(0., 1.)
            phase *= ((command.extension_duration() - command.age) / .055).clamp(0., 1.)
            gate = (phase * supported)[:, None] * load * travel * extension
        if self.policy_control:
            if self.position_pd:
                target = (length + PD_MAX_ADVANCE_M * request).clamp(MIN_LENGTH_M, MAX_LENGTH_M)
                self.last_request.copy_((target - MIN_LENGTH_M) / (MAX_LENGTH_M - MIN_LENGTH_M))
                applied = torch.where((gate > 0.) & (request > 0.) & (self.force_n > 0.),
                                      length + gate * (target - length), PD_RELEASE_LENGTH_M)
                return applied - self.nominal_length
            return requested * gate
        return self.force_n * gate

    def _action_force_cost(self):
        data = self.worlds.sim.data
        force = data.actuator_force[:, self.slide_actuators]
        pack, boot, ankle, axis = data.sensordata[:, self.guard_sensor_columns].unbind(dim=2)
        direction = torch.nn.functional.normalize(boot - pack, dim=-1)
        moment_per_newton = (torch.linalg.cross(boot - ankle, direction) * axis).sum(-1)
        contacts = data.sensordata[:, self.guard_contact_columns].reshape(
            self.worlds.count, 2, GUARD_CONTACTS_PER_FOOT, -1)
        normal_force = contacts[..., 1].clamp_min(0.)
        valid = (contacts[..., 0] > 0.) & (normal_force > 0.)
        normal_force = torch.where(valid, normal_force, 0.)
        loaded = normal_force.sum(-1) > .06 * self.worlds.device.weight
        if self.vertical_support:
            loaded = data.sensordata[:, self.load] > .06 * self.worlds.device.weight
        normals = contacts[..., 7:10]
        normals = torch.where(normals[..., 2:] < 0., -normals, normals)
        contact_arm = (torch.linalg.cross(contacts[..., 4:7] - ankle[:, :, None], normals)
                       * axis[:, :, None]).sum(-1)
        opposing_arm = contact_arm * -moment_per_newton.sign()[..., None]
        # Estimate how much more opposing torque the ground could provide
        # by shifting pressure from its current center to the loaded edge.
        best_arm = opposing_arm.masked_fill(~valid, -torch.inf).amax(-1)
        best_arm = torch.where(valid.any(-1), best_arm, 0.)
        reserve = SUPPORT_MOMENT_FRACTION * (
            (best_arm[..., None] - opposing_arm) * normal_force).sum(-1).clamp_min(0.)
        if self.vertical_support:
            # Normal-force magnitudes can be large for nearly opposed contacts.
            # Limit the reserve estimate to the measured upward boot reaction.
            lateral_pressure = (normal_force * torch.linalg.vector_norm(normals[..., :2], dim=-1)).sum(-1)
            torch.maximum(self.peak_horizontal_normal_sum_n, lateral_pressure, out=self.peak_horizontal_normal_sum_n)
            upward = data.sensordata[:, self.load]
            reserve *= (upward / normal_force.sum(-1).clamp_min(1e-6)).clamp(0., 1.)
        if self.mechanism in HUT_AFT_MECHANISMS:
            # Guard active eversion velocity and mechanical stop (>7 deg) without
            # penalizing flat-footed static side-slope compliance on rough lunar terrain.
            roll = data.qpos[:, self.ankle_roll_qpos]
            roll_vel = data.qvel[:, self.ankle_roll_dof]
            tau_dir = (moment_per_newton * self.ankle_roll_axis_sign).sign()
            worsening_vel = roll_vel * tau_dir
            vel_margin = ((radians(85.) - worsening_vel) / radians(65.)).clamp(0., 1.)
            worsening_pos = roll * tau_dir
            stop_margin = ((radians(8.8) - worsening_pos) / radians(1.8)).clamp(0., 1.)
            moment_limit = (18. + reserve) * vel_margin.minimum(stop_margin)
        else:
            roll = data.qpos[:, self.ankle_roll_qpos]
            increasing_tilt = roll * moment_per_newton > 0.
            margin = ((ANKLE_ROLL_GUARD_STOP - roll.abs()) /
                      (ANKLE_ROLL_GUARD_STOP - ANKLE_ROLL_GUARD_START)).clamp(0., 1.)
            moment_limit = (ANKLE_ROLL_MOMENT_LIMIT_NM + reserve) * torch.where(increasing_tilt, margin, 1.)
        excess_moment = (force * moment_per_newton.abs() - moment_limit).clamp_min(0.)
        unloaded_ratio = (force / MAX_FORCE_N) * ~loaded
        unloaded_force = unloaded_ratio.square().sum(-1)
        unloaded_linear = ((self.reward_cfg or {}).get('unloaded_strut_force_linear', 0.)
                           * unloaded_ratio.sum(-1))
        unsupported_roll = (excess_moment / ANKLE_ROLL_MOMENT_LIMIT_NM).clamp_max(3.).square().mean(-1)
        cost = (ACTION_FORCE_COSTS['unloaded_strut_force'] * unloaded_force
                + unloaded_linear
                + ACTION_FORCE_COSTS['unsupported_ankle_moment'] * unsupported_roll)
        if self.has_hinge:
            # Mild regularization against wasted airborne hinge torque while allowing
            # purposeful swing-leg alignment and full grounded sagittal propulsion/braking.
            hinge_unloaded = (data.actuator_force[:, self.hinge_actuators] / DEFAULT_HINGE_TORQUE_NM) * ~loaded
            cost = cost + .4 * hinge_unloaded.square().sum(-1)
        return cost

    def reset_controls(self, mask):
        # Reset the actuator command before forwarding the new episode pose.
        ctrl = self.worlds.sim.data.ctrl
        current = ctrl[:, self.slide_actuators]
        released = PD_RELEASE_LENGTH_M - self.nominal_length if self.position_pd else 0.
        ctrl[:, self.slide_actuators] = released if mask is None else torch.where(mask[:, None], released, current)
        if self.has_hinge:
            ctrl[:, self.hinge_actuators] = (
                0. if mask is None else torch.where(mask[:, None], 0., ctrl[:, self.hinge_actuators]))
        rows = slice(None) if mask is None else mask
        initial = ((self.nominal_length - MIN_LENGTH_M) / (MAX_LENGTH_M - MIN_LENGTH_M)
                   if self.position_pd else 0.)
        self.last_request[rows] = initial
        if self.has_hinge:
            self.last_hinge_request[rows] = 0.
        self.assist_phase[rows] = 0
        self.phase_force_impulse[rows] = 0.
        self.phase_support_seconds[rows] = 0.
        if self.load_sharing:
            self.phase_balance[rows] = 0.

        if self.vertical_support:
            self.unloaded_steps[rows] = 0
            self.remaining_touchdown_steps[rows] = 0
            self.touchdown_impulse_ns[rows] = 0.

        if self.roll_balance:
            self.filtered_roll[rows] = 0.
        if self.posture_balance:
            self.push_boot_impulse[rows] = 0.
        self._sample_force_scale(mask)

    def enable_load_sharing(self, *, vertical_support=False, posture_balance=False,
                            stance_transition=False, roll_balance=False, reward_overrides=None):
        if self.policy_io not in DIRECT_FORCE_REVISIONS:
            raise ValueError('Load sharing requires direct strut action control')
        self.load_sharing = True
        base_cfg = (STANCE_TRANSITION_REWARD if stance_transition else
                    POSTURE_SUPPORT_REWARD if posture_balance else
                    VERTICAL_SUPPORT_REWARD if vertical_support else
                    LOAD_SHARING_REWARD)
        self.reward_cfg = {**base_cfg, **(reward_overrides or {})}
        self.phase_balance = torch.zeros_like(self.efficiency_rate)
        self.phase_impulse_ns = self.peak_force_n.new_zeros((self.worlds.count, 2, 2))

        self.stance_transition = stance_transition
        self.posture_balance = posture_balance
        self.roll_balance = posture_balance or roll_balance
        if posture_balance:
            self.push_boot_impulse = torch.zeros_like(self.peak_force_n)
        if self.roll_balance:
            if not vertical_support:
                raise ValueError('Roll balance requires vertical support')
            self.filtered_roll = torch.zeros_like(self.peak_force_n)
            self.waist_roll_qadr = int(self.worlds.model.joint('robot/WaistRoll').qposadr[0])
            self.roll_deadband = self.filtered_roll.new_tensor([
                radians(POSTURE_SUPPORT_REWARD['pack_roll_free_deg']),
                radians(POSTURE_SUPPORT_REWARD['waist_roll_free_deg'])])
            self.roll_weights = self.filtered_roll.new_tensor([
                POSTURE_SUPPORT_REWARD['pack_roll_weight_per_rad2'],
                POSTURE_SUPPORT_REWARD['waist_roll_weight_per_rad2']])
        self.vertical_support = vertical_support
        if vertical_support:
            model = self.worlds.model
            self.reaction_columns = [[int(model.sensor(f'contacts/robot/{side}_sole_reaction').adr[0]) + i
                                      for i in range(3)] for side in STRUTS]
            self.distance_columns = [int(model.sensor(f'contacts/robot/{side}_sole_min_dist').adr[0])
                                     for side in STRUTS]
            self.unloaded_steps = torch.zeros_like(self.peak_force_n, dtype=torch.long)
            self.remaining_touchdown_steps = torch.zeros_like(self.unloaded_steps)
            self.touchdown_impulse_ns = torch.zeros_like(self.peak_force_n)
            self.completed_touchdowns = torch.zeros_like(self.unloaded_steps)
            self.completed_touchdown_impulse_ns = torch.zeros_like(self.peak_force_n)
            self.peak_upward_reaction_n = torch.zeros_like(self.peak_force_n)
            self.peak_horizontal_normal_sum_n = torch.zeros_like(self.peak_force_n)
            self.peak_lateral_force_n = torch.zeros_like(self.peak_force_n)
            self.peak_penetration_m = torch.zeros_like(self.peak_force_n)

    def _touchdown_cost(self, loaded, dt):
        data = self.worlds.sim.data
        reaction = data.sensordata[:, self.reaction_columns]
        weight = self.worlds.device.weight
        costs = VERTICAL_SUPPORT_REWARD
        command = self.commands
        descending = data.sensordata[:, command.com_velocity_xy.stop] < 0.
        landing = (command.stage == command.COMPRESS) | (descending & (command.stage != command.EXTEND))
        start = loaded & (self.unloaded_steps >= 2) & landing[:, None] & (self.remaining_touchdown_steps == 0)
        self.remaining_touchdown_steps.masked_fill_(start, round(costs['touchdown_window_s'] / dt))
        self.touchdown_impulse_ns.masked_fill_(start, 0.)
        threshold = costs['touchdown_impulse_bw_s'] * weight
        excess_before = (self.touchdown_impulse_ns - threshold).clamp_min(0.)
        active = self.remaining_touchdown_steps > 0
        self.touchdown_impulse_ns.add_(reaction[..., 2].clamp_min(0.) * active * dt)
        excess = (self.touchdown_impulse_ns - threshold).clamp_min(0.) - excess_before
        completed = self.remaining_touchdown_steps == 1
        self.completed_touchdowns.add_(completed)
        self.completed_touchdown_impulse_ns.add_(self.touchdown_impulse_ns * completed)
        self.remaining_touchdown_steps.sub_(active.long())
        self.unloaded_steps.copy_(torch.where(loaded, 0, (self.unloaded_steps + 1).clamp_max(2)))
        torch.maximum(self.peak_upward_reaction_n, reaction[..., 2], out=self.peak_upward_reaction_n)
        lateral = torch.linalg.vector_norm(reaction[..., :2], dim=-1)
        torch.maximum(self.peak_lateral_force_n, lateral, out=self.peak_lateral_force_n)
        depth = -data.sensordata[:, self.distance_columns]
        torch.maximum(self.peak_penetration_m, depth, out=self.peak_penetration_m)
        return costs['touchdown_impulse_weight'] * excess.sum(-1) / (weight * dt)

    def _regional_upward_loads(self):
        data = self.worlds.sim.data
        contacts = data.sensordata[:, self.guard_contact_columns].reshape(
            self.worlds.count, 2, GUARD_CONTACTS_PER_FOOT, 13)
        # This sensor reports boot -> terrain. Negate its world-Z force to get
        # the force on the boot, including friction on sloping terrain.
        normal, tangent = contacts[..., 7:10], contacts[..., 10:13]
        bitangent_z = normal[..., 0] * tangent[..., 1] - normal[..., 1] * tangent[..., 0]
        upward = -(contacts[..., 1] * normal[..., 2] + contacts[..., 2] * tangent[..., 2]
                   + contacts[..., 3] * bitangent_z) / self.worlds.device.weight
        heel, toe = self.commands.sole_probes().unbind(2)
        half_axis = (toe - heel) * .5
        longitudinal = ((contacts[..., 4:7] - ((heel + toe) * .5).unsqueeze(2))
                        * half_axis.unsqueeze(2)).sum(-1)
        boundary = (.33 * half_axis.square().sum(-1)).unsqueeze(2)
        upward = upward * (contacts[..., 0] > 0)
        return torch.stack(((upward * (longitudinal < -boundary)).sum(-1),
                            (upward * (longitudinal > boundary)).sum(-1)), dim=-1)

    def _stance_transition_rate(self):
        data = self.worlds.sim.data
        cfg = STANCE_TRANSITION_REWARD
        vz = data.sensordata[:, self.commands.com_velocity_xy.stop] / cfg['vertical_speed_mps']
        load = (data.sensordata[:, self.load] / self.worlds.device.weight).clamp_min(0.)
        # Each foot earns landing credit from its weaker region, only while
        # actually carrying the descending body. Heel lift is free during rise.
        regions = self._regional_upward_loads().amin(-1).minimum(.5 * load)
        flat = (regions / cfg['regional_load_bw']).clamp(0., 1.)
        braking = flat.mean(-1) * (-vz).clamp(0., 1.)
        # Reward the opportunity to assist: rise, support and axial extension
        # must coincide. Actuator force and signed work are absent from the score.
        extension = (data.actuator_velocity[:, self.slide_actuators] / cfg['rise_extension_mps']).clamp(0., 1.)
        available = extension * (load / cfg['rise_support_bw']).clamp_max(1.)
        rising = (.5 * available.mean(-1) + .5 * available.amin(-1)) * vz.clamp(0., 1.)
        return cfg['braking_sole_weight'] * braking + cfg['rising_support_weight'] * rising

    def _persistent_roll_cost(self, dt):
        data = self.worlds.sim.data
        w, x, y, z = data.sensordata[:, self.commands.torso_quat].unbind(-1)
        roll = torch.atan2(2. * (w*x + y*z), 1. - 2. * (x.square() + y.square()))
        angles = torch.stack((roll, data.qpos[:, self.waist_roll_qadr]), dim=-1)
        alpha = -expm1(-dt / POSTURE_SUPPORT_REWARD['posture_filter_seconds'])
        self.filtered_roll.lerp_(angles, alpha)
        persistent = (self.filtered_roll.abs() - self.roll_deadband).clamp_min(0.)
        # Charge the current tilt only while it reinforces the sustained bias.
        same_direction = (angles * self.filtered_roll.sign() - self.roll_deadband).clamp_min(0.)
        return (self.roll_weights * persistent * same_direction).sum(-1)

    def _load_sharing_rate(self, phase, loaded, dt):
        costs = self.reward_cfg or (VERTICAL_SUPPORT_REWARD if self.vertical_support else LOAD_SHARING_REWARD)
        weight = self.worlds.device.weight
        data = self.worlds.sim.data
        footprint = (data.sensordata[:, self.load] / weight).clamp_min(0.)
        supported = loaded & (phase != 0)[:, None]
        impulse = data.actuator_force[:, self.slide_actuators] * supported * dt
        changed = phase != self.assist_phase
        self.phase_force_impulse.masked_fill_(changed[:, None], 0.)
        self.phase_support_seconds.masked_fill_(changed[:, None], 0.)
        self.phase_force_impulse.add_(impulse)
        # A load spike cannot substitute for sustained support.
        self.phase_support_seconds.add_(footprint.clamp_max(1.) * supported * dt)
        self.phase_impulse_ns[:, 0].add_(impulse * (phase == 1)[:, None])
        self.phase_impulse_ns[:, 1].add_(impulse * (phase == 2)[:, None])
        impedance_io = self.policy_io in (IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO)
        band = .10 if impedance_io else costs['bilateral_force_share_deadband']
        strut_bal_w = 22. if impedance_io else costs['strut_impulse_balance']
        boot_bal_w = 10. if impedance_io else costs['boot_support_balance']
        balance = (strut_bal_w * excess_side_difference(self.phase_force_impulse, band) / weight
                   + boot_bal_w * excess_side_difference(self.phase_support_seconds, band))
        if self.posture_balance:
            push = phase == 2
            self.push_boot_impulse.masked_fill_(changed[:, None], 0.)
            self.push_boot_impulse.add_(footprint * supported * push[:, None] * dt)
            push_strut_w = 14. if impedance_io else POSTURE_SUPPORT_REWARD['push_extra_strut_impulse_balance']
            balance += POSTURE_SUPPORT_REWARD['push_boot_vertical_impulse_balance'] * excess_side_difference(
                self.push_boot_impulse, band)
            balance += push_strut_w * push * excess_side_difference(
                self.phase_force_impulse, band) / weight
        # Charge the change within each phase. Later support on the opposite
        # side can recover the charge for an initially asymmetric touchdown.
        rate = (balance - self.phase_balance.masked_fill(changed, 0.)) / dt
        self.phase_balance.copy_(balance)
        excess_load = (footprint.amax(-1) - costs['single_foot_overload_bw']).clamp_min(0.)
        if self.roll_balance:
            rate = rate + self._persistent_roll_cost(dt)
        if self.vertical_support:
            rate = rate + self._touchdown_cost(loaded, dt)
        return rate + costs['single_foot_overload_weight'] * excess_load

    def _bilateral_assist_cost(self, strut_power, dt):
        command = self.commands
        data = self.worlds.sim.data
        loaded = data.sensordata[:, self.load] > .06 * self.worlds.device.weight
        vz = data.sensordata[:, command.com_velocity_xy.stop]
        landing = (command.stage == command.COMPRESS) | (
            (command.stage == command.FLIGHT) & (vz < 0.) & loaded.any(-1))
        pushing = command.stage == command.EXTEND
        phase = landing.long() + 2 * pushing.long()
        work = strut_power.abs() * loaded * dt
        self.landing_strut_absolute_work_j.add_(work * landing[:, None])
        self.push_strut_absolute_work_j.add_(work * pushing[:, None])
        if self.load_sharing:
            rate = self._load_sharing_rate(phase, loaded, dt)
            self.assist_phase.copy_(phase)
            return rate
        changed = phase != self.assist_phase
        costs = ACTION_FORCE_EFFICIENCY_REWARD

        # Compare both rods over the completed phase, allowing different
        # contact times. Brief or genuinely unilateral support is exempt.
        both_supported = (self.phase_support_seconds >= costs['bilateral_min_loaded_seconds']).all(-1)
        mean_force = self.phase_force_impulse / self.phase_support_seconds.clamp_min(dt)
        total_force = mean_force.sum(-1)
        disparity = (mean_force[:, 0] - mean_force[:, 1]).abs() / total_force.clamp_min(1.)
        excess = (disparity - costs['bilateral_force_share_deadband']).clamp_min(0.)
        duration = self.phase_support_seconds.amax(-1)
        rate = costs['bilateral_force_balance'] * excess.square() * (
            total_force / (2. * MAX_FORCE_N)) * (duration / dt)
        rate = rate * (changed & both_supported & (self.assist_phase != 0))

        self.phase_force_impulse.masked_fill_(changed[:, None], 0.)
        self.phase_support_seconds.masked_fill_(changed[:, None], 0.)
        supported = loaded & (phase != 0)[:, None]
        self.phase_force_impulse.add_(data.actuator_force[:, self.slide_actuators] * supported * dt)
        self.phase_support_seconds.add_(supported * dt)
        self.assist_phase.copy_(phase)
        return rate

    def observe(self, dt):
        self.sample_seconds += dt
        self._observe_tensors(dt)

    def _observe_tensors(self, dt):
        command = self.commands
        data = self.worlds.sim.data
        power = data.actuator_force * data.actuator_velocity
        strut_power = power[:, self.slide_actuators]
        exo_slide_power = strut_power.clamp_min(0.)
        if self.has_hinge:
            hinge_power = power[:, self.hinge_actuators]
            exo_hinge_power = hinge_power.clamp_min(0.)
            exo_power = exo_slide_power + exo_hinge_power
        else:
            hinge_power = None
            exo_hinge_power = None
            exo_power = exo_slide_power
        human_power = power[:, self.human_actuators]
        human_positive = human_power.clamp_min(0.).sum(-1)
        human_absolute = human_power.abs().sum(-1)
        torque = data.actuator_force[:, self.human_actuators] * self.human_gear
        ratio_squared = (torque / self.human_torque_limits).square()
        leg_torque = torque[:, self.leg_columns]

        # Retain the historical positive-work statistics for comparison.
        weight = self.worlds.device.weight
        load = ((data.sensordata[:, self.load] / weight - .08) / .30).clamp(0., 1.)
        rising = ((data.sensordata[:, command.com_velocity_xy.stop] - .05) / .30).clamp(0., 1.)
        supported_power = (exo_power * load).sum(-1) * rising
        if self.policy_io not in PHASE_FREE_FORCE_REVISIONS:
            supported_power = supported_power * (command.stage == command.EXTEND)
        power_scale = weight * 1.0  # lunar body weight transported at 1 m/s
        if self.policy_io in DIRECT_FORCE_REVISIONS:
            costs = self.reward_cfg if self.load_sharing else ACTION_FORCE_EFFICIENCY_REWARD
            if costs.get('cot_energy_scale', False):
                power_scale = weight * command.target_speed.clamp_min(.9)
            human_negative = (-human_power).clamp_min(0.).sum(-1)
            human_energy_rate = (costs['positive_work_cost'] * human_positive
                                 + costs['negative_work_cost'] * human_negative)
            self.human_metabolic_proxy_j.add_(human_energy_rate * dt)
            normalized_power = human_energy_rate / power_scale
            if not self.load_sharing:
                normalized_power = normalized_power.clamp_max(6. * costs['positive_work_cost'])
            human_power_cost = costs['human_metabolic_proxy_power'] * normalized_power
            mean_torque_w = costs['human_mean_squared_torque']
            peak_torque_w = costs['peak_leg_squared_torque']
        else:
            human_power_cost = EFFICIENCY_REWARD['human_absolute_power'] * (
                human_absolute / power_scale).clamp_max(6.)
            mean_torque_w = EFFICIENCY_REWARD['human_mean_squared_torque']
            peak_torque_w = EFFICIENCY_REWARD['peak_leg_squared_torque']
        self.efficiency_rate.copy_(
            -human_power_cost
            - mean_torque_w * ratio_squared.mean(-1)
            - peak_torque_w * ratio_squared[:, self.leg_columns].amax(-1)
        )
        if self.policy_io in DIRECT_FORCE_REVISIONS:
            self.efficiency_rate.sub_(self._action_force_cost())
            self.efficiency_rate.sub_(self._bilateral_assist_cost(strut_power, dt))
            assist_share_w = (self.reward_cfg or {}).get('assist_work_share_weight', 0.)
            if assist_share_w > 0.:
                if self.policy_io in (IMPEDANCE_FORCE_IO, IMPEDANCE_HINGE_FORCE_IO):
                    # Bilateral-coupled work share (gated to moving speeds so 0.0 m/s stays calm at 0 N).
                    moving_share = (command.target_speed / .25).clamp(0., 1.)
                    exo_push_sides = exo_power * load
                    exo_brake_raw = (-strut_power).clamp_min(0.)
                    if self.has_hinge:
                        exo_brake_raw = exo_brake_raw + (-hinge_power).clamp_min(0.)
                    if self.mechanism in HUT_MECHANISMS:
                        # Strongly reward bilateral symmetric propulsion & braking (min-side weighted)
                        # so both struts actively assist rather than one strut dominating.
                        exo_push_eff = .40 * exo_push_sides.sum(-1) + 1.20 * exo_push_sides.amin(-1)
                        exo_brake_sides = exo_brake_raw * load
                        exo_brake_eff = .45 * exo_brake_sides.sum(-1) + 1.10 * exo_brake_sides.amin(-1)
                    else:
                        exo_push_eff = .4 * exo_push_sides.sum(-1) + 1.2 * exo_push_sides.amin(-1)
                        exo_brake_sides = exo_brake_raw * load
                        exo_brake_eff = .4 * exo_brake_sides.sum(-1) + 1.2 * exo_brake_sides.amin(-1)
                    push_share = exo_push_eff / (exo_push_sides.sum(-1) + human_positive + .12 * weight)
                    falling = ((-data.sensordata[:, command.com_velocity_xy.stop] - .05) / .30).clamp(0., 1.)
                    brake_share = exo_brake_eff / (exo_brake_sides.sum(-1) + human_negative + .12 * weight)
                    push_power_bonus = (exo_push_eff / (.35 * weight)).clamp(0., 1.2)
                    self.efficiency_rate.add_(assist_share_w * moving_share * (
                        (push_share + .25 * push_power_bonus) * rising + .85 * brake_share * falling))
                    # Encourage bilateral synchronized foot loading during COM rise so left and
                    # right struts stay loaded and extend together rather than one foot lifting early.
                    any_loaded = (load.amax(-1) > .15).float()
                    rising_load_diff = (load[:, 0] - load[:, 1]).abs()
                    rising_sync_bonus = load.amin(-1)
                    self.efficiency_rate.add_(moving_share * rising * any_loaded * (
                        18. * rising_sync_bonus - 24. * rising_load_diff.square()))
                    both_loaded = (load.amin(-1) > .20).float()
                    cmd_asym = ((self.last_request[:, 0] - self.last_request[:, 1]).abs() - .08).clamp_min(0.)
                    self.efficiency_rate.sub_(20. * both_loaded * cmd_asym.square())
                    if self.has_hinge:
                        hinge_asym = ((self.last_hinge_request[:, 0] - self.last_hinge_request[:, 1]).abs() - .15).clamp_min(0.)
                        self.efficiency_rate.sub_(6. * both_loaded * hinge_asym.square())
                else:
                    exo_push = (exo_power * load).sum(-1)
                    push_share = exo_push / (exo_push + human_positive + .15 * weight)
                    exo_brake = ((-strut_power).clamp_min(0.) * load).sum(-1)
                    falling = ((-data.sensordata[:, command.com_velocity_xy.stop] - .05) / .30).clamp(0., 1.)
                    brake_share = exo_brake / (exo_brake + human_negative + .15 * weight)
                    self.efficiency_rate.add_(assist_share_w * (push_share * rising + .5 * brake_share * falling))
            if self.stance_transition:
                self.efficiency_rate.add_(self._stance_transition_rate())
        else:
            # Soften the share denominator so small oscillations earn little credit.
            work_share = supported_power / (exo_power.sum(-1) + human_positive + .20 * power_scale)
            useful_power = (supported_power / (.25 * power_scale)).clamp_max(1.5)
            self.efficiency_rate.add_(EFFICIENCY_REWARD['supported_exo_work_share'] * work_share
                                      + EFFICIENCY_REWARD['supported_exo_positive_power'] * useful_power)

        self.positive_work_j.add_(exo_slide_power * dt)
        if self.has_hinge:
            self.hinge_positive_work_j.add_(exo_hinge_power * dt)
            self.hinge_absolute_work_j.add_(hinge_power.abs() * dt)
            torch.maximum(self.peak_hinge_torque_nm,
                          data.actuator_force[:, self.hinge_actuators].abs(),
                          out=self.peak_hinge_torque_nm)
        self.supported_work_j.add_(supported_power * dt)
        self.human_positive_work_j.add_(human_positive * dt)
        self.human_absolute_work_j.add_(human_absolute * dt)
        self.leg_torque_squared_integral.add_(leg_torque.square().mean(-1) * dt)
        torch.maximum(self.peak_leg_torque_nm, leg_torque.abs().amax(-1), out=self.peak_leg_torque_nm)
        torch.maximum(self.peak_force_n, data.actuator_force[:, self.slide_actuators], out=self.peak_force_n)

    def report(self):
        exo_work = self.positive_work_j.sum(-1)
        if self.has_hinge:
            exo_work = exo_work + self.hinge_positive_work_j.sum(-1)
        total_positive = exo_work + self.human_positive_work_j
        share = exo_work / total_positive.clamp_min(1e-6)
        rms = (self.leg_torque_squared_integral / max(self.sample_seconds, 1e-6)).sqrt()
        return {
            'mechanism': self.mechanism,
            'control': self.policy_io or 'scheduled_force',
            'force_limit_per_strut_n': self.force_n,
            'peak_force_per_env_n': self.peak_force_n.tolist(),
            'strut_positive_work_per_env_j': self.positive_work_j.tolist(),
            **({'hinge_torque_limit_per_side_nm': self.hinge_torque_nm,
                'peak_hinge_torque_per_env_nm': self.peak_hinge_torque_nm.tolist(),
                'hinge_positive_work_per_env_j': self.hinge_positive_work_j.tolist(),
                'hinge_absolute_work_per_env_j': self.hinge_absolute_work_j.tolist(),
                'exo_total_positive_work_per_env_j': exo_work.tolist()}
               if self.has_hinge else {}),
            'human_joint_positive_work_per_env_j': self.human_positive_work_j.tolist(),
            'human_joint_absolute_work_per_env_j': self.human_absolute_work_j.tolist(),
            **({'human_joint_metabolic_proxy_per_env_j': self.human_metabolic_proxy_j.tolist(),
                'landing_strut_absolute_work_per_env_j': self.landing_strut_absolute_work_j.tolist(),
                'push_strut_absolute_work_per_env_j': self.push_strut_absolute_work_j.tolist()}
               if self.policy_io in DIRECT_FORCE_REVISIONS else {}),
            **({'landing_strut_impulse_per_env_ns': self.phase_impulse_ns[:, 0].tolist(),
                'push_strut_impulse_per_env_ns': self.phase_impulse_ns[:, 1].tolist()}
               if self.load_sharing else {}),
            **({'touchdown_window_s': VERTICAL_SUPPORT_REWARD['touchdown_window_s'],
                'completed_touchdowns_per_env_side': self.completed_touchdowns.tolist(),
                'sum_touchdown_upward_impulse_per_env_side_ns': self.completed_touchdown_impulse_ns.tolist(),
                'peak_upward_net_reaction_per_env_side_n': self.peak_upward_reaction_n.tolist(),
                'peak_horizontal_net_reaction_per_env_side_n': self.peak_lateral_force_n.tolist(),
                'peak_sum_horizontal_normal_components_top12_per_env_side_n': self.peak_horizontal_normal_sum_n.tolist(),
                'deepest_sole_terrain_contact_per_env_side_m': self.peak_penetration_m.tolist()}
               if self.vertical_support else {}),
            'strut_supported_work_per_env_j': self.supported_work_j.tolist(),
            'strut_share_of_positive_work_per_env': share.tolist(),
            'human_leg_torque_rms_per_env_nm': rms.tolist(),
            'human_leg_torque_peak_per_env_nm': self.peak_leg_torque_nm.tolist(),
            'mechanical_work_sample_hz': 60,
        }
