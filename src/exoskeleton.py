"""Two powered telescoping struts with pinned backpack and boot mounts."""
from math import expm1, isfinite, radians

import mujoco
import numpy as np
import torch

MAX_FORCE_N = 500.
DEFAULT_FORCE_N = MAX_FORCE_N
STRUTS = ('left', 'right')
STRUT_ACTIONS = len(STRUTS)
STRUT_OBSERVATIONS = 5 * len(STRUTS)
MOTOR_POLICY_IO = 'motor_force_v1'
FORCE_POLICY_IO = 'motor_force_v2'
POLICY_TIMED_FORCE_IO = 'policy_timed_force_v1'
ACTION_FORCE_IO = 'action_force_v1'
FORCE_CONTROL_REVISIONS = (FORCE_POLICY_IO, POLICY_TIMED_FORCE_IO, ACTION_FORCE_IO)
PHASE_FREE_FORCE_REVISIONS = (POLICY_TIMED_FORCE_IO, ACTION_FORCE_IO)
POLICY_IO = 'position_pd_v1'
INITIAL_ASSIST_N = 80.
PD_KP = 4000.  # N/m
PD_KD = 60.  # N·s/m
PD_MAX_ADVANCE_M = .15
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


def validate_policy_io(revision):
    if revision not in (None, MOTOR_POLICY_IO, POLICY_IO, *FORCE_CONTROL_REVISIONS):
        raise ValueError(f'未知的外骨骼控制版本：{revision}')
    return revision


def exo_initial_state(source, target, *, initial_command=None, replace_strut_io=False):
    """Warm-start a new actuator mode without interpreting old rod commands."""
    expanded = dict(source)
    for key, fill in (('mlp.0.weight', 0.), ('obs_normalizer._mean', 0.),
                      ('obs_normalizer._var', 1.), ('obs_normalizer._std', 1.)):
        old, template = source[key], target[key]
        if replace_strut_io and old.shape == template.shape:
            value = old.clone()
            # The fourth feature of each rod changes command units.
            value[..., -STRUT_OBSERVATIONS + 3::5] = fill
            expanded[key] = value
            continue
        if (template.shape[:-1] != old.shape[:-1]
                or template.shape[-1] != old.shape[-1] + STRUT_OBSERVATIONS):
            raise ValueError(f'外骨骼观测形状不兼容：{key}')
        value = torch.full_like(template, fill)
        value[..., :old.shape[-1]] = old
        expanded[key] = value
    if initial_command is not None:
        for key, fill in (('mlp.4.weight', 0.), ('mlp.4.bias', initial_command),
                          ('distribution.std_param', .25)):
            old, template = source[key], target[key]
            if replace_strut_io:
                old = old[:-STRUT_ACTIONS]
            if (template.shape[1:] != old.shape[1:]
                    or template.shape[0] != old.shape[0] + STRUT_ACTIONS):
                raise ValueError(f'外骨骼动作形状不兼容：{key}')
            value = torch.full_like(template, fill)
            value[:old.shape[0]] = old
            expanded[key] = value
    return expanded


def mount_names(side):
    return f'robot/exo_{side}_pack', f'robot/exo_{side}_boot'


MIN_LENGTH_M = .63
MAX_LENGTH_M = 1.22
TUBE_LENGTH_M = .63
MECHANISM = 'hooke_slide_ball_v3'
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
PIN_COLOR = [.80, .45, .14, 1.]
SILVER_COLOR = [.69, .74, .77, 1.]


def _geom(body, name, kind, color, mass=0., **shape):
    return body.add_geom(name=name, type=kind, rgba=color, mass=mass,
                         contype=0, conaffinity=0, group=2, **shape)


def _cylinder(body, name, start, end, radius, color, mass=0.):
    return _geom(body, name, mujoco.mjtGeom.mjGEOM_CYLINDER, color, mass,
                 fromto=[*start, *end], size=[radius, 0., 0.])


def _site(body, name, position):
    body.add_site(name=name, pos=position, size=[.012, .012, .012], group=4, rgba=[0., 0., 0., 0.])


def _boot_point(sole, offset):
    point = np.empty(3)
    mujoco.mju_rotVecQuat(point, np.asarray(offset, dtype=float), sole.quat)
    return sole.pos + point


def add_exoskeleton(spec, *, pd_force=None, guard_action_force=False, contact_tangent=False):
    """Build Hooke–prismatic–ball chains that close against the boots."""
    backpack = spec.geom('robot/backpack')
    mounts = {}
    for side, sign in (('left', 1.), ('right', -1.)):
        sole = spec.geom(f'robot/{side}_sole')
        upper_name, lower_name = mount_names(side)
        upper = backpack.pos + np.array([.46 * backpack.size[0],
                                         sign * (backpack.size[1] + .018), -.68 * backpack.size[2]])
        # Place the boot mount in the sole's own frame, including its toe-out yaw.
        boot_offset = np.array([-.21 * sole.size[0], sign * (sole.size[1] + .010),
                                -sole.size[2] + SOCKET_RADIUS_M])
        _site(backpack.parent, upper_name, upper)
        _site(sole.parent, lower_name, _boot_point(sole, boot_offset))
        mounts[side] = (upper, boot_offset, sign)

    reference = spec.compile()
    rest = mujoco.MjData(reference)
    mujoco.mj_kinematics(reference, rest)
    box, sphere = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE
    for side, (upper, boot_offset, sign) in mounts.items():
        name = f'robot/exo_{side}'
        upper_name, lower_name = mount_names(side)
        torso = backpack.parent
        sole = spec.geom(f'robot/{side}_sole')
        foot = sole.parent
        torso_id = reference.body(torso.name).id
        world_axis = rest.site(lower_name).xpos - rest.site(upper_name).xpos
        radius = float(np.linalg.norm(world_axis))
        local_axis = rest.xmat[torso_id].reshape(3, 3).T @ (world_axis / radius)
        rotation = np.empty(4)
        mujoco.mju_quatZ2Vec(rotation, local_axis)

        # Fixed lower-side bracket. The two pins follow different axes.
        outward = np.array([0., sign, 0.])
        _geom(torso, f'{name}_pack_plate', box, YOKE_COLOR,
              pos=upper - .021 * outward, size=[.034, .010, .032])
        _cylinder(torso, f'{name}_pack_boss', upper - .023 * outward,
                  upper + .008 * outward, .019, YOKE_COLOR)
        yoke = torso.add_body(name=f'{name}_yoke', pos=upper, quat=rotation)
        yoke.add_joint(name=f'{name}_hooke_a', type=mujoco.mjtJoint.mjJNT_HINGE,
                       axis=[1., 0., 0.], damping=.045, armature=.001)
        _cylinder(yoke, f'{name}_outer_pin', [-.027, 0., 0.], [.027, 0., 0.], .011, PIN_COLOR, .10)
        sleeve = yoke.add_body(name=f'{name}_sleeve')
        sleeve.add_joint(name=f'{name}_hooke_b', type=mujoco.mjtJoint.mjJNT_HINGE,
                         axis=[0., 1., 0.], damping=.045, armature=.001)
        _cylinder(sleeve, f'{name}_inner_pin', [0., -.022, 0.], [0., .022, 0.], .012, SILVER_COLOR)
        _cylinder(sleeve, f'{name}_outer_tube', [0., 0., .010], [0., 0., TUBE_LENGTH_M], .020, YOKE_COLOR, .30)
        _cylinder(sleeve, f'{name}_guide_collar', [0., 0., TUBE_LENGTH_M-.028],
                  [0., 0., TUBE_LENGTH_M], .023, SILVER_COLOR)
        shaft = sleeve.add_body(name=f'{name}_shaft', pos=[0., 0., radius])
        slide = f'{name}_slide'
        shaft.add_joint(name=slide, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0., 0., 1.],
                        limited=True, range=[MIN_LENGTH_M-radius, MAX_LENGTH_M-radius],
                        damping=.20, armature=.005)
        _cylinder(shaft, f'{name}_sliding_shaft', [0., 0., -TUBE_LENGTH_M], [0., 0., 0.], .010, SILVER_COLOR, .27)
        _geom(shaft, f'{name}_ball', sphere, PIN_COLOR, .06, size=[.015, 0., 0.])
        _site(shaft, f'{name}_ball_pin', [0., 0., 0.])

        # Two inclined lugs connect the boot-side plate to the round socket.
        # The socket is tangent to the sole plane and the ball is above it.
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
        if pd_force is None:
            actuator.gainprm = [1.] + [0.] * 9
            actuator.ctrlrange = [0., MAX_FORCE_N]
            actuator.forcerange = [0., MAX_FORCE_N]
        else:
            cap = validate_assist_force(pd_force)
            actuator.set_to_position(kp=PD_KP if cap else 0., kv=PD_KD if cap else 0.)
            actuator.ctrlrange = [PD_RELEASE_LENGTH_M - radius, MAX_LENGTH_M - radius]
            actuator.forcerange = [0., cap or MAX_FORCE_N]
        spec.add_sensor(name=f'{name}_stroke', type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=slide)
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
        data.joint(f'{name}_hooke_a').qpos[0] = np.arctan2(-y, z)
        data.joint(f'{name}_hooke_b').qpos[0] = np.arctan2(x, np.hypot(y, z))
        data.joint(f'{name}_slide').qpos[0] = np.linalg.norm(v) - model.body(f'{name}_shaft').pos[2]
    mujoco.mj_kinematics(model, data)


class TelescopingAssist:
    """Select rod targets from the policy and the previous measured support state."""

    def __init__(self, worlds, force):
        from cpu_jump_task import lower_leg_actuators

        model = worlds.model
        self.force_n = validate_assist_force(force)
        self.worlds = worlds
        qpos = worlds.sim.data.qpos
        self.actuators = [model.actuator(f'robot/exo_{side}').id for side in STRUTS]
        self.stroke = [int(model.sensor(f'robot/exo_{side}_stroke').adr[0]) for side in STRUTS]
        self.nominal_length = torch.tensor([model.body(f'robot/exo_{side}_shaft').pos[2] for side in STRUTS],
                                           dtype=qpos.dtype, device=qpos.device)
        self.position_pd = bool(np.all(model.actuator_biastype[self.actuators] == mujoco.mjtBias.mjBIAS_AFFINE))
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
        self.peak_force_n = qpos.new_zeros((worlds.count, 2))
        self.last_request = torch.zeros_like(self.peak_force_n)
        self.positive_work_j = torch.zeros_like(self.peak_force_n)
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
        self.commands = commands
        self.policy_io = policy_io
        if policy_io == ACTION_FORCE_IO:
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

    def policy_actions(self, actions):
        if not self.policy_control:
            return self.commands(actions)
        human = self.commands(actions[:, :-STRUT_ACTIONS])
        return torch.cat((human, actions[:, -STRUT_ACTIONS:]), dim=-1)

    def observation(self, state):
        if not self.policy_control:
            return state
        data = self.worlds.sim.data
        length = self.nominal_length + data.sensordata[:, self.stroke]
        features = torch.stack((
            (length - MIN_LENGTH_M) / (MAX_LENGTH_M - MIN_LENGTH_M),
            data.actuator_velocity[:, self.actuators],
            data.actuator_force[:, self.actuators] / MAX_FORCE_N,
            self.last_request,
            (data.sensordata[:, self.load] / self.worlds.device.weight).clamp(0., 4.),
        ), dim=-1)
        return torch.cat((state, features.flatten(1)), dim=-1)

    def controls(self, actions=None):
        command = self.commands
        if command is None:
            raise RuntimeError('Bind the exoskeleton to the jump phase controller before stepping')
        if self.policy_control:
            if actions is None or actions.shape != self.last_request.shape:
                raise ValueError('每个环境必须提供左右两根支杆的动作')
            request = (actions.clamp(-1., 1.) + 1.) * .5
            if not self.position_pd:
                requested = self.force_n * request
                self.last_request.copy_(requested / MAX_FORCE_N)
                if self.policy_io == ACTION_FORCE_IO:
                    return requested
        data = self.worlds.sim.data
        weight = self.worlds.device.weight
        load = ((data.sensordata[:, self.load] / weight - .06) / .24).clamp(0., 1.)
        length = self.nominal_length + data.sensordata[:, self.stroke]
        travel = ((length - MIN_LENGTH_M) / .045).clamp(0., 1.)
        travel *= ((MAX_LENGTH_M - length) / .055).clamp(0., 1.)
        speed = data.actuator_velocity[:, self.actuators]
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
        force = data.actuator_force[:, self.actuators]
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
        return (ACTION_FORCE_COSTS['unloaded_strut_force'] * unloaded_force
                + unloaded_linear
                + ACTION_FORCE_COSTS['unsupported_ankle_moment'] * unsupported_roll)

    def reset_controls(self, mask):
        # Reset the actuator command before forwarding the new episode pose.
        ctrl = self.worlds.sim.data.ctrl
        current = ctrl[:, self.actuators]
        released = PD_RELEASE_LENGTH_M - self.nominal_length if self.position_pd else 0.
        ctrl[:, self.actuators] = released if mask is None else torch.where(mask[:, None], released, current)
        rows = slice(None) if mask is None else mask
        initial = ((self.nominal_length - MIN_LENGTH_M) / (MAX_LENGTH_M - MIN_LENGTH_M)
                   if self.position_pd else 0.)
        self.last_request[rows] = initial
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

    def enable_load_sharing(self, *, vertical_support=False, posture_balance=False,
                            stance_transition=False, roll_balance=False, reward_overrides=None):
        if self.policy_io != ACTION_FORCE_IO:
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
        extension = (data.actuator_velocity[:, self.actuators] / cfg['rise_extension_mps']).clamp(0., 1.)
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
        impulse = data.actuator_force[:, self.actuators] * supported * dt
        changed = phase != self.assist_phase
        self.phase_force_impulse.masked_fill_(changed[:, None], 0.)
        self.phase_support_seconds.masked_fill_(changed[:, None], 0.)
        self.phase_force_impulse.add_(impulse)
        # A load spike cannot substitute for sustained support.
        self.phase_support_seconds.add_(footprint.clamp_max(1.) * supported * dt)
        self.phase_impulse_ns[:, 0].add_(impulse * (phase == 1)[:, None])
        self.phase_impulse_ns[:, 1].add_(impulse * (phase == 2)[:, None])
        band = costs['bilateral_force_share_deadband']
        balance = (costs['strut_impulse_balance'] * excess_side_difference(self.phase_force_impulse, band) / weight
                   + costs['boot_support_balance'] * excess_side_difference(self.phase_support_seconds, band))
        if self.posture_balance:
            push = phase == 2
            self.push_boot_impulse.masked_fill_(changed[:, None], 0.)
            self.push_boot_impulse.add_(footprint * supported * push[:, None] * dt)
            balance += POSTURE_SUPPORT_REWARD['push_boot_vertical_impulse_balance'] * excess_side_difference(
                self.push_boot_impulse, band)
            balance += POSTURE_SUPPORT_REWARD['push_extra_strut_impulse_balance'] * push * excess_side_difference(
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
        self.phase_force_impulse.add_(data.actuator_force[:, self.actuators] * supported * dt)
        self.phase_support_seconds.add_(supported * dt)
        self.assist_phase.copy_(phase)
        return rate

    def observe(self, dt):
        command = self.commands
        data = self.worlds.sim.data
        power = data.actuator_force * data.actuator_velocity
        strut_power = power[:, self.actuators]
        exo_power = strut_power.clamp_min(0.)
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
        if self.policy_io == ACTION_FORCE_IO:
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
        if self.policy_io == ACTION_FORCE_IO:
            self.efficiency_rate.sub_(self._action_force_cost())
            self.efficiency_rate.sub_(self._bilateral_assist_cost(strut_power, dt))
            assist_share_w = (self.reward_cfg or {}).get('assist_work_share_weight', 0.)
            if assist_share_w > 0.:
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

        self.sample_seconds += dt
        self.positive_work_j.add_(exo_power * dt)
        self.supported_work_j.add_(supported_power * dt)
        self.human_positive_work_j.add_(human_positive * dt)
        self.human_absolute_work_j.add_(human_absolute * dt)
        self.leg_torque_squared_integral.add_(leg_torque.square().mean(-1) * dt)
        torch.maximum(self.peak_leg_torque_nm, leg_torque.abs().amax(-1), out=self.peak_leg_torque_nm)
        torch.maximum(self.peak_force_n, data.actuator_force[:, self.actuators], out=self.peak_force_n)

    def report(self):
        exo_work = self.positive_work_j.sum(-1)
        total_positive = exo_work + self.human_positive_work_j
        share = exo_work / total_positive.clamp_min(1e-6)
        rms = (self.leg_torque_squared_integral / max(self.sample_seconds, 1e-6)).sqrt()
        return {
            'mechanism': MECHANISM,
            'control': self.policy_io or 'scheduled_force',
            'force_limit_per_strut_n': self.force_n,
            'peak_force_per_env_n': self.peak_force_n.tolist(),
            'strut_positive_work_per_env_j': self.positive_work_j.tolist(),
            'human_joint_positive_work_per_env_j': self.human_positive_work_j.tolist(),
            'human_joint_absolute_work_per_env_j': self.human_absolute_work_j.tolist(),
            **({'human_joint_metabolic_proxy_per_env_j': self.human_metabolic_proxy_j.tolist(),
                'landing_strut_absolute_work_per_env_j': self.landing_strut_absolute_work_j.tolist(),
                'push_strut_absolute_work_per_env_j': self.push_strut_absolute_work_j.tolist()}
               if self.policy_io == ACTION_FORCE_IO else {}),
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
