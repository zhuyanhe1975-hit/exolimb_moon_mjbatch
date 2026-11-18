"""Two powered telescoping struts with pinned backpack and boot mounts."""
from math import isfinite

import mujoco
import numpy as np
import torch

MAX_FORCE_N = 500.
DEFAULT_FORCE_N = MAX_FORCE_N
STRUTS = ('left', 'right')
STRUT_ACTIONS = len(STRUTS)
STRUT_OBSERVATIONS = 5 * len(STRUTS)
MOTOR_POLICY_IO = 'motor_force_v1'
POLICY_IO = 'position_pd_v1'
INITIAL_ASSIST_N = 80.
PD_KP = 4000.  # N/m
PD_KD = 60.  # N·s/m
PD_MAX_ADVANCE_M = .15


def validate_assist_force(force):
    if not isfinite(force) or not 0. <= force <= MAX_FORCE_N:
        raise ValueError(f'每根支杆的助力须在 0–{MAX_FORCE_N:g} N 内')
    return float(force)


def validate_policy_io(revision):
    if revision not in (None, MOTOR_POLICY_IO, POLICY_IO):
        raise ValueError(f'未知的外骨骼控制版本：{revision}')
    return revision


def exo_initial_state(source, target, *, initial_command=None, from_motor=False):
    """Warm-start the position policy without transferring force commands."""
    expanded = dict(source)
    for key, fill in (('mlp.0.weight', 0.), ('obs_normalizer._mean', 0.),
                      ('obs_normalizer._var', 1.), ('obs_normalizer._std', 1.)):
        old, template = source[key], target[key]
        if from_motor and old.shape == template.shape:
            value = old.clone()
            # Fourth feature of each rod changes from a force to a length.
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
            if from_motor:
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


def add_exoskeleton(spec, *, pd_force=None):
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
        self.actuators = [model.actuator(f'robot/exo_{side}').id for side in STRUTS]
        self.stroke = [int(model.sensor(f'robot/exo_{side}_stroke').adr[0]) for side in STRUTS]
        self.nominal_length = torch.tensor([model.body(f'robot/exo_{side}_shaft').pos[2] for side in STRUTS],
                                           dtype=worlds.sim.data.qpos.dtype)
        self.position_pd = bool(np.all(model.actuator_biastype[self.actuators] == mujoco.mjtBias.mjBIAS_AFFINE))
        self.load = [int(model.sensor(f'robot/balance_{side}_load').adr[0]) for side in STRUTS]
        self.human_actuators = worlds.device.human_actuators
        human_ids = np.arange(model.nu)[self.human_actuators]
        self.leg_columns = np.flatnonzero(np.isin(human_ids, lower_leg_actuators(model))).tolist()
        self.human_gear = torch.tensor(model.actuator_gear[human_ids, 0], dtype=self.nominal_length.dtype)
        limits = np.abs(model.actuator_forcerange[human_ids]).max(axis=1)
        self.human_torque_limits = torch.tensor(limits, dtype=self.nominal_length.dtype) * self.human_gear.abs()
        self.commands = None
        self.policy_io = None
        self.peak_force_n = torch.zeros((worlds.count, 2))
        self.last_request = torch.zeros_like(self.peak_force_n)
        self.positive_work_j = torch.zeros_like(self.peak_force_n)
        self.human_positive_work_j = torch.zeros(worlds.count)
        self.human_absolute_work_j = torch.zeros_like(self.human_positive_work_j)
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
        data = self.worlds.sim.data
        supported = (command.stage == command.EXTEND) & (command.ground_frames > 0)
        phase = (command.age / .045).clamp(0., 1.)
        phase *= ((command.extension_duration() - command.age) / .055).clamp(0., 1.)
        weight = self.worlds.device.weight
        load = ((data.sensordata[:, self.load] / weight - .06) / .24).clamp(0., 1.)
        length = self.nominal_length + data.sensordata[:, self.stroke]
        travel = ((length - MIN_LENGTH_M) / .045).clamp(0., 1.)
        travel *= ((MAX_LENGTH_M - length) / .055).clamp(0., 1.)
        speed = data.actuator_velocity[:, self.actuators]
        extension = (speed >= 0.) * ((speed + .05) / .16).clamp(0., 1.)
        gate = (phase * supported)[:, None] * load * travel * extension
        if self.policy_control:
            if actions is None or actions.shape != self.last_request.shape:
                raise ValueError('每个环境必须提供左右两根支杆的动作')
            request = (actions.clamp(-1., 1.) + 1.) * .5
            if self.position_pd:
                target = (length + PD_MAX_ADVANCE_M * request).clamp(MIN_LENGTH_M, MAX_LENGTH_M)
                self.last_request.copy_((target - MIN_LENGTH_M) / (MAX_LENGTH_M - MIN_LENGTH_M))
                applied = torch.where((gate > 0.) & (request > 0.) & (self.force_n > 0.),
                                      length + gate * (target - length), PD_RELEASE_LENGTH_M)
                return applied - self.nominal_length
            requested = self.force_n * request
            self.last_request.copy_(requested / MAX_FORCE_N)
            return requested * gate
        return self.force_n * gate

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

    def observe(self, dt):
        command = self.commands
        data = self.worlds.sim.data
        power = data.actuator_force * data.actuator_velocity
        exo_power = power[:, self.actuators].clamp_min(0.)
        human_power = power[:, self.human_actuators]
        human_positive = human_power.clamp_min(0.).sum(-1)
        human_absolute = human_power.abs().sum(-1)
        torque = data.actuator_force[:, self.human_actuators] * self.human_gear
        ratio_squared = (torque / self.human_torque_limits).square()
        leg_torque = torque[:, self.leg_columns]

        # Credit work only on a loaded foot as the body rises during push-off.
        # Soften the share denominator so small oscillations earn little credit.
        weight = self.worlds.device.weight
        load = ((data.sensordata[:, self.load] / weight - .08) / .30).clamp(0., 1.)
        rising = ((data.sensordata[:, command.com_velocity_xy.stop] - .05) / .30).clamp(0., 1.)
        extending = command.stage == command.EXTEND
        supported_power = (exo_power * load).sum(-1) * rising * extending
        power_scale = weight * 1.0  # lunar body weight transported at 1 m/s
        work_share = supported_power / (exo_power.sum(-1) + human_positive + .20 * power_scale)
        useful_power = (supported_power / (.25 * power_scale)).clamp_max(1.5)
        self.efficiency_rate.copy_(
            EFFICIENCY_REWARD['supported_exo_work_share'] * work_share
            + EFFICIENCY_REWARD['supported_exo_positive_power'] * useful_power
            - EFFICIENCY_REWARD['human_absolute_power'] * (human_absolute / power_scale).clamp_max(6.)
            - EFFICIENCY_REWARD['human_mean_squared_torque'] * ratio_squared.mean(-1)
            - EFFICIENCY_REWARD['peak_leg_squared_torque'] * ratio_squared[:, self.leg_columns].amax(-1)
        )

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
            'strut_supported_work_per_env_j': self.supported_work_j.tolist(),
            'strut_share_of_positive_work_per_env': share.tolist(),
            'human_leg_torque_rms_per_env_nm': rms.tolist(),
            'human_leg_torque_peak_per_env_nm': self.peak_leg_torque_nm.tolist(),
            'mechanical_work_sample_hz': 60,
        }
