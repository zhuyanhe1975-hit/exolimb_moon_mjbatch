"""A repeated crouch, takeoff, flight and landing task under lunar gravity."""
from math import isfinite, sqrt, tau

import torch

from moon_device_core import META

STANCE_SECS = .55
FLIGHT_SECS = .80
PERIOD_SECS = STANCE_SECS + FLIGHT_SECS
REWARD_COLUMNS = {name: i for i, name in enumerate(META['rewards'])}
TIMED_JUMP_ACTUATION = 'sagittal_residual_v1'
CONTACT_JUMP_ACTUATION = 'contact_sagittal_residual_v2'
JUMP_ACTUATION = 'compliant_sagittal_residual_v3'
ADAPTIVE_JUMP_ACTUATION = 'adaptive_event_jump_v4'
REBOUND_JUMP_ACTUATION = 'rebound_event_jump_v5'
FLOW_JUMP_ACTUATION = 'flow_rebound_jump_v6'
FORWARD_JUMP_ACTUATION = 'forward_rebound_jump_v7'
FREE_HEIGHT_FORWARD_ACTUATION = 'forward_rebound_free_height_v8'
SPEED_TRACKING_FORWARD_ACTUATION = 'forward_speed_tracking_free_height_v9'
LONG_STRIDE_FORWARD_ACTUATION = 'forward_long_stride_free_height_v10'
PHASE_LEAN_FORWARD_ACTUATION = 'forward_phase_lean_free_height_v11'
HIGH_SPEED_LEAN_FORWARD_ACTUATION = 'forward_high_speed_lean_free_height_v12'
ABSORBING_HOP_FORWARD_ACTUATION = 'forward_absorbing_hop_free_height_v13'
BROAD_JUMP_FORWARD_ACTUATION = 'forward_broad_jump_free_height_v14'
COORDINATED_BROAD_JUMP_ACTUATION = 'forward_coordinated_broad_jump_free_height_v15'
COORDINATED_TAKEOFF_ACTUATION = 'forward_coordinated_takeoff_free_height_v16'
SUPPORTED_TAKEOFF_ACTUATION = 'forward_supported_takeoff_free_height_v17'
STANCE_FLOW_TAKEOFF_ACTUATION = 'forward_stance_flow_free_height_v18'
STANCE_STRIDE_TAKEOFF_ACTUATION = 'forward_stance_stride_free_height_v19'
LANDING_PREPARATION_ACTUATION = 'forward_landing_preparation_free_height_v20'
FULL_SOLE_FORWARD_ACTUATION = 'forward_full_sole_free_height_v21'
STABLE_FULL_SOLE_ACTUATION = 'forward_stable_full_sole_free_height_v22'
BALANCED_FULL_SOLE_ACTUATION = 'forward_balanced_full_sole_free_height_v23'
DISTRIBUTED_SOLE_ACTUATION = 'forward_distributed_sole_free_height_v24'
TERRAIN_ALIGNED_ACTUATION = 'forward_terrain_aligned_landing_free_height_v25'
LOW_SPEED_SOLE_ACTUATION = 'forward_low_speed_sole_landing_free_height_v26'
BILATERAL_SOLE_ACTUATION = 'forward_bilateral_sole_free_height_v27'
COORDINATED_LANDING_ACTUATION = 'forward_coordinated_landing_free_height_v28'
MIRRORED_LANDING_ACTUATION = 'forward_mirrored_landing_free_height_v29'
PHASE_MIRRORED_LANDING_ACTUATION = 'forward_phase_mirrored_landing_free_height_v30'
SOLE_PLANE_LANDING_ACTUATION = 'forward_sole_plane_mirror_free_height_v31'
FORWARD_FACING_BOOT_ACTUATION = 'forward_boot_yaw_free_height_v32'
SOLE_EDGE_PROBE_ACTUATIONS = (SOLE_PLANE_LANDING_ACTUATION, FORWARD_FACING_BOOT_ACTUATION)
SOLE_COLLISION_ACTUATIONS = (BILATERAL_SOLE_ACTUATION, COORDINATED_LANDING_ACTUATION,
                            MIRRORED_LANDING_ACTUATION, PHASE_MIRRORED_LANDING_ACTUATION,
                            *SOLE_EDGE_PROBE_ACTUATIONS)
SOLE_PROBE_ACTUATIONS = (FULL_SOLE_FORWARD_ACTUATION, STABLE_FULL_SOLE_ACTUATION,
                         BALANCED_FULL_SOLE_ACTUATION, DISTRIBUTED_SOLE_ACTUATION,
                         TERRAIN_ALIGNED_ACTUATION, LOW_SPEED_SOLE_ACTUATION,
                         *SOLE_COLLISION_ACTUATIONS)
FREE_HEIGHT_FORWARD_ACTUATIONS = (FREE_HEIGHT_FORWARD_ACTUATION, SPEED_TRACKING_FORWARD_ACTUATION,
                                 LONG_STRIDE_FORWARD_ACTUATION, PHASE_LEAN_FORWARD_ACTUATION,
                                 HIGH_SPEED_LEAN_FORWARD_ACTUATION, ABSORBING_HOP_FORWARD_ACTUATION,
                                 BROAD_JUMP_FORWARD_ACTUATION, COORDINATED_BROAD_JUMP_ACTUATION,
                                 COORDINATED_TAKEOFF_ACTUATION, SUPPORTED_TAKEOFF_ACTUATION,
                                 STANCE_FLOW_TAKEOFF_ACTUATION, STANCE_STRIDE_TAKEOFF_ACTUATION,
                                 LANDING_PREPARATION_ACTUATION, *SOLE_PROBE_ACTUATIONS)
FORWARD_JUMP_ACTUATIONS = (FORWARD_JUMP_ACTUATION, *FREE_HEIGHT_FORWARD_ACTUATIONS)
HEIGHT_JUMP_ACTUATIONS = (ADAPTIVE_JUMP_ACTUATION, REBOUND_JUMP_ACTUATION,
                         FLOW_JUMP_ACTUATION, FORWARD_JUMP_ACTUATION)
DEFAULT_JUMP_HEIGHT_RANGE = (.09, .20)
CROUCH_SECS = .32
EXTENSION_SECS = .17


class JumpCommands:
    """Use a symmetric crouch and extension as a prior for the learned residual."""

    def __init__(self, worlds):
        device = worlds.device
        device.action_scale = torch.full_like(device.default_q, .25)
        self.prior = torch.zeros_like(device.default_q)
        for side in ('Left', 'Right'):
            for joint, scale, low, high, direction in (
                ('HipPitch', .5, -.85, .5, -.5),
                ('Knee', .65, 0., 1.3, 1.),
                ('Ankle', .45, -.75, .65, -.5),
            ):
                index = device.action_index(f'robot/{side}{joint}')
                device.action_scale[index] = scale
                device.action_lo[index] = max(float(device.limits[index, 0]), low)
                device.action_hi[index] = min(float(device.limits[index, 1]), high)
                self.prior[index] = direction / scale

    def __call__(self, actions, time):
        phase = time.remainder(PERIOD_SECS)
        lower = (phase / CROUCH_SECS).clamp(0., 1.)
        extend = ((CROUCH_SECS + EXTENSION_SECS - phase) / EXTENSION_SECS).clamp(0., 1.)
        crouch = .55 * (1 - (torch.pi * lower.minimum(extend)).cos())
        return actions + crouch[:, None] * self.prior


class ContactJumpCommands(JumpCommands):
    """Advance the jump from sensed touchdown, compression and takeoff."""

    WAIT, COMPRESS, EXTEND, FLIGHT = range(4)
    DT = META['control_dt']

    def __init__(self, worlds):
        super().__init__(worlds)
        self.worlds = worlds
        self.knee_qpos = [int(worlds.model.joint(f'robot/{side}Knee').qposadr[0])
                          for side in ('Left', 'Right')]
        count = worlds.count
        self.stage = torch.full((count,), self.WAIT, dtype=torch.long)
        self.age = torch.zeros(count)
        self.ground_frames = torch.zeros_like(self.stage)
        self.air_frames = torch.zeros_like(self.stage)

    def advance(self, measured):
        data = self.worlds.sim.data
        self.update(foot_airborne(measured), data.qpos[:, self.knee_qpos].mean(-1),
                    data.qvel[:, 2])

    def update(self, airborne, knee_angle, vertical_speed):
        self.age += self.DT
        self.ground_frames = torch.where(airborne, 0, self.ground_frames + 1)
        self.air_frames = torch.where(airborne, self.air_frames + 1, 0)
        supported = self.ground_frames >= 2
        landed = ((self.stage == self.WAIT) | (self.stage == self.FLIGHT)) & supported
        push = (self.stage == self.COMPRESS) & self.ready_to_push(supported, knee_angle, vertical_speed)
        takeoff = ((self.stage == self.EXTEND)
                   | ((self.stage == self.COMPRESS) & (vertical_speed > .10)))
        takeoff &= self.air_frames >= 2
        retry = (self.stage == self.EXTEND) & supported & (self.age >= .36)
        next_stage = torch.where(landed | retry, self.COMPRESS, self.stage)
        next_stage = torch.where(push, self.EXTEND, next_stage)
        next_stage = torch.where(takeoff, self.FLIGHT, next_stage)
        self.age.masked_fill_(next_stage != self.stage, 0.)
        self.stage = next_stage

    def ready_to_push(self, supported, knee_angle, vertical_speed):
        return supported & ((self.age >= CROUCH_SECS)
                            | ((self.age >= .24) & (knee_angle >= .83)
                               & (vertical_speed >= -.25)))

    def phase(self):
        # A phase can hold at the end of its interval until the physical event
        # occurs. In particular, a long flight cannot wrap to a crouch in midair.
        vz = self.worlds.sim.data.qvel[:, 2]
        flight_age = torch.where(vz >= 0, self.age.clamp(max=FLIGHT_SECS / 2),
                                 self.age.clamp(min=FLIGHT_SECS / 2, max=FLIGHT_SECS - .001))
        reference = torch.full_like(self.age, PERIOD_SECS - .001)
        reference = torch.where(self.stage == self.COMPRESS, self.age.clamp(max=CROUCH_SECS), reference)
        reference = torch.where(self.stage == self.EXTEND,
                                CROUCH_SECS + self.age.clamp(max=STANCE_SECS - CROUCH_SECS - .001),
                                reference)
        return torch.where(self.stage == self.FLIGHT, STANCE_SECS + flight_age, reference)

    def reset(self, mask=None):
        rows = slice(None) if mask is None else mask
        self.stage[rows] = self.WAIT
        self.age[rows] = 0
        self.ground_frames[rows] = 0
        self.air_frames[rows] = 0


def positive_actuator_power(force, velocity):
    """Sum positive mechanical power of the native actuator transmissions."""
    return (force * velocity).clamp_min(0.).sum(-1)


def lower_leg_actuators(model):
    names = ('HipRoll', 'HipPitch', 'HipYaw', 'Knee', 'Ankle', 'AnkleRoll')
    return [model.actuator(f'robot/{side}{joint}').id
            for side in ('Left', 'Right') for joint in names]


def peak_torque_ratio(actuator_force, force_limits):
    # These direct joint drives use fixed gear: torque/torque_limit equals
    # actuator force/force_limit, including for a non-unit gear.
    return (actuator_force.abs() / force_limits).amax(dim=-1)


def validate_jump_height_range(height_range):
    low, high = height_range
    if not (0 < low < high and isfinite(high)):
        raise ValueError('跳高范围须为两个有限正数，且最小值小于最大值（米）')
    return float(low), float(high)


class CompliantJumpCommands(ContactJumpCommands):
    """Match knee angle at contact, then absorb faster landings more deeply."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self.previous_vz = torch.zeros_like(self.age)
        self.impact_speed = torch.zeros_like(self.age)
        self.start_knee = torch.zeros_like(self.age)
        self.peak_knee = torch.zeros_like(self.age)
        self.extension_start = torch.zeros_like(self.age)
        self.shock = torch.zeros_like(self.age)
        self.landing_window = torch.zeros_like(self.stage, dtype=torch.bool)
        self.leg_actuators = lower_leg_actuators(worlds.model)
        limits = torch.as_tensor(worlds.model.actuator_forcerange[self.leg_actuators],
                                 dtype=self.age.dtype)
        self.force_limits = limits.abs().amax(dim=-1)
        gravity = -float(worlds.model.opt.gravity[2])
        self.speed_scale = gravity * FLIGHT_SECS / 2
        self.power_scale = worlds.device.weight * self.speed_scale

    def crouch_reference(self):
        t = (self.age / CROUCH_SECS).clamp(0., 1.)
        flex = self.start_knee + (self.peak_knee - self.start_knee) * t.square() * (3 - 2 * t)
        t = (self.age / self.extension_duration()).clamp(0., 1.)
        extend = self.extension_start * (1 - t.square() * (3 - 2 * t))
        return torch.where(self.stage == self.COMPRESS, flex,
                           torch.where(self.stage == self.EXTEND, extend, 0.))

    def extension_duration(self):
        return EXTENSION_SECS

    def __call__(self, actions, phase):
        return actions + self.crouch_reference()[:, None] * self.prior

    def advance(self, measured):
        data = self.worlds.sim.data
        knee = data.qpos[:, self.knee_qpos].mean(-1)
        vz = data.qvel[:, 2]
        airborne = foot_airborne(measured)
        before = self.stage
        before_target = self.crouch_reference()
        first_contact = ~airborne & (self.ground_frames == 0)
        self.impact_speed = torch.where(first_contact, (-self.previous_vz).clamp_min(0.),
                                        self.impact_speed)
        self.update(airborne, knee, vz)
        landing = (self.stage == self.COMPRESS) & (before != self.COMPRESS)
        self.start_knee = torch.where(landing, knee.clamp(0., 1.2), self.start_knee)
        depth = (.84 + .29 * self.impact_speed).clamp(max=1.12).maximum(self.start_knee)
        self.peak_knee = torch.where(landing, depth, self.peak_knee)
        push = (self.stage == self.EXTEND) & (before == self.COMPRESS)
        self.extension_start = torch.where(push, before_target, self.extension_start)
        # Velocity change over one control step captures abrupt braking, including
        # the first contact frame before the contact debounce has completed.
        braking = ~airborne & (self.ground_frames <= round(.16 / self.DT))
        delta_v = (vz - self.previous_vz).clamp(0., 2 * self.speed_scale)
        self.landing_window = braking & (self.impact_speed > .15)
        self.shock = (delta_v / self.speed_scale).square() * self.landing_window
        self.previous_vz.copy_(vz)

    def efficiency_cost(self):
        data = self.worlds.sim.data
        human = self.worlds.device.human_actuators
        power = positive_actuator_power(data.actuator_force[:, human], data.actuator_velocity[:, human])
        peak = peak_torque_ratio(data.actuator_force[:, self.leg_actuators], self.force_limits)
        return (.16 * power / self.power_scale + .90 * peak.square() * self.landing_window
                ) * self.DT + .10 * self.shock

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        for field in (self.previous_vz, self.impact_speed, self.start_knee, self.peak_knee,
                      self.extension_start, self.shock):
            field[rows] = 0.
        self.landing_window[rows] = False


class AdaptiveJumpCommands(CompliantJumpCommands):
    """PPO requests the push based on support and a per-hop height command."""

    NUM_ACTIONS = 24
    def __init__(self, worlds, *, height_range=DEFAULT_JUMP_HEIGHT_RANGE, fixed_height=None):
        self.height_range = validate_jump_height_range(height_range)
        low, high = self.height_range
        if fixed_height is not None and not low <= fixed_height <= high:
            raise ValueError(f'目标跳高须在 {low:g}–{high:g} 米之间')
        super().__init__(worlds)
        self.fixed_height = fixed_height
        self.generator = torch.Generator().manual_seed(torch.initial_seed())
        self.target_height = torch.empty_like(self.age)
        self.push_request = torch.zeros_like(self.age)
        self.push_duration = torch.full_like(self.age, EXTENSION_SECS)
        self.takeoff_z = torch.zeros_like(self.age)
        self.peak_rise = torch.zeros_like(self.age)
        self.flight_target = torch.zeros_like(self.age)
        self.completed = torch.zeros_like(self.stage, dtype=torch.bool)
        self.event_reward = torch.zeros_like(self.age)
        self.gravity = -float(worlds.model.opt.gravity[2])
        self._sample_height()

    def max_landing_interval(self):
        flight_time = 2 * sqrt(2 * self.height_range[1] / self.gravity)
        return max(1.5 * PERIOD_SECS, flight_time + .65)

    def _sample_height(self, mask=None):
        if self.fixed_height is not None:
            self.target_height[slice(None) if mask is None else mask] = self.fixed_height
            return
        low, high = self.height_range
        targets = low + (high - low) * torch.rand(self.worlds.count, generator=self.generator)
        if mask is None:
            self.target_height.copy_(targets)
        else:
            self.target_height[mask] = targets[mask]

    def can_push(self, supported, knee_angle, vertical_speed):
        compressed = knee_angle >= (self.start_knee + .20).clamp(.48, .85)
        return supported & (self.age >= .16) & compressed & (vertical_speed >= -.40)

    def ready_to_push(self, supported, knee_angle, vertical_speed):
        return self.can_push(supported, knee_angle, vertical_speed) & (self.push_request > 0.)

    def extension_duration(self):
        return self.push_duration

    def __call__(self, actions, phase=None):
        self.push_request.copy_(actions[:, -1])
        return super().__call__(actions[:, :-1], phase)

    def observation(self, state):
        data = self.worlds.sim.data
        knee = data.qpos[:, self.knee_qpos].mean(-1)
        vz = data.qvel[:, 2]
        stages = torch.nn.functional.one_hot(self.stage, num_classes=4)
        ready = (self.stage == self.COMPRESS) & self.can_push(self.ground_frames >= 2, knee, vz)
        signals = torch.stack((self.age.clamp(max=1.), self.hop_observation_signal(),
                               vz / self.speed_scale, knee, ready, self.peak_rise / .15), -1)
        return torch.cat((state, stages, signals), -1)

    def hop_observation_signal(self):
        return self.target_height / .15

    def advance(self, measured):
        before = self.stage
        flight_age = self.age.clone()
        data = self.worlds.sim.data
        height = data.qpos[:, 2]
        vz = data.qvel[:, 2]
        airborne = foot_airborne(measured)
        self.peak_rise = torch.where((before == self.FLIGHT) & airborne,
                                     self.peak_rise.maximum(height - self.takeoff_z),
                                     self.peak_rise)
        super().advance(measured)
        push = (before == self.COMPRESS) & (self.stage == self.EXTEND)
        duration = .20 - .08 * (self.push_request / 2).clamp(0., 1.)
        self.push_duration = torch.where(push, duration, self.push_duration)
        takeoff = (before != self.FLIGHT) & (self.stage == self.FLIGHT)
        self.flight_target = torch.where(takeoff, self.target_height, self.flight_target)
        self.takeoff_z = torch.where(takeoff, height, self.takeoff_z)
        self.peak_rise = torch.where(takeoff, 0., self.peak_rise)
        self.completed = ((before == self.FLIGHT) & (self.stage == self.COMPRESS)
                          & self.valid_flight(flight_age)
                          & ~measured['wangyu/failed'])
        self.event_reward = self.flight_event_reward(takeoff, vz)
        if self.completed.any():
            self._sample_height(self.completed)

    def valid_flight(self, flight_age):
        min_rise = (.5 * self.flight_target).clamp(max=.045)
        return (flight_age >= .25) & (self.peak_rise >= min_rise)

    def flight_event_reward(self, takeoff, vz):
        projected_rise = vz.clamp_min(0.).square() / (2 * self.gravity)
        launch_match = torch.exp(-((projected_rise - self.flight_target) / .075).square())
        height_match = torch.exp(-((self.peak_rise - self.flight_target) / .065).square())
        return (.45 * launch_match * (takeoff & (vz > .18))
                + (.18 + .95 * height_match) * self.completed)

    def reward(self, measured):
        raw = measured['wangyu/rewards']
        data = self.worlds.sim.data
        vz = data.qvel[:, 2]
        flight = self.stage == self.FLIGHT
        target_speed = (2 * self.gravity * self.target_height).sqrt()
        wait = (self.stage == self.COMPRESS) & (self.age > .38)
        rate = (.55 + .55 * (self.peak_rise / self.flight_target.clamp_min(1e-6)).clamp(0., 1.) * flight
                + .65 * (vz / target_speed).clamp(0., 1.) * (self.stage == self.EXTEND)
                - .65 * wait + self.horizontal_rate())
        rate += jump_posture_cost(raw)
        failed = measured['wangyu/failed']
        return (torch.where(failed, -2., rate) * self.DT
                + self.event_reward * ~failed - self.efficiency_cost())

    def horizontal_rate(self):
        data = self.worlds.sim.data
        drift = (data.qpos[:, :2] - self.worlds.device.target[:, :2]).square().sum(-1)
        return -.8 * drift - .8 * data.qvel[:, :2].square().sum(-1)

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        for field in (self.push_request, self.takeoff_z, self.peak_rise, self.flight_target,
                      self.completed, self.event_reward):
            field[rows] = 0
        self.push_duration[rows] = EXTENSION_SECS
        self._sample_height(mask)


class ReboundJumpCommands(AdaptiveJumpCommands):
    """Blend from absorbing the landing into pushing off at the turnaround."""

    def can_push(self, supported, knee_angle, vertical_speed):
        compressed = knee_angle >= (self.start_knee + .12).clamp(.36, .70)
        return supported & (self.age >= .08) & compressed & (vertical_speed >= -.28)

    def ready_to_push(self, supported, knee_angle, vertical_speed):
        ready = self.can_push(supported, knee_angle, vertical_speed)
        turning_up = (vertical_speed >= -.08) & (vertical_speed > self.previous_vz)
        rising = vertical_speed >= .035
        stalled = (self.age >= .18) & (vertical_speed.abs() <= .06)
        return ready & ((self.push_request > 0.) | turning_up | rising | stalled)

    def crouch_reference(self):
        reference = super().crouch_reference()
        progress = (self.age / self.extension_duration()).clamp(0., 1.)
        # Continuous knee target at the switch with a small initial extension
        # slope. Avoid the flat start of the previous push curve.
        extend = self.extension_start * (1 - progress).square() * (1 + 1.65 * progress)
        return torch.where(self.stage == self.EXTEND, extend, reference)

    def reward(self, measured):
        velocity = self.worlds.sim.data.qvel[:, 2]
        hold = (self.stage == self.COMPRESS) & (self.ground_frames >= 2) & (self.age >= .12)
        near_stop = (1 - velocity.abs() / .12).clamp(0., 1.)
        return super().reward(measured) - 1.2 * hold * near_stop * self.DT


class FlowJumpCommands(AdaptiveJumpCommands):
    """Use the supported turnaround to shorten the time spent on the ground."""

    def ready_to_push(self, supported, knee_angle, vertical_speed):
        ready = self.can_push(supported, knee_angle, vertical_speed)
        turning = (vertical_speed >= -.08) & (vertical_speed > self.previous_vz)
        nearly_still = vertical_speed.abs() <= .07
        rebound = (self.age >= .23) & (turning | nearly_still)
        return ready & ((self.push_request > 0.) | rebound)

    def grounded_delay_cost(self):
        velocity = self.worlds.sim.data.qvel[:, 2]
        grounded = self.ground_frames >= 2
        compressing = (self.stage == self.COMPRESS) & (self.age >= .12)
        pushing = self.stage == self.EXTEND
        near_stop = (1 - velocity.abs() / .12).clamp(0., 1.)
        delay = .85 * (compressing | pushing) + 1.4 * compressing * near_stop
        return delay * grounded * self.DT

    def reward(self, measured):
        return super().reward(measured) - self.grounded_delay_cost()


def jump_observation(state, time):
    phase = time.remainder(PERIOD_SECS) * (tau / PERIOD_SECS)
    return torch.cat((state, phase.sin()[:, None], phase.cos()[:, None]), dim=-1)


def jump_reference(time, gravity):
    phase = time.remainder(PERIOD_SECS)
    in_flight = phase >= STANCE_SECS
    stance = (phase / STANCE_SECS).clamp(0., 1.)
    flight_time = (phase - STANCE_SECS).clamp_min(0.)
    launch_speed = gravity * FLIGHT_SECS / 2
    stance_height = -launch_speed * STANCE_SECS * stance * (1 - stance)
    stance_velocity = launch_speed * (2 * stance - 1)
    flight_height = launch_speed * flight_time - .5 * gravity * flight_time.square()
    flight_velocity = launch_speed - gravity * flight_time
    return (torch.where(in_flight, flight_height, stance_height),
            torch.where(in_flight, flight_velocity, stance_velocity), in_flight)


def jump_height(worlds):
    return worlds.sim.data.qpos[:, 2] - worlds.device.target[:, 2]


def foot_airborne(measured):
    return measured['wangyu/rewards'][:, REWARD_COLUMNS['airborne']].bool()


def jump_posture_cost(raw):
    return sum(weight * raw[:, REWARD_COLUMNS[name]] for name, weight in (
        ('gravity_tilt', -.8), ('centroidal_angular_momentum', -.008),
        ('action_change', -.004), ('joint_pose', -.10), ('joint_torques', -2e-6),
        ('torso_angular_velocity', -.15),
    ))


def jump_reward(worlds, measured, phase):
    raw = measured['wangyu/rewards']
    data = worlds.sim.data
    height = jump_height(worlds)
    velocity = data.qvel[:, :3]
    height_ref, velocity_ref, in_flight = jump_reference(phase, -float(worlds.model.opt.gravity[2]))
    airborne = foot_airborne(measured)
    height_match = torch.exp(-((height - height_ref) / .07).square())
    velocity_match = torch.exp(-((velocity[:, 2] - velocity_ref) / .4).square())
    flight_clearance = (height / .1).clamp(0., 1.) * (airborne & in_flight)
    drift = (data.qpos[:, :2] - worlds.device.target[:, :2]).square().sum(-1)
    rate = (1.2 * height_match + .7 * velocity_match + .9 * (airborne == in_flight)
            + .6 * flight_clearance - .8 * drift - .8 * velocity[:, :2].square().sum(-1))
    for name, weight in (
        ('gravity_tilt', -.8), ('centroidal_angular_momentum', -.008),
        ('action_change', -.004), ('joint_pose', -.10), ('joint_torques', -2e-6),
        ('torso_angular_velocity', -.15),
    ):
        rate += weight * raw[:, REWARD_COLUMNS[name]]
    return torch.where(measured['wangyu/failed'], -2., rate) * META['control_dt']


class JumpCounter:
    """Count complete ground-to-ground hops with measurable flight and rise."""

    def __init__(self, num_envs, *, max_interval=1.5 * PERIOD_SECS):
        self.max_landing_steps = round(max_interval / META['control_dt'])
        self.grounded_seen = torch.zeros(num_envs, dtype=torch.bool)
        self.air_steps = torch.zeros(num_envs, dtype=torch.long)
        self.peak_height = torch.zeros(num_envs)
        self.count = torch.zeros(num_envs, dtype=torch.long)
        self.streak = torch.zeros_like(self.count)
        self.best_streak = torch.zeros_like(self.count)
        self.since_landing = torch.zeros_like(self.count)

    def update(self, height, airborne, failed):
        flying = airborne & self.grounded_seen
        self.air_steps += flying
        self.peak_height = torch.where(flying, self.peak_height.maximum(height), self.peak_height)
        completed = (~airborne & ~failed & (self.air_steps >= round(.25 / META['control_dt']))
                     & (self.peak_height >= .06))
        self.record_completed(completed)
        self.grounded_seen |= ~airborne
        self.air_steps[~airborne] = 0
        self.peak_height[~airborne] = 0

    def record_completed(self, completed):
        self.count += completed
        self.since_landing += 1
        self.streak[self.since_landing > self.max_landing_steps] = 0
        self.streak += completed
        self.best_streak = self.best_streak.maximum(self.streak)
        self.since_landing[completed] = 0

    def reset(self, mask=None):
        rows = slice(None) if mask is None else mask
        self.grounded_seen[rows] = False
        self.air_steps[rows] = 0
        self.peak_height[rows] = 0
        self.streak[rows] = 0
        self.since_landing[rows] = 0
