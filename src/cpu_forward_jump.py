"""Forward speed command and heading-relative reward for repeated jumps."""
from math import isfinite

import torch

from cpu_jump_task import META, PERIOD_SECS, FlowJumpCommands, foot_airborne, jump_posture_cost

DEFAULT_SPEED_RANGE = (0., 1.5)
FORWARD_OBS_COLUMNS = 3
HEIGHT_COMMAND_COLUMN = META['obs_dim'] + 5


def validate_speed_range(speed_range):
    low, high = speed_range
    if not (0 <= low < high and isfinite(high)):
        raise ValueError('速度范围须为两个有限非负数，且最小值小于最大值（米/秒）')
    return float(low), float(high)


class ForwardJumpCommands(FlowJumpCommands):
    def __init__(self, worlds, *, speed_range=DEFAULT_SPEED_RANGE, fixed_speed=None, **kwargs):
        super().__init__(worlds, **kwargs)
        self.speed_range = validate_speed_range(speed_range)
        if fixed_speed is not None and not self.speed_range[0] <= fixed_speed <= self.speed_range[1]:
            raise ValueError('目标速度超出训练范围')
        self.fixed_speed = fixed_speed
        self.speed_generator = torch.Generator(device=self.age.device).manual_seed(torch.initial_seed() ^ 0x464F5257)
        self.target_speed = torch.empty_like(self.age)
        self.flight_speed_sum = torch.zeros_like(self.age)
        self.flight_speed_steps = torch.zeros_like(self.age)
        self.origin = torch.empty_like(worlds.sim.data.qpos[:, :2])
        self.heading = torch.empty_like(self.origin)
        self.reset()

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        data = self.worlds.sim.data
        self.origin[rows] = data.qpos[rows, :2]
        self.flight_speed_sum[rows] = 0.
        self.flight_speed_steps[rows] = 0.
        w, _, _, z = self.worlds.device.heading[rows].unbind(-1)
        self.heading[rows] = torch.stack((1 - 2 * z.square(), 2 * w * z), -1)
        low, high = self.speed_range
        if self.fixed_speed is None:
            samples = low + (high - low) * self.sample_speed_fraction()
            self.target_speed[rows] = samples[rows]
        else:
            self.target_speed[rows] = self.fixed_speed

    def sample_speed_fraction(self):
        return torch.rand(self.worlds.count, generator=self.speed_generator, device=self.age.device)

    def __call__(self, actions, phase=None):
        xy = self.worlds.sim.data.qpos[:, :2]
        progress = self.reference_progress(((xy - self.origin) * self.heading).sum(-1))
        self.worlds.device.target[:, :2] = self.origin + progress[:, None] * self.heading
        return super().__call__(actions, phase)

    def reference_progress(self, progress):
        return progress.clamp_min(0.) * (self.target_speed > .02)

    def track_velocity(self):
        xy = self.worlds.sim.data.qvel[:, :2]
        forward = (xy * self.heading).sum(-1)
        lateral = xy[:, 1] * self.heading[:, 0] - xy[:, 0] * self.heading[:, 1]
        return forward, lateral

    def observation(self, state):
        forward, lateral = self.track_velocity()
        speed_obs = torch.stack((self.target_speed, forward, lateral), -1)
        return torch.cat((super().observation(state), speed_obs), -1)

    def horizontal_rate(self):
        forward, lateral = self.track_velocity()
        sigma = .30 + .16 * self.target_speed
        error = forward - self.target_speed
        tracking = torch.exp(-(error / sigma).square())
        # A dense term supplies a forward gradient even when the inherited
        # stationary hopper is far below the commanded speed.
        improvement = (self.target_speed.square() + .01).sqrt() - (error.square() + .01).sqrt()
        flight = self.stage == self.FLIGHT
        return (2.8 * improvement + 1.2 * tracking) * (.5 + .5 * flight) - self.steering_cost(lateral)

    def steering_cost(self, lateral):
        xy = self.worlds.sim.data.qpos[:, :2]
        drift = (xy - self.worlds.device.target[:, :2]).square().sum(-1)
        facing = (self.body_forward_xy() * self.heading).sum(-1)
        return .7 * lateral.square() + .8 * drift + .6 * (1 - facing)

    def body_forward_xy(self):
        w, x, y, z = self.worlds.sim.data.qpos[:, 3:7].unbind(-1)
        return torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y + w * z)), -1)

    def advance(self, measured):
        previous_stage = self.stage
        super().advance(measured)
        took_off = (previous_stage != self.FLIGHT) & (self.stage == self.FLIGHT)
        self.flight_speed_sum[took_off] = 0.
        self.flight_speed_steps[took_off] = 0.
        forward, _ = self.track_velocity()
        flying = self.stage == self.FLIGHT
        self.flight_speed_sum += forward * flying
        self.flight_speed_steps += flying

    def reward(self, measured):
        return super().reward(measured) + .85 * self.completed * self.flight_speed_match()

    def flight_speed_match(self):
        air_speed = self.flight_speed_sum / self.flight_speed_steps.clamp_min(1.)
        return torch.exp(-((air_speed - self.target_speed) / (.40 + .16 * self.target_speed)).square())

    def course_finished(self):
        xy = self.worlds.sim.data.qpos[:, :2]
        relative = xy - self.origin
        progress = (relative * self.heading).sum(-1)
        margin = self.worlds.device.ground.half - 2.4
        return ((xy.abs().amax(-1) >= margin) & (progress > .4) & (self.target_speed > .02))


class FreeHeightForwardJumpCommands(ForwardJumpCommands):
    """Follow the speed command and allow the policy to choose its jump arc."""

    def max_landing_interval(self):
        return 1.5 * PERIOD_SECS

    def _sample_height(self, mask=None):
        # The parent uses a shared event controller and buffers across tasks.
        self.target_height[slice(None) if mask is None else mask] = 0.

    def hop_observation_signal(self):
        return self.ground_frames.clamp_max(12) / 12

    def valid_flight(self, flight_age):
        # Contact loss and a short flight are sufficient; no apex threshold.
        return flight_age >= .16

    def flight_event_reward(self, takeoff, vz):
        return .55 * self.completed

    def reward(self, measured):
        failed = measured['wangyu/failed']
        flying = self.stage == self.FLIGHT
        waiting = (self.stage == self.COMPRESS) & (self.age > .38)
        rate = (.45 + 1.65 * self.horizontal_rate() + .25 * flying - .65 * waiting
                + jump_posture_cost(measured['wangyu/rewards']))
        completion = self.event_reward + 1.25 * self.completed * self.flight_speed_match()
        return (torch.where(failed, -2., rate) * self.DT + completion * ~failed
                - self.efficiency_cost() - self.grounded_delay_cost())


class SpeedTrackingForwardJumpCommands(FreeHeightForwardJumpCommands):
    """Track speed through propulsion and flight, with measured hops as events."""

    def sample_speed_fraction(self):
        uniform = super().sample_speed_fraction()
        # Exact endpoints occur in 10% of resets each. The interior stays uniform.
        return ((uniform - .10) / .80).clamp(0., 1.)

    def reference_progress(self, progress):
        # Orthogonal course error remains in the observation and failure check.
        # Along-course drift at a near-zero command gets a continuous cost.
        return progress

    def horizontal_rate(self):
        forward, lateral = self.track_velocity()
        error = forward - self.target_speed
        # Smooth absolute error has an informative slope even far from target.
        improvement = (self.target_speed.square() + .04).sqrt() - (error.square() + .04).sqrt()
        tracking = torch.exp(-(error / .35).square())
        xy = self.worlds.sim.data.qpos[:, :2]
        displacement = ((xy - self.origin) * self.heading).sum(-1)
        anchor = (1. - self.target_speed / .25).clamp_min(0.)
        anchor_cost = .5 * anchor * displacement.clamp(-2., 2.).square()
        return 9. * improvement + 2. * tracking - self.steering_cost(lateral) - anchor_cost

    def speed_match(self, speed):
        return (1. + ((speed - self.target_speed) / .40).square()).reciprocal()

    def flight_event_reward(self, takeoff, vz):
        # Give credit for the speed from the landing-to-takeoff transition.
        speed, _ = self.track_velocity()
        return .10 * self.speed_match(speed) * takeoff

    def flight_speed_match(self):
        speed = self.flight_speed_sum / self.flight_speed_steps.clamp_min(1.)
        return self.speed_match(speed)

    def reward(self, measured):
        failed = measured['wangyu/failed']
        flying = self.stage == self.FLIGHT
        waiting = (self.stage == self.COMPRESS) & (self.age > .38)
        rate = (.65 + self.horizontal_rate() + .55 * flying - .50 * waiting
                + jump_posture_cost(measured['wangyu/rewards']))
        landing = (.30 + .45 * self.flight_speed_match()) * self.completed
        return (torch.where(failed, -1.25, rate * self.DT + self.event_reward + landing)
                - self.efficiency_cost() - self.grounded_delay_cost())


def long_hop_event_weight(cycle_seconds, forward_meters, speed_mps, target_period):
    period_fraction = cycle_seconds / target_period
    distance_fraction = forward_meters / (speed_mps.clamp_min(.15) * target_period)
    cycle_quality = period_fraction.minimum(distance_fraction).clamp(0., 1.).square()
    moving = (speed_mps / .35).clamp(0., 1.)
    return torch.lerp(torch.ones_like(cycle_quality), cycle_quality, moving)


class LongStrideForwardJumpCommands(SpeedTrackingForwardJumpCommands):
    """Favor longer ground-to-ground hops and phase-dependent torso lean."""

    economical_hops = False

    def hop_event_weight(self):
        if self.long_hop_period is not None:
            return self.long_hop_weight
        if self.economical_hops:
            return 1.
        return (self.flight_speed_steps * self.DT).clamp_max(1.2)

    def __init__(self, worlds, *, long_hop_period=None, **kwargs):
        if long_hop_period is not None and not (isfinite(long_hop_period) and long_hop_period > 0):
            raise ValueError('Long-hop target period must be finite and positive')
        self.long_hop_period = long_hop_period
        if long_hop_period is not None:
            self.economical_hops = True
            progress = worlds.sim.data.qpos.new_zeros(worlds.count)
            self.previous_landing = progress.clone()
            self.landing_elapsed = progress.clone()
            self.seen_landing = progress.bool()
            self.long_hop_weight = torch.ones_like(progress)
        self.takeoff_progress = worlds.sim.data.qpos.new_zeros(worlds.count)
        self.touchdown_progress = torch.zeros_like(self.takeoff_progress)
        self.hop_distance = torch.zeros_like(self.takeoff_progress)
        self.com_xy = self.sensor_columns(worlds.model, 'com_pos', 2)
        self.torso_quat = self.sensor_columns(worlds.model, 'torso_quat', 4)
        super().__init__(worlds, **kwargs)

    @staticmethod
    def sensor_columns(model, name, width):
        start = int(model.sensor(f'robot/balance_{name}').adr[0])
        return slice(start, start + width)

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        self.takeoff_progress[rows] = 0.
        self.touchdown_progress[rows] = 0.
        self.hop_distance[rows] = 0.
        if self.long_hop_period is not None:
            self.previous_landing[rows] = 0.
            self.landing_elapsed[rows] = 0.
            self.seen_landing[rows] = False
            self.long_hop_weight[rows] = 1.

    def effective_long_hop_period(self):
        return self.long_hop_period

    def advance(self, measured):
        before = self.stage
        super().advance(measured)
        com_xy = self.worlds.sim.data.sensordata[:, self.com_xy]
        progress = (com_xy * self.heading).sum(-1)
        took_off = (before != self.FLIGHT) & (self.stage == self.FLIGHT)
        touched_down = (before == self.FLIGHT) & (self.ground_frames == 1)
        torch.where(took_off, progress, self.takeoff_progress, out=self.takeoff_progress)
        torch.where(touched_down, progress, self.touchdown_progress, out=self.touchdown_progress)
        self.hop_distance = torch.where(
            self.completed, self.touchdown_progress - self.takeoff_progress, 0.)
        if self.long_hop_period is not None:
            self.landing_elapsed.add_(self.DT)
            cycle_seconds = torch.where(self.seen_landing, self.landing_elapsed, self.flight_speed_steps * self.DT)
            cycle_distance = torch.where(
                self.seen_landing, self.touchdown_progress - self.previous_landing, self.hop_distance)
            self.long_hop_weight.copy_(long_hop_event_weight(
                cycle_seconds, cycle_distance, self.target_speed, self.effective_long_hop_period()))
            torch.where(self.completed, self.touchdown_progress, self.previous_landing, out=self.previous_landing)
            self.landing_elapsed.masked_fill_(self.completed, 0.)
            self.seen_landing.logical_or_(self.completed)

    def flight_event_reward(self, takeoff, vz):
        return torch.zeros_like(self.age)

    def stride_reward(self):
        moving = (self.target_speed / .35).clamp(0., 1.)
        preferred_span = .18 + .60 * self.target_speed
        # A linear distance payout would only reward distance per second.
        stride_quality = (self.hop_distance / preferred_span).clamp(0., 1.).square()
        speed_fit = self.flight_speed_match()
        event_weight = self.hop_event_weight()
        quality = .75 * (1 - moving) * speed_fit + 2.8 * moving * stride_quality * (.35 + .65 * speed_fit)
        return event_weight * quality * self.completed

    def forward_pitch(self, quat):
        w, x, y, z = quat.unbind(-1)
        up_xy = 2 * torch.stack((x * z + w * y, y * z - w * x), -1)
        up_forward = (up_xy * self.heading).sum(-1)
        return torch.atan2(up_forward, 1 - 2 * (x.square() + y.square()))

    def posture_pitch(self, quat):
        return self.forward_pitch(quat)

    def torso_posture_cost(self):
        data = self.worlds.sim.data
        torso = self.posture_pitch(data.sensordata[:, self.torso_quat])
        pelvis = self.posture_pitch(data.qpos[:, 3:7])
        return self.lean_cost(torso, pelvis, data.qvel[:, 2])

    def lean_cost(self, torso, pelvis, vz):
        movement = (self.target_speed / .45).clamp(0., 1.)
        # Compression follows the incoming motion; ascent and descent come
        # from the measured vertical velocity, so the lean has no fixed clock.
        reference = torch.where(self.stage == self.COMPRESS, .10 + .12 * (-vz / .45).tanh(), 0.)
        reference = torch.where(self.stage == self.EXTEND, .20, reference)
        reference = torch.where(self.stage == self.FLIGHT, .06 + .16 * (vz / .50).tanh(), reference)
        outside_band = ((torso - movement * reference).abs() - .10).clamp_min(0.)
        backward = (-torso - .14).clamp_min(0.)
        backbend = (pelvis - torso - .12).clamp_min(0.)
        return 4. * outside_band.square() + 8. * backward.square() + 5. * backbend.square()

    def reward(self, measured):
        failed = measured['wangyu/failed']
        flying = self.stage == self.FLIGHT
        waiting = (self.stage == self.COMPRESS) & (self.age > .38)
        posture = jump_posture_cost(measured['wangyu/rewards']) - self.torso_posture_cost()
        air_credit = 0. if self.economical_hops else .55 * flying
        rate = .65 + .65 * self.horizontal_rate() + air_credit - .50 * waiting + posture
        return (torch.where(failed, -1.25, rate * self.DT + self.stride_reward())
                - self.efficiency_cost() - self.grounded_delay_cost())


class PhaseLeanForwardJumpCommands(LongStrideForwardJumpCommands):
    """Improve fast hops through forward propulsion and flexible phase-dependent lean."""

    com_speed_gain = 4.5

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        self.com_velocity_xy = self.sensor_columns(worlds.model, 'com_vel', 2)

    def high_speed_weight(self):
        return ((self.target_speed - .75) / 1.25).clamp(0., 1.)

    def horizontal_rate(self):
        # Whole-body travel cancels apparent pelvis speed from swinging the limbs.
        com_velocity = self.worlds.sim.data.sensordata[:, self.com_velocity_xy]
        forward = (com_velocity * self.heading).sum(-1)
        error = forward - self.target_speed
        improvement = (self.target_speed.square() + .09).sqrt() - (error.square() + .09).sqrt()
        return super().horizontal_rate() + self.com_speed_gain * self.high_speed_weight() * improvement

    def torso_lean_reference(self, vz):
        rising = (vz / .42).clamp_min(0.).tanh()
        falling = (-vz / .50).clamp_min(0.).tanh()
        air_lean = .015 + .19 * rising - .10 * falling
        landing_blend = (self.ground_frames * (self.DT / .12)).clamp(0., 1.)
        compression_lean = -.035 + landing_blend * (.135 + .06 * falling)
        reference = torch.where(self.stage == self.COMPRESS, compression_lean, 0.)
        reference = torch.where(self.stage == self.EXTEND, .15 + .07 * rising, reference)
        return torch.where(self.stage == self.FLIGHT, air_lean, reference)

    def lean_cost(self, torso, pelvis, vz):
        target = self.torso_lean_reference(vz)
        outside_band = ((torso - target).abs() - .045).clamp_min(0.)
        propulsion = (self.stage == self.EXTEND) | ((self.stage == self.FLIGHT) & (vz > 0.))
        backwards = (-torso - torch.where(propulsion, .025, .14)).clamp_min(0.)
        backbend = (pelvis - torso - .12).clamp_min(0.)
        cost = (self.lean_strength(propulsion) * outside_band.square()
                + 12. * backwards.square() + 5. * backbend.square())
        return torch.lerp(super().lean_cost(torso, pelvis, vz), cost, self.high_speed_weight())

    def lean_strength(self, propulsion):
        return 32.


class HighSpeedLeanForwardJumpCommands(PhaseLeanForwardJumpCommands):
    com_speed_gain = 6.

    def high_speed_weight(self):
        width = max(.10, self.speed_range[1] - .75)
        return ((self.target_speed - .75) / width).clamp(0., 1.)

    def lean_strength(self, propulsion):
        # Weight the same soft window more before the apex than on landing.
        return torch.where(propulsion, 72., 45.)


class AbsorbingHopForwardJumpCommands(HighSpeedLeanForwardJumpCommands):
    """Favor a sustained flight arc and use the landing to start knee flexion."""

    def __init__(self, worlds, **kwargs):
        self.com_z_column = self.sensor_columns(worlds.model, 'com_pos', 3).stop - 1
        self.knee_dofs = [int(worlds.model.joint(f'robot/{side}Knee').dofadr[0])
                          for side in ('Left', 'Right')]
        self.launch_com_z = worlds.sim.data.qpos.new_zeros(worlds.count)
        self.peak_com_rise = torch.zeros_like(self.launch_com_z)
        self.rising_pitch = torch.zeros_like(self.launch_com_z)
        self.descending_pitch = torch.zeros_like(self.launch_com_z)
        self.rising_steps = torch.zeros_like(self.launch_com_z)
        self.descending_steps = torch.zeros_like(self.launch_com_z)
        super().__init__(worlds, **kwargs)

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        for field in (self.launch_com_z, self.peak_com_rise, self.rising_pitch,
                      self.descending_pitch, self.rising_steps, self.descending_steps):
            field[rows] = 0.

    def advance(self, measured):
        previous_stage = self.stage
        super().advance(measured)
        data = self.worlds.sim.data
        flying = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        took_off = (previous_stage != self.FLIGHT) & flying
        z = data.sensordata[:, self.com_z_column]
        torch.where(took_off, z, self.launch_com_z, out=self.launch_com_z)
        self.peak_com_rise[took_off] = 0.
        self.peak_com_rise = torch.where(
            flying, self.peak_com_rise.maximum(z - self.launch_com_z), self.peak_com_rise)
        pitch = self.posture_pitch(data.sensordata[:, self.torso_quat])
        vz = self.flight_vertical_speed(data.qvel[:, 2])
        for total, steps, active in (
            (self.rising_pitch, self.rising_steps, flying & (vz > .05)),
            (self.descending_pitch, self.descending_steps, flying & (vz < -.05)),
        ):
            total[took_off] = 0.
            steps[took_off] = 0.
            total += pitch * active
            steps += active

    def flight_vertical_speed(self, pelvis_vz):
        return pelvis_vz

    def torso_lean_reference(self, vz):
        rise = (vz / .35).clamp_min(0.).tanh()
        fall = (-vz / .40).clamp_min(0.).tanh()
        in_air = .01 + .22 * rise - .025 * fall
        loading = (self.ground_frames * (self.DT / .14)).clamp(0., 1.)
        reference = torch.where(self.stage == self.COMPRESS, .11 * loading, 0.)
        reference = torch.where(self.stage == self.EXTEND, .14 + .07 * rise, reference)
        return torch.where(self.stage == self.FLIGHT, in_air, reference)

    def lean_cost(self, torso, pelvis, vz):
        baseline = super().lean_cost(torso, pelvis, vz)
        reference = self.torso_lean_reference(vz)
        propulsion = (self.stage == self.EXTEND) | ((self.stage == self.FLIGHT) & (vz > 0.))
        error = ((torso - reference).abs() - .035).clamp_min(0.)
        excessive_backbend = (pelvis - torso - .12).clamp_min(0.)
        coordinated = (torch.where(propulsion, 86., 62.) * error.square()
                       + 6. * excessive_backbend.square())
        return torch.lerp(baseline, coordinated, (self.target_speed / .55).clamp(0., 1.))

    def stride_reward(self):
        event_weight = self.hop_event_weight()
        # Monotonic, diminishing credit for real COM rise. No requested
        # height, matching error, or minimum apex is used.
        lift_quality = -torch.expm1(-self.peak_com_rise / .16)
        rise = self.rising_pitch / self.rising_steps.clamp_min(1.)
        fall = self.descending_pitch / self.descending_steps.clamp_min(1.)
        observed_arc = (self.rising_steps > 0) & (self.descending_steps > 0)
        movement = (self.target_speed / .45).clamp(0., 1.)
        arc = 4.2 * lift_quality + 1.7 * movement * self.flight_posture_quality(rise, fall) * observed_arc
        return super().stride_reward() + self.completed * event_weight * arc

    def flight_posture_quality(self, rise, fall):
        transition = ((rise - fall) / .16).clamp(0., 1.) * (rise / .14).clamp(0., 1.)
        return transition * torch.exp(-(fall / .13).square())

    def landing_flexion_rate(self):
        data = self.worlds.sim.data
        knee_speed = data.qvel[:, self.knee_dofs].mean(-1)
        absorbing = ((self.stage == self.COMPRESS) | (self.stage == self.FLIGHT)) & self.landing_window
        incoming = (self.impact_speed / .50).clamp(0., 1.)
        early = (1. - self.ground_frames * (self.DT / .18)).clamp(0., 1.)
        return 2.0 * (knee_speed / 1.8).tanh() * incoming * early * absorbing

    def reward(self, measured):
        return (super().reward(measured)
                + self.DT * self.landing_flexion_rate() * ~measured['wangyu/failed'])


class BroadJumpForwardCommands(AbsorbingHopForwardJumpCommands):
    """Open the torso at takeoff and fold forward to prepare for landing."""

    def flight_vertical_speed(self, pelvis_vz):
        com_vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        return torch.where(self.stage == self.FLIGHT, com_vz, pelvis_vz)

    def torso_lean_reference(self, vz):
        vz = self.flight_vertical_speed(vz)
        # Use actual knee extension to open up as support is lost. In the air,
        # the COM velocity sets the transition from opening to landing prep.
        knee = self.worlds.sim.data.qpos[:, self.knee_qpos].mean(-1)
        extension = ((self.extension_start - knee) / .34).clamp(0., 1.)
        push_lean = .095 - .245 * extension
        air_lean = .018 - .16 * (vz / .32).tanh()
        loading = (self.ground_frames * (self.DT / .15)).clamp(0., 1.)
        compression_lean = .105 + .035 * loading
        reference = torch.where(self.stage == self.COMPRESS, compression_lean, 0.)
        reference = torch.where(self.stage == self.EXTEND, push_lean, reference)
        reference = torch.where(self.stage == self.FLIGHT, air_lean, reference)
        return reference * (.65 + .35 * (self.target_speed / .55).clamp(0., 1.))

    def lean_cost(self, torso, pelvis, vz):
        reference = self.torso_lean_reference(vz)
        in_flight = self.stage == self.FLIGHT
        error = ((torso - reference).abs() - .025).clamp_min(0.)
        # Arm swing and pelvis rotation can contribute without excessive
        # lumbar extension or a persistent backward tilt.
        excessive_extension = (pelvis - torso - .24).clamp_min(0.)
        backwards = (-torso - .22).clamp_min(0.)
        return (torch.where(in_flight, 140., 88.) * error.square()
                + 10. * excessive_extension.square() + 12. * backwards.square())

    def flight_posture_quality(self, rise, fall):
        opens = (-rise / .11).clamp(0., 1.)
        prepares = (fall / .14).clamp(0., 1.)
        return opens * prepares * ((fall - rise) / .20).clamp(0., 1.)


class CoordinatedBroadJumpCommands(BroadJumpForwardCommands):
    """Keep arm swing coordinated and reduce fore-aft foot stagger."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        joints = ('ShoulderRoll', 'ShoulderPitch', 'ShoulderYaw', 'Elbow')
        self.arm_qpos = torch.tensor([
            [int(worlds.model.joint(f'robot/{side}{name}').qposadr[0]) for name in joints]
            for side in ('Left', 'Right')
        ], device=self.age.device)
        self.arm_neutral = torch.as_tensor(worlds.model.qpos0, dtype=self.age.dtype, device=self.age.device)[self.arm_qpos]
        # Reflection across the sagittal plane reverses roll and yaw.
        self.arm_mirror = torch.tensor([-1., 1., -1., 1.], device=self.age.device)
        self.arm_tolerance = torch.tensor([.17, .15, .20, .15], device=self.age.device)
        self.left_sole_xy = self.sensor_columns(worlds.model, 'left_pos', 2)
        self.right_sole_xy = self.sensor_columns(worlds.model, 'right_pos', 2)

    def arm_mirror_error(self):
        arms = self.worlds.sim.data.qpos[:, self.arm_qpos] - self.arm_neutral
        return (arms[:, 0] - self.arm_mirror * arms[:, 1]).abs()

    def foot_stagger(self):
        sensors = self.worlds.sim.data.sensordata
        sole_delta = sensors[:, self.left_sole_xy] - sensors[:, self.right_sole_xy]
        forward = torch.nn.functional.normalize(self.body_forward_xy(), dim=-1, eps=1e-6)
        return (sole_delta * forward).sum(-1).abs()

    def coordination_cost(self):
        arm_error = (self.arm_mirror_error() - self.arm_tolerance).clamp_min(0.)
        com_vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        descending = (-com_vz / .45).clamp(0., 1.) * (self.stage == self.FLIGHT)
        landing = ((self.stage == self.COMPRESS) & self.landing_window).float()
        align = descending.maximum(landing)
        foot_error = (self.foot_stagger() - (.085 - .035 * align)).clamp_min(0.)
        cost = 12. * arm_error.square().mean(-1) + (1. + align) * (foot_error / .18).square()
        return cost * (self.stage != self.WAIT)

    def reward(self, measured):
        return (super().reward(measured)
                - self.DT * self.coordination_cost() * ~measured['wangyu/failed'])


class CoordinatedTakeoffCommands(CoordinatedBroadJumpCommands):
    """Retain the broad-jump torso transition as the arm and leg motion aligns."""

    def __init__(self, worlds, **kwargs):
        self.takeoff_torso_pitch = worlds.sim.data.qpos.new_zeros(worlds.count)
        super().__init__(worlds, **kwargs)

    def reset(self, mask=None):
        super().reset(mask)
        self.takeoff_torso_pitch[slice(None) if mask is None else mask] = 0.

    def advance(self, measured):
        before = self.stage
        super().advance(measured)
        took_off = (before != self.FLIGHT) & (self.stage == self.FLIGHT)
        pitch = self.posture_pitch(self.worlds.sim.data.sensordata[:, self.torso_quat])
        torch.where(took_off, pitch, self.takeoff_torso_pitch, out=self.takeoff_torso_pitch)

    def lean_cost(self, torso, pelvis, vz):
        com_vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        next_to_takeoff = ((com_vz - .06) / .30).clamp(0., 1.) * (self.stage == self.EXTEND)
        early_flight = (1. - self.age / .18).clamp(0., 1.) * (self.stage == self.FLIGHT) * (com_vz > 0.)
        missed_opening = (torso + .055).clamp_min(0.)
        cost = (next_to_takeoff + early_flight) * (10. * missed_opening + 86. * missed_opening.square())
        return super().lean_cost(torso, pelvis, vz) + cost

    def stride_reward(self):
        landing_pitch = self.descending_pitch / self.descending_steps.clamp_min(1.)
        opens = ((.035 - self.takeoff_torso_pitch) / .14).clamp(0., 1.)
        folds = ((landing_pitch - .025) / .12).clamp(0., 1.)
        event_weight = self.hop_event_weight()
        quality = opens * (.45 + .55 * folds)
        return (super().stride_reward()
                + 1.8 * self.completed * event_weight * quality * (self.descending_steps > 0))


class SupportedTakeoffCommands(CoordinatedTakeoffCommands):
    """Credit a loaded compression and push, while charging for idle time."""

    def __init__(self, worlds, **kwargs):
        self.previous_com_vz = worlds.sim.data.qpos.new_zeros(worlds.count)
        self.support_time = torch.zeros_like(self.previous_com_vz)
        self.push_speed_gain = torch.zeros_like(self.previous_com_vz)
        self.takeoff_support_time = torch.zeros_like(self.previous_com_vz)
        self.takeoff_push_speed_gain = torch.zeros_like(self.previous_com_vz)
        self.sole_load_ratio = torch.zeros_like(self.previous_com_vz)
        self.sole_load_rise = torch.zeros_like(self.previous_com_vz)
        self.peak_support_load_ratio = torch.zeros_like(self.previous_com_vz)
        self.takeoff_peak_load_ratio = torch.zeros_like(self.previous_com_vz)
        super().__init__(worlds, **kwargs)
        self.sole_load_columns = [
            self.sensor_columns(worlds.model, f'{side}_load', 1).start for side in ('left', 'right')
        ]

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        for field in (self.previous_com_vz, self.support_time, self.push_speed_gain,
                      self.takeoff_support_time, self.takeoff_push_speed_gain,
                      self.sole_load_ratio, self.sole_load_rise,
                      self.peak_support_load_ratio, self.takeoff_peak_load_ratio):
            field[rows] = 0.

    def can_push(self, supported, knee_angle, vertical_speed):
        com_vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        # Leave a small window before the lowest point for a continuous reversal.
        near_turnaround = com_vz >= -.07
        return super().can_push(supported, knee_angle, com_vz) & near_turnaround

    def advance(self, measured):
        before = self.stage
        super().advance(measured)
        data = self.worlds.sim.data
        vz = data.sensordata[:, self.com_velocity_xy.stop]
        knee_speed = data.qvel[:, self.knee_dofs].mean(-1)
        load = data.sensordata[:, self.sole_load_columns].clamp_min(0.).sum(-1)
        load_ratio = load * ~foot_airborne(measured) / self.worlds.device.weight
        self.sole_load_rise.copy_((load_ratio - self.sole_load_ratio).clamp_min(0.))
        self.sole_load_ratio.copy_(load_ratio)
        self.peak_support_load_ratio = self.peak_support_load_ratio.maximum(load_ratio)
        # Cap each sample so a sharp landing force cannot replace sustained support.
        loaded = load_ratio.clamp_max(1.)
        compressing = self.stage == self.COMPRESS
        pushing = self.stage == self.EXTEND
        flexing = (knee_speed / .65).clamp(0., 1.) * compressing
        extending = (-knee_speed / .65).clamp(0., 1.) * pushing
        self.support_time += self.DT * loaded * (flexing + extending)
        # Upward COM speed gained during the supported push is a measure of
        # useful propulsion. The initial landing collision does not count.
        gain = (vz - self.previous_com_vz).clamp(0., .15)
        self.push_speed_gain += loaded * extending * gain * (vz > 0.)
        self.previous_com_vz.copy_(vz)
        took_off = (before != self.FLIGHT) & (self.stage == self.FLIGHT)
        torch.where(took_off, self.support_time, self.takeoff_support_time, out=self.takeoff_support_time)
        torch.where(took_off, self.push_speed_gain, self.takeoff_push_speed_gain, out=self.takeoff_push_speed_gain)
        torch.where(took_off, self.peak_support_load_ratio, self.takeoff_peak_load_ratio,
                    out=self.takeoff_peak_load_ratio)
        self.support_time[took_off] = 0.
        self.push_speed_gain[took_off] = 0.
        self.peak_support_load_ratio[took_off] = 0.

    def grounded_delay_cost(self):
        data = self.worlds.sim.data
        vz = data.sensordata[:, self.com_velocity_xy.stop]
        knee_speed = data.qvel[:, self.knee_dofs].abs().mean(-1)
        stance = (self.stage == self.COMPRESS) | (self.stage == self.EXTEND)
        idle = (self.age > .22) & (vz.abs() < .08) & (knee_speed < .28)
        overlong = self.age > .45
        return (1.4 * idle + .65 * overlong) * stance * (self.ground_frames >= 2) * self.DT

    def stride_reward(self):
        support_quality = (self.takeoff_support_time / .26).clamp(0., 1.).square()
        push_quality = (self.takeoff_push_speed_gain / .55).clamp(0., 1.)
        event_weight = self.hop_event_weight()
        support = 2.6 * support_quality * (.35 + .65 * push_quality)
        return super().stride_reward() + self.completed * event_weight * support

    def contact_load_cost(self):
        # Penalize tall peaks and sharp loading on touchdown. The quadratic
        # part makes the same load spread across several samples cheaper.
        excess = (self.sole_load_ratio - 2.5).clamp_min(0.)
        overload = excess.clamp_max(5.).square() + 10. * (excess - 5.).clamp_min(0.)
        loading_spike = (self.sole_load_rise - 1.25).clamp_min(0.)
        return (.10 * overload + self.landing_window * (.15 * overload + .16 * loading_spike)) * self.DT

    def reward(self, measured):
        return super().reward(measured) - self.contact_load_cost() * ~measured['wangyu/failed']


class StanceFlowTakeoffCommands(SupportedTakeoffCommands):
    """Link the loaded descent and knee flexion to a loaded rise and extension."""

    def __init__(self, worlds, **kwargs):
        self.stance_start_z = worlds.sim.data.qpos.new_zeros(worlds.count)
        self.stance_start_knee = torch.zeros_like(self.stance_start_z)
        self.stance_bottom_z = torch.zeros_like(self.stance_start_z)
        self.stance_peak_knee = torch.zeros_like(self.stance_start_z)
        self.stance_flow_quality = torch.zeros_like(self.stance_start_z)
        self.takeoff_flow_quality = torch.zeros_like(self.stance_start_z)
        self.stance_flow_rate = torch.zeros_like(self.stance_start_z)
        self.stance_seen = torch.zeros_like(self.stance_start_z, dtype=torch.bool)
        super().__init__(worlds, **kwargs)

    def reset(self, mask=None):
        super().reset(mask)
        rows = slice(None) if mask is None else mask
        for field in (self.stance_start_z, self.stance_start_knee, self.stance_bottom_z,
                      self.stance_peak_knee, self.stance_flow_quality, self.takeoff_flow_quality,
                      self.stance_flow_rate, self.stance_seen):
            field[rows] = 0

    def advance(self, measured):
        before = self.stage
        super().advance(measured)
        data = self.worlds.sim.data
        z = data.sensordata[:, self.com_z_column]
        vz = data.sensordata[:, self.com_velocity_xy.stop]
        knee = data.qpos[:, self.knee_qpos].mean(-1)
        knee_speed = data.qvel[:, self.knee_dofs].mean(-1)
        contact = (self.sole_load_ratio > .10) & (self.ground_frames > 0)
        arriving = (before == self.WAIT) | (before == self.FLIGHT)
        starting = contact & (~self.stance_seen | (arriving & (self.ground_frames == 1)))
        self.stance_seen |= starting
        # Fixed-size updates avoid CUDA boolean-index compaction on each step.
        torch.where(starting, z, self.stance_start_z, out=self.stance_start_z)
        torch.where(starting, z, self.stance_bottom_z, out=self.stance_bottom_z)
        torch.where(starting, knee, self.stance_start_knee, out=self.stance_start_knee)
        torch.where(starting, knee, self.stance_peak_knee, out=self.stance_peak_knee)
        self.stance_flow_quality[starting] = 0.
        torch.where(contact, self.stance_bottom_z.minimum(z), self.stance_bottom_z, out=self.stance_bottom_z)
        torch.where(contact, self.stance_peak_knee.maximum(knee), self.stance_peak_knee, out=self.stance_peak_knee)
        lowering = ((self.stance_start_z - self.stance_bottom_z) / .045).clamp(0., 1.)
        compression = ((self.stance_peak_knee - self.stance_start_knee) / .18).clamp(0., 1.)
        recovery = ((z - self.stance_bottom_z) / .065).clamp(0., 1.)
        extension = ((self.stance_peak_knee - knee) / .20).clamp(0., 1.)
        bent = lowering.minimum(compression)
        released = recovery.minimum(extension)
        # Capture what the legs did with load on the feet, even if the
        # event controller is still transitioning between its stance phases.
        torch.where(contact, (bent * released).sqrt(), self.stance_flow_quality, out=self.stance_flow_quality)
        absorbing = (-vz / .30).clamp(0., 1.) * (knee_speed / .8).clamp(0., 1.)
        propelling = (vz / .25).clamp(0., 1.) * (-knee_speed / 1.2).clamp(0., 1.) * bent
        self.stance_flow_rate = (absorbing + 4.5 * propelling) * self.sole_load_ratio.clamp_max(1.) * contact
        took_off = (before != self.FLIGHT) & (self.stage == self.FLIGHT)
        torch.where(took_off, self.stance_flow_quality, self.takeoff_flow_quality, out=self.takeoff_flow_quality)
        self.stance_flow_quality[took_off] = 0.
        self.stance_seen[took_off] = False

    def stride_reward(self):
        event_weight = self.hop_event_weight()
        return super().stride_reward() + 3.6 * self.completed * event_weight * self.takeoff_flow_quality

    def reward(self, measured):
        return super().reward(measured) + self.DT * self.stance_flow_rate * ~measured['wangyu/failed']


class StanceStrideTakeoffCommands(StanceFlowTakeoffCommands):
    """Restore long forward strides alongside the longer loaded stance."""

    def horizontal_rate(self):
        com_velocity = self.worlds.sim.data.sensordata[:, self.com_velocity_xy]
        forward = (com_velocity * self.heading).sum(-1)
        error = forward - self.target_speed
        improvement = (self.target_speed.square() + .05).sqrt() - (error.square() + .05).sqrt()
        moving = (self.target_speed / .35).clamp(0., 1.)
        return super().horizontal_rate() + 4. * moving * improvement

    def stride_reward(self):
        preferred_span = .22 + .77 * self.target_speed
        span = (self.hop_distance / preferred_span).clamp(0., 1.).square()
        event_weight = self.hop_event_weight()
        moving = (self.target_speed / .35).clamp(0., 1.)
        stride = 3.2 * moving * span * (.35 + .65 * self.flight_speed_match())
        return super().stride_reward() + self.completed * event_weight * stride


class LandingPreparationCommands(StanceStrideTakeoffCommands):
    """Soften the airborne knees and carry the landing into a loaded push."""

    def airborne_knee_cost(self):
        data = self.worlds.sim.data
        knees = data.qpos[:, self.knee_qpos]
        com_vz = data.sensordata[:, self.com_velocity_xy.stop]
        airborne = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        released = (self.age / .10).clamp(0., 1.)
        # Allow full extension at release. As the COM starts falling, increase
        # the desired flexion gradually without setting a flight time or apex.
        descent = (-com_vz / .55).clamp(0., 1.)
        lock = ((.15 - knees) / .15).clamp(0., 1.).square().mean(-1)
        preparation = ((.33 - knees) / .33).clamp(0., 1.).square().mean(-1)
        return (7. * lock + 6. * descent * preparation) * airborne * released

    def contact_load_cost(self):
        return 1.35 * super().contact_load_cost()

    def reward(self, measured):
        live = ~measured['wangyu/failed']
        return (super().reward(measured)
                + self.DT * (.65 * self.stance_flow_rate - self.airborne_knee_cost()) * live)


class FullSoleForwardJumpCommands(LandingPreparationCommands):
    """Place the boots on the local surface and preserve the rolling push-off."""

    sole_placement_weight = 1.

    def __init__(self, worlds, **kwargs):
        probes = [self.sensor_columns(worlds.model, f'{side}_{region}_pos', 3)
                  for side in ('left', 'right') for region in ('heel', 'toe')]
        self.sole_probe_columns = [column for probe in probes for column in range(probe.start, probe.stop)]
        super().__init__(worlds, **kwargs)

    def stationary_weight(self):
        return (1. - self.target_speed / .12).clamp(0., 1.)

    def reference_progress(self, progress):
        # Keep the starting position observable at zero speed.
        return (1. - self.stationary_weight()) * super().reference_progress(progress)

    def horizontal_rate(self):
        data = self.worlds.sim.data
        velocity = data.sensordata[:, self.com_velocity_xy]
        stationary = self.stationary_weight()
        displacement = (data.sensordata[:, self.com_xy] - self.origin).norm(dim=-1)
        displacement = self.stationary_distance_for_reward(displacement)
        hold_position = 11. * (displacement.square() + .025).sqrt() - 11. * (.025 ** .5)
        hold_speed = 9. * ((velocity.square().sum(-1) + .01).sqrt() - .10)
        forward = (velocity * self.heading).sum(-1)
        error = forward - self.target_speed
        speed_improvement = (self.target_speed.square() + .025).sqrt() - (error.square() + .025).sqrt()
        return (super().horizontal_rate() - stationary * (hold_position + hold_speed)
                + 9. * self.high_speed_weight() * speed_improvement)

    def stationary_distance_for_reward(self, displacement):
        return displacement

    def sole_probes(self):
        sensors = self.worlds.sim.data.sensordata
        return sensors[:, self.sole_probe_columns].reshape(-1, 2, 2, 3)

    def sole_clearances(self):
        probes = self.sole_probes()
        return probes[..., 2] - self.worlds.device.ground.height(probes[..., :2])

    def sole_placement_cost(self):
        data = self.worlds.sim.data
        gap = self.sole_clearances()
        uneven = ((gap[..., 0] - gap[..., 1]).abs() - .014).clamp_min(0.)
        error = (uneven / .065).clamp_max(2.).square()
        vz = data.sensordata[:, self.com_velocity_xy.stop]
        coming_down = ((-vz - .06) / .35).clamp(0., 1.)
        near_ground = ((.19 - gap.amin(-1)) / .15).clamp(0., 1.)
        approach = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        load = data.sensordata[:, self.sole_load_columns] / self.worlds.device.weight
        foot_loaded = ((load - .02) / .43).clamp(0., 1.)
        # Late extension can lift the heel. The largest cost is before or
        # during compression, when the foot should spread the arriving load.
        late_push = (self.stage == self.EXTEND) * (self.age / .13).clamp(0., 1.)
        stance = (self.ground_frames > 0) & (self.stage != self.WAIT)
        preparation = 5. * approach[:, None] * coming_down[:, None] * near_ground
        support = 8. * stance[:, None] * (1. - late_push[:, None]) * foot_loaded
        return ((preparation + support) * error).mean(-1)

    def contact_load_cost(self):
        return 1.26 * super().contact_load_cost()

    def reward(self, measured):
        failed = measured['wangyu/failed']
        return (super().reward(measured)
                - self.DT * self.sole_placement_weight * self.sole_placement_cost() * ~failed
                - (.65 + .65 * self.high_speed_weight()) * failed)


class StableFullSoleForwardJumpCommands(FullSoleForwardJumpCommands):
    """Track the zero-speed anchor without treating forward drift as a fall."""

    sole_placement_weight = .70

    def reference_progress(self, progress):
        # Keep the moving course reference for the safety termination.
        return progress

    def stationary_distance_for_reward(self, displacement):
        return displacement.clamp_max(.85)

    def observation(self, state):
        drift_column = 88  # Along-heading drift, before cross-track drift.
        drift = ((self.worlds.sim.data.qpos[:, :2] - self.origin) * self.heading).sum(-1)
        anchor = (drift / .3).clamp(-1.5, 1.5)
        state = state.clone()
        state[:, drift_column] = torch.lerp(state[:, drift_column], anchor, self.stationary_weight())
        return super().observation(state)

    def horizontal_rate(self):
        velocity = self.worlds.sim.data.sensordata[:, self.com_velocity_xy]
        drift = ((self.worlds.sim.data.sensordata[:, self.com_xy] - self.origin) * self.heading).sum(-1)
        forward = (velocity * self.heading).sum(-1)
        correction = (-.60 * drift).clamp(-.20, .20)
        stationary_tracking = torch.exp(-((forward - correction) / .16).square())
        fast_error = forward - self.target_speed
        speed_tracking = torch.exp(-(fast_error / .40).square())
        return (super().horizontal_rate() + 6. * stationary_tracking * self.stationary_weight()
                + 7. * speed_tracking * self.high_speed_weight())

    def reward(self, measured):
        # Keep the compression-to-push pattern competitive with changes in
        # foot placement and speed at the upper end of the command range.
        return (super().reward(measured)
                + .85 * self.DT * self.stance_flow_rate * ~measured['wangyu/failed'])


class BalancedFullSoleForwardJumpCommands(FullSoleForwardJumpCommands):
    """Balance endpoint tracking, grounded propulsion and boot placement."""

    sole_placement_weight = .85

    def sample_speed_fraction(self):
        # Keep the interior uniform and expose both endpoint commands often.
        uniform = torch.rand(self.worlds.count, generator=self.speed_generator, device=self.age.device)
        return ((uniform - .23) / .54).clamp(0., 1.)

    def reference_progress(self, progress):
        return progress

    def stationary_distance_for_reward(self, displacement):
        return displacement.clamp_max(.50)

    def horizontal_rate(self):
        velocity = self.worlds.sim.data.sensordata[:, self.com_velocity_xy]
        forward = (velocity * self.heading).sum(-1)
        cross = velocity[:, 1] * self.heading[:, 0] - velocity[:, 0] * self.heading[:, 1]
        stationary_fit = torch.exp(-velocity.square().sum(-1) / .14 ** 2)
        fast_fit = torch.exp(-((forward - self.target_speed) / .42).square())
        return (super().horizontal_rate() + 15. * self.stationary_weight() * stationary_fit
                + self.high_speed_weight() * (8. * fast_fit - 4. * cross.square()))

    def stride_reward(self):
        event_weight = self.hop_event_weight()
        return (super().stride_reward()
                + 2.7 * self.completed * event_weight * self.takeoff_flow_quality)


class DistributedSoleForwardJumpCommands(BalancedFullSoleForwardJumpCommands):
    """Favor real heel-to-forefoot support and leave a margin above a collapse."""

    sole_placement_weight = .62

    def sole_region_loads(self):
        probes = self.sole_probes()
        heel, toe = probes.unbind(dim=2)
        axis = (toe - heel) / 2.
        center = (toe + heel) / 2.
        sensors = self.worlds.sim.data.sensordata
        contact = sensors[:, self.worlds.device.contact_indices].reshape(-1, 2, 4, 8)
        relative = contact[..., 5:8] - center[:, :, None]
        region = (relative * axis[:, :, None]).sum(-1) / axis.square().sum(-1, keepdim=True)
        valid = (contact[..., 0] > 0) & (contact[..., 4] <= .002)
        normal = contact[..., 1].abs() / self.worlds.device.weight * valid
        heel = (normal * (region < -.33)).sum(-1).clamp(0., .45) / .45
        forefoot = (normal * (region > .33)).sum(-1).clamp(0., .45) / .45
        return heel, forefoot

    def supported_sole_rate(self):
        heel, forefoot = self.sole_region_loads()
        compressing = (self.ground_frames > 0) & (self.stage != self.EXTEND)
        return (2.2 * heel + 5.0 * heel.minimum(forefoot)).mean(-1) * compressing

    def reward(self, measured):
        live = ~measured['wangyu/failed']
        root_clearance_column = 90
        clearance = measured['wangyu/state'][:, root_clearance_column]
        low = ((.79 * self.worlds.device.q0[2] - clearance) / .14).clamp(0., 1.5)
        fast = self.high_speed_weight()
        support = self.supported_sole_rate() + 1.10 * self.stance_flow_rate
        return super().reward(measured) + self.DT * (support - 8. * fast * low.square()) * live


class TerrainAlignedLandingCommands(DistributedSoleForwardJumpCommands):
    """Blend a bounded ankle correction around touchdown using the local slope."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        self.ankle_actions = [worlds.device.action_index(f'robot/{side}Ankle')
                              for side in ('Left', 'Right')]
        self.ankle_qpos = [int(worlds.model.joint(f'robot/{side}Ankle').qposadr[0])
                           for side in ('Left', 'Right')]
        self.sole_span = torch.as_tensor([1.70 * worlds.model.geom(f'robot/{side}_sole').size[0]
                                          for side in ('left', 'right')], dtype=self.age.dtype, device=self.age.device)

    def __call__(self, actions, phase=None):
        commands = super().__call__(actions, phase)
        gap = self.sole_clearances()
        pitch_error = (gap[..., 0] - gap[..., 1]) / self.sole_span
        vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        descent = ((-vz - .02) / .30).clamp(0., 1.)
        near = ((.24 - gap.amin(-1)) / .20).clamp(0., 1.)
        flying = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        supporting = (self.ground_frames > 0) & (self.stage != self.WAIT)
        release = (self.stage == self.EXTEND) * (self.age / .11).clamp(0., 1.)
        influence = flying[:, None] * descent[:, None] * near
        influence += (supporting * (1. - release))[:, None] * near
        # Positive ankle rotation points the toe down. Use the actuator's
        # original position target and limits, with a small slope correction.
        ankles = commands[:, self.ankle_actions]
        commands[:, self.ankle_actions] = self.corrected_ankles(ankles, pitch_error, influence)
        return commands

    def corrected_ankles(self, ankles, pitch_error, influence):
        correction = (-.95 * pitch_error).clamp(-.22, .22) * influence
        scale = self.worlds.device.action_scale[self.ankle_actions]
        return ankles + correction / scale


class LowSpeedSoleLandingCommands(TerrainAlignedLandingCommands):
    """Allow greater ankle alignment where no forward propulsion is requested."""

    def corrected_ankles(self, ankles, pitch_error, influence):
        baseline = super().corrected_ankles(ankles, pitch_error, influence)
        device = self.worlds.device
        indices = self.ankle_actions
        default = device.default_q[indices]
        scale = device.action_scale[indices]
        requested = default + scale * ankles.clamp(-2., 2.)
        target = requested.clamp(device.action_lo[indices], device.action_hi[indices])
        current = self.worlds.sim.data.qpos[:, self.ankle_qpos]
        slope_target = current - pitch_error.atan()
        adjustment = (slope_target - target).clamp(-.46, .46)
        aligned = (target + adjustment * influence - default) / scale
        slow = (1. - self.target_speed / 1.2).clamp(0., 1.)
        return torch.lerp(baseline, aligned, slow[:, None])


class BilateralSoleLandingCommands(TerrainAlignedLandingCommands):
    """Coordinate left and right landings while jumping toward the torso front."""

    sole_placement_weight = .70
    required_sole_gap = .040

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        model = worlds.model
        self.separation_column = self.sensor_columns(model, 'sole_separation', 1).start
        self.hip_roll_actions = [worlds.device.action_index(f'robot/{side}HipRoll')
                                 for side in ('Left', 'Right')]
        self.outward = torch.tensor([1., -1.], device=self.age.device)

    def sole_separation_deficit(self):
        gap = self.worlds.sim.data.sensordata[:, self.separation_column]
        return ((self.required_sole_gap - gap) / self.required_sole_gap).clamp(0., 2.)

    def __call__(self, actions, phase=None):
        commands = super().__call__(actions, phase)
        indices = self.hip_roll_actions
        device = self.worlds.device
        scale = device.action_scale[indices]
        default = device.default_q[indices]
        raw = commands[:, indices]
        target = (default + scale * raw.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        # Hip roll opens the legs in the body frame as the two boots approach.
        stance_gain = torch.where(self.ground_frames > 0, .60, 1.)
        correction = .085 * self.sole_separation_deficit() * stance_gain
        commands[:, indices] = (target + correction[:, None] * self.outward - default) / scale
        return commands

    def corrected_ankles(self, ankles, pitch_error, influence):
        baseline = super().corrected_ankles(ankles, pitch_error, influence)
        # Increase dorsiflexion only on the foot that is too toe-down for its
        # own slope and further from flat than the other foot.
        pitch = pitch_error.atan()
        lag = (pitch - pitch.flip(-1).abs() - .045).clamp(0., .26)
        device = self.worlds.device
        indices = self.ankle_actions
        scale = device.action_scale[indices]
        default = device.default_q[indices]
        target = (default + scale * baseline.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        adjusted = (target - lag * influence - default) / scale
        return torch.where(lag * influence > 0., adjusted, baseline)

    def torso_forward_xy(self):
        w, x, y, z = self.worlds.sim.data.sensordata[:, self.torso_quat].unbind(-1)
        direction = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (w * z + x * y)), -1)
        return torch.nn.functional.normalize(direction, dim=-1, eps=1e-6)

    def torso_travel_cost(self):
        direction = self.torso_forward_xy()
        velocity = self.worlds.sim.data.sensordata[:, self.com_velocity_xy]
        along = (direction * velocity).sum(-1)
        cross = direction[:, 0] * velocity[:, 1] - direction[:, 1] * velocity[:, 0]
        moving = ((self.target_speed - .04) / .28).clamp(0., 1.)
        # A cosine term still distinguishes backwards from forward, and the
        # squared lateral speed penalizes the actual sideways motion.
        speed = velocity.norm(dim=-1)
        projected_direction = (along / speed.clamp_min(.05)).clamp(-1., 1.)
        return moving * (5.5 * cross.square() + 5. * (1. - projected_direction) * (speed / .45).clamp(0., 1.))

    def sole_placement_cost(self):
        base = super().sole_placement_cost()
        gap = self.sole_clearances()
        toe_down = ((gap[..., 0] - gap[..., 1] - .014) / .075).clamp(0., 2.)
        vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        descending = (self.stage == self.FLIGHT) & (self.ground_frames == 0) & (vz < -.10)
        near = ((.19 - gap.amin(dim=-1)) / .15).clamp(0., 1.)
        support = (self.ground_frames > 0) & (self.stage == self.COMPRESS)
        worst = (toe_down.square() * near).amax(-1)
        return base + (2.0 * descending + 3.0 * support) * worst

    def supported_sole_rate(self):
        heel, forefoot = self.sole_region_loads()
        compressing = (self.ground_frames > 0) & (self.stage != self.EXTEND)
        foot_support = 2.2 * heel + 5.0 * heel.minimum(forefoot)
        return (foot_support.mean(-1) + 2.5 * heel.amin(-1)) * compressing

    def reward(self, measured):
        live = ~measured['wangyu/failed']
        avoidance = 7. * self.sole_separation_deficit().square() + self.torso_travel_cost()
        return super().reward(measured) - self.DT * avoidance * live


class CoordinatedLandingCommands(BilateralSoleLandingCommands):
    """Bring the landing knee angles together before either boot takes the load."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        self.knee_actions = [worlds.device.action_index(f'robot/{side}Knee')
                             for side in ('Left', 'Right')]

    def landing_weight(self, gap):
        vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        descending = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        approach = descending * ((-vz - .06) / .35).clamp(0., 1.)
        landing = (self.stage == self.COMPRESS) * (1. - self.age / .16).clamp(0., 1.)
        near = ((.27 - gap.amin(dim=-1)) / .20).clamp(0., 1.)
        return (approach + landing)[:, None] * near

    def corrected_ankles(self, ankles, pitch_error, influence):
        baseline = super().corrected_ankles(ankles, pitch_error, influence)
        device = self.worlds.device
        data = self.worlds.sim.data
        indices = self.ankle_actions
        scale = device.action_scale[indices]
        neutral = device.default_q[indices]
        target = (neutral + scale * baseline.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        pitch = pitch_error.atan()
        lag = ((pitch - pitch.flip(-1) - .05) / .16).clamp(0., 1.)
        toe_down = ((pitch - .075) / .13).clamp(0., 1.)
        slope_target = data.qpos[:, self.ankle_qpos] - pitch
        extra = (target - slope_target).clamp(0., .35)
        return baseline - extra * lag * toe_down * influence / scale

    def __call__(self, actions, phase=None):
        commands = super().__call__(actions, phase)
        data = self.worlds.sim.data
        device = self.worlds.device
        knees = data.qpos[:, self.knee_qpos]
        gap = self.sole_clearances()
        downturned = ((gap[..., 0] - gap[..., 1] - .035) / .08).clamp(0., 1.)
        extra_bend = (knees - knees.flip(-1) - .18).clamp(0., .20)
        correction = extra_bend * downturned * self.landing_weight(gap)
        indices = self.knee_actions
        scale = device.action_scale[indices]
        neutral = device.default_q[indices]
        current = commands[:, indices]
        target = (neutral + scale * current.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        target = (target - correction - neutral) / scale
        commands[:, indices] = torch.where(correction > 0., target, current)
        return commands

    def reward(self, measured):
        knees = self.worlds.sim.data.qpos[:, self.knee_qpos]
        mismatch = ((knees[:, 0] - knees[:, 1]).abs() - .16).clamp_min(0.)
        landing = self.landing_weight(self.sole_clearances()).amax(-1)
        penalty = 7.0 * landing * (mismatch / .43).clamp_max(2.).square()
        penalty += 4. * self.sole_separation_deficit().square()
        return super().reward(measured) - self.DT * penalty * ~measured['wangyu/failed']


class MirroredLandingCommands(BilateralSoleLandingCommands):
    """Use a soft sagittal mirror for paired joints before terrain adaptation."""

    required_sole_gap = .050

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        joints = ('HipRoll', 'HipPitch', 'HipYaw', 'Knee', 'Ankle', 'AnkleRoll',
                  'ShoulderRoll', 'ShoulderPitch', 'ShoulderYaw', 'Elbow')
        model = worlds.model
        pairs = [[model.joint(f'robot/{side}{joint}') for joint in joints]
                 for side in ('Left', 'Right')]
        self.mirror_actions = torch.tensor([[worlds.device.action_index(joint.name) for joint in row] for row in pairs], device=self.age.device)
        self.mirror_qpos = torch.tensor([[int(joint.qposadr[0]) for joint in row] for row in pairs], device=self.age.device)
        self.mirror_sign = torch.tensor([-1., 1., -1., 1., 1., -1., -1., 1., -1., 1.], device=self.age.device)
        # Radians of free side-to-side error before the soft constraint begins.
        self.mirror_tolerance = torch.tensor([.15, .12, .14, .16, .19, .18, .22, .23, .23, .22], device=self.age.device)
        self.mirror_gain = torch.tensor([.62, .76, .65, .76, .62, .64, .50, .50, .45, .50], device=self.age.device)
        self.mirror_neutral = torch.as_tensor(model.qpos0, dtype=self.age.dtype, device=self.age.device)[self.mirror_qpos]

    def mirror_joint_targets(self, actions, strength=1.):
        indices = self.mirror_actions
        device = self.worlds.device
        neutral = device.default_q[indices]
        scale = device.action_scale[indices]
        raw = actions[:, indices]
        target = (neutral + scale * raw.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        angles = target - neutral
        left, right = angles.unbind(dim=1)
        difference = left - self.mirror_sign * right
        # Leave the deadband and let the policy retain some extra asymmetry.
        excess = (difference.abs() - self.mirror_tolerance).clamp_min(0.)
        correction = .5 * self.mirror_gain * strength * difference.sign() * excess
        adjusted = torch.stack((left - correction, right + self.mirror_sign * correction), dim=1)
        coupled = actions.clone()
        paired = (correction != 0.)[:, None]
        coupled[:, indices] = torch.where(paired, adjusted / scale, raw)
        return coupled

    def __call__(self, actions, phase=None):
        return super().__call__(self.mirror_joint_targets(actions), phase)

    def mirror_motion_cost(self):
        positions = self.worlds.sim.data.qpos[:, self.mirror_qpos] - self.mirror_neutral
        left, right = positions.unbind(dim=1)
        error = ((left - self.mirror_sign * right).abs() - self.mirror_tolerance).clamp_min(0.)
        lower = (error[:, :6] / .45).clamp_max(2.).square().mean(-1)
        arms = (error[:, 6:] / .45).clamp_max(2.).square().mean(-1)
        descending = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop] < -.08
        landing = ((self.stage == self.FLIGHT) & descending) | (self.stage == self.COMPRESS)
        return (5.5 * lower + 4.5 * landing * lower + 2.3 * arms) * (self.stage != self.WAIT)

    def reward(self, measured):
        return super().reward(measured) - self.DT * self.mirror_motion_cost() * ~measured['wangyu/failed']


class PhaseMirroredLandingCommands(MirroredLandingCommands):
    """Focus action coupling on landing preparation and initial support."""

    def mirror_joint_targets(self, actions, strength=1.):
        vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        descending = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        approach = descending * ((-vz - .04) / .32).clamp(0., 1.)
        absorbing = (self.stage == self.COMPRESS) * (1. - self.age / .20).clamp(0., 1.)
        # Leave more freedom in the push, where the sides can unload at
        # slightly different times on rough ground.
        phase = .23 + .48 * (approach + absorbing)
        return super().mirror_joint_targets(actions, strength=phase[:, None])


class SolePlaneMirroredLandingCommands(PhaseMirroredLandingCommands):
    """Use the local fore-aft and cross-foot slope to support both heels."""

    def __init__(self, worlds, **kwargs):
        super().__init__(worlds, **kwargs)
        model = worlds.model
        slices = [self.sensor_columns(model, f'{side}_{edge}_edge_pos', 3)
                  for side in ('left', 'right') for edge in ('negative', 'positive')]
        self.edge_columns = [col for sl in slices for col in range(sl.start, sl.stop)]
        self.sole_width = torch.tensor([1.7 * model.geom(f'robot/{side}_sole').size[1]
                                        for side in ('left', 'right')], dtype=self.age.dtype, device=self.age.device)
        ankle_joints = [[model.joint(f'robot/{side}{joint}')
                         for joint in ('Ankle', 'AnkleRoll')] for side in ('Left', 'Right')]
        self.plane_actions = torch.tensor([[worlds.device.action_index(joint.name) for joint in pair] for pair in ankle_joints], device=self.age.device)
        self.plane_qpos = torch.tensor([[int(joint.qposadr[0]) for joint in pair] for pair in ankle_joints], device=self.age.device)

    def sole_edge_clearances(self):
        sensors = self.worlds.sim.data.sensordata
        points = sensors[:, self.edge_columns].reshape(-1, 2, 2, 3)
        return points[..., 2] - self.worlds.device.ground.height(points[..., :2])

    def __call__(self, actions, phase=None):
        commands = super().__call__(actions, phase)
        data = self.worlds.sim.data
        device = self.worlds.device
        gap = self.sole_clearances()
        edges = self.sole_edge_clearances()
        pitch = ((gap[..., 0] - gap[..., 1]) / self.sole_span).atan()
        roll = ((edges[..., 1] - edges[..., 0]) / self.sole_width).atan()
        vz = data.sensordata[:, self.com_velocity_xy.stop]
        flight = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        approach = flight * ((-vz - .05) / .29).clamp(0., 1.)
        supporting = (self.ground_frames > 0) & (self.stage != self.EXTEND) & (self.stage != self.WAIT)
        loading = supporting * (1. - self.ground_frames * (self.DT / .17)).clamp(0., 1.)
        pitch_near = ((.27 - gap.amin(dim=-1)) / .18).clamp(0., 1.)
        edge_near = ((.27 - edges.amin(dim=-1)) / .18).clamp(0., 1.)
        pitch_need = ((pitch - .045) / .17).clamp(0., 1.)
        roll_need = ((roll.abs() - .045) / .17).clamp(0., 1.)
        # A laterally tilted sole still needs correction when heel and toe are level.
        strength = .68 * (approach + .62 * loading)[:, None, None] * torch.stack(
            (pitch_near * pitch_need, edge_near * roll_need), dim=-1)
        indices = self.plane_actions
        neutral, scale = device.default_q[indices], device.action_scale[indices]
        raw = commands[:, indices]
        target = (neutral + scale * raw.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        aligned = data.qpos[:, self.plane_qpos] - torch.stack((pitch, roll), dim=-1)
        adjust = (aligned - target).clamp(-.50, .50) * strength
        commands[:, indices] = torch.where(adjust != 0., (target + adjust - neutral) / scale, raw)
        return commands

    def contact_load_cost(self):
        slow = (1. - self.target_speed / .90).clamp(0., 1.)
        return super().contact_load_cost() * (1. + .80 * slow)


class ForwardFacingBootCommands(SolePlaneMirroredLandingCommands):
    """Center boot yaw on the torso while preserving the fitted toe-out."""

    def __init__(self, worlds, *, compact_arm_swing=False, suit_ankle_roll=False, suit_ankle_pitch=False,
                 slope_adaptive_posture=False, suit_major_joints=False, relax_propulsion=False, **kwargs):
        self.suit_major_joints = bool(suit_major_joints)
        self.symmetric_startup = (worlds.exoskeleton is not None
                                  and getattr(worlds.exoskeleton, 'mechanism', None)
                                  in ('hut_ball_aft_v6', 'hut_hinge_aft_v7', 'hut_hinge_hoof_v8'))
        # Event-driven 3-stage startup: crouch in place to store energy, launch forward
        # with an immediate 0.75 m/s target, and ramp at 0.75 m/s^2 so full 1.5 m/s
        # long-stride cruise is reached within ~1.0s (right after the first hop).
        self.startup_base_mps = .75
        self.startup_accel_mps2 = .75
        self.startup_ramp_s = 1.5
        if self.suit_major_joints:
            self.first_takeoff_done = worlds.sim.data.qpos.new_zeros(worlds.count, dtype=torch.bool)
            self.first_landing_done = worlds.sim.data.qpos.new_zeros(worlds.count, dtype=torch.bool)
            self.startup_elapsed = worlds.sim.data.qpos.new_zeros(worlds.count)
            self.command_speed = worlds.sim.data.qpos.new_zeros(worlds.count)
            self.startup_foot_origin = worlds.sim.data.qpos.new_zeros((worlds.count, 2))
            self.peak_sole_clearance = worlds.sim.data.qpos.new_zeros(worlds.count)
            self.hip_qpos = [int(worlds.model.joint(f'robot/{side}HipPitch').qposadr[0]) for side in ('Left', 'Right')]
        super().__init__(worlds, **kwargs)
        model = worlds.model
        self.slope_adaptive_posture = bool(slope_adaptive_posture)
        self.relax_propulsion = bool(relax_propulsion)
        if suit_ankle_roll:
            # The final action clip at ±2 then spans the physical roll limit.
            device = worlds.device
            for side in ('Left', 'Right'):
                index = device.action_index(f'robot/{side}AnkleRoll')
                device.action_scale[index] = .5 * device.limits[index].abs().amax()
        if suit_ankle_pitch:
            # Keep enough action authority to span the full [-18, +15] deg
            # pressurized boot range even over crouch/slope priors, while
            # softening the crouch prior to stay comfortably inside -18 deg.
            device = worlds.device
            for side in ('Left', 'Right'):
                index = device.action_index(f'robot/{side}Ankle')
                scale = .35
                device.action_scale[index] = scale
                self.prior[index] = -.14 / scale
        if self.slope_adaptive_posture:
            # Unlock full physical WaistPitch flexion so the suit can lean
            # forward on uphill slopes to land flat within ankle limits.
            device = worlds.device
            waist = device.action_index('robot/WaistPitch')
            device.action_scale[waist] = .35
            device.action_lo[waist] = max(float(device.limits[waist, 0]), -.55)
            device.action_hi[waist] = min(float(device.limits[waist, 1]), .61)
        if self.suit_major_joints:
            device = worlds.device
            # Match waist and hip-yaw/roll action ranges to pressurized suit limits
            # so actions map smoothly across the stiffer EVA joint ROM.
            for name in ('robot/WaistYaw', 'robot/WaistRoll', 'robot/WaistPitch',
                         'robot/LeftHipYaw', 'robot/RightHipYaw',
                         'robot/LeftHipRoll', 'robot/RightHipRoll'):
                idx = device.action_index(name)
                lo, hi = float(device.limits[idx, 0]), float(device.limits[idx, 1])
                device.action_lo[idx] = lo
                device.action_hi[idx] = hi
                device.action_scale[idx] = max(.5 * max(abs(lo), abs(hi)), .15)
            self.waist_qpos = [int(model.joint(f'robot/Waist{a}').qposadr[0]) for a in ('Yaw', 'Roll', 'Pitch')]
            self.waist_dofs = [int(model.joint(f'robot/Waist{a}').dofadr[0]) for a in ('Yaw', 'Roll', 'Pitch')]
            self.hip_pitch_actions = [device.action_index(f'robot/{side}HipPitch') for side in ('Left', 'Right')]
            self.knee_pitch_actions = [device.action_index(f'robot/{side}Knee') for side in ('Left', 'Right')]
        if self.relax_propulsion:
            device = worlds.device
            # Allow full physical HipPitch range and slight Knee extension
            # overdrive (-0.18 rad) so fast takeoff push-off delivers full leg
            # torque without hitting control clamps.
            knee_max = 1.22 if self.suit_major_joints else 1.45
            # Under 1/6g lunar gravity, a 1.0 rad (57 deg) knee crouch prior
            # bends faster than gravity can lower the body on startup, yanking
            # the feet off the ground into a deep mid-air tuck. Match the crouch
            # prior to lunar gravity when suit_major_joints is enabled.
            knee_prior = .55 if self.suit_major_joints else 1.
            hip_prior = -.45 if self.suit_major_joints else -.5
            for side in ('Left', 'Right'):
                hip = device.action_index(f'robot/{side}HipPitch')
                device.action_scale[hip] = .65
                device.action_lo[hip] = max(float(device.limits[hip, 0]), -1.15)
                device.action_hi[hip] = min(float(device.limits[hip, 1]), .78)
                self.prior[hip] = hip_prior / .65
                knee = device.action_index(f'robot/{side}Knee')
                device.action_scale[knee] = .75
                device.action_lo[knee] = -.18
                device.action_hi[knee] = min(float(device.limits[knee, 1]), knee_max)
                self.prior[knee] = knee_prior / .75
        self.hip_yaw_actions = [worlds.device.action_index(f'robot/{side}HipYaw')
                                for side in ('Left', 'Right')]
        self.bunny_hop_mirror = (worlds.exoskeleton is not None
                                 and getattr(worlds.exoskeleton, 'mechanism', None) == 'hut_ring_clevis_v5')
        if self.bunny_hop_mirror:
            # True two-footed lunar bunny hop for HUT-mounted bilateral struts:
            # tighten sagittal leg mirror deadband (still leaving ±3.5-5 deg for uneven rocks).
            self.mirror_tolerance = torch.tensor([.12, .06, .12, .08, .12, .12, .22, .23, .23, .22], device=self.age.device)
            self.mirror_gain = torch.tensor([.75, .90, .75, .90, .75, .75, .50, .50, .45, .50], device=self.age.device)
        if self.symmetric_startup:
            self.startup_mirror_tol = torch.tensor([.10, .07, .10, .08, .12, .12, .20, .20, .20, .20], device=self.age.device)
            self.startup_mirror_gain = torch.tensor([.85, .94, .85, .92, .75, .75, .50, .50, .45, .50], device=self.age.device)
        self.compact_arm_swing = worlds.exoskeleton is not None and (
            worlds.exoskeleton.position_pd or compact_arm_swing)
        if self.compact_arm_swing:
            self.shoulder_actions = [
                worlds.device.action_index(f'robot/{side}Shoulder{axis}')
                for side in ('Left', 'Right') for axis in ('Pitch', 'Roll', 'Yaw')
            ]
            # Draw the upper arms inward and leave room for balance corrections.
            self.shoulder_trim = self.age.new_tensor([0., -.14, 0., 0., .14, 0.])
            self.shoulder_gain = self.age.new_tensor([.38, .42, .70] * 2)
        boot_rotation = torch.as_tensor(
            [model.geom(f'robot/{side}_sole').quat.tolist() for side in ('left', 'right')],
            dtype=self.age.dtype, device=self.age.device)
        w, x, y, z = boot_rotation.unbind(-1)
        self.neutral_toe_out = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))

    def terrain_slope(self):
        xy = self.worlds.sim.data.sensordata[:, self.com_xy]
        fwd = self.heading
        # Slight lookahead (+0.55 m ahead vs -0.35 m behind) so the posture
        # anticipates uphill/downhill landings during descent as well as stance.
        h_ahead = self.worlds.device.ground.height(xy + .55 * fwd)
        h_behind = self.worlds.device.ground.height(xy - .35 * fwd)
        return torch.atan2(h_ahead - h_behind, torch.full_like(h_ahead, .90)).clamp(-.42, .42)

    def slope_torso_shift(self):
        if not self.slope_adaptive_posture:
            return 0.
        slope = self.terrain_slope()
        # Lean forward prominently on uphill slopes (+2.0 * slope) so both
        # pelvis and waist pitch into the hill, shifting the COM over the
        # midfoot and flexing the hips enough for -18 deg stiff boots to land
        # flat even during long bounding leaps; stay upright / lean back on
        # downhill slopes (+1.1 * slope). Gentler when nearly stationary.
        moving = .65 + .35 * (self.target_speed / .50).clamp(0., 1.)
        return moving * torch.where(slope >= 0., 2.0 * slope, 1.1 * slope)

    def posture_pitch(self, quat):
        pitch = self.forward_pitch(quat)
        return pitch - self.slope_torso_shift() if self.slope_adaptive_posture else pitch

    def lean_cost(self, torso, pelvis, vz):
        if not self.slope_adaptive_posture:
            return super().lean_cost(torso, pelvis, vz)
        # Add speed-proportional forward lean (+4.6 deg at 1.5 m/s) plus dynamic
        # acceleration lean (+5.7 deg max while accelerating from rest) so fast
        # lunar bounding launches forward instead of leaning backward on takeoff.
        com_vel = self.worlds.sim.data.sensordata[:, self.com_velocity_xy]
        forward = (com_vel * self.heading).sum(-1)
        shortfall = (self.target_speed - forward).clamp_min(0.)
        accel_gain = .18 if (self.symmetric_startup and self.suit_major_joints) else (.07 if self.symmetric_startup else .16)
        accel_lean = accel_gain * (shortfall / 1.2).clamp(0., 1.) * (self.target_speed > .25)
        speed_lean = .08 * (self.target_speed / 1.5).clamp(0., 1.) + accel_lean
        reference = self.torso_lean_reference(vz) + speed_lean
        in_pre_takeoff = (self.symmetric_startup and self.suit_major_joints)
        if in_pre_takeoff:
            # Physically, the heavy Wangyu PLSS backpack puts standing COM at -12 cm (-6.6 deg,
            # just 9 mm ahead of the heel edge). Shifting COM over the forefeet (+3..+9 deg)
            # while keeping boots flat (ankle -17 deg, knee 35 deg, hip -45 deg) geometrically
            # requires torso_pitch ~ 28..38 deg (0.49..0.66 rad).
            pre_to = (~self.first_takeoff_done) & (self.command_speed > .20)
            pre_ref = torch.where(self.stage == self.EXTEND, .38, .54)
            reference = torch.where(pre_to, pre_ref, reference)
        in_flight = self.stage == self.FLIGHT
        slope = self.terrain_slope()
        uphill = (slope / .14).clamp(0., 1.)
        # Guide torso cleanly along the slope-shifted reference (±3.2 deg
        # deadband) so uphill forward lean and downhill upright/back posture
        # stay prominent even during powerful long-stride bounding hops.
        error = ((torso - reference).abs() - .055).clamp_min(0.)
        excessive_extension = (pelvis - torso - .24).clamp_min(0.)
        # On flat/downhill keep natural backpack balance (waist up to ~22 deg);
        # on uphill tighten waist-pelvis lag and guide pelvis forward so hips
        # flex and shins stay within the -18 deg ankle dorsiflexion budget.
        lag_allow = .38 - .16 * uphill
        pelvis_lag = (torso - pelvis - lag_allow).clamp_min(0.)
        pelvis_uphill = uphill * (-pelvis - .18).clamp_min(0.)
        backwards = (-torso - .22).clamp_min(0.)
        com_vz = self.worlds.sim.data.sensordata[:, self.com_velocity_xy.stop]
        next_to_takeoff = ((com_vz - .06) / .30).clamp(0., 1.) * (self.stage == self.EXTEND)
        early_flight = (1. - self.age / .18).clamp(0., 1.) * in_flight * (com_vz > 0.)
        missed_opening = (torso - speed_lean + .04).clamp_min(0.)
        uphill_relief = (1. - slope.clamp_min(0.) / .20).clamp(.15, 1.)
        accel_relief = (1. - shortfall / .60).clamp(.15, 1.)
        takeoff_cost = uphill_relief * accel_relief * (next_to_takeoff + early_flight) * (
            3. * missed_opening + 12. * missed_opening.square())
        if in_pre_takeoff:
            takeoff_cost = torch.where(pre_to, torch.zeros_like(takeoff_cost), takeoff_cost)
        cost = (torch.where(in_flight, 120., 110.) * error.square()
                + 14. * excessive_extension.square()
                + (12. + 32. * uphill) * pelvis_lag.square()
                + 55. * pelvis_uphill.square()
                + 16. * backwards.square() + takeoff_cost)
        if self.symmetric_startup and self.suit_major_joints:
            # Allow physically required torso forward lean (~38 deg) before first takeoff so
            # the heavy backpack COM moves forward over planted feet, then return to upright
            # cruise dive limits (16 deg) once airborne / cruising.
            in_first_hop = (~self.first_landing_done) & (self.startup_elapsed < self.startup_ramp_s)
            startup_blend = in_first_hop.float()
            dive_lim = torch.where(pre_to, .68, .28)
            startup_dive = (torso - dive_lim).clamp_min(0.)
            waist_arch = (pelvis - torso - .08).clamp_min(0.)
            cruise_dive = torch.where(pre_to, (torso - .68).clamp_min(0.), (torso - .28).clamp_min(0.))
            cost = (cost + 65. * startup_blend * startup_dive.square()
                    + (25. + 30. * startup_blend) * waist_arch.square()
                    + 40. * cruise_dive.square())
        return cost

    def sole_placement_cost(self):
        if not self.slope_adaptive_posture:
            return super().sole_placement_cost()
        data = self.worlds.sim.data
        gap = self.sole_clearances()
        uphill = (self.terrain_slope() / .14).clamp(0., 1.)
        uneven = ((gap[..., 0] - gap[..., 1]).abs() - .014).clamp_min(0.)
        u = uneven / .055
        # Smooth quadratic-to-linear Huber penalty: never saturates at large toe-down angles.
        error = torch.where(u <= 1.5, u.square(), 3. * u - 2.25)
        toe_down = ((gap[..., 0] - gap[..., 1] - .014) / .055).clamp_min(0.)
        toe_error = torch.where(toe_down <= 1.5, toe_down.square(), 3. * toe_down - 2.25)
        vz = data.sensordata[:, self.com_velocity_xy.stop]
        coming_down = ((-vz - .06) / .35).clamp(0., 1.)
        near_ground = ((.22 - gap.amin(-1)) / .18).clamp(0., 1.)
        # Cover both flight descent and accidental ungrounded COMPRESS (e.g.
        # rapid knee tuck right after spawn under 1/6g) so boots cannot point
        # steeply downward with zero penalty.
        ungrounded_compress = (self.stage == self.COMPRESS) & (self.ground_frames == 0)
        approach = ((self.stage == self.FLIGHT) | ungrounded_compress) & (self.ground_frames == 0)
        prep_weight = torch.where(ungrounded_compress, torch.ones_like(coming_down), coming_down)
        load = data.sensordata[:, self.sole_load_columns] / self.worlds.device.weight
        foot_loaded = ((load - .02) / .43).clamp(0., 1.)
        late_push = (self.stage == self.EXTEND) * (self.age / .13).clamp(0., 1.)
        stance = (self.ground_frames > 0) & (self.stage != self.WAIT)
        preparation = 8. * approach[:, None] * prep_weight[:, None] * near_ground
        support = 14. * stance[:, None] * (1. - late_push[:, None]) * foot_loaded
        base = ((preparation + support) * error).mean(-1)
        descending = (approach & (vz < -.10)) | ungrounded_compress
        compress = (self.ground_frames > 0) & (self.stage == self.COMPRESS)
        worst_toe = (toe_error * near_ground).amax(-1)
        return base + (3.2 * descending + (5.5 + 4.5 * uphill) * compress) * worst_toe

    def supported_sole_rate(self):
        rate = super().supported_sole_rate()
        if not self.slope_adaptive_posture:
            return rate
        uphill = (self.terrain_slope() / .14).clamp(0., 1.)
        return (1.5 + 1.5 * uphill) * rate

    def boot_yaw_error(self):
        heel, toe = self.sole_probes()[..., :2].unbind(dim=2)
        direction = toe - heel
        torso = self.torso_forward_xy()
        dot = (torso[:, None] * direction).sum(-1)
        cross = torso[:, 0, None] * direction[..., 1] - torso[:, 1, None] * direction[..., 0]
        offset = torch.atan2(cross, dot) - self.neutral_toe_out
        return torch.atan2(offset.sin(), offset.cos())

    def mirror_joint_targets(self, actions, strength=1.):
        if not (self.symmetric_startup and self.suit_major_joints):
            return super().mirror_joint_targets(actions, strength)
        coupled = super().mirror_joint_targets(actions, strength)
        opening = ~self.first_takeoff_done
        transition = self.first_takeoff_done & (~self.first_landing_done) & (self.startup_elapsed < self.startup_ramp_s)
        # Keep gentle symmetric coupling (0.35, with 4 deg terrain-adaptation deadband) at
        # cruise so both legs and both exoskeleton struts land and push off together.
        extra_strength = torch.where(opening, 1., torch.where(transition, .55, .35))
        indices = self.mirror_actions
        device = self.worlds.device
        neutral = device.default_q[indices]
        scale = device.action_scale[indices]
        raw = coupled[:, indices]
        target = (neutral + scale * raw.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        angles = target - neutral
        left, right = angles.unbind(dim=1)
        difference = left - self.mirror_sign * right
        excess = (difference.abs() - self.startup_mirror_tol).clamp_min(0.)
        correction = .5 * self.startup_mirror_gain * extra_strength[:, None] * difference.sign() * excess
        adjusted = torch.stack((left - correction, right + self.mirror_sign * correction), dim=1)
        out = coupled.clone()
        paired = (correction != 0.)[:, None]
        out[:, indices] = torch.where(paired, adjusted / scale, raw)
        return out

    def mirror_motion_cost(self):
        base = super().mirror_motion_cost()
        if not (self.symmetric_startup and self.suit_major_joints):
            return base
        positions = self.worlds.sim.data.qpos[:, self.mirror_qpos] - self.mirror_neutral
        left, right = positions.unbind(dim=1)
        err = ((left - self.mirror_sign * right).abs() - self.startup_mirror_tol).clamp_min(0.)
        lower_err = (err[:, :6] / .35).clamp_max(2.5).square().mean(-1)
        hip_knee_err = (err[:, (1, 3)] / .25).clamp_max(3.).square().mean(-1)
        opening = (~self.first_takeoff_done).float()
        transition = (self.first_takeoff_done & (~self.first_landing_done) & (self.startup_elapsed < self.startup_ramp_s)).float()
        steady = ((opening == 0.) & (transition == 0.)).float()
        # Enforce strong symmetry during launch and gentle two-foot bounding symmetry at cruise
        # (retaining the 4 deg deadband for rough lunar slopes) to eliminate single-leg galloping.
        startup_cost = (opening * (10. * lower_err + 12. * hip_knee_err)
                        + transition * (5. * lower_err + 6. * hip_knee_err)
                        + steady * (3.0 * lower_err + 4.5 * hip_knee_err))
        return base + startup_cost

    def __call__(self, actions, phase=None):
        commands = super().__call__(actions, phase)
        error = self.boot_yaw_error()
        difference = error.sign() * (error.abs() - .055).clamp(0., .30)
        flying = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        # Make the yaw correction primarily in the air so it does not twist a
        # loaded boot against the ground.
        gain = .62 * flying + .10 * (self.stage == self.COMPRESS)
        indices = self.hip_yaw_actions
        device = self.worlds.device
        raw = commands[:, indices]
        neutral, scale = device.default_q[indices], device.action_scale[indices]
        target = (neutral + scale * raw.clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        correction = difference * gain[:, None]
        commands[:, indices] = torch.where(correction != 0., (target - correction - neutral) / scale, raw)
        if self.compact_arm_swing:
            indices = self.shoulder_actions
            neutral, scale = device.default_q[indices], device.action_scale[indices]
            target = (neutral + scale * commands[:, indices].clamp(-2., 2.)).clamp(
                device.action_lo[indices], device.action_hi[indices])
            commands[:, indices] = (self.shoulder_trim + self.shoulder_gain * (target - neutral)) / scale
        if self.suit_major_joints:
            # During the opening crouch before first takeoff, allow full athletic
            # knee flexion (<= 52 deg) to lower COM and store elastic/muscular energy
            # while keeping feet grounded and preventing premature backward hip swing.
            opening = ~self.first_takeoff_done
            k_idx = self.knee_pitch_actions
            k_neutral, k_scale = device.default_q[k_idx], device.action_scale[k_idx]
            k_target = (k_neutral + k_scale * commands[:, k_idx].clamp(-2., 2.)).clamp(
                device.action_lo[k_idx], device.action_hi[k_idx])
            k_capped = k_target.clamp_max(.90)
            commands[:, k_idx] = torch.where(
                opening[:, None], (k_capped - k_neutral) / k_scale, commands[:, k_idx])
            opening_pre_push = opening & (self.stage != self.EXTEND)
            h_idx = self.hip_pitch_actions
            h_neutral, h_scale = device.default_q[h_idx], device.action_scale[h_idx]
            h_target = (h_neutral + h_scale * commands[:, h_idx].clamp(-2., 2.)).clamp(
                device.action_lo[h_idx], device.action_hi[h_idx])
            if self.symmetric_startup:
                # Guard against backward kick only until first landing (or startup_ramp_s),
                # while permitting strong forward EXTEND push-off (up to +15 deg).
                in_startup = (~self.first_landing_done) & (self.startup_elapsed < self.startup_ramp_s)
                hip_cap = torch.where(self.stage == self.EXTEND, .26, .15)[:, None]
                h_capped = h_target.minimum(hip_cap)
                commands[:, h_idx] = torch.where(
                    in_startup[:, None], (h_capped - h_neutral) / h_scale, commands[:, h_idx])
            else:
                h_capped = h_target.clamp_max(.14)
                commands[:, h_idx] = torch.where(
                    opening_pre_push[:, None], (h_capped - h_neutral) / h_scale, commands[:, h_idx])
        return commands

    def reset(self, mask=None):
        super().reset(mask)
        if self.suit_major_joints:
            rows = slice(None) if mask is None else mask
            self.first_takeoff_done[rows] = False
            self.first_landing_done[rows] = False
            self.startup_elapsed[rows] = 0.
            self.peak_sole_clearance[rows] = 0.
            self.command_speed[rows] = self.target_speed[rows]
            if self.symmetric_startup:
                self.target_speed[rows] = self.command_speed[rows].clamp_max(self.startup_base_mps)
                foot_mid_xy = self.sole_probes()[..., :2].mean(dim=(1, 2))
                self.startup_foot_origin[rows] = foot_mid_xy[rows]

    def ready_to_push(self, supported, knee_angle, vertical_speed):
        ready = super().ready_to_push(supported, knee_angle, vertical_speed)
        if not (self.symmetric_startup and self.suit_major_joints):
            return ready
        # Before the first forward takeoff, wait until the heavy backpack COM has shifted
        # forward over the planted feet (body_tilt >= +1.5 deg) or a 0.55s cap so the first
        # push-off launches forward instead of jumping straight up and kicking backward.
        pre_launch = (~self.first_takeoff_done) & (self.command_speed > .20)
        probes = self.sole_probes()
        foot_mid = probes.mean(dim=(1, 2))
        com_xyz = self.worlds.sim.data.sensordata[:, self.com_xy.start:self.com_xy.start + 3]
        com_ahead = ((com_xyz[:, :2] - foot_mid[:, :2]) * self.heading).sum(-1)
        body_tilt = torch.atan2(com_ahead, (com_xyz[:, 2] - foot_mid[:, 2]).clamp_min(.25))
        com_ready = (body_tilt >= .025) | (self.age >= .55)
        return torch.where(pre_launch, ready & com_ready, ready)

    def grounded_delay_cost(self):
        cost = super().grounded_delay_cost()
        if self.symmetric_startup and self.suit_major_joints:
            # Do not penalize brief in-place crouch energy storage before the first forward takeoff.
            in_opening_crouch = (~self.first_takeoff_done) & (self.stage == self.COMPRESS) & (self.age < .55)
            return torch.where(in_opening_crouch, torch.zeros_like(cost), cost)
        return cost

    def effective_long_hop_period(self):
        if self.symmetric_startup and self.suit_major_joints and self.long_hop_period is not None:
            # First hop targets a crisp forward launch (~0.95s), then immediately
            # transitions to full long-distance bounding period once first landing completes.
            speed_frac = (self.target_speed / 1.0).clamp(0., 1.)
            cruise_period = .95 + (self.long_hop_period - .95) * speed_frac
            return torch.where(self.first_landing_done, cruise_period, torch.full_like(cruise_period, .95))
        return self.long_hop_period

    def advance(self, measured):
        before = self.stage
        super().advance(measured)
        if self.suit_major_joints:
            self.first_takeoff_done |= (self.stage == self.FLIGHT)
            self.first_landing_done |= self.completed
            self.startup_elapsed.add_(self.DT)
            if self.symmetric_startup:
                v_cap = self.startup_base_mps + self.startup_accel_mps2 * self.startup_elapsed
                self.target_speed.copy_(self.command_speed.minimum(v_cap))
                took_off = (before != self.FLIGHT) & (self.stage == self.FLIGHT)
                flying = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
                min_clear = self.sole_clearances().amin(dim=(-1, -2)).clamp_min(0.)
                torch.where(
                    took_off, min_clear,
                    torch.where(flying, self.peak_sole_clearance.maximum(min_clear), self.peak_sole_clearance),
                    out=self.peak_sole_clearance)

    def stride_reward(self):
        base = super().stride_reward()
        if not (self.symmetric_startup and self.suit_major_joints):
            return base
        # Strongly reward genuine lunar bounding arc height (COM rise ~15..24 cm and
        # both-boot minimum sole clearance ~20..30 cm) on completed hops so the astronaut
        # clears rough lunar terrain cleanly instead of skimming the ground and tripping.
        moving = (self.command_speed / .40).clamp(0., 1.)
        rise_score = (self.peak_com_rise / .18).clamp(0., 1.35)
        clear_score = (self.peak_sole_clearance / .22).clamp(0., 1.35)
        high_arc = self.completed * moving * (24.0 * rise_score + 22.0 * clear_score)
        return base + high_arc

    def waist_motion_cost(self):
        data = self.worlds.sim.data
        q = data.qpos[:, self.waist_qpos]  # Yaw, Roll, Pitch
        v = data.qvel[:, self.waist_dofs]
        # Keep pressurized waist twisting/side-bending calm and realistic.
        yaw_angle = (q[:, 0].abs() - .08).clamp_min(0.)
        roll_angle = (q[:, 1].abs() - .10).clamp_min(0.)
        yaw_vel = (v[:, 0].abs() - .45).clamp_min(0.)
        roll_vel = (v[:, 1].abs() - .55).clamp_min(0.)
        pitch_vel = (v[:, 2].abs() - 1.20).clamp_min(0.)
        return (28. * yaw_angle.square() + 20. * roll_angle.square()
                + 5.5 * yaw_vel.square() + 4.5 * roll_vel.square() + 2.5 * pitch_vel.square())

    def natural_leg_cost(self):
        data = self.worlds.sim.data
        knee = data.qpos[:, self.knee_qpos]  # (W, 2)
        probes = self.sole_probes()
        heel, toe = probes.unbind(dim=2)
        # Positive sole_down means heel above toe (toe pointing toward ground).
        sole_down = ((heel[..., 2] - toe[..., 2]) / self.sole_span).clamp(-1., 1.).asin()
        in_extend = self.stage == self.EXTEND
        # Prevent feet from tilting nearly vertical (-55..-65 deg) during
        # startup, compression, or flight, while allowing natural athletic
        # toe-roll (up to ~35.5 deg) during powerful forward EXTEND push-off.
        allow = torch.where(in_extend, .62, .38)[:, None]
        steep_sole = (sole_down - allow).clamp_min(0.)
        sole_weight = torch.where(in_extend, 35., torch.where(self.first_takeoff_done, 85., 255.))
        sole_cost = sole_weight * steep_sole.square().amax(-1)
        # Prevent rapid leg tucking off the ground right after spawn during COMPRESS.
        ungrounded_compress = ((self.stage == self.COMPRESS) & (self.ground_frames == 0)).float()
        tuck_cost = ungrounded_compress * (20. + 60. * (knee - .30).clamp_min(0.).square().mean(-1))
        # Allow deep athletic crouch (<= 52 deg) both for opening energy storage and fast landings.
        crouch_lim = torch.full_like(self.startup_elapsed, .90)[:, None]
        deep_crouch = (knee - crouch_lim).clamp_min(0.)
        crouch_cost = (self.stage == self.COMPRESS).float() * 45. * deep_crouch.square().mean(-1)
        opening_weight = torch.where(self.first_takeoff_done, 1., 3.)
        cost = sole_cost + opening_weight * (tuck_cost + crouch_cost)
        if self.symmetric_startup:
            # Directly penalize fore-aft foot splitting during initial crouch/launch
            # and first hop flight, then transition immediately to steady-state cruise.
            foot_xy = probes[..., :2].mean(dim=2)  # (W, 2, 2)
            foot_dx = ((foot_xy[:, 0] - foot_xy[:, 1]) * self.heading).sum(-1).abs()
            opening = (~self.first_takeoff_done).float()
            first_flight = (self.first_takeoff_done & (~self.first_landing_done) & (self.startup_elapsed < self.startup_ramp_s)).float()
            steady = ((opening == 0.) & (first_flight == 0.)).float()
            open_split = (foot_dx - .14).clamp_min(0.)
            flight_split = (foot_dx - .16).clamp_min(0.)
            steady_split = (foot_dx - .15).clamp_min(0.)
            cost = (cost
                    + opening * (12. * open_split.clamp_max(.6) + 40. * open_split.clamp_max(.6).square())
                    + first_flight * 20. * flight_split.clamp_max(.6).square()
                    + steady * (10. * steady_split.clamp_max(.6) + 38. * steady_split.clamp_max(.6).square()))
            # Prevent both legs kicking backward behind the torso during startup or flight.
            hip_q = data.qpos[:, self.hip_qpos]  # (W, 2)
            startup_blend = torch.where(
                self.first_landing_done,
                torch.zeros_like(self.startup_elapsed),
                (1. - self.startup_elapsed / self.startup_ramp_s).clamp(0., 1.),
            )
            hip_allow = torch.where(in_extend, .26, torch.where(startup_blend > .05, .14, .21))[:, None]
            kick_back = (hip_q - hip_allow).clamp_min(0.)
            kick_w = torch.where(in_extend, 25., 28. + 32. * startup_blend)
            cost = cost + kick_w * kick_back.clamp_max(.6).square().mean(-1)
            # Event-driven launch guard: keep FULL weight (1.0) through the entire first hop
            # (crouch + launch + first flight + first touchdown) so the policy cannot cheat by
            # hopping vertically and swinging both feet backward mid-air; release immediately
            # once first_landing_done completes.
            moving_cmd = (self.command_speed / .20).clamp(0., 1.)
            launch_w = (~self.first_landing_done).float()
            foot_mid = probes.mean(dim=(1, 2))  # (W, 3)
            foot_back = -((foot_mid[:, :2] - self.startup_foot_origin) * self.heading).sum(-1)
            # Tight 2 cm tolerance across the entire first hop to eliminate backward foot-swing.
            back_tol = torch.where(~self.first_landing_done, .02, .04)
            back_exc = (foot_back - back_tol).clamp_min(0.).clamp_max(.60)
            com_xyz = data.sensordata[:, self.com_xy.start:self.com_xy.start + 3]
            com_vel = data.sensordata[:, self.com_velocity_xy]
            com_vx = (com_vel * self.heading).sum(-1)
            com_ahead = ((com_xyz[:, :2] - foot_mid[:, :2]) * self.heading).sum(-1)
            body_tilt = torch.atan2(com_ahead, (com_xyz[:, 2] - foot_mid[:, 2]).clamp_min(.25))
            slope = self.terrain_slope()
            # Keep startup (10/13 deg) and cruise (12/15 deg) body tilt upright, and tighten
            # forward tilt allowance on downhills so fast bounding never dives on steep lunar slopes.
            in_fc = (self.stage == self.FLIGHT) | (self.stage == self.COMPRESS)
            start_allow = torch.where(in_fc, .175, .225)
            cruise_allow = torch.where(in_fc, .21, .26)
            downhill = (-slope).clamp_min(0.)
            tilt_allow = (torch.lerp(cruise_allow, start_allow, launch_w) - .55 * downhill).clamp_min(.08)
            tilt_exc = (body_tilt - slope.clamp_min(0.) - tilt_allow).clamp_min(0.).clamp_max(.50)
            tilt_lin_w = 40. + 95. * launch_w
            tilt_quad_w = 120. + 260. * launch_w
            flight_tuck = ((self.stage == self.FLIGHT).float()
                           * (knee - .50).clamp_min(0.).clamp_max(.80).square().mean(-1))
            # Heavily penalize jumping vertically in place (low forward vx) during the 1st flight,
            # which is the exact precursor to swinging both feet backward mid-air.
            first_flight_mask = (self.first_takeoff_done & (~self.first_landing_done) & (self.stage == self.FLIGHT)).float()
            min_launch_vx = (0.50 * (self.command_speed / 1.0).clamp(0., 1.)).clamp_max(.55)
            vertical_hop_shortfall = (min_launch_vx - com_vx).clamp_min(0.)
            vertical_hop_cost = first_flight_mask * (90. * vertical_hop_shortfall + 220. * vertical_hop_shortfall.square())
            cost = (cost
                    + moving_cmd * (launch_w * (420. * back_exc + 1200. * back_exc.square() + 45. * flight_tuck)
                                    + vertical_hop_cost
                                    + tilt_lin_w * tilt_exc + tilt_quad_w * tilt_exc.square()))
        return cost

    def in_place_crouch_launch_reward(self):
        """Reward grounded in-place knee flexion + COM forward shift, and genuine forward 1st takeoff."""
        if not (self.symmetric_startup and self.suit_major_joints):
            return self.worlds.sim.data.qpos.new_zeros(self.worlds.count)
        moving_cmd = (self.command_speed / .20).clamp(0., 1.)
        data = self.worlds.sim.data
        knee_mean = data.qpos[:, self.knee_qpos].mean(-1)
        probes = self.sole_probes()
        foot_mid = probes.mean(dim=(1, 2))
        foot_back = -((foot_mid[:, :2] - self.startup_foot_origin) * self.heading).sum(-1)
        com_xyz = data.sensordata[:, self.com_xy.start:self.com_xy.start + 3]
        com_vel = data.sensordata[:, self.com_velocity_xy]
        com_vx = (com_vel * self.heading).sum(-1)
        com_ahead = ((com_xyz[:, :2] - foot_mid[:, :2]) * self.heading).sum(-1)
        body_tilt = torch.atan2(com_ahead, (com_xyz[:, 2] - foot_mid[:, 2]).clamp_min(.25))
        # 1) Grounded in-place crouch: keep feet planted (no backward shift), bend knees 25..50 deg,
        #    and shift heavy backpack COM forward from -6.7 deg toward +4..+11 deg.
        in_crouch = (~self.first_takeoff_done) & (self.stage == self.COMPRESS) & (self.ground_frames >= 2)
        anchored = torch.exp(-(foot_back.clamp_min(0.) / .02).square())
        crouch_score = ((knee_mean - .30) / .35).clamp(0., 1.) * (1. - ((knee_mean - .90) / .20).clamp(0., 1.))
        com_shift = ((body_tilt + .08) / .22).clamp(0., 1.) * torch.exp(-((body_tilt - .14) / .10).square())
        crouch_bonus = in_crouch.float() * 24.0 * anchored * crouch_score * com_shift
        # 2) Genuine forward 1st flight: reward forward velocity while feet stay ahead of origin.
        in_first_flight = self.first_takeoff_done & (~self.first_landing_done) & (self.stage == self.FLIGHT)
        no_back_swing = torch.exp(-(foot_back.clamp_min(0.) / .03).square())
        fwd_launch = (com_vx / .65).clamp(0., 1.2)
        flight_bonus = in_first_flight.float() * 36.0 * no_back_swing * fwd_launch
        # 3) Dense upward takeoff drive (grounded EXTEND) and airborne clearance reward +
        #    low-skim penalty across all moving hops so the astronaut launches with vz ~0.75 m/s
        #    and clears rough lunar terrain by 18..30 cm instead of skimming at 7 cm.
        com_vz = data.sensordata[:, self.com_velocity_xy.stop]
        move_arc = (self.command_speed / .40).clamp(0., 1.)
        target_vz = .68 + .12 * ((self.command_speed - .50) / 1.0).clamp(0., 1.)
        up_drive = (com_vz / target_vz).clamp(0., 1.15)
        in_extend_push = (self.stage == self.EXTEND) & (self.ground_frames > 0)
        extend_lift_bonus = in_extend_push.float() * 42.0 * up_drive
        in_air = (self.stage == self.FLIGHT) & (self.ground_frames == 0)
        min_clear = self.sole_clearances().amin(dim=(-1, -2)).clamp_min(0.)
        air_clear_bonus = in_air.float() * 45.0 * ((min_clear - .05) / .20).clamp(0., 1.3)
        mid_flight = in_air & (self.age > .08) & (com_vz > -.25)
        low_skim = ((.15 - min_clear) / .15).clamp(0., 1.)
        skim_penalty = mid_flight.float() * 35.0 * low_skim.square()
        return moving_cmd * (crouch_bonus + flight_bonus) + move_arc * (
            extend_lift_bonus + air_clear_bonus - skim_penalty)

    def reward(self, measured):
        error = self.boot_yaw_error()
        boot_error = ((error.abs() - .075).clamp_min(0.) / .55).clamp_max(2.).square().mean(-1)
        center_error = ((error.mean(-1).abs() - .045).clamp_min(0.) / .55).clamp_max(2.).square()
        cost = (2.4 * boot_error + 2.6 * center_error) * (self.stage != self.WAIT)
        if self.suit_major_joints:
            cost = cost + self.waist_motion_cost() + self.natural_leg_cost()
        bonus = self.in_place_crouch_launch_reward()
        return super().reward(measured) + self.DT * (bonus - cost) * ~measured['wangyu/failed']


def replace_height_command_input(state):
    """Initialize contact duration at zero weight; retain other learned inputs."""
    state = dict(state)
    column = HEIGHT_COMMAND_COLUMN
    for key, fill in (('mlp.0.weight', 0.), ('obs_normalizer._mean', 0.),
                      ('obs_normalizer._var', 1.), ('obs_normalizer._std', 1.)):
        state[key] = state[key].clone()
        state[key][..., column] = fill
    return state


def forward_initial_state(source, target):
    """Append three zero-weight speed inputs without altering the inherited actor."""
    expanded = {}
    input_keys = {'obs_normalizer._mean', 'obs_normalizer._var',
                  'obs_normalizer._std', 'mlp.0.weight'}
    for key, value in target.items():
        old = source[key]
        if key in input_keys:
            if value.shape[:-1] != old.shape[:-1] or value.shape[-1] - old.shape[-1] != FORWARD_OBS_COLUMNS:
                raise ValueError(f'Incompatible base observation shape at {key}')
            value = value.clone()
            if key == 'mlp.0.weight':
                value.zero_()
            value[..., :old.shape[-1]].copy_(old)
            expanded[key] = value
        else:
            expanded[key] = old
    return expanded
