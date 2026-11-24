"""RSL-RL environment for the native CPU stand task."""
from concurrent.futures import ThreadPoolExecutor
import os

from rsl_rl.env import VecEnv
from tensordict import TensorDict
import torch

from cpu_forward_jump import DEFAULT_SPEED_RANGE, ForwardFacingBootCommands
from cpu_jump_task import (DEFAULT_JUMP_HEIGHT_RANGE, FLOW_JUMP_ACTUATION,
                           FORWARD_FACING_BOOT_ACTUATION, FlowJumpCommands)
from cpu_simulation import CpuWorlds
from moon_device_core import META
from moon_scene import (APOLLO_MASS_PROFILE, MASS_PROFILES, RELAXED_PROPULSION_PROFILE,
                        SUIT_ANKLE_PITCH_PROFILE, SUIT_ANKLE_ROLL_PROFILE, SUIT_MAJOR_JOINTS_PROFILE,
                        WANGYU_EVA_PROFILE, scene_spec)
from suit_proxy_geometry import LEGACY


class CpuStandEnv(VecEnv):
    device = 'cpu'
    task = 'stand'
    num_actions = 23
    sole_probes = False
    sole_collision = False
    sole_edge_probes = False
    max_episode_length = round(10 / META['control_dt'])

    def __init__(self, num_envs, terrain_idx, pushes, physics_backend, workers, seed, *,
                 geometry=LEGACY, inward_heading=False, terrain_indices=None, terrain_size=16,
                 assist_force=None, hinge_torque=None,
                 exo_policy_io=None, exo_mechanism=None, vertical_support=False, stance_transition=False,
                 suit_ankle_pitch=False, suit_major_joints=False, relax_propulsion=False,
                 apollo_mass=False, mass_profile=None):
        self.num_envs = num_envs
        self.workers = min(workers, num_envs)
        self.device = 'cuda' if physics_backend == 'warp' else 'cpu'
        self.cfg = {
            'backend': 'mujoco_warp_cuda' if physics_backend == 'warp' else 'native_mujoco_cpu',
            'physics_backend': physics_backend, 'task': self.task,
            'num_envs': num_envs, 'terrain_size_m': terrain_size,
            'terrain_idx': terrain_idx, 'pushes': pushes, 'seed': seed,
            'physics_workers': self.workers,
            'geometry': geometry,
        }
        if terrain_indices is not None:
            self.cfg['terrain_indices'] = list(terrain_indices)
            self.cfg['spawn_sampling'] = 'independent_position_and_36_uniform_headings'
        pd_force = None
        suit_ankle_roll = False
        if assist_force is not None:
            from exoskeleton import (ACTION_FORCE_COSTS, DEFAULT_HINGE_TORQUE_NM, DIRECT_FORCE_REVISIONS,
                                     HINGE_MECHANISMS, MECHANISM, POLICY_IO, validate_hinge_torque)
            mechanism = exo_mechanism or MECHANISM
            self.cfg.update(assist_force_n=assist_force, exo_mechanism=mechanism)
            if mechanism in HINGE_MECHANISMS:
                resolved_hinge = validate_hinge_torque(
                    DEFAULT_HINGE_TORQUE_NM if hinge_torque is None else hinge_torque)
                self.cfg['hinge_torque_nm'] = resolved_hinge
                hinge_torque = resolved_hinge
            suit_ankle_roll = exo_policy_io in DIRECT_FORCE_REVISIONS
            if suit_ankle_roll:
                self.cfg.update(ankle_roll_profile=SUIT_ANKLE_ROLL_PROFILE,
                                exo_action_force_costs=ACTION_FORCE_COSTS.copy())
            if exo_policy_io == POLICY_IO:
                pd_force = assist_force
        if suit_ankle_pitch:
            self.cfg['ankle_pitch_profile'] = SUIT_ANKLE_PITCH_PROFILE
        if suit_major_joints:
            self.cfg['suit_major_joints_profile'] = SUIT_MAJOR_JOINTS_PROFILE
        if relax_propulsion:
            self.cfg['propulsion_profile'] = RELAXED_PROPULSION_PROFILE
        resolved_mass = mass_profile or (APOLLO_MASS_PROFILE if apollo_mass else None)
        if resolved_mass is not None:
            if resolved_mass not in MASS_PROFILES:
                raise ValueError(f'Unknown suit mass profile: {resolved_mass}')
            self.cfg['apollo_mass_profile'] = resolved_mass
        spec = scene_spec(terrain_size=terrain_size, geometry=geometry, sole_probes=self.sole_probes,
                          exoskeleton=assist_force is not None, exo_pd_force=pd_force,
                          exo_mechanism=exo_mechanism,
                          sole_collision=self.sole_collision, sole_edge_probes=self.sole_edge_probes,
                          suit_ankle_roll=suit_ankle_roll, suit_ankle_pitch=suit_ankle_pitch,
                          suit_major_joints=suit_major_joints, relax_propulsion=relax_propulsion,
                          mass_profile=resolved_mass,
                          vertical_support=vertical_support, stance_transition=stance_transition,
                          tiled_100m=(terrain_size == 100 and physics_backend == 'warp'))
        if physics_backend == 'warp':
            from warp_exo_worlds import WarpExoWorlds
            self.worlds = WarpExoWorlds(spec, num_envs, terrain_idx, pushes=pushes,
                                        terrain_indices=terrain_indices, seed=seed,
                                        assist_force=assist_force, hinge_torque=hinge_torque,
                                        terrain_size=terrain_size)
            self.cfg['spawn_sampling'] = 'independent_position_and_36_uniform_headings_gpu_with_replacement'
            if terrain_size == 100:
                self.cfg['warp_100m_tiles'] = f'{len(self.worlds.device.ground.zmin)}x20m'
            self.cfg['warp_cuda_graphs'] = self.worlds.use_cuda_graphs
            self.cfg['measurement_cuda_graph'] = self.worlds.device.use_measurement_graph
            self.cfg['batched_terrain_queries'] = self.worlds.device.batch_measurement_terrain
        else:
            self.worlds = CpuWorlds(spec, num_envs, terrain_idx, terrain_size=terrain_size, pushes=pushes,
                                    use_mjbatch=physics_backend == 'mjbatch', workers=self.workers,
                                    optimize_cpu=True, inward_heading=inward_heading,
                                    terrain_indices=terrain_indices, spawn_seed=seed,
                                    assist_force=assist_force, hinge_torque=hinge_torque)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.weights = torch.tensor(list(META['rewards'].values()), device=self.device)
        self.pool = None

    def __enter__(self):
        if self.cfg['physics_backend'] == 'threaded' and self.workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def get_observations(self):
        measured = self.worlds.device.get_measurements(self.worlds.sim)
        return self._observations(measured)

    def _observations(self, measured):
        state = self.policy_state(measured['wangyu/state'])
        return TensorDict({'policy': state, 'critic': state}, batch_size=[self.num_envs])

    def policy_state(self, state):
        return state.clone()

    def reward(self, measured):
        return (measured['wangyu/rewards'] @ self.weights) * META['control_dt']

    def commands(self, actions):
        return actions

    def advance_task(self, measured):
        pass

    def reset_task(self, mask):
        pass

    def episode_timeout(self):
        return self.episode_length_buf >= self.max_episode_length

    def step(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError('Non-finite PPO actions')
        measured = self.worlds.step(self.commands(actions), pool=self.pool, workers=self.workers)
        self.advance_task(measured)
        rewards = self.reward(measured)
        self.episode_length_buf += 1
        failed = measured['wangyu/failed']
        timeouts = self.episode_timeout() & ~failed
        done = failed | timeouts
        if done.any():
            self.worlds.reset(done)
            self.reset_task(done)
            self.episode_length_buf.masked_fill_(done, 0)
            if self.worlds.device.use_measurement_graph:
                reset_obs = self.worlds.device.get_measurements(self.worlds.sim)
                measured['wangyu/state'] = torch.where(
                    done[:, None], reset_obs['wangyu/state'], measured['wangyu/state'])
            else:
                ids = done.nonzero(as_tuple=True)[0]
                reset_obs = self.worlds.device.get_measurements(self.worlds.sim, env_ids=ids)
                measured['wangyu/state'][ids] = reset_obs['wangyu/state']
        return self._observations(measured), rewards, done, {'time_outs': timeouts}


class CpuJumpEnv(CpuStandEnv):
    task = 'jump'
    num_actions = FlowJumpCommands.NUM_ACTIONS
    controller = FlowJumpCommands
    actuation = FLOW_JUMP_ACTUATION
    objective = 'event_height_efficiency_and_continuous_rebound'
    has_height_command = True

    def __init__(self, *args, height_range=DEFAULT_JUMP_HEIGHT_RANGE, controller_options=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.jump_commands = self.controller(self.worlds, height_range=height_range,
                                             **(controller_options or {}))
        self.cfg['jump_actuation'] = self.actuation
        if self.has_height_command:
            self.cfg['jump_height_range_m'] = self.jump_commands.height_range
        self.cfg['jump_objective'] = self.objective

    def commands(self, actions):
        return self.jump_commands(actions)

    def policy_state(self, state):
        return self.jump_commands.observation(state)

    def reward(self, measured):
        return self.jump_commands.reward(measured)

    def advance_task(self, measured):
        self.jump_commands.advance(measured)

    def reset_task(self, mask):
        self.jump_commands.reset(mask)


class CpuForwardJumpEnv(CpuJumpEnv):
    task = 'forward_jump'
    controller = ForwardFacingBootCommands
    actuation = FORWARD_FACING_BOOT_ACTUATION
    objective = 'forward_with_phase_mirroring_and_boot_yaw_alignment'
    sole_probes = True
    sole_collision = True
    sole_edge_probes = True
    has_height_command = False

    def __init__(self, *args, speed_range=DEFAULT_SPEED_RANGE, exo_policy_io=None, long_hop_period=None,
                 suit_ankle_pitch=False, slope_adaptive_posture=False,
                 suit_major_joints=False, relax_propulsion=False, **kwargs):
        from exoskeleton import DIRECT_FORCE_REVISIONS, FORCE_CONTROL_REVISIONS

        controller_options = {'speed_range': speed_range,
                              'compact_arm_swing': exo_policy_io in FORCE_CONTROL_REVISIONS,
                              'suit_ankle_roll': exo_policy_io in DIRECT_FORCE_REVISIONS,
                              'suit_ankle_pitch': suit_ankle_pitch,
                              'slope_adaptive_posture': slope_adaptive_posture,
                              'suit_major_joints': suit_major_joints,
                              'relax_propulsion': relax_propulsion,
                              'long_hop_period': long_hop_period}
        super().__init__(*args, inward_heading=True, controller_options=controller_options,
                         exo_policy_io=exo_policy_io, suit_ankle_pitch=suit_ankle_pitch,
                         suit_major_joints=suit_major_joints, relax_propulsion=relax_propulsion, **kwargs)
        self.cfg['jump_speed_range_mps'] = self.jump_commands.speed_range
        if slope_adaptive_posture:
            self.cfg['slope_adaptive_posture'] = True
        if self.worlds.exoskeleton is not None:
            self.worlds.exoskeleton.bind(self.jump_commands, policy_io=exo_policy_io)
        if exo_policy_io is not None:
            from exoskeleton import validate_policy_io
            if self.worlds.exoskeleton is None:
                raise ValueError('Learned strut actions require an exoskeleton')
            self.num_actions += len(self.worlds.exoskeleton.actuators)
            self.cfg['exo_policy_io'] = validate_policy_io(exo_policy_io)

    def commands(self, actions):
        assist = self.worlds.exoskeleton
        return assist.policy_actions(actions) if assist is not None else super().commands(actions)

    def policy_state(self, state):
        state = super().policy_state(state)
        assist = self.worlds.exoskeleton
        return assist.observation(state) if assist is not None else state

    def episode_timeout(self):
        return super().episode_timeout() | self.jump_commands.course_finished()



class CpuExoJumpEnv(CpuForwardJumpEnv):
    task = 'exo_jump'
    objective = 'forward_jump_with_supported_exoskeleton_work_sharing'

    def __init__(self, *args, assist_force, stance_center_weight=0., exo_policy_io=None, load_sharing=False,
                 vertical_support=False, posture_balance=False, stance_transition=False, economical_hops=False,
                 long_hop_period=None, roll_balance=False, reward_overrides=None, speed_track_boost=0.,
                 assist_drop_prob=0., **kwargs):
        from exoskeleton import (ACTION_FORCE_EFFICIENCY_REWARD, DIRECT_FORCE_REVISIONS,
                                 EFFICIENCY_REWARD, PHASE_FREE_FORCE_REVISIONS, POLICY_IO)

        posture_balance = posture_balance or stance_transition
        economical_hops = economical_hops or long_hop_period is not None
        vertical_support = vertical_support or posture_balance or economical_hops or roll_balance
        super().__init__(*args, assist_force=assist_force, exo_policy_io=exo_policy_io or POLICY_IO,
                         vertical_support=vertical_support, stance_transition=stance_transition,
                         long_hop_period=long_hop_period, **kwargs)
        self.cfg['exo_efficiency_reward'] = EFFICIENCY_REWARD.copy()
        if self.cfg['exo_policy_io'] in DIRECT_FORCE_REVISIONS:
            self.cfg['exo_efficiency_reward'] = ACTION_FORCE_EFFICIENCY_REWARD.copy()
        elif self.cfg['exo_policy_io'] in PHASE_FREE_FORCE_REVISIONS:
            self.cfg['exo_efficiency_reward']['revision'] = 'phase_free_supported_work_v1'
        if load_sharing or vertical_support:
            self.worlds.exoskeleton.enable_load_sharing(
                vertical_support=vertical_support, posture_balance=posture_balance,
                stance_transition=stance_transition, roll_balance=roll_balance,
                reward_overrides=reward_overrides)
            self.cfg['exo_efficiency_reward'] = self.worlds.exoskeleton.reward_cfg.copy()
            self.cfg['jump_objective'] = 'human_energy_and_bilateral_grounded_assistance'
        if assist_drop_prob:
            self.worlds.exoskeleton.enable_assist_drop(assist_drop_prob)
            self.cfg['assist_drop_prob'] = float(assist_drop_prob)
        if economical_hops:
            self.jump_commands.economical_hops = True
            self.cfg['economical_hops'] = True
        if roll_balance:
            self.cfg['roll_balance'] = True
        if long_hop_period is not None:
            self.cfg['long_hop_period_s'] = long_hop_period
        self.stance_center_weight = stance_center_weight
        if stance_center_weight:
            self.cfg['stance_center_weight'] = stance_center_weight
        self.speed_track_boost = speed_track_boost
        if speed_track_boost:
            self.cfg['speed_track_boost'] = speed_track_boost
        if self.device == 'cuda' and os.environ.get('MOON_WARP_COMPILE', '1') == '1':
            self.commands = torch.compile(self.commands, mode='default')
            self.advance_task = torch.compile(self.advance_task, mode='default')
            self.reward = torch.compile(self.reward, mode='default')
            self.policy_state = torch.compile(self.policy_state, mode='default')
            self.worlds.exoskeleton._observe_tensors = torch.compile(
                self.worlds.exoskeleton._observe_tensors, mode='default')
            self.worlds.exoskeleton.controls = torch.compile(
                self.worlds.exoskeleton.controls, mode='default')
            self.worlds.device.set_substep_commands = torch.compile(
                self.worlds.device.set_substep_commands, mode='default')
            self.cfg['warp_torch_compile'] = True

    def stance_center_cost(self):
        root = self.worlds.sim.data.qpos
        w, x, y, z = root[:, 3:7].unbind(-1)
        left = torch.stack((-2 * (w * z + x * y), 1 - 2 * (y.square() + z.square())), dim=-1)
        left = torch.nn.functional.normalize(left, dim=-1)
        midpoint = self.jump_commands.sole_probes()[..., :2].mean(dim=(1, 2))
        offset = ((midpoint - root[:, :2]) * left).sum(-1).abs()
        # Allow small terrain and balance adjustments around the pelvis midline.
        return ((offset - .03).clamp_min(0.) / .12).clamp_max(2.).square()

    def reward(self, measured):
        rate = self.worlds.exoskeleton.efficiency_rate
        if self.stance_center_weight:
            rate = rate - self.stance_center_weight * self.stance_center_cost()
        if self.speed_track_boost:
            cmd = self.jump_commands
            com_vel = self.worlds.sim.data.sensordata[:, cmd.com_velocity_xy]
            forward = (com_vel * cmd.heading).sum(-1)
            target = cmd.target_speed
            # Calibrated cruise speed tracking across 0.5..1.5 m/s: remove low/mid-speed
            # feedforward bias so 0.5 and 1.0 m/s track accurately without overshooting,
            # while giving a modest +0.05 m/s high-speed dip offset near 1.5 m/s.
            moving = (target / .25).clamp(0., 1.)
            high = 1. + .8 * (target - .60).clamp_min(0.)
            effective_target = target + .05 * ((target - 1.0) / .50).clamp(0., 1.)
            error = forward - effective_target
            improve = (effective_target.square() + .006).sqrt() - (error.square() + .006).sqrt()
            shortfall = (effective_target - forward).clamp_min(0.)
            overshoot = (forward - (effective_target + .05)).clamp_min(0.)
            underspeed_pen = shortfall + .65 * shortfall.square()
            overspeed_pen = .80 * overshoot + 2.4 * overshoot.square()
            still = (1. - target / .15).clamp(0., 1.)
            still_pen = 20. * still * forward.square()
            rate = (rate
                    + self.speed_track_boost * moving * high * (improve - .75 * underspeed_pen - overspeed_pen)
                    - still_pen)
        return super().reward(measured) + rate * self.jump_commands.DT * ~measured['wangyu/failed']


class CpuExoHingeJumpEnv(CpuExoJumpEnv):
    task = 'exo_hinge_jump'
    objective = 'forward_jump_with_sagittal_hinge_exoskeleton_assistance'

    def __init__(self, *args, assist_force=None, hinge_torque=None, exo_mechanism=None,
                 exo_policy_io=None, **kwargs):
        from exoskeleton import (DEFAULT_FORCE_N, DEFAULT_HINGE_TORQUE_NM,
                                 HUT_HINGE_AFT_MECHANISM, IMPEDANCE_HINGE_FORCE_IO)
        super().__init__(
            *args,
            assist_force=DEFAULT_FORCE_N if assist_force is None else assist_force,
            hinge_torque=DEFAULT_HINGE_TORQUE_NM if hinge_torque is None else hinge_torque,
            exo_mechanism=exo_mechanism or HUT_HINGE_AFT_MECHANISM,
            exo_policy_io=exo_policy_io or IMPEDANCE_HINGE_FORCE_IO,
            **kwargs,
        )
