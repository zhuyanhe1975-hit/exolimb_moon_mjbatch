"""RSL-RL environment for the native CPU stand task."""
from concurrent.futures import ThreadPoolExecutor

from rsl_rl.env import VecEnv
from tensordict import TensorDict
import torch

from cpu_forward_jump import DEFAULT_SPEED_RANGE, ForwardFacingBootCommands
from cpu_jump_task import (DEFAULT_JUMP_HEIGHT_RANGE, FLOW_JUMP_ACTUATION,
                           FORWARD_FACING_BOOT_ACTUATION, FlowJumpCommands)
from cpu_simulation import CpuWorlds
from moon_device_core import META
from moon_scene import scene_spec
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
                 geometry=LEGACY, inward_heading=False, terrain_indices=None, assist_force=None,
                 exo_policy_io=None):
        self.num_envs = num_envs
        self.workers = min(workers, num_envs)
        self.cfg = {
            'backend': 'native_mujoco_cpu', 'physics_backend': physics_backend, 'task': self.task,
            'num_envs': num_envs, 'terrain_idx': terrain_idx, 'pushes': pushes, 'seed': seed,
            'physics_workers': self.workers,
            'geometry': geometry,
        }
        if terrain_indices is not None:
            self.cfg['terrain_indices'] = list(terrain_indices)
            self.cfg['spawn_sampling'] = 'independent_position_and_36_uniform_headings'
        pd_force = None
        if assist_force is not None:
            from exoskeleton import MECHANISM, POLICY_IO
            self.cfg.update(assist_force_n=assist_force, exo_mechanism=MECHANISM)
            if exo_policy_io == POLICY_IO:
                pd_force = assist_force
        spec = scene_spec(geometry=geometry, sole_probes=self.sole_probes,
                          exoskeleton=assist_force is not None, exo_pd_force=pd_force,
                          sole_collision=self.sole_collision, sole_edge_probes=self.sole_edge_probes)
        self.worlds = CpuWorlds(spec, num_envs, terrain_idx, pushes=pushes,
                               use_mjbatch=physics_backend == 'mjbatch', workers=self.workers,
                               optimize_cpu=True, inward_heading=inward_heading,
                               terrain_indices=terrain_indices, spawn_seed=seed,
                               assist_force=assist_force)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.weights = torch.tensor(list(META['rewards'].values()))
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
            self.episode_length_buf[done] = 0
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

    def __init__(self, *args, speed_range=DEFAULT_SPEED_RANGE, exo_policy_io=None, **kwargs):
        super().__init__(*args, inward_heading=True, controller_options={'speed_range': speed_range},
                         exo_policy_io=exo_policy_io, **kwargs)
        self.cfg['jump_speed_range_mps'] = self.jump_commands.speed_range
        if self.worlds.exoskeleton is not None:
            self.worlds.exoskeleton.bind(self.jump_commands, policy_io=exo_policy_io)
        if exo_policy_io is not None:
            from exoskeleton import STRUT_ACTIONS, validate_policy_io
            if self.worlds.exoskeleton is None:
                raise ValueError('Learned strut actions require an exoskeleton')
            self.num_actions += STRUT_ACTIONS
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

    def __init__(self, *args, assist_force, **kwargs):
        from exoskeleton import EFFICIENCY_REWARD, POLICY_IO

        super().__init__(*args, assist_force=assist_force, exo_policy_io=POLICY_IO, **kwargs)
        self.cfg['exo_efficiency_reward'] = EFFICIENCY_REWARD.copy()

    def reward(self, measured):
        efficiency = self.worlds.exoskeleton.efficiency_rate * self.jump_commands.DT
        return super().reward(measured) + efficiency * ~measured['wangyu/failed']
