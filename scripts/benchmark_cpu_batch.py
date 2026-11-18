"""Check CPU backend parity and time the stand task's rollout sampling."""
import argparse
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import mujoco
import numpy as np
import torch
from rsl_rl.models import MLPModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cpu_stand_env import CpuStandEnv


def make_env(count, workers, backend, seed):
    torch.manual_seed(seed)
    return CpuStandEnv(count, 12, True, backend, workers, seed)


def assert_same(left, right):
    if not torch.equal(left, right):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def integration_states(env):
    worlds = env.worlds
    if worlds.data is None:
        return worlds._batch.state.copy()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    states = np.empty((env.num_envs, mujoco.mj_stateSize(worlds.model, state_spec)))
    for data, state in zip(worlds.data, states):
        mujoco.mj_getState(worlds.model, data, state, state_spec)
    return states


def check_parity(count, workers):
    seed = 42
    with (make_env(count, workers, 'threaded', seed) as old,
          make_env(count, workers, 'mjbatch', seed) as new):
        assert_same(old.get_observations()['policy'], new.get_observations()['policy'])
        # Exercise a failure and a distinct timeout on the first control step.
        for env in (old, new):
            env.episode_length_buf[:2] = env.max_episode_length - 1
            env.worlds.device.target[1, 0] += 2
            env.worlds.device.wait[2] = 0
        actions_rng = torch.Generator().manual_seed(seed)
        rng = torch.get_rng_state()
        for step in range(240):
            actions = torch.randn((count, old.num_actions), generator=actions_rng) * .25
            torch.set_rng_state(rng)
            old_obs, old_reward, old_done, old_info = old.step(actions)
            next_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            new_obs, new_reward, new_done, new_info = new.step(actions)
            assert_same(next_rng, torch.get_rng_state())
            rng = next_rng
            for key in ('policy', 'critic'):
                assert_same(old_obs[key], new_obs[key])
            assert_same(old_reward, new_reward)
            assert_same(old_done, new_done)
            assert_same(old_info['time_outs'], new_info['time_outs'])
            assert_same(old.episode_length_buf, new.episode_length_buf)
            for name in ('qpos', 'qvel', 'time', 'sensordata', 'actuator_force'):
                assert_same(getattr(old.worlds.sim.data, name), getattr(new.worlds.sim.data, name))
            if step == 0:
                assert old_done[:2].tolist() == [True, True]
                assert old_info['time_outs'][:2].tolist() == [True, False]
                assert old.worlds.sim.data.time[:2].tolist() == [0, 0]
                assert old.worlds.device.remaining[2] > 0

        mask = torch.arange(count) % 3 == 1
        for env in (old, new):
            before = integration_states(env)
            torch.set_rng_state(rng)
            env.worlds.reset(mask)
            after = integration_states(env)
            np.testing.assert_array_equal(before[~mask.numpy()], after[~mask.numpy()])
            assert torch.all(env.worlds.sim.data.time[mask] == 0)
        np.testing.assert_array_equal(integration_states(old), integration_states(new))
        assert_same(old.get_observations()['policy'], new.get_observations()['policy'])
    print(f'{count:2} envs: exact parity for 240 steps, including pushes, '
          'failure, timeout and masked reset', flush=True)


def sample(count, workers, backend, seed, warmup, steps):
    with make_env(count, workers, backend, seed) as env:
        obs = env.get_observations()
        actor = MLPModel(obs, {'actor': ['policy']}, 'actor', env.num_actions,
                         hidden_dims=(128, 128), activation='elu', obs_normalization=True,
                         distribution_cfg={'class_name': 'rsl_rl.modules.GaussianDistribution',
                                           'init_std': .25, 'std_type': 'scalar'}).eval()
        env.episode_length_buf.random_(0, env.max_episode_length)
        with torch.inference_mode():
            for _ in range(warmup):
                obs, _, _, _ = env.step(actor(obs, stochastic_output=True))
            start = perf_counter()
            for _ in range(steps):
                obs, _, _, _ = env.step(actor(obs, stochastic_output=True))
            return count * steps / (perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physics-workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.physics_workers, args.steps, args.repeats) < 1:
        parser.error('counts must be positive')
    torch.set_num_threads(1)
    print(f'MuJoCo {mujoco.__version__} | {args.physics_workers} physics threads | '
          f'1 Torch thread | {args.repeats} repeats of {args.steps} timed steps', flush=True)
    for count in (16, 64):
        check_parity(count, args.physics_workers)
        rates = {'threaded': [], 'mjbatch': []}
        for repeat in range(args.repeats):
            backends = list(rates) if repeat % 2 == 0 else list(reversed(rates))
            for backend in backends:
                rate = sample(count, args.physics_workers, backend, 42 + repeat, 100, args.steps)
                rates[backend].append(rate)
                print(f'{count:2} envs | repeat {repeat + 1} | {backend:8} | '
                      f'{rate:,.0f} transitions/s', flush=True)
        old, new = (median(rates[backend]) for backend in ('threaded', 'mjbatch'))
        print(f'{count:2} envs median | threaded {old:,.0f} | mjbatch {new:,.0f} '
              f'transitions/s | {new / old:.2f}x', flush=True)


if __name__ == '__main__':
    main()
