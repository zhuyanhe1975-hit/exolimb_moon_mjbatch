"""Local batched PPO training with native MuJoCo CPU physics."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def default_threads(num_envs):
    """Scale the native pool from the measured small and large batch settings."""
    available = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                 else (os.cpu_count() or 1))
    physics_threads = min(available, num_envs, 16, max(4, (3 * num_envs + 15) // 16))
    torch_threads = 1 if num_envs < 32 else 2 if num_envs < 256 else 4
    return physics_threads, min(available, torch_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--task', choices=('stand', 'jump', 'forward_jump', 'exo_jump'), default='stand')
    parser.add_argument('--jump-height-range', nargs=2, type=float, metavar=('MIN_M', 'MAX_M'),
                        help='uniform jump goal range in meters (jump task only)')
    parser.add_argument('--jump-speed-range', nargs=2, type=float, metavar=('MIN_MPS', 'MAX_MPS'),
                        help='forward jump speed range in m/s (default: 0 1.5)')
    parser.add_argument('--assist-force', type=float, metavar='NEWTONS',
                        help='per-strut peak force in the exoskeleton task (default: 500 N)')
    parser.add_argument('--geometry', choices=('legacy', 'fitted'), default='fitted',
                        help='suit collision geometry (default: fitted)')
    parser.add_argument('--steps-per-env', type=int, default=32)
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument('--terrain-idx', type=int, choices=range(1, 65), default=12, metavar='1–64')
    terrain.add_argument('--terrain-idxs', nargs='+', type=int, choices=range(1, 65), metavar='1–64',
                         help='split forward-jump worlds across these terrains with independent headings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--physics-backend', choices=('mjbatch', 'threaded'), default='mjbatch',
                        help='CPU physics backend (default: mjbatch)')
    parser.add_argument('--physics-workers', type=int, help='physics threads (default: automatic)')
    parser.add_argument('--torch-threads', type=int, help='PyTorch threads (default: automatic)')
    parser.add_argument('--update-device', choices=('auto', 'cpu', 'mps'), default='auto',
                        help='PPO update device (default: auto); rollouts stay on CPU')
    parser.add_argument('--pushes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--init-checkpoint', type=Path,
                        help='initialize the jump actor/critic from a compatible checkpoint')
    args = parser.parse_args()
    forward_tasks = ('forward_jump', 'exo_jump')
    if args.assist_force is not None and args.task != 'exo_jump':
        parser.error('--assist-force 仅用于外骨骼跳跃任务')
    if args.terrain_idxs is not None:
        if args.task not in forward_tasks:
            parser.error('--terrain-idxs 仅用于前进跳跃任务')
        if len(set(args.terrain_idxs)) != len(args.terrain_idxs) or len(args.terrain_idxs) > args.num_envs:
            parser.error('地形编号必须唯一，且每块地形至少有一个训练环境')
        args.terrain_idx = args.terrain_idxs[0]
    if args.jump_height_range is not None and args.task != 'jump':
        parser.error('--jump-height-range 仅用于原地跳跃任务')
    if args.jump_speed_range is not None and args.task not in forward_tasks:
        parser.error('--jump-speed-range 只能用于前进跳跃任务')
    if args.init_checkpoint is not None and args.task == 'stand':
        parser.error('--init-checkpoint 只能用于跳跃任务')
    if args.task == 'forward_jump' and args.init_checkpoint is None:
        args.init_checkpoint = ROOT / 'bests/jump_flow_rebound_2026-11-18/model_149.pt'
    if args.task == 'exo_jump' and args.init_checkpoint is None:
        args.init_checkpoint = ROOT / 'bests/forward_jump_boot_yaw_2026-11-18/model_104.pt'
    physics_threads, torch_threads = default_threads(args.num_envs)
    if args.physics_workers is None:
        args.physics_workers = physics_threads
    if args.torch_threads is None:
        args.torch_threads = torch_threads
    if min(args.num_envs, args.iterations, args.steps_per_env, args.torch_threads, args.physics_workers) < 1:
        parser.error('environment, iteration, rollout and thread counts must be positive')
    if args.num_envs * args.steps_per_env < 4:
        parser.error('each rollout must contain at least 4 transitions')

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from cpu_forward_jump import (DEFAULT_SPEED_RANGE, forward_initial_state,
                                  replace_height_command_input, validate_speed_range)
    from cpu_jump_task import (DEFAULT_JUMP_HEIGHT_RANGE, FORWARD_JUMP_ACTUATIONS,
                               FORWARD_FACING_BOOT_ACTUATION, FREE_HEIGHT_FORWARD_ACTUATIONS, HEIGHT_JUMP_ACTUATIONS,
                               validate_jump_height_range)
    from cpu_stand_env import CpuExoJumpEnv, CpuForwardJumpEnv, CpuJumpEnv, CpuStandEnv
    from provenance import save as save_terrain

    mps_available = torch.backends.mps.is_available()
    if args.update_device == 'auto':
        # Only default to MPS at the rollout size verified on the Mac.
        transitions = args.num_envs * args.steps_per_env
        args.update_device = 'mps' if mps_available and transitions >= 1024 * 32 else 'cpu'
    if args.update_device == 'mps' and not mps_available:
        parser.error('MPS is not available in this PyTorch environment')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    init_state = None
    init_info = {}
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            parser.error(f'检查点不存在：{args.init_checkpoint}')
        init_state = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        init_info = init_state.get('infos') or {}
        init_actuation = init_info.get('jump_actuation')
        if init_actuation not in HEIGHT_JUMP_ACTUATIONS and init_actuation not in FORWARD_JUMP_ACTUATIONS:
            parser.error('初始化需要兼容的事件驱动跳跃检查点')
        if args.task == 'jump' and init_actuation in FORWARD_JUMP_ACTUATIONS:
            parser.error('原地跳跃任务需要原地跳跃检查点初始化')
        if args.task == 'exo_jump' and init_actuation != FORWARD_FACING_BOOT_ACTUATION:
            parser.error('外骨骼实验需要 v32 前进跳跃策略或由它续训的外骨骼策略')
        if init_info.get('geometry') != args.geometry:
            parser.error('初始化检查点的几何版本须与训练配置一致')
    env_class = {'stand': CpuStandEnv, 'jump': CpuJumpEnv,
                 'forward_jump': CpuForwardJumpEnv, 'exo_jump': CpuExoJumpEnv}[args.task]
    task_options = {}
    migrate_exo_policy = False
    try:
        if args.task == 'jump':
            task_options['height_range'] = validate_jump_height_range(
                args.jump_height_range or DEFAULT_JUMP_HEIGHT_RANGE)
        elif args.task in forward_tasks:
            task_options['speed_range'] = validate_speed_range(args.jump_speed_range or DEFAULT_SPEED_RANGE)
        if args.task == 'exo_jump':
            from exoskeleton import DEFAULT_FORCE_N, POLICY_IO, validate_assist_force, validate_policy_io
            requested = DEFAULT_FORCE_N if args.assist_force is None else args.assist_force
            task_options['assist_force'] = validate_assist_force(requested)
            migrate_exo_policy = validate_policy_io(init_info.get('exo_policy_io')) != POLICY_IO
    except ValueError as exc:
        parser.error(str(exc))
    env = env_class(args.num_envs, args.terrain_idx, args.pushes,
                    args.physics_backend, args.physics_workers, args.seed,
                    geometry=args.geometry, terrain_indices=args.terrain_idxs, **task_options)
    env.cfg['update_device'] = args.update_device
    if args.init_checkpoint is not None:
        env.cfg['init_checkpoint'] = str(args.init_checkpoint.resolve())
    model = dict(class_name='rsl_rl.models.MLPModel', hidden_dims=(128, 128),
                 activation='elu', obs_normalization=True)
    algorithm = ('cpu_sampling_ppo.CpuSamplingMpsPPO' if args.update_device == 'mps'
                 else 'rsl_rl.algorithms.PPO')
    cfg = dict(
        num_steps_per_env=args.steps_per_env, save_interval=25, logger='tensorboard',
        obs_groups={'actor': ['policy'], 'critic': ['critic']},
        actor=dict(model, distribution_cfg=dict(class_name='rsl_rl.modules.GaussianDistribution',
                                                init_std=.25, std_type='scalar')),
        critic=dict(model),
        algorithm=dict(class_name=algorithm, num_learning_epochs=5,
                       num_mini_batches=4, learning_rate=3e-4, schedule='adaptive',
                       gamma=.995, lam=.95, entropy_coef=.001, desired_kl=.01,
                       clip_param=.2, max_grad_norm=1., value_loss_coef=1.,
                       use_clipped_value_loss=True, normalize_advantage_per_mini_batch=False),
    )
    log_root = 'cpu_exo_train' if args.task == 'exo_jump' else 'cpu_train'
    log_dir = (args.log_dir or ROOT / 'logs' / log_root /
               datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')).resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    save_terrain(log_dir)
    (log_dir / 'config.json').write_text(json.dumps(
        {'env': env.cfg, 'runner': cfg, 'iterations': args.iterations,
         'torch_threads': args.torch_threads}, indent=2) + '\n')
    terrain_description = ','.join(map(str, args.terrain_idxs or [args.terrain_idx]))
    print(f'[CPU train] task={args.task}, geometry={args.geometry}, '
          f'{args.num_envs} worlds on terrain {terrain_description}, '
          f'{args.physics_backend} with {env.workers} physics threads and '
          f'{args.torch_threads} PyTorch threads, PPO update={args.update_device} → {log_dir}', flush=True)
    if 'jump_height_range_m' in env.cfg:
        low, high = env.cfg['jump_height_range_m']
        print(f'[CPU train] Jump height range: {low:g}–{high:g} m', flush=True)
    if 'jump_speed_range_mps' in env.cfg:
        low, high = env.cfg['jump_speed_range_mps']
        print(f'[CPU train] Forward jump speed range: {low:g}–{high:g} m/s; free jump height', flush=True)

    if args.task == 'exo_jump':
        print(f'[CPU train] Exoskeleton position PD: up to {env.cfg["assist_force_n"]:g} N per strut', flush=True)

    class LocalRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            keys = ('task', 'jump_actuation', 'jump_height_range_m', 'jump_speed_range_mps',
                    'geometry', 'assist_force_n', 'exo_mechanism', 'exo_efficiency_reward', 'exo_policy_io')
            task_info = {key: env.cfg[key] for key in keys if key in env.cfg}
            super().save(path, {**(infos or {}), **task_info})

    runner = LocalRunner(env, cfg, log_dir=str(log_dir), device='cpu')
    if init_state is not None:
        if args.task in forward_tasks:
            if init_actuation not in FREE_HEIGHT_FORWARD_ACTUATIONS:
                for key, model in (('actor_state_dict', runner.alg.get_policy()),
                                   ('critic_state_dict', runner.alg.critic)):
                    state = init_state[key]
                    if init_actuation not in FORWARD_JUMP_ACTUATIONS:
                        state = forward_initial_state(state, model.state_dict())
                    init_state[key] = replace_height_command_input(state)
            if init_actuation != env.actuation:
                # Reopen exploration on the new objective without changing actor means.
                actor_state = init_state['actor_state_dict']
                std_key = 'distribution.std_param'
                actor_state[std_key] = actor_state[std_key].clamp_min(.25)
        if migrate_exo_policy:
            from exoskeleton import INITIAL_ASSIST_N, MOTOR_POLICY_IO, PD_KP, PD_MAX_ADVANCE_M, exo_initial_state
            force_cap = env.cfg['assist_force_n']
            initial_force = min(init_info.get('assist_force_n', INITIAL_ASSIST_N), force_cap)
            position_command = 2. * min(initial_force / (PD_KP * PD_MAX_ADVANCE_M), 1.) - 1.
            for key, model in (('actor_state_dict', runner.alg.get_policy()),
                               ('critic_state_dict', runner.alg.critic)):
                command = position_command if key == 'actor_state_dict' else None
                init_state[key] = exo_initial_state(
                    init_state[key], model.state_dict(), initial_command=command,
                    from_motor=init_info.get('exo_policy_io') == MOTOR_POLICY_IO,
                )
        runner.alg.load(init_state, {'actor': True, 'critic': True}, strict=True)
        print(f'[CPU train] Initialized from {args.init_checkpoint}', flush=True)
    with env:
        runner.learn(args.iterations, init_at_random_ep_len=True)
    if args.task == 'exo_jump':
        (log_dir / 'training_efficiency.json').write_text(
            json.dumps(env.worlds.exoskeleton.report(), indent=2) + '\n')
    print(f'[CPU train] checkpoint: {log_dir / f"model_{runner.current_learning_iteration}.pt"}')


if __name__ == '__main__':
    main()
