"""Batched PPO training with native MuJoCo or CUDA Warp physics."""
import argparse
from datetime import datetime
import json
import math
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
    parser.add_argument('--exo-control', choices=('position-pd', 'force', 'force-policy', 'force-action', 'impedance-force'),
                        help='exoskeleton actuator mode (defaults to the checkpoint mode or position-pd)')
    parser.add_argument('--exo-mechanism', choices=('hooke_slide_ball_v3', 'waist_clevis_v4', 'hut_ring_clevis_v5', 'hut_ball_aft_v6'),
                        help='exoskeleton mount geometry (default: checkpoint mechanism or hooke_slide_ball_v3)')
    parser.add_argument('--apollo-mass', action='store_true',
                        help='apply NASA Apollo A7LB + PLSS/OPS lunar EVA mass & inertia proportions')
    parser.add_argument('--wangyu-eva', action='store_true',
                        help='apply China CMSA Wangyu (望宇) lunar spacesuit mass, inertia, bearing/bellows mobility, and regolith boot friction')
    parser.add_argument('--assist-drop-prob', type=float, default=0., metavar='PROB',
                        help='per-episode probability of zero/partial exoskeleton assist drop for robustness')
    parser.add_argument('--stance-transition', action='store_true',
                        help='train distributed boot loading and bilateral support during COM rise')
    parser.add_argument('--economical-hops', action='store_true',
                        help='train with completed-hop quality independent of flight duration')
    parser.add_argument('--long-hop-period', type=float, metavar='SECONDS',
                        help='favor longer landing-to-landing cycles at the requested forward speed')
    parser.add_argument('--keep-strut-heads', action='store_true',
                        help='keep the two force output heads intact when initializing load sharing')
    parser.add_argument('--reset-strut-heads', action='store_true',
                        help='symmetrize and re-center both strut output heads (+0.35 bias) to break dead-zone bias')
    parser.add_argument('--roll-balance', action='store_true',
                        help='penalize persistent pack and waist roll on either side')
    parser.add_argument('--posture-balance', action='store_true',
                        help='penalize persistent pack and waist roll and asymmetric push support')
    parser.add_argument('--vertical-support', action='store_true',
                        help='use world-frame vertical boot reaction for load sharing and touchdown impulse')
    parser.add_argument('--load-sharing', action='store_true',
                        help='train direct force actions with human energy and bilateral support impulse costs')
    parser.add_argument('--stance-center-weight', type=float, default=0.,
                        help='exoskeleton boot-midpoint reward weight (default: 0)')
    parser.add_argument('--human-energy-weight', type=float, metavar='WEIGHT',
                        help='override human_metabolic_proxy_power weight for load-sharing exo training')
    parser.add_argument('--torque-weight', type=float, metavar='WEIGHT',
                        help='override human_mean_squared_torque and peak_leg_squared_torque weights')
    parser.add_argument('--unloaded-linear-cost', type=float, metavar='WEIGHT',
                        help='linear unloaded strut force penalty in action_force_v1')
    parser.add_argument('--cot-energy-scale', action='store_true',
                        help='scale human energy cost by max(target_speed, 0.5) to optimize cost of transport')
    parser.add_argument('--assist-share-weight', type=float, metavar='WEIGHT',
                        help='bounded grounded propulsion/braking exoskeleton work-share bonus weight')
    parser.add_argument('--speed-track-boost', type=float, default=0., metavar='WEIGHT',
                        help='additional high-speed COM velocity tracking weight for exo_jump')
    parser.add_argument('--suit-ankle-pitch', action='store_true',
                        help='apply pressurized lunar boot ankle pitch limit (-18..+15 deg) and bellows stiffness/damping')
    parser.add_argument('--slope-adaptive-posture', action='store_true',
                        help='shift torso lean reference with terrain slope, unlock WaistPitch flexion, and use non-saturating flat-sole landing')
    parser.add_argument('--suit-major-joints', action='store_true',
                        help='apply pressurized EVA spacesuit damping, restoring stiffness, and ROM constraints to waist/hip/knee/arm joints')
    parser.add_argument('--relax-propulsion', action='store_true',
                        help='relax leg/waist PD velocity brake (kd), peak torque limits, and HipPitch/Knee extension control bounds')
    parser.add_argument('--lr', type=float, metavar='RATE',
                        help='override PPO learning rate')
    parser.add_argument('--lr-schedule', choices=('auto', 'fixed', 'adaptive'), default='auto',
                        help='PPO learning rate schedule (default: auto)')
    parser.add_argument('--min-init-std', type=float, metavar='STD',
                        help='minimum action std floor when initializing from a checkpoint')
    parser.add_argument('--min-strut-std', type=float, metavar='STD',
                        help='minimum strut action std floor when initializing from a checkpoint')
    parser.add_argument('--geometry', choices=('legacy', 'fitted', 'fitted_3box', 'bodyfit_v2_hull', 'fit3d_18v'), default='fitted',
                        help='suit collision geometry (default: fitted)')
    parser.add_argument('--steps-per-env', type=int, default=32)
    parser.add_argument('--terrain-size', type=int, choices=(16, 100), default=16,
                        help='lunar terrain size in meters: 16 (64-patch bank) or 100 (100x100m continuous)')
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument('--terrain-idx', type=int, choices=range(1, 65), default=None, metavar='1–64')
    terrain.add_argument('--terrain-idxs', nargs='+', type=int, choices=range(1, 65), metavar='1–64',
                         help='split forward-jump worlds across these terrains with independent headings')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--physics-backend', choices=('mjbatch', 'threaded', 'warp'), default='mjbatch',
                        help='physics backend (warp uses CUDA for the exoskeleton task)')
    parser.add_argument('--physics-workers', type=int, help='physics threads (default: automatic)')
    parser.add_argument('--torch-threads', type=int, help='PyTorch threads (default: automatic)')
    parser.add_argument('--update-device', choices=('auto', 'cpu', 'mps', 'cuda'), default='auto',
                        help='PPO update device (auto chooses CUDA with Warp)')
    parser.add_argument('--pushes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--init-checkpoint', type=Path,
                        help='initialize the jump actor/critic from a compatible checkpoint')
    args = parser.parse_args()
    if args.long_hop_period is not None:
        if args.task != 'exo_jump' or not (math.isfinite(args.long_hop_period) and args.long_hop_period > 0):
            parser.error('--long-hop-period requires exo_jump and a finite positive period')
        args.economical_hops = True
        args.keep_strut_heads = True
    args.posture_balance = args.posture_balance or args.stance_transition
    args.vertical_support = args.vertical_support or args.posture_balance or args.economical_hops or args.roll_balance
    args.load_sharing = args.load_sharing or args.vertical_support
    forward_tasks = ('forward_jump', 'exo_jump')
    full_cuda = args.physics_backend == 'warp'
    if full_cuda and (args.task != 'exo_jump' or args.update_device not in ('auto', 'cuda')):
        parser.error('--physics-backend warp requires --task exo_jump and --update-device cuda or auto')
    if not math.isfinite(args.stance_center_weight) or args.stance_center_weight < 0:
        parser.error('--stance-center-weight must be finite and nonnegative')
    if args.stance_center_weight and (args.task != 'exo_jump' or args.init_checkpoint is None):
        parser.error('--stance-center-weight requires exo_jump and an explicit --init-checkpoint')
    if args.load_sharing and args.task != 'exo_jump':
        parser.error('--load-sharing requires exo_jump')
    if args.assist_force is not None and args.task != 'exo_jump':
        parser.error('--assist-force 仅用于外骨骼跳跃任务')
    if args.exo_control is not None and args.task != 'exo_jump':
        parser.error('--exo-control 仅用于外骨骼跳跃任务')
    if args.terrain_size == 100:
        if args.terrain_idx is not None or args.terrain_idxs is not None:
            parser.error('--terrain-size 100 cannot be combined with --terrain-idx / --terrain-idxs')
    elif args.terrain_idxs is not None:
        if args.task not in forward_tasks:
            parser.error('--terrain-idxs 仅用于前进跳跃任务')
        if len(set(args.terrain_idxs)) != len(args.terrain_idxs) or len(args.terrain_idxs) > args.num_envs:
            parser.error('地形编号必须唯一，且每块地形至少有一个训练环境')
        args.terrain_idx = args.terrain_idxs[0]
    elif args.terrain_idx is None:
        args.terrain_idx = 12
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
    if full_cuda:
        args.update_device = 'cuda'
    elif args.update_device == 'auto':
        # Only default to MPS at the rollout size verified on the Mac.
        transitions = args.num_envs * args.steps_per_env
        args.update_device = 'mps' if mps_available and transitions >= 1024 * 32 else 'cpu'
    if args.update_device == 'mps' and not mps_available:
        parser.error('MPS is not available in this PyTorch environment')
    if args.update_device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available in this PyTorch environment')
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
        init_geom = init_info.get('geometry')
        fitted_family = {'fitted', 'fitted_3box', 'bodyfit_v2_hull', 'fit3d_18v'}
        if init_geom != args.geometry:
            if init_geom in fitted_family and args.geometry in fitted_family:
                print(f'[CPU train] Geometry transfer: {init_geom} -> {args.geometry}', flush=True)
            else:
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
            from moon_scene import (APOLLO_MASS_PROFILE, MASS_PROFILES, RELAXED_PROPULSION_PROFILE,
                                    SUIT_ANKLE_PITCH_PROFILE, SUIT_MAJOR_JOINTS_PROFILE, WANGYU_EVA_PROFILE)
            init_pitch_profile = init_info.get('ankle_pitch_profile')
            if init_pitch_profile not in (None, SUIT_ANKLE_PITCH_PROFILE):
                parser.error(f'未知的初始化脚踝俯仰配置：{init_pitch_profile}')
            if args.suit_ankle_pitch or init_pitch_profile == SUIT_ANKLE_PITCH_PROFILE:
                task_options['suit_ankle_pitch'] = True
            if args.slope_adaptive_posture or init_info.get('slope_adaptive_posture'):
                task_options['slope_adaptive_posture'] = True
            init_major_profile = init_info.get('suit_major_joints_profile')
            if init_major_profile not in (None, SUIT_MAJOR_JOINTS_PROFILE):
                parser.error(f'未知的初始化大关节航天服配置：{init_major_profile}')
            if args.suit_major_joints or init_major_profile == SUIT_MAJOR_JOINTS_PROFILE:
                task_options['suit_major_joints'] = True
            init_prop_profile = init_info.get('propulsion_profile')
            if init_prop_profile not in (None, RELAXED_PROPULSION_PROFILE):
                parser.error(f'未知的初始化驱动放宽配置：{init_prop_profile}')
            if args.relax_propulsion or init_prop_profile == RELAXED_PROPULSION_PROFILE:
                task_options['relax_propulsion'] = True
            init_mass_profile = init_info.get('apollo_mass_profile')
            if init_mass_profile not in (None, *MASS_PROFILES):
                parser.error(f'未知的初始化航天服质量配置：{init_mass_profile}')
            if args.wangyu_eva:
                task_options['mass_profile'] = WANGYU_EVA_PROFILE
            elif args.apollo_mass:
                task_options['mass_profile'] = APOLLO_MASS_PROFILE
            elif init_mass_profile is not None:
                task_options['mass_profile'] = init_mass_profile
        if args.task == 'exo_jump':
            from exoskeleton import (ACTION_FORCE_IO, DEFAULT_FORCE_N, DIRECT_FORCE_REVISIONS,
                                     FORCE_CONTROL_REVISIONS, FORCE_POLICY_IO, IMPEDANCE_FORCE_IO,
                                     MECHANISM, MECHANISMS, POLICY_IO, POLICY_TIMED_FORCE_IO,
                                     validate_assist_force, validate_policy_io)
            requested = DEFAULT_FORCE_N if args.assist_force is None else args.assist_force
            task_options['assist_force'] = validate_assist_force(requested)
            task_options['stance_center_weight'] = args.stance_center_weight
            source_exo_io = validate_policy_io(init_info.get('exo_policy_io'))
            target_exo_io = {'position-pd': POLICY_IO, 'force': FORCE_POLICY_IO,
                             'force-policy': POLICY_TIMED_FORCE_IO,
                             'force-action': ACTION_FORCE_IO,
                             'impedance-force': IMPEDANCE_FORCE_IO}.get(args.exo_control)
            if target_exo_io is None:
                target_exo_io = source_exo_io if source_exo_io in FORCE_CONTROL_REVISIONS else POLICY_IO
            if args.load_sharing and target_exo_io not in DIRECT_FORCE_REVISIONS:
                parser.error('--load-sharing requires force-action or impedance-force control')
            init_mechanism = init_info.get('exo_mechanism')
            if init_mechanism not in (None, *MECHANISMS):
                parser.error(f'未知的初始化外骨骼机构：{init_mechanism}')
            task_options['exo_mechanism'] = args.exo_mechanism or init_mechanism or MECHANISM
            task_options['load_sharing'] = args.load_sharing
            task_options['vertical_support'] = args.vertical_support
            task_options['posture_balance'] = args.posture_balance
            task_options['roll_balance'] = args.roll_balance
            task_options['stance_transition'] = args.stance_transition
            task_options['economical_hops'] = args.economical_hops
            task_options['long_hop_period'] = args.long_hop_period
            task_options['exo_policy_io'] = target_exo_io
            if args.assist_drop_prob:
                if not (math.isfinite(args.assist_drop_prob) and 0. <= args.assist_drop_prob <= 1.):
                    parser.error('--assist-drop-prob must be in [0, 1]')
                task_options['assist_drop_prob'] = args.assist_drop_prob
            reward_overrides = {}
            if args.human_energy_weight is not None:
                if not (math.isfinite(args.human_energy_weight) and args.human_energy_weight >= 0):
                    parser.error('--human-energy-weight must be finite and nonnegative')
                reward_overrides['human_metabolic_proxy_power'] = args.human_energy_weight
            if args.torque_weight is not None:
                if not (math.isfinite(args.torque_weight) and args.torque_weight >= 0):
                    parser.error('--torque-weight must be finite and nonnegative')
                reward_overrides['human_mean_squared_torque'] = args.torque_weight
                reward_overrides['peak_leg_squared_torque'] = args.torque_weight
            if args.unloaded_linear_cost is not None:
                if not (math.isfinite(args.unloaded_linear_cost) and args.unloaded_linear_cost >= 0):
                    parser.error('--unloaded-linear-cost must be finite and nonnegative')
                reward_overrides['unloaded_strut_force_linear'] = args.unloaded_linear_cost
            if args.cot_energy_scale:
                reward_overrides['cot_energy_scale'] = True
            if args.assist_share_weight is not None:
                if not (math.isfinite(args.assist_share_weight) and args.assist_share_weight >= 0):
                    parser.error('--assist-share-weight must be finite and nonnegative')
                reward_overrides['assist_work_share_weight'] = args.assist_share_weight
            if args.speed_track_boost:
                if not (math.isfinite(args.speed_track_boost) and args.speed_track_boost >= 0):
                    parser.error('--speed-track-boost must be finite and nonnegative')
                task_options['speed_track_boost'] = args.speed_track_boost
            if reward_overrides:
                if not (args.load_sharing or args.vertical_support):
                    parser.error('reward weight overrides require --load-sharing / --vertical-support')
                task_options['reward_overrides'] = reward_overrides
            migrate_exo_policy = source_exo_io is None or source_exo_io != target_exo_io
    except ValueError as exc:
        parser.error(str(exc))
    env = env_class(args.num_envs, args.terrain_idx, args.pushes,
                    args.physics_backend, args.physics_workers, args.seed,
                    geometry=args.geometry, terrain_indices=args.terrain_idxs,
                    terrain_size=args.terrain_size, **task_options)
    mix_strut_heads = args.load_sharing and not (args.posture_balance or args.roll_balance or args.keep_strut_heads) and init_state is not None and not migrate_exo_policy
    if mix_strut_heads:
        env.cfg['strut_head_warm_start_mix'] = .5
    env.cfg['update_device'] = args.update_device
    if args.init_checkpoint is not None:
        env.cfg['init_checkpoint'] = str(args.init_checkpoint.resolve())
    model = dict(class_name='rsl_rl.models.MLPModel', hidden_dims=(128, 128),
                 activation='elu', obs_normalization=True)
    learner_config = {'class_name': 'rsl_rl.algorithms.PPO'}
    if args.update_device != 'cpu' and not full_cuda:
        learner_config = {
            'class_name': 'cpu_sampling_ppo.CpuSamplingAcceleratedPPO',
            'learner_device': args.update_device,
        }
    cfg = dict(
        num_steps_per_env=args.steps_per_env, save_interval=25, logger='tensorboard',
        obs_groups={'actor': ['policy'], 'critic': ['critic']},
        actor=dict(model, distribution_cfg=dict(class_name='rsl_rl.modules.GaussianDistribution',
                                                init_std=.25, std_type='scalar')),
        critic=dict(model),
        algorithm=dict(**learner_config, num_learning_epochs=5,
                       num_mini_batches=4,
                       learning_rate=(args.lr if args.lr is not None
                                      else (1e-4 if args.stance_center_weight else 3e-4)),
                       schedule=(('fixed' if args.stance_center_weight else 'adaptive')
                                 if args.lr_schedule == 'auto' else args.lr_schedule),
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
    terrain_description = ('100m' if args.terrain_size == 100
                           else ','.join(map(str, args.terrain_idxs or [args.terrain_idx])))
    physics_description = ('Warp CUDA' if full_cuda
                           else f'{args.physics_backend} with {env.workers} physics threads')
    print(f'[Train] task={args.task}, geometry={args.geometry}, '
          f'{args.num_envs} worlds on terrain {terrain_description}, '
          f'{physics_description}, rollout={env.device}, '
          f'PPO update={args.update_device} → {log_dir}', flush=True)
    if 'jump_height_range_m' in env.cfg:
        low, high = env.cfg['jump_height_range_m']
        print(f'[CPU train] Jump height range: {low:g}–{high:g} m', flush=True)
    if 'jump_speed_range_mps' in env.cfg:
        low, high = env.cfg['jump_speed_range_mps']
        print(f'[CPU train] Forward jump speed range: {low:g}–{high:g} m/s; free jump height', flush=True)

    if args.task == 'exo_jump':
        print(f'[Train] Exoskeleton {env.cfg["exo_policy_io"]}: '
              f'up to {env.cfg["assist_force_n"]:g} N per strut', flush=True)

    class LocalRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            keys = ('task', 'jump_actuation', 'jump_height_range_m', 'jump_speed_range_mps',
                    'geometry', 'terrain_size_m', 'assist_force_n', 'exo_mechanism', 'exo_efficiency_reward',
                    'exo_policy_io', 'stance_center_weight', 'ankle_roll_profile', 'ankle_pitch_profile',
                    'suit_major_joints_profile', 'propulsion_profile', 'apollo_mass_profile',
                    'slope_adaptive_posture', 'exo_action_force_costs', 'economical_hops',
                    'long_hop_period_s', 'roll_balance')
            task_info = {key: env.cfg[key] for key in keys if key in env.cfg}
            super().save(path, {**(infos or {}), **task_info})

    runner = LocalRunner(env, cfg, log_dir=str(log_dir), device=env.device)
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
            from exoskeleton import INITIAL_ASSIST_N, PD_KP, PD_MAX_ADVANCE_M, exo_initial_state
            force_cap = env.cfg['assist_force_n']
            if target_exo_io in FORCE_CONTROL_REVISIONS:
                # Start both force outputs at 80 N or the configured cap before gating.
                command_fraction = min(INITIAL_ASSIST_N / force_cap, 1.) if force_cap else 0.
            else:
                initial_force = min(init_info.get('assist_force_n', INITIAL_ASSIST_N), force_cap)
                command_fraction = min(initial_force / (PD_KP * PD_MAX_ADVANCE_M), 1.)
            for key, model in (('actor_state_dict', runner.alg.get_policy()),
                               ('critic_state_dict', runner.alg.critic)):
                command = 2. * command_fraction - 1. if key == 'actor_state_dict' else None
                init_state[key] = exo_initial_state(
                    init_state[key], model.state_dict(), initial_command=command,
                    replace_strut_io=source_exo_io is not None,
                )
        if args.reset_strut_heads and args.task == 'exo_jump':
            actor = init_state['actor_state_dict']
            w = actor['mlp.4.weight'].clone()
            b = actor['mlp.4.bias'].clone()
            w_mean = w[-2:].mean(dim=0, keepdim=True) * .25
            w[-2:] = w_mean.expand(2, -1)
            b[-2:] = .35
            actor['mlp.4.weight'] = w
            actor['mlp.4.bias'] = b
        elif mix_strut_heads:
            # Bring the two inherited output means closer for exploration. The
            # learned heads remain independent throughout optimization and replay.
            actor = init_state['actor_state_dict']
            for key in ('mlp.4.weight', 'mlp.4.bias'):
                rods = actor[key][-2:]
                rods.lerp_(rods.mean(dim=0, keepdim=True), .5)
        if args.min_init_std is not None or args.min_strut_std is not None:
            actor = init_state['actor_state_dict']
            std_key = 'distribution.std_param'
            if std_key in actor:
                std_val = actor[std_key].clone()
                if args.min_init_std is not None:
                    std_val.clamp_min_(args.min_init_std)
                if args.min_strut_std is not None and args.task == 'exo_jump':
                    std_val[-2:].clamp_min_(args.min_strut_std)
                actor[std_key] = std_val
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
