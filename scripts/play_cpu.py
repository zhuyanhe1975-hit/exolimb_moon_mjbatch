"""Native MuJoCo CPU policy replay, independent of CUDA/Warp/REAF."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from zipfile import is_zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from provenance import training_env_config

EXO_CHECKPOINT = ROOT / 'bests/exo_jump_position_pd500_2026-11-19/model_199.pt'
EXO_BASE_CHECKPOINT = ROOT / 'bests/forward_jump_boot_yaw_2026-11-18/model_104.pt'
BEST_CHECKPOINTS = {
    'exo_jump': EXO_CHECKPOINT,
    'forward_jump': EXO_BASE_CHECKPOINT,
    'jump': ROOT / 'bests/jump_flow_rebound_2026-11-18/model_149.pt',
}
DEFAULT_BEST_TASK = 'exo_jump'
IMPORTED_CHECKPOINT = ROOT / 'logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt'


def training_terrain(run_dir):
    terrain = training_env_config(run_dir).get('terrain_idx')
    return terrain if type(terrain) is int and 1 <= terrain <= 64 else None


def latest_training_checkpoint(log_dir, task=None):
    candidates = []
    for run in log_dir.iterdir() if log_dir.is_dir() else ():
        if not run.is_dir() or not (run / 'terrain_manifest.json').is_file():
            continue
        if training_terrain(run) is None:
            continue
        if task is not None and training_env_config(run).get('task', 'stand') != task:
            continue
        for checkpoint in run.glob('model_*.pt'):
            match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
            if match and checkpoint.is_file():
                try:
                    candidates.append((checkpoint.stat().st_mtime_ns, int(match[1]), checkpoint))
                except FileNotFoundError:
                    continue
    for _, _, checkpoint in sorted(candidates, reverse=True):
        # PyTorch writes the ZIP directory at the end of a checkpoint save.
        if is_zipfile(checkpoint):
            return checkpoint
    return None


def parse_args(argv=None):
    entry = os.environ.get('MOON_RUN_ENTRY', 'run.sh')
    latest_default = entry == 'run_cpu.sh'
    parser = argparse.ArgumentParser(
        prog=f'./{entry}', allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='月面策略 CPU 回放：classic/Studio 原生窗口、物理代理或航天服蒙皮、共享月面多人，无随机扰动。',
        epilog="""示例：
  ./run.sh
  ./run.sh --list-bests
  ./run.sh --best --viewer studio --viz skin --jump-speed 1.0
  ./run.sh --best jump --jump-height 0.15
  ./run.sh --viewer studio --viz skin --num-envs 8
  ./run.sh --terrain-idx 24
  ./run.sh --checkpoint /absolute/path/model_200.pt
  ./run.sh --headless --seconds 5 --report outputs/cpu_check.json

窗口快捷键：/切换跟踪，逗号上一人，句号下一人，A总览，Esc退出跟踪。
空格暂停/继续，R复位，[ / ] 切换地形，关闭窗口退出。
施力：双击选中身体，Control+右键拖动施加力，Control+左键拖动施加转矩。
物理120Hz / 策略60Hz；窗口按1×实时播放；无窗口模式全速运行。
支持 classic/studio 与 physx/skin，自身动力学和策略均在CPU上；各人独立仿真，不相互碰撞。
run.sh 专用：--backend cpu（默认）或 --backend playground（CUDA/Warp 站立回放）。
环境变量：CPU_PYTHON 指定 Python。Linux 在没有 .venv-cpu 时复用本机的 Playground 环境。
""")
    parser.add_argument('--episode-seconds', type=int, default=10, help='单回合仿真时限（秒）')
    parser.add_argument('--headless', action='store_true', help='不打开窗口；必须设置 --seconds')
    parser.add_argument('--seconds', type=float, default=0, help='运行的仿真秒数；默认0表示持续运行到窗口关闭')
    parser.add_argument('--terrain-idx', '--terrain_idx', type=int, choices=range(1, 65), metavar='1–64',
                        help='16米地形编号，默认采用检查点的训练地形或第12块')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')
    default_policy = '最新本地训练检查点' if latest_default else '导入的站立策略'
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--checkpoint', '--resume-path', '--resume_path', type=Path,
                        help=f'指定策略检查点；无任务或策略选择参数时默认使用{default_policy}')
    policy.add_argument('--best', nargs='?', const='default', metavar='TASK',
                        choices=('default', *BEST_CHECKPOINTS),
                        help='使用归档策略。可选 jump、forward_jump、exo_jump；省略任务时采用 --task 或 exo_jump')
    policy.add_argument('--latest-training', action='store_true',
                        help='自动选择最新本地训练检查点，可用 --task 筛选')
    policy.add_argument('--list-bests', action='store_true',
                        help='列出当前归档及对应检查点，然后退出')
    parser.add_argument('--task', choices=('stand', 'jump', 'forward_jump', 'exo_jump'),
                        help='按任务筛选检查点，也可与 --best 一起指定归档任务')
    height_args = parser.add_mutually_exclusive_group()
    height_args.add_argument('--jump-height', type=float, metavar='M',
                             help='设定高度指令型跳跃策略的目标跳高（米），例如 0.15')
    height_args.add_argument('--jump-height-cm', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--jump-speed', type=float, metavar='MPS',
                        help='设定前进跳跃的目标速度（米/秒），例如 1.0')
    parser.add_argument('--assist-force', type=float, metavar='NEWTONS',
                        help='外骨骼每根支杆的助力上限（0–500 N，默认采用检查点记录的值）')
    parser.add_argument('--require-force-control', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--geometry', choices=('legacy', 'fitted'),
                        help='物理几何版本，默认采用检查点的训练配置')
    parser.add_argument('--terrain-transfer', '--terrain_transfer', action='store_true', help='显式允许检查点向当前地形迁移')
    parser.add_argument('--report', type=Path, help='退出时保存 JSON 验证报告')
    compatible = parser.add_argument_group('计算与显示')
    compatible.add_argument('--device', choices=('cpu',), default='cpu', help='计算设备，仅支持cpu')
    compatible.add_argument('--viewer', choices=('classic','studio'), default='classic', help='窗口类型，默认classic')
    compatible.add_argument('--viz', choices=('physx','skin'), default='physx', help='物理代理几何或航天服蒙皮，默认physx')
    compatible.add_argument('--num-envs', '--num_envs', '--nworld', type=int, default=1, help='独立环境/人数，默认1；多人共享同一块地形')
    args = parser.parse_args(argv)
    if args.list_bests:
        for task, path in BEST_CHECKPOINTS.items():
            default = '（默认）' if task == DEFAULT_BEST_TASK else ''
            missing = '' if path.is_file() else '（文件缺失）'
            print(f'{task}{default}  {path.relative_to(ROOT)}{missing}')
        parser.exit()
    if args.best is not None:
        if args.best != 'default' and args.task not in (None, args.best):
            parser.error(f'--best {args.best} 与 --task {args.task} 不一致')
        args.task = args.task or (DEFAULT_BEST_TASK if args.best == 'default' else args.best)
        if args.task not in BEST_CHECKPOINTS:
            parser.error(f'任务 {args.task} 没有归档检查点，可用 --list-bests 查看')
        args.checkpoint = BEST_CHECKPOINTS[args.task]
    forward_tasks = ('forward_jump', 'exo_jump')
    if args.assist_force is not None:
        if args.task not in (None, 'exo_jump'):
            parser.error('--assist-force 只能用于外骨骼跳跃任务')
        args.task = 'exo_jump'
    fixed_height = args.jump_height
    if args.jump_height_cm is not None:
        fixed_height = args.jump_height_cm / 100
    if fixed_height is not None:
        if args.task == 'stand':
            parser.error('--jump-height 只能用于跳跃策略')
    if args.jump_speed is not None:
        if args.task is not None and args.task not in forward_tasks:
            parser.error('--jump-speed 只能用于前进跳跃策略')
        if args.task is None and args.checkpoint is None:
            args.task = 'forward_jump'
    elif fixed_height is not None and args.task is None and args.checkpoint is None:
        args.task = 'jump'
    if args.num_envs <= 0:
        parser.error('--num-envs 必须是正整数')
    import math
    if args.episode_seconds < 1:
        parser.error('--episode-seconds 必须为正整数')
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds 必须是有限的非负数')
    if args.headless and args.seconds <= 0:
        parser.error('--headless 必须配合 --seconds > 0')
    if args.seconds > 0 and round(args.seconds * 60) < 1:
        parser.error('--seconds 至少需要一个控制步（约0.0167秒）')
    if args.checkpoint is None and (args.latest_training or latest_default or args.task):
        log_root = 'cpu_exo_train' if args.task == 'exo_jump' else 'cpu_train'
        args.checkpoint = latest_training_checkpoint(ROOT / 'logs' / log_root, task=args.task)
        if args.checkpoint is None and args.task == 'exo_jump':
            args.checkpoint = EXO_CHECKPOINT if EXO_CHECKPOINT.is_file() else EXO_BASE_CHECKPOINT
        if args.checkpoint is None:
            if args.task == 'forward_jump':
                parser.error('未找到前进跳跃检查点，请先运行 ./train_forward_jump.sh')
            if args.task == 'jump':
                parser.error('未找到本地跳跃检查点，请先运行 ./train_jump.sh')
            print('[CPU] 未找到已保存的本地训练检查点，加载导入策略。', flush=True)
    if args.checkpoint is None:
        args.checkpoint = IMPORTED_CHECKPOINT
    if args.terrain_idx is None:
        args.terrain_idx = training_terrain(args.checkpoint.parent) or 12
    if not args.checkpoint.is_file():
        parser.error(f'检查点不存在：{args.checkpoint}')
    return parser, args, fixed_height


def main():
    parser, args, fixed_height = parse_args()
    forward_tasks = ('forward_jump', 'exo_jump')
    import mujoco
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from cpu_jump_task import (BILATERAL_SOLE_ACTUATION, COORDINATED_LANDING_ACTUATION,
                               MIRRORED_LANDING_ACTUATION, PHASE_MIRRORED_LANDING_ACTUATION,
                               FORWARD_FACING_BOOT_ACTUATION, SOLE_COLLISION_ACTUATIONS,
                               SOLE_EDGE_PROBE_ACTUATIONS, SOLE_PLANE_LANDING_ACTUATION,
                               SOLE_PROBE_ACTUATIONS)
    from cpu_simulation import CpuWorlds
    from moon_scene import (RELAXED_PROPULSION_PROFILE, SUIT_ANKLE_PITCH_PROFILE,
                            SUIT_MAJOR_JOINTS_PROFILE, scene_spec)
    from provenance import verify_checkpoint
    from suit_proxy_geometry import LEGACY, PROFILES
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    verify_checkpoint(args.checkpoint, allow_transfer=args.terrain_transfer)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    infos = checkpoint.get('infos') or {}
    env_config = training_env_config(args.checkpoint.parent)
    task = infos.get('task', env_config.get('task', 'stand'))
    if task not in ('stand', 'jump', *forward_tasks):
        parser.error(f'未知检查点任务：{task}')
    if args.task == 'exo_jump' and task == 'forward_jump':
        task = 'exo_jump'
    if args.task is not None and args.task != task:
        parser.error(f'检查点任务为 {task}，与 --task {args.task} 不一致')
    if args.jump_speed is not None and task not in forward_tasks:
        parser.error('--jump-speed 只能用于前进跳跃策略')
    actuation = infos.get('jump_actuation', env_config.get('jump_actuation'))
    assist_force = None
    exo_policy_io = None
    exo_pd_force = None
    stance_transition = False
    posture_balance = False
    roll_balance = False
    vertical_support = False
    suit_ankle_roll = False
    ankle_pitch_profile = infos.get('ankle_pitch_profile', env_config.get('ankle_pitch_profile'))
    if ankle_pitch_profile not in (None, SUIT_ANKLE_PITCH_PROFILE):
        parser.error(f'未知检查点脚踝俯仰配置：{ankle_pitch_profile}')
    suit_ankle_pitch = ankle_pitch_profile == SUIT_ANKLE_PITCH_PROFILE
    major_profile = infos.get('suit_major_joints_profile', env_config.get('suit_major_joints_profile'))
    if major_profile not in (None, SUIT_MAJOR_JOINTS_PROFILE):
        parser.error(f'未知检查点大关节航天服配置：{major_profile}')
    suit_major_joints = major_profile == SUIT_MAJOR_JOINTS_PROFILE
    prop_profile = infos.get('propulsion_profile', env_config.get('propulsion_profile'))
    if prop_profile not in (None, RELAXED_PROPULSION_PROFILE):
        parser.error(f'未知检查点驱动放宽配置：{prop_profile}')
    relax_propulsion = prop_profile == RELAXED_PROPULSION_PROFILE
    if args.require_force_control and task != 'exo_jump':
        parser.error('请指定原生力控外骨骼检查点')
    if task == 'exo_jump':
        if actuation != FORWARD_FACING_BOOT_ACTUATION:
            parser.error('外骨骼回放需要 v32 前进跳跃策略或由它续训的外骨骼策略')
        from exoskeleton import (ACTION_FORCE_IO, DEFAULT_FORCE_N, FORCE_CONTROL_REVISIONS, LOAD_SHARING_REWARD,
                                 POLICY_IO, POSTURE_SUPPORT_REWARD, STANCE_TRANSITION_REWARD, VERTICAL_SUPPORT_REWARD, validate_assist_force, validate_policy_io)
        reward_revision = infos.get('exo_efficiency_reward', env_config.get('exo_efficiency_reward', {})).get('revision')
        stance_transition = reward_revision == STANCE_TRANSITION_REWARD['revision']
        posture_balance = stance_transition or reward_revision == POSTURE_SUPPORT_REWARD['revision']
        roll_balance = infos.get('roll_balance', env_config.get('roll_balance', False))
        vertical_support = posture_balance or roll_balance or reward_revision == VERTICAL_SUPPORT_REWARD['revision']
        default_force = infos.get('assist_force_n', DEFAULT_FORCE_N)
        try:
            assist_force = validate_assist_force(
                default_force if args.assist_force is None else args.assist_force)
            exo_policy_io = validate_policy_io(infos.get('exo_policy_io', env_config.get('exo_policy_io')))
            exo_pd_force = assist_force if exo_policy_io == POLICY_IO else None
            suit_ankle_roll = exo_policy_io == ACTION_FORCE_IO
        except ValueError as exc:
            parser.error(str(exc))
        if args.require_force_control and exo_policy_io not in FORCE_CONTROL_REVISIONS:
            parser.error('请指定原生力控外骨骼检查点')
    height_range = None
    if task != 'stand':
        from cpu_jump_task import (DEFAULT_JUMP_HEIGHT_RANGE, FREE_HEIGHT_FORWARD_ACTUATIONS,
                                   HEIGHT_JUMP_ACTUATIONS,
                                   validate_jump_height_range)
        if fixed_height is not None and actuation in FREE_HEIGHT_FORWARD_ACTUATIONS:
            parser.error('当前前进跳跃策略使用自由高度，请省略 --jump-height')
        if actuation in HEIGHT_JUMP_ACTUATIONS:
            try:
                height_range = validate_jump_height_range(infos.get(
                    'jump_height_range_m', env_config.get('jump_height_range_m', DEFAULT_JUMP_HEIGHT_RANGE)))
            except (TypeError, ValueError) as exc:
                parser.error(f'检查点跳高范围无效：{exc}')
    if fixed_height is not None:
        if height_range is None:
            parser.error('此检查点不支持指定跳高，请选用新版事件驱动跳跃策略')
        low, high = height_range
        if not low <= fixed_height <= high:
            parser.error(f'目标跳高须在 {low:g}–{high:g} 米之间')
    speed_range = None
    if task in forward_tasks:
        from cpu_forward_jump import DEFAULT_SPEED_RANGE, validate_speed_range
        try:
            speed_range = validate_speed_range(infos.get(
                'jump_speed_range_mps', env_config.get('jump_speed_range_mps', DEFAULT_SPEED_RANGE)))
        except (TypeError, ValueError) as exc:
            parser.error(f'检查点速度范围无效：{exc}')
        low, high = speed_range
        if args.jump_speed is not None and not low <= args.jump_speed <= high:
            parser.error(f'目标速度须在 {low:g}–{high:g} 米/秒之间')
    trained_geometry = infos.get('geometry', env_config.get('geometry', LEGACY))
    geometry = args.geometry or trained_geometry
    if geometry not in PROFILES:
        parser.error(f'未知检查点物理几何：{geometry}')
    if geometry != trained_geometry:
        print(f'[CPU] Geometry override: checkpoint={trained_geometry}; playback={geometry}', flush=True)
    startup_started=time.monotonic()
    print(f'[Startup] Preparing {args.num_envs} independent CPU environments...',flush=True)
    physics_spec = scene_spec(geometry=geometry, sole_probes=actuation in SOLE_PROBE_ACTUATIONS,
                              exoskeleton=assist_force is not None, exo_pd_force=exo_pd_force,
                              sole_collision=actuation in SOLE_COLLISION_ACTUATIONS,
                              sole_edge_probes=actuation in SOLE_EDGE_PROBE_ACTUATIONS,
                              suit_ankle_roll=suit_ankle_roll, suit_ankle_pitch=suit_ankle_pitch,
                              suit_major_joints=suit_major_joints, relax_propulsion=relax_propulsion,
                              vertical_support=vertical_support, stance_transition=stance_transition)
    worlds = CpuWorlds(physics_spec, args.num_envs, args.terrain_idx,
                       edge_start=args.require_force_control, inward_heading=task in forward_tasks, assist_force=assist_force)
    model, data, sim, device = worlds.model, worlds.data[0], worlds.sim, worlds.device
    print(f'[Startup] CPU environments ready in {time.monotonic()-startup_started:.2f}s',flush=True)
    jump_counter = None
    jump_commands = None
    event_jump = None
    adaptive_jump = None
    forward_jump = None
    track_hop_distance = False
    track_support = False
    if task != 'stand':
        from cpu_jump_task import (ABSORBING_HOP_FORWARD_ACTUATION, ADAPTIVE_JUMP_ACTUATION,
                                   BALANCED_FULL_SOLE_ACTUATION, BROAD_JUMP_FORWARD_ACTUATION,
                                   CONTACT_JUMP_ACTUATION, DISTRIBUTED_SOLE_ACTUATION,
                                   COORDINATED_BROAD_JUMP_ACTUATION, COORDINATED_TAKEOFF_ACTUATION, FLOW_JUMP_ACTUATION,
                                   FORWARD_JUMP_ACTUATIONS, FREE_HEIGHT_FORWARD_ACTUATION,
                                   FULL_SOLE_FORWARD_ACTUATION, LOW_SPEED_SOLE_ACTUATION,
                                   HIGH_SPEED_LEAN_FORWARD_ACTUATION, JUMP_ACTUATION,
                                   LANDING_PREPARATION_ACTUATION, LONG_STRIDE_FORWARD_ACTUATION,
                                   PHASE_LEAN_FORWARD_ACTUATION, REBOUND_JUMP_ACTUATION,
                                   SPEED_TRACKING_FORWARD_ACTUATION, STANCE_FLOW_TAKEOFF_ACTUATION,
                                   STANCE_STRIDE_TAKEOFF_ACTUATION, SUPPORTED_TAKEOFF_ACTUATION,
                                   STABLE_FULL_SOLE_ACTUATION, TERRAIN_ALIGNED_ACTUATION,
                                   TIMED_JUMP_ACTUATION,
                                   AdaptiveJumpCommands, FlowJumpCommands, ReboundJumpCommands,
                                   CompliantJumpCommands, ContactJumpCommands, JumpCommands,
                                   JumpCounter, foot_airborne, jump_height, jump_observation)
        height_controllers = {ADAPTIVE_JUMP_ACTUATION: AdaptiveJumpCommands,
                              REBOUND_JUMP_ACTUATION: ReboundJumpCommands,
                              FLOW_JUMP_ACTUATION: FlowJumpCommands}
        if actuation in FORWARD_JUMP_ACTUATIONS:
            from cpu_forward_jump import (AbsorbingHopForwardJumpCommands, BalancedFullSoleForwardJumpCommands,
                                          BilateralSoleLandingCommands, BroadJumpForwardCommands,
                                          CoordinatedBroadJumpCommands, CoordinatedLandingCommands,
                                          CoordinatedTakeoffCommands,
                                          DistributedSoleForwardJumpCommands, ForwardFacingBootCommands, ForwardJumpCommands,
                                          FreeHeightForwardJumpCommands, FullSoleForwardJumpCommands,
                                          HighSpeedLeanForwardJumpCommands, LandingPreparationCommands,
                                          LongStrideForwardJumpCommands, LowSpeedSoleLandingCommands,
                                          MirroredLandingCommands, PhaseMirroredLandingCommands,
                                          PhaseLeanForwardJumpCommands,
                                          SpeedTrackingForwardJumpCommands, StableFullSoleForwardJumpCommands,
                                          SolePlaneMirroredLandingCommands, StanceFlowTakeoffCommands,
                                          StanceStrideTakeoffCommands,
                                          SupportedTakeoffCommands, TerrainAlignedLandingCommands)
            if actuation in FREE_HEIGHT_FORWARD_ACTUATIONS:
                controller = {
                    FREE_HEIGHT_FORWARD_ACTUATION: FreeHeightForwardJumpCommands,
                    SPEED_TRACKING_FORWARD_ACTUATION: SpeedTrackingForwardJumpCommands,
                    LONG_STRIDE_FORWARD_ACTUATION: LongStrideForwardJumpCommands,
                    PHASE_LEAN_FORWARD_ACTUATION: PhaseLeanForwardJumpCommands,
                    HIGH_SPEED_LEAN_FORWARD_ACTUATION: HighSpeedLeanForwardJumpCommands,
                    ABSORBING_HOP_FORWARD_ACTUATION: AbsorbingHopForwardJumpCommands,
                    BROAD_JUMP_FORWARD_ACTUATION: BroadJumpForwardCommands,
                    COORDINATED_BROAD_JUMP_ACTUATION: CoordinatedBroadJumpCommands,
                    COORDINATED_TAKEOFF_ACTUATION: CoordinatedTakeoffCommands,
                    SUPPORTED_TAKEOFF_ACTUATION: SupportedTakeoffCommands,
                    STANCE_FLOW_TAKEOFF_ACTUATION: StanceFlowTakeoffCommands,
                    STANCE_STRIDE_TAKEOFF_ACTUATION: StanceStrideTakeoffCommands,
                    LANDING_PREPARATION_ACTUATION: LandingPreparationCommands,
                    FULL_SOLE_FORWARD_ACTUATION: FullSoleForwardJumpCommands,
                    STABLE_FULL_SOLE_ACTUATION: StableFullSoleForwardJumpCommands,
                    BALANCED_FULL_SOLE_ACTUATION: BalancedFullSoleForwardJumpCommands,
                    DISTRIBUTED_SOLE_ACTUATION: DistributedSoleForwardJumpCommands,
                    TERRAIN_ALIGNED_ACTUATION: TerrainAlignedLandingCommands,
                    LOW_SPEED_SOLE_ACTUATION: LowSpeedSoleLandingCommands,
                    BILATERAL_SOLE_ACTUATION: BilateralSoleLandingCommands,
                    COORDINATED_LANDING_ACTUATION: CoordinatedLandingCommands,
                    MIRRORED_LANDING_ACTUATION: MirroredLandingCommands,
                    PHASE_MIRRORED_LANDING_ACTUATION: PhaseMirroredLandingCommands,
                    SOLE_PLANE_LANDING_ACTUATION: SolePlaneMirroredLandingCommands,
                    FORWARD_FACING_BOOT_ACTUATION: ForwardFacingBootCommands,
                }[actuation]
                controller_options = {'compact_arm_swing': True, 'suit_ankle_roll': suit_ankle_roll} if (
                    task == 'exo_jump' and exo_policy_io in FORCE_CONTROL_REVISIONS) else {}
                if suit_ankle_pitch:
                    controller_options['suit_ankle_pitch'] = True
                if suit_major_joints:
                    controller_options['suit_major_joints'] = True
                if relax_propulsion:
                    controller_options['relax_propulsion'] = True
                if infos.get('slope_adaptive_posture', env_config.get('slope_adaptive_posture', False)):
                    controller_options['slope_adaptive_posture'] = True
                period = infos.get('long_hop_period_s', env_config.get('long_hop_period_s'))
                if period is not None:
                    controller_options['long_hop_period'] = period
                forward_jump = controller(
                    worlds, speed_range=speed_range, fixed_speed=args.jump_speed, **controller_options)
            else:
                forward_jump = ForwardJumpCommands(
                    worlds, speed_range=speed_range, fixed_speed=args.jump_speed,
                    height_range=height_range, fixed_height=fixed_height)
            if infos.get('economical_hops', env_config.get('economical_hops', False)):
                forward_jump.economical_hops = True
            adaptive_jump = event_jump = jump_commands = forward_jump
            track_hop_distance = isinstance(forward_jump, LongStrideForwardJumpCommands)
            track_support = isinstance(forward_jump, SupportedTakeoffCommands)
        elif actuation in height_controllers:
            controller = height_controllers[actuation]
            adaptive_jump = event_jump = jump_commands = controller(
                worlds, height_range=height_range, fixed_height=fixed_height)
        elif actuation == JUMP_ACTUATION:
            event_jump = jump_commands = CompliantJumpCommands(worlds)
        elif actuation == CONTACT_JUMP_ACTUATION:
            event_jump = jump_commands = ContactJumpCommands(worlds)
        elif actuation == TIMED_JUMP_ACTUATION:
            jump_commands = JumpCommands(worlds)
        elif actuation is not None:
            parser.error(f'未知跳跃动作配置：{actuation}')
        counter_options = {'max_interval': adaptive_jump.max_landing_interval()} if adaptive_jump else {}
        jump_counter = JumpCounter(args.num_envs, **counter_options)

    assist = worlds.exoskeleton
    if assist is not None:
        assist.bind(forward_jump, policy_io=exo_policy_io)
        if vertical_support or reward_revision == LOAD_SHARING_REWARD['revision']:
            assist.enable_load_sharing(vertical_support=vertical_support, posture_balance=posture_balance,
                                      stance_transition=stance_transition, roll_balance=roll_balance)
        mode = exo_policy_io or 'scheduled force'
        print(f'[CPU] Exoskeleton {mode}: up to {assist_force:g} N per strut', flush=True)

    def jump_phase():
        return event_jump.phase() if event_jump is not None else sim.data.time

    def policy_observation():
        state = device.get_measurements(sim)['wangyu/state']
        if adaptive_jump is not None:
            state = adaptive_jump.observation(state)
            return assist.observation(state) if assist is not None else state
        return jump_observation(state, jump_phase()) if task != 'stand' else state

    obs = TensorDict({'policy': policy_observation()}, batch_size=[args.num_envs])
    num_actions = adaptive_jump.NUM_ACTIONS if adaptive_jump is not None else 23
    if exo_policy_io is not None:
        num_actions += len(assist.actuators)
    actor = MLPModel(obs, {'actor': ['policy']}, 'actor', num_actions,
        hidden_dims=(128, 128), activation='elu', obs_normalization=True,
        distribution_cfg={'class_name': 'rsl_rl.modules.GaussianDistribution', 'init_std': .25, 'std_type': 'scalar'})
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    del checkpoint
    actor.eval()
    print(f'[CPU] MuJoCo {mujoco.__version__}; physics=CPU; policy=CPU; terrain={args.terrain_idx}; worlds={args.num_envs}; 120Hz physics / 60Hz control', flush=True)
    print(f'[CPU] Checkpoint: {args.checkpoint}', flush=True)
    print(f'[CPU] Task: {task}; geometry={geometry}', flush=True)
    if fixed_height is not None:
        print(f'[CPU] Jump height target: {fixed_height:g} m', flush=True)
    if args.jump_speed is not None:
        print(f'[CPU] Forward jump speed target: {args.jump_speed:g} m/s', flush=True)
    state = {'paused': False}
    reset_requested = threading.Event()
    terrain_requested = []
    def key(keycode):
        if keycode == 32:
            state['paused'] = not state['paused']
        elif keycode in (ord('R'), ord('r')):
            reset_requested.set()
        elif keycode in (ord('['), ord(']')):
            terrain_requested.append(-1 if keycode == ord('[') else 1)
            reset_requested.set()
    if args.headless:
        context = nullcontext(None)
    else:
        from cpu_viewer import launch
        context = launch(physics_spec, model, data, args.viewer, args.viz,
                         key, state, reset_requested, simulator=sim)
    steps = failures = timeouts = 0
    episode_steps = torch.zeros(args.num_envs, dtype=torch.long)
    if forward_jump is not None:
        forward_distance = torch.zeros(args.num_envs)
        forward_air_distance = torch.zeros_like(forward_distance)
        forward_air_steps = torch.zeros_like(episode_steps)
    if track_hop_distance:
        completed_hop_distance = torch.zeros_like(forward_distance)
        longest_forward_hop = torch.zeros_like(forward_distance)
    if track_support:
        completed_support_time = torch.zeros_like(forward_distance)
        completed_push_speed_gain = torch.zeros_like(forward_distance)
        completed_peak_load = torch.zeros_like(forward_distance)
    start = time.monotonic()
    max_action = 0.
    with context as viewer, torch.inference_mode():
        if viewer:
            viewer.cam.lookat[:] = [0,0,1.] if args.num_envs > 1 else data.qpos[:3]
            viewer.cam.distance = 18 if args.num_envs > 1 else 5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = args.viz == 'skin'
            print('[CPU] Window ready. /=follow on/off; comma/period=previous/next person; A=overview; Esc=free camera; Space=pause; R=reset; [ / ]=terrain.', flush=True)
        while (viewer is None or viewer.is_running()) and (args.seconds == 0 or steps < round(args.seconds * 60)):
            tick = time.monotonic()
            lock = viewer.lock() if viewer else nullcontext()
            with lock:
                if reset_requested.is_set():
                    while terrain_requested:
                        args.terrain_idx = (args.terrain_idx - 1 + terrain_requested.pop(0)) % 64 + 1
                        device.select_terrain(args.terrain_idx)
                    worlds.reset(); state.pop('mouse_forces',None); episode_steps.zero_(); reset_requested.clear()
                    if jump_counter is not None:
                        jump_counter.reset()
                    if event_jump is not None:
                        event_jump.reset()
                    print(f'[CPU] Reset terrain={args.terrain_idx}', flush=True)
                if not state['paused']:
                    obs['policy'] = policy_observation()
                    action = actor(obs)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('Non-finite policy action')
                    max_action = max(max_action, float(action.abs().max()))
                    if assist is not None:
                        action = assist.policy_actions(action)
                    elif adaptive_jump is not None:
                        action = adaptive_jump(action)
                    elif jump_commands is not None:
                        action = jump_commands(action, jump_phase())
                    if forward_jump is not None:
                        before_xy = sim.data.qpos[:, :2].clone()
                    worlds.mouse_forces[:] = state.get('mouse_forces',0.)
                    measured = worlds.step(action)
                    if event_jump is not None:
                        event_jump.advance(measured)
                    steps += 1; episode_steps += 1
                    failed = measured['wangyu/failed']
                    timeout = episode_steps >= args.episode_seconds * 60
                    if forward_jump is not None:
                        timeout |= forward_jump.course_finished()
                        distance = ((sim.data.qpos[:, :2] - before_xy) * forward_jump.heading).sum(-1)
                        forward_distance += distance
                        airborne = foot_airborne(measured) & (forward_jump.stage == forward_jump.FLIGHT)
                        forward_air_distance += distance * airborne
                        forward_air_steps += airborne
                        if track_hop_distance:
                            completed_hop_distance += forward_jump.hop_distance
                            longest_forward_hop = longest_forward_hop.maximum(forward_jump.hop_distance)
                        if track_support:
                            completed_support_time += forward_jump.completed * forward_jump.takeoff_support_time
                            completed_push_speed_gain += forward_jump.completed * forward_jump.takeoff_push_speed_gain
                            completed_peak_load += forward_jump.completed * forward_jump.takeoff_peak_load_ratio
                    done = failed | timeout
                    if adaptive_jump is not None:
                        jump_counter.record_completed(adaptive_jump.completed)
                    elif jump_counter is not None:
                        jump_counter.update(jump_height(worlds), foot_airborne(measured), failed)
                    if bool(done.any()):
                        failures += int(failed.sum()); timeouts += int((timeout & ~failed).sum())
                        worlds.reset(done)
                        episode_steps[done] = 0
                        if jump_counter is not None:
                            jump_counter.reset(done)
                        if event_jump is not None:
                            event_jump.reset(done)
                    if steps % 60 == 0:
                        print(f'[CPU] steps={steps} simulated={steps / 60:.1f}s failures={failures} timeouts={timeouts}', flush=True)
            if viewer:
                viewer.sync()
                time.sleep(max(0, 1 / (60 * state.get('speed', 1.)) - (time.monotonic() - tick)))
    report = dict(backend='native_mujoco_cpu', policy_device='cpu', mujoco=mujoco.__version__,
        task=task, geometry=geometry, viewer=args.viewer, viz=args.viz, num_envs=args.num_envs,
        transitions=steps * args.num_envs,
        checkpoint=str(args.checkpoint), terrain_idx=args.terrain_idx, steps=steps,
        simulated_seconds=steps / 60, wall_seconds=time.monotonic() - start,
        failure_resets=failures, timeout_resets=timeouts, max_raw_action=max_action,
        finite=True, training=False, exact_replay=False)
    if worlds.exoskeleton is not None:
        report['exoskeleton'] = worlds.exoskeleton.report()
    if jump_counter is not None:
        report.update(jump_actuation=actuation, jumps_per_env=jump_counter.count.tolist(),
                      best_jump_streak_per_env=jump_counter.best_streak.tolist())
    if height_range is not None:
        report['jump_height_range_m'] = height_range
        report['jump_height_target_m'] = fixed_height
    if forward_jump is not None:
        report.update(jump_speed_range_mps=speed_range, jump_speed_target_mps=args.jump_speed,
                      forward_distance_per_env_m=forward_distance.tolist(),
                      airborne_forward_distance_per_env_m=forward_air_distance.tolist(),
                      mean_forward_speed_per_env_mps=(forward_distance / max(steps / 60, 1e-6)).tolist(),
                      mean_airborne_forward_speed_per_env_mps=(
                          forward_air_distance / (forward_air_steps.clamp_min(1) / 60)).tolist())
    if track_hop_distance:
        report.update(mean_forward_hop_distance_per_env_m=(
                          completed_hop_distance / jump_counter.count.clamp_min(1)).tolist(),
                      longest_forward_hop_per_env_m=longest_forward_hop.tolist())
    if track_support:
        hops = jump_counter.count.clamp_min(1)
        report.update(mean_loaded_working_support_per_env_s=(completed_support_time / hops).tolist(),
                      mean_supported_push_speed_gain_per_env_mps=(completed_push_speed_gain / hops).tolist(),
                      mean_sampled_peak_sole_load_per_env_lunar_bodyweights=(completed_peak_load / hops).tolist())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
