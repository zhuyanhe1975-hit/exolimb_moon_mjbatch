"""Native MuJoCo CPU policy replay, independent of CUDA/Warp/REAF."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(
        prog='./run.sh', allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='月面策略 CPU 回放：classic/Studio 原生窗口、物理代理或航天服蒙皮、共享月面多人，无随机扰动。',
        epilog="""示例：
  ./run.sh
  ./run.sh --viewer studio --viz skin --num-envs 8
  ./run.sh --terrain-idx 24
  ./run.sh --checkpoint /absolute/path/model_200.pt
  ./run.sh --headless --seconds 5 --report outputs/cpu_check.json

窗口快捷键：/切换跟踪，逗号上一人，句号下一人，A总览，Esc退出跟踪。
空格暂停/继续，R复位，[ / ] 切换地形，关闭窗口退出。
施力：双击选中身体，Control+右键拖动施加力，Control+左键拖动施加转矩。
物理120Hz / 策略60Hz；窗口按1×实时播放；无窗口模式全速运行。
支持 classic/studio 与 physx/skin，自身动力学和策略均在CPU上；各人独立仿真，不相互碰撞。
环境变量：CPU_PYTHON 指定 Python，默认项目内 .venv-cpu/bin/python。
""")
    parser.add_argument('--headless', action='store_true', help='不打开窗口；必须设置 --seconds')
    parser.add_argument('--seconds', type=float, default=0, help='运行的仿真秒数；默认0表示持续运行到窗口关闭')
    parser.add_argument('--terrain-idx', '--terrain_idx', type=int, choices=range(1, 65), metavar='1–64', default=12, help='16米地形编号，默认12')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认42')
    parser.add_argument('--checkpoint', '--resume-path', '--resume_path', type=Path,
        default=ROOT / 'logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt', help='策略检查点；默认导入的 model_200.pt')
    parser.add_argument('--terrain-transfer', '--terrain_transfer', action='store_true', help='显式允许检查点向当前地形迁移')
    parser.add_argument('--report', type=Path, help='退出时保存 JSON 验证报告')
    compatible = parser.add_argument_group('计算与显示')
    compatible.add_argument('--device', choices=('cpu',), default='cpu', help='计算设备，仅支持cpu')
    compatible.add_argument('--viewer', choices=('classic','studio'), default='classic', help='窗口类型，默认classic')
    compatible.add_argument('--viz', choices=('physx','skin'), default='physx', help='物理代理几何或航天服蒙皮，默认physx')
    compatible.add_argument('--num-envs', '--num_envs', '--nworld', type=int, default=1, help='独立环境/人数，默认1；多人共享同一块地形')
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error('--num-envs 必须是正整数')
    import math
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds 必须是有限的非负数')
    if args.headless and args.seconds <= 0:
        parser.error('--headless 必须配合 --seconds > 0')
    if args.seconds > 0 and round(args.seconds * 60) < 1:
        parser.error('--seconds 至少需要一个控制步（约0.0167秒）')
    if not args.checkpoint.is_file():
        parser.error(f'检查点不存在：{args.checkpoint}')
    import mujoco
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from cpu_simulation import CpuWorlds
    from moon_scene import scene_spec
    from provenance import verify_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    verify_checkpoint(args.checkpoint, allow_transfer=args.terrain_transfer)
    startup_started=time.monotonic()
    print(f'[Startup] Preparing {args.num_envs} independent CPU environments...',flush=True)
    physics_spec = scene_spec()
    worlds = CpuWorlds(physics_spec, args.num_envs, args.terrain_idx)
    model, data, sim, device = worlds.model, worlds.data[0], worlds.sim, worlds.device
    print(f'[Startup] CPU environments ready in {time.monotonic()-startup_started:.2f}s',flush=True)
    obs = TensorDict({'policy': device.get_measurements(sim)['wangyu/state']}, batch_size=[args.num_envs])
    actor = MLPModel(obs, {'actor': ['policy']}, 'actor', 23,
        hidden_dims=(128, 128), activation='elu', obs_normalization=True,
        distribution_cfg={'class_name': 'rsl_rl.modules.GaussianDistribution', 'init_std': .25, 'std_type': 'scalar'})
    actor.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['actor_state_dict'], strict=True)
    actor.eval()
    print(f'[CPU] MuJoCo {mujoco.__version__}; physics=CPU; policy=CPU; terrain={args.terrain_idx}; worlds={args.num_envs}; 120Hz physics / 60Hz control', flush=True)
    print('[CPU] Imported policy transferred to current terrain; native CPU trajectory is not claimed equivalent to Warp.', flush=True)
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
                    print(f'[CPU] Reset terrain={args.terrain_idx}', flush=True)
                if not state['paused']:
                    obs['policy'] = device.get_measurements(sim)['wangyu/state']
                    action = actor(obs)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('Non-finite policy action')
                    max_action = max(max_action, float(action.abs().max()))
                    worlds.mouse_forces[:] = state.get('mouse_forces',0.)
                    measured = worlds.step(action)
                    steps += 1; episode_steps += 1
                    failed = measured['wangyu/failed']
                    timeout = episode_steps >= 600
                    done = failed | timeout
                    if bool(done.any()):
                        failures += int(failed.sum()); timeouts += int((timeout & ~failed).sum())
                        worlds.reset(done)
                        episode_steps[done] = 0
                    if steps % 60 == 0:
                        print(f'[CPU] steps={steps} simulated={steps / 60:.1f}s failures={failures} timeouts={timeouts}', flush=True)
            if viewer:
                viewer.sync()
                time.sleep(max(0, 1 / (60 * state.get('speed', 1.)) - (time.monotonic() - tick)))
    report = dict(backend='native_mujoco_cpu', policy_device='cpu', mujoco=mujoco.__version__,
        viewer=args.viewer, viz=args.viz, num_envs=args.num_envs,
        transitions=steps * args.num_envs,
        checkpoint=str(args.checkpoint), terrain_idx=args.terrain_idx, steps=steps,
        simulated_seconds=steps / 60, wall_seconds=time.monotonic() - start,
        failure_resets=failures, timeout_resets=timeouts, max_raw_action=max_action,
        finite=True, training=False, exact_replay=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
