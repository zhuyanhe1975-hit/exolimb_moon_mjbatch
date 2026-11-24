"""Train, preview reference, or replay the Wangyu lunar fall-recovery task with detached hut_hinge_aft_v7 struts."""
import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

BEST_REPLANT_CHECKPOINT = (
    ROOT / 'bests' / 'replant_wangyu_hut_hinge_hoof_v12_2026-11-24' / 'model_999.pt'
)


def latest_replant_checkpoint(root, *, task):
    """Return the newest saved checkpoint belonging to the requested replant task."""
    candidates = []
    for run in root.iterdir() if root.is_dir() else ():
        if not run.is_dir():
            continue
        cfg_file = run / 'config.json'
        if not cfg_file.is_file():
            continue
        try:
            if json.loads(cfg_file.read_text()).get('task') != task:
                continue
        except (OSError, json.JSONDecodeError):
            continue
        for p in run.glob('model_*.pt'):
            if re.fullmatch(r'model_[0-9]+\.pt', p.name) and p.is_file():
                candidates.append(p)
    return max(candidates, key=lambda p: (p.parent.name, int(p.stem[6:])), default=None)


def runner_config():
    model = dict(class_name='rsl_rl.models.MLPModel', hidden_dims=[512, 256],
                 activation='elu', obs_normalization=True)
    # Canonical DeepMimic / BeyondMimic residual-tracking PPO configuration:
    # [512, 256] capacity for 527-frame whole-body tracking, calm init_std=0.12,
    # steady adaptive LR starting at 2e-4 with tight desired_kl=0.008 and gamma=0.99.
    return dict(
        num_steps_per_env=32, save_interval=25, logger='tensorboard',
        obs_groups={'actor': ['policy'], 'critic': ['policy']},
        actor=dict(model, distribution_cfg=dict(class_name='rsl_rl.modules.GaussianDistribution',
                                                init_std=.12, std_type='scalar')),
        critic=model,
        algorithm=dict(
            class_name='rsl_rl.algorithms.PPO',
            num_learning_epochs=5, num_mini_batches=4, learning_rate=.0002, schedule='adaptive',
            gamma=.99, lam=.95, entropy_coef=.0005, desired_kl=.008, clip_param=.2, max_grad_norm=1.,
            value_loss_coef=1., use_clipped_value_loss=True, normalize_advantage_per_mini_batch=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('check', help='compile physics/display specs and verify SHA-locked reference without stepping')

    preview = modes.add_parser('preview', help='kinematically preview the SHA-verified Wangyu get-up reference')
    preview.add_argument('--viewer', choices=('studio', 'classic'), default='studio',
                         help='native viewer window (default: studio)')
    preview.add_argument('--viz', choices=('skin', 'physx'), default='skin',
                         help='astronaut skin PBR or proxy physics geometry (default: skin)')
    preview.add_argument('--speed', type=float, default=1.0, help='playback speed multiplier (default: 1.0)')
    preview.add_argument('--loops', type=int, default=0, help='number of loops (0 = infinite in GUI, 1 in headless)')
    preview.add_argument('--no-deploy', action='store_true',
                         help='skip the prone magnetic-undock & forward deploy prelude')
    preview.add_argument('--headless', action='store_true')

    train = modes.add_parser('train')
    train.add_argument('--num-envs', type=int, default=8192, help='parallel CUDA physics worlds (default 8192)')
    train.add_argument('--iterations', type=int, default=2000)
    train.add_argument('--rsi-prob', type=float, default=0.5,
                       help='fraction of training resets initialized along intermediate reference frames (default: 0.5)')
    train.add_argument('--init-from', type=str, default=None,
                       help='optional checkpoint path (or "best") to warm-start actor weights & obs normalizer')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--log-dir', type=Path, help='optional output directory under logs/wangyu_replant')

    play = modes.add_parser('play')
    play.add_argument('checkpoint', type=Path, nargs='?',
                      help='defaults to newest saved v12 checkpoint or archived best')
    play.add_argument('--best', action='store_true',
                      help='replay the archived best Wangyu-Replant-v12 (magnetic hoof) checkpoint under bests/')
    play.add_argument('--episodes', type=int, default=0,
                      help='number of episodes (0 = keep GUI window open for R-reset; 1 in headless)')
    play.add_argument('--viewer', choices=('studio', 'classic'), default='studio',
                      help='native viewer window (default: studio)')
    play.add_argument('--viz', choices=('skin', 'physx'), default='skin',
                      help='astronaut skin PBR or proxy physics geometry (default: skin)')
    play.add_argument('--no-deploy', action='store_true',
                      help='skip the prone magnetic-undock & forward deploy prelude')
    play.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    import mujoco
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from wangyu_replant import (DT, GEOMETRY, MECHANISM, TASK, ReplantEnv,
                                build_spec, load_replant_reference)

    spec = build_spec(training_sensors=args.mode in ('check', 'train'))
    model = spec.compile()
    ref = load_replant_reference()

    if args.mode == 'check':
        print(f'{TASK}: {model.nu} actuators ({MECHANISM}, {GEOMETRY}), '
              f'neq={model.neq}, gravity={model.opt.gravity.tolist()}, dt={model.opt.timestep}, '
              f'ref={ref["revision"]} ({ref["n_frames"]} frames, sha256={ref["sha256"][:12]})')
        return

    if args.mode == 'preview':
        if args.speed <= 0:
            parser.error('speed must be positive')
        from replant_deploy import DEPLOY_STEPS, ProneDeployPrelude
        data = mujoco.MjData(model)
        n_frames = ref['n_frames']
        prelude = None if args.no_deploy else ProneDeployPrelude(model, data, ref)
        prelude_frames = DEPLOY_STEPS if prelude is not None else 0
        total_frames = prelude_frames + n_frames
        max_loops = args.loops if args.loops > 0 else (1 if args.headless else 0)
        state = {
            'paused': False,
            'speed': float(args.speed),
            'sync_geom_rgba': ('robot/exo_left_mag_led', 'robot/exo_right_mag_led'),
        }
        reset_requested = threading.Event()

        def on_key(keycode):
            if keycode == 32:
                state['paused'] = not state['paused']
            elif keycode in (ord('R'), ord('r')):
                reset_requested.set()

        if prelude is not None:
            prelude.kinematic_frame(0)
        else:
            data.qpos[:] = ref['qpos'][0]
            data.qvel[:] = 0.
            data.ctrl[:] = ref['ctrl'][0]
            mujoco.mj_forward(model, data)

        if args.headless:
            context = nullcontext(None)
        else:
            from cpu_viewer import launch
            context = launch(spec, model, data, args.viewer, args.viz,
                             on_key, state, reset_requested)

        with context as viewer:
            if viewer:
                viewer.cam.distance = 3.2
                viewer.cam.lookat[:] = [0, 0, .55]
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -18
                print(f'[Replant Preview] {ref["revision"]} (deploy={prelude_frames} + ref={n_frames} frames, '
                      f'{total_frames * DT:.2f}s, sha256={ref["sha256"][:12]}). '
                      'Space=pause; R=restart; /=follow; A=overview; Esc=free camera.', flush=True)
            k = 0
            completed = 0
            while viewer is None or viewer.is_running():
                start = time.monotonic()
                lock = viewer.lock() if viewer else nullcontext()
                with lock:
                    if reset_requested.is_set():
                        k = 0
                        reset_requested.clear()
                    if prelude is not None and k < prelude_frames:
                        prelude.kinematic_frame(k)
                    else:
                        rk = k - prelude_frames
                        data.qpos[:] = ref['qpos'][rk]
                        data.qvel[:] = ref['qvel'][rk]
                        data.ctrl[:] = ref['ctrl'][rk]
                        data.time = float(prelude_frames * DT + ref['times'][rk])
                        mujoco.mj_forward(model, data)
                    if not state['paused']:
                        k += 1
                        if k >= total_frames:
                            completed += 1
                            if max_loops > 0 and completed >= max_loops:
                                break
                            k = 0
                if viewer:
                    viewer.sync()
                    time.sleep(max(0., DT / max(state.get('speed', 1.), 1e-3) - (time.monotonic() - start)))
        print(json.dumps(dict(
            mode='preview', revision=ref['revision'], sha256=ref['sha256'][:12],
            deploy_frames=prelude_frames, frames=n_frames, loops_completed=completed,
            duration_s=round(float(total_frames * DT), 3),
        )), flush=True)
        return

    torch.set_num_threads(1)
    if args.mode == 'train':
        if min(args.num_envs, args.iterations) < 1:
            parser.error('num-envs and iterations must be positive')
        if not (0. <= args.rsi_prob <= 1.):
            parser.error('rsi-prob must be within [0, 1]')
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            parser.error('replant training requires CUDA PyTorch and MuJoCo Warp')
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from warp_replant import WarpReplantEnv
        env = WarpReplantEnv(model, args.num_envs, reference=ref, rsi_prob=args.rsi_prob)
        folder = (args.log_dir or ROOT / 'logs' / 'wangyu_replant' /
                  datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        cfg = runner_config()
        init_ckpt = None
        if args.init_from is not None:
            init_ckpt = BEST_REPLANT_CHECKPOINT if args.init_from == 'best' else Path(args.init_from)
            if not init_ckpt.is_file():
                parser.error(f'--init-from checkpoint not found: {init_ckpt}')
            cfg['actor']['distribution_cfg']['init_std'] = 0.08
            cfg['algorithm']['learning_rate'] = 0.0001
            cfg['algorithm']['desired_kl'] = 0.012
        (folder / 'config.json').write_text(json.dumps(dict(
            task=TASK, geometry=GEOMETRY, exo_mechanism=MECHANISM,
            reference_revision=ref['revision'], reference_sha256=ref['sha256'],
            init_from=str(init_ckpt) if init_ckpt is not None else None,
            rsi_prob=args.rsi_prob, env=env.cfg, runner=cfg, seed=args.seed,
            num_envs=args.num_envs, iterations=args.iterations,
        ), indent=2) + '\n')

        class Runner(OnPolicyRunner):
            def save(self, path, infos=None):
                # Publish only complete checkpoints to concurrent replay processes.
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(dir=Path(path).parent, suffix='.tmp', delete=False) as tmp:
                    temporary = Path(tmp.name)
                try:
                    state = self.alg.save()
                    state.update(iter=self.current_learning_iteration,
                                 infos=dict(infos or {}, task=TASK, geometry=GEOMETRY,
                                            exo_mechanism=MECHANISM,
                                            reference_sha256=ref['sha256']))
                    torch.save(state, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                self.logger.save_model(path, self.current_learning_iteration)

        runner = Runner(env, cfg, log_dir=str(folder), device=str(env.device))
        if init_ckpt is not None:
            loaded = torch.load(init_ckpt, map_location=env.device, weights_only=True)
            runner.alg.actor.load_state_dict(loaded['actor_state_dict'], strict=True)
            critic_loaded = False
            if 'critic_state_dict' in loaded:
                try:
                    runner.alg.critic.load_state_dict(loaded['critic_state_dict'], strict=True)
                    critic_loaded = True
                except RuntimeError:
                    critic_loaded = False
            with torch.no_grad():
                dist = getattr(runner.alg.actor, 'distribution', None)
                if dist is not None and hasattr(dist, 'std_param'):
                    dist.std_param.fill_(0.08)
                elif dist is not None and hasattr(dist, 'std'):
                    dist.std.fill_(0.08)
            print(f'[Replant] Warm-started actor{"+critic" if critic_loaded else ""} & obs normalizer '
                  f'from {init_ckpt} (std_param=0.08, lr={cfg["algorithm"]["learning_rate"]}, '
                  f'desired_kl={cfg["algorithm"]["desired_kl"]})', flush=True)
        if hasattr(runner, 'logger') and hasattr(runner.logger, 'rewbuffer'):
            buf_size = max(100, min(4096, args.num_envs // 2))
            runner.logger.rewbuffer = deque(maxlen=buf_size)
            runner.logger.lenbuffer = deque(maxlen=buf_size)
        from rollout_logger import install_rollout_logger
        install_rollout_logger(runner, enabled=True)
        print(f'{TASK}: {args.num_envs} Warp worlds ({MECHANISM}, {GEOMETRY}, rsi={args.rsi_prob:.2f}, '
              f'ref={ref["sha256"][:12]}), rollout and PPO on {env.device}, logs={folder}', flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
        return

    if args.episodes < 0:
        parser.error('episodes must be non-negative')
    max_episodes = args.episodes if args.episodes > 0 else (1 if args.headless else 0)
    if args.best:
        if not BEST_REPLANT_CHECKPOINT.is_file():
            parser.error(f'archived best checkpoint not found: {BEST_REPLANT_CHECKPOINT}')
        args.checkpoint = BEST_REPLANT_CHECKPOINT
    elif args.checkpoint is None:
        args.checkpoint = latest_replant_checkpoint(ROOT / 'logs' / 'wangyu_replant', task=TASK)
        if args.checkpoint is None and BEST_REPLANT_CHECKPOINT.is_file():
            args.checkpoint = BEST_REPLANT_CHECKPOINT
        if args.checkpoint is None:
            parser.error(f'logs/wangyu_replant 下尚无 {TASK} 的已保存 model_<迭代>.pt，请先运行 ./replant.sh train')
    print(f'[Replant] 加载检查点 {args.checkpoint}', flush=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    ckpt_task = (saved.get('infos') or {}).get('task')
    if ckpt_task not in (TASK, 'Wangyu-Replant-v11'):
        parser.error(f'checkpoint must belong to {TASK} (or Wangyu-Replant-v11), got {ckpt_task}')
    if ckpt_task == 'Wangyu-Replant-v11':
        spec = build_spec(training_sensors=False, knee_hard_limit=False)
        model = spec.compile()
    bundled_ref = args.checkpoint.parent / 'reference'
    play_ref = load_replant_reference(bundled_ref) if (bundled_ref / 'motion.npz').is_file() else ref
    env = ReplantEnv(model, 1, reference=play_ref, rsi_prob=0.0)
    runner = OnPolicyRunner(env, runner_config(), device='cpu')
    actor = runner.get_inference_policy('cpu')
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    world = env.worlds[0]
    from replant_deploy import ProneDeployPrelude, handoff_to_recovery
    prelude = None if args.no_deploy else ProneDeployPrelude(model, world.d, world.ref)
    state = {
        'paused': False,
        'sync_geom_rgba': ('robot/exo_left_mag_led', 'robot/exo_right_mag_led'),
    }
    reset_requested = threading.Event()

    def on_key(keycode):
        if keycode == 32:
            state['paused'] = not state['paused']
        elif keycode in (ord('R'), ord('r')):
            reset_requested.set()

    if args.headless:
        context = nullcontext(None)
    else:
        from cpu_viewer import launch
        context = launch(spec, model, world.d, args.viewer, args.viz,
                         on_key, state, reset_requested)

    with context as viewer, torch.inference_mode():
        if viewer:
            viewer.cam.distance = 3.5
            viewer.cam.lookat[:] = [0, 0, .65]
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            print('[Replant] Window ready. Space=pause; R=restart from docked prone; /=follow; A=overview; Esc=free camera.',
                  flush=True)
        episode = 0
        while (max_episodes == 0 or episode < max_episodes) and (viewer is None or viewer.is_running()):
            episode += 1
            if prelude is not None:
                prelude.reset()
                in_prelude = True
                last_stage = prelude.stage
            else:
                world.reset(start_frame=0)
                in_prelude = False
                last_stage = 'rl'
            state.pop('mouse_forces', None)
            reset_requested.clear()
            success = failed = False
            reported = False
            while viewer is None or viewer.is_running():
                start = time.monotonic()
                restarted = False
                lock = viewer.lock() if viewer else nullcontext()
                with lock:
                    if reset_requested.is_set():
                        reset_requested.clear()
                        restarted = True
                    elif not state['paused'] and not failed:
                        if 'mouse_forces' in state:
                            world.d.xfrc_applied[:] = state['mouse_forces'][0]
                        else:
                            world.d.xfrc_applied[:] = 0.
                        if in_prelude:
                            finished_prelude = prelude.step()
                            if prelude.stage != last_stage:
                                last_stage = prelude.stage
                                if viewer is not None:
                                    print(f'[Replant Deploy] stage -> {last_stage}', flush=True)
                            if finished_prelude:
                                handoff_to_recovery(world)
                                in_prelude = False
                                if viewer is not None:
                                    print('[Replant Deploy] 前撑点就位，无缝切入 RL 脱困策略', flush=True)
                        else:
                            action = actor(env.get_observations())[0].numpy()
                            _, step_success, step_failed = world.step(action)
                            success = success or step_success
                            failed = failed or step_failed
                if restarted:
                    break
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, DT / state.get('speed', 1.) - (time.monotonic() - start)))
                done = (not in_prelude) and (success or failed or world.steps >= env.max_episode_length)
                if done and not reported:
                    reported = True
                    deploy_s = round((prelude.step_idx * DT) if prelude is not None else 0.0, 2)
                    print(dict(episode=episode, deploy_s=deploy_s, phase=world.phase,
                               seconds=round(world.steps * DT, 2),
                               success=success, failed=failed), flush=True)
                    if viewer is not None and max_episodes == 0:
                        print('[Replant] 本轮完成，窗口保持打开（按 R 从趴地磁吸态重新开始）', flush=True)
                    else:
                        break
            if viewer and not viewer.is_running():
                break


if __name__ == '__main__':
    main()
