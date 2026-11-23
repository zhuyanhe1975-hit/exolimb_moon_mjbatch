"""Train or replay the independent Wangyu lunar fall recovery task."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def latest_replant_checkpoint(root):
    # Run names are sortable creation timestamps; an empty newer run is skipped.
    candidates = (p for p in root.glob('*/model_*.pt')
                  if re.fullmatch(r'model_[0-9]+\.pt', p.name) and p.is_file())
    return max(candidates, key=lambda p: (p.parent.name, int(p.stem[6:])), default=None)


def runner_config():
    model = dict(class_name='rsl_rl.models.MLPModel', hidden_dims=[128,128],
                 activation='elu', obs_normalization=True)
    return dict(num_steps_per_env=32, save_interval=25, logger='tensorboard',
                obs_groups={'actor':['policy'],'critic':['policy']},
                actor=dict(model,distribution_cfg=dict(class_name='rsl_rl.modules.GaussianDistribution',
                                                       init_std=.3,std_type='scalar')),
                critic=model, algorithm=dict(class_name='rsl_rl.algorithms.PPO',
                    num_learning_epochs=5,num_mini_batches=4,learning_rate=.0003,schedule='adaptive',
                    gamma=.998,lam=.95,entropy_coef=.001,desired_kl=.01,clip_param=.2,max_grad_norm=1.,
                    value_loss_coef=1.,use_clipped_value_loss=True,normalize_advantage_per_mini_batch=False))


def draw_replant_mounts(viewer, world):
    """Show pack brackets and the telescoping middle sleeve in the native viewer."""
    import mujoco
    import numpy as np
    scene = viewer.user_scn
    scene.ngeom = 0
    for side, sign in (('left', 1), ('right', -1)):
        mount = world.d.body(f'{side}_rod')
        tip = world.d.geom(f'{side}_tip').xpos
        axis = mount.xmat.reshape(3, 3)[:, 2]
        torso_y = world.d.xmat[world.torso].reshape(3, 3)[:, 1]
        for start, end, radius in (
            (mount.xpos - sign*.12*torso_y, mount.xpos, .025),
            (mount.xpos - .50*axis, tip + .03*axis, .018),
        ):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([.40, .43, .46, 1.], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
            scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    modes.add_parser('check', help='compile model without resetting or stepping physics')
    train = modes.add_parser('train')
    train.add_argument('--num-envs',type=int,default=8192,help='parallel CUDA physics worlds (default 8192)')
    train.add_argument('--iterations',type=int,default=2000)
    train.add_argument('--seed',type=int,default=42)
    play = modes.add_parser('play')
    play.add_argument('checkpoint',type=Path,nargs='?',help='defaults to the newest saved replant checkpoint')
    play.add_argument('--episodes',type=int,default=1)
    play.add_argument('--headless',action='store_true')
    args = parser.parse_args()
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from wangyu_replant import TASK, DT, ReplantEnv, build_model
    model = build_model(training_sensors=args.mode in ('check','train'))
    if args.mode=='check':
        print(f'{TASK}: {model.nu} actuators, gravity={model.opt.gravity}, dt={model.opt.timestep}')
        return
    torch.set_num_threads(1)
    if args.mode=='train':
        if min(args.num_envs,args.iterations)<1:
            parser.error('num-envs and iterations must be positive')
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            parser.error('replant training requires CUDA PyTorch and MuJoCo Warp')
        from warp_replant import WarpReplantEnv
        env = WarpReplantEnv(model,args.num_envs)
        folder = ROOT/'logs'/'wangyu_replant'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        folder.mkdir(parents=True,exist_ok=False)
        cfg = runner_config()
        (folder/'config.json').write_text(json.dumps(dict(task=TASK,env=env.cfg,runner=cfg,seed=args.seed,
            num_envs=args.num_envs,iterations=args.iterations),indent=2)+'\n')

        class Runner(OnPolicyRunner):
            def save(self,path,infos=None):
                # Publish only complete checkpoints to concurrent replay processes.
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(dir=Path(path).parent,suffix='.tmp',delete=False) as tmp:
                    temporary = Path(tmp.name)
                try:
                    state = self.alg.save()
                    state.update(iter=self.current_learning_iteration,infos=dict(infos or {},task=TASK))
                    torch.save(state,temporary)
                    os.replace(temporary,path)
                finally:
                    temporary.unlink(missing_ok=True)
                self.logger.save_model(path,self.current_learning_iteration)

        print(f'{TASK}: {args.num_envs} Warp worlds, rollout and PPO on {env.device}, logs={folder}',flush=True)
        Runner(env,cfg,log_dir=str(folder),device=str(env.device)).learn(args.iterations)
        return
    if args.episodes<1:
        parser.error('episodes must be positive')
    if args.checkpoint is None:
        args.checkpoint = latest_replant_checkpoint(ROOT/'logs'/'wangyu_replant')
        if args.checkpoint is None:
            parser.error('logs/wangyu_replant 下尚无已保存的 model_<迭代>.pt')
    print(f'[Replant] 加载检查点 {args.checkpoint}',flush=True)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (saved.get('infos') or {}).get('task') != TASK:
        parser.error(f'checkpoint must belong to {TASK}')
    env = ReplantEnv(model,1)
    runner = OnPolicyRunner(env,runner_config(),device='cpu')
    actor = runner.get_inference_policy('cpu')
    actor.load_state_dict(saved['actor_state_dict'],strict=True)
    world = env.worlds[0]
    context = nullcontext()
    if not args.headless:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(model,world.d)
    with context as viewer, torch.inference_mode():
        if viewer:
            viewer.cam.distance=3.5
            viewer.cam.lookat[:]=[0,0,.65]
        for episode in range(args.episodes):
            world.reset()
            while world.steps < env.max_episode_length and (viewer is None or viewer.is_running()):
                start = time.monotonic()
                action = actor(env.get_observations())[0].numpy()
                _,success,failed = world.step(action)
                if viewer:
                    with viewer.lock():
                        draw_replant_mounts(viewer, world)
                    viewer.sync()
                    time.sleep(max(0,DT-(time.monotonic()-start)))
                if success or failed:
                    break
            if viewer and not viewer.is_running():
                break
            print(dict(episode=episode+1,phase=world.phase,seconds=world.steps*DT,
                       success=success,failed=failed),flush=True)


if __name__=='__main__':
    main()
