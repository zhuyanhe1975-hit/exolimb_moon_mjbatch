"""Bounded deterministic first-episode evaluation; never updates a policy."""
import argparse,dataclasses,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoints',nargs='+',type=Path)
p.add_argument('--worlds',type=int,default=64)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
import torch
from reaf.simulation.torch import flat_scene
from mujoco_playground._src.rl import rng
from rsl_rl.models import MLPModel
from lunar_stand_training import runner_config,training_env
from training_logs import LunarTrainingVecEnv
results=[]
for path in args.checkpoints:
    rng.seed(42);cfg=runner_config();ecfg=training_env()
    ecfg.sim.nworld=args.worlds;ecfg.sim.xpu_device='cuda:0'
    env=flat_scene.build_env(ecfg)
    try:
        wrapped=LunarTrainingVecEnv(env,cfg=cfg.vecenv_wrapper_cfg);obs=wrapped.get_observations()
        params=dataclasses.asdict(cfg.actor);params.pop('class_name')
        actor=MLPModel(obs,cfg.obs_groups,'actor',23,**params).to('cuda:0')
        actor.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=False)['actor_state_dict']);actor.eval()
        active=torch.ones(args.worlds,dtype=torch.bool,device='cuda:0')
        lengths=torch.zeros(args.worlds,device='cuda:0');returns=lengths.clone();timeouts=torch.zeros_like(active)
        with torch.inference_mode():
            for _ in range(600):
                obs,reward,done,extras=wrapped.step(actor(obs))
                lengths+=active;returns+=torch.where(active,reward,0.)
                timeouts|=active & extras.get('time_outs',torch.zeros_like(active))
                active&=~done.bool()
                if not bool(active.any()):break
        result={'checkpoint':str(path),'worlds':args.worlds,'push_force_N':30.,'deterministic':True,
            'first_episode_mean_seconds':float(lengths.mean()/60),
            'first_episode_median_seconds':float(lengths.median()/60),
            'ten_second_success_fraction':float((lengths>=600).float().mean()),
            'failure_fraction':float(((lengths<600)&~timeouts).float().mean()),
            'first_episode_mean_return':float(returns.mean())}
        results.append(result);print(json.dumps(result),flush=True)
    finally:env.close()
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(results,indent=2)+'\n')
