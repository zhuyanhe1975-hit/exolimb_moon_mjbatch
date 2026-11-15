"""Bounded, explicitly requested policy replay through the real Playground env."""
import dataclasses,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import moon_task
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
moon_task.register();torch.manual_seed(42)
entry=env_registry.get(moon_task.TASK_ID)
print('[Replay check] building 1 CUDA world, pushes off',flush=True)
env=entry.env_builder(nworld=1,device='cuda:0',pushes=False)
wrapped=RslRlVecEnvWrapper(env,cfg=entry.runner_cfg.vecenv_wrapper_cfg)
runner=OnPolicyRunner(wrapped,dataclasses.asdict(entry.runner_cfg),device='cuda:0')
p=ROOT/'logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt'
runner.load(str(p),load_cfg={'actor':True,'critic':True},map_location='cuda:0')
policy=runner.get_inference_policy(device='cuda:0');obs=wrapped.get_observations()
failures=0;maximum=0.;start=time.monotonic()
with torch.inference_mode():
 for i in range(300):
  action=policy(obs);maximum=max(maximum,float(action.abs().max()))
  obs,reward,done,extra=wrapped.step(action)
  assert torch.isfinite(action).all() and torch.isfinite(reward).all()
  failures+=int(done.sum())
  if i%60==0:print(f'[Replay check] step={i} reward={float(reward[0]):.6f} resets={failures}',flush=True)
report={'checkpoint':str(p),'worlds':1,'seconds':5,'steps':300,'reset_count':failures,'max_raw_action':maximum,'wall_seconds':time.monotonic()-start,'pushes':False,'terrain_transfer':True,'training':False}
(ROOT/'outputs/playground_play/check_model200.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True);env.close()
