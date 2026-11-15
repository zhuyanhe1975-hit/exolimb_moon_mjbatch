"""Bounded foreground sampling benchmark. No PPO updates/checkpoints/viewer."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
import cProfile
import pstats
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--worlds',type=int,default=8192)
parser.add_argument('--steps',type=int,default=128)
parser.add_argument('--warmup',type=int,default=32)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--profile',action='store_true')
parser.add_argument('--check-graph',action='store_true')
args=parser.parse_args()
for key in ('MOON_MEASUREMENT_GRAPH','MOON_SPARSE_RESET','MOON_BATCHED_LOGS','MOON_BATCHED_TERRAIN'):
    os.environ.setdefault(key,'1')
if not 1<=args.steps<=512 or not 1<=args.warmup<=128:parser.error('bounded steps/warmup required')
import torch
from reaf.simulation.torch import flat_scene
from mujoco_playground._src.rl import rng
from rsl_rl.models import MLPModel
from lunar_stand_training import training_env,runner_config
from training_logs import LunarTrainingVecEnv
rng.seed(42);cfg=runner_config();ecfg=training_env();ecfg.sim.nworld=args.worlds;ecfg.sim.xpu_device='cuda:0'
env=None
try:
    env=flat_scene.build_env(ecfg)
    wrapper=LunarTrainingVecEnv(env,cfg=cfg.vecenv_wrapper_cfg,
        batched_logs=os.environ.get('MOON_BATCHED_LOGS','1')=='1')
    obs=wrapper.get_observations()
    params=dataclasses.asdict(cfg.actor);params.pop('class_name')
    actor=MLPModel(obs,cfg.obs_groups,'actor',23,**params).to('cuda:0')
    checkpoint=ROOT/'logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt'
    actor.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=False)['actor_state_dict'])
    actor.eval()
    wrapper.episode_length_buf=torch.randint(600,(args.worlds,),device='cuda:0')
    dev=next(d for d in env.device_layer._devices if hasattr(d,'_measurement_values'));sim=env.device_layer.sim
    checked=0;ended=torch.zeros((),device='cuda:0',dtype=torch.long)
    profiler=cProfile.Profile() if args.profile else None
    with torch.inference_mode():
        for i in range(args.warmup+args.steps):
            if i==args.warmup:
                torch.cuda.synchronize();start=time.perf_counter()
                if profiler:profiler.enable()
            obs,reward,done,extras=wrapper.step(actor(obs))
            if i>=args.warmup:ended+=done.sum()
            if args.check_graph and i in (0,15,31):
                captured=dev.get_measurements(sim)
                saved={k:v.clone() for k,v in captured.items()}
                eager=dev._measurement_values(sim)
                terrain_mode=os.environ.get('MOON_BATCHED_TERRAIN')
                os.environ['MOON_BATCHED_TERRAIN']='0'
                legacy_terrain=dev._measurement_values(sim)
                if terrain_mode is None:os.environ.pop('MOON_BATCHED_TERRAIN')
                else:os.environ['MOON_BATCHED_TERRAIN']=terrain_mode
                torch.testing.assert_close(eager,legacy_terrain,rtol=0,atol=0)
                torch.testing.assert_close(captured['wangyu/state'],eager[:,:146],rtol=0,atol=0)
                torch.testing.assert_close(captured['wangyu/rewards'],eager[:,146:155],rtol=0,atol=0)
                assert torch.equal(captured['wangyu/failed'],eager[:,155].bool())
                before_rng=torch.cuda.get_rng_state();dev.get_measurements(sim)
                assert torch.equal(before_rng,torch.cuda.get_rng_state())
                for k in saved:assert torch.equal(saved[k],captured[k])
                if i==0:
                    # Prove the host-visible guard remains active after capture.
                    from unittest.mock import patch
                    velocity=sim.data.qvel[:].clone()
                    sim.data.qvel[0,0]=float('nan')
                    try:
                        with patch('moon_device.torch.save') as save:
                            try:dev.get_measurements(sim)
                            except FloatingPointError:pass
                            else:raise AssertionError('Captured numerical guard missed NaN')
                            assert save.called
                    finally:sim.data.qvel[:]=velocity
                checked+=1
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if profiler:profiler.disable()
    result={'worlds':args.worlds,'steps':args.steps,'warmup':args.warmup,'seconds':elapsed,
        'environment_steps_per_second':args.worlds*args.steps/elapsed,'completed_episodes':int(ended),
        'graph_checks':checked,'profiled':args.profile,'policy':str(checkpoint),
        'switches':{k:os.environ.get(k,'1') for k in ('MOON_MEASUREMENT_GRAPH','MOON_BATCHED_LOGS','MOON_SPARSE_RESET','MOON_BATCHED_TERRAIN')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if profiler:
        with args.output.with_suffix('.profile.txt').open('w') as f:pstats.Stats(profiler,stream=f).sort_stats('cumtime').print_stats(40)
finally:
    if env is not None:env.close()
