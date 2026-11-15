"""Training contract tests: CPU tensors/native compilation, never sim steps/PPO."""
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import sys
import numpy as np
import pytest
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lunar_stand_training import runner_config,training_env,REWARD_WEIGHTS,signature
from moon_device import MoonDevice,META
from moon_scene import scene_spec


def test_mjlab_ppo_and_environment_contract():
    cfg=runner_config()
    assert (cfg.nworld,cfg.num_steps_per_env,cfg.max_iterations,cfg.save_interval)==(8192,32,2000,25)
    assert cfg.algorithm.learning_rate==3e-4 and cfg.algorithm.entropy_coef==.001
    assert cfg.algorithm.gamma==.995 and cfg.algorithm.lam==.95
    env=training_env(pushes=False,push_force=200.,push_torque=50.)
    params=env.device_layer.devices['wangyu'].runtime_params
    assert params=={'pushes':False,'push_force':200.,'push_torque':50.,'resample_terrain':True}
    assert env.episode_length_secs==10. and env.sim.control_timestep==1/60
    assert {name:term.weight for name,term in env.task_layer.rewards.items()}==REWARD_WEIGHTS
    assert REWARD_WEIGHTS['torso_velocity']==-4. and REWARD_WEIGHTS['joint_pose']==-.25
    assert 'capture_risk' not in REWARD_WEIGHTS
    assert signature(push_force=30.)[1]!=signature(push_force=200.)[1]


def test_policy_contract_loads_imported_normalization_and_weights():
    from rsl_rl.models import MLPModel
    cfg=runner_config();obs=TensorDict({'policy':torch.zeros(2,146),'critic':torch.zeros(2,146)},batch_size=[2])
    saved=torch.load(ROOT/'logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt',weights_only=False,map_location='cpu')
    for key,size in (('actor',23),('critic',1)):
        params=dataclasses.asdict(getattr(cfg,key));params.pop('class_name')
        model=MLPModel(obs,cfg.obs_groups,key,size,**params)
        model.load_state_dict(saved[key+'_state_dict'],strict=True)
        model.eval()
        with torch.inference_mode():out=model(obs)
        assert out.shape==(2,size) and torch.isfinite(out).all()


def test_batched_push_and_partial_reset_contract():
    m=scene_spec().compile();n=4
    sim=SimpleNamespace(mj_model=m,
        model=SimpleNamespace(geom_dataid=torch.tensor(np.repeat(m.geom_dataid[None],n,0)),geom_pos=torch.tensor(np.repeat(m.geom_pos[None],n,0))),
        data=SimpleNamespace(qpos=torch.tensor(np.repeat(m.qpos0[None],n,0),dtype=torch.float32),qvel=torch.zeros(n,m.nv),ctrl=torch.zeros(n,23),xfrc_applied=torch.zeros(n,m.nbody,6)))
    device=MoonDevice(pushes=True,push_force=30.,push_torque=50.)
    device.initialize(sim);device.reset(sim)
    device.wait.zero_()
    device.set_substep_commands(sim,{'wangyu/action':torch.zeros(n,23)})
    pulse=sim.data.xfrc_applied[:,device.root].clone()
    assert torch.equal(pulse[:,2],torch.zeros(n))
    assert pulse[:,:2].abs().max()<=30 and pulse[:,3:].abs().max()<=50
    assert pulse.abs().max()>0
    device.set_substep_commands(sim,{'wangyu/action':torch.zeros(n,23)})
    torch.testing.assert_close(sim.data.xfrc_applied[:,device.root],pulse)
    before=sim.data.qpos.clone();ctrl=sim.data.ctrl.clone()
    mask=torch.tensor([True,False,False,False]);device.reset(sim,mask)
    torch.testing.assert_close(sim.data.qpos[1:],before[1:])
    torch.testing.assert_close(sim.data.ctrl[1:],ctrl[1:])
    torch.testing.assert_close(sim.data.xfrc_applied[1:,device.root],pulse[1:])
    assert sim.data.xfrc_applied[0,device.root].abs().sum()==0


def test_new_native_training_checkpoint_wins_over_import_name(tmp_path):
    from checkpoint_selection import latest_checkpoint
    imported=tmp_path/'2026-11-15_import_mjlab';imported.mkdir();(imported/'model_200.pt').touch()
    fresh=tmp_path/'2026-11-15_11-30-00_stand';fresh.mkdir()
    (fresh/'training_config.json').write_text('{}')
    (fresh/'model_25.pt').touch();(fresh/'model_100.pt').touch()
    model,reason=latest_checkpoint(tmp_path)
    assert model==fresh/'model_100.pt' and 'native' in reason
