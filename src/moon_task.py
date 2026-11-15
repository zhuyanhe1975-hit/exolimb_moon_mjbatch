"""Current MuJoCo Playground 1.x / REAF configuration."""
import json
import hashlib
from pathlib import Path
import mujoco
import numpy as np
import torch
from reaf.core.termination_checker import TerminationResult
from reaf.simulation.torch import flat_scene as config
from moon_device import MoonDevice,ASSET,META

TASK_ID='WangyuMoonStandLunarV15'
REWARD_NAMES=tuple(META['rewards'])

from moon_scene import scene_spec

def reward(features,index):return features['wangyu/rewards'][:,index]
def failed(features):
    return torch.where(features['wangyu/failed'],int(TerminationResult.TERMINATE),0)

def moon_env(pushes=True,push_force=30.,push_torque=0.,resample_terrain=True,shared_terrain=False,terrain_size=16,terrain_idx=None):
    observations={'state':config.Obs('wangyu/state').configure(output_shape=(146,))}
    return config.Environment(base_scene=scene_spec(terrain_size),
        device_layer=config.DeviceLayer(devices={'wangyu':config.Component(MoonDevice,pushes=pushes,push_force=push_force,push_torque=push_torque,resample_terrain=resample_terrain,**({'shared_terrain':True} if shared_terrain else {}),**({'terrain_size':terrain_size} if terrain_size!=16 else {}),**({'terrain_idx':terrain_idx} if terrain_idx is not None else {}))}),
        task_layer=config.TaskLayer(
            rewards={name:config.Reward(reward,index=i,weight=META['rewards'][name]) for i,name in enumerate(REWARD_NAMES)},
            terminations={'fallen_or_outside':config.Term(failed)}),
        observation_adapter={key:config.ObsGroup(terms=observations) for key in ('policy','critic')},
        sim=config.Sim(control_timestep=META['control_dt'],simulation_timestep=META['physics_dt'],
            gravity=(0,0,-1.635),nconmax=128,njmax=512),episode_length_secs=10.)

def register():
    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl.rsl_config import RslRlOnPolicyRunnerCfg
    cfg=RslRlOnPolicyRunnerCfg()
    cfg.nworld=8192;cfg.clip_actions=None;cfg.experiment_name='wangyu_moon_playground_v15_detail3'
    cfg.obs_groups={'actor':['policy'],'critic':['critic']}
    cfg.actor.hidden_dims=(128,128);cfg.critic.hidden_dims=(128,128)
    cfg.actor.obs_normalization=True;cfg.critic.obs_normalization=True
    cfg.actor.distribution_cfg.init_std=.25
    cfg.algorithm.gamma=.995
    env_registry.register(TASK_ID,cfg,moon_env)
    from lunar_stand_training import register_training_task
    register_training_task()
