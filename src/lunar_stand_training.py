"""Playground lunar-standing training task, matching the MJLab v15 MDP/PPO."""
import hashlib
import json
import math
from mujoco_playground._src.rl.rsl_rl.rsl_config import RslRlOnPolicyRunnerCfg
from reaf.simulation.torch import flat_scene as config
from moon_device import META
from moon_task import moon_env
from provenance import verify_assets

TASK_ID='WangyuMoonStandLunarTrainV1'
EXPERIMENT='wangyu_moon_playground_stand_v1_detail3'
MDP_REVISION='lunar_stand_v2_terminal_fix'
REWARD_WEIGHTS=dict(META['rewards'])
REWARD_COLUMNS={name:i for i,name in enumerate(REWARD_WEIGHTS)}


def runner_config():
    cfg=RslRlOnPolicyRunnerCfg()
    cfg.seed=42;cfg.nworld=8192;cfg.max_iterations=2000
    cfg.num_steps_per_env=32;cfg.save_interval=25
    cfg.experiment_name=EXPERIMENT
    cfg.obs_groups={'actor':['policy'],'critic':['critic']}
    # REAF wrapper rejects clip_actions. MoonDevice applies the original clip2.
    cfg.clip_actions=None
    cfg.actor.hidden_dims=(128,128);cfg.critic.hidden_dims=(128,128)
    cfg.actor.activation='elu';cfg.critic.activation='elu'
    cfg.actor.obs_normalization=True;cfg.critic.obs_normalization=True
    cfg.actor.distribution_cfg.init_std=.25
    cfg.actor.distribution_cfg.std_type='scalar'
    ppo=cfg.algorithm
    ppo.num_learning_epochs=5;ppo.num_mini_batches=4
    ppo.learning_rate=3e-4;ppo.schedule='adaptive'
    ppo.gamma=.995;ppo.lam=.95;ppo.entropy_coef=.001
    ppo.desired_kl=.01;ppo.clip_param=.2;ppo.max_grad_norm=1.
    ppo.value_loss_coef=1.;ppo.use_clipped_value_loss=True
    ppo.normalize_advantage_per_mini_batch=False
    return cfg


def physical_metric(features,column,divisor=1.,square_root=False):
    value=features['wangyu/rewards'][:,column]/divisor
    return value.clamp_min(0).sqrt() if square_root else value


def training_env(pushes=True,push_force=30.,push_torque=0.):
    if not all(math.isfinite(x) and x>=0 for x in (push_force,push_torque)):
        raise ValueError('Push force/torque amplitudes must be finite and nonnegative')
    cfg=moon_env(pushes=pushes,push_force=push_force,push_torque=push_torque)
    for name,term in cfg.task_layer.rewards.items():term.weight=REWARD_WEIGHTS[name]
    for name,term,root,divisor in (
        ('torso_speed_m_s','torso_velocity',True,1.),
        ('torso_angular_speed_rad_s','torso_angular_velocity',True,1.),
        ('momentum_kg_m2_s','centroidal_angular_momentum',True,1.),
        ('joint_pose_rms_rad','joint_pose',True,1.),
        ('actuator_torque_rms_Nm','joint_torques',True,23.),
        ('airborne_fraction','airborne',False,1.),
    ):
        cfg.task_layer.metrics[name]=config.Metric(physical_metric,
            column=REWARD_COLUMNS[term],square_root=root,divisor=divisor)
    return cfg


def validate_runner(cfg):
    for name in ('nworld','max_iterations','num_steps_per_env','save_interval'):
        value=getattr(cfg,name)
        if isinstance(value,bool) or not isinstance(value,int) or value<=0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg.nworld*cfg.num_steps_per_env<cfg.algorithm.num_mini_batches:
        raise ValueError('Rollout must contain at least one sample per minibatch')


def signature(pushes=True,push_force=30.,push_torque=0.):
    mdp={'task':TASK_ID,'revision':MDP_REVISION,'physics_revision':META['physics_revision'],
         'state_sha256':verify_assets(),'observation_dim':146,'action_dim':23,
         'physics_dt':META['physics_dt'],'control_dt':META['control_dt'],
         'episode_seconds':10.,'rewards':REWARD_WEIGHTS,
         'pushes':pushes,'push_force':push_force,'push_torque':push_torque}
    return mdp,hashlib.sha256(json.dumps(mdp,sort_keys=True).encode()).hexdigest()


def update_config(cfg,overrides):
    for key,value in overrides.items():
        obj=cfg;*path,leaf=key.split('.')
        for part in path:obj=getattr(obj,part)
        if not hasattr(obj,leaf):raise ValueError(f'Unknown runner option: {key}')
        setattr(obj,leaf,value)


def register_training_task():
    from mujoco_playground._src.rl import env_registry
    env_registry.register(TASK_ID,runner_config(),training_env)
