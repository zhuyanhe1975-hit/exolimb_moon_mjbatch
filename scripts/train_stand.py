"""Train lunar standing in Playground. Fresh PPO by default; never opens a viewer."""
from pathlib import Path
import dataclasses
from datetime import datetime
import json
import os
import shutil
import sys
from zoneinfo import ZoneInfo
from absl import app,flags
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

flags.DEFINE_integer('num_envs',None,'Number of independent worlds; default 8192.')
flags.DEFINE_alias('num-envs','num_envs')
flags.DEFINE_integer('iterations',None,'Number of additional PPO updates; default 2000.')
flags.DEFINE_alias('max-iterations','iterations')
flags.DEFINE_string('device','cuda:0','Training device.')
flags.DEFINE_integer('seed',42,'Random seed.')
flags.DEFINE_bool('pushes',True,'Enable independent horizontal force/torque pulses.')
flags.DEFINE_float('push_force',30.,'World X/Y force amplitude in N; Z remains zero.')
flags.DEFINE_float('push_torque',0.,'World XYZ torque amplitude in Nm.')
flags.DEFINE_alias('push-force','push_force');flags.DEFINE_alias('push-torque','push_torque')
flags.DEFINE_string('resume_path',None,'Resume same task: actor, critic, optimizer and iteration.')
flags.DEFINE_string('init_from',None,'Initialize actor/critic from checkpoint; fresh optimizer/iteration.')
flags.DEFINE_alias('init-from','init_from')
flags.DEFINE_bool('terrain_transfer',False,'Explicitly allow initialization on a changed terrain.')
flags.DEFINE_alias('terrain-transfer','terrain_transfer')
flags.DEFINE_string('logdir',None,'New output directory; default logs/<timestamp>_<experiment>.')
flags.DEFINE_string('overrides',None,'JSON runner overrides; e.g. {"save_interval":25}.')
flags.DEFINE_bool('compile_policy',False,'Experiment: torch.compile actor and critic in default mode; startup compilation can be slow.')
flags.DEFINE_alias('compile-policy','compile_policy')
flags.DEFINE_string('env_overrides',None,'JSON with pushes, push_force and/or push_torque.')
flags.DEFINE_bool('dry_run',False,'Compile/check configuration on CPU only; no reset, stepping or PPO.')
flags.DEFINE_alias('dry-run','dry_run')
FLAGS=flags.FLAGS


def prepare():
    for key in ('MOON_MEASUREMENT_GRAPH','MOON_SPARSE_RESET','MOON_BATCHED_LOGS','MOON_ROLLOUT_LOGGER','MOON_BATCHED_TERRAIN'):
        os.environ.setdefault(key,os.environ.get('MOON_FAST_TRAIN','1'))
    from lunar_stand_training import runner_config,update_config,validate_runner,signature,EXPERIMENT
    cfg=runner_config()
    update_config(cfg,json.loads(FLAGS.overrides) if FLAGS.overrides else {})
    if FLAGS.compile_policy:
        if cfg.torch_compile_mode not in (None,'default'):
            raise ValueError('compile-policy conflicts with torch_compile_mode override')
        cfg.torch_compile_mode='default'
    if FLAGS.num_envs is not None:cfg.nworld=FLAGS.num_envs
    if FLAGS.iterations is not None:cfg.max_iterations=FLAGS.iterations
    cfg.seed=FLAGS.seed;validate_runner(cfg)
    if FLAGS.seed<0:raise ValueError('seed must be nonnegative')
    params={'pushes':FLAGS.pushes,'push_force':FLAGS.push_force,'push_torque':FLAGS.push_torque}
    extra=json.loads(FLAGS.env_overrides) if FLAGS.env_overrides else {}
    unknown=set(extra)-set(params)
    if unknown:raise ValueError(f'Unknown environment options: {sorted(unknown)}')
    params.update(extra)
    if not isinstance(params['pushes'],bool):raise ValueError('pushes must be boolean')
    mdp,digest=signature(**params)
    if FLAGS.resume_path and FLAGS.init_from:raise ValueError('Choose resume_path or init_from')
    source=FLAGS.resume_path or FLAGS.init_from
    if source:
        source=Path(source).expanduser().resolve()
        if not source.is_file():raise FileNotFoundError(source)
        from provenance import verify_checkpoint
        verify_checkpoint(source,allow_transfer=FLAGS.terrain_transfer)
        if FLAGS.resume_path:
            config_file=source.parent/'training_config.json'
            if not config_file.is_file():raise ValueError('This is not a native training run. Use --init-from for imported weights.')
            previous=json.loads(config_file.read_text())
            if previous['mdp_sha256']!=digest:
                raise ValueError('Resume MDP differs. Use --init-from with explicit transfer instead.')
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')
    logdir=Path(FLAGS.logdir).expanduser().resolve() if FLAGS.logdir else ROOT/'logs'/f'{stamp}_{EXPERIMENT}'
    record={'task':mdp['task'],'mode':'resume' if FLAGS.resume_path else 'initialize' if FLAGS.init_from else 'fresh',
            'source_checkpoint':str(source) if source else None,'mdp':mdp,'mdp_sha256':digest,
            'runner':dataclasses.asdict(cfg),'device':FLAGS.device,'logdir':str(logdir)}
    record['performance']={key:os.environ.get(key,'1') for key in
        ('MOON_MEASUREMENT_GRAPH','MOON_SPARSE_RESET','MOON_BATCHED_LOGS','MOON_ROLLOUT_LOGGER','MOON_BATCHED_TERRAIN')}
    return cfg,params,source,logdir,record


def main(argv):
    if len(argv)>1:raise app.UsageError('Unexpected positional arguments: '+str(argv[1:]))
    from lunar_stand_training import training_env
    cfg,params,source,logdir,record=prepare()
    env_cfg=training_env(**params)
    # Compilation validates the native task without starting a simulator.
    model=env_cfg.base_scene.compile()
    assert (model.nu,model.nq,model.nv,model.nhfield)==(23,30,29,64)
    print(json.dumps(record,indent=2),flush=True)
    if FLAGS.dry_run:
        print('[Dry run] CPU compile passed; no reset/step/PPO/viewer; no run directory created.',flush=True)
        return
    import torch
    from mujoco_playground._src.rl import rng
    from training_logs import LunarTrainingVecEnv
    from reaf.simulation.torch import flat_scene
    from rsl_rl.runners import OnPolicyRunner
    from provenance import save,digest
    if logdir.exists():raise FileExistsError(f'Refusing to overwrite existing run: {logdir}')
    logdir.mkdir(parents=True)
    save(logdir)
    snapshot=logdir/'terrain';snapshot.mkdir()
    for name in ('state.npz','metadata.json','manifest.json'):
        shutil.copy2(ROOT/'assets/playground'/name,snapshot/name)
    record['source_checkpoint_sha256']=digest(source) if source else None
    record['source_sha256']={name:digest(ROOT/'src'/name) for name in ('lunar_stand_training.py','moon_device.py','moon_scene.py','balance_features.py','local_lunar_terrain.py','training_logs.py','tensor_graph.py','rollout_logger.py','measurement_terrain.py')}
    (logdir/'training_config.json').write_text(json.dumps(record,indent=2)+'\n')
    env=None
    try:
        rng.seed(cfg.seed)
        env_cfg.sim.nworld=cfg.nworld;env_cfg.sim.xpu_device=FLAGS.device
        env=flat_scene.build_env(env_cfg)
        wrapped=LunarTrainingVecEnv(env,cfg=cfg.vecenv_wrapper_cfg,
            batched_logs=os.environ.get('MOON_BATCHED_LOGS','1')=='1')
        runner=OnPolicyRunner(wrapped,dataclasses.asdict(cfg),log_dir=str(logdir),device=FLAGS.device)
        from rollout_logger import install_rollout_logger
        install_rollout_logger(runner,enabled=os.environ.get('MOON_ROLLOUT_LOGGER','1')=='1')
        if source:
            runner.load(str(source),map_location=FLAGS.device,load_cfg={
                'actor':True,'critic':True,'optimizer':bool(FLAGS.resume_path),
                'iteration':bool(FLAGS.resume_path)})
        print(f'[Train] {cfg.nworld} worlds; {cfg.max_iterations} additional updates; '
              f'{cfg.num_steps_per_env} steps/update; pushes={params}; output={logdir}',flush=True)
        runner.learn(num_learning_iterations=cfg.max_iterations,init_at_random_ep_len=not bool(FLAGS.resume_path))
        (logdir/'status.json').write_text(json.dumps({'status':'complete','iteration':runner.current_learning_iteration},indent=2)+'\n')
    except BaseException as error:
        import traceback
        (logdir/'status.json').write_text(json.dumps({'status':'failed','error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':app.run(main)
