"""One-time CPU static export; only this migration utility imports MJLab."""
import sys,json,dataclasses
from pathlib import Path
import numpy as np
SOURCE=Path('/home/yhzhu/myWorks/exolimb_moon_mjlab')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src'))
from mjlab.scene import Scene
import lunar_finetune_task as task
import local_lunar_terrain as terrain
from terrain_provenance import save
cfg=task.terrain_stand_cfg()
cfg.scene.num_envs=1
scene=Scene(cfg.scene,'cpu')
scene.spec.option.timestep=cfg.sim.mujoco.timestep
scene.spec.option.gravity=cfg.sim.mujoco.gravity
model=scene.compile()
cfg.sim.mujoco.apply(model)
out=ROOT/'assets/playground';out.mkdir(exist_ok=True)
scene.write(out/'scene')
import mujoco
xml_model=mujoco.MjModel.from_xml_path(str(out/'scene/scene.xml'))
cfg.sim.mujoco.apply(xml_model)
mujoco.mj_saveLastXML(str(out/'scene/scene.xml'),xml_model)
# Export exact compiled arrays separately: XML exporters can omit inline hfield data.
arrays={k:np.array(getattr(model,k)) for k in ('qpos0','hfield_data','hfield_size','actuator_trnid','jnt_qposadr','jnt_dofadr','jnt_range')}
arrays.update(terrain.bank_data())
sole,cover=terrain.sole_sampling(model);arrays.update(sole_points=sole,sole_cover=np.array(cover))
np.savez_compressed(out/'state.npz',**arrays)
save(out/'terrain')
metadata={'source':str(SOURCE),'task':task.TASK_ID,'physics_revision':task.PHYSICS_REVISION,
 'obs_dim':task.OBS_DIM,'action_dim':23,'episode_seconds':cfg.episode_length_s,
 'control_dt':cfg.sim.mujoco.timestep*cfg.decimation,'physics_dt':cfg.sim.mujoco.timestep,
 'rewards':task.REWARD_WEIGHTS,'push':task.PUSH_PARAMS,
 'joints':[model.joint(i).name for i in range(1,model.njnt)],
 'terminations':{k:{'function':v.func.__name__,'params':str(v.params)} for k,v in cfg.terminations.items()}}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
print(json.dumps(metadata,indent=2,default=str))
