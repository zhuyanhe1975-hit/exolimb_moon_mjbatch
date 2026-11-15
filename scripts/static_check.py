"""CPU compilation/kinematics and adapter contract checks; never mj_step."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
import ast,json,sys
import numpy as np
import mujoco
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from moon_scene import scene_spec
for path in [ROOT/'launch.py',* (ROOT/'src').glob('*.py')]:
    ast.parse(path.read_text(),filename=str(path))
    if path.parent==ROOT/'src':
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('mjlab'),path
            if isinstance(node,ast.Import):assert all(not n.name.startswith('mjlab') for n in node.names),path
spec=scene_spec();model=spec.compile();data=mujoco.MjData(model)
assert model.nu==23 and model.nq==30 and model.nv==29 and model.nhfield==64
assert abs(model.opt.gravity[2]+1.635)<1e-10
for side in ('left','right'):assert model.sensor('contacts/robot/'+side+'_sole').dim[0]==32
with np.load(ROOT/'assets/playground/state.npz') as z:
    np.testing.assert_allclose(model.hfield_data,z['hfield_data'],atol=2e-7)
    np.testing.assert_allclose(model.qpos0,z['qpos0'],atol=2e-6)
    assert len(set(model.actuator_trnid[:,0]))==23
# Interface stubs only allow testing pure adapter methods without installing REAF.
# This does not claim integration validation of the actual REAF runtime.
for name in ('dm_env','reaf','reaf.simulation'):
    sys.modules[name]=ModuleType(name)
sys.modules['dm_env'].specs=SimpleNamespace(Array=lambda *a,**kw:None)
sys.modules['reaf.simulation'].torch=SimpleNamespace(Device=object)
from moon_device import MoonDevice
n=8
sim=SimpleNamespace(mj_model=model,
    model=SimpleNamespace(geom_dataid=torch.tensor(np.repeat(model.geom_dataid[None],n,0)),geom_pos=torch.tensor(np.repeat(model.geom_pos[None],n,0))),
    data=SimpleNamespace(qpos=torch.tensor(np.repeat(model.qpos0[None],n,0),dtype=torch.float32),qvel=torch.zeros(n,model.nv),ctrl=torch.zeros(n,23),xfrc_applied=torch.zeros(n,model.nbody,6),sensordata=torch.zeros(n,model.nsensordata),actuator_force=torch.zeros(n,23)))
device=MoonDevice(pushes=False);device.initialize(sim);device.reset(sim)
for i in range(n):
    data.qpos[:]=sim.data.qpos[i].numpy();data.qvel[:]=0
    model.geom_dataid[:]=sim.model.geom_dataid[i].numpy();model.geom_pos[:]=sim.model.geom_pos[i].numpy()
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_comVel(model,data);mujoco.mj_subtreeVel(model,data)
    mujoco.mj_sensorPos(model,data);mujoco.mj_sensorVel(model,data)
    sim.data.sensordata[i]=torch.tensor(data.sensordata,dtype=torch.float32)
result=device.get_measurements(sim)
assert result['wangyu/state'].shape==(n,146)
assert result['wangyu/rewards'].shape==(n,9)
assert torch.isfinite(result['wangyu/state']).all()
# Action clipping after default offset, then map joint order to actuator order.
a=torch.linspace(-3,3,23).repeat(n,1)
device.set_substep_commands(sim,{'wangyu/action':a})
expected=(device.q0[device.qadr]+.25*a.clamp(-2,2)).maximum(torch.maximum(device.limits[:,0],torch.tensor(-.5))).minimum(torch.minimum(device.limits[:,1],torch.tensor(.5)))
torch.testing.assert_close(sim.data.ctrl,expected[:,device.aorder])
device.on_step(sim);torch.testing.assert_close(device.action,a.clamp(-2,2))
# Partial reset must not overwrite ongoing worlds' command or action state.
keep=sim.data.qpos[1:].clone();mask=torch.zeros(n,dtype=torch.bool);mask[0]=True
device.reset(sim,mask);torch.testing.assert_close(sim.data.qpos[1:],keep)
report={'validation':'CPU compilation, native kinematics/sensors, tensor adapter only',
 'dynamic_simulation':False,'training':False,'playground_runtime_integration':False,
 'joints':23,'observations':146,'hfields':64,'test_worlds':n}
out=ROOT/'outputs';out.mkdir(exist_ok=True);(out/'static_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
