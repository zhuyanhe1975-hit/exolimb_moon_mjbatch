"""Static real Filament resource check: selected actor only, switch, then clear."""
from pathlib import Path
import sys,os
from types import SimpleNamespace
import mujoco,numpy as np,torch
from mujoco import mjrf
from mujoco.rendering.filament import renderer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_scene import scene_spec
from cpu_simulation import CpuWorlds
from moon_viewer import display_spec
from multiworld_viewer import sync_grid
from filament_model_scene import FilamentModelScene
os.environ.update(MOON_VIEWER='studio',MOON_FILAMENT_SCENE='1')
torch.manual_seed(42)
w=CpuWorlds(scene_spec(),8,12)
m=display_spec(w.sim).compile();d=mujoco.MjData(m)
sync_grid(SimpleNamespace(_simulator=w.sim,_world_id=0),m,d)
original=m.mat_rgba.copy();pose=d.qpos.copy()
context=mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL))
r=renderer.Renderer(context);scene=FilamentModelScene(context,r.scene('scn'),m)
camera=mujoco.MjvCamera();camera.distance=24
options=mujoco.MjvOption();options.geomgroup[4]=options.geomgroup[5]=0
perturb=mujoco.MjvPerturb()
for selected in (None,2,7,None):
 perturb.select=0 if selected is None else m.body(f'world_{selected}/robot/Pelvis').id
 scene.sync(1200,900,m,d,perturb,camera,options,[])
 for i in range(m.nskin):
  material=scene._content.get_renderable(mujoco.mjtObj.mjOBJ_SKIN,i).get_material()
  belongs=m.skin(i).name.startswith(f'world_{selected}/') if selected is not None else False
  assert material.selected==belongs,(selected,m.skin(i).name,material.selected)
 np.testing.assert_array_equal(m.mat_rgba,original)
 np.testing.assert_array_equal(d.qpos,pose)
print('[Selection highlight] 8 people: selected actor only, switch, clear; shared materials and physics unchanged',flush=True)
del scene,r,context
