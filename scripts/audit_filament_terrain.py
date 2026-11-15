"""Static Filament source/display comparison. No policy or physics stepping.

Run each mode in a fresh process: audit_filament_terrain.py source | display.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from mujoco import mjrf
from mujoco.studio import native_viewer as native
from mujoco.rendering.filament import renderer
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moon_scene import scene_spec
from moon_viewer import display_spec
mode=sys.argv[1] if len(sys.argv)>1 else 'display'
if mode not in ('source','display'):raise SystemExit('Expected source or display')
spec=scene_spec()
if mode=='display':spec=display_spec(SimpleNamespace(mj_spec=spec))
model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
ctx=mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL))
rend=renderer.Renderer(ctx)
scene=native._viewer.ModelScene(ctx,rend.scene('scn'),model)
cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
cam.lookat[:]=[0,0,0];cam.distance=12;cam.azimuth=135;cam.elevation=-35
opt=mujoco.MjvOption();pert=mujoco.MjvPerturb()
pixels=np.empty((720,960,3),dtype=np.uint8);rend.target('out',pixels)
for _ in range(5):
    scene.sync(960,720,model,data,pert,cam,opt,[])
    rend.render([renderer.View(scene='scn',target='out',camera=mujoco.mjv_camera2GLCamera(model,data,cam),
        headlight=True,headlight_color=(.6,.6,.6),headlight_intensity=scene.get_headlight_intensity())])
out=ROOT/'outputs/playground_play/studio_audit';out.mkdir(exist_ok=True)
Image.fromarray(pixels).save(out/(mode+'_static.png'))
del scene,rend,ctx
