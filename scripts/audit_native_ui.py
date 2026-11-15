"""Fixed ImGui widgets -> mjrf static screenshot; no native window or simulation."""
from pathlib import Path
import sys
import numpy as np
import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import renderer
from mujoco.studio.dear_imgui import dear_imgui as imgui
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from native_ui_bridge import NativeUiBridge
im=imgui.CreateContext();io=imgui.GetIO();io.DisplaySize=imgui.Vec2(900,600);io.BackendFlags|=16
ctx=mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL))
r=renderer.Renderer(ctx);bridge=NativeUiBridge(ctx,r.scene('ux'),im)
pixels=np.empty((600,900,3),np.uint8);r.target('out',pixels)
for _ in range(4):
    imgui.NewFrame()
    imgui.SetNextWindowPos(imgui.Vec2(10,10));imgui.SetNextWindowSize(imgui.Vec2(400,520))
    imgui.Begin('Options');imgui.Text('Simulation');imgui.Button('Reset');imgui.SameLine();imgui.Button('Run')
    imgui.Separator();imgui.Text('Rendering');imgui.Checkbox('Skin',True);imgui.Checkbox('Shadows',True)
    imgui.Text('Terrain: 12 / 64');imgui.Text('Keyboard: [ previous, ] next')
    imgui.End()
    imgui.SetNextWindowPos(imgui.Vec2(430,10));imgui.SetNextWindowSize(imgui.Vec2(450,520))
    imgui.Begin('Inspector');imgui.Text('Joints / Controls / Sensors');imgui.Text('UI text and controls must remain visible.');imgui.End()
    bridge.update()
    camera=mjrf.Camera(orthographic=True,pos=[0,0,1],forward=[0,0,-1],up=[0,1,0],frustum_top=0,
        frustum_near=0,frustum_far=1,frustum_center=450,frustum_width=450,frustum_bottom=600)
    r.render([renderer.View(scene='ux',target='out',camera=camera,post_processing=False,skybox=False,haze=False,shadows=False,reflections=False)])
out=ROOT/'outputs/render_reference/ui_fixed.png';out.parent.mkdir(exist_ok=True)
Image.fromarray(pixels).save(out);print('Static UI audit:',out,flush=True)
del bridge,r,ctx
imgui.DestroyContext(im)
