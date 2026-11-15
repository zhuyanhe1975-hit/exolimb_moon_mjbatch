"""Static MuJoCo 4.0.1 Filament audit using the production Studio scene class.

No policy loading, physics stepping, training, playback, or native window.
"""
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from mujoco import mjrf
from mujoco.rendering.filament import renderer
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from moon_scene import scene_spec
from moon_viewer import display_spec
from filament_model_scene import FilamentModelScene
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='reference_final')
parser.add_argument('--azimuth',type=float,default=148.)
parser.add_argument('--closeup',action='store_true',help='Fixed visor closeup; still zero physics steps')
parser.add_argument('--cloth-closeup',action='store_true',help='Fixed upper body fabric closeup')
parser.add_argument('--source-cloth',action='store_true',help='Audit only: use uncalibrated source cloth maps')
parser.add_argument('--environment',default='lunar',help='lunar, royal, or KTX file')
args=parser.parse_args();os.environ['MOON_FILAMENT_SCENE']='1'
os.environ['MOON_REFLECTION_ENV']=args.environment
spec=display_spec(SimpleNamespace(mj_spec=scene_spec(),nworld=1),single=True)
if args.source_cloth:
    for part,index in (('Suit',4),('Boot',3)):
        spec.texture(f'skin_tex_{index}').file=str(ROOT/f'assets/wangyu_skin/{part}_BaseColor.png')
        for suffix in ('Metallic','Roughness','Normal'):
            spec.texture(f'pbr_{part}_{suffix}').file=str(ROOT/f'assets/suit_pbr/{part}_{suffix}.png')
    spec.texture('pbr_Helmet_Normal').file=str(ROOT/'assets/suit_pbr/Helmet_Normal.png')
model=spec.compile()
data=mujoco.MjData(model)
for i in range(64):model.geom_group[model.geom('terrain' if i==0 else f'display_terrain_{i:02d}').id]=0 if i==21 else 5
options=mujoco.MjvOption();options.geomgroup[4]=0;options.geomgroup[5]=0
mujoco.mj_forward(model,data)
hit=mujoco.mj_ray(model,data,np.array([0.,0.,100.]),np.array([0.,0.,-1.]),
    np.array([1,0,0,0,0,0],dtype=np.uint8),True,-1,None)
if hit<0:raise RuntimeError('Static placement ray missed ground')
data.qpos[2]+=100-hit+.02;mujoco.mj_forward(model,data)
ctx=mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL))
rend=renderer.Renderer(ctx);scene=FilamentModelScene(ctx,rend.scene('scn'),model)
cam=mujoco.MjvCamera();cam.lookat[:]=data.qpos[:3]+[0,0,.2];cam.distance=2.9;cam.azimuth=args.azimuth;cam.elevation=-10
if args.closeup:
    cam.lookat[:]=data.qpos[:3]+[0,0,.83];cam.distance=.75
if args.cloth_closeup:
    cam.lookat[:]=data.qpos[:3]+[0,0,.5];cam.distance=1.15
pixels=np.empty((900,900,3),dtype=np.uint8);rend.target('out',pixels)
for _ in range(6):
    scene.sync(900,900,model,data,mujoco.MjvPerturb(),cam,options,[])
    rend.render([renderer.View(scene='scn',target='out',camera=mujoco.mjv_camera2GLCamera(model,data,cam),
        headlight=False,headlight_intensity=scene.get_headlight_intensity())])
out=ROOT/'outputs/render_reference';out.mkdir(parents=True,exist_ok=True)
Image.fromarray(pixels).save(out/(args.label+'.png'))
(out/(args.label+'.json')).write_text(json.dumps({'mujoco':mujoco.mj_versionString(),
    'scene_class':'filament_model_scene.FilamentModelScene (same class used by StudioNativeViewer)',
    'static':True,'physics_steps':0,'native_window_tested':False,'source_cloth':args.source_cloth,
    'sun_count':int((model.light_type==mujoco.mjtLightType.mjLIGHT_DIRECTIONAL).sum()),
    'image_light_count':int((model.light_type==mujoco.mjtLightType.mjLIGHT_IMAGE).sum()),
    'headlight_intensity':scene.get_headlight_intensity(),'camera':str(cam),
    'visor':{'visible_skin':'skin_Suit_Shield_0','material':'skin_mat_visor',
             'color_texture':'visor_color','metallic_texture':'visor_metallic',
             'roughness_texture':'visor_roughness','normal_texture':'visor_normal',
             'gold_roughness':json.loads((ROOT/'assets/visor_pbr/manifest.json').read_text())['roughness'],
             'thin_film_interference':False,'scene_reflection':args.environment}},indent=2)+'\n')
print('Saved static audit:',out/(args.label+'.png'),flush=True)
del scene,rend,ctx
