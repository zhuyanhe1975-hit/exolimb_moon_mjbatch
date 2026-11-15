"""Pick visible deformed skins, with exact surface anchors and occlusion."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from mujoco.studio import ux

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from skin_picking import NativeSkinPicker,screen_ray
from moon_scene import scene_spec
from moon_viewer import display_spec
from cpu_simulation import CpuWorlds


def project(scene,point,aspect):
    camera=mujoco.mjv_averageCamera(scene.camera[0],scene.camera[1])
    position,forward,up=(np.zeros(3) for _ in range(3))
    mujoco.mjv_cameraInModel(position,forward,up,scene)
    right=np.cross(forward,up)
    q=point-position
    factor=camera.frustum_near/np.dot(q,forward)
    horizontal=np.dot(q,right)*factor
    vertical=np.dot(q,up)*factor
    width=camera.frustum_width or .5*aspect*(camera.frustum_top-camera.frustum_bottom)
    return .5+.5*(horizontal-camera.frustum_center)/width,1-(vertical-camera.frustum_bottom)/(camera.frustum_top-camera.frustum_bottom)


class SkinPickingTest(unittest.TestCase):
    def test_exact_surface_and_occlusion(self):
        spec=mujoco.MjSpec.from_string('''<mujoco><worldbody>
        <body name="front" pos="0 0 0"/><body name="back" pos="0 1 0"/>
        </worldbody><deformable>
        <skin name="front_skin" vertex="-1 0 -1 1 0 -1 0 0 1" face="0 1 2">
        <bone body="front" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 1 2" vertweight="1 1 1"/></skin>
        <skin name="back_skin" vertex="-1 0 -1 1 0 -1 0 0 1" face="0 1 2">
        <bone body="back" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 1 2" vertweight="1 1 1"/></skin>
        </deformable></mujoco>''')
        model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        camera=mujoco.MjvCamera();camera.lookat[:]=0;camera.distance=5;camera.azimuth=90;camera.elevation=0
        options=mujoco.MjvOption();options.flags[mujoco.mjtVisFlag.mjVIS_SKIN]=1
        scene=mujoco.MjvScene(model,maxgeom=10)
        mujoco.mjv_updateScene(model,data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        target=np.array([.1,0.,-.1])
        # Choose the nearer surface using the actual view direction.
        position=np.zeros(3);forward=np.zeros(3);up=np.zeros(3)
        mujoco.mjv_cameraInModel(position,forward,up,scene)
        near='front' if position[1]<0 else 'back'
        target[1]=0 if near=='front' else 1
        x,y=project(scene,target,1.5)
        picker=NativeSkinPicker()
        result=picker.pick(model,data,camera,x,y,1.5,options)
        self.assertEqual(result.body,model.body(near).id)
        self.assertEqual(result.skin,model.skin(near+'_skin').id)
        np.testing.assert_allclose(result.point,target,atol=2e-6)
        # A selection anchor must lie on the clicked surface, not a corner vertex.
        self.assertGreater(np.linalg.norm(result.point-np.array([-1,target[1],-1])),1)
        options.flags[mujoco.mjtVisFlag.mjVIS_SKIN]=0
        self.assertEqual(picker.pick(model,data,camera,x,y,1.5,options).body,-1)

    def test_real_suit_single_and_eight_people(self):
        picker=NativeSkinPicker()
        for count in (1,8):
            torch.manual_seed(42)
            worlds=CpuWorlds(scene_spec(),count,12)
            with patch.dict(os.environ,{'MOON_VIEWER':'studio','MOON_FILAMENT_SCENE':'1'}):
                model=display_spec(worlds.sim).compile()
            data=mujoco.MjData(model)
            if count==1:data.qpos[:]=worlds.data[0].qpos
            else:
                from multiworld_viewer import sync_grid
                sync_grid(SimpleNamespace(_simulator=worlds.sim,_world_id=0),model,data)
            mujoco.mj_forward(model,data)
            options=mujoco.MjvOption();options.geomgroup[4]=options.geomgroup[5]=0
            options.flags[mujoco.mjtVisFlag.mjVIS_SKIN]=1
            camera=mujoco.MjvCamera();camera.distance=5;camera.azimuth=135;camera.elevation=-20
            root=model.body(('world_0/' if count>1 else '')+'robot/Pelvis').id
            camera.lookat[:]=data.xpos[root]+[0,0,.2]
            scene=mujoco.MjvScene(model,maxgeom=2000)
            mujoco.mjv_updateScene(model,data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            hits=0;parts=set();before=data.qpos.copy()
            for y in np.linspace(.25,.75,11):
                for x in np.linspace(.3,.7,11):
                    p=np.zeros(3);g=np.array([-1],np.int32);f=g.copy();s=g.copy()
                    body=mujoco.mjv_select(model,data,options,1.5,x,1-y,scene,p,g,f,s)
                    if s[0]<0:continue
                    hits+=1;parts.add(int(s[0]))
                    with self.subTest(count=count,x=x,y=y):
                        result=picker.pick(model,data,camera,x,y,1.5,options)
                        self.assertEqual((result.body,result.skin),(body,int(s[0])))
                        origin,direction=screen_ray(picker.scene,x,y,1.5)
                        np.testing.assert_allclose(np.cross(result.point-origin,direction),0,atol=2e-6)
            self.assertGreater(hits,10);self.assertGreater(len(parts),2)
            np.testing.assert_array_equal(before,data.qpos)
            # Repeated picking while tracking must not move/smooth the camera.
            camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING;camera.trackbodyid=root
            lookat=camera.lookat.copy()
            for _ in range(3):picker.pick(model,data,camera,.5,.5,1.5,options)
            np.testing.assert_array_equal(camera.lookat,lookat)


if __name__=='__main__':unittest.main()
