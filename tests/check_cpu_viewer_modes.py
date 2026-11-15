"""Bounded native-window integration check; run each backend in its Mac entrypoint."""
import argparse
from pathlib import Path
import sys
import threading
from unittest.mock import patch
import numpy as np
import torch
import mujoco

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpu_simulation import CpuWorlds
from cpu_viewer import launch
from moon_scene import scene_spec

p=argparse.ArgumentParser()
p.add_argument('--viewer',choices=('classic','studio'),required=True)
p.add_argument('--viz',choices=('physx','skin'),required=True)
args=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(42)
worlds=CpuWorlds(scene_spec(),8,12)
state={'paused':False};reset=threading.Event();keys=[]
with launch(worlds.sim.mj_spec,worlds.model,worlds.data[0],args.viewer,args.viz,
            keys.append,state,reset,simulator=worlds.sim) as viewer:
    viewer.sync()
    physical=worlds.sim.data.qpos.clone()
    activation=int(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION])
    def key(character, expected_world, tracking):
        if args.viewer=='classic':
            # Callback queue is fed by the native render thread in production.
            viewer._keys.put(ord(character))
            viewer.sync()
        else:
            from mujoco.studio.dear_imgui import dear_imgui as imgui
            enum={'.':'Period',',':'Comma','/':'Slash','A':'A','[':'LeftBracket',']':'RightBracket'}[character]
            chord=int(getattr(imgui.Key,enum))
            with patch.object(imgui,'IsKeyChordPressed',side_effect=lambda key,*a: int(key)==chord), \
                 patch.object(imgui,'IsAnyItemActive',return_value=False):
                viewer.sync()
        assert viewer.controls.current==expected_world,(character,state)
        assert viewer.controls.tracking==tracking,(character,state)
    key('.',1,True)
    target=viewer.controls.roots[1]
    assert viewer.native.perturb.select==target
    key('/',1,False)
    key(',',0,True)
    key(',',7,True)
    key('A',7,False)
    assert viewer.controls.overview
    key('A',7,True)
    key('[',7,True);key(']',7,True)
    assert keys==[ord('['),ord(']')],keys
    assert int(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION])==activation
    # The same native perturbation goes to world_7, independent of visual style.
    perturb=viewer.native.perturb
    perturb.select=viewer.controls.roots[7]
    scene=mujoco.MjvScene(viewer.model,maxgeom=2000)
    mujoco.mjv_updateScene(viewer.model,viewer.data,viewer.opt,None,viewer.cam,
                          mujoco.mjtCatBit.mjCAT_ALL,scene)
    mujoco.mjv_initPerturb(viewer.model,viewer.data,scene,perturb)
    perturb.active=mujoco.mjtPertBit.mjPERT_TRANSLATE
    perturb.refselpos[:]=viewer.data.xpos[perturb.select]
    perturb.refselpos[0]+=.1
    with viewer.lock():viewer.update_mouse_interaction()
    forces=state['mouse_forces'].copy()
    assert np.linalg.norm(forces[7])>0
    np.testing.assert_array_equal(forces[:7],0)
    assert viewer.controls.current==7
    perturb.active=0
    with viewer.lock():viewer.update_mouse_interaction()
    np.testing.assert_array_equal(state['mouse_forces'],0)
    torch.testing.assert_close(worlds.sim.data.qpos,physical)
    assert (viewer.model.nskin>0)==(args.viz=='skin')
    if args.viewer=='studio' and args.viz=='skin':
        from types import SimpleNamespace
        from mujoco.studio import ux
        from mujoco.studio.dear_imgui import dear_imgui as imgui
        # Go through App's real double-click handler and patched Pick route.
        pick=None
        for y in np.linspace(.35,.65,9):
            for x in np.linspace(.4,.6,9):
                hit=ux.Pick(viewer.model,viewer.data,viewer.cam,x,y,1.5,viewer.opt)
                if hit.skin>=0 and viewer.controls.body_world.get(hit.body)==7:
                    pick=(x,y,hit);break
            if pick:break
        assert pick is not None,'no visible skin sample for selected world'
        x,y,hit=pick
        io=SimpleNamespace(WantCaptureMouse=False,DisplaySize=SimpleNamespace(x=1200,y=800),
            MousePos=SimpleNamespace(x=x*1200,y=y*800),MouseDelta=SimpleNamespace(x=0,y=0),
            KeyCtrl=False,KeySuper=False,KeyAlt=False,KeyShift=False)
        with patch.object(imgui,'GetIO',return_value=io), \
             patch.object(imgui,'IsMouseDown',return_value=False), \
             patch.object(imgui,'IsMouseDoubleClicked',side_effect=lambda b: b==imgui.MouseButton.Left), \
             patch.object(ux,'move_camera_with_mouse',return_value=None):
            viewer._studio_app.handle_mouse_events()
        assert viewer.native.perturb.select==hit.body
        local=viewer.native.perturb.localpos
        anchor=viewer.data.xpos[hit.body]+viewer.data.xmat[hit.body].reshape(3,3)@local
        np.testing.assert_allclose(anchor,hit.point,atol=2e-6)
        print('[Skin click check] Studio double-click selects visible skin body and exact surface anchor',flush=True)
        # Physical Control and Command must yield identical perturbations, and
        # neither drag may simultaneously orbit the camera.
        saved_camera=(viewer.cam.lookat.copy(),viewer.cam.azimuth,viewer.cam.elevation)
        outputs=[]
        for control,command in ((True,False),(False,True)):
            io.KeyCtrl,io.KeySuper=control,command
            io.MouseDelta.x=12;io.MouseDelta.y=8
            viewer.native.perturb.active=0
            with patch.object(imgui,'GetIO',return_value=io), \
                 patch.object(imgui,'IsMouseDown',side_effect=lambda b: b==imgui.MouseButton.Right), \
                 patch.object(imgui,'IsMouseDoubleClicked',return_value=False), \
                 patch.object(ux,'move_camera_with_mouse') as orbit:
                viewer._studio_app.handle_mouse_events()
                orbit.assert_not_called()
            viewer.update_mouse_interaction()
            outputs.append(state['mouse_forces'].copy())
            assert np.linalg.norm(outputs[-1])>0
        np.testing.assert_allclose(outputs[0],outputs[1],atol=1e-6)
        np.testing.assert_array_equal(viewer.cam.lookat,saved_camera[0])
        assert (viewer.cam.azimuth,viewer.cam.elevation)==saved_camera[1:]
        print('[Modifier check] Studio Control/Command produce identical forces without moving camera',flush=True)
    print(f'[Interaction check] {args.viewer}/{args.viz}: selection, tracking, overview, terrain keys, force mapping/release passed',flush=True)
