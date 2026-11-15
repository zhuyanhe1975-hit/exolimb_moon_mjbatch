"""CPU lifecycle tests for retained public mjrf scene and overlay resources."""
import sys
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import filament_model_scene as module


def test_scene_and_overlay_resources_are_reused(monkeypatch):
    created=[];members=[];updates=[]
    class Renderable:
        def set_geom_mesh(self,*args):pass
        def set_size(self,*args):pass
        def set_pose(self,*args):pass
        def set_material(self,*args):pass
    def create(params):
        renderable=Renderable();created.append(renderable);return renderable
    class Content:
        def __init__(self,*args):
            updates.append('constructed')
            self._renderables=SimpleNamespace(set_options=lambda options:None)
        def update(self,data):updates.append('updated')
    monkeypatch.setattr(module,'ModelScene',Content)
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    context=SimpleNamespace(create_renderable=create)
    scene=SimpleNamespace(add_renderable=members.append,remove_renderable=members.remove)
    adapter=module.FilamentModelScene(context,scene,model)
    geom=mujoco.MjvGeom();mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,
        np.ones(3)*.01,np.zeros(3),np.eye(3).ravel(),np.ones(4))
    options=mujoco.MjvOption();camera=mujoco.MjvCamera();perturb=mujoco.MjvPerturb()
    for _ in range(3):adapter.sync(640,480,model,data,perturb,camera,options,[geom])
    assert len(created)==len(members)==1 and updates.count('constructed')==1
    adapter.sync(640,480,model,data,perturb,camera,options,[])
    assert not members
    adapter.sync(640,480,model,data,perturb,camera,options,[geom])
    assert len(created)==1 and len(members)==1
    del adapter
    assert not members
