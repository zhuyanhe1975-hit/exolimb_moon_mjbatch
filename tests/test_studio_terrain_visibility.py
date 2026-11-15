"""CPU model of Filament's option-delta cache, compared with classic masks."""
from pathlib import Path
from types import SimpleNamespace
import sys
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from studio_terrain import refresh_terrain_visibility


class CachedScene:
    # Match ModelRenderables::SetOptions/SetVisibility: group edits alone
    # don't update scene membership; only option deltas trigger changes.
    def __init__(self,model,options):
        self.groups=options.geomgroup.copy()
        self.visible=self.groups[model.geom_group].astype(bool)

    def sync(self,width,height,model,data,perturb,camera,options,extra_geoms):
        for group in range(6):
            if self.groups[group]!=options.geomgroup[group]:
                self.visible[model.geom_group==group]=bool(options.geomgroup[group])
        self.groups=options.geomgroup.copy()


def test_studio_terrain_membership_matches_classic_across_field_switches():
    geoms=''.join(f'<geom name="g{i}" type="sphere" size=".1" group="5"/>' for i in range(64))
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+geoms+'</worldbody></mujoco>')
    model.geom_group[0]=0
    options=mujoco.MjvOption();options.geomgroup[5]=0
    scene=CachedScene(model,options)
    viewer=SimpleNamespace(model=model,data=None,perturb=None,camera=None,
        vis_options=options,extra_geoms=[],_model_scene=scene,
        _window=SimpleNamespace(GetWidth=lambda:640,GetHeight=lambda:480))
    for active in (21,63,0,21,21,1):
        model.geom_group[:]=5;model.geom_group[active]=0
        # Without invalidation, a normal Studio sync retains stale membership.
        if active==21 and scene.visible[0]:
            scene.sync(640,480,model,None,None,None,options,[])
            assert scene.visible[0] and not scene.visible[active]
        refresh_terrain_visibility(viewer)
        scene.sync(640,480,model,None,None,None,options,[])
        classic_visible=options.geomgroup[model.geom_group].astype(bool)
        np.testing.assert_array_equal(scene.visible,classic_visible)
        np.testing.assert_array_equal(np.flatnonzero(scene.visible),[active])
    assert options.geomgroup[0] and not options.geomgroup[5]


def test_skin_terrain_snapshot_matches_classic_native_scene(monkeypatch):
    from moon_scene import scene_spec
    from moon_viewer import display_spec
    from moon_studio_plugins import TerrainReceiver,TerrainSnapshot
    monkeypatch.setenv('MOON_FILAMENT_SCENE','1')
    monkeypatch.setenv('MOON_VIEWER','studio')
    model=display_spec(SimpleNamespace(mj_spec=scene_spec(),nworld=1)).compile()
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    options=mujoco.MjvOption();options.geomgroup[4]=0;options.geomgroup[5]=0
    receiver=TerrainReceiver();receiver._viewer=SimpleNamespace(model=model,vis_options=options)
    scene=mujoco.MjvScene(model,maxgeom=300);camera=mujoco.MjvCamera()
    heights=model.hfield_data.copy();ids=model.geom_dataid.copy()
    for active in (0,21,63,0):
        receiver.on_terrain(TerrainSnapshot(active))
        mujoco.mjv_updateScene(model,data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        fields=[g for g in scene.geoms[:scene.ngeom] if g.type==mujoco.mjtGeom.mjGEOM_HFIELD]
        assert len(fields)==1
        gid=model.geom('terrain' if active==0 else f'display_terrain_{active:02d}').id
        assert fields[0].objid==gid
        assert model.geom_dataid[gid]==active
        assert model.geom_bodyid[gid]==0
        assert fields[0].rgba[3]==1
        np.testing.assert_allclose(fields[0].pos,data.geom_xpos[gid])
    np.testing.assert_array_equal(model.hfield_data,heights)
    np.testing.assert_array_equal(model.geom_dataid,ids)
