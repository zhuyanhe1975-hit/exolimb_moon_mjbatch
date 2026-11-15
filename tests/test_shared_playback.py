"""Static eight-person, one-surface playback contract; no physics stepping."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import mujoco
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_scene import scene_spec
from moon_device import MoonDevice
from moon_viewer import display_spec
from multiworld_viewer import world_count,sync_grid,grid_forces
from moon_studio_plugins import TerrainReceiver,TerrainSnapshot


@pytest.mark.parametrize('n,skin,terrain_size,terrain_idx',[(8,True,16,None),(4,False,16,None),(4,False,100,None),(4,False,16,1),(4,False,16,64)])
def test_people_share_one_collision_and_display_surface(monkeypatch,n,skin,terrain_size,terrain_idx):
    monkeypatch.setenv('MOON_VIEWER','classic')
    monkeypatch.setenv('MOON_FILAMENT_SCENE','1' if skin else '0')
    spec=scene_spec(terrain_size);source=spec.compile()
    def batch(value):return torch.tensor(np.repeat(value[None],n,axis=0))
    native=mujoco.MjData(source)
    sim=SimpleNamespace(nworld=n,mj_spec=spec,mj_model=source,
        model=SimpleNamespace(**{k:batch(getattr(source,k)) for k in
            ('geom_dataid','geom_pos','geom_quat','geom_size','geom_rgba')}),
        data=SimpleNamespace(**{k:batch(getattr(native,k)) for k in
            ('qpos','qvel','ctrl','xfrc_applied','mocap_pos','mocap_quat')}))
    device=MoonDevice(pushes=False,resample_terrain=False,shared_terrain=True,terrain_size=terrain_size,terrain_idx=terrain_idx)
    device.initialize(sim);device.reset(sim)
    gid=source.geom('terrain').id
    ids=sim.model.geom_dataid[:,gid].clone()
    assert torch.equal(ids,ids[:1].expand_as(ids))
    if terrain_idx is not None:assert int(ids[0])==terrain_idx-1
    xy=sim.data.qpos[:,:2].numpy()
    distances=((xy[:,None]-xy[None,:])**2).sum(-1)
    np.fill_diagonal(distances,np.inf)
    assert distances.min()>2.**2
    assert np.ptp(xy,axis=0).min()>6.
    assert np.abs(xy).max()<=(45. if terrain_size==100 else 5.)
    old=sim.data.qpos.clone()
    device.reset(sim,torch.arange(n)==1)
    torch.testing.assert_close(sim.data.qpos[2:],old[2:])
    assert torch.equal(sim.model.geom_dataid[:,gid],ids)
    single=display_spec(sim,single=True).compile()
    model=display_spec(sim).compile();data=mujoco.MjData(model)
    assert model.nlight==single.nlight==(2 if skin else 1)
    assert np.count_nonzero(model.light_type==mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)==1
    assert model.light(0).name=='world_0/sun'
    np.testing.assert_array_equal(model.light_diffuse,single.light_diffuse)
    np.testing.assert_array_equal(model.light_ambient,single.light_ambient)
    np.testing.assert_array_equal(model.light_specular,single.light_specular)
    assert model.nmesh==single.nmesh
    assert model.nmat==single.nmat
    assert model.ntex==single.ntex
    for field in ('mesh_vert','mesh_face','tex_data','hfield_data'):
        assert getattr(model,field).size==getattr(single,field).size
    for w in range(n):
        for i in range(single.nskin):
            assert model.skin_matid[w*single.nskin+i]==model.skin_matid[i]
    assert world_count(model)==n and model.nskin==n*single.nskin
    assert model.nhfield==source.nhfield==(1 if terrain_size==100 else 64)
    assert model.hfield_data.size==source.hfield_data.size
    assert np.count_nonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_HFIELD)==source.nhfield
    np.testing.assert_array_equal(model.hfield_data,source.hfield_data)
    for w in range(1,n):
        with pytest.raises(KeyError):model.geom(f'world_{w}/terrain')
        assert not any(model.hfield(h).name.startswith(f'world_{w}/') for h in range(model.nhfield))
    backend=SimpleNamespace(_simulator=sim,_world_id=0)
    sim.data.time=torch.zeros(n)
    sync_grid(backend,model,data)
    for w in range(1,n):
        for src,dst in backend._grid_maps[w][3]:
            name=source.geom(src).name
            assert model.geom_dataid[dst]==model.geom_dataid[model.geom('world_0/'+name).id]
    receiver=TerrainReceiver();receiver._viewer=SimpleNamespace(model=model,vis_options=mujoco.MjvOption())
    receiver.on_terrain(TerrainSnapshot(int(ids[0]),tuple(int(x) for x in ids)))
    visible=(model.geom_type==mujoco.mjtGeom.mjGEOM_HFIELD)&(model.geom_group==0)
    assert visible.sum()==1
    assert model.geom(np.flatnonzero(visible)[0]).name.startswith('world_0/')
    for w in range(n):
        qa=backend._grid_maps[w][0]
        np.testing.assert_allclose(data.qpos[qa:qa+source.nq],sim.data.qpos[w].numpy())
    if terrain_size==100:
        np.testing.assert_array_equal(source.hfield_size[0,:2],[50.,50.])
        assert source.hfield_nrow[0]==source.hfield_ncol[0]==1001
        from large_terrain import half_extent
        assert half_extent(model)==50.
        native_data=mujoco.MjData(source);mujoco.mj_forward(source,native_data)
        for point in ((-44.,-44.),(.3,1.1),(40.,-20.)):
            expected=float(device.ground.height(torch.tensor([point]*n)) [0])
            hit=mujoco.mj_ray(source,native_data,np.array([*point,1000.]),np.array([0.,0.,-1.]),
                None,True,-1,None)
            assert 1000.-hit==pytest.approx(expected,abs=5e-5)
    # Shared static assets must not couple the per-person skin deformation.
    scene=mujoco.MjvScene(model,maxgeom=model.ngeom+100)
    options=mujoco.MjvOption();camera=mujoco.MjvCamera()
    mujoco.mjv_updateScene(model,data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
    before=scene.skinvert.copy().reshape(-1,3)
    sim.data.qpos[3,0]+=.25
    sync_grid(backend,model,data)
    mujoco.mjv_updateScene(model,data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
    delta=scene.skinvert.reshape(-1,3)-before
    expected=np.zeros_like(delta)
    expected[3*single.nskinvert:4*single.nskinvert,0]=.25
    np.testing.assert_allclose(delta,expected,atol=2e-6)
    # Exercise the native classic scene, using an independent CPU model copy.
    # Exactly one rendered hfield must remain after each shared terrain switch.
    for patch in ((0,) if terrain_size==100 else (0,21,63)):
        sim.model.geom_dataid[:,gid]=patch
        sim.model.geom_pos[:,gid,2]=device.ground.zmin[patch]
        sync_grid(backend,model,data)
        import copy
        copied=copy.deepcopy(model)
        copied_data=copy.deepcopy(data)
        mujoco.mj_forward(copied,copied_data)
        options.geomgroup[5]=0
        mujoco.mjv_updateScene(copied,copied_data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        terrains=[g for g in scene.geoms[:scene.ngeom] if g.type==mujoco.mjtGeom.mjGEOM_HFIELD]
        assert len(terrains)==1 and terrains[0].dataid==patch
        assert scene.nlight==single.nlight+1 # native mjv includes one default headlight slot
    if terrain_size==16:
        # A keyboard selection invalidates fixed-first-reset terrain state and
        # changes all collision worlds while keeping the compiled scene/assets.
        identity=id(model);asset_ids=model.geom_dataid.copy()
        device.select_terrain(12);device.reset(sim)
        assert (sim.model.geom_dataid[:,gid]==11).all()
        sync_grid(backend,model,data)
        assert id(model)==identity
        np.testing.assert_array_equal(model.geom_dataid,asset_ids)
        assert np.count_nonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_HFIELD)&(model.geom_group==0))==1
    # Dragging a displayed body maps to its own world only.
    forces=np.zeros((model.nbody,6));body=backend._grid_maps[3][2][0][1];forces[body,0]=7
    routed=grid_forces(backend,forces)
    assert routed[3].sum()==7 and routed.sum()==7
    sim.model.geom_dataid[1,gid]=(sim.model.geom_dataid[0,gid]+1)%64
    with pytest.raises(ValueError,match='collision fields differ'):sync_grid(backend,model,data)
