"""Independent display compilation and native geometry equivalence, no stepping."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_scene import scene_spec
from moon_viewer import display_spec

def test_display_terrain_preserves_physics_and_native_surface():
    spec=scene_spec();source=spec.compile()
    original=source.geom_bodyid.copy();heights=source.hfield_data.copy()
    display=display_spec(SimpleNamespace(mj_spec=spec)).compile()
    np.testing.assert_array_equal(source.geom_bodyid,original)
    np.testing.assert_array_equal(source.hfield_data,heights)
    assert source.geom_bodyid[source.geom('terrain').id]!=0
    assert display.geom_bodyid[display.geom('terrain').id]==0
    assert (display.nq,display.nv,display.nbody,display.nmocap)==(source.nq,source.nv,source.nbody,source.nmocap)
    source_data=mujoco.MjData(source);display_data=mujoco.MjData(display)
    robot_pairs=[(i,source.geom(display.geom(i).name).id) for i in range(display.ngeom) if display.geom(i).name.startswith('robot/')]
    with np.load(Path(__file__).resolve().parents[1]/'assets/playground/state.npz') as archive:
        zmin=archive['zmin'].copy()
    mask=np.array([0,0,0,0,0,1],dtype=np.uint8)
    for patch in (0,21,63):
        gid=source.geom('terrain').id
        source.geom_dataid[gid]=source.hfield(f'site20_local_{patch:02d}').id
        source.geom_pos[gid,2]=zmin[patch];source.geom_group[gid]=5
        for index in range(64):
            dst=display.geom('terrain' if index==0 else f'display_terrain_{index:02d}').id
            display.geom_rgba[dst,3]=float(index==patch);display.geom_group[dst]=5
        active=display.geom('terrain' if patch==0 else f'display_terrain_{patch:02d}').id
        mujoco.mj_forward(source,source_data);mujoco.mj_forward(display,display_data)
        for dst,src in robot_pairs:
            np.testing.assert_allclose(display_data.geom_xpos[dst],source_data.geom_xpos[src],atol=1e-12)
            np.testing.assert_allclose(display_data.geom_xmat[dst],source_data.geom_xmat[src],atol=1e-12)
        np.testing.assert_allclose(display_data.geom_xpos[active],source_data.geom_xpos[gid],atol=1e-12)
        for xy in ((.12,.23),(-.45,.67),(1.1,-.3)):
            ray=np.array([*xy,10.]);direction=np.array([0.,0.,-1.])
            a=mujoco.mj_ray(source,source_data,ray,direction,mask,True,-1,None)
            b=mujoco.mj_ray(display,display_data,ray,direction,mask,True,-1,None)
            assert a>0 and b>0
            assert abs(a-b)<1e-10
