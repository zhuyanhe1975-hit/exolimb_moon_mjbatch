"""Static safety/spacing checks on exported geometry; no reset or step loops."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from playback_spawns import distributed_spawns
from local_lunar_terrain import sole_sampling,source_height
from moon_scene import scene_spec


@pytest.mark.parametrize('patch',[0,21,63])
def test_random_spaced_safe_placements_and_partial_reset(patch):
    root=Path(__file__).resolve().parents[1]
    with np.load(root/'assets/playground/state.npz') as z:
        bank={k:z[k].copy() for k in ('normalized','height_range','zmin')}
    foot,cover=sole_sampling(scene_spec().compile())
    result=distributed_spawns(bank,patch,foot,cover,8,np.empty((0,2)),42)
    xy,yaw,padding=result
    repeat=distributed_spawns(bank,patch,foot,cover,8,np.empty((0,2)),42)
    for a,b in zip(result,repeat):np.testing.assert_array_equal(a,b)
    other=distributed_spawns(bank,patch,foot,cover,8,np.empty((0,2)),43)
    assert not np.array_equal(xy,other[0])
    distances=np.linalg.norm(xy[:,None]-xy[None],axis=-1);np.fill_diagonal(distances,np.inf)
    assert distances.min()>3. and np.ptp(xy,axis=0).min()>8.
    assert (abs(xy)<=5.).all()
    c=np.cos(yaw)[:,None];s=np.sin(yaw)[:,None]
    points=np.stack((xy[:,None,0]+c*foot[None,:,0]-s*foot[None,:,1],
                     xy[:,None,1]+s*foot[None,:,0]+c*foot[None,:,1]),-1)
    heights=source_height(points,{'half_extent_m':8.,'height_range_m':bank['height_range'][patch],
        'zmin_m':bank['zmin'][patch]},bank['normalized'][patch])
    root_z=(heights-foot[None,:,2]).max(-1)+.015+padding
    gaps=root_z[:,None]+foot[None,:,2]-heights
    assert gaps.min()>=.0145 and gaps.max()<=.20
    assert abs(points).max()<6.
    # Replacement avoids the people that are still standing elsewhere.
    replacement=distributed_spawns(bank,patch,foot,cover,1,xy[1:],44)[0]
    assert np.linalg.norm(replacement-xy[1:],axis=-1).min()>2.
