"""CPU exact parity of combined native-triangle terrain queries."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from local_lunar_terrain import HALF, LocalSurface
from measurement_terrain import measurement_terrain


def test_batched_queries_match_separate_sensor_and_qpos_roots():
    # Use real exported terrain, without compiling a simulator or allocating CUDA.
    ground=LocalSurface.__new__(LocalSurface)
    with np.load(ROOT/'assets/playground/state.npz') as bank:
        ground.grid=torch.tensor(bank['normalized'],dtype=torch.float32)
        ground.zmin=torch.tensor(bank['zmin'],dtype=torch.float32)
        ground.height_range=torch.tensor(bank['height_range'],dtype=torch.float32)
    ground.half=HALF
    ground.step=2*ground.half/(ground.grid.shape[1]-1)
    n=129;generator=torch.Generator().manual_seed(1002)
    ground.patch_ids=torch.randint(64,(n,),generator=generator)
    coordinates=torch.linspace(-.5,.5,5)
    ground.scan_points=torch.cartesian_prod(coordinates,coordinates)
    root=torch.randn(n,7,generator=generator)
    root[:,:2]*=5 # Includes queries outside the field, exercising clamping.
    root[:,3:7]/=root[:,3:7].norm(dim=-1,keepdim=True)
    left=root[:,:2]+torch.randn(n,2,generator=generator)
    right=root[:,:2]+torch.randn(n,2,generator=generator)
    qpos=root[:,:2]+.65 # Deliberately different from the native sensor position.
    reference=(ground.height(torch.stack((root[:,:2],left,right),1)),
               ground.scan(root),ground.height(qpos))
    queried=[];original=ground.height
    def count(xy):
        queried.append(xy.shape)
        return original(xy)
    ground.height=count
    actual=measurement_terrain(ground,root,left,right,qpos)
    assert queried==[torch.Size([n,29,2])]
    for expected,result in zip(reference,actual):
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
    assert not torch.equal(actual[0][:,0],actual[2])
