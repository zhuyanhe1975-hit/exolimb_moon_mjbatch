"""CPU reset equivalence; no dynamic simulation, CUDA or training."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_device import MoonDevice
from moon_scene import scene_spec


def _make_device(model,n):
    sim=SimpleNamespace(mj_model=model,
        model=SimpleNamespace(geom_dataid=torch.tensor(np.repeat(model.geom_dataid[None],n,0)),
                              geom_pos=torch.tensor(np.repeat(model.geom_pos[None],n,0))),
        data=SimpleNamespace(qpos=torch.tensor(np.repeat(model.qpos0[None],n,0),dtype=torch.float32),
                             qvel=torch.zeros(n,model.nv),ctrl=torch.zeros(n,23),
                             xfrc_applied=torch.zeros(n,model.nbody,6)))
    dev=MoonDevice(pushes=True,push_force=30.,push_torque=50.)
    dev.initialize(sim)
    return dev,sim


def _state(dev,sim):
    return {**{name:value.clone() for name,value in vars(sim.data).items()},
            **{'model/'+name:value.clone() for name,value in vars(sim.model).items()},
            **{'device/'+name:getattr(dev,name).clone() for name in
               ('action','previous','pending','target','heading','wait','remaining','wrench')},
            'patch_ids':dev.ground.patch_ids.clone(),'rng':torch.get_rng_state().clone()}


def test_sparse_reset_matches_full_batch_rng_and_state(monkeypatch):
    model=scene_spec().compile();n=11
    masks=[None,torch.arange(n)%3==0,torch.zeros(n,dtype=torch.bool),
           torch.ones(n,dtype=torch.bool),torch.arange(n)==n-1]
    results=[];counts=[]
    for mode in ('0','1'):
        monkeypatch.setenv('MOON_SPARSE_RESET',mode)
        dev,sim=_make_device(model,n)
        original=dev.ground.height;queries=[]
        def counted(xy,env_ids=None):
            queries.append(xy.shape[0])
            return original(xy,env_ids=env_ids)
        dev.ground.height=counted
        torch.manual_seed(5142);states=[]
        for mask in masks:
            # Exercise nonzero state that must survive unselected resets.
            sim.data.qvel.add_(.03);dev.wait.zero_()
            dev.set_substep_commands(sim,{'wangyu/action':torch.rand(n,23)})
            dev.on_step(sim)
            before=_state(dev,sim)
            dev.reset(sim,mask)
            after=_state(dev,sim);states.append(after)
            if mask is not None:
                for key in before.keys()-{'rng'}:
                    torch.testing.assert_close(after[key][~mask],before[key][~mask],rtol=0,atol=0)
        results.append(states);counts.append(queries)
    for reference,actual in zip(*results):
        for key in reference:
            torch.testing.assert_close(actual[key],reference[key],rtol=0,atol=0,msg=key)
    assert counts[0]==[n]*len(masks)
    assert counts[1]==[n,4,0,n,1]


@pytest.mark.parametrize('mode',['0','1'])
def test_sparse_reset_keeps_whole_sole_guard(monkeypatch,mode):
    monkeypatch.setenv('MOON_SPARSE_RESET',mode)
    dev,sim=_make_device(scene_spec().compile(),3)
    dev.ground.clearance_padding.fill_(1.)
    with pytest.raises(ValueError,match='whole-sole'):
        dev.reset(sim,torch.tensor([False,True,False]))


def test_playback_keeps_terrain_while_training_resamples():
    model=scene_spec().compile();dev,sim=_make_device(model,11)
    dev.resample_terrain=False
    dev.reset(sim);patches=dev.ground.patch_ids.clone();positions=sim.data.qpos.clone()
    for _ in range(3):
        dev.reset(sim)
        assert torch.equal(patches,dev.ground.patch_ids)
    assert not torch.equal(positions,sim.data.qpos)
    dev.resample_terrain=True
    dev.reset(sim)
    assert not torch.equal(patches,dev.ground.patch_ids)
