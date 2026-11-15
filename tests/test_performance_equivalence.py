"""CPU semantic checks for the reduced feature path; no simulation or CUDA."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from balance_features import features,SENSOR_DIM,STATE_DIM


def test_standing_features_match_consumed_legacy_columns():
    generator=torch.Generator().manual_seed(137)
    x=torch.randn(67,STATE_DIM+SENSOR_DIM+30,generator=generator)
    # Include no-load, unilateral support, and positive load worlds.
    for offset in (STATE_DIM+35,STATE_DIM+42):
        x[:,offset]=torch.linspace(-100.,400.,len(x))
    reference=features(x,180.,1.635)
    state,penalties=features(x,180.,1.635,standing=True)
    assert state.shape==(67,143)
    torch.testing.assert_close(state,torch.cat((reference[:,:95],reference[:,97:145]),-1),rtol=0,atol=0)
    torch.testing.assert_close(penalties,reference[:,[146,147,148,157]],rtol=0,atol=0)
    saved=state.clone()
    features(x+1,180.,1.635,standing=True)
    assert torch.equal(saved,state)


def test_batched_contacts_match_per_foot_path():
    contact=torch.randn(67,2,4,8,generator=torch.Generator().manual_seed(3))
    contact[0,0,0,5]=float('nan')
    per_foot=[]
    for side in range(2):
        foot=contact[:,side]
        valid=(foot[:,:,0]>0)&(foot[:,:,1:4].norm(dim=-1)>.5)&(foot[:,:,4]<=.002)&torch.isfinite(foot[:,:,5:8]).all(-1)
        per_foot.append(valid.any(-1))
    valid=(contact[...,0]>0)&(contact[...,1:4].norm(dim=-1)>.5)&(contact[...,4]<=.002)&torch.isfinite(contact[...,5:8]).all(-1)
    assert torch.equal(~valid.any(dim=(1,2)),~(per_foot[0]|per_foot[1]))
