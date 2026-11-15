from pathlib import Path
from types import SimpleNamespace
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from training_logs import LunarTrainingVecEnv
from moon_task import failed
from reaf.core.termination_checker import TerminationResult


def test_observation_survives_reaf_inplace_step_and_partial_reset():
    wrapper=LunarTrainingVecEnv.__new__(LunarTrainingVecEnv)
    buffer=torch.arange(12,dtype=torch.float32).reshape(3,4)
    timestep=SimpleNamespace(observation={'policy':buffer,'critic':buffer})
    old=wrapper._get_flat_obs(timestep)
    expected=buffer.clone()
    buffer.add_(7)  # REAF overwrites its device output at the next step.
    new=wrapper._get_flat_obs(timestep)
    reset_expected=new['policy'].clone()
    buffer[1]=0  # REAF splices a partial reset into the same buffer.
    for value in old.values():assert torch.equal(value,expected)
    for value in new.values():assert torch.equal(value,reset_expected)


def test_falls_terminate_rather_than_timeout_bootstrap():
    result=failed({'wangyu/failed':torch.tensor([False,True])})
    assert result.tolist()==[0,int(TerminationResult.TERMINATE)]
    assert result[1]!=int(TerminationResult.TRUNCATE)
