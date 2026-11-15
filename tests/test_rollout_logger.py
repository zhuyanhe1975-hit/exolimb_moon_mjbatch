"""Compare CPU logger results with upstream, including delayed episode flush."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rsl_rl.utils.logger import Logger
from rollout_logger import install_rollout_logger


class Writer:
    def __init__(self):self.values={}
    def add_scalar(self,key,value,it):self.values[(key,it)]=float(value)


def make_logger(rnd=False):
    logger=Logger(None,{'algorithm':{'rnd_cfg':rnd},'num_steps_per_env':4},None,64,False,1,0,'cpu')
    logger.writer=Writer()
    return logger


def log(logger,it):
    logger.log(it=it,start_it=0,total_it=4,collect_time=.1,learn_time=.02,
        loss_dict={'value':.2},learning_rate=.001,action_std=torch.tensor([.25]),rnd_weight=None)


def test_rollout_logger_window_extras_and_no_done_match_upstream(capsys):
    eager=make_logger()
    runner=SimpleNamespace(logger=make_logger())
    writer=runner.logger.writer
    fast=install_rollout_logger(runner)
    assert fast.writer is writer and install_rollout_logger(runner) is fast
    for it in range(4):
        for step in range(4):
            rewards=torch.arange(64,dtype=torch.float32)*.01 + (it+step)*.2
            # 128+ endings exercises the latest-100 selection. Entirely empty
            # rollout before and after must preserve previous windows.
            dones=torch.zeros(64,dtype=torch.long)
            if it in (1,3):
                dones[((torch.arange(64)+step)%2)==0]=1
            extras={'log':{'step/value':torch.tensor(float(step)),
                'step/double':torch.tensor(step*.5,dtype=torch.float64)}}
            if step in (0,3):extras['log']['episode/sparse']=torch.tensor([float(step*2)])
            eager.process_env_step(rewards,dones,extras)
            fast.process_env_step(rewards,dones,extras)
            torch.testing.assert_close(fast.cur_reward_sum,eager.cur_reward_sum)
            torch.testing.assert_close(fast.cur_episode_length,eager.cur_episode_length)
        log(eager,it);log(fast,it)
        assert list(fast.rewbuffer)==list(eager.rewbuffer)
        assert list(fast.lenbuffer)==list(eager.lenbuffer)
        assert len(fast.rewbuffer)<=100
        assert fast._rollout_index==0 and fast.ep_extras==[]
        for key,value in eager.writer.values.items():
            assert fast.writer.values[key]==pytest.approx(value)
        if it>0:
            assert fast.writer.values[('Perf/wall_cycle_time',it)]>0
            assert fast.writer.values[('Perf/wall_cycle_fps',it)]>0
    assert ('Perf/wall_cycle_time',0) not in fast.writer.values
    capsys.readouterr()


def test_rollout_logger_rnd_falls_back_without_delaying_windows():
    eager=make_logger(rnd=True)
    fast=install_rollout_logger(SimpleNamespace(logger=make_logger(rnd=True)))
    rewards=torch.ones(64);dones=torch.ones(64);intrinsic=torch.full((64,),.3)
    for logger in (eager,fast):logger.process_env_step(rewards,dones,{},intrinsic)
    for name in ('rewbuffer','lenbuffer','erewbuffer','irewbuffer'):
        assert list(getattr(fast,name))==list(getattr(eager,name))


def test_logging_disabled_does_not_accumulate():
    logger=make_logger();logger.writer=None
    fast=install_rollout_logger(SimpleNamespace(logger=logger))
    fast.process_env_step(torch.ones(64),torch.ones(64),{})
    assert fast._rollout_index==0 and not fast.rewbuffer
