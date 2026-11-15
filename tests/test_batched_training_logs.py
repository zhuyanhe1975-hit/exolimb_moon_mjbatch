"""CPU differential tests against the installed upstream transition wrapper."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
from dm_env import specs

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from training_logs import LunarTrainingVecEnv


class FakeEnv:
    nworld = 4
    xpu_device = 'cpu'

    def __init__(self):
        self.device_layer = SimpleNamespace(control_timestep=1/60)
        self.task_layer = SimpleNamespace(get_episode_metrics=self.metrics)
        self.index = 0
        self.reset_masks = []
        self.metric_masks = []

    def action_spec(self):
        return {'wangyu/action': specs.Array((23,),np.float32)}

    def reset(self):
        self.observation = torch.zeros(4,3)
        return SimpleNamespace(observation={'policy': self.observation.clone()})

    def step(self, actions):
        assert tuple(actions) == ('wangyu/action',)
        self.index += 1
        masks = ([False]*4,[True,False,True,False],[False]*4,[False,True,True,True],[True]*4)
        done = torch.tensor(masks[(self.index-1)%len(masks)])
        # Even-numbered worlds terminate; odd-numbered worlds time out.
        discount = torch.tensor([0.,1.,0.,1.])
        steps = torch.where(done,2,1)
        self.observation = torch.arange(12,dtype=torch.float32).reshape(4,3) + self.index
        terms = {}
        for i in range(9):
            value = (torch.arange(4,dtype=torch.float32)+1)*(i+1)*self.index
            terms['term'+str(i)] = value
            terms['weighted/term'+str(i)] = value*(-.01)*(1/60)
        reward = torch.stack([v for k,v in terms.items() if k.startswith('weighted/')]).sum(0)
        return SimpleNamespace(observation={'policy': self.observation.clone()},
            step_type=steps,discount=discount,reward={'reward':reward,'terms':terms})

    def metrics(self, mask):
        self.metric_masks.append(mask.clone())
        return {'test_metric': mask.float().sum()+self.index}

    def default_reset_options(self):
        return {'test': 1}

    def reset_with_options(self, *, options, env_mask):
        assert options == {'test':1}
        self.reset_masks.append(env_mask.clone())
        self.observation[env_mask] = -self.index
        return SimpleNamespace(observation={'policy':self.observation.clone()},
            step_type=torch.where(env_mask,0,1))


@pytest.mark.parametrize('finite_horizon',[False,True])
def test_batched_logs_preserve_upstream_transitions_and_episode_means(finite_horizon):
    configs = SimpleNamespace(is_finite_horizon=finite_horizon)
    reference = LunarTrainingVecEnv(FakeEnv(),cfg=configs,max_episode_length=600,batched_logs=False)
    batched = LunarTrainingVecEnv(FakeEnv(),cfg=configs,max_episode_length=600,batched_logs=True)
    retained = []
    for _ in range(12):
        action = torch.zeros(4,23)
        expected = reference.step(action)
        actual = batched.step(action)
        torch.testing.assert_close(actual[0]['policy'],expected[0]['policy'])
        torch.testing.assert_close(actual[1],expected[1])
        torch.testing.assert_close(actual[2],expected[2])
        assert actual[3].keys() == expected[3].keys()
        for key in ('log','reward_terms_per_world'):
            assert actual[3][key].keys() == expected[3][key].keys()
            for name,value in expected[3][key].items():
                torch.testing.assert_close(actual[3][key][name],value)
        torch.testing.assert_close(actual[3]['step_type'],expected[3]['step_type'])
        if not finite_horizon:
            torch.testing.assert_close(actual[3]['time_outs'],expected[3]['time_outs'])
        torch.testing.assert_close(batched.episode_length_buf,reference.episode_length_buf)
        for key,row in zip(batched._reward_keys,batched._batched_episode_sums):
            torch.testing.assert_close(row,reference._episode_sums[key])
        for value,saved in retained:
            torch.testing.assert_close(value,saved)
        retained.extend((value,value.clone()) for value in actual[3]['log'].values())
    for attribute in ('reset_masks','metric_masks'):
        first=getattr(reference.env,attribute);second=getattr(batched.env,attribute)
        assert len(first)==len(second)
        for a,b in zip(first,second):assert torch.equal(a,b)
