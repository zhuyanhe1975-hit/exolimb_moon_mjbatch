"""Regression check for CPU rollout and learner synchronization."""
import math
from pathlib import Path
import sys
import unittest

from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cpu_sampling_ppo import CpuSamplingMpsPPO


@unittest.skipUnless(torch.backends.mps.is_available(), 'requires Apple MPS')
class CpuSamplingMpsTest(unittest.TestCase):
    def test_learner_updates_cpu_rollout_models(self):
        obs = TensorDict({'state': torch.arange(12).float().reshape(4, 3) / 10}, batch_size=[4])
        groups = {'actor': ['state'], 'critic': ['state']}
        actor = MLPModel(
            obs, groups, 'actor', 2, hidden_dims=(8,), obs_normalization=True,
            distribution_cfg={'class_name': 'rsl_rl.modules.GaussianDistribution',
                              'init_std': .25, 'std_type': 'scalar'},
        )
        critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=(8,), obs_normalization=True)
        storage = RolloutStorage('rl', 4, 2, obs, [2], 'cpu')
        ppo = CpuSamplingMpsPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1)
        initial_weight = next(actor.mlp.parameters()).detach().clone()

        for _ in range(2):
            with torch.inference_mode():
                for _ in range(2):
                    actions = ppo.act(obs)
                    self.assertEqual(actions.device.type, 'cpu')
                    rewards = torch.arange(4).float()
                    done = torch.tensor([False, False, False, True])
                    ppo.process_env_step(obs, rewards, done, {'time_outs': done})
                ppo.compute_returns(obs)
            self.assertTrue(all(math.isfinite(value) for value in ppo.update().values()))
            self.assertEqual(storage.step, 0)
            for sampler, learner in ((actor, ppo.learner.actor), (critic, ppo.learner.critic)):
                for key, tensor in sampler.state_dict().items():
                    self.assertEqual(tensor.device.type, 'cpu')
                    torch.testing.assert_close(tensor, learner.state_dict()[key].cpu(), rtol=0, atol=0)
            self.assertEqual(next(iter(ppo.optimizer.state.values()))['exp_avg'].device.type, 'mps')
        self.assertFalse(torch.equal(initial_weight, next(actor.mlp.parameters())))


if __name__ == '__main__':
    unittest.main()
