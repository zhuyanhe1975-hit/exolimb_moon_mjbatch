"""RSL-RL PPO with CPU rollouts and a separate MPS learner."""
from copy import deepcopy

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class CpuSamplingMpsPPO(PPO):
    """Keep inference and returns on CPU and transfer one rollout per update."""

    def __init__(self, actor, critic, storage, *, device="cpu", **kwargs):
        if (device != "cpu" or actor.is_recurrent or critic.is_recurrent
                or any(kwargs.get(key) for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"))
                or kwargs.get("optimizer", "adam") != "adam"):
            raise ValueError("The CPU/MPS stand trainer requires feedforward, single-process Adam PPO")
        super().__init__(actor, critic, storage, device=device, **kwargs)
        learner_storage = RolloutStorage(
            "rl", storage.num_envs, storage.num_transitions_per_env,
            storage.observations[0], storage.actions_shape, "mps",
        )
        self.learner = PPO(deepcopy(actor), deepcopy(critic), learner_storage,
                           device="mps", **kwargs)
        self.optimizer = torch.optim.Adam(
            self.learner.optimizer.param_groups[0]["params"], lr=self.learning_rate, fused=True,
        )
        self.learner.optimizer = self.optimizer

    def update(self):
        source, target = self.storage, self.learner.storage
        # PPO consumes a complete rollout here, including CPU-computed GAE.
        target.observations.copy_(source.observations, non_blocking=True)
        for field in ("actions", "values", "returns", "advantages", "actions_log_prob"):
            getattr(target, field).copy_(getattr(source, field), non_blocking=True)
        if target.distribution_params is None:
            target.distribution_params = tuple(
                torch.empty_like(tensor, device="mps") for tensor in source.distribution_params)
        for dst, src in zip(target.distribution_params, source.distribution_params, strict=True):
            dst.copy_(src, non_blocking=True)
        loss = self.learner.update()
        # load_state_dict copies both weights and running observation statistics.
        self.actor.load_state_dict(self.learner.actor.state_dict())
        self.critic.load_state_dict(self.learner.critic.state_dict())
        self.learning_rate = self.learner.learning_rate
        source.clear()
        return loss

    def load(self, loaded_dict, load_cfg, strict):
        iteration = super().load(loaded_dict, load_cfg, strict)
        self.learner.actor.load_state_dict(self.actor.state_dict())
        self.learner.critic.load_state_dict(self.critic.state_dict())
        self.learner.learning_rate = self.learning_rate
        return iteration
