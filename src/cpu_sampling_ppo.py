"""RSL-RL PPO with CPU rollouts and a separate accelerator learner."""
from copy import deepcopy

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


class CpuSamplingAcceleratedPPO(PPO):
    """Keep inference and returns on CPU and transfer one rollout per update."""

    def __init__(self, actor, critic, storage, *, learner_device, device="cpu", **kwargs):
        if (device != "cpu" or actor.is_recurrent or critic.is_recurrent
                or any(kwargs.get(key) for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"))
                or kwargs.get("optimizer", "adam") != "adam"):
            raise ValueError("CPU sampling with an accelerator requires feedforward, single-process Adam PPO")
        super().__init__(actor, critic, storage, device=device, **kwargs)
        self.learner_device = torch.device(learner_device)
        learner_storage = RolloutStorage(
            "rl", storage.num_envs, storage.num_transitions_per_env,
            storage.observations[0], storage.actions_shape, learner_device,
        )
        self.learner = PPO(deepcopy(actor), deepcopy(critic), learner_storage,
                           device=learner_device, **kwargs)
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
                torch.empty_like(tensor, device=self.learner_device) for tensor in source.distribution_params)
        for dst, src in zip(target.distribution_params, source.distribution_params, strict=True):
            dst.copy_(src, non_blocking=True)
        model_pairs = ((self.actor, self.learner.actor), (self.critic, self.learner.critic))
        for sampler, learner in model_pairs:
            learner.obs_normalizer.load_state_dict(sampler.obs_normalizer.state_dict())
        loss = self.learner.update()
        # RSL-RL replaces the sampler's _std buffer under inference_mode during rollout.
        with torch.inference_mode():
            for sampler, learner in model_pairs:
                sampler.load_state_dict(learner.state_dict())
        self.learning_rate = self.learner.learning_rate
        source.clear()
        return loss

    def load(self, loaded_dict, load_cfg, strict):
        iteration = super().load(loaded_dict, load_cfg, strict)
        self.learner.actor.load_state_dict(self.actor.state_dict())
        self.learner.critic.load_state_dict(self.critic.state_dict())
        self.learner.learning_rate = self.learning_rate
        return iteration


class CpuSamplingMpsPPO(CpuSamplingAcceleratedPPO):
    def __init__(self, actor, critic, storage, **kwargs):
        super().__init__(actor, critic, storage, learner_device="mps", **kwargs)
