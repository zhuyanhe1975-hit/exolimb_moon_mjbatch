"""Give reward logs explicit meanings without changing the RL transition."""
from mujoco_playground._src.rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
import dm_env
import tensordict
import torch


def clarify_reward_logs(log):
    """Move raw sums aside; the reward namespace contains weighted contributions."""
    renamed = {}
    for key, value in log.items():
        for period in ('episode', 'step'):
            prefix = f'reward_{period}/'
            if key.startswith(prefix):
                term = key[len(prefix):]
                if term.startswith('weighted/'):
                    key = prefix + term[len('weighted/'):]
                else:
                    key = f'raw_{period}/' + term
                break
        if key in renamed:
            raise ValueError(f'Duplicate training log key: {key}')
        renamed[key] = value
    return renamed


class LunarTrainingVecEnv(RslRlVecEnvWrapper):
    def _get_flat_obs(self,timestep):
        # REAF's single-term observation groups can alias its reusable device
        # output buffers. PPO stores this state only AFTER env.step(), so the
        # returned state must own storage across subsequent steps and resets.
        observations=super()._get_flat_obs(timestep)
        return {key:value.clone() for key,value in observations.items()}

    def __init__(self, *args, batched_logs=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batched_logs = batched_logs
        self._reward_keys = None
        self._batched_episode_sums = None

    def step(self, actions):
        if self.batched_logs:
            obs, reward, dones, extras = self._batched_step(actions)
        else:
            obs, reward, dones, extras = super().step(actions)
        extras['log'] = clarify_reward_logs(extras['log'])
        return obs, reward, dones, extras

    def _batched_step(self, actions):
        """Same transition/reset as upstream, with fixed-shape reward reductions.

        Upstream gathers the completed worlds separately for every reward term.
        Boolean gathers on CUDA require a dynamic output size and synchronize.
        Reducing a masked term-by-world tensor avoids these repeated gathers.
        The single done check and all native reset/metric handling are retained.
        """
        if self.clip_actions is not None:
            raise ValueError('clip_actions is not supported.')
        action_key = next(iter(self.action_spec()))
        self._timestep = self.env.step({action_key: actions})
        rew_dict = self._timestep.reward
        if not isinstance(rew_dict, dict):
            raise TypeError('Lunar training requires named REAF reward terms')
        reward = rew_dict['reward']
        terms = rew_dict['terms']
        self._episode_length_buf += 1
        is_last = self._timestep.step_type == dm_env.StepType.LAST
        terminated = is_last & (self._timestep.discount == 0)
        truncated = is_last & (self._timestep.discount > 0)
        done_mask = terminated | truncated
        dones = done_mask.to(dtype=torch.long)

        keys = tuple(terms)
        if self._reward_keys is None:
            if not keys:
                raise ValueError('Lunar training requires nonempty reward terms')
            self._reward_keys = keys
        elif keys != self._reward_keys:
            raise ValueError('Reward term order changed during training')
        values = torch.stack(tuple(terms.values()))
        if self._batched_episode_sums is None:
            self._batched_episode_sums = torch.zeros_like(values)
        self._batched_episode_sums.add_(values)
        step_means = values.mean(dim=1)
        extras = {'step_type': self._timestep.step_type,
                  'reward_terms_per_world': terms,
                  'log': dict(zip(('reward_step/' + key for key in keys),step_means.unbind()))}

        if done_mask.any():
            for key, value in self.env.task_layer.get_episode_metrics(done_mask).items():
                extras['log']['Metrics/' + key] = value
            episode_means = torch.where(done_mask[None],self._batched_episode_sums,0.).sum(dim=1) / done_mask.sum()
            extras['log'].update(zip(('reward_episode/' + key for key in keys),episode_means.unbind()))
            self._batched_episode_sums.masked_fill_(done_mask[None],0.)
            self._episode_length_buf.masked_fill_(done_mask,0)
            self._timestep = self.env.reset_with_options(
                options=self.env.default_reset_options(),env_mask=done_mask)
            extras['step_type'] = self._timestep.step_type

        obs_dict = self._get_flat_obs(self._timestep)
        if self.cfg is not None and not self.cfg.is_finite_horizon:
            extras['time_outs'] = truncated
        return tensordict.TensorDict(obs_dict,batch_size=[self.num_envs]),reward,dones,extras
