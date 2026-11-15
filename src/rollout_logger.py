"""Batch episode bookkeeping at rollout boundaries without changing PPO."""
from time import perf_counter
import torch
from rsl_rl.utils.logger import Logger


class RolloutLogger(Logger):
    """Keep the upstream writer/format and last-100-episode window semantics.

    Endings are ordered by control step, then ascending world ID, just as in
    upstream. RND retains upstream bookkeeping. This task's extras are scalars;
    unsupported vector extras raise rather than silently changing averaging.
    """

    def initialize_rollout_buffers(self):
        self._batch_enabled = not bool(self.cfg['algorithm']['rnd_cfg'])
        self._rollout_index = 0
        self._wall_boundary = None
        steps = self.cfg['num_steps_per_env']
        self._ended_values = torch.empty((steps,self.num_envs,2),device=self.device,
            dtype=self.cur_reward_sum.dtype)
        self._ended_masks = torch.empty((steps,self.num_envs),device=self.device,dtype=torch.bool)

    def process_env_step(self,rewards,dones,extras,intrinsic_rewards=None):
        if not self._batch_enabled:
            return super().process_env_step(rewards,dones,extras,intrinsic_rewards)
        if self.writer is None:
            return
        if intrinsic_rewards is not None:
            raise ValueError('Unexpected intrinsic rewards in non-RND rollout logger')
        if self._rollout_index >= len(self._ended_masks):
            raise RuntimeError('Rollout logger requires log() after each configured rollout')
        if 'episode' in extras:
            self.ep_extras.append(extras['episode'])
        elif 'log' in extras:
            self.ep_extras.append(extras['log'])
        self.cur_reward_sum.add_(rewards)
        self.cur_episode_length.add_(1)
        mask = dones > 0
        row = self._rollout_index
        self._ended_values[row,:,0].copy_(self.cur_reward_sum)
        self._ended_values[row,:,1].copy_(self.cur_episode_length)
        self._ended_masks[row].copy_(mask)
        self.cur_reward_sum.masked_fill_(mask,0)
        self.cur_episode_length.masked_fill_(mask,0)
        self._rollout_index += 1

    def _flush_endings(self):
        if not self._rollout_index:
            return
        # A single dynamic selection and host transfer per rollout. Earlier
        # completed episodes cannot survive the upstream maxlen=100 deques.
        count = self._rollout_index
        last = max(self.rewbuffer.maxlen,self.lenbuffer.maxlen)
        ids = self._ended_masks[:count].reshape(-1).nonzero(as_tuple=False).flatten()[-last:]
        rows = self._ended_values[:count].reshape(-1,2).index_select(0,ids).cpu().tolist()
        self.rewbuffer.extend(row[0] for row in rows)
        self.lenbuffer.extend(row[1] for row in rows)
        self._rollout_index = 0

    def _reduce_extras(self):
        if not self.ep_extras:
            return
        by_key = {}
        for extra in self.ep_extras:
            for key,value in extra.items():
                value = torch.as_tensor(value,device=self.device)
                if value.numel() != 1:
                    raise ValueError(f'RolloutLogger requires scalar extras: {key}')
                by_key.setdefault(key,[]).append(value.reshape(1))
        # Missing episode keys are excluded, preserving upstream weighting.
        self.ep_extras = [{key:torch.cat([torch.empty(0,device=self.device),*values]).mean()
            for key,values in by_key.items()}]

    def log(self,*args,**kwargs):
        if self.writer is not None and self._batch_enabled:
            self._flush_endings()
            self._reduce_extras()
        super().log(*args,**kwargs)
        completed = perf_counter()
        if self.writer is not None and self._wall_boundary is not None:
            elapsed = completed-self._wall_boundary
            it = kwargs['it'] if 'it' in kwargs else args[0]
            samples = self.cfg['num_steps_per_env']*self.num_envs*self.gpu_world_size
            # End-of-log to end-of-log: includes previous checkpoint time,
            # collection, learning and current logging; first cycle omitted.
            self.writer.add_scalar('Perf/wall_cycle_time',elapsed,it)
            self.writer.add_scalar('Perf/wall_cycle_fps',samples/elapsed,it)
        self._wall_boundary = completed


def install_rollout_logger(runner,enabled=True):
    """Replace this runner's logger only; retain writer/state and shared libraries."""
    if isinstance(runner.logger,RolloutLogger):
        return runner.logger
    if type(runner.logger) is not Logger:
        raise TypeError('Expected the upstream RSL Logger')
    logger = RolloutLogger.__new__(RolloutLogger)
    logger.__dict__.update(runner.logger.__dict__)
    logger.initialize_rollout_buffers()
    logger._batch_enabled = logger._batch_enabled and enabled
    runner.logger = logger
    return logger
