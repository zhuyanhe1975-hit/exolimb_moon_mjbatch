"""Evaluate fixed-speed CPU jumps by terrain and by slope relative to heading."""
import argparse
import json
from pathlib import Path
import sys

import torch
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from cpu_stand_env import CpuForwardJumpEnv
from provenance import verify_checkpoint

SLOPES = ('level', 'uphill', 'downhill', 'cross_slope')
METRICS = ('steps', 'distance', 'speed_error_integral', 'threshold_time', 'jumps',
           'stride_sum', 'failures', 'falls', 'timeouts')


def initial_course_slopes(ground, xy, heading, env_ids=None):
    """Classify the local plane using 1 m fore-aft and transverse baselines."""
    lateral = torch.stack((-heading[:, 1], heading[:, 0]), -1)
    points = xy[:, None] + .5 * torch.stack((heading, -heading, lateral, -lateral), dim=1)
    z = ground.height(points, env_ids=env_ids)
    along, across = z[:, 0] - z[:, 1], z[:, 2] - z[:, 3]
    grade = (along.square() + across.square()).sqrt().atan().rad2deg()
    slope = torch.where(across.abs() > along.abs(), 3, torch.where(along > 0, 1, 2))
    return torch.where(grade < 3., 0, slope), grade


class EvaluationEnv(CpuForwardJumpEnv):
    def start_audit(self, speed):
        self.jump_commands.fixed_speed = speed
        self.jump_commands.reset()
        self.speed = speed
        self.last_xy = self.worlds.sim.data.qpos[:, :2].clone()
        self.slope = torch.empty(self.num_envs, dtype=torch.long)
        self.slope_grade = torch.empty(self.num_envs)
        patches = self.worlds.device.ground.patch_ids
        self.patch_ids, self.patch_rows = torch.unique(patches, sorted=True, return_inverse=True)
        self.sums = torch.zeros((len(self.patch_ids) * len(SLOPES), len(METRICS)), dtype=torch.float64)
        self._classify(torch.arange(self.num_envs))
        self.initial_classes = {name: int((self.slope == idx).sum()) for idx, name in enumerate(SLOPES)}
        self.initial_grade_range_deg = [float(self.slope_grade.min()), float(self.slope_grade.max())]

    def _classify(self, ids):
        self.slope[ids], self.slope_grade[ids] = initial_course_slopes(
            self.worlds.device.ground, self.last_xy[ids], self.jump_commands.heading[ids], env_ids=ids)

    def reset_task(self, mask):
        super().reset_task(mask)
        ids = mask.nonzero(as_tuple=True)[0]
        self.last_xy[ids] = self.worlds.sim.data.qpos[ids, :2]
        self._classify(ids)

    def advance_task(self, measured):
        super().advance_task(measured)
        command = self.jump_commands
        data = self.worlds.sim.data
        delta = ((data.qpos[:, :2] - self.last_xy) * command.heading).sum(-1)
        self.last_xy.copy_(data.qpos[:, :2])
        forward = (data.sensordata[:, command.com_velocity_xy] * command.heading).sum(-1)
        error = (forward - self.speed).abs()
        failed = measured['wangyu/failed']
        state = measured['wangyu/state']
        tilt = (state[:, 48] > -.5) | (state[:, 51] > -.5)
        low_pelvis = (data.qpos[:, 2] - self.worlds.device.ground.height(data.qpos[:, :2])
                      < .70 * self.worlds.device.q0[2])
        falls = failed & (tilt | low_pelvis)
        timeout = ((self.episode_length_buf + 1 >= self.max_episode_length)
                   | command.course_finished()) & ~failed
        values = torch.stack((torch.ones_like(delta), delta, error * command.DT,
                              error <= .15, command.completed, command.hop_distance,
                              failed, falls, timeout), dim=-1)
        self.sums.index_add_(0, self.patch_rows * len(SLOPES) + self.slope, values.double())

    def audit(self):
        def summarize(values):
            row = dict(zip(METRICS, values.tolist()))
            seconds = row['steps'] * self.jump_commands.DT
            if not seconds:
                return {'simulated_seconds': 0.}
            mean_speed = row['distance'] / seconds
            return {'simulated_seconds': seconds, 'forward_mean_mps': mean_speed,
                    'mean_speed_error_mps': mean_speed - self.speed,
                    'instantaneous_speed_mae_mps': row['speed_error_integral'] / seconds,
                    'fraction_within_0_15_mps': row['threshold_time'] / row['steps'],
                    'completed_jumps': int(row['jumps']),
                    'jumps_per_minute': 60 * row['jumps'] / seconds,
                    'mean_jump_stride_m': row['stride_sum'] / row['jumps'] if row['jumps'] else None,
                    'failures': int(row['failures']), 'falls': int(row['falls']),
                    'falls_per_minute': 60 * row['falls'] / seconds, 'timeouts': int(row['timeouts'])}

        grid = self.sums.reshape(len(self.patch_ids), len(SLOPES), len(METRICS))
        report = {
            'target_speed_mps': self.speed, 'initial_slope_counts': self.initial_classes,
            'initial_slope_magnitude_range_deg': self.initial_grade_range_deg,
            'overall': summarize(grid.sum((0, 1))),
            'by_slope': {name: summarize(grid[:, i].sum(0)) for i, name in enumerate(SLOPES)},
            'by_terrain': {int(patch) + 1: {
                'overall': summarize(grid[i].sum(0)),
                'by_slope': {name: summarize(grid[i, k]) for k, name in enumerate(SLOPES)},
            } for i, patch in enumerate(self.patch_ids)},
        }
        if self.worlds.exoskeleton is not None:
            report['exoskeleton'] = self.worlds.exoskeleton.report()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--terrain-idxs', type=int, nargs='+', required=True)
    parser.add_argument('--assist-force', type=float, metavar='NEWTONS',
                        help='mount the exoskeleton and use this limit per strut; 0 is the unpowered control')
    parser.add_argument('--speeds', type=float, nargs='+', default=[0., .5, 1., 1.5])
    parser.add_argument('--worlds-per-terrain', type=int, default=16)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 1 or args.worlds_per_terrain < 1:
        parser.error('seconds and worlds per terrain must be positive')
    if any(not 0 <= speed <= 1.5 for speed in args.speeds):
        parser.error('speeds must be within 0–1.5 m/s')
    torch.set_num_threads(1)
    verify_checkpoint(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    info = state['infos']
    if info.get('jump_actuation') != EvaluationEnv.actuation:
        parser.error('checkpoint does not use the current forward-jump controller')
    assist_force = args.assist_force
    exo_policy_io = None
    if assist_force is None and info.get('task') == 'exo_jump':
        assist_force = info['assist_force_n']
    if assist_force is not None:
        from exoskeleton import validate_assist_force, validate_policy_io
        try:
            assist_force = validate_assist_force(assist_force)
            exo_policy_io = validate_policy_io(info.get('exo_policy_io'))
        except ValueError as exc:
            parser.error(str(exc))
    count = args.worlds_per_terrain * len(args.terrain_idxs)
    reports = []
    for speed in args.speeds:
        torch.manual_seed(args.seed)
        env = EvaluationEnv(count, args.terrain_idxs[0], False, 'mjbatch', min(12, count), args.seed,
                            geometry=info['geometry'], terrain_indices=args.terrain_idxs, assist_force=assist_force,
                            exo_policy_io=exo_policy_io)
        env.start_audit(speed)
        with torch.inference_mode(), env:
            obs = env.get_observations()
            actor = MLPModel(obs, {'actor': ['policy']}, 'actor', env.num_actions,
                             hidden_dims=(128, 128), activation='elu', obs_normalization=True,
                             distribution_cfg={'class_name': 'rsl_rl.modules.GaussianDistribution',
                                               'init_std': .25, 'std_type': 'scalar'})
            actor.load_state_dict(state['actor_state_dict'], strict=True)
            actor.eval()
            for _ in range(round(args.seconds / env.jump_commands.DT)):
                obs, _, _, _ = env.step(actor(obs))
        reports.append(env.audit())
        print(f'{speed:g} m/s across {len(args.terrain_idxs)} terrains: {reports[-1]["overall"]}', flush=True)
    report = {'checkpoint': str(args.checkpoint), 'terrain_indices': args.terrain_idxs, 'seed': args.seed,
              'worlds_per_terrain': args.worlds_per_terrain, 'seconds_per_world': args.seconds,
              'pushes': False, 'assist_force_n': assist_force, 'coarse_slope_threshold_deg': 3,
              'slope_attribution': 'At episode spawn, using two perpendicular 1 m baselines. '
                                   'Steeper lateral component is cross-slope, else positive or negative forward grade.',
              'speeds': reports}
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
