"""Check forward commands and inheritance of the original jumping policy."""
from math import sqrt
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cpu_forward_jump import (AbsorbingHopForwardJumpCommands, BilateralSoleLandingCommands,
                              BroadJumpForwardCommands,
                              CoordinatedBroadJumpCommands, CoordinatedLandingCommands,
                              CoordinatedTakeoffCommands,
                              DistributedSoleForwardJumpCommands,
                              ForwardFacingBootCommands, ForwardJumpCommands, FreeHeightForwardJumpCommands,
                              FullSoleForwardJumpCommands,
                              HighSpeedLeanForwardJumpCommands, LandingPreparationCommands,
                              LongStrideForwardJumpCommands, MirroredLandingCommands,
                              PhaseMirroredLandingCommands, SolePlaneMirroredLandingCommands,
                              SpeedTrackingForwardJumpCommands, StanceFlowTakeoffCommands,
                              StanceStrideTakeoffCommands,
                              SupportedTakeoffCommands, TerrainAlignedLandingCommands,
                              forward_initial_state, long_hop_event_weight)
from cpu_jump_task import REWARD_COLUMNS
from cpu_simulation import CpuWorlds
from moon_scene import scene_spec


class ForwardJumpTaskTest(unittest.TestCase):
    def test_default_commands_use_zero_to_one_point_five_meters_per_second(self):
        worlds = CpuWorlds(scene_spec(), 4, 12)
        commands = FreeHeightForwardJumpCommands(worlds)
        self.assertEqual(commands.speed_range, (0., 1.5))
        keep = commands.target_speed[1:].clone()
        selected = torch.tensor([True, False, False, False])
        commands.speed_generator.manual_seed(42)
        samples = []
        for _ in range(6):
            commands.reset(selected)
            samples.append(float(commands.target_speed[0]))
            torch.testing.assert_close(commands.target_speed[1:], keep)
        self.assertTrue(all(0. <= speed <= 1.5 for speed in samples))
        self.assertGreater(max(samples), 1.)

    def test_free_height_counts_landings_independently_of_apex(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = FreeHeightForwardJumpCommands(worlds, fixed_speed=.5)
        commands.stage[:] = torch.tensor([commands.FLIGHT, commands.FLIGHT, commands.WAIT])
        commands.age[:] = .20
        commands.peak_rise[:] = torch.tensor([.01, .30, .30])
        contact = {'wangyu/rewards': torch.zeros(3, len(REWARD_COLUMNS)),
                   'wangyu/failed': torch.zeros(3, dtype=torch.bool)}
        commands.advance(contact)
        self.assertFalse(commands.completed.any())
        commands.advance(contact)
        self.assertEqual(commands.completed.tolist(), [True, True, False])
        torch.testing.assert_close(commands.event_reward, torch.tensor([.55, .55, 0.]))
        torch.testing.assert_close(commands.target_height, torch.zeros(3))

    def test_speed_tracking_improves_throughout_push_and_flight(self):
        worlds = CpuWorlds(scene_spec(), 4, 12)
        commands = SpeedTrackingForwardJumpCommands(worlds, speed_range=(0., 2.), fixed_speed=2.)
        speeds = torch.tensor([.4, 1., 2., 2.6])
        worlds.sim.data.qvel[:, :2] = speeds[:, None] * commands.heading
        commands.stage.fill_(commands.EXTEND)
        pushing = commands.horizontal_rate()
        commands.stage.fill_(commands.FLIGHT)
        torch.testing.assert_close(commands.horizontal_rate(), pushing)
        self.assertLess(float(pushing[0]), float(pushing[1]))
        self.assertLess(float(pushing[1]), float(pushing[2]))
        self.assertLess(float(pushing[3]), float(pushing[2]))

    def test_tracking_samples_exact_endpoints_and_keeps_soft_stationary_anchor(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = SpeedTrackingForwardJumpCommands(worlds)
        commands.speed_generator.manual_seed(42)
        samples = torch.cat([commands.sample_speed_fraction() for _ in range(30)])
        self.assertTrue((samples == 0.).any())
        self.assertTrue((samples == 1.).any())
        self.assertTrue(((samples > 0.) & (samples < 1.)).any())
        commands.target_speed.zero_()
        displacement = torch.tensor([-1.3, 0., 1.3])
        worlds.sim.data.qpos[:, :2] = commands.origin + displacement[:, None] * commands.heading
        commands(torch.zeros(3, 24))
        torch.testing.assert_close(worlds.device.target[:, :2], worlds.sim.data.qpos[:, :2])
        rates = commands.horizontal_rate()
        self.assertLess(float(rates[0]), float(rates[1]))
        self.assertLess(float(rates[2]), float(rates[1]))

    def test_stride_reward_prefers_one_long_jump_to_two_short_jumps(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = LongStrideForwardJumpCommands(worlds, fixed_speed=1.5)
        commands.completed.fill_(True)
        commands.flight_speed_steps[:] = torch.tensor([18., 36., 36.])
        commands.flight_speed_sum[:] = 1.5 * commands.flight_speed_steps
        commands.hop_distance[:] = torch.tensor([.45, .90, -.90])
        reward = commands.stride_reward()
        self.assertGreater(float(reward[1]), 2 * float(reward[0]))
        self.assertEqual(float(reward[2]), 0.)
        commands.completed.zero_()
        torch.testing.assert_close(commands.stride_reward(), torch.zeros(3))

    def test_long_hop_weight_requires_both_time_and_forward_distance(self):
        seconds = torch.tensor([.85, 1.7, 1.7, 3.4, 1.7, .10, 1.7])
        meters = torch.tensor([.85, 1.7, .85, 3.4, -.2, 0., 3.4])
        speed = torch.tensor([1., 1., 1., 1., 1., 0., 2.])
        weights = long_hop_event_weight(seconds, meters, speed, 1.7)
        torch.testing.assert_close(weights, torch.tensor([.25, 1., .25, 1., 0., 1., 1.]))
        self.assertGreater(weights[1].item(), 2 * weights[0].item())

    def test_economical_hops_pay_equal_quality_once_per_completed_flight(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        cmd = LongStrideForwardJumpCommands(worlds, fixed_speed=1.)
        cmd.flight_speed_steps[:] = torch.tensor([18., 54., 54.])
        cmd.flight_speed_sum.copy_(cmd.flight_speed_steps)
        cmd.completed[:] = torch.tensor([True, True, False])
        cmd.hop_distance.fill_(.75)
        original = cmd.stride_reward()
        torch.testing.assert_close(original[1], 3 * original[0])
        cmd.economical_hops = True
        credit = cmd.stride_reward()
        torch.testing.assert_close(credit[0], credit[1])
        self.assertGreater(credit[0].item(), 0.)
        self.assertEqual(credit[2].item(), 0.)

    def test_stride_uses_first_touchdown_position_after_confirmed_takeoff(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = LongStrideForwardJumpCommands(worlds, fixed_speed=1.)
        data = worlds.sim.data
        commands.stage.fill_(commands.EXTEND)
        rewards = torch.zeros(2, len(REWARD_COLUMNS))
        failed = torch.tensor([False, True])
        measured = {'wangyu/rewards': rewards, 'wangyu/failed': failed}
        rewards[:, REWARD_COLUMNS['airborne']] = 1.
        data.sensordata[:, commands.com_xy] = 0.
        for _ in range(14):
            commands.advance(measured)
        rewards[:, REWARD_COLUMNS['airborne']] = 0.
        data.sensordata[:, commands.com_xy] = .62 * commands.heading
        commands.advance(measured)
        torch.testing.assert_close(commands.hop_distance, torch.zeros(2))
        data.sensordata[:, commands.com_xy] = .95 * commands.heading
        commands.advance(measured)
        torch.testing.assert_close(commands.hop_distance, torch.tensor([.62, 0.]))
        commands.reset(torch.tensor([True, False]))
        torch.testing.assert_close(commands.hop_distance, torch.zeros(2))

    def test_torso_allows_backwards_lean_on_descent_and_discourages_backbend(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = LongStrideForwardJumpCommands(worlds, fixed_speed=1.)
        data = worlds.sim.data
        commands.heading[:] = torch.tensor([1., 0.])
        angles = torch.tensor([.20, -.08, -.34]) / 2
        quat = torch.stack((angles.cos(), torch.zeros_like(angles),
                            angles.sin(), torch.zeros_like(angles)), -1)
        data.sensordata[:, commands.torso_quat] = quat
        data.qpos[:, 3:7] = torch.tensor([1., 0., 0., 0.])
        commands.stage.fill_(commands.EXTEND)
        pushing = commands.torso_posture_cost()
        commands.stage.fill_(commands.FLIGHT)
        data.qvel[:, 2] = -.70
        descending = commands.torso_posture_cost()
        self.assertLess(float(pushing[0]), float(pushing[1]))
        self.assertLess(float(descending[1]), float(descending[0]))
        self.assertGreater(float(descending[2]), float(descending[1]))

    def test_high_speed_posture_favors_forward_propulsion_and_relaxed_descent(self):
        worlds = CpuWorlds(scene_spec(), 4, 12)
        commands = HighSpeedLeanForwardJumpCommands(worlds, fixed_speed=1.5)
        torch.testing.assert_close(commands.high_speed_weight(), torch.ones(4))
        commands.stage[:] = torch.tensor(
            [commands.EXTEND, commands.FLIGHT, commands.FLIGHT, commands.COMPRESS])
        commands.ground_frames[-1] = 2
        vz = torch.tensor([.6, .6, -.6, -.6])
        reference = commands.torso_lean_reference(vz)
        self.assertTrue((reference[:2] > .15).all())
        self.assertTrue((reference[2:] < .03).all())
        forwards = commands.lean_cost(torch.full((4,), .18), torch.zeros(4), vz)
        backwards = commands.lean_cost(torch.full((4,), -.08), torch.zeros(4), vz)
        self.assertTrue((forwards[:2] < backwards[:2]).all())
        self.assertTrue((forwards[2:] > backwards[2:]).all())

    def test_high_speed_whole_body_progress_adds_reward_without_affecting_slow_commands(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = HighSpeedLeanForwardJumpCommands(worlds)
        commands.target_speed[:] = torch.tensor([.5, 1.5, 1.5])
        data = worlds.sim.data
        data.sensordata[:, commands.com_velocity_xy] = torch.tensor([1., .5, 1.5])[:, None] * commands.heading
        baseline = LongStrideForwardJumpCommands.horizontal_rate(commands)
        gain = commands.horizontal_rate() - baseline
        self.assertEqual(float(gain[0]), 0.)
        self.assertLess(float(gain[1]), float(gain[2]))

    def test_completed_arc_rewards_rise_and_relaxation_without_a_minimum_height(self):
        worlds = CpuWorlds(scene_spec(), 4, 12)
        commands = AbsorbingHopForwardJumpCommands(worlds, fixed_speed=1.5)
        commands.completed[:] = torch.tensor([True, True, True, False])
        commands.flight_speed_steps.fill_(36.)
        commands.flight_speed_sum.fill_(54.)
        commands.hop_distance.fill_(1.)
        commands.peak_com_rise[:] = torch.tensor([.02, .18, .18, .18])
        commands.rising_steps.fill_(12.)
        commands.descending_steps.fill_(12.)
        commands.rising_pitch.fill_(1.68)
        commands.descending_pitch[:] = torch.tensor([1.68, 1.68, 0., 0.])
        reward = commands.stride_reward()
        self.assertGreater(float(reward[0]), 0.)
        self.assertGreater(float(reward[1]), float(reward[0]))
        self.assertGreater(float(reward[2]), float(reward[1]))
        self.assertEqual(float(reward[3]), 0.)

    def test_landing_flexion_credit_requires_early_contact_and_incoming_speed(self):
        worlds = CpuWorlds(scene_spec(), 4, 12)
        commands = AbsorbingHopForwardJumpCommands(worlds, fixed_speed=1.5)
        commands.stage[:] = torch.tensor([commands.COMPRESS, commands.COMPRESS,
                                         commands.COMPRESS, commands.EXTEND])
        commands.landing_window.fill_(True)
        commands.ground_frames[:] = torch.tensor([2, 14, 2, 2])
        commands.impact_speed.fill_(.65)
        qvel = worlds.sim.data.qvel
        qvel[:, commands.knee_dofs] = torch.tensor([2., 2., -2., 2.])[:, None]
        score = commands.landing_flexion_rate()
        self.assertGreater(float(score[0]), 0.)
        self.assertEqual(float(score[1]), 0.)
        self.assertLess(float(score[2]), 0.)
        self.assertEqual(float(score[3]), 0.)

    def test_broad_jump_opens_on_extension_and_folds_before_landing(self):
        worlds = CpuWorlds(scene_spec(), 4, 12)
        commands = BroadJumpForwardCommands(worlds, fixed_speed=1.5)
        commands.stage[:] = torch.tensor([commands.EXTEND, commands.EXTEND,
                                         commands.FLIGHT, commands.FLIGHT])
        commands.extension_start.fill_(.85)
        data = worlds.sim.data
        data.qpos[:2, commands.knee_qpos] = torch.tensor([[.85], [.4]])
        data.sensordata[:, commands.com_velocity_xy.stop] = torch.tensor([0., 0., .65, -.65])
        reference = commands.torso_lean_reference(torch.tensor([0., 0., -.4, .4]))
        self.assertTrue((reference[[0, 3]] > .08).all())
        self.assertTrue((reference[[1, 2]] < -.08).all())
        quality = commands.flight_posture_quality(torch.tensor([-.11, .11]), torch.tensor([.14, -.14]))
        torch.testing.assert_close(quality, torch.tensor([1., 0.]))

    def test_coordinated_arms_compare_mirrored_angles_with_tolerance(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = CoordinatedBroadJumpCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.EXTEND)
        data = worlds.sim.data
        left = torch.tensor([.30, -.55, -.24, -.45])
        right = left * commands.arm_mirror
        offsets = torch.stack((left.expand(3, -1), right.expand(3, -1)), dim=1).clone()
        offsets[1, 1, 1] += .10
        offsets[2, 1, 1] += .55
        data.qpos[:, commands.arm_qpos] = commands.arm_neutral + offsets
        data.sensordata[:, commands.left_sole_xy] = 0.
        data.sensordata[:, commands.right_sole_xy] = 0.
        loss = commands.coordination_cost()
        torch.testing.assert_close(loss[:2], torch.zeros(2))
        self.assertGreater(float(loss[2]), 0.)

    def test_foot_alignment_uses_body_heading_and_emphasizes_descent(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = CoordinatedBroadJumpCommands(worlds, fixed_speed=1.5)
        data = worlds.sim.data
        data.qpos[:, commands.arm_qpos] = commands.arm_neutral
        data.qpos[:, 3:7] = torch.tensor([sqrt(.5), 0., 0., sqrt(.5)])
        data.sensordata[:, commands.left_sole_xy] = torch.tensor([[.25, 0.], [.25, .25], [.25, .25]])
        data.sensordata[:, commands.right_sole_xy] = 0.
        commands.stage.fill_(commands.FLIGHT)
        data.sensordata[:, commands.com_velocity_xy.stop] = torch.tensor([-.6, .6, -.6])
        torch.testing.assert_close(commands.foot_stagger(), torch.tensor([0., .25, .25]))
        loss = commands.coordination_cost()
        self.assertLess(float(loss[0]), float(loss[1]))
        self.assertLess(float(loss[1]), float(loss[2]))

    def test_takeoff_extension_credit_targets_liftoff_and_preserves_landing_preparation(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = CoordinatedTakeoffCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.FLIGHT)
        commands.age[:] = torch.tensor([.02, .35, .02])
        vz = torch.tensor([.55, .55, -.55])
        worlds.sim.data.sensordata[:, commands.com_velocity_xy.stop] = vz
        torso = torch.full((3,), .05)
        pelvis = torch.zeros(3)
        old_cost = CoordinatedBroadJumpCommands.lean_cost(commands, torso, pelvis, vz)
        extra_cost = commands.lean_cost(torso, pelvis, vz) - old_cost
        self.assertGreater(float(extra_cost[0]), 0.)
        torch.testing.assert_close(extra_cost[1:], torch.zeros(2))

        commands.completed.fill_(True)
        commands.flight_speed_steps.fill_(50.)
        commands.descending_steps.fill_(15.)
        commands.descending_pitch.fill_(1.8)
        commands.takeoff_torso_pitch[:] = torch.tensor([-.08, .05, -.08])
        commands.completed[-1] = False
        score = commands.stride_reward()
        self.assertGreater(float(score[0]), float(score[1]))
        self.assertEqual(float(score[2]), 0.)

    def test_working_support_counts_loaded_extension_and_carries_it_to_landing(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = SupportedTakeoffCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.EXTEND)
        data = worlds.sim.data
        data.qvel[:, commands.knee_dofs] = torch.tensor([[-1.], [-1.], [0.]])
        data.sensordata[:, commands.sole_load_columns] = 2 * worlds.device.weight
        data.sensordata[:, commands.com_velocity_xy.stop] = .20
        raw = torch.zeros(3, len(REWARD_COLUMNS))
        raw[1, REWARD_COLUMNS['airborne']] = 1.
        measured = {'wangyu/rewards': raw, 'wangyu/failed': torch.zeros(3, dtype=torch.bool)}
        commands.advance(measured)
        torch.testing.assert_close(commands.support_time, torch.tensor([commands.DT, 0., 0.]))
        torch.testing.assert_close(commands.push_speed_gain, torch.tensor([.15, 0., 0.]))

        raw[:, REWARD_COLUMNS['airborne']] = 1.
        commands.advance(measured)
        commands.advance(measured)
        torch.testing.assert_close(commands.support_time, torch.zeros(3))
        torch.testing.assert_close(commands.takeoff_support_time, torch.tensor([commands.DT, 0., 0.]))
        torch.testing.assert_close(commands.takeoff_push_speed_gain, torch.tensor([.15, 0., 0.]))
        torch.testing.assert_close(commands.takeoff_peak_load_ratio, torch.tensor([4., 0., 4.]))
        commands.completed.fill_(True)
        commands.flight_speed_steps.fill_(35.)
        reward = commands.stride_reward()
        self.assertGreater(float(reward[0]), float(reward[1]))

    def test_working_support_waives_old_ground_penalty_for_active_push(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = SupportedTakeoffCommands(worlds, fixed_speed=1.5)
        commands.stage[:] = torch.tensor([commands.COMPRESS, commands.EXTEND, commands.COMPRESS])
        commands.ground_frames.fill_(14)
        commands.age.fill_(.27)
        worlds.sim.data.qvel[:, commands.knee_dofs] = torch.tensor([[.8], [-.8], [0.]])
        worlds.sim.data.sensordata[:, commands.com_velocity_xy.stop] = 0.
        penalty = commands.grounded_delay_cost()
        torch.testing.assert_close(penalty[:2], torch.zeros(2))
        self.assertGreater(float(penalty[2]), 0.)

    def test_supported_extension_starts_when_the_center_of_mass_turns(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = SupportedTakeoffCommands(worlds, fixed_speed=1.5)
        commands.age.fill_(.24)
        commands.previous_com_vz.fill_(-.26)
        worlds.sim.data.sensordata[:, commands.com_velocity_xy.stop] = torch.tensor([-.20, -.04, .02])
        ready = commands.can_push(
            torch.ones(3, dtype=torch.bool), torch.full((3,), .8), torch.full((3,), .20))
        self.assertEqual(ready.tolist(), [False, True, True])

    def test_equal_load_integral_costs_more_when_concentrated_into_a_peak(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = SupportedTakeoffCommands(worlds, fixed_speed=1.5)
        commands.landing_window.fill_(True)
        commands.sole_load_ratio[:] = torch.tensor([6., 3.])
        commands.sole_load_rise.copy_(commands.sole_load_ratio)
        first = commands.contact_load_cost()
        commands.sole_load_rise.zero_()
        later = commands.contact_load_cost()
        self.assertGreater(float(first[0] + later[0]), float(first[1] + 3 * later[1]))

    def test_stance_flow_requires_both_loaded_flexion_and_loaded_reextension(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = StanceFlowTakeoffCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.COMPRESS)
        data = worlds.sim.data
        data.sensordata[:, commands.sole_load_columns] = worlds.device.weight
        data.sensordata[:, commands.com_z_column] = 1.
        data.qpos[:, commands.knee_qpos] = .35
        raw = torch.zeros(2, len(REWARD_COLUMNS))
        measured = {'wangyu/rewards': raw, 'wangyu/failed': torch.zeros(2, dtype=torch.bool)}
        commands.advance(measured)

        data.sensordata[:, commands.com_z_column] = .93
        data.sensordata[:, commands.com_velocity_xy.stop] = -.25
        data.qpos[:, commands.knee_qpos] = .68
        data.qvel[:, commands.knee_dofs] = .9
        commands.advance(measured)
        torch.testing.assert_close(commands.stance_flow_quality, torch.zeros(2))

        data.sensordata[:, commands.com_z_column] = .995
        data.sensordata[:, commands.com_velocity_xy.stop] = .30
        data.qpos[0, commands.knee_qpos] = .45
        data.qvel[:, commands.knee_dofs] = torch.tensor([[-1.4], [0.]])
        commands.advance(measured)
        self.assertGreater(float(commands.stance_flow_rate[0]), float(commands.stance_flow_rate[1]))
        torch.testing.assert_close(commands.stance_flow_quality, torch.tensor([1., 0.]))

        raw[:, REWARD_COLUMNS['airborne']] = 1.
        data.qvel[:, 2] = .35
        commands.advance(measured)
        commands.advance(measured)
        torch.testing.assert_close(commands.takeoff_flow_quality, torch.tensor([1., 0.]))

    def test_stance_stride_bonus_prefers_larger_forward_hops_and_respects_zero_speed(self):
        worlds = CpuWorlds(scene_spec(), 3, 12)
        commands = StanceStrideTakeoffCommands(worlds)
        commands.target_speed[:] = torch.tensor([.5, .5, 0.])
        commands.hop_distance[:] = torch.tensor([.25, .65, .65])
        commands.completed.fill_(True)
        commands.flight_speed_steps.fill_(48.)
        baseline = StanceFlowTakeoffCommands.stride_reward(commands)
        added = commands.stride_reward() - baseline
        self.assertGreater(float(added[1]), float(added[0]))
        self.assertGreater(float(added[0]), 0.)
        self.assertEqual(float(added[2]), 0.)

    def test_landing_preparation_cost_respects_contact_and_each_knee(self):
        worlds = CpuWorlds(scene_spec(), 6, 12)
        commands = LandingPreparationCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.FLIGHT)
        commands.age.fill_(.15)
        data = worlds.sim.data
        data.qpos[:, commands.knee_qpos] = torch.tensor([
            [0., 0.], [0., 0.], [0., .4], [.2, .2], [0., 0.], [0., 0.],
        ])
        data.sensordata[:, commands.com_velocity_xy.stop] = torch.tensor([
            .30, -.50, -.50, -.50, -.50, -.50,
        ])
        commands.ground_frames[4] = 1
        commands.age[5] = 0.
        cost = commands.airborne_knee_cost()
        self.assertGreater(float(cost[1]), float(cost[0]))
        self.assertGreater(float(cost[2]), float(cost[3]))
        self.assertEqual(float(cost[4]), 0.)
        self.assertEqual(float(cost[5]), 0.)

    def test_full_sole_reward_allows_a_heel_lift_late_in_push_off(self):
        worlds = CpuWorlds(scene_spec(geometry='fitted', sole_probes=True), 3, 12)
        commands = FullSoleForwardJumpCommands(worlds, fixed_speed=1.5)
        self.assertTrue(torch.isfinite(commands.sole_clearances()).all())
        commands.stage[:] = torch.tensor([commands.COMPRESS, commands.COMPRESS, commands.EXTEND])
        commands.ground_frames.fill_(3)
        commands.age.fill_(.2)
        data = worlds.sim.data
        data.sensordata[:, commands.sole_load_columns] = worlds.device.weight
        gaps = torch.tensor([[[.08, .0], [.08, .0]], [[.01, .0], [.01, .0]],
                             [[.08, .0], [.08, .0]]])
        with patch.object(commands, 'sole_clearances', return_value=gaps):
            cost = commands.sole_placement_cost()
        self.assertGreater(float(cost[0]), 0.)
        self.assertEqual(cost[1:].tolist(), [0., 0.])

        commands.target_speed[:] = torch.tensor([0., .5, 1.5])
        torch.testing.assert_close(commands.reference_progress(torch.ones(3)), torch.tensor([0., 1., 1.]))

    def test_distributed_support_requires_solved_rear_and_forefoot_contact(self):
        worlds = CpuWorlds(scene_spec(geometry='fitted', sole_probes=True), 3, 12)
        commands = DistributedSoleForwardJumpCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.COMPRESS)
        commands.ground_frames.fill_(3)
        probes = commands.sole_probes()
        samples = torch.zeros(3, 2, 4, 8)
        samples[:, :, 0:2, 0] = 1.
        samples[:, :, 0:2, 1] = .5 * worlds.device.weight
        samples[:, :, 0:2, 5:8] = probes
        samples[1, :, 0, 0] = 0.  # An inactive heel slot carries no weight.
        samples[2, :, 1, 4] = .1  # A distant forefoot slot is not contact.
        worlds.sim.data.sensordata[:, worlds.device.contact_indices] = samples.flatten(-2)
        distributed, front_only, heel_only = commands.supported_sole_rate()
        self.assertGreater(float(distributed), float(heel_only))
        self.assertGreater(float(heel_only), float(front_only))
        self.assertEqual(float(front_only), 0.)

    def test_ankle_alignment_only_changes_touchdown_and_loaded_ankle_targets(self):
        worlds = CpuWorlds(scene_spec(geometry='fitted', sole_probes=True), 3, 12)
        commands = TerrainAlignedLandingCommands(worlds, fixed_speed=1.5)
        commands.stage[:] = torch.tensor([commands.FLIGHT, commands.FLIGHT, commands.EXTEND])
        commands.ground_frames[2] = 3
        commands.age.fill_(.20)
        data = worlds.sim.data
        data.sensordata[:, commands.com_velocity_xy.stop] = torch.tensor([-.5, .5, .3])
        actions = torch.zeros(3, 24)
        gaps = torch.tensor([[[.060, .010], [.010, .060]]]).repeat(3, 1, 1)
        baseline = DistributedSoleForwardJumpCommands.__call__(commands, actions)
        with patch.object(commands, 'sole_clearances', return_value=gaps):
            changed = commands(actions) - baseline
        left, right = commands.ankle_actions
        self.assertLess(float(changed[0, left]), 0.)
        self.assertGreater(float(changed[0, right]), 0.)
        self.assertEqual(int(torch.count_nonzero(changed)), 2)

    def test_bilateral_landing_targets_the_lagging_ankle_and_forward_travel(self):
        worlds = CpuWorlds(scene_spec(geometry='fitted', sole_probes=True, sole_collision=True), 3, 12)
        commands = BilateralSoleLandingCommands(worlds, fixed_speed=1.5)
        angles = torch.tensor([[.04, .60], [.60, .04], [.04, .04]])
        base = TerrainAlignedLandingCommands.corrected_ankles(
            commands, torch.zeros_like(angles), angles, torch.ones_like(angles))
        aligned = commands.corrected_ankles(torch.zeros_like(angles), angles, torch.ones_like(angles))
        self.assertLess(float(aligned[0, 1]), float(base[0, 1]))
        self.assertLess(float(aligned[1, 0]), float(base[1, 0]))
        torch.testing.assert_close(aligned[[0, 1], [0, 1]], base[[0, 1], [0, 1]])
        torch.testing.assert_close(aligned[2], base[2])

        data = worlds.sim.data
        data.sensordata[:, commands.torso_quat] = torch.tensor([1., 0., 0., 0.])
        data.sensordata[:, commands.com_velocity_xy] = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        forward, sideways, backward = commands.torso_travel_cost()
        self.assertEqual(float(forward), 0.)
        self.assertGreater(float(sideways), float(forward))
        self.assertGreater(float(backward), float(forward))
        data.sensordata[:, commands.separation_column] = torch.tensor([.08, .02, -.02])
        torch.testing.assert_close(commands.sole_separation_deficit(), torch.tensor([0., .5, 1.5]))

    def test_landing_knee_correction_only_releases_the_more_bent_toe_down_leg(self):
        worlds = CpuWorlds(scene_spec(geometry='fitted', sole_probes=True, sole_collision=True), 3, 12)
        commands = CoordinatedLandingCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.FLIGHT)
        data = worlds.sim.data
        data.qpos[:, commands.knee_qpos] = torch.tensor([[.4, .9], [.9, .4], [.4, .9]])
        data.sensordata[:, commands.com_velocity_xy.stop] = torch.tensor([-.45, -.45, .40])
        gaps = torch.tensor([[[.011, .01], [.145, .01]],
                             [[.145, .01], [.011, .01]],
                             [[.011, .01], [.145, .01]]])
        actions = torch.ones(3, 24)
        with patch.object(commands, 'sole_clearances', return_value=gaps):
            before = BilateralSoleLandingCommands.__call__(commands, actions)[:, commands.knee_actions]
            after = commands(actions)[:, commands.knee_actions]
        delta = after - before
        self.assertLess(float(delta[0, 1]), 0.)
        self.assertLess(float(delta[1, 0]), 0.)
        self.assertEqual(int(torch.count_nonzero(delta)), 2)

    def test_mirror_constraint_reduces_large_differences_and_allows_small_ones(self):
        worlds = CpuWorlds(scene_spec(geometry='fitted', sole_probes=True, sole_collision=True), 3, 12)
        commands = MirroredLandingCommands(worlds, fixed_speed=1.5)
        left, right = commands.mirror_actions[:, 3]
        left_roll, right_roll = commands.mirror_actions[:, 0]
        raw = torch.zeros(3, 24)
        raw[0, right] = 1.4
        raw[1, right] = .10
        raw[2, [left_roll, right_roll]] = torch.tensor([.8, -.8])
        mirrored = commands.mirror_joint_targets(raw)
        self.assertGreater(float(mirrored[0, left]), 0.)
        self.assertLess(float(mirrored[0, right]), 1.4)
        self.assertGreater(float(mirrored[0, right] - mirrored[0, left]), 0.)
        torch.testing.assert_close(mirrored[1:], raw[1:])
        torch.testing.assert_close(mirrored[:, -1], raw[:, -1])

    def test_sole_plane_adjusts_pitch_and_roll_without_changing_ascent(self):
        spec = scene_spec(geometry='fitted', sole_collision=True, sole_edge_probes=True)
        worlds = CpuWorlds(spec, 2, 12)
        commands = SolePlaneMirroredLandingCommands(worlds, fixed_speed=1.5)
        commands.stage.fill_(commands.FLIGHT)
        sensors = worlds.sim.data.sensordata
        sensors[:, commands.com_velocity_xy.stop] = torch.tensor([-.4, .4])
        gaps = torch.tensor([[[.01, .01], [.13, .01]]]).expand(2, -1, -1)
        edges = torch.tensor([[[.01, .01], [.01, .06]]]).expand(2, -1, -1)
        raw = torch.zeros(2, 24)
        with patch.object(commands, 'sole_clearances', return_value=gaps), patch.object(
                commands, 'sole_edge_clearances', return_value=edges):
            base = PhaseMirroredLandingCommands.__call__(commands, raw)
            adjusted = commands(raw)
        delta = adjusted - base
        ids = commands.plane_actions[1]
        self.assertEqual(adjusted.dtype, raw.dtype)
        ankle_dofs = commands.plane_qpos[1]
        target = worlds.sim.data.qpos[0, ankle_dofs] - torch.atan(torch.stack(
            (.12 / commands.sole_span[1], .05 / commands.sole_width[1])))
        scale = worlds.device.action_scale[ids]
        neutral = worlds.device.default_q[ids]
        pre = neutral + scale * base[0, ids]
        post = neutral + scale * adjusted[0, ids]
        self.assertTrue(((post - target).abs() < (pre - target).abs()).all())
        self.assertEqual(int(torch.count_nonzero(delta)), 2)

        # With level heel-to-toe clearance, both signs of lateral tilt still
        # need correction on approach. The ascending world remains unchanged.
        level_pitch = torch.full_like(gaps, .04)
        tilted_edges = torch.tensor([[[.01, .08], [.08, .01]]]).expand(2, -1, -1)
        with patch.object(commands, 'sole_clearances', return_value=level_pitch), patch.object(
                commands, 'sole_edge_clearances', return_value=tilted_edges):
            base = PhaseMirroredLandingCommands.__call__(commands, raw)
            adjusted = commands(raw)
        correction = adjusted - base
        self.assertLess(float(correction[0, commands.plane_actions[0, 1]]), 0.)
        self.assertGreater(float(correction[0, commands.plane_actions[1, 1]]), 0.)
        self.assertEqual(int(torch.count_nonzero(correction)), 2)

    def test_boot_yaw_preserves_toe_out_and_measures_rotation_relative_to_torso(self):
        spec = scene_spec(geometry='fitted', sole_collision=True, sole_edge_probes=True)
        commands = ForwardFacingBootCommands(CpuWorlds(spec, 2, 12), fixed_speed=1.5)
        data = commands.worlds.sim.data
        chest_yaw = torch.tensor([.8, -1.1])
        chest_quat = torch.zeros(2, 4)
        chest_quat[:, 0] = (chest_yaw / 2).cos()
        chest_quat[:, 3] = (chest_yaw / 2).sin()
        data.sensordata[:, commands.torso_quat] = chest_quat
        toe_yaw = chest_yaw[:, None] + commands.neutral_toe_out + torch.tensor([[0.], [.5]])
        probes = torch.zeros(2, 2, 2, 3)
        probes[:, :, 1, :2] = torch.stack((toe_yaw.cos(), toe_yaw.sin()), dim=-1) * .25
        with patch.object(commands, 'sole_probes', return_value=probes):
            torch.testing.assert_close(commands.boot_yaw_error(), torch.tensor([[0., 0.], [.5, .5]]))

    def test_forward_velocity_is_projected_onto_the_spawn_heading(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        worlds.device.heading[:] = torch.tensor([[1., 0., 0., 0.],
                                                 [.5 * sqrt(2), 0., 0., .5 * sqrt(2)]])
        commands = ForwardJumpCommands(worlds, fixed_speed=1.)
        worlds.sim.data.qvel[:, :2] = torch.tensor([[1., .25], [-.25, 1.]])
        forward, lateral = commands.track_velocity()
        torch.testing.assert_close(forward, torch.ones(2))
        torch.testing.assert_close(lateral, torch.full((2,), .25))
        worlds.sim.data.qpos[:, :2] = commands.origin + torch.tensor([[1.5, .1], [.1, 1.5]])
        commands(torch.zeros(2, 24))
        torch.testing.assert_close(worlds.device.target[:, :2] - commands.origin,
                                   torch.tensor([[1.5, 0.], [0., 1.5]]))

    def test_added_inputs_preserve_inherited_actor_output(self):
        groups = {'actor': ['policy']}
        obs = TensorDict({'policy': torch.randn(4, 6)}, batch_size=[4])
        extended = TensorDict({'policy': torch.cat((obs['policy'], torch.randn(4, 3)), -1)},
                              batch_size=[4])
        kwargs = dict(hidden_dims=(8,), obs_normalization=True)
        base = MLPModel(obs, groups, 'actor', 24, **kwargs)
        forward = MLPModel(extended, groups, 'actor', 24, **kwargs)
        forward.load_state_dict(forward_initial_state(base.state_dict(), forward.state_dict()))
        torch.testing.assert_close(forward(extended), base(obs))


if __name__ == '__main__':
    unittest.main()
