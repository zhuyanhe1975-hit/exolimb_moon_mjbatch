"""Check the lunar jump reference and ground-to-ground completion metric."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cpu_jump_task import (FLIGHT_SECS, PERIOD_SECS, STANCE_SECS, REWARD_COLUMNS,
                           AdaptiveJumpCommands, ContactJumpCommands, FlowJumpCommands, ReboundJumpCommands,
                           CompliantJumpCommands, JumpCounter, jump_reference,
                           peak_torque_ratio, positive_actuator_power)
from cpu_simulation import CpuWorlds
from moon_scene import scene_spec


class JumpTaskTest(unittest.TestCase):
    def test_flow_push_requires_compression_but_does_not_wait_at_turnaround(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = FlowJumpCommands(worlds)
        commands.stage[:] = commands.COMPRESS
        commands.age[:] = .23
        commands.start_knee[:] = .30
        commands.ground_frames[:] = 2
        commands.previous_vz[:] = -.19
        commands.push_request[:] = -1.
        commands.update(torch.tensor([False, False]), torch.tensor([.66, .36]),
                        torch.tensor([-.05, -.05]))
        self.assertEqual(commands.stage.tolist(), [commands.EXTEND, commands.COMPRESS])

    def test_rebound_begins_during_velocity_turn_with_ground_support(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = ReboundJumpCommands(worlds)
        commands.stage[:] = commands.COMPRESS
        commands.age[:] = .12
        commands.start_knee[:] = .22
        commands.ground_frames[:] = 2
        commands.previous_vz[:] = -.20
        commands.push_request[:] = -1.
        knee = torch.tensor([.50, .50])
        vz = torch.tensor([-.06, -.06])
        commands.update(torch.tensor([False, True]), knee, vz)
        self.assertEqual(commands.stage.tolist(), [commands.EXTEND, commands.COMPRESS])

    def test_random_targets_resample_only_selected_worlds_in_training_range(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = AdaptiveJumpCommands(worlds, height_range=(.10, .50))
        unchanged = commands.target_height[1].clone()
        values = []
        for _ in range(32):
            commands.reset(torch.tensor([True, False]))
            values.append(float(commands.target_height[0]))
            self.assertEqual(commands.target_height[1], unchanged)
        self.assertTrue(all(.10 <= height <= .50 for height in values))
        self.assertEqual(len(set(values)), len(values))

    def test_fixed_inference_height_survives_landing_and_partial_reset(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = AdaptiveJumpCommands(worlds, fixed_height=.15)
        data = worlds.sim.data
        commands.stage[:] = commands.FLIGHT
        commands.age[:] = .5
        commands.flight_target[:] = .15
        commands.takeoff_z.copy_(data.qpos[:, 2] - .15)
        commands.peak_rise[:] = .15
        contact = {'wangyu/rewards': torch.zeros(2, 9),
                   'wangyu/failed': torch.zeros(2, dtype=torch.bool)}
        for _ in range(2):
            commands.advance(contact)
        self.assertTrue(commands.completed.all())
        torch.testing.assert_close(commands.target_height, torch.full((2,), .15))
        commands.reset(torch.tensor([True, False]))
        torch.testing.assert_close(commands.target_height, torch.full((2,), .15))

    def test_learned_push_and_height_change_require_physical_events(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = AdaptiveJumpCommands(worlds)
        data = worlds.sim.data
        data.qpos[:, commands.knee_qpos] = .75
        contact = {'wangyu/rewards': torch.zeros(2, 9),
                   'wangyu/failed': torch.zeros(2, dtype=torch.bool)}
        actions = torch.zeros(2, 24)
        actions[:, -1] = -1.
        initial_target = commands.target_height.clone()
        for _ in range(80):
            self.assertEqual(commands(actions).shape, (2, 23))
            commands.advance(contact)
        self.assertEqual(commands.stage.tolist(), [commands.COMPRESS, commands.COMPRESS])
        torch.testing.assert_close(commands.target_height, initial_target)

        data.qpos[:, commands.knee_qpos] = .95
        actions[0, -1] = 1.5
        commands(actions)
        commands.advance(contact)
        self.assertEqual(commands.stage.tolist(), [commands.EXTEND, commands.COMPRESS])
        self.assertLess(commands.push_duration[0], commands.push_duration[1])
        contact['wangyu/rewards'][0, REWARD_COLUMNS['airborne']] = 1
        data.qvel[0, 2] = .6
        for _ in range(2):
            commands.advance(contact)
        self.assertEqual(int(commands.stage[0]), commands.FLIGHT)
        self.assertEqual(float(commands.flight_target[0]), float(initial_target[0]))
        data.qpos[0, 2] += .12
        for _ in range(20):
            commands.advance(contact)
        torch.testing.assert_close(commands.target_height, initial_target)
        contact['wangyu/rewards'][0, REWARD_COLUMNS['airborne']] = 0
        for _ in range(2):
            commands.advance(contact)
        self.assertEqual(commands.completed.tolist(), [True, False])
        self.assertNotEqual(float(commands.target_height[0]), float(initial_target[0]))
        self.assertEqual(float(commands.target_height[1]), float(initial_target[1]))

    def test_energy_and_peak_torque_use_motor_sign_and_limits(self):
        force = torch.tensor([[18., -45., 0.]])
        velocity = torch.tensor([[2., 1., 3.]])
        self.assertEqual(float(positive_actuator_power(force, velocity)), 36.)
        limits = torch.tensor([30., 90., 40.])
        torch.testing.assert_close(peak_torque_ratio(force, limits), torch.tensor([.6]))

    def test_landing_starts_at_measured_knee_and_adapts_to_impact(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = CompliantJumpCommands(worlds)
        data = worlds.sim.data
        data.qpos[:, commands.knee_qpos] = .25
        contact = {'wangyu/rewards': torch.zeros((2, 9))}
        commands.previous_vz[:] = torch.tensor([-.25, -.65])
        commands.advance(contact)
        commands.advance(contact)
        torch.testing.assert_close(commands.crouch_reference(), torch.full((2,), .25))
        self.assertGreater(commands.peak_knee[1], commands.peak_knee[0])
        data.actuator_force[:, commands.leg_actuators] = commands.force_limits
        self.assertTrue((commands.efficiency_cost() > 0).all())
        commands.landing_window.zero_()
        commands.shock.zero_()
        self.assertTrue((commands.efficiency_cost() == 0).all())

    def test_phase_waits_for_contact_and_takeoff_in_each_world(self):
        worlds = CpuWorlds(scene_spec(), 2, 12)
        commands = ContactJumpCommands(worlds)
        still_airborne = torch.tensor([True, True])
        one_touchdown = torch.tensor([False, True])
        knee = torch.zeros(2)
        vertical_speed = worlds.sim.data.qvel[:, 2]
        vertical_speed.fill_(-.5)

        for _ in range(90):
            commands.update(still_airborne, knee, vertical_speed)
        self.assertEqual(commands.stage.tolist(), [commands.WAIT, commands.WAIT])
        commands.update(one_touchdown, knee, vertical_speed)
        self.assertEqual(commands.stage.tolist(), [commands.WAIT, commands.WAIT])
        commands.update(one_touchdown, knee, vertical_speed)
        self.assertEqual(commands.stage.tolist(), [commands.COMPRESS, commands.WAIT])
        self.assertEqual(float(commands.phase()[0]), 0.)

        knee[0] = .9
        vertical_speed[0] = 0.
        for _ in range(15):
            commands.update(one_touchdown, knee, vertical_speed)
        self.assertEqual(commands.stage.tolist(), [commands.EXTEND, commands.WAIT])
        vertical_speed[0] = .5
        commands.update(one_touchdown, knee, vertical_speed)
        self.assertEqual(int(commands.stage[0]), commands.EXTEND)
        commands.update(still_airborne, knee, vertical_speed)
        commands.update(still_airborne, knee, vertical_speed)
        self.assertEqual(int(commands.stage[0]), commands.FLIGHT)

        vertical_speed[0] = -.5
        for _ in range(100):
            commands.update(still_airborne, knee, vertical_speed)
        self.assertGreaterEqual(float(commands.phase()[0]), STANCE_SECS)
        self.assertLess(float(commands.phase()[0]), PERIOD_SECS)
        commands.update(one_touchdown, knee, vertical_speed)
        commands.update(one_touchdown, knee, vertical_speed)
        self.assertEqual(commands.stage.tolist(), [commands.COMPRESS, commands.WAIT])

    def test_reference_has_crouch_ballistic_apex_and_period(self):
        height, velocity, flying = jump_reference(torch.tensor([
            STANCE_SECS / 2, STANCE_SECS + FLIGHT_SECS / 2, PERIOD_SECS + STANCE_SECS / 2,
        ]), gravity=1.635)
        self.assertLess(height[0], 0.)
        self.assertAlmostEqual(float(height[1]), 1.635 * FLIGHT_SECS ** 2 / 8, places=5)
        torch.testing.assert_close(height[0], height[2])
        torch.testing.assert_close(velocity, torch.zeros_like(velocity))
        self.assertEqual(flying.tolist(), [False, True, False])

    def test_counter_rejects_reset_airtime_and_tracks_consecutive_landings(self):
        counter = JumpCounter(2)
        flying = torch.tensor([True, True])
        grounded = ~flying
        failed = torch.tensor([False, True])
        height = torch.full((2,), .09)
        for _ in range(24):
            counter.update(height, flying, grounded)
        counter.update(height, grounded, grounded)
        self.assertEqual(counter.count.tolist(), [0, 0])
        for _ in range(2):
            for _ in range(24):
                counter.update(height, flying, grounded)
            counter.update(height, grounded, failed)
        self.assertEqual(counter.count.tolist(), [2, 0])
        self.assertEqual(counter.best_streak.tolist(), [2, 0])
        counter.reset()
        self.assertEqual(counter.streak.tolist(), [0, 0])


if __name__ == '__main__':
    unittest.main()
