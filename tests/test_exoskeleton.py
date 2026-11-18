"""Static transmission, phase gating and control-layout checks for the exoskeleton."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from cpu_stand_env import CpuExoJumpEnv, CpuForwardJumpEnv
from exoskeleton import (MAX_LENGTH_M, MIN_LENGTH_M, MOTOR_POLICY_IO, PD_KD, PD_KP, PD_RELEASE_LENGTH_M,
                         STRUT_ACTIONS, STRUT_OBSERVATIONS, STRUTS, exo_initial_state,
                         fit_struts_to_pose, mount_names, validate_assist_force)
from moon_scene import scene_spec


class ExoskeletonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = scene_spec(geometry='fitted', exoskeleton=True).compile()

    def test_hooke_slide_ball_assembly_and_positive_slider_force(self):
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(model.neq, 2)
        self.assertTrue(np.all(model.eq_type == mujoco.mjtEq.mjEQ_CONNECT))
        self.assertEqual(validate_assist_force(500.), 500.)
        with self.assertRaises(ValueError):
            validate_assist_force(500.01)
        for side, requested in zip(STRUTS, (500., 600.), strict=True):
            name = f'robot/exo_{side}'
            a, b, slide = (model.joint(f'{name}_{suffix}') for suffix in ('hooke_a', 'hooke_b', 'slide'))
            self.assertEqual(a.type[0], mujoco.mjtJoint.mjJNT_HINGE)
            self.assertEqual(b.type[0], mujoco.mjtJoint.mjJNT_HINGE)
            self.assertAlmostEqual(float(data.xaxis[a.id] @ data.xaxis[b.id]), 0.)
            self.assertEqual(slide.type[0], mujoco.mjtJoint.mjJNT_SLIDE)
            np.testing.assert_allclose(slide.range + model.body(f'{name}_shaft').pos[2],
                                       [MIN_LENGTH_M, MAX_LENGTH_M])
            motor = model.actuator(name)
            np.testing.assert_array_equal(motor.ctrlrange, [0., 500.])
            np.testing.assert_array_equal(motor.forcerange, [0., 500.])
            data.ctrl[motor.id] = requested
        data.joint('robot/LeftKnee').qpos[0] = .37
        data.joint('robot/RightHipPitch').qpos[0] = -.18
        data.joint('robot/WaistPitch').qpos[0] = .12
        fit_struts_to_pose(model, data)
        mujoco.mj_forward(model, data)
        for side in STRUTS:
            name = f'robot/exo_{side}'
            ball, boot = data.site(f'{name}_ball_pin').xpos, data.site(mount_names(side)[1]).xpos
            np.testing.assert_allclose(ball, boot, atol=1e-9)
            aid = model.actuator(name).id
            dof = int(model.joint(f'{name}_slide').dofadr[0])
            self.assertAlmostEqual(data.actuator_force[aid], 500.)
            self.assertAlmostEqual(data.qfrc_actuator[dof], 500.)
        eq_rows = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self.assertEqual(int(eq_rows.sum()), 6)
        np.testing.assert_allclose(data.efc_pos[:data.nefc][eq_rows], 0., atol=1e-9)
        self.assertEqual(data.time, 0.)

    def test_compact_mounts_clear_the_backpack_profile_and_sole_plane(self):
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        pack = model.geom('robot/backpack')
        rear, front = pack.pos[0] + np.array([-1., 1.]) * pack.size[0]
        for side, sign in (('left', 1.), ('right', -1.)):
            name = f'robot/exo_{side}'
            upper = model.site(mount_names(side)[0]).pos
            pack_plate = model.geom(f'{name}_pack_plate')
            self.assertGreater(upper[0], pack.pos[0])
            self.assertGreaterEqual(pack_plate.pos[0] - pack_plate.size[0], rear)
            self.assertLessEqual(pack_plate.pos[0] + pack_plate.size[0], front)
            sole = model.geom(f'robot/{side}_sole')
            up = data.geom_xmat[sole.id].reshape(3, 3)[:, 2]
            sole_bottom = data.geom_xpos[sole.id] - sole.size[2] * up
            gaps = {}
            for geom_id in range(model.ngeom):
                geom = model.geom(geom_id)
                if not geom.name.startswith(name):
                    continue
                axis = data.geom_xmat[geom_id].reshape(3, 3).T @ up
                if geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
                    half_height = np.abs(axis) @ geom.size
                elif geom.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    half_height = geom.size[0]
                else:
                    self.assertEqual(geom.type[0], mujoco.mjtGeom.mjGEOM_CYLINDER)
                    half_height = np.linalg.norm(axis[:2]) * geom.size[0] + abs(axis[2]) * geom.size[1]
                gaps[geom.name] = (data.geom_xpos[geom_id] - sole_bottom) @ up - half_height
            self.assertAlmostEqual(min(gaps.values()), 0., places=8)
            for edge in ('fore', 'aft'):
                self.assertAlmostEqual(gaps[f'{name}_socket_{edge}'], 0., places=8)
                lug = model.geom(f'{name}_boot_lug_{edge}')
                axis = data.geom_xmat[lug.id].reshape(3, 3)[:, 2]
                axis = -np.sign(axis @ up) * axis
                self.assertLess(axis @ up, -.5)
                self.assertGreater(sign * axis[1], .5)
            self.assertAlmostEqual(gaps[f'{name}_ball'], .005, places=8)

    def test_assistance_requires_loaded_extension_and_only_the_loaded_boot(self):
        env = CpuForwardJumpEnv(4, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.)
        assist = env.worlds.exoskeleton
        data = env.worlds.sim.data
        cmd = env.jump_commands
        cmd.stage[:] = torch.tensor([cmd.EXTEND, cmd.COMPRESS, cmd.FLIGHT, cmd.EXTEND])
        cmd.ground_frames[:] = 4
        cmd.age[:] = .10
        cmd.push_duration[:] = .20
        data.sensordata[:, assist.load] = env.worlds.device.weight
        data.sensordata[0, assist.load[1]] = 0.
        data.sensordata[:, assist.stroke] = 1.10 - assist.nominal_length
        data.actuator_velocity[:, assist.actuators] = .25
        data.actuator_velocity[3, assist.actuators[0]] = -.20
        data.sensordata[3, assist.stroke[1]] = 1.60 - assist.nominal_length[1]
        torch.testing.assert_close(assist.controls(), torch.tensor(
            [[500., 0.], [0., 0.], [0., 0.], [0., 0.]]))
        data.ctrl[:, assist.actuators] = 500.
        assist.reset_controls(torch.tensor([True, False, True, False]))
        torch.testing.assert_close(data.ctrl[:, assist.actuators], torch.tensor(
            [[0., 0.], [500., 500.], [0., 0.], [500., 500.]]))
        assist.force_n = 0.
        self.assertEqual(torch.count_nonzero(assist.controls()).item(), 0)

    def test_human_action_layout_ignores_the_added_mechanism_joints(self):
        args = (2, 12, False, 'mjbatch', 2, 42)
        torch.manual_seed(42)
        existing = CpuForwardJumpEnv(*args, geometry='fitted')
        torch.manual_seed(42)
        exo = CpuExoJumpEnv(*args, geometry='fitted', assist_force=0.)
        old_obs = existing.get_observations()['policy']
        self.assertEqual(exo.get_observations()['policy'].shape, (2, old_obs.shape[1] + STRUT_OBSERVATIONS))
        self.assertEqual(exo.num_actions, existing.num_actions + STRUT_ACTIONS)
        self.assertEqual(exo.worlds.model.nu, existing.worlds.model.nu + 2)
        self.assertEqual(exo.worlds.model.nq, existing.worlds.model.nq + 6)
        self.assertEqual(len(exo.worlds.device.qadr), 23)
        torch.testing.assert_close(exo.worlds.sim.data.qpos[:, exo.worlds.device.qadr],
                                   existing.worlds.sim.data.qpos[:, existing.worlds.device.qadr])
        raw = torch.linspace(-.6, .6, existing.num_actions).repeat(2, 1)
        strut_actions = torch.tensor([[.8, -.3], [-.3, .8]])
        exo_cmd = exo.commands(torch.cat((raw, strut_actions), dim=-1))
        torch.testing.assert_close(exo_cmd[:, -STRUT_ACTIONS:], strut_actions)
        released_controls = exo.worlds.sim.data.ctrl[:, -2:].clone()
        existing.worlds.device.set_substep_commands(existing.worlds.sim, {'wangyu/action': existing.commands(raw)})
        exo.worlds.device.set_substep_commands(exo.worlds.sim, {'wangyu/action': exo_cmd[:, :-STRUT_ACTIONS]})
        unchanged = [i for i in range(existing.worlds.model.nu)
                     if 'Shoulder' not in existing.worlds.model.actuator(i).name]
        torch.testing.assert_close(exo.worlds.sim.data.ctrl[:, unchanged],
                                   existing.worlds.sim.data.ctrl[:, unchanged])
        torch.testing.assert_close(exo.worlds.sim.data.ctrl[:, -2:], released_controls)
        self.assertEqual(torch.count_nonzero(exo.worlds.sim.data.actuator_force[:, -2:]).item(), 0)

    def test_policy_motor_actions_reach_controls_and_feed_back_into_observation(self):
        env = CpuForwardJumpEnv(2, 12, False, 'threaded', 1, 42, geometry='fitted', assist_force=500.,
                                exo_policy_io=MOTOR_POLICY_IO)
        worlds, cmd = env.worlds, env.jump_commands
        assist, data = worlds.exoskeleton, worlds.sim.data
        cmd.stage[:] = cmd.EXTEND
        cmd.ground_frames[:] = 4
        cmd.age[:] = .10
        cmd.push_duration[:] = .20
        data.sensordata[:, assist.load] = worlds.device.weight
        data.sensordata[1, assist.load[1]] = 0.
        data.sensordata[:, assist.stroke] = 1.10 - assist.nominal_length
        data.actuator_velocity[:, assist.actuators] = .25
        actions = torch.zeros((2, env.num_actions))
        actions[:, -2:] = torch.tensor([[-1., .2], [2., 1.]])
        expected = torch.tensor([[0., 300.], [500., 0.]])

        def copy_controls(start, end, controls, forces):
            for row in range(start, end):
                worlds.data[row].ctrl[:] = controls[row]

        with patch.object(worlds, '_step_range', side_effect=copy_controls):
            worlds.step(env.commands(actions))
        torch.testing.assert_close(data.actuator_force[:, assist.actuators], expected)
        observation = env.get_observations()['policy'][:, -STRUT_OBSERVATIONS:].reshape(2, 2, 5)
        torch.testing.assert_close(observation[..., 2], expected / 500.)
        torch.testing.assert_close(observation[..., 3], torch.tensor([[0., .6], [1., 1.]]))
        worlds.reset(torch.tensor([True, False]))
        observation = env.get_observations()['policy'][:, -STRUT_OBSERVATIONS:].reshape(2, 2, 5)
        torch.testing.assert_close(observation[0, :, 2:4], torch.zeros((2, 2)))
        torch.testing.assert_close(observation[1, :, 3], torch.ones(2))
        self.assertTrue(all(data.time == 0. for data in worlds.data))

    def test_policy_position_targets_drive_native_pd_and_release_when_unloaded(self):
        env = CpuExoJumpEnv(2, 12, False, 'threaded', 1, 42, geometry='fitted', assist_force=500.)
        worlds, cmd = env.worlds, env.jump_commands
        assist, data = worlds.exoskeleton, worlds.sim.data
        model = worlds.model
        ids = assist.actuators
        np.testing.assert_array_equal(model.actuator_gainprm[ids, 0], [PD_KP, PD_KP])
        np.testing.assert_array_equal(model.actuator_biasprm[ids, 1:3], [[-PD_KP, -PD_KD]] * 2)
        self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        for native in worlds.data:
            for side, nominal in zip(STRUTS, assist.nominal_length.tolist(), strict=True):
                joint = native.joint(f'robot/exo_{side}_slide')
                joint.qpos[:] = 1.0 - nominal
                joint.qvel[:] = .25
            mujoco.mj_forward(model, native)
        worlds.read_native()
        cmd.stage[:] = cmd.EXTEND
        cmd.ground_frames[:] = 4
        cmd.age[:] = .10
        cmd.push_duration[:] = .20
        data.sensordata[:, assist.load] = worlds.device.weight
        data.sensordata[1, assist.load[1]] = 0.
        actions = torch.zeros((2, env.num_actions))
        actions[:, -2:] = torch.tensor([[-1., .2], [2., 1.]])

        def copy_controls(start, end, controls, forces):
            for row in range(start, end):
                worlds.data[row].ctrl[:] = controls[row]

        with patch.object(worlds, '_step_range', side_effect=copy_controls):
            worlds.step(env.commands(actions))
        expected = torch.tensor([[0., PD_KP * .09 - PD_KD * .25], [500., 0.]])
        torch.testing.assert_close(data.ctrl[:, ids] + assist.nominal_length,
                                   torch.tensor([[PD_RELEASE_LENGTH_M, 1.09], [1.15, PD_RELEASE_LENGTH_M]]))
        torch.testing.assert_close(data.actuator_force[:, ids], expected)
        features = env.get_observations()['policy'][:, -STRUT_OBSERVATIONS:].reshape(2, 2, 5)
        targets = torch.tensor([[1., 1.09], [1.15, 1.15]])
        torch.testing.assert_close(features[..., 2], expected / 500.)
        torch.testing.assert_close(features[..., 3], (targets - MIN_LENGTH_M) / (MAX_LENGTH_M - MIN_LENGTH_M))
        worlds.reset(torch.tensor([True, False]))
        torch.testing.assert_close(data.actuator_force[0, ids], torch.zeros(2))
        torch.testing.assert_close(data.actuator_force[1, ids], expected[1])
        self.assertTrue(all(native.time == 0. for native in worlds.data))

    def test_compact_shoulders_reduce_swing_without_changing_other_actions(self):
        env = CpuExoJumpEnv(2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.)
        controller = env.jump_commands
        device = env.worlds.device
        indices = controller.shoulder_actions
        actions = torch.linspace(-.8, .8, env.num_actions).repeat(2, 1)
        actions[:, indices] = torch.tensor([[-5.], [5.]])
        self.assertTrue(controller.compact_arm_swing)
        compact = env.commands(actions)
        controller.compact_arm_swing = False
        original = env.commands(actions)
        other = [i for i in range(compact.shape[1]) if i not in indices]
        torch.testing.assert_close(compact[:, other], original[:, other])
        scale = device.action_scale[indices]
        old_target = (scale * original[:, indices].clamp(-2., 2.)).clamp(
            device.action_lo[indices], device.action_hi[indices])
        expected = (old_target * torch.tensor([.38, .42, .70] * 2)
                    + torch.tensor([0., -.14, 0., 0., .14, 0.]))
        torch.testing.assert_close(scale * compact[:, indices], expected)
        self.assertLess(float(expected[:, [0, 3]].abs().max()), .20)

    def test_native_pd_force_override_and_zero_assist(self):
        for cap in (80., 0.):
            model = scene_spec(geometry='fitted', exoskeleton=True, exo_pd_force=cap).compile()
            data = mujoco.MjData(model)
            for side in STRUTS:
                name = f'robot/exo_{side}'
                actuator = model.actuator(name)
                data.ctrl[actuator.id] = actuator.ctrlrange[1]
                data.joint(f'{name}_slide').qpos[:] = -.20
                data.joint(f'{name}_slide').qvel[:] = -.10
            mujoco.mj_forward(model, data)
            for side in STRUTS:
                self.assertAlmostEqual(data.actuator_force[model.actuator(f'robot/exo_{side}').id], cap)
            self.assertEqual(data.time, 0.)

    def test_legacy_weights_preserve_body_outputs_when_strut_channels_are_added(self):
        config = dict(class_name='rsl_rl.modules.GaussianDistribution', init_std=.25, std_type='scalar')
        old_obs = TensorDict({'policy': torch.randn(3, 159)}, batch_size=[3])
        new_obs = TensorDict({'policy': torch.cat((old_obs['policy'], torch.randn(3, 10)), dim=-1)},
                             batch_size=[3])
        for actor in (True, False):
            kwargs = dict(hidden_dims=(128, 128), activation='elu', obs_normalization=True,
                          distribution_cfg=config.copy() if actor else None)
            size = 24 if actor else 1
            old = MLPModel(old_obs, {'policy': ['policy']}, 'policy', size, **kwargs)
            new = MLPModel(new_obs, {'policy': ['policy']}, 'policy', size + (STRUT_ACTIONS if actor else 0), **kwargs)
            command = -.68 if actor else None
            new.load_state_dict(exo_initial_state(old.state_dict(), new.state_dict(), initial_command=command))
            result = new(new_obs)
            torch.testing.assert_close(result[:, :size], old(old_obs))
            if actor:
                torch.testing.assert_close(result[:, size:], torch.full((3, 2), command))

    def test_work_sharing_credits_loaded_rising_push_and_costs_human_effort(self):
        env = CpuExoJumpEnv(6, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=80.)
        assist, cmd = env.worlds.exoskeleton, env.jump_commands
        data = env.worlds.sim.data
        cmd.stage[:] = cmd.EXTEND
        cmd.stage[2] = cmd.COMPRESS
        data.sensordata[:, assist.load] = env.worlds.device.weight
        data.sensordata[1, assist.load] = 0.
        data.sensordata[:, cmd.com_velocity_xy.stop] = .40
        data.sensordata[3, cmd.com_velocity_xy.stop] = -.30
        data.actuator_force[:, assist.actuators] = 60.
        data.actuator_velocity[:, assist.actuators] = .30
        data.actuator_velocity[4:, assist.actuators] = -.30
        data.actuator_force[:, assist.human_actuators] = 15.
        data.actuator_force[5, assist.human_actuators] = 30.
        data.actuator_velocity[:, assist.human_actuators] = -.20
        assist.observe(cmd.DT)
        self.assertGreater(float(assist.efficiency_rate[0]), float(assist.efficiency_rate[1]))
        torch.testing.assert_close(assist.efficiency_rate[1:5], assist.efficiency_rate[1].expand(4))
        self.assertLess(float(assist.efficiency_rate[5]), float(assist.efficiency_rate[4]))
        torch.testing.assert_close(cmd.efficiency_cost(), torch.zeros(6))
        torch.testing.assert_close(assist.supported_work_j, torch.tensor([36. * cmd.DT, 0., 0., 0., 0., 0.]))
        report = assist.report()
        self.assertEqual(report['human_joint_positive_work_per_env_j'], [0.] * 6)
        self.assertGreater(report['human_joint_absolute_work_per_env_j'][0], 0.)
        self.assertEqual(report['human_leg_torque_peak_per_env_nm'][5], 30.)


if __name__ == '__main__':
    unittest.main()
