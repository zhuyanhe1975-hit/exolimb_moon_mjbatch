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

    def test_persistent_posture_cost_and_push_impulse_reset(self):
        from exoskeleton import ACTION_FORCE_IO

        env = CpuExoJumpEnv(2, 12, False, 'threaded', 1, 42, geometry='fitted',
                            assist_force=500., exo_policy_io=ACTION_FORCE_IO, posture_balance=True)
        assist = env.worlds.exoskeleton
        cmd = env.jump_commands
        data = env.worlds.sim.data
        dt = cmd.DT
        angle = torch.deg2rad(torch.tensor([10., -10.]))
        quat = data.sensordata[:, cmd.torso_quat]
        quat.zero_()
        quat[:, 0] = (angle / 2).cos()
        quat[:, 1] = (angle / 2).sin()
        data.qpos[:, assist.waist_roll_qadr] = angle
        torch.testing.assert_close(assist._persistent_roll_cost(dt), torch.zeros(2))
        for _ in range(120):
            cost = assist._persistent_roll_cost(dt)
        torch.testing.assert_close(cost[0], cost[1])
        self.assertGreater(cost[0].item(), 0.)
        assist.reset_controls(torch.tensor([True, False]))
        self.assertEqual(assist.filtered_roll[0].abs().sum().item(), 0.)
        self.assertGreater(assist.filtered_roll[1].abs().sum().item(), 0.)
        data.sensordata[:, assist.load] = torch.tensor([[1., 0.], [0., 1.]]) * env.worlds.device.weight
        cmd.stage.fill_(cmd.EXTEND)
        assist._bilateral_assist_cost(torch.zeros_like(assist.peak_force_n), dt)
        torch.testing.assert_close(assist.push_boot_impulse, torch.tensor([[dt, 0.], [0., dt]]))
        cmd.stage.fill_(cmd.FLIGHT)
        assist._bilateral_assist_cost(torch.zeros_like(assist.peak_force_n), dt)
        self.assertEqual(assist.push_boot_impulse.abs().sum().item(), 0.)

    def test_stance_transition_uses_world_vertical_regions_and_free_force_actions(self):
        from exoskeleton import ACTION_FORCE_IO

        env = CpuExoJumpEnv(2, 12, False, 'threaded', 1, 42, geometry='fitted',
                            assist_force=500., exo_policy_io=ACTION_FORCE_IO, stance_transition=True)
        worlds = env.worlds
        assist = worlds.exoskeleton
        model = worlds.model
        for data in worlds.data:
            data.qpos[2] -= .04
            data.qvel[0] = .6
            mujoco.mj_forward(model, data)
        worlds.read_native()
        expected = np.zeros((2, 2, 2))
        ground = model.geom('terrain').id
        probes = env.jump_commands.sole_probes().numpy()
        for w, data in enumerate(worlds.data):
            for side, name in enumerate(('left', 'right')):
                sole = model.geom(f'robot/{name}_sole').id
                heel, toe = probes[w, side]
                axis = (toe - heel) * .5
                for i, contact in enumerate(data.contact):
                    if set(contact.geom) != {ground, sole}:
                        continue
                    region = np.dot(contact.pos - (heel + toe) * .5, axis) / np.dot(axis, axis)
                    if abs(region) <= .33:
                        continue
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, wrench)
                    sign = 1 if contact.geom[1] == sole else -1
                    expected[w, side, int(region > 0)] += sign * (contact.frame.reshape(3, 3).T @ wrench[:3])[2]
        self.assertGreater(expected.max(), 1.)
        np.testing.assert_allclose(assist._regional_upward_loads().numpy(),
                                   expected / worlds.device.weight, rtol=1e-5, atol=1e-5)
        sim = worlds.sim.data
        sim.sensordata[:, assist.load] = worlds.device.weight
        vz = env.jump_commands.com_velocity_xy.stop
        sim.sensordata[:, vz] = -.3
        from unittest.mock import patch
        region_loads = torch.tensor([[[.15, .15], [.15, .15]], [[.30, 0.], [0., .30]]])
        with patch.object(assist, '_regional_upward_loads', return_value=region_loads):
            torch.testing.assert_close(assist._stance_transition_rate(), torch.tensor([22., 0.]))
            sim.sensordata[:, vz] = 0.
            torch.testing.assert_close(assist._stance_transition_rate(), torch.zeros(2))
        sim.sensordata[:, vz] = .3
        sim.actuator_velocity[:, assist.actuators] = torch.tensor([[.2, -.2], [-.2, .2]])
        torch.testing.assert_close(assist._stance_transition_rate(), torch.tensor([6.5, 6.5]))
        sim.sensordata[:, assist.load] = 0.
        torch.testing.assert_close(assist._stance_transition_rate(), torch.zeros(2))
        torch.testing.assert_close(assist.controls(torch.ones((2,2))), torch.full((2,2), 500.))

    def test_vertical_boot_feedback_and_touchdown_window(self):
        from exoskeleton import ACTION_FORCE_IO

        env = CpuExoJumpEnv(2, 12, False, 'threaded', 1, 42, geometry='fitted', assist_force=500.,
                            exo_policy_io=ACTION_FORCE_IO, vertical_support=True)
        worlds = env.worlds
        model = worlds.model
        native = worlds.data[0]
        native.qpos[2] -= .04
        native.qvel[0] = .6
        mujoco.mj_forward(model, native)
        worlds.read_native()
        assist = worlds.exoskeleton
        ground = model.geom('terrain').id
        expected = np.zeros((2, 3))
        for side, name in enumerate(('left', 'right')):
            boot = model.geom(f'robot/{name}_sole').id
            for index, contact in enumerate(native.contact):
                if set(contact.geom) != {ground, boot}:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, native, index, wrench)
                sign = 1 if contact.geom[1] == boot else -1
                expected[side] += sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
        self.assertGreater(np.max(expected[:, 2]), 1.)
        self.assertGreater(np.linalg.norm(expected[:, :2]), 1.)
        sensors = worlds.sim.data.sensordata
        np.testing.assert_allclose(sensors[0, assist.reaction_columns], expected, atol=1e-3, rtol=1e-5)
        np.testing.assert_allclose(sensors[0, assist.load], expected[:, 2].clip(min=0), atol=1e-3)
        # The two worlds receive the same synthetic contact. Reset only the first
        # world in the middle of the 50 ms window.
        env.jump_commands.stage.fill_(env.jump_commands.COMPRESS)
        ground_mask = torch.zeros_like(assist.peak_force_n, dtype=torch.bool)
        dt = env.jump_commands.DT
        for _ in range(2):
            assist._touchdown_cost(ground_mask, dt)
        ground_mask[:, 0] = True
        sensors[:, assist.reaction_columns[0][2]] = 4 * worlds.device.weight
        assist._touchdown_cost(ground_mask, dt)
        assist.reset_controls(torch.tensor([True, False]))
        assist._touchdown_cost(ground_mask, dt)
        cost = assist._touchdown_cost(ground_mask, dt)
        self.assertEqual(assist.completed_touchdowns.tolist(), [[0, 0], [1, 0]])
        self.assertAlmostEqual(float(assist.completed_touchdown_impulse_ns[1, 0]),
                               .2 * worlds.device.weight, places=4)
        torch.testing.assert_close(cost * dt, torch.tensor([0., .2]))

    def test_sharing_charges_cumulative_impulse_and_allows_staggered_contact(self):
        from exoskeleton import ACTION_FORCE_IO

        env = CpuExoJumpEnv(2, 12, False, 'threaded', 1, 42, geometry='fitted',
                            assist_force=500., exo_policy_io=ACTION_FORCE_IO, load_sharing=True)
        assist = env.worlds.exoskeleton
        data = env.worlds.sim.data
        cmd = env.jump_commands
        dt = cmd.DT
        weight = env.worlds.device.weight
        # The worlds are left/right reflections with identical absolute strut work.
        loads = torch.tensor([[1., 0.], [0., 1.]])
        force = 100.
        data.actuator_force[:, assist.actuators] = force
        cmd.stage.fill_(cmd.COMPRESS)
        def sample(load):
            data.sensordata[:, assist.load] = load * weight
            return assist._bilateral_assist_cost(torch.ones_like(load), dt)

        first = sample(loads)
        self.assertGreater(float(first[0]), 0.)
        torch.testing.assert_close(first[0], first[1])
        torch.testing.assert_close(sample(loads.flip(-1)), -first)
        torch.testing.assert_close(assist.phase_impulse_ns[:, 0], torch.full_like(loads, force * dt))
        cmd.stage.fill_(cmd.EXTEND)
        torch.testing.assert_close(sample(loads), first)
        assist.reset_controls(torch.tensor([True, False]))
        self.assertEqual(float(assist.phase_balance[0]), 0.)
        self.assertGreater(float(assist.phase_balance[1]), 0.)
        torch.testing.assert_close(assist.phase_impulse_ns[:, 1], loads * force * dt)
        # A stationary overloaded boot incurs an impact cost even with equal forces.
        both = torch.ones_like(loads)
        cmd.stage.fill_(cmd.FLIGHT)
        data.sensordata[:, cmd.com_velocity_xy.stop] = 1.
        torch.testing.assert_close(sample(both * 8.), torch.full_like(first, 4.))
        torch.testing.assert_close(sample(both * 0.), torch.zeros_like(first))
        torch.testing.assert_close(assist.controls(both), both * 500.)

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
            kwargs = dict(hidden_dims=(128, 128), activation='elu', obs_normalization=True)
            size = 24 if actor else 1
            old = MLPModel(old_obs, {'policy': ['policy']}, 'policy', size,
                           distribution_cfg=config.copy() if actor else None, **kwargs)
            new = MLPModel(new_obs, {'policy': ['policy']}, 'policy', size + (STRUT_ACTIONS if actor else 0),
                           distribution_cfg=config.copy() if actor else None, **kwargs)
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

    def test_high_assist_reward_overrides_scale_cot_unloaded_and_work_share(self):
        from exoskeleton import ACTION_FORCE_IO
        overrides = {
            'human_metabolic_proxy_power': 6.5,
            'human_mean_squared_torque': 3.0,
            'peak_leg_squared_torque': 3.0,
            'unloaded_strut_force_linear': 1.5,
            'cot_energy_scale': True,
            'assist_work_share_weight': 10.0,
        }
        env = CpuExoJumpEnv(
            4, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=ACTION_FORCE_IO, load_sharing=True, vertical_support=True,
            posture_balance=True, economical_hops=True, reward_overrides=overrides,
            speed_track_boost=8.0,
        )
        assist, cmd = env.worlds.exoskeleton, env.jump_commands
        data = env.worlds.sim.data
        self.assertEqual(env.cfg['exo_efficiency_reward']['human_metabolic_proxy_power'], 6.5)
        self.assertEqual(env.cfg['speed_track_boost'], 8.0)
        cmd.stage[:] = cmd.EXTEND
        cmd.target_speed[:] = torch.tensor([1.0, 1.0, 0.5, 1.5])
        data.sensordata[:, assist.load] = env.worlds.device.weight
        data.sensordata[1, assist.load] = 0.
        data.sensordata[:, cmd.com_velocity_xy.stop] = .40
        data.actuator_force[:, assist.actuators] = 100.
        data.actuator_velocity[:, assist.actuators] = .30
        data.actuator_force[2:, assist.actuators] = 0.
        data.actuator_velocity[2:, assist.actuators] = 0.
        data.actuator_force[:, assist.human_actuators] = 20.
        data.actuator_velocity[:, assist.human_actuators] = .25
        assist.observe(cmd.DT)
        self.assertGreater(float(assist.efficiency_rate[0]), float(assist.efficiency_rate[1]))
        self.assertGreater(float(assist.efficiency_rate[3]), float(assist.efficiency_rate[2]))

    def test_suit_ankle_pitch_profile_limits_and_bellows_dynamics(self):
        from math import radians
        from exoskeleton import ACTION_FORCE_IO
        from moon_scene import (SUIT_ANKLE_PITCH_DAMPING, SUIT_ANKLE_PITCH_LIMITS,
                                SUIT_ANKLE_PITCH_PROFILE, SUIT_ANKLE_PITCH_STIFFNESS)

        env = CpuExoJumpEnv(
            2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=ACTION_FORCE_IO, load_sharing=True, vertical_support=True,
            suit_ankle_pitch=True,
        )
        self.assertEqual(env.cfg['ankle_pitch_profile'], SUIT_ANKLE_PITCH_PROFILE)
        model = env.worlds.model
        device = env.worlds.device
        for side in ('Left', 'Right'):
            joint = model.joint(f'robot/{side}Ankle')
            np.testing.assert_allclose(joint.range, SUIT_ANKLE_PITCH_LIMITS, atol=1e-6)
            self.assertAlmostEqual(float(model.dof_damping[int(joint.dofadr[0])]), SUIT_ANKLE_PITCH_DAMPING)
            self.assertAlmostEqual(float(model.jnt_stiffness[joint.id]), SUIT_ANKLE_PITCH_STIFFNESS)
            self.assertAlmostEqual(float(model.qpos_spring[int(joint.qposadr[0])]), 0.)
            idx = device.action_index(f'robot/{side}Ankle')
            self.assertAlmostEqual(float(device.action_lo[idx]), -radians(18.), places=6)
            self.assertAlmostEqual(float(device.action_hi[idx]), radians(15.), places=6)
            self.assertAlmostEqual(float(device.action_scale[idx]), .35, places=6)
            self.assertAlmostEqual(float(env.jump_commands.prior[idx] * device.action_scale[idx]), -.14, places=6)

    def test_slope_adaptive_posture_shifts_lean_and_unlocks_waist_pitch(self):
        from exoskeleton import ACTION_FORCE_IO
        env = CpuExoJumpEnv(
            2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=ACTION_FORCE_IO, load_sharing=True, vertical_support=True,
            suit_ankle_pitch=True, slope_adaptive_posture=True,
        )
        self.assertTrue(env.cfg['slope_adaptive_posture'])
        device, cmd = env.worlds.device, env.jump_commands
        waist = device.action_index('robot/WaistPitch')
        self.assertAlmostEqual(float(device.action_scale[waist]), .35, places=6)
        self.assertGreater(float(device.action_hi[waist]), .60)
        # Verify uphill slope produces positive forward torso shift and downhill produces negative shift
        cmd.target_speed[:] = 1.0
        with patch.object(cmd, 'terrain_slope', return_value=torch.tensor([.20, -.20])):
            shift = cmd.slope_torso_shift()
            self.assertGreater(float(shift[0]), .30)
            self.assertLess(float(shift[1]), -.15)
            # An upright torso (0 deg) has negative relative pitch on uphill (needs forward lean)
            # and positive relative pitch on downhill
            upright = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
            rel = cmd.posture_pitch(upright)
            self.assertLess(float(rel[0]), -.30)
            self.assertGreater(float(rel[1]), .15)
            # Stationary target uses a gentler (0.65x) slope shift to preserve static balance
            cmd.target_speed[:] = 0.0
            static_shift = cmd.slope_torso_shift()
            torch.testing.assert_close(static_shift, .65 * shift)

    def test_suit_major_joints_and_relaxed_propulsion_profiles(self):
        from math import radians
        from exoskeleton import ACTION_FORCE_IO
        from moon_scene import RELAXED_PROPULSION_PROFILE, SUIT_MAJOR_JOINTS_PROFILE

        env = CpuExoJumpEnv(
            2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=ACTION_FORCE_IO, load_sharing=True, vertical_support=True,
            suit_ankle_pitch=True, slope_adaptive_posture=True,
            suit_major_joints=True, relax_propulsion=True,
        )
        self.assertEqual(env.cfg['suit_major_joints_profile'], SUIT_MAJOR_JOINTS_PROFILE)
        self.assertEqual(env.cfg['propulsion_profile'], RELAXED_PROPULSION_PROFILE)
        model, device = env.worlds.model, env.worlds.device
        # Waist & HipYaw pressurized EVA damping, stiffness, and tightened ranges
        wy = model.joint('robot/WaistYaw')
        self.assertAlmostEqual(float(model.dof_damping[wy.dofadr[0]]), 16., places=5)
        self.assertAlmostEqual(float(model.jnt_stiffness[wy.id]), 35., places=5)
        np.testing.assert_allclose(model.jnt_range[wy.id], [-radians(15.), radians(15.)], atol=1e-6)
        wp = model.joint('robot/WaistPitch')
        self.assertAlmostEqual(float(model.dof_damping[wp.dofadr[0]]), 12., places=5)
        self.assertAlmostEqual(float(model.jnt_stiffness[wp.id]), 12., places=5)
        # Relaxed leg propulsion PD brake (kd 25->12), higher peak torque (260 Nm), and unlocked HipPitch/Knee bounds
        lk = model.actuator('robot/LeftKnee')
        self.assertAlmostEqual(float(model.actuator_gainprm[lk.id, 0]), 380., places=5)
        self.assertAlmostEqual(float(model.actuator_biasprm[lk.id, 2]), -12., places=5)
        np.testing.assert_allclose(model.actuator_forcerange[lk.id], [-260., 260.], atol=1e-6)
        hip_idx = device.action_index('robot/LeftHipPitch')
        knee_idx = device.action_index('robot/LeftKnee')
        self.assertAlmostEqual(float(device.action_hi[hip_idx]), radians(30.), places=5)
        self.assertAlmostEqual(float(device.action_lo[knee_idx]), -.18, places=5)
        self.assertAlmostEqual(float(device.action_hi[knee_idx]), 1.22, places=5)

    def test_apollo_mass_waist_clevis_and_impedance_force_assist(self):
        from exoskeleton import (IMPEDANCE_FORCE_IO, MECHANISM, STRUT_EXTRA_OBS,
                                 WAIST_CLEVIS_MECHANISM, WAIST_MAX_LENGTH_M, WAIST_MIN_LENGTH_M)
        from moon_scene import APOLLO_CHEST_MASS_KG, APOLLO_MASS_PROFILE, APOLLO_PACK_MASS_KG, scene_spec

        # 1. Compare ankle roll moment arm: waist_clevis_v4 mid-plane vs legacy hooke_slide_ball_v3
        def max_roll_moment_per_n(mech):
            spec = scene_spec(geometry='fitted', sole_probes=True, sole_collision=True,
                              sole_edge_probes=True, exoskeleton=True, exo_mechanism=mech,
                              suit_ankle_roll=True, apollo_mass=(mech == WAIST_CLEVIS_MECHANISM))
            m = spec.compile()
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            arms = []
            for side in ('left', 'right'):
                pack = d.sensor(f'robot/exo_{side}_pack_pos').data
                boot = d.sensor(f'robot/exo_{side}_boot_pos').data
                ankle = d.sensor(f'robot/exo_{side}_ankle_pos').data
                axis = d.sensor(f'robot/exo_{side}_ankle_axis').data
                direction = (boot - pack) / np.linalg.norm(boot - pack)
                arms.append(abs(float(np.dot(np.cross(boot - ankle, direction), axis))))
            return max(arms), m

        legacy_arm, legacy_m = max_roll_moment_per_n(MECHANISM)
        clevis_arm, clevis_m = max_roll_moment_per_n(WAIST_CLEVIS_MECHANISM)
        # Mid-plane bifurcated boot clevis cuts ankle roll moment arm dramatically (>70% reduction)
        self.assertLess(clevis_arm, .30 * legacy_arm)
        # Apollo mass proportions: Torso = 82.5 kg (chest 31.5 + PLSS/OPS 51.0), Head = 8.5 kg, total ~157 kg
        torso_id = clevis_m.body('robot/Torso').id
        self.assertAlmostEqual(float(clevis_m.body_mass[torso_id]), APOLLO_CHEST_MASS_KG + APOLLO_PACK_MASS_KG, places=4)
        self.assertLess(float(clevis_m.body_ipos[torso_id, 0]), float(legacy_m.body_ipos[legacy_m.body('robot/Torso').id, 0]) - .04)
        self.assertAlmostEqual(float(clevis_m.body_mass[clevis_m.body('robot/Head').id]), 8.5, places=4)
        pelvis_id = clevis_m.body('robot/Pelvis').id
        self.assertAlmostEqual(float(clevis_m.body_subtreemass[pelvis_id]), 156.96, delta=1.5)
        # Upper mount attaches to Pelvis (waist harness), not Torso
        self.assertEqual(int(clevis_m.body('robot/exo_left_yoke').parentid[0]), pelvis_id)

        # 2. Impedance force control: 168-dim obs, grounded load gate, SLIP spring-damper, bilateral share
        env = CpuExoJumpEnv(
            2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=IMPEDANCE_FORCE_IO, exo_mechanism=WAIST_CLEVIS_MECHANISM,
            apollo_mass=True, load_sharing=True, vertical_support=True, posture_balance=True,
            suit_ankle_pitch=True, slope_adaptive_posture=True,
            suit_major_joints=True, relax_propulsion=True,
            reward_overrides={'assist_work_share_weight': 20.0},
        )
        self.assertEqual(env.cfg['apollo_mass_profile'], APOLLO_MASS_PROFILE)
        self.assertEqual(env.cfg['exo_mechanism'], WAIST_CLEVIS_MECHANISM)
        self.assertEqual(env.cfg['exo_policy_io'], IMPEDANCE_FORCE_IO)
        obs = env.get_observations()['policy']
        self.assertEqual(obs.shape[-1], 169 + STRUT_EXTRA_OBS)
        assist = env.worlds.exoskeleton
        self.assertEqual(assist.mechanism, WAIST_CLEVIS_MECHANISM)
        torch.testing.assert_close(assist.min_length_m, torch.full((2,), WAIST_MIN_LENGTH_M), atol=1e-4, rtol=0)
        torch.testing.assert_close(assist.max_length_m, torch.full((2,), WAIST_MAX_LENGTH_M), atol=1e-4, rtol=0)

        data = env.worlds.sim.data
        # Airborne (load = 0) -> zero force even at max policy command
        data.sensordata[:, assist.load] = 0.
        data.sensordata[:, assist.stroke] = -.15
        data.actuator_velocity[:, assist.actuators] = -.4
        air_force = assist.controls(torch.ones((2, 2)))
        torch.testing.assert_close(air_force, torch.zeros((2, 2)))
        # Loaded + compressed + descending -> stronger elastic+damping catch than uncompressed
        data.sensordata[:, assist.load] = env.worlds.device.weight
        comp_force = assist.controls(torch.full((2, 2), .4))  # u = 0.7
        data.sensordata[:, assist.stroke] = 0.
        data.actuator_velocity[:, assist.actuators] = 0.
        ext_force = assist.controls(torch.full((2, 2), .4))
        self.assertTrue(bool((comp_force > ext_force).all()))
        # Bilateral push share rewards symmetric L+R more than unilateral same-total power
        cmd = env.jump_commands
        cmd.stage[:] = cmd.EXTEND
        cmd.target_speed[:] = 1.0
        data.sensordata[:, cmd.com_velocity_xy.stop] = .40
        data.actuator_force[:, assist.human_actuators] = 10.
        data.actuator_velocity[:, assist.human_actuators] = .20
        data.actuator_velocity[:, assist.actuators] = .30
        data.actuator_force[0, assist.actuators] = torch.tensor([150., 150.])  # symmetric 90W
        data.actuator_force[1, assist.actuators] = torch.tensor([300., 0.])    # unilateral 90W
        assist.last_request[0] = torch.tensor([.6, .6])
        assist.last_request[1] = torch.tensor([1.0, 0.0])
        assist.observe(cmd.DT)
        self.assertGreater(float(assist.efficiency_rate[0]), float(assist.efficiency_rate[1]))

        # 3. Wangyu (望宇) lunar EVA profile: ~169 kg total (70 kg astronaut + 97 kg suit + 2 kg exo),
        #    91 kg Torso (39 kg HUT + 52 kg PLSS), Wangyu bearing/bellows mobility, 0.82 regolith boot friction.
        from math import radians
        from moon_scene import WANGYU_EVA_PROFILE, WANGYU_SOLE_FRICTION
        w_spec = scene_spec(geometry='fitted', sole_probes=True, sole_collision=True, sole_edge_probes=True,
                            exoskeleton=True, exo_mechanism=WAIST_CLEVIS_MECHANISM,
                            suit_ankle_roll=True, suit_ankle_pitch=True, suit_major_joints=True,
                            relax_propulsion=True, mass_profile=WANGYU_EVA_PROFILE)
        wm = w_spec.compile()
        self.assertAlmostEqual(float(wm.body_mass[wm.body('robot/Torso').id]), 91.0, places=4)
        self.assertAlmostEqual(float(wm.body_subtreemass[wm.body('robot/Pelvis').id]), 168.66, delta=1.5)
        self.assertAlmostEqual(float(wm.geom_friction[wm.geom('robot/left_sole').id, 0]), WANGYU_SOLE_FRICTION, places=5)
        np.testing.assert_allclose(wm.jnt_range[wm.joint('robot/WaistPitch').id], [-radians(15.), radians(35.)], atol=1e-6)
        np.testing.assert_allclose(wm.jnt_range[wm.joint('robot/LeftAnkle').id], [-radians(20.), radians(18.)], atol=1e-6)

        # 4. Option B: Wangyu HUT bottom rigid ring upper mount + centered mid-plane boot Y-fork (hut_ring_clevis_v5)
        from exoskeleton import HUT_MAX_LENGTH_M, HUT_MIN_LENGTH_M, HUT_RING_CLEVIS_MECHANISM
        hut_arm, hut_m = max_roll_moment_per_n(HUT_RING_CLEVIS_MECHANISM)
        self.assertLess(hut_arm, .25 * legacy_arm)
        self.assertEqual(int(hut_m.body('robot/exo_left_yoke').parentid[0]), hut_m.body('robot/Torso').id)
        hut_env = CpuExoJumpEnv(
            2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=IMPEDANCE_FORCE_IO, exo_mechanism=HUT_RING_CLEVIS_MECHANISM,
            mass_profile=WANGYU_EVA_PROFILE, load_sharing=True, vertical_support=True, posture_balance=True,
            suit_ankle_pitch=True, slope_adaptive_posture=True,
            suit_major_joints=True, relax_propulsion=True,
        )
        self.assertEqual(hut_env.worlds.exoskeleton.mechanism, HUT_RING_CLEVIS_MECHANISM)
        torch.testing.assert_close(hut_env.worlds.exoskeleton.min_length_m, torch.full((2,), HUT_MIN_LENGTH_M), atol=1e-4, rtol=0)
        torch.testing.assert_close(hut_env.worlds.exoskeleton.max_length_m, torch.full((2,), HUT_MAX_LENGTH_M), atol=1e-4, rtol=0)
        # Closed-loop HUT WaistRoll safety guard: 0 N at v=0, full 500 N when upright,
        # tapers unilateral differential to 280 N when pushing further into an excessive >13 deg roll.
        h_assist = hut_env.worlds.exoskeleton
        h_data = hut_env.worlds.sim.data
        h_data.sensordata[:, h_assist.stroke] = -.12
        h_data.actuator_velocity[:, h_assist.actuators] = -.3
        h_data.sensordata[0, h_assist.load] = torch.tensor([0., hut_env.worlds.device.weight])
        h_data.sensordata[1, h_assist.load] = hut_env.worlds.device.weight
        hut_env.jump_commands.target_speed[:] = 0.0
        torch.testing.assert_close(h_assist.controls(torch.ones((2, 2))), torch.zeros((2, 2)))
        hut_env.jump_commands.target_speed[:] = 1.0
        h_data.qpos[:, h_assist.waist_roll_qpos] = 0.0
        upright_f = h_assist.controls(torch.ones((2, 2)))
        self.assertAlmostEqual(float(upright_f[0, 0]), 0., places=4)
        self.assertAlmostEqual(float(upright_f[0, 1]), 500., places=3)
        torch.testing.assert_close(upright_f[1], torch.full((2,), 500.), atol=1e-3, rtol=0)
        # Excessive negative roll (-13 deg) -> right strut (-y) worsening roll is tapered to 280 N
        # while bilateral loading still delivers full 500 N per strut.
        h_data.qpos[:, h_assist.waist_roll_qpos] = -radians(13.)
        rolled_f = h_assist.controls(torch.ones((2, 2)))
        self.assertAlmostEqual(float(rolled_f[0, 1]), 280., places=3)
        torch.testing.assert_close(rolled_f[1], torch.full((2,), 500.), atol=1e-3, rtol=0)

        # 5. Aft HUT Ring + Ultra-Simple Coaxial Outer-Sole Ball + Active Ankle-Roll Force Governor (hut_ball_aft_v6)
        from exoskeleton import (AFT_BALL_MAX_LENGTH_M, AFT_BALL_MIN_LENGTH_M,
                                 HUT_BALL_AFT_MECHANISM, fit_struts_to_pose)
        b_arm, b_m = max_roll_moment_per_n(HUT_BALL_AFT_MECHANISM)
        self.assertGreater(b_arm, .05)  # simplified outer-sole ball joint (~9 cm arm)
        self.assertEqual(int(b_m.body('robot/exo_left_yoke').parentid[0]), b_m.body('robot/Torso').id)
        # Verify 0.000 mm rod/cylinder coaxiality & 0.000 mm ball/socket closure across deep squat + ankle roll
        bd = mujoco.MjData(b_m)
        bd.qpos[b_m.joint('robot/WaistPitch').qposadr[0]] = radians(16.)
        bd.qpos[b_m.joint('robot/LeftHipPitch').qposadr[0]] = radians(-42.)
        bd.qpos[b_m.joint('robot/LeftKnee').qposadr[0]] = radians(65.)
        bd.qpos[b_m.joint('robot/LeftAnkle').qposadr[0]] = radians(-18.)
        bd.qpos[b_m.joint('robot/LeftAnkleRoll').qposadr[0]] = radians(8.)
        fit_struts_to_pose(b_m, bd)
        mujoco.mj_forward(b_m, bd)
        pa = bd.site('robot/exo_left_ball_pin').xpos
        pb = bd.site('robot/exo_left_boot').xpos
        self.assertLess(float(np.linalg.norm(pa - pb)), 1e-4)
        p_out = bd.geom('robot/exo_left_outer_tube').xpos
        p_in = bd.geom('robot/exo_left_sliding_shaft').xpos
        z_out = bd.geom('robot/exo_left_outer_tube').xmat.reshape(3, 3)[:, 2]
        coax_err = float(np.linalg.norm((p_in - p_out) - np.dot(p_in - p_out, z_out) * z_out))
        self.assertLess(coax_err, 1e-6)
        b_env = CpuExoJumpEnv(
            2, 12, False, 'mjbatch', 2, 42, geometry='fitted', assist_force=500.,
            exo_policy_io=IMPEDANCE_FORCE_IO, exo_mechanism=HUT_BALL_AFT_MECHANISM,
            mass_profile=WANGYU_EVA_PROFILE, load_sharing=True, vertical_support=True, posture_balance=True,
            suit_ankle_pitch=True, slope_adaptive_posture=True,
            suit_major_joints=True, relax_propulsion=True,
        )
        b_assist = b_env.worlds.exoskeleton
        self.assertEqual(b_assist.mechanism, HUT_BALL_AFT_MECHANISM)
        torch.testing.assert_close(b_assist.min_length_m, torch.full((2,), AFT_BALL_MIN_LENGTH_M), atol=1e-4, rtol=0)
        torch.testing.assert_close(b_assist.max_length_m, torch.full((2,), AFT_BALL_MAX_LENGTH_M), atol=1e-4, rtol=0)
        # Verify closed-loop Ankle Inversion/Eversion Force Governor:
        # 1) Static steady stance (even on 5 deg side-slope, roll_vel=0) -> full assist (>450 N)
        # 2) Active rapid eversion (roll_vel = [-90, +90] deg/s) -> governor shuts off force to 0 N!
        # 3) Opposite roll velocity (inversion) -> strut push HELPS restore level -> full assist!
        b_data = b_env.worlds.sim.data
        b_env.jump_commands.target_speed[:] = 1.0
        b_data.qpos[:, b_assist.waist_roll_qpos] = 0.0
        b_data.sensordata[:, b_assist.stroke] = -.12
        b_data.actuator_velocity[:, b_assist.actuators] = -.3
        b_data.sensordata[:, b_assist.load] = 3.0 * b_env.worlds.device.weight
        b_data.qpos[:, b_assist.ankle_roll_qpos] = torch.tensor([-radians(5.), radians(5.)])
        b_data.qvel[:, b_assist.ankle_roll_dof] = 0.0
        flat_f = b_assist.controls(torch.ones((2, 2)))
        self.assertGreater(float(flat_f[0, 0]), 450.)
        self.assertGreater(float(flat_f[0, 1]), 450.)
        b_data.qvel[:, b_assist.ankle_roll_dof] = torch.tensor([-radians(90.), radians(90.)])
        evert_f = b_assist.controls(torch.ones((2, 2)))
        torch.testing.assert_close(evert_f, torch.zeros((2, 2)), atol=1e-3, rtol=0)
        b_data.qvel[:, b_assist.ankle_roll_dof] = torch.tensor([radians(90.), -radians(90.)])
        restoring_f = b_assist.controls(torch.ones((2, 2)))
        self.assertGreater(float(restoring_f[0, 0]), 450.)
        self.assertGreater(float(restoring_f[0, 1]), 450.)


if __name__ == '__main__':
    unittest.main()
