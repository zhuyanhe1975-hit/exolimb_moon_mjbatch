"""Static, kinematic, reference, anti-penetration, DeepMimic tracking, lookahead, and parity checks for Wangyu-Replant-v11."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from exoskeleton import (DETACHED_HINGE_MAX_RAD, DETACHED_HINGE_MIN_RAD,
                         DETACHED_ROLL_LIMIT_RAD, HUT_HINGE_HOOF_MECHANISM, mount_names)
from moon_scene import WANGYU_EVA_PROFILE, scene_spec
from moon_viewer import display_spec
from replant import BEST_REPLANT_CHECKPOINT, latest_replant_checkpoint
from wangyu_replant import (ARM_ACTUATORS, CONTACT_GEOMS, GEOMETRY, KEY_BODY_GEOMS,
                            MECHANISM, REF_DIR, RESIDUAL_ARM_RAD, RESIDUAL_LEG_RAD,
                            RESIDUAL_STRUT_RAD, RESIDUAL_STRUT_ROLL_RAD,
                            RESIDUAL_STRUT_SLIDE_M, SIDES, STRUT_INWARD_ROLL_MAX_RAD,
                            TASK, Recovery, ReplantEnv, actuator_residual_scales,
                            build_model, build_spec, load_replant_reference,
                            mimic_reward_np, quat_rot_error_np)


def _collides(m, g1: str, g2: str) -> bool:
    a, b = m.geom(g1), m.geom(g2)
    return bool((int(a.contype[0]) & int(b.conaffinity[0])) or (int(b.contype[0]) & int(a.conaffinity[0])))


class ReplantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = build_spec(training_sensors=False)
        cls.model = cls.spec.compile()
        cls.train_model = build_model(training_sensors=True)
        cls.ref = load_replant_reference()

    def test_strut_anti_penetration_sensors_and_small_residual_bounds(self):
        attached = scene_spec(
            geometry=GEOMETRY, mass_profile=WANGYU_EVA_PROFILE,
            suit_ankle_roll=True, suit_ankle_pitch=True, suit_major_joints=True,
            relax_propulsion=True, exoskeleton=True, exo_mechanism=HUT_HINGE_HOOF_MECHANISM,
        ).compile()
        m = self.model
        self.assertEqual(attached.neq, 2)
        self.assertEqual(m.neq, 2)
        self.assertTrue(np.all(m.eq_active0 == 0))
        self.assertEqual(m.nu, 29)
        self.assertEqual(m.nhfield, 0)
        self.assertEqual(m.geom('terrain').type[0], mujoco.mjtGeom.mjGEOM_PLANE)

        arm_set = set(ARM_ACTUATORS)
        scales = actuator_residual_scales(m)
        for i in range(m.nu):
            name = m.actuator(i).name
            if name in ('robot/exo_left', 'robot/exo_right'):
                self.assertLessEqual(scales[i], RESIDUAL_STRUT_SLIDE_M + 1e-9)
            elif name in ('robot/exo_left_roll', 'robot/exo_right_roll'):
                self.assertLessEqual(scales[i], RESIDUAL_STRUT_ROLL_RAD + 1e-9)
            elif name.startswith('robot/exo_'):
                self.assertLessEqual(scales[i], RESIDUAL_STRUT_RAD + 1e-9)
            elif name in arm_set:
                self.assertLessEqual(scales[i], RESIDUAL_ARM_RAD + 1e-9)
            else:
                self.assertLessEqual(scales[i], RESIDUAL_LEG_RAD + 1e-9)

        np.testing.assert_allclose(m.joint('robot/exo_left_hooke_roll').range,
                                   [-DETACHED_ROLL_LIMIT_RAD, STRUT_INWARD_ROLL_MAX_RAD], atol=1e-9)
        np.testing.assert_allclose(m.joint('robot/exo_right_hooke_roll').range,
                                   [-STRUT_INWARD_ROLL_MAX_RAD, DETACHED_ROLL_LIMIT_RAD], atol=1e-9)

        for side in SIDES:
            upper_site = mount_names(side)[0]
            np.testing.assert_allclose(m.site(upper_site).pos, attached.site(upper_site).pos, atol=1e-9)
            np.testing.assert_allclose(m.joint(f'robot/exo_{side}_hinge_pitch').range,
                                       [DETACHED_HINGE_MIN_RAD, DETACHED_HINGE_MAX_RAD], atol=1e-9)
            np.testing.assert_allclose(m.joint(f'robot/exo_{side}_slide').range,
                                       attached.joint(f'robot/exo_{side}_slide').range, atol=1e-9)

        for s_geom in ('robot/exo_left_outer_tube', 'robot/exo_left_sliding_shaft', 'robot/exo_left_hoof_pad'):
            for h_geom in ('robot/left_thigh', 'robot/left_shin', 'robot/left_sole_hull',
                           'robot/left_forearm_hand', 'robot/chest', 'robot/pelvis'):
                self.assertTrue(_collides(m, s_geom, h_geom))
        self.assertTrue(_collides(m, 'terrain', 'robot/exo_left_hoof_pad'))
        self.assertFalse(_collides(m, 'terrain', 'robot/exo_left_outer_tube'))
        self.assertFalse(_collides(m, 'terrain', 'robot/exo_left_hoof_shell'))
        self.assertTrue(_collides(m, 'robot/chest', 'robot/left_forearm_hand'))
        self.assertFalse(_collides(m, 'robot/left_thigh', 'robot/right_thigh'))
        self.assertFalse(_collides(m, 'robot/left_thigh', 'robot/pelvis'))
        self.assertFalse(_collides(m, 'robot/exo_left_sliding_shaft', 'robot/exo_right_sliding_shaft'))

        tm = self.train_model
        for i, gname in enumerate(CONTACT_GEOMS):
            sensor = tm.sensor(f'replant/{gname}')
            self.assertEqual(int(sensor.adr[0]), i * 28)
            self.assertEqual(int(sensor.dim[0]), 28)
        ang = tm.sensor('replant/angmom')
        self.assertEqual(int(ang.adr[0]), 168)
        self.assertEqual(int(ang.dim[0]), 3)

    def test_sha_locked_reference_with_kneel_settle_and_8_key_bodies(self):
        ref = self.ref
        self.assertEqual(ref['qpos'].shape[1], self.model.nq)
        self.assertEqual(ref['qvel'].shape[1], self.model.nv)
        self.assertEqual(ref['ctrl'].shape[1], self.model.nu)
        self.assertEqual(ref['key_targets'].shape, (ref['n_frames'], len(KEY_BODY_GEOMS), 3))
        self.assertLess(float(ref['times'][-1]), 9.5)
        for key in ('qpos', 'qvel', 'ctrl', 'key_targets', 'height', 'upright'):
            self.assertTrue(np.isfinite(ref[key]).all(), f'Non-finite values in reference {key}')

        audit = ref['manifest']['audit']
        self.assertLessEqual(max(audit['prone_hand_gaps_m']), 0.005)
        self.assertGreaterEqual(audit['min_strut_human_gap_m'], 0.025)
        self.assertLessEqual(abs(float(ref['sole_targets'][-1, 0, 0] - ref['sole_targets'][-1, 1, 0])), 0.005)
        k_replant_end = int(np.argmin(np.abs(ref['times'] - float(ref['t_replant_end_s']))))
        leg_gnd = ref['shin_contact_mask'][:k_replant_end + 1] | ref['sole_contact_mask'][:k_replant_end + 1]
        self.assertTrue(np.all(leg_gnd.all(axis=1)))
        self.assertTrue(np.all(ref['sole_contact_mask'][k_replant_end + 25:].any(axis=1)))
        # Verify tighter EVA comfort ROM caps on hip/knee/ankle/waist in ref v8.
        for name, lo_deg, hi_deg in (
            ('robot/LeftHipPitch', -80.5, 18.5),
            ('robot/LeftKnee', 2.5, 100.5),
            ('robot/RightKnee', 2.5, 100.5),
            ('robot/LeftAnkle', -15.5, 14.5),
            ('robot/RightAnkle', -15.5, 14.5),
            ('robot/WaistPitch', -11.5, 22.5),
        ):
            aid = self.model.actuator(name).id
            deg = np.rad2deg(ref['ctrl'][:, aid])
            self.assertGreaterEqual(float(deg.min()), lo_deg, f'{name} below EVA ref min')
            self.assertLessEqual(float(deg.max()), hi_deg, f'{name} above EVA ref max')

        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'ref'
            shutil.copytree(REF_DIR, dst)
            manifest = json.loads((dst / 'manifest.json').read_text())
            manifest['files']['assets/wangyu_replant_ref/motion.npz'] = '0' * 64
            (dst / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_replant_reference(dst)

    def test_deepmimic_obs_signed_rotation_error_and_early_termination(self):
        world = Recovery(self.model, reference=self.ref, terminate_on_ref_div=True)
        self.assertTrue(world.from_prone)
        obs = world.observe()
        self.assertEqual(obs.shape[0], 187)
        self.assertTrue(np.isfinite(obs).all())

        # Signed rotation error distinguishes forward (+20 deg) vs backward (-20 deg) lean!
        q_up = np.array([1., 0., 0., 0.], dtype=np.float64)
        a = np.deg2rad(20.)
        q_fwd = np.array([np.cos(a / 2), 0., np.sin(a / 2), 0.])
        q_bwd = np.array([np.cos(a / 2), 0., -np.sin(a / 2), 0.])
        rv_f, ang_f = quat_rot_error_np(q_fwd, q_up)
        rv_b, ang_b = quat_rot_error_np(q_bwd, q_up)
        self.assertAlmostEqual(ang_f, a, places=5)
        self.assertAlmostEqual(ang_b, a, places=5)
        self.assertLess(rv_f[1], 0.)
        self.assertGreater(rv_b[1], 0.)

        # Zero-residual standing holds cleanly with high DeepMimic reward (> 0.85).
        zero = np.zeros(self.model.nu, dtype=np.float32)
        world.reset(start_frame=self.ref['n_frames'] - 1)
        for _ in range(30):
            r, _, failed = world.step(zero)
            self.assertFalse(failed)
            self.assertGreaterEqual(r, 0.)
            self.assertLessEqual(r, 1.)
        self.assertGreater(r, 0.85)
        self.assertGreater(world.upright(), 0.98)

        # Backward tip-over (> 45 deg rot error) triggers clean DeepMimic early termination!
        world.reset(start_frame=self.ref['n_frames'] - 1)
        world.steps = 20
        b = np.deg2rad(-60.)
        world.d.qpos[3:7] = [np.cos(b / 2), 0., np.sin(b / 2), 0.]
        mujoco.mj_forward(self.model, world.d)
        _, _, failed = world.step(zero)
        self.assertTrue(failed)

    def test_studio_skin_and_classic_physx_display_specs_compile_cleanly(self):
        old_viewer = os.environ.get('MOON_VIEWER')
        old_fil = os.environ.get('MOON_FILAMENT_SCENE')
        try:
            for viewer, viz, expect_skin in (('studio', 'skin', True), ('classic', 'physx', False)):
                os.environ['MOON_VIEWER'] = viewer
                os.environ['MOON_FILAMENT_SCENE'] = '1' if viz == 'skin' else '0'
                shown = display_spec(SimpleNamespace(mj_spec=self.spec, nworld=1)).compile()
                self.assertEqual((shown.nq, shown.nv, shown.nbody, shown.nmocap),
                                 (self.model.nq, self.model.nv, self.model.nbody, self.model.nmocap))
                self.assertEqual(shown.nskin > 0, expect_skin)
                self.assertEqual(shown.nhfield, 0)
        finally:
            if old_viewer is None:
                os.environ.pop('MOON_VIEWER', None)
            else:
                os.environ['MOON_VIEWER'] = old_viewer
            if old_fil is None:
                os.environ.pop('MOON_FILAMENT_SCENE', None)
            else:
                os.environ['MOON_FILAMENT_SCENE'] = old_fil

    def test_latest_replant_checkpoint_filters_by_task_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run = root / '2026-11-24_11-27-34_793540'
            old_run.mkdir()
            (old_run / 'config.json').write_text(json.dumps({'task': 'Wangyu-Replant-v10'}))
            (old_run / 'model_275.pt').write_bytes(b'old')
            self.assertIsNone(latest_replant_checkpoint(root, task=TASK))
            new_run = root / '2026-11-24_11-40-00_000001'
            new_run.mkdir()
            (new_run / 'config.json').write_text(json.dumps({'task': TASK}))
            (new_run / 'model_25.pt').write_bytes(b'new')
            self.assertEqual(latest_replant_checkpoint(root, task=TASK), new_run / 'model_25.pt')

    def test_deepmimic_reward_and_quat_error_cpu_gpu_parity(self):
        import torch
        from warp_replant import mimic_reward_torch, quat_rot_error_torch

        rng = np.random.default_rng(7)
        n = 32
        q_cur = rng.normal(size=(n, 4)).astype(np.float32)
        q_cur /= np.linalg.norm(q_cur, axis=-1, keepdims=True)
        q_ref_quat = rng.normal(size=(n, 4)).astype(np.float32)
        q_ref_quat /= np.linalg.norm(q_ref_quat, axis=-1, keepdims=True)

        cpu_rv, cpu_ang = [], []
        for i in range(n):
            rv, ang = quat_rot_error_np(q_cur[i], q_ref_quat[i])
            cpu_rv.append(rv)
            cpu_ang.append(ang)
        gpu_rv, gpu_ang = quat_rot_error_torch(torch.from_numpy(q_cur), torch.from_numpy(q_ref_quat))
        np.testing.assert_allclose(gpu_rv.numpy(), np.stack(cpu_rv), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(gpu_ang.numpy(), np.array(cpu_ang, dtype=np.float32), atol=1e-5, rtol=1e-5)

        q_act = rng.uniform(-1., 1., size=(n, 29)).astype(np.float32)
        q_ref = rng.uniform(-1., 1., size=(n, 29)).astype(np.float32)
        qd_act = rng.uniform(-2., 2., size=(n, 29)).astype(np.float32)
        qd_ref = rng.uniform(-2., 2., size=(n, 29)).astype(np.float32)
        rpos = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
        rpos_ref = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
        rvel = rng.uniform(-1., 1., size=(n, 6)).astype(np.float32)
        rvel_ref = rng.uniform(-1., 1., size=(n, 6)).astype(np.float32)
        keys = rng.uniform(-0.5, 0.5, size=(n, 8, 3)).astype(np.float32)
        keys_ref = rng.uniform(-0.5, 0.5, size=(n, 8, 3)).astype(np.float32)
        load_bw = rng.uniform(0., 0.8, size=(n, 6)).astype(np.float32)
        tip_c = (np.arange(n) % 2 == 0)
        act_rate = rng.uniform(0., 0.5, size=n).astype(np.float32)
        tau_rms_sq = rng.uniform(0., 0.5, size=n).astype(np.float32)
        tau_peak_sq = rng.uniform(0., 0.8, size=n).astype(np.float32)
        rom_exc_sq = rng.uniform(0., 0.1, size=n).astype(np.float32)
        hoof_slip_sq = rng.uniform(0., 1.5, size=n).astype(np.float32)

        cpu_r = np.array([
            mimic_reward_np(q_act[i], q_ref[i], qd_act[i], qd_ref[i],
                            rpos[i], rpos_ref[i], float(cpu_ang[i]), rvel[i], rvel_ref[i],
                            keys[i], keys_ref[i], load_bw[i], bool(tip_c[i]),
                            float(act_rate[i]), float(tau_rms_sq[i]),
                            float(tau_peak_sq[i]), float(rom_exc_sq[i]),
                            float(hoof_slip_sq[i]))
            for i in range(n)
        ], dtype=np.float32)
        gpu_r = mimic_reward_torch(
            torch.from_numpy(q_act), torch.from_numpy(q_ref),
            torch.from_numpy(qd_act), torch.from_numpy(qd_ref),
            torch.from_numpy(rpos), torch.from_numpy(rpos_ref),
            gpu_ang, torch.from_numpy(rvel), torch.from_numpy(rvel_ref),
            torch.from_numpy(keys), torch.from_numpy(keys_ref),
            torch.from_numpy(load_bw), torch.from_numpy(tip_c),
            torch.from_numpy(act_rate), torch.from_numpy(tau_rms_sq),
            torch.from_numpy(tau_peak_sq), torch.from_numpy(rom_exc_sq),
            torch.from_numpy(hoof_slip_sq),
        ).numpy()
        np.testing.assert_allclose(gpu_r, cpu_r, atol=1e-5, rtol=1e-5)
        self.assertTrue(np.all((gpu_r >= 0.) & (gpu_r <= 1.)))
        # Verify strictly lower reward when grounded hoof slips vs stationary.
        r_still = mimic_reward_np(q_act[0], q_act[0], qd_act[0], qd_act[0],
                                  rpos[0], rpos[0], 0.0, rvel[0], rvel[0],
                                  keys[0], keys[0], np.array([.5, .5, 0., 0., .1, .1]), True,
                                  0.0, 0.0, 0.0, 0.0, 0.0)
        r_skid = mimic_reward_np(q_act[0], q_act[0], qd_act[0], qd_act[0],
                                 rpos[0], rpos[0], 0.0, rvel[0], rvel[0],
                                 keys[0], keys[0], np.array([.5, .5, 0., 0., .1, .1]), True,
                                 0.0, 0.0, 0.0, 0.0, 1.0)
        self.assertLess(r_skid, r_still - 0.05)

        env = ReplantEnv(self.model, 2, reference=self.ref, rsi_prob=0.5)
        _, _, _, extras = env.step(torch.zeros((2, env.num_actions)))
        for key in ('Replant/phase', 'Replant/upright', 'Replant/height',
                    'Replant/hoof_slip_mps', 'Replant/angmom_nms',
                    'Replant/success_rate', 'Replant/prone_success_rate'):
            self.assertIn(key, extras['log'])

    def test_archived_best_bundles_integrity(self):
        import hashlib
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from replant import runner_config

        root = Path(__file__).resolve().parents[1]
        v11_ckpt = root / 'bests' / 'replant_wangyu_hut_hinge_aft_v11_2026-11-24' / 'model_1999.pt'
        self.assertTrue(BEST_REPLANT_CHECKPOINT.is_file())
        self.assertTrue(v11_ckpt.is_file())
        for ckpt, expected_task in ((BEST_REPLANT_CHECKPOINT, TASK), (v11_ckpt, 'Wangyu-Replant-v11')):
            bundle = ckpt.parent
            archive = json.loads((bundle / 'archive.json').read_text())
            self.assertEqual(archive['task'], expected_task)
            self.assertEqual(archive['selected_checkpoint'], ckpt.name)
            for rel, expected in archive['sha256'].items():
                target = bundle / rel
                h = hashlib.sha256(target.read_bytes()).hexdigest()
                self.assertEqual(h, expected, f'SHA mismatch for archived {bundle.name}/{rel}')
        # Verify archived best actor & critic weights load strictly into v12 and std_param resets cleanly to 0.08.
        env = ReplantEnv(self.model, 1, reference=self.ref, rsi_prob=0.0)
        runner = OnPolicyRunner(env, runner_config(), device='cpu')
        saved = torch.load(BEST_REPLANT_CHECKPOINT, map_location='cpu', weights_only=True)
        runner.alg.actor.load_state_dict(saved['actor_state_dict'], strict=True)
        runner.alg.critic.load_state_dict(saved['critic_state_dict'], strict=True)
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.fill_(0.08)
        np.testing.assert_allclose(runner.alg.actor.distribution.std_param.detach().numpy(), 0.08)
        # Verify legacy v11 replay model restores original knee limit while v12 enforces 108 deg EVA cap.
        legacy_m = build_spec(training_sensors=False, knee_hard_limit=False).compile()
        self.assertAlmostEqual(float(np.rad2deg(self.model.joint('robot/LeftKnee').range[1])), 108.0, places=3)
        self.assertGreater(float(np.rad2deg(legacy_m.joint('robot/LeftKnee').range[1])), 120.0)

    def test_prone_magnetic_undock_and_forward_deploy_zero_penetration(self):
        from replant_deploy import ProneDeployPrelude, handoff_to_recovery

        world = Recovery(self.model, reference=self.ref, terminate_on_ref_div=False)
        prelude = ProneDeployPrelude(self.model, world.d, self.ref)
        # Starts magnetically welded to both boots in prone Exoskeleton Mode.
        self.assertTrue(np.all(world.d.eq_active == 1))
        for side in SIDES:
            np.testing.assert_allclose(world.d.site(f'robot/exo_{side}_hoof_dock').xpos,
                                       world.d.site(f'robot/exo_{side}_boot_dock').xpos, atol=1e-7)

        strut_gids = [self.model.geom(n).id for n in (
            'robot/exo_left_outer_tube', 'robot/exo_left_sliding_shaft', 'robot/exo_left_hoof_pad',
            'robot/exo_right_outer_tube', 'robot/exo_right_sliding_shaft', 'robot/exo_right_hoof_pad',
        )]
        tip_set = {self.model.geom(f'robot/exo_{s}_hoof_pad').id for s in SIDES}
        human_gids = [i for i in range(self.model.ngeom)
                      if (self.model.geom_contype[i] & 2) != 0 and i not in tip_set]
        min_human_clear = 1e9
        while not prelude.done:
            prelude.step()
            if prelude.step_idx % 2 == 0:
                for sg in strut_gids:
                    for hg in human_gids:
                        min_human_clear = min(
                            min_human_clear,
                            float(mujoco.mj_geomDistance(self.model, world.d, sg, hg, 2., None)),
                        )
        # Magnetic welds released, zero strut/hoof penetration into human body, hooves seated near front anchor.
        self.assertTrue(np.all(world.d.eq_active == 0))
        self.assertGreater(min_human_clear, 0.005, f'Strut/hoof penetrated human during deploy: {min_human_clear*100:.2f} cm')
        handoff_to_recovery(world)
        tip_err = np.linalg.norm(world.d.geom_xpos[world.tips] - self.ref['tip_targets'][0], axis=-1)
        self.assertLess(float(tip_err.max()), 0.05, f'Front anchor handoff error too large: {tip_err}')


if __name__ == '__main__':
    unittest.main()
