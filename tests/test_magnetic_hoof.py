"""Tests for the damped ball-joint bionic hoof and magnetic Exo <-> Exolimb mode switching (hut_hinge_hoof_v8)."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from exo_mode_switch import ExoMode, MagneticHoofController, solve_strut_ik_to_world_point
from exoskeleton import (HOOF_BALL_DAMPING, HOOF_BALL_STIFFNESS,
                         HOOF_SOLE_HEIGHT_M, HUT_HINGE_HOOF_MECHANISM, STRUTS,
                         fit_struts_to_pose)
from moon_scene import WANGYU_EVA_PROFILE, scene_spec
from moon_viewer import display_spec
from preview_hoof_mode_switch import build_hoof_demo_spec
from suit_proxy_geometry import FIT3D_18V


class MagneticHoofTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exo_spec = build_hoof_demo_spec(detached=False)
        cls.exo_model = cls.exo_spec.compile()
        cls.limb_spec = build_hoof_demo_spec(detached=True)
        cls.limb_model = cls.limb_spec.compile()

    def test_unified_layout_damped_ball_joint_and_zero_initial_weld_error(self):
        m_exo, m_limb = self.exo_model, self.limb_model
        # Exact same state & actuator layout in both initial modes -> runtime switching needs zero recompile.
        self.assertEqual((m_exo.nq, m_exo.nv, m_exo.nu, m_exo.nbody, m_exo.neq),
                         (m_limb.nq, m_limb.nv, m_limb.nu, m_limb.nbody, m_limb.neq))
        self.assertEqual(m_exo.nu, 29)
        self.assertEqual(m_exo.neq, 2)
        self.assertTrue(np.all(m_exo.eq_type == mujoco.mjtEq.mjEQ_WELD))
        self.assertTrue(np.all(m_exo.eq_active0 == 1))
        self.assertTrue(np.all(m_limb.eq_active0 == 0))

        d = mujoco.MjData(m_exo)
        mujoco.mj_forward(m_exo, d)
        for side in STRUTS:
            name = f'robot/exo_{side}'
            jnt = m_exo.joint(f'{name}_hoof_ball')
            self.assertEqual(jnt.type[0], mujoco.mjtJoint.mjJNT_BALL)
            # Smooth elastomer centering (limited=False) avoids Warp's extra _limit_ball kernel.
            self.assertEqual(int(jnt.limited[0]), 0)
            dof = int(jnt.dofadr[0])
            self.assertAlmostEqual(float(m_exo.dof_damping[dof]), HOOF_BALL_DAMPING)
            self.assertAlmostEqual(float(jnt.stiffness[0]), HOOF_BALL_STIFFNESS)

            # Hoof dock site mates flush to boot dock site with 0 mm / 0 deg error.
            p_h = d.site(f'{name}_hoof_dock').xpos
            p_b = d.site(f'{name}_boot_dock').xpos
            R_h = d.site(f'{name}_hoof_dock').xmat.reshape(3, 3)
            R_b = d.site(f'{name}_boot_dock').xmat.reshape(3, 3)
            np.testing.assert_allclose(p_h, p_b, atol=1e-9)
            np.testing.assert_allclose(R_h, R_b, atol=1e-9)

            # Docked hoof bottom clears ground by ~4 mm above boot sole plane (no walking snag).
            sole_z = float(d.site(f'{name}_hoof_sole_site').xpos[2])
            self.assertGreater(sole_z, 0.002)
            self.assertLess(sole_z, 0.008)

            # In detached (Exolimb) mode, ONLY the bottom rubber box (hoof_pad) collides on the hoof body.
            hoof_body_id = m_limb.body(f'{name}_hoof').id
            colliding_hoof_geoms = [
                m_limb.geom(gid).name for gid in range(m_limb.ngeom)
                if int(m_limb.geom_bodyid[gid]) == hoof_body_id
                and (int(m_limb.geom_contype[gid]) != 0 or int(m_limb.geom_conaffinity[gid]) != 0)
            ]
            self.assertEqual(colliding_hoof_geoms, [f'{name}_hoof_pad'])
            g_limb = m_limb.geom(f'{name}_hoof_pad')
            self.assertEqual(int(g_limb.type[0]), mujoco.mjtGeom.mjGEOM_BOX)
            self.assertEqual(int(g_limb.condim[0]), 4)
            self.assertGreater(float(g_limb.friction[0]), 1.2)
            self.assertEqual((int(g_limb.contype[0]), int(g_limb.conaffinity[0])), (2, 1))
            # Default build keeps hoof_pad in the compile-time BVH for runtime mode switching;
            # MagneticHoofController disables it when docked and re-enables on undock.
            g_exo = m_exo.geom(f'{name}_hoof_pad')
            self.assertEqual((int(g_exo.contype[0]), int(g_exo.conaffinity[0])), (2, 1))
            self.assertEqual(mujoco.mj_name2id(m_exo, mujoco.mjtObj.mjOBJ_GEOM, f'{name}_hoof_cleat_0'), -1)

        # 6-DOF branch + 6-constraint weld = 0 redundant constraints, 0 residual!
        eq_rows = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self.assertEqual(int(eq_rows.sum()), 12)
        np.testing.assert_allclose(d.efc_pos[:d.nefc][eq_rows], 0., atol=1e-9)

    def test_fit_struts_to_pose_solves_all_6_dof_including_ball_joint_across_squat(self):
        m = self.exo_model
        d = mujoco.MjData(m)
        d.joint('robot/LeftHipPitch').qpos[0] = -0.55
        d.joint('robot/LeftKnee').qpos[0] = 0.95
        d.joint('robot/LeftAnkle').qpos[0] = -0.25
        d.joint('robot/RightHipPitch').qpos[0] = -0.30
        d.joint('robot/RightKnee').qpos[0] = 0.55
        d.joint('robot/WaistPitch').qpos[0] = 0.18
        fit_struts_to_pose(m, d)
        mujoco.mj_forward(m, d)

        for side in STRUTS:
            name = f'robot/exo_{side}'
            np.testing.assert_allclose(d.site(f'{name}_hoof_dock').xpos,
                                       d.site(f'{name}_boot_dock').xpos, atol=1e-9)
            np.testing.assert_allclose(d.site(f'{name}_hoof_dock').xmat,
                                       d.site(f'{name}_boot_dock').xmat, atol=1e-9)
        eq_rows = d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        np.testing.assert_allclose(d.efc_pos[:d.nefc][eq_rows], 0., atol=1e-8)

    def test_runtime_mode_switching_undock_brace_and_magnetic_redock(self):
        m = build_hoof_demo_spec(detached=False).compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ctrl = MagneticHoofController(m, d)
        self.assertEqual(ctrl.overall_mode, ExoMode.EXO_ASSIST)

        # Keep human joints gently held at neutral standing while testing live strut mode switch.
        q0 = d.qpos.copy()
        human_aids = [i for i in range(m.nu) if not m.actuator(i).name.startswith('robot/exo_')]

        def hold_human():
            for aid in human_aids:
                jnt = int(m.actuator_trnid[aid, 0])
                d.ctrl[aid] = float(q0[m.jnt_qposadr[jnt]])

        # 1) Release magnetic lock -> UNDOCKING -> EXOLIMB_BRACE
        ctrl.request_exolimb()
        self.assertEqual(ctrl.overall_mode, ExoMode.UNDOCKING)
        self.assertTrue(np.all(d.eq_active == 0))
        for _ in range(80):
            hold_human()
            ctrl.step_control()
            mujoco.mj_step(m, d)
        self.assertEqual(ctrl.overall_mode, ExoMode.EXOLIMB_BRACE)
        for side in STRUTS:
            self.assertGreater(ctrl.status(side).dock_distance_m, 0.04)

        # 2) Plant hooves forward-outward at an angle on the lunar plane -> damped ball joint self-aligns flat!
        pelvis = d.xpos[m.body('robot/Pelvis').id].copy()
        brace_targets = {
            'left': np.array([pelvis[0] + 0.20, pelvis[1] + 0.34, HOOF_SOLE_HEIGHT_M]),
            'right': np.array([pelvis[0] + 0.20, pelvis[1] - 0.34, HOOF_SOLE_HEIGHT_M]),
        }
        for _ in range(140):
            hold_human()
            ctrl.step_control(brace_targets_world=brace_targets)
            mujoco.mj_step(m, d)
        for side in STRUTS:
            hoof_id = m.body(f'robot/exo_{side}_hoof').id
            hoof_up_z = float(d.xmat[hoof_id].reshape(3, 3)[2, 2])
            sole_z = float(d.site(f'robot/exo_{side}_hoof_sole_site').xpos[2])
            self.assertGreater(hoof_up_z, 0.94, f'{side} hoof failed to self-align flat on ground')
            self.assertLess(abs(sole_z), 0.025, f'{side} hoof did not reach ground plane')

        # 3) Command magnetic redock -> DOCKING -> snaps mjEQ_WELD back into EXO_ASSIST!
        ctrl.request_exoskeleton()
        self.assertEqual(ctrl.overall_mode, ExoMode.DOCKING)
        for _ in range(260):
            hold_human()
            ctrl.step_control()
            mujoco.mj_step(m, d)
            if ctrl.overall_mode == ExoMode.EXO_ASSIST:
                break
        self.assertEqual(ctrl.overall_mode, ExoMode.EXO_ASSIST)
        self.assertTrue(np.all(d.eq_active == 1))
        for side in STRUTS:
            st = ctrl.status(side)
            self.assertTrue(st.welded)
            self.assertLess(st.dock_distance_m, 0.010)
            self.assertLess(st.dock_angle_deg, 10.0)

    def test_studio_skin_and_classic_physx_display_specs_compile_with_hoof_mesh(self):
        old_viewer = os.environ.get('MOON_VIEWER')
        old_fil = os.environ.get('MOON_FILAMENT_SCENE')
        try:
            for viewer, viz, expect_skin in (('studio', 'skin', True), ('classic', 'physx', False)):
                os.environ['MOON_VIEWER'] = viewer
                os.environ['MOON_FILAMENT_SCENE'] = '1' if viz == 'skin' else '0'
                shown = display_spec(SimpleNamespace(mj_spec=self.exo_spec, nworld=1)).compile()
                self.assertEqual((shown.nq, shown.nv, shown.nbody),
                                 (self.exo_model.nq, self.exo_model.nv, self.exo_model.nbody))
                self.assertEqual(shown.nskin > 0, expect_skin)
                for side in STRUTS:
                    # Hoof seat + sole pad + boot magnetic dock remain visible in group 2 in both modes.
                    for part in ('hoof_socket', 'hoof_shell', 'hoof_pad',
                                 'hoof_mag_plate', 'boot_mag_plate', 'mag_led'):
                        self.assertEqual(int(shown.geom(f'robot/exo_{side}_{part}').group[0]), 2)
        finally:
            if old_viewer is None:
                os.environ.pop('MOON_VIEWER', None)
            else:
                os.environ['MOON_VIEWER'] = old_viewer
            if old_fil is None:
                os.environ.pop('MOON_FILAMENT_SCENE', None)
            else:
                os.environ['MOON_FILAMENT_SCENE'] = old_fil

    def test_full_scene_spec_accepts_hut_hinge_hoof_mechanism(self):
        # Verify compatibility with standard terrain scene_spec as well.
        spec = scene_spec(geometry=FIT3D_18V, mass_profile=WANGYU_EVA_PROFILE,
                          suit_ankle_roll=True, suit_ankle_pitch=True, suit_major_joints=True,
                          relax_propulsion=True, exoskeleton=True,
                          exo_mechanism=HUT_HINGE_HOOF_MECHANISM)
        m = spec.compile()
        self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot/exo_left_hoof_ball'), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, 'robot/exo_right_mag_dock'), 0)

    def test_cpu_viewer_extra_key_routes_L_without_stealing_studio_M(self):
        from cpu_viewer import CpuDisplay
        received = []
        disp = CpuDisplay.__new__(CpuDisplay)
        disp.kind = 'classic'
        disp.controls = SimpleNamespace(handle=lambda k: False)
        disp.state = {'extra_keys': (('L', 'L'),)}
        disp.key = received.append
        self.assertTrue(disp.handle_key('L'))
        self.assertEqual(received, [ord('L')])
        self.assertFalse(disp.handle_key('M'))
        self.assertEqual(received, [ord('L')])


if __name__ == '__main__':
    unittest.main()
