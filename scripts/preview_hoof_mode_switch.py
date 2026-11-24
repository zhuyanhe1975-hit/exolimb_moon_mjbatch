"""Inspect and interactively toggle the damped ball-joint magnetic hoof (Exoskeleton <-> Exolimb mode).

Usage:
  ./scripts/preview_hoof_mode_switch.py --headless          # static + mode-switch verification (no window)
  python scripts/preview_hoof_mode_switch.py                # Studio + Wangyu skin window (L toggles mode)
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from exo_mode_switch import ExoMode, MagneticHoofController  # noqa: E402
from exoskeleton import (HOOF_SOLE_HEIGHT_M, HUT_HINGE_HOOF_MECHANISM,  # noqa: E402
                         STRUTS, fit_struts_to_pose)
from moon_scene import WANGYU_EVA_PROFILE, scene_spec  # noqa: E402
from suit_proxy_geometry import FIT3D_18V  # noqa: E402


def build_hoof_demo_spec(*, detached: bool = False):
    spec = scene_spec(
        geometry=FIT3D_18V,
        mass_profile=WANGYU_EVA_PROFILE,
        suit_ankle_roll=True,
        suit_ankle_pitch=True,
        suit_major_joints=True,
        relax_propulsion=True,
        exoskeleton=True,
        exo_mechanism=HUT_HINGE_HOOF_MECHANISM,
        exo_detached=detached,
    )
    ground = spec.geom('terrain')
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.hfieldname = ''
    ground.pos = [0, 0, 0]
    ground.size = [8, 8, .1]
    ground.contype, ground.conaffinity = 1, 2
    for hfield in list(spec.hfields):
        spec.delete(hfield)
    return spec


def _brace_targets(model: mujoco.MjModel, data: mujoco.MjData, *, forward_offset: float = 0.14):
    """Wide tripod ground-bracing targets for the damped ball-joint hooves."""
    pelvis = data.xpos[model.body('robot/Pelvis').id]
    return {
        'left': np.array([pelvis[0] + forward_offset, pelvis[1] + 0.36, HOOF_SOLE_HEIGHT_M], dtype=np.float64),
        'right': np.array([pelvis[0] + forward_offset, pelvis[1] - 0.36, HOOF_SOLE_HEIGHT_M], dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', choices=('studio', 'classic'), default='studio')
    parser.add_argument('--viz', choices=('skin', 'physx'), default='skin')
    parser.add_argument('--start-mode', choices=('exo', 'exolimb'), default='exo')
    parser.add_argument('--headless', action='store_true',
                        help='run static geometry & kinematic dock inspection without opening a window')
    args = parser.parse_args()

    spec = build_hoof_demo_spec(detached=(args.start_mode == 'exolimb'))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl = MagneticHoofController(model, data)

    if args.headless:
        report = {
            'mechanism': HUT_HINGE_HOOF_MECHANISM,
            'nq': int(model.nq),
            'nv': int(model.nv),
            'nu': int(model.nu),
            'neq': int(model.neq),
            'initial_mode': ctrl.overall_mode.value,
            'sides': {},
        }
        for side in STRUTS:
            st = ctrl.status(side)
            report['sides'][side] = {
                'welded': st.welded,
                'dock_distance_mm': round(st.dock_distance_m * 1e3, 4),
                'dock_angle_deg': round(st.dock_angle_deg, 4),
                'hoof_ground_gap_mm': round(st.hoof_ground_gap_m * 1e3, 2),
            }
        print(json.dumps(report, indent=2), flush=True)
        return

    # Interactive mode: keep astronaut stably upright while user presses L to toggle Exo <-> Exolimb.
    human_aids = [i for i in range(model.nu) if not model.actuator(i).name.startswith('robot/exo_')]
    q0 = data.qpos.copy()
    pelvis_id = model.body('robot/Pelvis').id
    pelvis_p0 = data.xpos[pelvis_id].copy()
    state = {
        'paused': False,
        'toggle_req': False,
        # L (Latch/Lock) is unused by MuJoCo Studio/Classic (M stays Studio's native COM toggle).
        'extra_keys': (('L', 'L'),),
        'sync_geom_rgba': tuple(f'robot/exo_{s}_mag_led' for s in STRUTS),
    }
    reset_requested = threading.Event()

    def on_key(keycode):
        if keycode == 32:
            state['paused'] = not state['paused']
        elif keycode in (ord('R'), ord('r')):
            reset_requested.set()
        elif keycode in (ord('L'), ord('l')):
            state['toggle_req'] = True

    from cpu_viewer import launch
    with launch(spec, model, data, args.viewer, args.viz, on_key, state, reset_requested) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -16
        print('[Magnetic Hoof Demo] Space=pause; L=toggle magnetic dock (Exo <-> Exolimb hoof brace); '
              'R=reset; M=Studio COM; /=follow; A=overview; Esc=free camera.', flush=True)
        dt_vis = 1.0 / 60.0
        substeps = max(1, int(round(dt_vis / model.opt.timestep)))
        last_mode = ctrl.overall_mode
        while viewer.is_running():
            t0 = time.monotonic()
            with viewer.lock():
                if reset_requested.is_set():
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = q0
                    fit_struts_to_pose(model, data)
                    mujoco.mj_forward(model, data)
                    ctrl = MagneticHoofController(model, data)
                    last_mode = ctrl.overall_mode
                    reset_requested.clear()
                if state['toggle_req']:
                    new_mode = ctrl.toggle_mode()
                    print(f'[Magnetic Hoof] Requested -> {new_mode.value}', flush=True)
                    state['toggle_req'] = False
                if not state['paused']:
                    for aid in human_aids:
                        jnt = int(model.actuator_trnid[aid, 0])
                        data.ctrl[aid] = float(q0[model.jnt_qposadr[jnt]])
                    targets = _brace_targets(model, data) if ctrl.overall_mode == ExoMode.EXOLIMB_BRACE else None
                    for _ in range(substeps):
                        # Gentle overhead harness support on pelvis so astronaut stays upright during close-up inspection.
                        dp = pelvis_p0 - data.xpos[pelvis_id]
                        dv = -data.qvel[:3]
                        ang_vel = -data.qvel[3:6]
                        data.xfrc_applied[pelvis_id, :3] = 400.0 * dp + 60.0 * dv
                        data.xfrc_applied[pelvis_id, 3:] = 40.0 * ang_vel
                        ctrl.step_control(brace_targets_world=targets)
                        mujoco.mj_step(model, data)
                    if ctrl.overall_mode != last_mode:
                        last_mode = ctrl.overall_mode
                        st_l, st_r = ctrl.status('left'), ctrl.status('right')
                        print(f'[Magnetic Hoof] State -> {last_mode.value} '
                              f'(welded={st_l.welded}/{st_r.welded}, '
                              f'dock_dist={st_l.dock_distance_m*1e3:.1f}/{st_r.dock_distance_m*1e3:.1f} mm)',
                              flush=True)
            viewer.sync()
            time.sleep(max(0.0, dt_vis - (time.monotonic() - t0)))


if __name__ == '__main__':
    main()
