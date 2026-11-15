"""CPU display synchronizes terrain/pose without changing physical state."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cpu_viewer import CpuDisplay
from moon_scene import scene_spec
from moon_viewer import display_spec


class CpuDisplayTest(unittest.TestCase):
    def test_terrain_pose_and_resource_reuse(self):
        spec = scene_spec()
        physics = spec.compile()
        data = mujoco.MjData(physics)
        for viz in ('physx', 'skin'):
            with self.subTest(viz=viz), patch.dict(os.environ, {
                'MOON_VIEWER': 'studio', 'MOON_FILAMENT_SCENE': '1' if viz == 'skin' else '0'}):
                model = display_spec(SimpleNamespace(mj_spec=spec, nworld=1), single=True).compile()
                display = CpuDisplay.__new__(CpuDisplay)
                display.source_model, display.source_data = physics, data
                display.model, display.data = model, mujoco.MjData(model)
                display.kind, display._active = 'studio', None
                self.assertEqual(model.nq, physics.nq)
                self.assertEqual(model.nv, physics.nv)
                self.assertEqual(model.nhfield, 64)
                self.assertEqual(model.nskin > 0, viz == 'skin')
                resources = (model.nskin, model.ntex, model.nmesh, model.nhfield)
                hfield_data = model.hfield_data.copy()
                for active in (0, 11, 63, 0):
                    physics.geom('terrain').dataid[0] = active
                    data.qpos[:] = physics.qpos0
                    data.qpos[0] = active / 100
                    mujoco.mj_forward(physics, data)
                    before = data.qpos.copy()
                    display._copy_state()
                    np.testing.assert_array_equal(data.qpos, before)
                    np.testing.assert_array_equal(display.data.qpos, data.qpos)
                    np.testing.assert_array_equal(display.data.sensordata, data.sensordata)
                    fields = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD) & (model.geom_group == 0))
                    self.assertEqual(len(fields), 1)
                    self.assertEqual(model.geom_dataid[fields[0]], active)
                    self.assertEqual(resources, (model.nskin, model.ntex, model.nmesh, model.nhfield))
                    np.testing.assert_array_equal(hfield_data, model.hfield_data)

    def test_eight_world_partial_reset_and_shared_skin(self):
        import torch
        from cpu_simulation import CpuWorlds
        torch.manual_seed(42)
        spec = scene_spec()
        worlds = CpuWorlds(spec, 8, 12)
        worlds.step(torch.zeros(8, 23))
        ongoing = [d.qpos.copy() for d in worlds.data[1:]]
        times = [d.time for d in worlds.data[1:]]
        targets = worlds.device.target[1:].clone()
        mask = torch.zeros(8, dtype=torch.bool); mask[0] = True
        worlds.reset(mask)
        for d, pose, timestamp in zip(worlds.data[1:], ongoing, times):
            np.testing.assert_array_equal(d.qpos, pose)
            self.assertEqual(d.time, timestamp)
        torch.testing.assert_close(worlds.device.target[1:], targets)
        self.assertEqual(worlds.data[0].time, 0)
        with patch.dict(os.environ, {'MOON_VIEWER':'studio', 'MOON_FILAMENT_SCENE':'1'}):
            single = display_spec(worlds.sim, single=True).compile()
            with warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                multi = display_spec(worlds.sim).compile()
            self.assertEqual(multi.nkey, 0)
        self.assertEqual(multi.nq, single.nq * 8)
        self.assertEqual(multi.nskin, single.nskin * 8)
        self.assertEqual((multi.ntex,multi.nmesh,multi.nhfield,multi.nlight),
                         (single.ntex,single.nmesh,single.nhfield,single.nlight))
        display = CpuDisplay.__new__(CpuDisplay)
        display.source_model, display.source_data = worlds.model, worlds.data[0]
        display.model, display.data = multi, mujoco.MjData(multi)
        display.kind, display._active = 'studio', None
        display._grid_backend = SimpleNamespace(_simulator=worlds.sim, _world_id=0)
        for terrain in (12, 64, 1):
            worlds.device.select_terrain(terrain)
            worlds.reset()
            display._copy_state()
            for i in range(8):
                adr = int(multi.joint(f'world_{i}/'+worlds.model.joint(0).name).qposadr[0])
                np.testing.assert_allclose(display.data.qpos[adr:adr+worlds.model.nq],
                                           worlds.data[i].qpos, atol=1e-6)
            active = np.flatnonzero((multi.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD) & (multi.geom_group == 0))
            self.assertEqual(len(active), 1)
            self.assertEqual(multi.geom_dataid[active[0]], terrain-1)
        display.state = {}
        display.data.xfrc_applied[:] = 0
        body = multi.body('world_7/'+worlds.model.body(worlds.device.root).name).id
        display.data.xfrc_applied[body,0] = 100
        display.capture_mouse_forces()
        forces = display.state['mouse_forces']
        self.assertEqual(forces[7,worlds.device.root,0],100)
        np.testing.assert_array_equal(forces[:7],0)
        display.data.xfrc_applied[:] = 0
        display.capture_mouse_forces()
        np.testing.assert_array_equal(display.state['mouse_forces'],0)

    def test_mouse_force_changes_only_target_world_and_releases(self):
        import torch
        from cpu_simulation import CpuWorlds
        torch.manual_seed(42)
        baseline = CpuWorlds(scene_spec(), 2, 12)
        torch.manual_seed(42)
        pushed = CpuWorlds(scene_spec(), 2, 12)
        root = pushed.device.root
        action = torch.zeros(2,23)
        pushed.mouse_forces[1,root,0] = 100
        baseline.step(action); pushed.step(action)
        np.testing.assert_array_equal(baseline.data[0].qvel,pushed.data[0].qvel)
        self.assertGreater(np.linalg.norm(pushed.data[1].qvel-baseline.data[1].qvel),1e-5)
        self.assertEqual(pushed.data[1].xfrc_applied[root,0],100)
        pushed.mouse_forces[:] = 0
        pushed.step(action)
        np.testing.assert_array_equal(pushed.data[1].xfrc_applied,0)


if __name__ == '__main__':
    unittest.main()
