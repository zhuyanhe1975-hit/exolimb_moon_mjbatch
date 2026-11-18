"""Verify that the skin-aligned proxies preserve mass and sole reference planes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from local_lunar_terrain import sole_sampling
from cpu_simulation import CpuWorlds
from moon_scene import scene_spec
from moon_viewer import display_spec


class SuitProxyGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = scene_spec().compile()
        cls.fitted = scene_spec(geometry='fitted')
        cls.model = cls.fitted.compile()

    def test_mass_floor_planes_and_touch_sites(self):
        for field in ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia',
                      'geom_contype', 'geom_conaffinity', 'qpos0'):
            np.testing.assert_array_equal(getattr(self.source, field), getattr(self.model, field))
        points, cover = sole_sampling(self.model)
        self.assertGreater(cover, 0.)
        self.assertLess(np.linalg.norm(points[:, :2], axis=-1).max(), .5)
        for side, sign in (('left', 1), ('right', -1)):
            old = self.source.geom(f'robot/{side}_sole')
            new = self.model.geom(f'robot/{side}_sole')
            site = self.model.site(f'robot/balance_{side}_sole')
            self.assertAlmostEqual(old.pos[2] - old.size[2], new.pos[2] - new.size[2])
            np.testing.assert_allclose(site.pos, [*new.pos[:2], new.pos[2] - new.size[2]])
            np.testing.assert_array_equal(new.quat, site.quat)
            self.assertGreater(new.quat[3] * sign, 0.)

    def test_sole_pair_backstop_is_detected_and_excluded_from_ground_load(self):
        spec = scene_spec(geometry='fitted', sole_probes=True, sole_collision=True)
        worlds = CpuWorlds(spec, 1, 12)
        model, data = worlds.model, worlds.data[0]
        data.qpos[:] = model.qpos0
        data.qpos[2] += 4.
        data.joint('robot/LeftHipRoll').qpos[:] = -.19
        data.joint('robot/RightHipRoll').qpos[:] = .19
        mujoco.mj_forward(model, data)
        feet = {model.geom(f'robot/{side}_sole').id for side in ('left', 'right')}
        self.assertTrue(any(set(contact.geom) == feet for contact in data.contact[:data.ncon]))
        self.assertLess(float(data.sensor('robot/balance_sole_separation').data[0]), 0.)
        worlds.read_native()
        self.assertEqual(worlds.sim.data.sensordata[:, worlds._sole_load_columns].tolist(), [[0., 0.]])
        self.assertEqual(data.time, 0.)

    def test_footprint_covers_the_lower_boot_mesh(self):
        with patch.dict('os.environ', {'MOON_FILAMENT_SCENE': '1', 'MOON_VIEWER': 'studio'}):
            shown = display_spec(SimpleNamespace(mj_spec=self.fitted, nworld=1)).compile()
        data = mujoco.MjData(shown)
        mujoco.mj_forward(shown, data)
        option = mujoco.MjvOption()
        option.skingroup[1] = 1
        scene = mujoco.MjvScene(shown, maxgeom=2000)
        mujoco.mjv_updateScene(shown, data, option, mujoco.MjvPerturb(), mujoco.MjvCamera(),
                              mujoco.mjtCatBit.mjCAT_ALL, scene)
        skin = shown.skin('skin_Boot_0').id
        start = shown.skin_vertadr[skin]
        boots = scene.skinvert.reshape(-1, 3)[start:start + shown.skin_vertnum[skin]]
        for side, sign in (('left', 1), ('right', -1)):
            geo = shown.geom(f'robot/{side}_sole')
            vertices = boots[(sign * boots[:, 1] > 0) & (boots[:, 2] < .040)]
            self.assertGreater(len(vertices), 50)
            local = (vertices - data.geom_xpos[geo.id]) @ data.geom_xmat[geo.id].reshape(3, 3)
            self.assertGreater(np.mean((np.abs(local[:, :2]) <= geo.size[:2] + .001).all(-1)), .98)


if __name__ == '__main__':
    unittest.main()
