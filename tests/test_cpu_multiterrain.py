"""Check per-world terrain assignment on both local CPU physics backends."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from cpu_simulation import CpuWorlds
from moon_scene import scene_spec


class MultiterrainTest(unittest.TestCase):
    def test_mjbatch_matches_threaded_and_preserves_live_worlds_on_reset(self):
        spec = scene_spec(geometry='fitted', sole_probes=True, sole_collision=True, sole_edge_probes=True)
        opts = dict(terrain_indices=(12, 54), spawn_seed=17, optimize_cpu=True, workers=2)
        torch.manual_seed(17)
        batch = CpuWorlds(spec, 4, 12, use_mjbatch=True, **opts)
        torch.manual_seed(17)
        native = CpuWorlds(spec, 4, 12, **opts)
        torch.testing.assert_close(batch.device.ground.patch_ids, torch.tensor([11, 11, 53, 53]))
        geom = batch.device.ground.geom_id
        self.assertIs(batch._model_for(0), batch._model_for(1))
        self.assertIsNot(batch._model_for(0), batch._model_for(2))
        for i, model in enumerate(batch._batch.models):
            self.assertEqual(model.geom_dataid[geom], batch.sim.model.geom_dataid[i, geom])
            self.assertAlmostEqual(model.geom_pos[geom, 2], float(batch.sim.model.geom_pos[i, geom, 2]))
        action = torch.zeros((4, 23))
        for _ in range(5):
            got, expected = batch.step(action), native.step(action)
            torch.testing.assert_close(got['wangyu/state'], expected['wangyu/state'], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(got['wangyu/failed'], expected['wangyu/failed'])
        live = torch.tensor([False, True, False, True])
        prior = batch.sim.data.qpos[live].clone()
        headings = batch.device.heading[live].clone()
        batch.reset(~live)
        torch.testing.assert_close(batch.sim.data.qpos[live], prior)
        torch.testing.assert_close(batch.device.heading[live], headings)
        torch.testing.assert_close(batch.device.ground.patch_ids, torch.tensor([11, 11, 53, 53]))
        np.testing.assert_array_equal(batch.device.spawn_sampler.patches, [11, 11, 53, 53])


if __name__ == '__main__':
    unittest.main()
