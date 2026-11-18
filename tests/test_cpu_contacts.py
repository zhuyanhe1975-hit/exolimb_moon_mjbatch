"""Contact overlays must use solved CPU worlds, not display collisions."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpu_viewer import CpuDisplay


class CpuContactsTest(unittest.TestCase):
    def test_solver_points_forces_and_disable_both_backends(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
          </worldbody></mujoco>''')
        sources=[mujoco.MjData(model) for _ in range(2)]
        sources[1].qpos[0]=2
        for data in sources: mujoco.mj_forward(model,data)
        self.assertTrue(all(data.ncon and data.efc_force[0]>0 for data in sources))
        for kind in ('classic','studio'):
            display=CpuDisplay.__new__(CpuDisplay)
            display.kind=kind
            display.source_model,display.source_data=model,sources[0]
            display.simulator=SimpleNamespace(native_data=sources)
            display.native=SimpleNamespace(user_scn=mujoco.MjvScene(model,maxgeom=100),extra_geoms=[])
            display.opt=mujoco.MjvOption()
            for points,forces in ((True,False),(False,True),(True,True),(False,False)):
                display.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=points
                display.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=forces
                display.update_contacts()
                scene=display.native.user_scn if kind=='classic' else display._contact_scene
                self.assertEqual(scene.ngeom,2*(points+forces))
                geoms=list(scene.geoms[:scene.ngeom])
                if points:
                    spheres=[g for g in geoms if g.type==mujoco.mjtGeom.mjGEOM_SPHERE]
                    np.testing.assert_allclose([g.pos[0] for g in spheres],[0,2])
                if forces:
                    arrows=[g for g in geoms if g.type==mujoco.mjtGeom.mjGEOM_ARROW]
                    self.assertTrue(all(g.size[2]>0 and np.isfinite(g.mat).all() for g in arrows))
                if kind=='studio': self.assertEqual(len(display.native.extra_geoms),scene.ngeom)


if __name__=='__main__': unittest.main()
