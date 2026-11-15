"""Native camera tracks the selected actor; cycling never edits physics state."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cpu_camera import CameraControls
from moon_scene import scene_spec
from moon_viewer import display_spec


class CameraTest(unittest.TestCase):
    def build(self,count,kind='studio',viz='physx'):
        source = scene_spec()
        with patch.dict(os.environ,{'MOON_VIEWER':kind,'MOON_FILAMENT_SCENE':'1' if viz=='skin' else '0'}):
            model = display_spec(SimpleNamespace(mj_spec=source,nworld=count)).compile()
        data = mujoco.MjData(model)
        for i in range(count):
            name = (f'world_{i}/' if count>1 else '')+'robot/Pelvis'
            # Freejoint qpos starts at the first joint of its own actor.
            adr = int(model.body(name).jntadr[0])
            qa = int(model.jnt_qposadr[adr])
            data.qpos[qa] = i
        mujoco.mj_forward(model,data)
        camera = mujoco.MjvCamera()
        camera.distance=18; camera.elevation=-20; camera.azimuth=135
        selected=[]
        controls = CameraControls(model,data,camera,count,{},selected.append)
        return controls,selected

    def test_cycle_track_wrap_overview_and_single_actor(self):
        from itertools import product
        for count,kind,viz in product((1,8),('classic','studio'),('physx','skin')):
            with self.subTest(count=count,kind=kind,viz=viz),redirect_stdout(io.StringIO()):
                c, selected = self.build(count,kind,viz)
                before = c.data.qpos.copy()
                for key in ('T','N','P'):
                    self.assertFalse(c.handle(key))
                c.handle('/')
                self.assertTrue(c.tracking)
                c.handle(',')
                self.assertEqual(c.current,count-1)
                self.assertEqual(c.camera.trackbodyid,c.roots[count-1])
                self.assertEqual(selected[-1],c.roots[count-1])
                c.handle('.')
                self.assertEqual(c.current,0)
                self.assertTrue(c.tracking)
                distance=c.camera.distance
                c.handle('A');self.assertTrue(c.overview);self.assertFalse(c.tracking)
                c.handle('A');self.assertFalse(c.overview);self.assertTrue(c.tracking)
                self.assertEqual(c.camera.distance,distance)
                c.handle('Escape');self.assertFalse(c.tracking)
                c.handle('.');self.assertTrue(c.tracking)
                c.handle('/');self.assertFalse(c.tracking)
                self.assertEqual(c.state['selected_world'],c.current)
                self.assertEqual(c.state['camera_mode'],'free')
                np.testing.assert_array_equal(c.data.qpos,before)

    def test_native_camera_follows_motion_then_stops(self):
        with redirect_stdout(io.StringIO()):
            c,_ = self.build(8)
            c.select(7);c.set_tracking(True)
            scene = mujoco.MjvScene(c.model,maxgeom=2000)
            option = mujoco.MjvOption()
            def render_camera():
                mujoco.mjv_updateScene(c.model,c.data,option,None,c.camera,
                                      mujoco.mjtCatBit.mjCAT_ALL,scene)
            render_camera()
            previous = c.camera.lookat.copy()
            joint = int(c.model.body(c.roots[7]).jntadr[0])
            qa = int(c.model.jnt_qposadr[joint])
            c.data.qpos[qa] += 2
            mujoco.mj_forward(c.model,c.data)
            for _ in range(30):render_camera()
            self.assertGreater(c.camera.lookat[0]-previous[0],.1)
            c.set_tracking(False)
            frozen = c.camera.lookat.copy()
            c.data.qpos[qa] += 2
            mujoco.mj_forward(c.model,c.data)
            render_camera()
            np.testing.assert_array_equal(c.camera.lookat,frozen)


if __name__=='__main__':unittest.main()
