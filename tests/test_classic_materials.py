"""Classic's lighting fallback preserves the same skin assets and physical model."""
from pathlib import Path
from types import SimpleNamespace
import os,sys,unittest
from unittest.mock import patch
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_scene import scene_spec
from moon_viewer import display_spec


class ClassicMaterialTest(unittest.TestCase):
 def test_shared_skin_and_nonduplicated_classic_lighting(self):
  source=scene_spec();physics=source.compile()
  with patch.dict(os.environ,{'MOON_VIEWER':'studio','MOON_FILAMENT_SCENE':'1'}):
   studio=display_spec(SimpleNamespace(mj_spec=source,nworld=1)).compile()
  models=[]
  for count in (1,8):
   with patch.dict(os.environ,{'MOON_VIEWER':'classic','MOON_FILAMENT_SCENE':'1'}):
    model=display_spec(SimpleNamespace(mj_spec=source,nworld=count)).compile()
   models.append(model)
   self.assertEqual(model.nskin,studio.nskin*count)
   self.assertEqual(model.nlight,2)
   self.assertEqual(model.vis.headlight.active,0)
   self.assertFalse((model.light_type==mujoco.mjtLightType.mjLIGHT_IMAGE).any())
   self.assertTrue((model.light_ambient>0).any())
   self.assertFalse((model.tex_nchannel==1).any())
   self.assertTrue((model.mat_reflectance==0).all())
   if count==1:
    np.testing.assert_array_equal(model.skin_vert,studio.skin_vert)
    np.testing.assert_array_equal(model.skin_texcoord,studio.skin_texcoord)
    robot=[i for i in range(physics.nbody) if physics.body(i).name.startswith('robot/')]
    np.testing.assert_array_equal(model.body_mass[robot],physics.body_mass[robot])
    np.testing.assert_array_equal(model.body_mass,studio.body_mass)
    np.testing.assert_array_equal(model.hfield_data,physics.hfield_data)
    np.testing.assert_array_equal(model.qpos0,physics.qpos0)
  self.assertEqual((models[0].ntex,models[0].nmesh,models[0].nmat),
                   (models[1].ntex,models[1].nmesh,models[1].nmat))


if __name__=='__main__':unittest.main()
