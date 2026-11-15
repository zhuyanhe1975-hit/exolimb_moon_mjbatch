"""64 actors retain exactly one declaration of every shared render asset."""
from pathlib import Path
from types import SimpleNamespace
import os,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_scene import scene_spec
from moon_viewer import display_spec


class LargeDisplayAssetsTest(unittest.TestCase):
 def test_no_duplicate_assets_before_compilation(self):
  source=scene_spec()
  for viewer in ('classic','studio'):
   with self.subTest(viewer=viewer),patch.dict(os.environ,{'MOON_VIEWER':viewer,'MOON_FILAMENT_SCENE':'1'}):
    simulator=SimpleNamespace(mj_spec=source,nworld=64)
    single=display_spec(simulator,single=True)
    multi=display_spec(simulator)
    for collection in ('textures','materials','meshes','hfields','lights'):
     self.assertEqual(len(list(getattr(multi,collection))),len(list(getattr(single,collection))))
    self.assertEqual(len(list(multi.skins)),64*len(list(single.skins)))
    for collection in ('textures','materials','meshes'):
     self.assertTrue(all(asset.name.startswith('world_0/') for asset in getattr(multi,collection)))
    for element in list(multi.geoms)+list(multi.skins):
     for attr in ('meshname','material'):
      name=getattr(element,attr,'')
      if name:self.assertTrue(name.startswith('world_0/'),name)
    skins={skin.name:skin for skin in multi.skins}
    self.assertIn('world_0/skin_Suit_0',skins)
    self.assertIn('world_63/skin_Suit_0',skins)
    self.assertNotEqual(skins['world_0/skin_Suit_0'].id,skins['world_63/skin_Suit_0'].id)


if __name__=='__main__':unittest.main()
