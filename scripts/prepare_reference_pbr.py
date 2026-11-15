"""Convert cached Isaac preview textures to MuJoCo PNG/ORM, no renderer launch."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
source=Path(os.environ.get('ZBOT_MOON_DIR',str(Path.home()/'myWorks/zbot_exoarm_moon')))/'assets/lunar_sites/southpole_site20/LunarRegolith4k'
out=ROOT/'assets/regolith_pbr'
if (out/'manifest.json').exists():raise SystemExit('Refusing to overwrite prepared material assets')
paths={suffix:source/f'LunarRegolith4k_{suffix}.jpg' for suffix in ('diff','rough','nor_gl','ao')}
for path in paths.values():
    if not path.is_file():raise FileNotFoundError(path)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
out.mkdir(parents=True,exist_ok=True)
Image.open(paths['diff']).convert('RGB').save(out/'diffuse.png')
rough=np.asarray(Image.open(paths['rough']).convert('L'),dtype=np.float32)/255
occlusion=np.asarray(Image.open(paths['ao']).convert('L'),dtype=np.float32)/255
orm=np.stack((.65+.35*occlusion,.87+.13*rough,np.zeros_like(rough)),-1)
Image.fromarray(np.round(255*orm).astype(np.uint8)).save(out/'orm.png')
normal=np.asarray(Image.open(paths['nor_gl']).convert('RGB'),dtype=np.float32)/127.5-1
normal[...,:2]*=.8
normal[...,1]*=-1 # Isaac OmniPBR flip_tangent_v=true
normal/=np.linalg.norm(normal,axis=-1,keepdims=True).clip(1e-6)
Image.fromarray(np.round((normal+1)*127.5).clip(0,255).astype(np.uint8)).save(out/'normal.png')
(out/'manifest.json').write_text(json.dumps({
    'reference':str(ROOT.parent/'exolimb_moon_mjlab/scripts/preview_isaacsim.py'),
    'reference_sha256':sha(ROOT.parent/'exolimb_moon_mjlab/scripts/preview_isaacsim.py'),
    'source_sha256':{str(path):sha(path) for path in paths.values()},
    'files':{name:sha(out/name) for name in ('diffuse.png','orm.png','normal.png')},
    'uv_period_m':4.,'roughness_influence':.13,'ao_influence':.35,'normal_strength':.8,
    'normal_flip_tangent_v':True,'note':'Texture channel mapping approximates OmniPBR in Filament; different BRDF, lighting units and renderer.'},indent=2)+'\n')
print('Prepared cached reference maps:',out)
