"""Explicit static asset generation requested by user; no simulation or policy."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from local_lunar_terrain import base_height
from lunar_details import generate,metadata
from large_terrain import ASSET,sha
if (ASSET/'state.npz').exists():raise SystemExit('Refusing to overwrite existing 100m asset')
with np.load(ROOT/'assets/playground/state.npz') as z:
    index=int(np.flatnonzero((abs(z['centers'])<=945).all(-1)&(z['spawn_counts']>0))[0])
    center=z['centers'][index].copy();altitude=float(z['altitudes'][index])
    spawn=z['spawn_xy'][index:index+1].copy();counts=z['spawn_counts'][index:index+1].copy()
axis=np.linspace(-50.,50.,1001)
heights=np.empty((1001,1001))
for row in range(0,1001,100):
    for col in range(0,1001,100):
        xx,yy=np.meshgrid(axis[col:col+100],axis[row:row+100]);xy=np.stack((xx,yy),-1)
        heights[row:row+100,col:col+100]=base_height(xy+center)-altitude+generate(xy,center)
low=heights.min();span=np.ptp(heights)
ASSET.mkdir(parents=True,exist_ok=True)
np.savez_compressed(ASSET/'state.npz',normalized=((heights-low)/span).astype(np.float32)[None],
    zmin=[low],height_range=[span],spawn_xy=spawn,spawn_counts=counts,centers=center[None],altitudes=[altitude])
(ASSET/'metadata.json').write_text(json.dumps({'extent_m':[100,100],'half_extent_m':50.,'resolution_m':.1,
    'shape':[1001,1001],'source_center_m':center.tolist(),'removed_altitude_m':altitude,
    'detail':metadata(),'generator_sha256':{name:sha(ROOT/name) for name in ('scripts/build_100m_terrain.py','src/lunar_details.py','src/local_lunar_terrain.py')},'exact_replay':False,'note':'New 100m crop, not a stretched 16m patch; legacy assets unchanged.'},indent=2)+'\n')
files=['assets/site20/heightfield.npz','assets/site20/detail_config.json','assets/playground/state.npz',
       'assets/playground_100m/state.npz','assets/playground_100m/metadata.json']
(ASSET/'manifest.json').write_text(json.dumps({'files':{name:sha(ROOT/name) for name in files}},indent=2)+'\n')
print('Saved',ASSET,'shape',heights.shape,'height range',float(span))
