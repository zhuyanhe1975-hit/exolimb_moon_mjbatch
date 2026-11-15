"""Explicit 100m playback asset; immutable legacy training assets stay separate."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/playground_100m'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def bank_data():
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        if sha(ROOT/name)!=value:raise ValueError(f'100m terrain SHA mismatch: {name}')
    with np.load(ASSET/'state.npz') as z:bank={k:z[k].copy() for k in z.files}
    bank['half_extent_m']=50.
    return bank


def configure(spec):
    bank=bank_data()
    for h in list(spec.hfields):
        if h.name!='site20_local_00':spec.delete(h)
    h=spec.hfield('site20_local_00')
    h.nrow=h.ncol=1001
    h.userdata=bank['normalized'][0].ravel().tolist()
    h.size=[50.,50.,float(bank['height_range'][0]),1.]
    spec.geom('terrain').pos=[0,0,float(bank['zmin'][0])]
    spec.add_numeric(name='moon_terrain_half',data=[50.])
    print('[Terrain transfer] 100x100m, 0.1m samples; state_sha256='+sha(ASSET/'state.npz')+'; legacy policy terrain transfer, not exact replay',flush=True)
    return spec


def half_extent(model):
    try:return float(model.numeric('moon_terrain_half').data[0])
    except KeyError:
        try:return float(model.numeric('world_0/moon_terrain_half').data[0])
        except KeyError:return 8.
