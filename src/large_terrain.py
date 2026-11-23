"""Explicit 100m playback asset; immutable legacy training assets stay separate."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/playground_100m'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


WARP_TILE_SIZE_M = 20
WARP_TILE_GRID = 6


@lru_cache(maxsize=1)
def bank_data():
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        if sha(ROOT/name)!=value:raise ValueError(f'100m terrain SHA mismatch: {name}')
    with np.load(ASSET/'state.npz') as z:bank={k:z[k].copy() for k in z.files}
    bank['half_extent_m']=50.
    return bank


@lru_cache(maxsize=4)
def tiled_bank_data(tile_size_m=WARP_TILE_SIZE_M, grid=WARP_TILE_GRID):
    """Exact overlapping crops of the verified 100x100m terrain for float32 Warp.

    MuJoCo Warp's float32 hfield prism collider extends every triangle down to -base_z.
    On the full 100m map (27.58m vertical span), high-elevation cells form >200:1 needle
    prisms that cause float32 EPA contact explosions. Cropping into overlapping local
    tiles preserves every height, triangle, and slope at 0.1m resolution while keeping
    per-tile elevation spans within ~4-6.5m.
    """
    bank = bank_data()
    full_h = (bank['normalized'][0].astype(np.float64) * float(bank['height_range'][0])
              + float(bank['zmin'][0]))
    n_pts = int(round(tile_size_m * 10)) + 1
    if n_pts > full_h.shape[0]:
        raise ValueError(f'Tile size {tile_size_m}m exceeds 100m terrain')
    starts = np.linspace(0, full_h.shape[0] - n_pts, grid).round().astype(int)
    tiles, zmins, spans = [], [], []
    for r0 in starts:
        for c0 in starts:
            sub = full_h[r0:r0 + n_pts, c0:c0 + n_pts]
            low = float(sub.min())
            span = max(float(sub.max() - low), 1e-3)
            tiles.append(((sub - low) / span).astype(np.float32))
            zmins.append(low)
            spans.append(span)
    count = len(tiles)
    return {
        'normalized': np.stack(tiles, axis=0),
        'zmin': np.asarray(zmins, dtype=np.float32),
        'height_range': np.asarray(spans, dtype=np.float32),
        'half_extent_m': tile_size_m / 2.0,
        'spawn_xy': np.zeros((count, 1, 2), dtype=np.float32),
        'spawn_counts': np.ones(count, dtype=np.int32),
    }


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


def configure_tiled(spec, tile_size_m=WARP_TILE_SIZE_M, grid=WARP_TILE_GRID):
    bank = tiled_bank_data(tile_size_m, grid)
    count = len(bank['normalized'])
    n_pts = int(bank['normalized'].shape[1])
    half = float(bank['half_extent_m'])
    keep = {f'site20_local_{i:02d}' for i in range(count)}
    for h in list(spec.hfields):
        if h.name not in keep:
            spec.delete(h)
    for i in range(count):
        h = spec.hfield(f'site20_local_{i:02d}')
        h.nrow = h.ncol = n_pts
        h.userdata = bank['normalized'][i].ravel().tolist()
        h.size = [half, half, float(bank['height_range'][i]), 1.]
    spec.geom('terrain').pos = [0, 0, float(bank['zmin'][0])]
    spec.add_numeric(name='moon_terrain_half', data=[half])
    print(f'[Terrain transfer] 100x100m tiled ({grid}x{grid}={count} x {tile_size_m}m, 0.1m samples, '
          f'max span={bank["height_range"].max():.2f}m); state_sha256=' + sha(ASSET/'state.npz'), flush=True)
    return spec


def half_extent(model):
    try:return float(model.numeric('moon_terrain_half').data[0])
    except KeyError:
        try:return float(model.numeric('world_0/moon_terrain_half').data[0])
        except KeyError:return 8.
