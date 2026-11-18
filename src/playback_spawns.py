"""Spaced playback starts on the saved terrain, without regenerating geometry."""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
from local_lunar_terrain import HALF,source_height


def _safe_placements(xy, yaw, foot, padding, height_meta, grid):
    c, s = np.cos(yaw), np.sin(yaw)
    points = np.stack(
        (xy[:, None, 0] + c[:, None] * foot[None, :, 0] - s[:, None] * foot[None, :, 1],
         xy[:, None, 1] + s[:, None] * foot[None, :, 0] + c[:, None] * foot[None, :, 1]),
        axis=-1,
    )
    heights = source_height(points, height_meta, grid)
    z = (heights - foot[None, :, 2]).max(-1) + .015 + padding
    gaps = z[:, None] + foot[None, :, 2] - heights
    return np.isfinite(gaps).all(-1) & (gaps.min(-1) >= .0145) & (gaps.max(-1) <= .20)


def distributed_spawns(bank,patch,sole_points,cover,count,occupied,seed,*,inward_heading=False):
    rng=np.random.default_rng(seed)
    # Keep spawn and the allowed 1m drift clear of the terrain boundary.
    half=float(bank.get('half_extent_m',HALF))
    step=2*half/(bank['normalized'].shape[1]-1)
    limit=half-5. if half>8 else 5.
    axis=np.linspace(-limit,limit,81 if half>8 else 41)
    xx,yy=np.meshgrid(axis,axis)
    xy=np.stack((xx,yy),-1).reshape(-1,2)
    xy+=rng.uniform(-.04,.04,xy.shape)
    xy=np.clip(xy,-limit,limit)
    yaw=rng.uniform(-np.pi,np.pi,len(xy))
    if inward_heading:
        yaw=np.where(np.linalg.norm(xy,axis=1)>.5,np.arctan2(-xy[:,1],-xy[:,0]),yaw)
    foot=np.asarray(sole_points)
    grid=bank['normalized'][patch]
    span=float(bank['height_range'][patch]);low=float(bank['zmin'][patch])
    zgrid=grid*span+low
    a=zgrid[:-1,:-1];b=zgrid[:-1,1:];d=zgrid[1:,1:];e=zgrid[1:,:-1]
    slope=np.maximum(np.hypot(d-e,e-a),np.hypot(b-a,d-b))/step
    radius=int(np.ceil(np.linalg.norm(foot[:,:2],axis=1).max()/step))+2
    bound=maximum_filter(slope,size=2*radius+1,mode='nearest')
    ij=np.floor((xy+half)/step).astype(int)
    padding=bound[ij[:,1],ij[:,0]]*cover+2e-6
    occupied=np.asarray(occupied).reshape(-1,2)
    height_meta = {'half_extent_m': half, 'height_range_m': span, 'zmin_m': low}
    # An initial random pick needs the full valid set. A small partial reset
    # can check candidates in greedy distance order and discard invalid ones.
    check_on_pick = len(occupied) > 0 and count <= 64
    if not check_on_pick:
        valid = _safe_placements(xy, yaw, foot, padding, height_meta, grid)
        xy, yaw, padding = xy[valid], yaw[valid], padding[valid]
        if len(xy) < count:
            raise ValueError(f'Only {len(xy)} safe playback placements for {count} people')
    distance=np.full(len(xy),np.inf)
    # Compute nearest occupied distances in native code with bounded scratch space.
    for start in range(0,len(occupied),128):
        block=cdist(occupied[start:start+128],xy,metric='sqeuclidean')
        np.minimum(distance,block.min(axis=0),out=distance)
    x,y=xy.T.copy()
    selected=[]
    while len(selected) < count:
        index=int(rng.integers(len(xy))) if not selected and not len(occupied) else int(distance.argmax())
        if distance[index] == -np.inf:
            raise ValueError(f'Only {len(selected)} safe playback placements for {count} people')
        distance[index] = -np.inf
        if check_on_pick:
            candidate = slice(index, index + 1)
            if not _safe_placements(
                xy[candidate], yaw[candidate], foot, padding[candidate], height_meta, grid
            )[0]:
                continue
        selected.append(index)
        np.minimum(distance,(x-x[index])**2+(y-y[index])**2,out=distance)
    return xy[selected],yaw[selected],padding[selected]


class IndependentSpawns:
    """Sample position and heading independently from a safe product set."""

    def __init__(self, bank, patches, sole_points, cover, seed):
        self.patches = np.asarray(patches)
        self.positions = {}
        self.headings = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        position_seed, heading_seed = np.random.SeedSequence(seed).spawn(2)
        self.position_rng = np.random.default_rng(position_seed)
        self.heading_rng = np.random.default_rng(heading_seed)
        half = float(bank.get('half_extent_m', HALF))
        axis = np.linspace(-2.4, 2.4, 13)
        xy = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        step = 2 * half / (bank['normalized'].shape[1] - 1)
        radius = int(np.ceil(np.linalg.norm(sole_points[:, :2], axis=-1).max() / step)) + 2
        ij = np.floor((xy + half) / step).astype(int)
        for patch in np.unique(self.patches):
            grid = bank['normalized'][patch]
            span, low = float(bank['height_range'][patch]), float(bank['zmin'][patch])
            heights = grid * span + low
            a, b = heights[:-1, :-1], heights[:-1, 1:]
            c, d = heights[1:, :-1], heights[1:, 1:]
            slopes = np.maximum(np.hypot(d-c, c-a), np.hypot(b-a, d-b)) / step
            bound = maximum_filter(slopes, size=2*radius+1, mode='nearest')
            padding = bound[ij[:, 1], ij[:, 0]] * cover + 2e-6
            meta = dict(half_extent_m=half, height_range_m=span, zmin_m=low)
            candidates = np.arange(len(xy))
            for yaw in self.headings:
                good = _safe_placements(xy[candidates], np.full(len(candidates), yaw),
                                        sole_points, padding[candidates], meta, grid)
                candidates = candidates[good]
            if not len(candidates):
                raise ValueError(f'Terrain {patch + 1} has no starts safe for all headings')
            self.positions[patch] = (xy[candidates], padding[candidates])

    def sample(self, indices):
        xy = np.empty((len(indices), 2))
        padding = np.empty(len(indices))
        patches = self.patches[indices]
        for patch in np.unique(patches):
            rows = patches == patch
            positions, margins = self.positions[patch]
            count = int(rows.sum())
            chosen = self.position_rng.choice(len(positions), count, replace=count > len(positions))
            xy[rows] = positions[chosen]
            padding[rows] = margins[chosen]
        yaw = self.heading_rng.choice(self.headings, len(indices))
        return xy, yaw, padding
