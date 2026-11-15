"""Spaced playback starts on the saved terrain, without regenerating geometry."""
import numpy as np
from scipy.ndimage import maximum_filter
from local_lunar_terrain import HALF,source_height


def distributed_spawns(bank,patch,sole_points,cover,count,occupied,seed):
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
    c=np.cos(yaw);s=np.sin(yaw);foot=np.asarray(sole_points)
    points=np.stack((xy[:,None,0]+c[:,None]*foot[None,:,0]-s[:,None]*foot[None,:,1],
                     xy[:,None,1]+s[:,None]*foot[None,:,0]+c[:,None]*foot[None,:,1]),-1)
    grid=bank['normalized'][patch]
    span=float(bank['height_range'][patch]);low=float(bank['zmin'][patch])
    zgrid=grid*span+low
    a=zgrid[:-1,:-1];b=zgrid[:-1,1:];d=zgrid[1:,1:];e=zgrid[1:,:-1]
    slope=np.maximum(np.hypot(d-e,e-a),np.hypot(b-a,d-b))/step
    radius=int(np.ceil(np.linalg.norm(foot[:,:2],axis=1).max()/step))+2
    bound=maximum_filter(slope,size=2*radius+1,mode='nearest')
    ij=np.floor((xy+half)/step).astype(int)
    padding=bound[ij[:,1],ij[:,0]]*cover+2e-6
    heights=source_height(points,{'half_extent_m':half,'height_range_m':span,'zmin_m':low},grid)
    z=(heights-foot[None,:,2]).max(-1)+.015+padding
    gaps=z[:,None]+foot[None,:,2]-heights
    valid=np.isfinite(gaps).all(-1)&(gaps.min(-1)>=.0145)&(gaps.max(-1)<=.20)
    xy=xy[valid];yaw=yaw[valid];padding=padding[valid]
    if len(xy)<count:raise ValueError(f'Only {len(xy)} safe playback placements for {count} people')
    occupied=np.asarray(occupied).reshape(-1,2)
    distance=np.full(len(xy),np.inf)
    for point in occupied:distance=np.minimum(distance,((xy-point)**2).sum(-1))
    selected=[]
    for _ in range(count):
        index=int(rng.integers(len(xy))) if not selected and not len(occupied) else int(distance.argmax())
        selected.append(index)
        distance=np.minimum(distance,((xy-xy[index])**2).sum(-1))
        distance[selected]=-np.inf
    return xy[selected],yaw[selected],padding[selected]
