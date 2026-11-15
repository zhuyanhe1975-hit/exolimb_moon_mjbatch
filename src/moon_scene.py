from pathlib import Path
import mujoco
import numpy as np
ASSET=Path(__file__).resolve().parents[1]/"assets/playground"

def scene_spec(terrain_size=16):
    from provenance import verify_assets
    verify_assets()
    spec=mujoco.MjSpec.from_file(str(ASSET/'scene/scene.xml'))
    spec.modelname='exolimb_moon_mujoco_playground'
    with np.load(ASSET/'state.npz') as z:
        for i in range(64):
            h=next(h for h in spec.hfields if h.name==f'site20_local_{i:02d}')
            h.userdata=z['normalized'][i].ravel().tolist()
            h.size=[8.,8.,float(z['height_range'][i]),1.]
        next(g for g in spec.geoms if g.name=='terrain').pos=[0.,0.,float(z['zmin'][0])]
        spec.option.timestep=1/120
        spec.option.gravity=[0.,0.,-1.635]
    # Native contact sensors: preserve four slots per sole and same load filter.
    # Do not use a stale host-side contact cache or synthesize TOUCH from state.
    for side in ('left','right'):
        spec.add_sensor(name=f'contacts/robot/{side}_sole',type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,objname=f'robot/{side}_sole',
            reftype=mujoco.mjtObj.mjOBJ_GEOM,refname='terrain',intprm=[27,2,4])
    if terrain_size==100:
        from large_terrain import configure
        configure(spec)
    elif terrain_size!=16:raise ValueError('Supported terrain sizes: 16, 100')
    return spec

