from pathlib import Path
import mujoco
import numpy as np
from suit_proxy_geometry import FITTED, LEGACY, PROFILES, fit_suit_proxies
ASSET=Path(__file__).resolve().parents[1]/"assets/playground"
SOLE_TERRAIN_CONTACTS = 12

def scene_spec(terrain_size=16, *, geometry=LEGACY, sole_probes=False, sole_collision=False,
               sole_edge_probes=False, exoskeleton=False, exo_pd_force=None):
    if geometry not in PROFILES:
        raise ValueError(f'Unknown suit geometry profile: {geometry}')
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
    if geometry == FITTED:
        fit_suit_proxies(spec)
    if sole_probes or sole_edge_probes:
        add_sole_probes(spec, include_edges=sole_edge_probes)
    if sole_collision:
        restore_sole_collision(spec)
    if exoskeleton:
        from exoskeleton import add_exoskeleton
        add_exoskeleton(spec, pd_force=exo_pd_force)
    return spec


def restore_sole_collision(spec):
    """Let native contact backstop the desired separation of the two boots."""
    exclusion = next(item for item in spec.excludes if item.name == 'robot/no_foot_self_collision')
    spec.delete(exclusion)
    spec.add_sensor(name='robot/balance_sole_separation', type=mujoco.mjtSensor.mjSENS_GEOMDIST,
                    objtype=mujoco.mjtObj.mjOBJ_GEOM, objname='robot/left_sole',
                    reftype=mujoco.mjtObj.mjOBJ_GEOM, refname='robot/right_sole', cutoff=.10)
    for side in ('left', 'right'):
        # Ground-only force is required once the boots can touch each other.
        spec.add_sensor(name=f'contacts/robot/{side}_sole_terrain_load',
                        type=mujoco.mjtSensor.mjSENS_CONTACT,
                        objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=f'robot/{side}_sole',
                        reftype=mujoco.mjtObj.mjOBJ_GEOM, refname='terrain',
                        intprm=[3, 2, SOLE_TERRAIN_CONTACTS])


def add_sole_probes(spec, *, include_edges=False):
    """Measure the front and rear of each collision sole in world coordinates."""
    for side in ('left', 'right'):
        geom = spec.geom(f'robot/{side}_sole')
        points = [('heel', 0, -1), ('toe', 0, 1)]
        if include_edges:
            points.extend((('negative_edge', 1, -1), ('positive_edge', 1, 1)))
        for region, axis, direction in points:
            offset = np.array([0., 0., -geom.size[2]])
            offset[axis] = direction * .85 * geom.size[axis]
            rotated = np.empty(3)
            mujoco.mju_rotVecQuat(rotated, offset, geom.quat)
            name = f'robot/balance_{side}_{region}'
            geom.parent.add_site(name=name, pos=geom.pos + rotated, size=[.001, .001, .001],
                                 group=4, rgba=[0., 0., 0., 0.])
            spec.add_sensor(name=f'{name}_pos', type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name)

