from math import radians
from pathlib import Path
import mujoco
import numpy as np
from suit_proxy_geometry import FITTED, LEGACY, PROFILES, fit_suit_proxies
ASSET=Path(__file__).resolve().parents[1]/"assets/playground"
SOLE_TERRAIN_CONTACTS = 12
SUIT_ANKLE_ROLL_LIMIT = radians(8.)
SUIT_ANKLE_ROLL_DAMPING = 8.
SUIT_ANKLE_ROLL_PROFILE = 'suit_roll_8deg_damped_v1'
SUIT_ANKLE_PITCH_LIMITS = (-radians(18.), radians(15.))
SUIT_ANKLE_PITCH_DAMPING = 8.
SUIT_ANKLE_PITCH_STIFFNESS = 15.
SUIT_ANKLE_PITCH_PROFILE = 'suit_pitch_18_15deg_bellows_v1'
SUIT_MAJOR_JOINTS_PROFILE = 'suit_eva_major_joints_v1'
SUIT_MAJOR_JOINT_SPECS = {
    # Pressurized EVA hard upper torso + waist brief/bearing resists fast twist/bend.
    'robot/WaistYaw': ((-radians(15.), radians(15.)), 16., 35.),
    'robot/WaistRoll': ((-radians(14.), radians(14.)), 16., 35.),
    'robot/WaistPitch': ((-radians(15.), radians(28.)), 12., 12.),
    # Pressurized thigh/hip brief restricts fast axial twist and lateral splay.
    'robot/LeftHipYaw': ((-radians(18.), radians(18.)), 10., 18.),
    'robot/RightHipYaw': ((-radians(18.), radians(18.)), 10., 18.),
    'robot/LeftHipRoll': ((-radians(25.), radians(25.)), 7., 12.),
    'robot/RightHipRoll': ((-radians(25.), radians(25.)), 7., 12.),
    # Leg pitch convolutes add realistic suit resistance and athletic EVA hip
    # extension (+30 deg relative to pelvis, ~16-20 deg behind vertical when
    # leaning forward) without choking forward flexion (-90 deg) or fast takeoff.
    'robot/LeftHipPitch': ((-radians(90.), radians(30.)), 4., 6.),
    'robot/RightHipPitch': ((-radians(90.), radians(30.)), 4., 6.),
    'robot/LeftKnee': (None, 3.5, 4.),
    'robot/RightKnee': (None, 3.5, 4.),
    # Arm convolutes keep upper limbs calm and weighty.
    'robot/LeftShoulderRoll': (None, 5., 8.),
    'robot/LeftShoulderPitch': (None, 5., 8.),
    'robot/LeftShoulderYaw': (None, 5., 8.),
    'robot/LeftElbow': (None, 5., 8.),
    'robot/RightShoulderRoll': (None, 5., 8.),
    'robot/RightShoulderPitch': (None, 5., 8.),
    'robot/RightShoulderYaw': (None, 5., 8.),
    'robot/RightElbow': (None, 5., 8.),
}
RELAXED_PROPULSION_PROFILE = 'propulsion_relaxed_v1'
RELAXED_ACTUATOR_SPECS = {
    # Reduce PD velocity brake (kd 25->12) and raise peak leg torque (180->260 Nm)
    # so explosive takeoff extension is not choked by internal actuator damping.
    'robot/LeftHipPitch': (380., 12., 260.),
    'robot/RightHipPitch': (380., 12., 260.),
    'robot/LeftKnee': (380., 12., 260.),
    'robot/RightKnee': (380., 12., 260.),
    'robot/LeftHipRoll': (320., 18., 200.),
    'robot/RightHipRoll': (320., 18., 200.),
    # Give waist enough torque headroom to smoothly counter pressurized bellows.
    'robot/WaistPitch': (140., 10., 110.),
    'robot/WaistRoll': (140., 10., 110.),
    'robot/WaistYaw': (100., 12., 80.),
}


def vertical_load_columns(model):
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'contacts/robot/left_sole_reaction') < 0:
        return None
    return [int(model.sensor(f'contacts/robot/{side}_sole_reaction').adr[0]) + 2
            for side in ('left', 'right')]


def scene_spec(terrain_size=16, *, geometry=LEGACY, sole_probes=False, sole_collision=False,
               sole_edge_probes=False, exoskeleton=False, exo_pd_force=None, suit_ankle_roll=False,
               suit_ankle_pitch=False, suit_major_joints=False, relax_propulsion=False,
               vertical_support=False, stance_transition=False):
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
    if suit_ankle_roll:
        for side in ('Left', 'Right'):
            joint = spec.joint(f'robot/{side}AnkleRoll')
            joint.range = [-SUIT_ANKLE_ROLL_LIMIT, SUIT_ANKLE_ROLL_LIMIT]
            joint.damping[0] = SUIT_ANKLE_ROLL_DAMPING
    if suit_ankle_pitch:
        for side in ('Left', 'Right'):
            joint = spec.joint(f'robot/{side}Ankle')
            joint.range = list(SUIT_ANKLE_PITCH_LIMITS)
            joint.damping[0] = SUIT_ANKLE_PITCH_DAMPING
            joint.stiffness[0] = SUIT_ANKLE_PITCH_STIFFNESS
            joint.springref = 0.
    if suit_major_joints:
        for name, (limits, damping, stiffness) in SUIT_MAJOR_JOINT_SPECS.items():
            joint = spec.joint(name)
            if limits is not None:
                joint.range = list(limits)
            joint.damping[0] = damping
            joint.stiffness[0] = stiffness
            joint.springref = 0.
    if relax_propulsion:
        for name, (kp, kd, force_lim) in RELAXED_ACTUATOR_SPECS.items():
            act = spec.actuator(name)
            act.gainprm[0] = kp
            act.biasprm[1] = -kp
            act.biasprm[2] = -kd
            act.forcerange = [-force_lim, force_lim]
    if sole_probes or sole_edge_probes:
        add_sole_probes(spec, include_edges=sole_edge_probes)
    if sole_collision:
        restore_sole_collision(spec, vertical_support=vertical_support)
    if exoskeleton:
        from exoskeleton import add_exoskeleton
        add_exoskeleton(spec, pd_force=exo_pd_force, guard_action_force=suit_ankle_roll, contact_tangent=stance_transition)
    return spec


def restore_sole_collision(spec, *, vertical_support=False):
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

        if vertical_support:
            # Order defines the direction of the world-frame resultant on the boot.
            for field, dataspec, reduction in (('reaction', 2, 3), ('min_dist', 8, 1)):
                spec.add_sensor(name=f'contacts/robot/{side}_sole_{field}',
                                type=mujoco.mjtSensor.mjSENS_CONTACT,
                                objtype=mujoco.mjtObj.mjOBJ_GEOM, objname='terrain',
                                reftype=mujoco.mjtObj.mjOBJ_GEOM, refname=f'robot/{side}_sole',
                                intprm=[dataspec, reduction, 1])


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

