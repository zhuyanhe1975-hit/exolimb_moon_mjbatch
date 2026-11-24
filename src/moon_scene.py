from math import radians
from pathlib import Path
import mujoco
import numpy as np
from suit_proxy_geometry import (BODYFIT_HULL, FIT3D_18V, FITTED, FITTED_3BOX, LEGACY, PROFILES,
                                 apply_3box_sole_collision, apply_bodyfit_hull_collision,
                                 apply_fit3d_18v_collision, fit_suit_proxies)
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
APOLLO_MASS_PROFILE = 'apollo_a7lb_mass_v1'
WANGYU_EVA_PROFILE = 'wangyu_lunar_eva_v1'
MASS_PROFILES = (APOLLO_MASS_PROFILE, WANGYU_EVA_PROFILE)
# NASA Apollo A7LB lunar surface EVA mass proportions (~75 kg astronaut + ~80 kg EMU/PLSS/OPS = ~155 kg human+suit, ~157 kg w/ exo):
# PLSS (36 kg loaded) + OPS (15 kg) = 51.0 kg backpack + 31.5 kg chest/HUT/DCU = 82.5 kg on Torso (~53% of total mass).
APOLLO_CHEST_MASS_KG = 31.5
APOLLO_PACK_MASS_KG = 51.0
APOLLO_BODY_MASSES = {
    'robot/Pelvis': 15.0,
    'robot/Head': 8.5,
    'robot/LeftUpperArm': 3.0,
    'robot/RightUpperArm': 3.0,
    'robot/LeftForearmHand': 3.0,
    'robot/RightForearmHand': 3.0,
    'robot/LeftThigh': 9.5,
    'robot/RightThigh': 9.5,
    'robot/LeftShin': 5.5,
    'robot/RightShin': 5.5,
    'robot/LeftFoot': 3.5,
    'robot/RightFoot': 3.5,
}
# China CMSA Wangyu (望宇) lunar spacesuit proportions:
# ~70 kg astronaut + ~97 kg Wangyu lunar suit (~20% lighter than 120 kg Feitian orbital suit) = ~167 kg (~169 kg w/ exo).
# Integrated rear-entry PLSS backpack = 52.0 kg (~54% of suit mass, ~31% of total) + 39.0 kg HUT/chest/hatch = 91.0 kg Torso.
WANGYU_CHEST_MASS_KG = 39.0
WANGYU_PACK_MASS_KG = 52.0
WANGYU_PACK_COM_LOCAL = (-0.275, -0.0029, 0.395)  # more compact/integrated rear-entry PLSS than stacked Apollo PLSS+OPS
WANGYU_SOLE_FRICTION = 0.82  # treaded anti-dust lunar boot on compacted lunar regolith
WANGYU_BODY_MASSES = {
    'robot/Pelvis': 16.0,
    'robot/Head': 8.8,
    'robot/LeftUpperArm': 3.2,
    'robot/RightUpperArm': 3.2,
    'robot/LeftForearmHand': 3.1,
    'robot/RightForearmHand': 3.1,
    'robot/LeftThigh': 9.8,
    'robot/RightThigh': 9.8,
    'robot/LeftShin': 5.8,
    'robot/RightShin': 5.8,
    'robot/LeftFoot': 3.8,
    'robot/RightFoot': 3.8,
}
# Wangyu low-gravity pressurized (~30-40 kPa) rotary bearings & flexible lunar bellows:
# supports deep bending/squatting/kneeling while keeping realistic pressurized damping/restoring torque.
WANGYU_ANKLE_PITCH_LIMITS = (-radians(20.), radians(18.))
WANGYU_ANKLE_PITCH_DAMPING = 7.5
WANGYU_ANKLE_PITCH_STIFFNESS = 14.0
WANGYU_MAJOR_JOINT_SPECS = {
    'robot/WaistYaw': ((-radians(18.), radians(18.)), 14., 28.),
    'robot/WaistRoll': ((-radians(14.), radians(14.)), 15., 32.),
    'robot/WaistPitch': ((-radians(15.), radians(35.)), 10., 10.),
    'robot/LeftHipYaw': ((-radians(20.), radians(20.)), 8., 14.),
    'robot/RightHipYaw': ((-radians(20.), radians(20.)), 8., 14.),
    'robot/LeftHipRoll': ((-radians(25.), radians(25.)), 6.5, 11.),
    'robot/RightHipRoll': ((-radians(25.), radians(25.)), 6.5, 11.),
    'robot/LeftHipPitch': ((-radians(100.), radians(30.)), 3.5, 5.),
    'robot/RightHipPitch': ((-radians(100.), radians(30.)), 3.5, 5.),
    'robot/LeftKnee': (None, 3.0, 3.5),
    'robot/RightKnee': (None, 3.0, 3.5),
    'robot/LeftShoulderRoll': (None, 4.5, 7.),
    'robot/LeftShoulderPitch': (None, 4.5, 7.),
    'robot/LeftShoulderYaw': (None, 4.5, 7.),
    'robot/LeftElbow': (None, 4.5, 7.),
    'robot/RightShoulderRoll': (None, 4.5, 7.),
    'robot/RightShoulderPitch': (None, 4.5, 7.),
    'robot/RightShoulderYaw': (None, 4.5, 7.),
    'robot/RightElbow': (None, 4.5, 7.),
}
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


def _box_inertia(mass, half_sizes):
    hx, hy, hz = [float(v) for v in half_sizes]
    return (mass / 3.) * np.diag([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])


def _apply_mass_profile(spec, body_masses, chest_mass, pack_mass, pack_com=None):
    """Apply realistic EVA suit + PLSS mass and exact parallel-axis inertia."""
    compiled = spec.compile()
    for name, target_mass in body_masses.items():
        src = compiled.body(name)
        body = spec.body(name)
        scale = float(target_mass) / max(float(src.mass[0]), 1e-6)
        body.mass = float(target_mass)
        body.ipos = src.ipos
        body.iquat = src.iquat
        body.inertia = src.inertia * scale
        body.explicitinertial = True

    pack = spec.geom('robot/backpack')
    c_chest = np.array([0., 0., .227], dtype=float)
    c_pack = np.asarray(pack.pos if pack_com is None else pack_com, dtype=float)
    m_c, m_p = float(chest_mass), float(pack_mass)
    m_tot = m_c + m_p
    com = (m_c * c_chest + m_p * c_pack) / m_tot
    i_c = _box_inertia(m_c, [.14, .19, .22])
    i_p = _box_inertia(m_p, pack.size)
    d_c = c_chest - com
    d_p = c_pack - com
    tensor = (i_c + m_c * (float(d_c @ d_c) * np.eye(3) - np.outer(d_c, d_c))
              + i_p + m_p * (float(d_p @ d_p) * np.eye(3) - np.outer(d_p, d_p)))
    eigvals, eigvecs = np.linalg.eigh(tensor)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1.
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, eigvecs.ravel())
    torso = spec.body('robot/Torso')
    torso.mass = float(m_tot)
    torso.ipos = com.tolist()
    torso.iquat = quat.tolist()
    torso.inertia = np.maximum(eigvals, 1e-4).tolist()
    torso.explicitinertial = True


def apply_apollo_mass_distribution(spec):
    """Apply NASA Apollo A7LB + PLSS/OPS lunar EVA mass & inertia proportions."""
    _apply_mass_profile(spec, APOLLO_BODY_MASSES, APOLLO_CHEST_MASS_KG, APOLLO_PACK_MASS_KG)


def apply_wangyu_eva_physics(spec):
    """Apply China CMSA Wangyu (望宇) lunar spacesuit mass, inertia, and regolith sole friction."""
    _apply_mass_profile(spec, WANGYU_BODY_MASSES, WANGYU_CHEST_MASS_KG,
                        WANGYU_PACK_MASS_KG, pack_com=WANGYU_PACK_COM_LOCAL)
    for side in ('left', 'right'):
        sole = spec.geom(f'robot/{side}_sole')
        sole.friction[0] = WANGYU_SOLE_FRICTION


def scene_spec(terrain_size=16, *, geometry=LEGACY, sole_probes=False, sole_collision=False,
               sole_edge_probes=False, exoskeleton=False, exo_pd_force=None, exo_mechanism=None,
               exo_detached=False, exo_hoof_pad_collision=True,
               suit_ankle_roll=False, suit_ankle_pitch=False, suit_major_joints=False,
               relax_propulsion=False, apollo_mass=False, mass_profile=None,
               vertical_support=False, stance_transition=False, tiled_100m=False):
    if geometry not in PROFILES:
        raise ValueError(f'Unknown suit geometry profile: {geometry}')
    if mass_profile is None and apollo_mass:
        mass_profile = APOLLO_MASS_PROFILE
    if mass_profile not in (None, *MASS_PROFILES):
        raise ValueError(f'Unknown suit mass profile: {mass_profile}')
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
        from large_terrain import configure, configure_tiled
        (configure_tiled if tiled_100m else configure)(spec)
    elif terrain_size!=16:raise ValueError('Supported terrain sizes: 16, 100')
    if geometry in (FITTED, FITTED_3BOX, BODYFIT_HULL, FIT3D_18V):
        fit_suit_proxies(spec)
    is_wangyu = mass_profile == WANGYU_EVA_PROFILE
    if mass_profile == APOLLO_MASS_PROFILE:
        apply_apollo_mass_distribution(spec)
    elif is_wangyu:
        apply_wangyu_eva_physics(spec)
    if suit_ankle_roll:
        for side in ('Left', 'Right'):
            joint = spec.joint(f'robot/{side}AnkleRoll')
            joint.range = [-SUIT_ANKLE_ROLL_LIMIT, SUIT_ANKLE_ROLL_LIMIT]
            joint.damping[0] = SUIT_ANKLE_ROLL_DAMPING
    if suit_ankle_pitch:
        pitch_lim = WANGYU_ANKLE_PITCH_LIMITS if is_wangyu else SUIT_ANKLE_PITCH_LIMITS
        pitch_damp = WANGYU_ANKLE_PITCH_DAMPING if is_wangyu else SUIT_ANKLE_PITCH_DAMPING
        pitch_stiff = WANGYU_ANKLE_PITCH_STIFFNESS if is_wangyu else SUIT_ANKLE_PITCH_STIFFNESS
        for side in ('Left', 'Right'):
            joint = spec.joint(f'robot/{side}Ankle')
            joint.range = list(pitch_lim)
            joint.damping[0] = pitch_damp
            joint.stiffness[0] = pitch_stiff
            joint.springref = 0.
    if suit_major_joints:
        joint_specs = WANGYU_MAJOR_JOINT_SPECS if is_wangyu else SUIT_MAJOR_JOINT_SPECS
        for name, (limits, damping, stiffness) in joint_specs.items():
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
        from exoskeleton import MECHANISM, add_exoskeleton
        add_exoskeleton(spec, pd_force=exo_pd_force, guard_action_force=suit_ankle_roll,
                        contact_tangent=stance_transition, mechanism=exo_mechanism or MECHANISM,
                        detached=exo_detached, hoof_pad_collision=exo_hoof_pad_collision)
    if geometry == FITTED_3BOX:
        apply_3box_sole_collision(spec)
    elif geometry == BODYFIT_HULL:
        apply_bodyfit_hull_collision(spec)
    elif geometry == FIT3D_18V:
        apply_fit3d_18v_collision(spec)
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

