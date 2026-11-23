"""Body-local boxes and convex hulls fitted to the native astronaut skin at rest."""
import hashlib
import json
from math import cos, radians, sin
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODYFIT_HULL_ASSET = ROOT / 'assets/body_fit_hull'
BOOT_HULL_18V_ASSET = ROOT / 'assets/boot_hull_18v'

LEGACY = 'legacy'
FITTED = 'fitted'
FITTED_3BOX = 'fitted_3box'
BODYFIT_HULL = 'bodyfit_v2_hull'
FIT3D_18V = 'fit3d_18v'
PROFILES = (LEGACY, FITTED, FITTED_3BOX, BODYFIT_HULL, FIT3D_18V)

# Optimal 3-box flat-bottom composite fitted to the 16-direction below-ankle boot
# support polygon (IoU = 0.927 vs bodyfit_v2_hull, +5.6% PPO iter cost on Warp).
# Each tuple is (cx, cy_left, hx, hy, yaw_rad_left) in robot/LeftFoot local frame.
SOLE_3BOX_LEFT = (
    (0.0988, 0.0457, 0.1033, 0.0469, -0.0003),
    (0.1347, 0.0058, 0.0983, 0.0471, 0.4409),
    (-0.0009, -0.0003, 0.0695, 0.0688, -0.0634),
)


def fit_suit_proxies(spec):
    """Fit the collision soles and the non-colliding backpack envelope."""
    legacy = spec.compile()
    # The export infers inertia from primitive geoms. Freeze the original
    # body mass properties before moving/resizing the geom that provided them.
    for name in ('robot/Torso', 'robot/LeftFoot', 'robot/RightFoot'):
        source = legacy.body(name)
        body = spec.body(name)
        body.mass = float(source.mass[0])
        body.ipos = source.ipos
        body.iquat = source.iquat
        body.inertia = source.inertia
        body.explicitinertial = True

    # Lower boot shells from skin_Boot_0 in the LeftFoot/RightFoot frames.
    # Mirror the fitted toe-out yaw and preserve the original flat sole plane.
    for side, sign in (('left', 1), ('right', -1)):
        geom = spec.geom(f'robot/{side}_sole')
        bottom = geom.pos[2] - geom.size[2]
        half_height = .0462
        yaw_half = radians(8.2 * sign) / 2
        quat = [cos(yaw_half), 0., 0., sin(yaw_half)]
        geom.pos = [.0768, sign * .0201, bottom + half_height]
        geom.size = [.1535, .0663, half_height]
        geom.quat = quat

        sole_site = spec.site(f'robot/balance_{side}_sole')
        sole_site.pos = [.0768, sign * .0201, bottom]
        sole_site.size = [.1585, .0713, .015]
        sole_site.quat = quat

    # Rear shell of skin_Suit_0, expressed in the Torso frame. Keep its
    # original collision masks; this box represents the carried mass visually.
    pack = spec.geom('robot/backpack')
    pack.pos = [-.2916, -.0029, .4431]
    pack.size = [.0955, .280, .4732]


def apply_3box_sole_collision(spec):
    """Replace single-box sole collision with the 3-box flat-bottom boot proxy.

    Keep robot/{left,right}_sole as a non-colliding reference box (group 4) so
    existing sole span/width/mount queries remain unchanged, and retarget all
    sole contact and separation sensors to the LeftFoot/RightFoot bodies.
    """
    body_map = {}
    for side, sign in (('left', 1), ('right', -1)):
        geom = spec.geom(f'robot/{side}_sole')
        body_name = f'robot/{side.title()}Foot'
        body_map[geom.name] = body_name
        bottom = float(geom.pos[2]) - float(geom.size[2])
        half_height = float(geom.size[2])
        contype, conaffinity = int(geom.contype), int(geom.conaffinity)
        friction = list(geom.friction)
        rgba = list(geom.rgba)
        solref = list(geom.solref)
        solimp = list(geom.solimp)
        condim = int(geom.condim)
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 4
        geom.rgba = [0., 0., 0., 0.]
        for idx, (cx, cy, hx, hy, yaw) in enumerate(SOLE_3BOX_LEFT):
            yaw_half = (yaw * sign) / 2.0
            quat = [cos(yaw_half), 0., 0., sin(yaw_half)]
            geom.parent.add_geom(
                name=f'robot/{side}_sole_box_{idx}',
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[cx, sign * cy, bottom + half_height],
                size=[hx, hy, half_height],
                quat=quat,
                contype=contype,
                conaffinity=conaffinity,
                condim=condim,
                friction=friction,
                solref=solref,
                solimp=solimp,
                mass=0.0,
                group=0,
                rgba=rgba,
            )
    for sensor in spec.sensors:
        if sensor.objtype == mujoco.mjtObj.mjOBJ_GEOM and sensor.objname in body_map:
            sensor.objtype = mujoco.mjtObj.mjOBJ_BODY
            sensor.objname = body_map[sensor.objname]
        if sensor.reftype == mujoco.mjtObj.mjOBJ_GEOM and sensor.refname in body_map:
            sensor.reftype = mujoco.mjtObj.mjOBJ_BODY
            sensor.refname = body_map[sensor.refname]


def _load_hull_asset_data(asset_dir, checked_files):
    manifest = json.loads((asset_dir / 'manifest.json').read_text())
    for rel in checked_files:
        expected = manifest['files'][rel]
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Boot hull asset SHA mismatch: {rel}')
    data = json.loads((asset_dir / 'fit.json').read_text())
    skin_sha = hashlib.sha256((ROOT / 'assets/wangyu_skin/skin.xml').read_bytes()).hexdigest()
    if data['source_sha256'] != skin_sha:
        raise ValueError('Boot hull source SHA mismatch against assets/wangyu_skin/skin.xml')
    return data


def _apply_mesh_hull_collision(spec, fit):
    """Replace single-box sole collision with SHA-verified convex hull meshes.

    Preserves robot/{left,right}_sole as a non-colliding reference box (group 4)
    so span/width/mount/probe queries remain unchanged, and retargets all sole
    contact and separation sensors to robot/{left,right}_sole_hull.
    """
    geom_map = {}
    for side in ('left', 'right'):
        geom = spec.geom(f'robot/{side}_sole')
        foot = fit['feet'][side]
        hull = foot['hull']
        mesh_name = f'robot/{side}_sole_hull_mesh'
        hull_geom_name = f'robot/{side}_sole_hull'
        geom_map[geom.name] = hull_geom_name
        contype, conaffinity = int(geom.contype), int(geom.conaffinity)
        friction = list(geom.friction)
        rgba = list(geom.rgba)
        solref = list(geom.solref)
        solimp = list(geom.solimp)
        condim = int(geom.condim)
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 4
        geom.rgba = [0., 0., 0., 0.]
        if not any(m.name == mesh_name for m in spec.meshes):
            spec.add_mesh(
                name=mesh_name,
                uservert=np.asarray(hull['vertices'], dtype=np.float32).ravel().tolist(),
                userface=np.asarray(hull['faces'], dtype=np.int32).ravel().tolist(),
                inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
            )
        geom.parent.add_geom(
            name=hull_geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=[0., 0., 0.],
            quat=[1., 0., 0., 0.],
            contype=contype,
            conaffinity=conaffinity,
            condim=condim,
            friction=friction,
            solref=solref,
            solimp=solimp,
            mass=0.0,
            group=0,
            rgba=rgba,
        )
    for sensor in spec.sensors:
        if sensor.objtype == mujoco.mjtObj.mjOBJ_GEOM and sensor.objname in geom_map:
            sensor.objname = geom_map[sensor.objname]
        if sensor.reftype == mujoco.mjtObj.mjOBJ_GEOM and sensor.refname in geom_map:
            sensor.refname = geom_map[sensor.refname]


def apply_bodyfit_hull_collision(spec):
    fit = _load_hull_asset_data(
        BODYFIT_HULL_ASSET, ('assets/wangyu_skin/skin.xml', 'assets/body_fit_hull/fit.json'))
    _apply_mesh_hull_collision(spec, fit)


def apply_fit3d_18v_collision(spec):
    fit = _load_hull_asset_data(
        BOOT_HULL_18V_ASSET, ('assets/wangyu_skin/skin.xml', 'assets/boot_hull_18v/fit.json'))
    _apply_mesh_hull_collision(spec, fit)
