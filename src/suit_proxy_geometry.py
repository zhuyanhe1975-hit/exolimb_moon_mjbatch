"""Body-local boxes fitted to the native astronaut skin at its rest pose."""
from math import cos, radians, sin

LEGACY = 'legacy'
FITTED = 'fitted'
PROFILES = (LEGACY, FITTED)


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
