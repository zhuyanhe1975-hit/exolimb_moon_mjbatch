"""Display-only material/lighting reference from the original Isaac preview."""
from pathlib import Path
from functools import lru_cache
import hashlib,json,os
import mujoco
ASSET=Path(__file__).resolve().parents[1]/'assets/regolith_pbr'
# Filament's measured-metal reference gives gold F0 in sRGB, not linear RGB.
# MaterialManager's default BaseColorFactor input is also sRGB; do not apply
# another gamma conversion here. This models an optically thick gold coating,
# not wavelength-dependent interference/transmission through a thin film.
# https://google.github.io/filament/main/filament.html#materialsystem/parameterization/remapping
VISOR_GOLD_SRGB=(1.00,.85,.57,1.)
VISOR_ROUGHNESS=.04  # polished mirror; Filament clamps its GGX lobe safely


@lru_cache(maxsize=1)
def verify_maps():
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((ASSET/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Regolith texture SHA mismatch: {name}')


@lru_cache(maxsize=1)
def verify_lunar_ibl():
    asset=ASSET.parent/'lunar_ibl'
    manifest=json.loads((asset/'manifest.json').read_text())
    if hashlib.sha256((ASSET/'diffuse.png').read_bytes()).hexdigest()!=manifest['source_sha256']:
        raise ValueError('Lunar IBL source albedo SHA mismatch')
    if hashlib.sha256((ASSET/'normal.png').read_bytes()).hexdigest()!=manifest['normal_sha256']:
        raise ValueError('Lunar IBL source normal SHA mismatch')
    for name,expected in manifest['files'].items():
        if hashlib.sha256((asset/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Lunar IBL SHA mismatch: {name}')
    return asset


def add_reference_environment(spec):
    verify_maps()
    material=spec.add_material(name='filament_lunar_regolith',rgba=[1,1,1,1],
        metallic=0.,roughness=1.,specular=.26,texuniform=True,texrepeat=[.25,.25])
    for name,role,space in (
        ('diffuse',mujoco.mjtTextureRole.mjTEXROLE_RGB,mujoco.mjtColorSpace.mjCOLORSPACE_SRGB),
        ('normal',mujoco.mjtTextureRole.mjTEXROLE_NORMAL,mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR),
        ('orm',mujoco.mjtTextureRole.mjTEXROLE_ORM,mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)):
        texture=spec.add_texture(name='regolith_'+name,type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(ASSET/(name+'.png')),vflip=True,colorspace=space)
        material.textures[role]=texture.name
    for geom in spec.geoms:
        if geom.name=='terrain' or geom.name.startswith('display_terrain_'):
            geom.material=material.name
            # White lets the color map set albedo without another grey multiplier.
            geom.rgba=[1,1,1,1]
    sun=spec.light('sun')
    sun.dir=[-2**-.5,0,-2**-.5]
    sun.diffuse=[1,1,1];sun.specular=[1,1,1]
    # A positive intensity selects MuJoCo 4.0.1 photometric lighting and
    # suppresses its built-in studio/room IBL. USD 1750 is not the same unit;
    # 80000 keeps MuJoCo's former sun calibration without the fallback room.
    sun.intensity=80000.
    # Sharp black-sky/ground boundary and real regolith albedo variation,
    # replacing the builtin sigmoid gradient that washed out visor reflections.
    # This shared static plane approximation does not ray trace the active
    # heightfield or nearby people. Radiance is albedo*cos(45deg)/pi relative
    # to the sun, hence use the SAME intensity as the directional light.
    environment=os.environ.get('MOON_REFLECTION_ENV','lunar')
    if environment=='lunar':
        ibl=verify_lunar_ibl()
        # KTX carries both the reflection cube AND diffuse irradiance SH.
        spec.add_texture(name='lunar_bounce_ibl',type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(ibl/'lunar.ktx'),colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)
    else:
        if environment=='royal':
            path=ASSET.parent/'reflection_reference/royal_esplanade.ktx'
            manifest=json.loads(path.with_suffix('.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:
                raise ValueError('Reference HDRI SHA mismatch')
        else:path=Path(environment).expanduser().resolve()
        if path.suffix.lower()!='.ktx' or not path.is_file():
            raise ValueError('Reflection environment must be lunar, royal, or an existing KTX cubemap')
        # MuJoCo stores compressed KTX as an opaque 2D blob; the KTX header
        # defines its six cube faces (same convention as upstream pbr.xml).
        spec.add_texture(name='lunar_bounce_ibl',type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(path),colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)
    spec.worldbody.add_light(name='lunar_bounce',type=mujoco.mjtLightType.mjLIGHT_IMAGE,
        texture='lunar_bounce_ibl',intensity=sun.intensity if environment=='lunar' else 30000.)
    spec.visual.headlight.ambient=[0,0,0]
    spec.visual.headlight.diffuse=[0,0,0]
    spec.visual.headlight.specular=[0,0,0]
    for name,value in (('fallback.head_light_intensity',0),
                       ('fallback.environment_light_intensity',0),
                       ('shadows.map_size',4096),('ao.enabled',1),('bloom.enabled',0)):
        spec.add_numeric(name='filament.'+name,data=[value])



@lru_cache(maxsize=1)
def verify_suit_maps():
    asset=ASSET.parent/'suit_pbr'
    manifest=json.loads((asset/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        if hashlib.sha256((asset/name).read_bytes()).hexdigest()!=value:
            raise ValueError(f'Suit texture SHA mismatch: {name}')
    return asset


@lru_cache(maxsize=1)
def verify_visor_maps():
    asset=ASSET.parent/'visor_pbr'
    manifest=json.loads((asset/'manifest.json').read_text())
    for item in manifest['sources'].values():
        if hashlib.sha256((ASSET.parents[1]/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Visor source atlas SHA mismatch')
    for name,expected in manifest['files'].items():
        if hashlib.sha256((asset/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Visor texture SHA mismatch: {name}')
    return asset


@lru_cache(maxsize=1)
def verify_clean_cloth():
    asset=ASSET.parent/'clean_cloth'
    manifest=json.loads((asset/'manifest.json').read_text())
    for name,expected in manifest['sources'].items():
        if hashlib.sha256((ASSET.parents[1]/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Clean cloth source SHA mismatch: {name}')
    for name,expected in manifest['files'].items():
        if hashlib.sha256((asset/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Clean cloth texture SHA mismatch: {name}')
    return asset


def add_suit_maps(spec):
    asset=verify_suit_maps()
    for part in ('Helmet','Boot','Suit'):
        for suffix,role in (('Metallic',mujoco.mjtTextureRole.mjTEXROLE_METALLIC),
                            ('Roughness',mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS),
                            ('Normal',mujoco.mjtTextureRole.mjTEXROLE_NORMAL)):
            name=f'pbr_{part}_{suffix}'
            spec.add_texture(name=name,type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=str(asset/f'{part}_{suffix}.png'),vflip=True,
                colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)
            for index in ({'Helmet':(0,1),'Boot':(2,),'Suit':(3,)}[part]):
                material=spec.material(f'skin_mat_{index}')
                material.textures[role]=name
                material.metallic=1.;material.roughness=1.;material.reflectance=.5
    # Glass is the occluded inner visor. The visible Shield face is handled
    # separately below; helmet-side parts remain dielectric shell/frame.
    shell=spec.material('skin_mat_1')
    shell.metallic=0.
    shell.textures[mujoco.mjtTextureRole.mjTEXROLE_METALLIC]=''
    visor=spec.material('skin_mat_0')
    visor.rgba=VISOR_GOLD_SRGB
    visor.metallic=1.;visor.roughness=VISOR_ROUGHNESS
    # mjMaterial.reflectance is MuJoCo's planar reflection blend, NOT the
    # dielectric Fresnel control called reflectance in Filament's own docs.
    # Curved visor reflections come from metallic F0 + the shared scene IBL.
    visor.reflectance=0.
    for role in (mujoco.mjtTextureRole.mjTEXROLE_RGB,mujoco.mjtTextureRole.mjTEXROLE_METALLIC,
                 mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS,mujoco.mjtTextureRole.mjTEXROLE_NORMAL):
        visor.textures[role]=''
    # The VISIBLE face is Suit_Shield, not the occluded Glass mesh. Shield's
    # frame shares the same skin, so preserve its dielectric pixels via a mask.
    maps=verify_visor_maps()
    front=spec.add_material(name='skin_mat_visor',rgba=[1,1,1,1],metallic=1.,roughness=1.,reflectance=0.)
    for name,role,color_space in (
        ('visor_color',mujoco.mjtTextureRole.mjTEXROLE_RGB,mujoco.mjtColorSpace.mjCOLORSPACE_SRGB),
        ('visor_metallic',mujoco.mjtTextureRole.mjTEXROLE_METALLIC,mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR),
        ('visor_roughness',mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS,mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR),
        ('visor_normal',mujoco.mjtTextureRole.mjTEXROLE_NORMAL,mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR),
        ('shell_color',mujoco.mjtTextureRole.mjTEXROLE_RGB,mujoco.mjtColorSpace.mjCOLORSPACE_SRGB),
        ('shell_roughness',mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS,mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)):
        spec.add_texture(name=name,type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(maps/(name+'.png')),vflip=True,colorspace=color_space)
        (shell if name.startswith('shell_') else front).textures[role]=name
    # Repoint existing shared texture assets: no duplicate per-person maps.
    clean=verify_clean_cloth()
    for part,index in (('Suit',4),('Boot',3)):
        spec.texture(f'skin_tex_{index}').file=str(clean/f'{part}_BaseColor.png')
        for suffix in ('Metallic','Roughness','Normal'):
            spec.texture(f'pbr_{part}_{suffix}').file=str(clean/f'{part}_{suffix}.png')
    spec.texture('pbr_Helmet_Normal').file=str(clean/'Helmet_Normal.png')
