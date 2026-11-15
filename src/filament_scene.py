"""Filament presentation assets on the CPU display copy only."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import os

ASSETS=Path(__file__).resolve().parents[1]/'assets'


def add_astronaut(spec):
    root=ET.parse(ASSETS/'wangyu_skin/skin.xml').getroot()
    for texture in root.findall('./asset/texture'):
        spec.add_texture(name=texture.get('name'),type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(ASSETS/texture.get('file')),vflip=True,colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
    for i,material in enumerate(root.findall('./asset/material')):
        mat=spec.add_material(name=material.get('name'),
            rgba=np.fromstring(material.get('rgba'),sep=' '),
            metallic=1. if i==0 else 0.,roughness=.15 if i==0 else .85)
        if material.get('texture'):
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]=material.get('texture')
    from reference_materials import add_suit_maps
    add_suit_maps(spec)
    for skin in root.findall('./deformable/skin'):
        bones=skin.findall('bone')
        def floats(element,key):return np.fromstring(element.get(key,''),sep=' ').tolist()
        material='skin_mat_visor' if skin.get('name')=='skin_Suit_Shield_0' else skin.get('material')
        spec.add_skin(name=skin.get('name'),material=material,group=1,
            vert=floats(skin,'vertex'),texcoord=floats(skin,'texcoord'),
            face=np.fromstring(skin.get('face'),sep=' ',dtype=int).tolist(),
            bodyname=['robot/'+b.get('body') for b in bones],
            bindpos=[x for b in bones for x in floats(b,'bindpos')],
            bindquat=[x for b in bones for x in floats(b,'bindquat')],
            vertid=[np.fromstring(b.get('vertid'),sep=' ',dtype=int).tolist() for b in bones],
            vertweight=[floats(b,'vertweight') for b in bones])
    # Proxy geoms stay in the model for picking/state layout, hidden by group.
    for geom in spec.geoms:
        if geom.name.startswith('robot/'):geom.group=4
    if os.environ.get('MOON_FILAMENT_TERRAIN_PBR','1')=='1':
        from reference_materials import add_reference_environment
        add_reference_environment(spec)
    return spec
