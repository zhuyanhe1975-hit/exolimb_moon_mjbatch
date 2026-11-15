"""Separate visible visor coating from the shared Helmet atlas (CPU only).

Suit_Shield includes both visor and white frame; the occluded Glass skin is
not the visible front surface. Keep original UVs/skin/bones, use a coating
mask instead of making every part using Helmet_MAT metallic.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from reference_materials import verify_suit_maps,VISOR_GOLD_SRGB,VISOR_ROUGHNESS


def build():
    suit=verify_suit_maps()
    source={'color':ROOT/'assets/wangyu_skin/Helmet_BaseColor.png',
            'roughness':suit/'Helmet_Roughness.png','normal':suit/'Helmet_Normal.png'}
    color=np.array(Image.open(source['color']).convert('RGB'))
    rgb=color.astype(np.int16)
    # The source's painted gold islands identify the visor coating, while the
    # same Shield skin's white frame and attachments remain dielectric.
    mask=(rgb[:,:,0]-rgb[:,:,2]>45)&(rgb[:,:,1]-rgb[:,:,2]>25)&(rgb[:,:,0]>rgb[:,:,1])
    rough=np.array(Image.open(source['roughness']).convert('RGB'))
    normal=np.array(Image.open(source['normal']).convert('RGB'))
    visor_color=color.copy();visor_color[mask]=np.rint(np.array(VISOR_GOLD_SRGB[:3])*255)
    visor_rough=rough.copy();visor_rough[mask]=round(VISOR_ROUGHNESS*255)
    visor_normal=normal.copy();visor_normal[mask]=[128,128,255]
    shell_color=color.copy();shell_color[mask]=[243,246,248]
    shell_rough=rough.copy();shell_rough[mask]=round(.45*255)
    maps={'visor_color.png':visor_color,'visor_metallic.png':np.repeat((mask*255).astype(np.uint8)[...,None],3,axis=-1),
          'visor_roughness.png':visor_rough,'visor_normal.png':visor_normal,
          'shell_color.png':shell_color,'shell_roughness.png':shell_rough}
    out=ROOT/'assets/visor_pbr';out.mkdir(exist_ok=True)
    for name,data in maps.items():Image.fromarray(data).save(out/name)
    (out/'manifest.json').write_text(json.dumps({
        'sources':{key:{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for key,p in source.items()},
        'files':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in maps},
        'visible_skin':'skin_Suit_Shield_0','mask':'source gold islands: R-B>45, G-B>25, R>G',
        'gold_pixels':int(mask.sum()),'gold_srgb':VISOR_GOLD_SRGB,'roughness':VISOR_ROUGHNESS,
        'shell':'gold-painted atlas islands remapped to dielectric white for helmet shell only',
        'geometry_modified':False},indent=2)+'\n')
    print(out)


if __name__=='__main__':build()
