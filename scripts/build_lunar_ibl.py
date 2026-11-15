"""Build a shared lunar reflection approximation; no physics or rendering.

Black sky and a normal-mapped Lambertian plane viewed from 1.7m. Not a capture of
the active heightfield or surrounding actors. Source albedo is SHA-verified.
Cubemap is Y-up (Filament applies MuJoCo's +90deg X rotation when lighting).
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from reference_materials import verify_maps, ASSET
from prepare_reflection_environment import irradiance_sh,write_ktx


def build():
    verify_maps()
    source=ASSET/'diffuse.png'
    normal_source=ASSET/'normal.png'
    albedo=np.asarray(Image.open(source).convert('RGB'),dtype=np.float64)/255
    albedo=np.where(albedo<=.04045,albedo/12.92,((albedo+.055)/1.055)**2.4)
    # Use the same converted tangent normal map as the visible ground. Albedo
    # alone contains little relief contrast and reads as a flat gold hemisphere.
    normal=np.asarray(Image.open(normal_source).convert('RGB'),dtype=np.float32)/127.5-1
    normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-6)
    size=512
    u,v=np.meshgrid(2*(np.arange(size)+.5)/size-1,2*(np.arange(size)+.5)/size-1)
    one=np.ones_like(u)
    directions=((one,-v,-u),(-one,-v,u),(u,one,v),(u,-one,-v),(u,-v,one),(-u,-v,-one))
    faces=[]
    linear_faces=[]
    for x,y,z in directions:
        ground=y<0
        distance=1.7/np.maximum(-y,1e-6)
        # 4m texture repeat, matching the scene material. Direction length
        # cancels in ray/plane intersection. No pseudo atmosphere or sun disk:
        # the real directional scene light supplies the direct sun highlight.
        px=np.floor(np.remainder(x*distance/4,1)*albedo.shape[1]).astype(int)
        py=np.floor(np.remainder(z*distance/4,1)*albedo.shape[0]).astype(int)
        n=normal[py,px]
        cosine=np.maximum((n[...,0]+n[...,2])*2**-.5,0)
        radiance=albedo[py,px]*cosine[...,None]/np.pi
        radiance[~ground]=0
        linear_faces.append(radiance)
        srgb=np.where(radiance<=.0031308,12.92*radiance,1.055*radiance**(1/2.4)-.055)
        faces.append(np.rint(np.clip(srgb,0,1)*255).astype(np.uint8))
    output=ROOT/'assets/lunar_ibl';output.mkdir(exist_ok=True)
    target=output/'lunar.png'
    Image.fromarray(np.concatenate(faces,axis=0)).save(target)
    cube=np.stack(linear_faces)
    sh=irradiance_sh(cube)
    mips=[cube]
    while cube.shape[1]>1:
        n=cube.shape[1]//2
        cube=cube.reshape(6,n,2,n,2,3).mean(axis=(2,4));mips.append(cube)
    write_ktx(output/'lunar.ktx',mips,sh)
    manifest={'source':str(source.relative_to(ROOT)),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'normal_source':str(normal_source.relative_to(ROOT)),
              'normal_sha256':hashlib.sha256(normal_source.read_bytes()).hexdigest(),
              'files':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('lunar.png','lunar.ktx')},
              'irradiance_sh':sh.tolist(),
              'layout':'RLUDFB','face_size':size,'colorspace':'sRGB',
              'model':'black sky / normal-mapped Lambertian plane, sun elevation 45 degrees',
              'view_height_m':1.7,'texture_repeat_m':4,
              'limitations':'shared static approximation; no active terrain or actor reflections'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(target)


if __name__=='__main__':build()
