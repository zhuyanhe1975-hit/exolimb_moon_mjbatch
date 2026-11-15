"""Display-only clean fabric calibration from immutable source PBR maps.

Preserves UVs, labels, seam color detail and non-fabric panels. No relighting
painted into albedo, no model/physics changes and no runtime texture edits.
"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from reference_materials import verify_suit_maps


def soften_normal(image,strength):
    n=image.astype(np.float32)/127.5-1
    weight=np.asarray(strength)[...,None]
    n*=weight
    n[...,2]+=1-np.asarray(strength)
    n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-6)
    return np.rint((n+1)*127.5).clip(0,255).astype(np.uint8)


def build():
    original=verify_suit_maps();out=ROOT/'assets/clean_cloth';out.mkdir(exist_ok=True)
    sources={};files={};settings={}
    def load(path):
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return np.array(Image.open(path).convert('RGB'))
    def save(name,data):
        path=out/name;Image.fromarray(data).save(path)
        files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for part in ('Suit','Boot'):
        color=load(ROOT/f'assets/wangyu_skin/{part}_BaseColor.png')
        rgb=color.astype(np.float32);level=rgb.mean(-1);spread=rgb.max(-1)-rgb.min(-1)
        white=(spread<18)&(level>185)
        red=(rgb[...,0]>150)&(rgb[...,0]>rgb[...,1]*1.5)&(rgb[...,0]>rgb[...,2]*1.5)
        fabric=white|red
        # Original white cloth is ~213 sRGB. Lift it by 22 code values while
        # retaining per-pixel seam variations and leaving colored labels alone.
        cleaned=color.copy();cleaned[white]=np.clip(rgb[white]+22,0,245).astype(np.uint8)
        save(part+'_BaseColor.png',cleaned)
        metallic=load(original/(part+'_Metallic.png'));metallic[fabric]=0
        save(part+'_Metallic.png',metallic)
        rough=load(original/(part+'_Roughness.png'));rough[fabric]=np.maximum(rough[fabric],209)
        save(part+'_Roughness.png',rough)
        strength=np.where(fabric,.28,.55).astype(np.float32)
        save(part+'_Normal.png',soften_normal(load(original/(part+'_Normal.png')),strength))
        settings[part]={'fabric_normal_strength':.28,'other_normal_strength':.55,
                        'white_albedo_lift_srgb':22,'fabric_metallic':0,'fabric_roughness_min':209/255}
    save('Helmet_Normal.png',soften_normal(load(original/'Helmet_Normal.png'),.45))
    (out/'manifest.json').write_text(json.dumps({'sources':sources,'files':files,'settings':settings,
        'helmet_shell_normal_strength':.45,'scope':'display fabric only; visor mirror unchanged'},indent=2)+'\n')
    print(out)


if __name__=='__main__':build()
