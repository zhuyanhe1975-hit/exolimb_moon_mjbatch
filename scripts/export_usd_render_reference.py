"""Read the original USD material graph with pxr only; never starts Kit."""
from pathlib import Path
import hashlib,json
from pxr import Usd,UsdShade,UsdGeom
ROOT=Path(__file__).resolve().parents[1]
source=Path.home()/'myWorks/zbot_exoarm_moon/outputs/wangyu_new_preview/wangyu_rigged.usdc'
stage=Usd.Stage.Open(str(source));shaders={};bindings={}
def serial(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'path'):return value.path
    try:return list(value)
    except TypeError:return str(value)
for prim in stage.Traverse():
    if prim.IsA(UsdShade.Shader):
        shader=UsdShade.Shader(prim)
        shaders[str(prim.GetPath())]={'id':shader.GetIdAttr().Get(),'inputs':{
            i.GetBaseName():{'value':serial(i.Get()),'connections':[str(p) for p in i.GetAttr().GetConnections()]}
            for i in shader.GetInputs()}}
    if prim.IsA(UsdGeom.Mesh):
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bindings[str(prim.GetPath())]=str(material.GetPath())
out=ROOT/'assets/suit_pbr/usd_reference.json'
out.write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'shaders':shaders,'bindings':bindings},indent=2)+'\n')
print('Saved static USD graph:',out)
