"""Display native MuJoCo-deformed suit vertices in USD; no physics influence."""
from pathlib import Path
import mujoco
import numpy as np


class SuitDisplay:
    def __init__(self, stage, sensors):
        from pxr import UsdGeom,UsdShade,Sdf,Gf,Vt
        self.sensors=sensors;self.meshes=[];m=sensors.model
        self.scene=mujoco.MjvScene(m,maxgeom=256)
        self.option=mujoco.MjvOption();self.camera=mujoco.MjvCamera()
        self.refresh()
        textures={1:'Helmet_BaseColor.png',2:'Boot_BaseColor.png',3:'Suit_BaseColor.png'}
        materials={}
        for matid in range(m.nmat):
            material=UsdShade.Material.Define(stage,f'/World/Looks/Suit{matid}')
            shader=UsdShade.Shader.Define(stage,f'/World/Looks/Suit{matid}/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            color=m.mat_rgba[matid,:3];shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,color)))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.22 if matid==0 else .8)
            shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.75 if matid==0 else 0.)
            if matid in textures:
                texture=UsdShade.Shader.Define(stage,f'/World/Looks/Suit{matid}/Texture');texture.CreateIdAttr('UsdUVTexture')
                source=Path(__file__).resolve().parents[1]/'assets/wangyu_skin'/textures[matid]
                texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(source)))
                texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
                uv=UsdShade.Shader.Define(stage,f'/World/Looks/Suit{matid}/UV');uv.CreateIdAttr('UsdPrimvarReader_float2')
                uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
                texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
                texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
                shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(),'rgb')
            shader.CreateOutput('surface',Sdf.ValueTypeNames.Token);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            materials[matid]=material
        for i in range(m.nskin):
            mesh=UsdGeom.Mesh.Define(stage,f'/World/Suit/Skin{i}')
            a,n=int(m.skin_faceadr[i]),int(m.skin_facenum[i]);faces=m.skin_face[a:a+n]
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n,3,dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel().astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            a,n=int(m.skin_texcoordadr[i]),int(m.skin_vertnum[i])
            if a>=0:
                uv=m.skin_texcoord[a:a+n].copy()
                # USD textures use bottom-origin UV, whereas MuJoCo's native
                # texture loader uses the vflip=true source convention here.
                # Preserve the source Blender UV, with no second image flip.
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[int(m.skin_matid[i])])
            self.meshes.append(mesh)
        # Hide only source debug shapes. Rigid bodies and shoe collisions stay active.
        for geom in sensors.manifest['geoms']:
            body=sensors.manifest['bodies'][geom['body']-1]['name']
            UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/Robot/{body}/{geom["name"]}')).MakeInvisible()
        self.update()

    def update(self):
        from pxr import Vt
        m=self.sensors.model
        self.refresh()
        for i,mesh in enumerate(self.meshes):
            a,n=int(m.skin_vertadr[i]),int(m.skin_vertnum[i])
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.scene.skinvert.reshape(-1,3)[a:a+n].copy()))
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(self.scene.skinnormal.reshape(-1,3)[a:a+n].copy()));mesh.SetNormalsInterpolation('vertex')

    def refresh(self):
        # 3.12 deprecates mjv_updateSkin but does not expose its replacement in
        # Python. The supported native scene update deforms active skins too.
        # This allocates no renderer and advances no physics.
        mujoco.mjv_updateScene(self.sensors.model,self.sensors.data,self.option,None,
                              self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
