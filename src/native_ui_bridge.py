"""ImGui draw-data adapter using public MuJoCo 4.0.1 mjrf buffers."""
from pathlib import Path
import importlib,sys
import numpy as np
import mujoco
from mujoco import mjrf
from mujoco.studio.dear_imgui import dear_imgui as imgui


def reader():
    folder=Path(__file__).resolve().parents[1]/'outputs/native_ui'
    sys.path.insert(0,str(folder))
    try:return importlib.import_module('_moon_draw_data')
    except ImportError as error:
        raise RuntimeError('Build the local UI adapter with the Playground Python: scripts/build_native_ui.py') from error


class NativeUiBridge:
    def __init__(self,context,scene,im_context):
        self.ctx=context;self.scene=scene;self.im_context=int(im_context);self.api=reader()
        self.textures={};self._texture_data={};self.next_id=1;self.meshes=[];self.renderables=[];self.shown=0;self.buffers=[]
        probe=self.api.probe(self.im_context)
        io=imgui.GetIO()
        io.BackendFlags|=24 # public upload path supports textures and large vertex offsets
        if tuple(probe[:2])!=(io.DisplaySize.x,io.DisplaySize.y):raise RuntimeError('ImGui SDK/context ABI mismatch')

    def upload_image(self,tex_id,pixels,width,height,bpp):
        if not pixels:
            self.textures.pop(tex_id,None);self._texture_data.pop(tex_id,None);return 0
        if not tex_id:tex_id=self.next_id;self.next_id+=1
        raw=np.frombuffer(pixels,np.uint8).reshape(height,width,bpp)
        if bpp==1:raw=np.concatenate((np.full((height,width,3),255,np.uint8),raw),axis=-1)
        if bpp==3:raw=np.concatenate((raw,np.full((height,width,1),255,np.uint8)),axis=-1)
        config=mjrf.TextureConfig(width=width,height=height,sampler_type=mujoco.mjtTexture.mjTEXTURE_2D,
            format=mjrf.PixelFormat.PIXEL_FORMAT_RGBA8,color_space=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR)
        texture=self.ctx.create_texture(config);data=mjrf.TextureData();data.set_image(raw.tobytes());texture.upload(data)
        self.textures[tex_id]=texture;self._texture_data[tex_id]=data
        return tex_id

    def update(self):
        imgui.SetCurrentContext(self.im_context)
        imgui.Render();self.buffers=[]
        for pointer,tex_id,w,h,bpp,pixels in self.api.textures(self.im_context):
            tex_id=self.upload_image(tex_id,pixels,w,h,bpp);self.api.ack_texture(pointer,tex_id)
        width,height,sx,sy,lists=self.api.draw(self.im_context)
        used=0
        for list_idx,(nv,ni,vertices,indices,commands) in enumerate(lists):
            if not nv or not ni:continue
            corrected=np.frombuffer(indices,np.uint16).astype(np.uint32)
            for offset,count,base,texid,clip in commands:corrected[offset:offset+count]+=base
            while list_idx>=len(self.meshes):self.meshes.append(None)
            if self.meshes[list_idx] is None or nv>self.meshes[list_idx][1] or ni>self.meshes[list_idx][2]:
                attrs=[mjrf.VertexAttribute(usage=usage,type=kind) for usage,kind in (
                    (mjrf.VertexAttributeUsage.VERTEX_ATTRIBUTE_USAGE_POSITION,mjrf.VertexAttributeType.VERTEX_ATTRIBUTE_TYPE_FLOAT2),
                    (mjrf.VertexAttributeUsage.VERTEX_ATTRIBUTE_USAGE_UV,mjrf.VertexAttributeType.VERTEX_ATTRIBUTE_TYPE_FLOAT2),
                    (mjrf.VertexAttributeUsage.VERTEX_ATTRIBUTE_USAGE_COLOR,mjrf.VertexAttributeType.VERTEX_ATTRIBUTE_TYPE_UBYTE4))]
                cfg=mjrf.MeshConfig(max_vertices=2**max(1,(nv-1).bit_length()),max_indices=2**max(1,(ni-1).bit_length()),
                    attributes=attrs,interleaved=True,index_type=mjrf.IndexType.INDEX_TYPE_U32,
                    primitive_type=mjrf.MeshPrimitiveType.MESH_PRIMITIVE_TYPE_TRIANGLES)
                self.meshes[list_idx]=(self.ctx.create_mesh(cfg),cfg.max_vertices,cfg.max_indices,cfg)
            mesh,_,_,cfg=self.meshes[list_idx]
            data=mjrf.MeshData();data.set_vertices(cfg,vertices.ljust(cfg.max_vertices*20,b'\0'));data.set_indices(cfg,corrected.tobytes().ljust(cfg.max_indices*4,b'\0'));mesh.upload(data);self.ctx.flush()
            self.buffers.append(data)
            for offset,count,base,texid,clip in commands:
                if used==len(self.renderables):self.renderables.append(self.ctx.create_renderable(mjrf.RenderableParams(cast_shadow=False,receive_shadow=False,blend_order=used+1)))
                renderable=self.renderables[used]
                material=mjrf.Material();material.decor=True
                if texid in self.textures:material.color_texture=self.textures[texid]
                x0,y0,x1,y1=clip
                x0=np.floor(np.clip(x0*sx,0,width*sx));x1=np.ceil(np.clip(x1*sx,x0,width*sx))
                y0=np.floor(np.clip(y0*sy,0,height*sy));y1=np.ceil(np.clip(y1*sy,y0,height*sy))
                material.scissor=[x0,height*sy-y1,x1-x0,y1-y0]
                renderable.set_material(material);renderable.set_mesh(mesh,offset,count);renderable.set_size(np.array([sx,sy,1],np.float32))
                renderable.set_pose(np.zeros(3,np.float32),np.eye(3,dtype=np.float32).ravel())
                if used>=self.shown:self.scene.add_renderable(renderable)
                used+=1
        for i in range(used,self.shown):self.scene.remove_renderable(self.renderables[i])
        self.shown=used

    def __del__(self):
        for renderable in self.renderables[:self.shown]:self.scene.remove_renderable(renderable)
