"""Convert a Radiance HDR panorama to a linear HDR KTX1 reflection cubemap.

Offline CPU asset conversion, no simulation. Mips use GGX importance sampling
instead of blurring the original panorama in display/sRGB space.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def read_hdr(path):
    with Path(path).open('rb') as f:
        if not f.readline().startswith(b'#?'):raise ValueError('Not Radiance HDR')
        while f.readline().strip():pass
        axes=f.readline().decode().split()
        if axes[0]!='-Y' or axes[2]!='+X':raise ValueError('Expected -Y +X panorama')
        height,width=int(axes[1]),int(axes[3])
        rgbe=np.empty((height,width,4),dtype=np.uint8)
        for y in range(height):
            header=f.read(4)
            if header[:2]!=b'\x02\x02' or int.from_bytes(header[2:],'big')!=width:
                raise ValueError('Expected scanline RLE')
            for channel in range(4):
                x=0
                while x<width:
                    count=f.read(1)[0]
                    if count==0:raise ValueError('Invalid RLE run')
                    if count>128:
                        count-=128;value=f.read(1)[0]
                        rgbe[y,x:x+count,channel]=value
                    else:rgbe[y,x:x+count,channel]=np.frombuffer(f.read(count),np.uint8)
                    x+=count
                if x!=width:raise ValueError('Invalid scanline length')
    return np.ldexp(rgbe[:,:,:3].astype(np.float32),rgbe[:,:,3:4].astype(np.int32)-136)


def sample(image,d):
    d=d/np.maximum(np.linalg.norm(d,axis=-1,keepdims=True),1e-12)
    x=(.5+np.arctan2(d[...,2],d[...,0])/(2*np.pi))*image.shape[1]-.5
    y=np.arccos(np.clip(d[...,1],-1,1))/np.pi*image.shape[0]-.5
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    fx=(x-x0)[...,None];fy=(y-y0)[...,None]
    def pixel(dx,dy):return image[np.clip(y0+dy,0,image.shape[0]-1),(x0+dx)%image.shape[1]]
    return (pixel(0,0)*(1-fx)+pixel(1,0)*fx)*(1-fy)+(pixel(0,1)*(1-fx)+pixel(1,1)*fx)*fy


def directions(size):
    u,v=np.meshgrid(2*(np.arange(size)+.5)/size-1,2*(np.arange(size)+.5)/size-1)
    one=np.ones_like(u)
    d=np.stack([(one,-v,-u),(-one,-v,u),(u,one,v),(u,-one,-v),(u,-v,one),(-u,-v,-one)],axis=-1)
    # Above array is (component, height, width, face).
    d=d.transpose(3,1,2,0)
    return d/np.linalg.norm(d,axis=-1,keepdims=True)


def irradiance_sh(cube):
    """Filament's 3-band irradiance polynomial, including cosine and 1/pi.

    Follows IndirectLight::Builder::radiance / CubemapSH::preprocessSHForShader.
    A constant radiance L must reconstruct L, not pi*L or zero.
    """
    side=cube.shape[1];d=directions(side);x,y,z=np.moveaxis(d,-1,0)
    u,v=np.meshgrid(2*(np.arange(side)+.5)/side-1,2*(np.arange(side)+.5)/side-1)
    weight=np.broadcast_to((1+u*u+v*v)**-1.5,cube.shape[:-1]).copy()
    weight*=4*np.pi/weight.sum()
    poly=np.stack([np.ones_like(x),y,z,x,x*y,y*z,3*z*z-1,z*x,x*x-y*y],axis=-1)
    a=np.array([.2820947918,-.4886025119,.4886025119,-.4886025119,
                1.0925484306,-1.0925484306,.3153915653,-1.0925484306,.5462742153])
    convolution=np.array([1,2/3,2/3,2/3,1/4,1/4,1/4,1/4,1/4])
    integrated=np.einsum('...i,...c,...->ic',poly,cube,weight,optimize=True)
    return integrated*(a*a*convolution)[:,None]


def write_ktx(target,mips,sh):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    side=mips[0].shape[1]
    # This is the exact SH metadata consumed by MuJoCo's Ktx1Reader. Without
    # it ModelLights configures reflections but no diffuse irradiance.
    value=b'sh\0'+(''.join(' '.join(f'{v:.9g}' for v in row)+'\n' for row in sh)).encode()
    metadata=struct.pack('<I',len(value))+value+b'\0'*((-len(value))%4)
    with target.open('wb') as f:
        f.write(b'\xabKTX 11\xbb\r\n\x1a\n')
        f.write(struct.pack('<13I',0x04030201,0x140B,2,0x1907,0x881B,0x1907,side,side,0,0,6,len(mips),len(metadata)))
        f.write(metadata)
        for data in mips:
            face_bytes=data.shape[1]*data.shape[2]*3*2
            f.write(struct.pack('<I',face_bytes))
            for face in data:
                raw=np.clip(face,0,65504).astype('<f2').tobytes()
                f.write(raw);f.write(b'\0'*((-len(raw))%4))


def prefilter(image,n,roughness,count=128):
    if roughness==0:return sample(image,n)
    up=np.zeros_like(n);up[...,1]=1
    up[np.abs(n[...,1])>.99]=[1,0,0]
    tangent=np.cross(up,n);tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    bitangent=np.cross(n,tangent)
    result=np.zeros_like(n);weight=np.zeros(n.shape[:-1])
    a=roughness**2
    for i in range(count):
        # Hammersley: stratification + base-2 radical inverse.
        bits=i;inverse=0.;factor=.5
        while bits:
            inverse+=(bits&1)*factor;factor*=.5;bits>>=1
        phi=2*np.pi*i/count
        cosine=np.sqrt((1-inverse)/(1+(a*a-1)*inverse))
        sine=np.sqrt(max(0,1-cosine*cosine))
        h=tangent*(np.cos(phi)*sine)+bitangent*(np.sin(phi)*sine)+n*cosine
        l=2*np.sum(n*h,axis=-1,keepdims=True)*h-n
        w=np.maximum(np.sum(n*l,axis=-1),0)
        result+=sample(image,l)*w[...,None];weight+=w
    return result/np.maximum(weight[...,None],1e-12)


def build(source,target,size=256,source_url=''):
    image=read_hdr(source)
    levels=int(np.log2(size))+1
    if size&(size-1):raise ValueError('Face size must be a power of two')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    mips=[]
    for level in range(levels):
        side=max(1,size>>level)
        lod=level/(levels-1)
        roughness=(1.7-np.sqrt(2.89-2.8*lod))/1.4
        mips.append(prefilter(image,directions(side),roughness if level else 0))
    sh=irradiance_sh(mips[0]);write_ktx(target,mips,sh)
    target.with_suffix('.json').write_text(json.dumps({
        'source':str(source),'source_url':source_url,
        'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'format':'KTX1 RGB16F linear','face_size':size,'mip_levels':levels,
        'prefilter':'GGX importance sampling, 128 samples per texel',
        'irradiance_sh':sh.tolist(),
        'purpose':'reference HDRI reflection, not the lunar scene'},indent=2)+'\n')
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('target',type=Path)
    p.add_argument('--source-url',default='')
    a=p.parse_args();build(a.source,a.target,source_url=a.source_url)
