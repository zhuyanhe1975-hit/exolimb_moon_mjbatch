"""CPU checks for HDR cube axes, energy and KTX dynamic range."""
from pathlib import Path
import struct
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_reflection_environment import build,directions,prefilter,read_hdr,irradiance_sh


def test_diffuse_irradiance_energy_and_ground_hemisphere_direction():
    d=directions(32)
    constant=np.ones_like(d)*[.1,.2,.3]
    for cube,expected in ((constant,[[.1,.2,.3]]*3),
                          (constant*(d[...,1:2]<0),[[0,0,0],[.1,.2,.3],[.05,.1,.15]])):
        sh=irradiance_sh(cube)
        n=np.array([[0,1,0],[0,-1,0],[1,0,0]])
        x,y,z=n.T
        poly=np.stack([np.ones(3),y,z,x,x*y,y*z,3*z*z-1,z*x,x*x-y*y],axis=-1)
        np.testing.assert_allclose(poly@sh,expected,atol=1e-4)


def test_cube_axes_and_constant_radiance_survive_ggx_filter():
    np.testing.assert_allclose(directions(1).reshape(6,3),
                               [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    image=np.broadcast_to([8.,4.,2.],(4,8,3))
    filtered=prefilter(image,directions(4),.7)
    np.testing.assert_allclose(filtered,np.broadcast_to([8,4,2],filtered.shape),atol=1e-6)


def test_hdr_to_ktx_keeps_values_above_one_and_all_cube_faces(tmp_path):
    source=tmp_path/'constant.hdr'
    with source.open('wb') as f:
        f.write(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 4 +X 8\n')
        for _ in range(4):
            f.write(bytes([2,2,0,8]))
            for value in [128,64,32,132]:f.write(bytes([136,value]))
    np.testing.assert_allclose(read_hdr(source)[0,0],[8,4,2])
    target=tmp_path/'constant.ktx';build(source,target,size=8)
    raw=target.read_bytes();header=struct.unpack('<13I',raw[12:64])
    assert header[4]==0x881B and header[10:12]==(6,4)  # RGB16F, cube, full mips
    assert header[12]>0  # SH is mandatory for diffuse cloth illumination
    assert raw[68:71]==b'sh\0'
    start=64+header[12]
    face_size=struct.unpack_from('<I',raw,start)[0]
    assert face_size==8*8*3*2
    for i in range(6):
        values=np.frombuffer(raw,dtype='<f2',count=8*8*3,offset=start+4+i*face_size).reshape(-1,3)
        np.testing.assert_allclose(values,np.broadcast_to([8,4,2],values.shape))
