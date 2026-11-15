"""Read real ImGui draw lists without a graphics context or native window."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from native_ui_bridge import reader
from mujoco.studio.dear_imgui import dear_imgui as imgui


def test_reader_matches_pinned_imgui_context_and_large_vertex_offsets():
    api=reader();context=imgui.CreateContext()
    try:
        io=imgui.GetIO();io.DisplaySize=imgui.Vec2(900,600);io.BackendFlags|=24
        assert api.probe(context)[:3]==(900.,600.,1)
        imgui.NewFrame();imgui.SetNextWindowSize(imgui.Vec2(500,500));imgui.Begin('UI vertex-offset regression');imgui.End();imgui.Render()
        for pointer,texid,w,h,bpp,pixels in api.textures(context):api.ack_texture(pointer,texid or 1)
        imgui.NewFrame();imgui.Begin('UI vertex-offset regression')
        draw=imgui.GetWindowDrawList()
        for _ in range(18000):draw.AddRectFilled(imgui.Vec2(30,30),imgui.Vec2(31,31),0xffffffff)
        imgui.End();imgui.Render()
        width,height,sx,sy,lists=api.draw(context)
        assert (width,height)==(900.,600.)
        found=False
        for nv,ni,vertices,indices,commands in lists:
            assert len(vertices)==nv*20 and len(indices)==ni*2
            corrected=np.frombuffer(indices,np.uint16).astype(np.uint32)
            for offset,count,base,texture,clip in commands:
                corrected[offset:offset+count]+=base
                if base:found=True
            if ni:assert corrected.max()<nv
            if nv>65535:assert corrected.max()>65535
        assert found
    finally:imgui.DestroyContext(context)
