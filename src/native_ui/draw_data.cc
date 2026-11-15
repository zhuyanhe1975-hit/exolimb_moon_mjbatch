// Project-local read-only ImGui draw data bridge. ImGui 1.92.9b headers match
// the pinned MuJoCo 4.0.1 SDK; no shared binaries are changed or relinked.
#include <imgui.h>
#include <imgui_internal.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
namespace py=pybind11;
PYBIND11_MODULE(_moon_draw_data,m) {
 m.def("probe",[](uintptr_t p){auto*c=(ImGuiContext*)p;return py::make_tuple(c->IO.DisplaySize.x,c->IO.DisplaySize.y,c->Viewports.Size,IMGUI_VERSION);});
 m.def("textures",[](uintptr_t p){auto*c=(ImGuiContext*)p;py::list out;for(auto*t:c->PlatformIO.Textures){
  if(t->Status==ImTextureStatus_WantCreate||t->Status==ImTextureStatus_WantUpdates){
   out.append(py::make_tuple((uintptr_t)t,(uint64_t)t->TexID,t->Width,t->Height,t->BytesPerPixel,py::bytes((char*)t->GetPixels(),t->Width*t->Height*t->BytesPerPixel)));
  }}return out;});
 m.def("ack_texture",[](uintptr_t p,uint64_t id){auto*t=(ImTextureData*)p;t->SetTexID((ImTextureID)id);t->SetStatus(ImTextureStatus_OK);});
 m.def("draw",[](uintptr_t p){auto*c=(ImGuiContext*)p;auto&d=c->Viewports[0]->DrawDataP;py::list lists;for(auto*l:d.CmdLists){py::list commands;
  for(auto&cmd:l->CmdBuffer){if(cmd.UserCallback)continue;commands.append(py::make_tuple(cmd.IdxOffset,cmd.ElemCount,cmd.VtxOffset,(uint64_t)cmd.GetTexID(),py::make_tuple(cmd.ClipRect.x,cmd.ClipRect.y,cmd.ClipRect.z,cmd.ClipRect.w)));}
  lists.append(py::make_tuple(l->VtxBuffer.Size,l->IdxBuffer.Size,py::bytes((char*)l->VtxBuffer.Data,l->VtxBuffer.Size*sizeof(ImDrawVert)),py::bytes((char*)l->IdxBuffer.Data,l->IdxBuffer.Size*sizeof(ImDrawIdx)),commands));}
  return py::make_tuple(c->IO.DisplaySize.x,c->IO.DisplaySize.y,c->IO.DisplayFramebufferScale.x,c->IO.DisplayFramebufferScale.y,lists);});
}
