"""Project-local native presentation adapter for pinned MuJoCo4 Studio dev.

NativeViewer setup follows the installed Apache-2.0 MuJoCo Studio implementation
(Copyright 2026 DeepMind Technologies Limited). Shared package files are untouched.
The C++ mjrf API already supports a null target for direct window presentation;
the installed Python Renderer only implements explicit offscreen readback targets.
"""
def install(*, ui_bridge="local"):
    from studio_shortcuts import install as install_shortcuts
    install_shortcuts()
    from mujoco import mjrf
    from mujoco.rendering.filament import renderer
    from mujoco.studio import native_viewer as native

    class WindowRenderer(renderer.Renderer):
        def __init__(self,context,window):
            super().__init__(context)
            self._native_window=window
            self._audit_frames=0
        def render(self,views):
            if any(view.target for view in views):
                return super().render(views)
            width=int(self._native_window.GetWidth()*self._native_window.GetScale())
            height=int(self._native_window.GetHeight()*self._native_window.GetScale())
            requests=[]
            for view in views:
                request=mjrf.RenderRequest()
                request.scene=self._scenes[view.scene]
                # Default target=None selects the native window swapchain.
                request.camera=self._to_mjrf_camera(view.camera)
                request.viewport.left=int(view.viewport[0]*width)
                request.viewport.bottom=int(view.viewport[1]*height)
                request.viewport.width=int(view.viewport[2]*width)
                request.viewport.height=int(view.viewport[3]*height)
                request.draw_mode=view.draw_mode
                for field in ('post_processing','reflections','shadows','skybox','haze',
                              'headlight','headlight_color','headlight_intensity'):
                    setattr(request,field,getattr(view,field))
                requests.append(request)
            frame=self._ctx.render(requests,[])
            self._ctx.wait_for_frame(frame)
            self._audit_frames+=1
            import os
            if self._audit_frames in (30,600) and os.environ.get("STUDIO_RENDER_AUDIT")=="1":
                import dataclasses,json
                from pathlib import Path
                import numpy as np
                from PIL import Image
                out=Path(__file__).resolve().parents[1]/'outputs/playground_play/studio_audit'
                out.mkdir(parents=True,exist_ok=True)
                pixels=np.empty((height,width,3),dtype=np.uint8)
                self.target('audit',pixels)
                renderer.Renderer.render(self,[dataclasses.replace(view,target='audit') for view in views])
                Image.fromarray(pixels).save(out/('scene.png' if self._audit_frames==30 else 'scene_after_resets.png'))
                (out/'requests.json').write_text(json.dumps([{'scene':v.scene,'viewport':list(v.viewport),'headlight':bool(v.headlight),'headlight_intensity':float(v.headlight_intensity),'camera':str(v.camera)} for v in views],indent=2))
                print('[Studio audit] saved same-camera scene readback',flush=True)

    class StudioNativeViewer(native.NativeViewer):
        def prepare_next_frame(self):
            import os
            limit=int(os.environ.get('STUDIO_AUDIT_MAX_FRAMES','0'))
            if limit and self._native_frames>=limit:
                self.dispatch(native.messages.ExitEvent())
                return False
            return super().prepare_next_frame()
        def __init__(self,config,endpoint,**kwargs):
            native.viewer_protocol.Viewer.__init__(self,config,endpoint,**kwargs)
            self._window=native.window.Window(config.title,config.width,config.height,config.gfx or 'opengl')
            self._model_scene=None
            self._imgui_bridge=None
            self._renderer=None
            self._native_frames=0
            from pathlib import Path
            Path(__file__).resolve().parents[1].joinpath("outputs/playground_play/studio_audit").mkdir(parents=True,exist_ok=True)
            try:
                context=mjrf.Context(mjrf.ContextConfig(
                    graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL,
                    native_window=self._window.GetNativeWindowHandle()))
                self._renderer=WindowRenderer(context,self._window)
                self._renderer_model_id=id(None)
                native.imgui.SetCurrentContext(native.window.GetImGuiContext())
                native.ux.set_imgui_context(native.window.GetImGuiContext())
                native.ux.set_implot_context(native.window.GetImPlotContext())
                native.implot.set_imgui_context(native.window.GetImGuiContext())
                native.implot.set_implot_context(native.window.GetImPlotContext())
                if ui_bridge == "native":
                    self._imgui_bridge=native._viewer.ImguiBridge(context,self._renderer.scene('ux'))
                else:
                    from native_ui_bridge import NativeUiBridge
                    self._imgui_bridge=NativeUiBridge(context,self._renderer.scene('ux'),native.window.GetImGuiContext())
                self.dispatch(native.viewer_protocol.ViewerInitEvent(viewer=self))
            except BaseException:
                self.close()
                raise
        def _sync_renderer(self,model):
            replaced=id(model)!=self._renderer_model_id
            if replaced:
                from filament_model_scene import FilamentModelScene
                self._model_scene=None
                self._model_scene=FilamentModelScene(self._renderer.context,self._renderer.scene('scn'),model)
                self._renderer_model_id=id(model)
            if replaced or getattr(self,'_moon_terrain_visibility_changed',False):
                from studio_terrain import refresh_terrain_visibility
                refresh_terrain_visibility(self)
                self._moon_terrain_visibility_changed=False
        def sync(self):
            # The native window can render before the sim's ModelEvent arrives.
            # Initialise the display on the actual model, not on frame number 0.
            if (self.model.nq>=3 and getattr(self,'_moon_display_model_id',None)!=id(self.model)
                and getattr(self,'_moon_camera_state_model_id',None)==id(self.model)):
                self._moon_display_model_id=id(self.model)
                self.vis_options.geomgroup[0]=1
                self.vis_options.geomgroup[5]=0
                self.vis_options.flags[native.mujoco.mjtVisFlag.mjVIS_STATIC]=1
                import os
                if os.environ.get('MOON_FILAMENT_SCENE')=='1':
                    self.vis_options.geomgroup[4]=0
                    self.vis_options.skingroup[1]=1
                    self.vis_options.flags[native.mujoco.mjtVisFlag.mjVIS_SKIN]=1
                    self.vis_options.flags[native.mujoco.mjtVisFlag.mjVIS_STATIC]=1
                self.camera.lookat[:]=self.data.qpos[:3]+[0.,0.,.25]
                self.camera.distance=5.
                self.camera.elevation=-25.
                from multiworld_viewer import world_count
                count=world_count(self.model)
                if count:
                    self.camera.lookat[:]=[0,0,1.]
                    from large_terrain import half_extent
                    self.camera.distance=150. if half_extent(self.model)>8 else 18.
                    self.camera.elevation=-50.
            super().sync()
            self._native_frames+=1
            import os
            if self.model.nq>=3 and self._native_frames in (1,120):
                import numpy as np
                terrain_ids=np.flatnonzero(self.model.geom_type==int(native.mujoco.mjtGeom.mjGEOM_HFIELD))
                active=[int(g) for g in terrain_ids if self.model.geom_group[g]==0]
                print('[Studio terrain] active='+str([
                    (self.model.geom(g).name,int(self.model.geom_dataid[g]),
                     self.data.geom_xpos[g].tolist()) for g in active])+
                    '; groups='+str(self.vis_options.geomgroup.tolist())+
                    '; static='+str(bool(self.vis_options.flags[native.mujoco.mjtVisFlag.mjVIS_STATIC])),flush=True)
            if self.model.nq>=3 and self._native_frames in (1,120) and os.environ.get("STUDIO_RENDER_AUDIT")=="1":
                import json
                from pathlib import Path
                from multiworld_viewer import world_count
                gid=self.model.geom('world_0/terrain' if world_count(self.model) else 'terrain').id
                record={'hfield_id':int(self.model.geom_dataid[gid]),'geom_pos':self.model.geom_pos[gid].tolist(),'geom_xpos':self.data.geom_xpos[gid].tolist(),'geom_rgba':self.model.geom_rgba[gid].tolist(),'geom_group':int(self.model.geom_group[gid]),'visible_groups':self.vis_options.geomgroup.tolist(),'camera_lookat':self.camera.lookat.tolist(),'camera_distance':self.camera.distance,'camera_elevation':self.camera.elevation,'camera_azimuth':self.camera.azimuth,'root':self.data.qpos[:3].tolist(),'nlight':self.model.nlight}
                record['active_terrains']=[{
                    'name':self.model.geom(g).name,'hfield_id':int(self.model.geom_dataid[g]),
                    'geom_group':int(self.model.geom_group[g]),
                    'geom_xpos':self.data.geom_xpos[g].tolist(),
                    'geom_rgba':self.model.geom_rgba[g].tolist(),
                    'body_id':int(self.model.geom_bodyid[g])} for g in active]
                record['static_visible']=bool(self.vis_options.flags[native.mujoco.mjtVisFlag.mjVIS_STATIC])
                record['frame']=self._native_frames
                Path(__file__).resolve().parents[1].joinpath(f'outputs/playground_play/studio_audit/model_frame_{self._native_frames}.json').write_text(json.dumps(record,indent=2))
            if self._native_frames in (1,120):
                print(f'[Studio native] rendered {self._native_frames} frames to desktop window; '
                      f'{self._window.GetWidth()}x{self._window.GetHeight()}',flush=True)
    native.NativeViewer=StudioNativeViewer
