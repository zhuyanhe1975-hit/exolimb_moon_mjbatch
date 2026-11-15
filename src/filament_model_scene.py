"""Studio-compatible scene using MuJoCo 4.0.1 public mjrf resources.

The installed Studio C++ ModelScene and public mjrf path produce different
results in fixed-pose audits. Keep the model/resources alive once, use the
public ModelScene, and draw native mjv diagnostic primitives without the
unbound raw-struct ModelDecorations.update signature in this build.
"""
import numpy as np
import mujoco
from mujoco import mjrf
# Loading Studio registers the packaged filament: shader resource provider.
# No window/Kit/simulation is created by this import.
from mujoco import studio as _studio_resources
from mujoco.rendering.filament.model_scene import ModelScene


class FilamentModelScene:
    def __init__(self,context,scene,model):
        self._overlays=[];self._shown=0
        self._context=context;self._scene=scene;self._model=model
        import time
        started=time.monotonic()
        print(f'[Startup] Uploading Studio render resources: {model.nskin} skins, {model.ntex} shared textures...',flush=True)
        self._content=ModelScene(context,model,scene)
        print(f'[Startup] Studio render resources ready in {time.monotonic()-started:.2f}s',flush=True)
        from selection_highlight import SelectionHighlight
        self._selection=SelectionHighlight(model,self._content)
        self._diagnostics=mujoco.MjvScene(model,maxgeom=model.ngeom+2000)
        self._overlays=[]
        self._overlay_types=[]
        self._shown=0

    def get_headlight_intensity(self):
        if np.any(self._model.light_intensity>0):return 0.
        try:return float(self._model.numeric('filament.fallback.head_light_intensity').data[0])
        except KeyError:return 40000.

    def sync(self,width,height,model,data,perturb,camera,vis_options,extra_geoms):
        self._content._renderables.set_options(vis_options)
        self._content.update(data)
        self._selection.update(perturb.select)
        # Keep picking/camera/diagnostic geometry derived by native mjv.
        mujoco.mjv_updateScene(model,data,vis_options,perturb,camera,
                              mujoco.mjtCatBit.mjCAT_ALL,self._diagnostics)
        primitives=[g for g in self._diagnostics.geoms[:self._diagnostics.ngeom]
                    if g.category==mujoco.mjtCatBit.mjCAT_DECOR]
        primitives.extend(extra_geoms)
        supported={mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE,
                   mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_BOX,
                   mujoco.mjtGeom.mjGEOM_ELLIPSOID,mujoco.mjtGeom.mjGEOM_ARROW,
                   mujoco.mjtGeom.mjGEOM_ARROW1,mujoco.mjtGeom.mjGEOM_ARROW2,
                   mujoco.mjtGeom.mjGEOM_LINE,mujoco.mjtGeom.mjGEOM_LINEBOX}
        primitives=[g for g in primitives if g.type in supported]
        for i,geom in enumerate(primitives):
            if i==len(self._overlays):
                self._overlays.append(self._context.create_renderable(mjrf.RenderableParams()))
                self._overlay_types.append(None)
            renderable=self._overlays[i]
            if self._overlay_types[i]!=geom.type:
                renderable.set_geom_mesh(mujoco.mjtGeom(geom.type),8,16,4)
                self._overlay_types[i]=geom.type
            renderable.set_size(geom.size)
            renderable.set_pose(geom.pos,geom.mat.ravel())
            material=mjrf.Material();material.decor=True;material.color=geom.rgba
            renderable.set_material(material)
            if i>=self._shown:self._scene.add_renderable(renderable)
        for i in range(len(primitives),self._shown):self._scene.remove_renderable(self._overlays[i])
        self._shown=len(primitives)

    def __del__(self):
        for renderable in self._overlays[:self._shown]:self._scene.remove_renderable(renderable)
