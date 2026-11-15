"""Project-local Studio skin picking using native, freshly skinned vertices."""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass(frozen=True)
class PickResult:
    body: int = -1
    geom: int = -1
    flex: int = -1
    skin: int = -1
    dist: float = -1.
    point: object = None


def screen_ray(scene, x, y, aspect):
    """Match native mjv_select; x/y are normalized with a top-left origin."""
    camera = mujoco.mjv_averageCamera(scene.camera[0],scene.camera[1])
    position, forward, up = (np.zeros(3) for _ in range(3))
    mujoco.mjv_cameraInModel(position,forward,up,scene)
    left = np.cross(up,forward)
    left /= np.linalg.norm(left)
    width = camera.frustum_width or .5*aspect*(camera.frustum_top-camera.frustum_bottom)
    vertical = camera.frustum_bottom+(1-y)*(camera.frustum_top-camera.frustum_bottom)
    horizontal = -(camera.frustum_center+(2*x-1)*width)
    if camera.orthographic:
        position += up*vertical+left*horizontal
        direction = forward.copy()
    else:
        direction = forward*camera.frustum_near+up*vertical+left*horizontal
    direction /= np.linalg.norm(direction)
    return position,direction


class NativeSkinPicker:
    def __init__(self):
        self.model = None
        self.scene = None

    def pick(self, model, data, camera, x, y, aspect, options):
        if not (0<=x<=1 and 0<=y<=1 and aspect>0):
            return PickResult(point=np.zeros(3))
        if model is not self.model:
            self.model = model
            self.scene = mujoco.MjvScene(model,maxgeom=model.ngeom+2000)
        # Never step physics or let a pick advance tracking-camera smoothing.
        import copy
        pick_camera = copy.copy(camera)
        if pick_camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
            pick_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_updateScene(model,data,options,None,pick_camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        point=np.zeros(3)
        geom=np.array([-1],dtype=np.int32);flex=geom.copy();skin=geom.copy()
        body = mujoco.mjv_select(model,data,options,aspect,x,1-y,self.scene,point,geom,flex,skin)
        origin,direction = screen_ray(self.scene,x,y,aspect)
        distance = float(np.linalg.norm(point-origin)) if body>=0 else -1.
        if skin[0]>=0:
            i=int(skin[0]);fa=int(model.skin_faceadr[i]);nf=int(model.skin_facenum[i])
            va=int(model.skin_vertadr[i]);nv=int(model.skin_vertnum[i])
            faces=np.ascontiguousarray(model.skin_face[fa:fa+nf],dtype=np.int32)
            vertices=np.ascontiguousarray(self.scene.skinvert.reshape(-1,3)[va:va+nv],dtype=np.float32)
            # mjv_select returns a nearest vertex as the skin selection point.
            # Anchor perturbations to the actual surface/ray intersection instead.
            distance=mujoco.mju_raySkin(nf,nv,faces,vertices,origin,direction,None)
            if distance>=0:point=origin+distance*direction
        return PickResult(int(body),int(geom[0]),int(flex[0]),int(skin[0]),distance,point)


def install():
    """Replace only this process's Python Studio Pick call, preserving geom mode."""
    from mujoco.studio import ux
    if getattr(ux.Pick,'_moon_skin_picker',False):return
    original=ux.Pick
    picker=NativeSkinPicker()
    def pick(model,data,camera,x,y,aspect,options):
        if model.nskin and options.flags[mujoco.mjtVisFlag.mjVIS_SKIN]:
            return picker.pick(model,data,camera,x,y,aspect,options)
        return original(model,data,camera,x,y,aspect,options)
    pick._moon_skin_picker=True
    ux.Pick=pick
