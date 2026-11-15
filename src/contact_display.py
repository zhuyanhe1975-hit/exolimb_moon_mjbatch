"""Classic viewer overlays from authoritative simulation contacts and forces."""
import mujoco
import numpy as np


def draw_contacts(model,data,scene,*,points=True,forces=True):
    """Draw native contact positions and world-space solver forces (0.003 m/N)."""
    scene.ngeom=0
    force=np.zeros(6)
    for i in range(data.ncon):
        contact=data.contact[i]
        if points and scene.ngeom<scene.maxgeom:
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([.012]*3),contact.pos,np.eye(3).ravel(),np.array([1.,.2,.2,1.]))
            scene.ngeom+=1
        if forces and contact.efc_address>=0 and scene.ngeom<scene.maxgeom:
            mujoco.mj_contactForce(model,data,i,force)
            world_force=contact.frame.reshape(3,3).T@force[:3]
            if np.linalg.norm(world_force)<1e-6:continue
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,.9,1.,1.]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.008,
                contact.pos,contact.pos+.003*world_force)
            scene.ngeom+=1


class ContactDisplay:
    def __init__(self,simulator):
        self.sim=simulator
        self.model=simulator.mj_model
        self.data=mujoco.MjData(self.model)

    def read(self,world):
        if str(self.sim.impl).lower().endswith('cpp'):
            return self.sim.data.obj
        import mujoco_warp
        mujoco_warp.get_data_into(self.data,self.model,self.sim.data.obj,world_id=world)
        return self.data

    def update(self,window,world):
        points=bool(window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])
        forces=bool(window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])
        if not (points or forces):
            window.user_scn.ngeom=0
            return
        source=self.read(world)
        draw_contacts(self.model,source,window.user_scn,points=points,forces=forces)
