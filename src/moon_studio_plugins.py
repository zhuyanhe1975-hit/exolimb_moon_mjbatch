"""Explicit simulation -> viewer model-field sync, per Studio Snapshot contract."""
from dataclasses import dataclass
import numpy as np
import mujoco
from mujoco.studio import messages,plugin

@dataclass
class TerrainSnapshot(messages.Snapshot):
    active_field:int
    active_fields:tuple=()
    revision:int=0

@dataclass
class TerrainRequest(messages.Event):
    """Request current terrain after loading/replacing the display model."""

@dataclass
class TerrainAck(messages.Event):
    revision:int

@dataclass
class ContactOptionsSnapshot(messages.Snapshot):
    enabled:bool

@dataclass
class ContactsSnapshot(messages.Snapshot):
    positions:np.ndarray
    forces:np.ndarray


class ContactSender(plugin.SimPlugin):
    def __init__(self,simulator,selection):
        super().__init__()
        from contact_display import ContactDisplay
        self.contacts=ContactDisplay(simulator);self.selection=selection;self.enabled=False

    @messages.handler
    def on_options(self,event:ContactOptionsSnapshot):
        self.enabled=event.enabled
        if self.enabled:self._send_contacts()

    @messages.handler
    def on_step(self,event:messages.PostStepEvent):
        if not self.enabled:return
        self._send_contacts()

    def _send_contacts(self):
        data=self.contacts.read(self.selection['world'])
        positions=np.array(data.contact.pos[:data.ncon],copy=True)
        if 'offsets' in self.selection:positions+=self.selection['offsets'][self.selection['world']]
        forces=np.zeros((data.ncon,3));force=np.zeros(6)
        for i in range(data.ncon):
            if data.contact[i].efc_address>=0:
                mujoco.mj_contactForce(self.contacts.model,data,i,force)
                forces[i]=data.contact[i].frame.reshape(3,3).T@force[:3]
        self.handle.send_to_viewer(ContactsSnapshot(positions,forces))


class ContactReceiver(plugin.ViewerPlugin):
    def __init__(self):
        super().__init__()
        self.snapshot=None;self._enabled=None;self._owned=set()

    @messages.handler
    def on_contacts(self,event:ContactsSnapshot):self.snapshot=event

    @messages.handler
    def on_update(self,event:messages.UpdateEvent):
        viewer=self.viewer
        points=bool(viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])
        forces=bool(viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])
        enabled=points or forces
        if enabled!=self._enabled:
            viewer.send_to_sim(ContactOptionsSnapshot(enabled));self._enabled=enabled
        viewer.extra_geoms[:]=[g for g in viewer.extra_geoms if id(g) not in self._owned]
        self._owned=set()
        if not enabled or self.snapshot is None:return
        for pos,force in zip(self.snapshot.positions,self.snapshot.forces):
            if points:
                geom=mujoco.MjvGeom()
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012]*3),pos,np.eye(3).ravel(),np.array([1.,.2,.2,1.]))
                viewer.extra_geoms.append(geom);self._owned.add(id(geom))
            if forces and np.linalg.norm(force)>1e-6:
                geom=mujoco.MjvGeom()
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,.9,1.,1.]))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.008,pos,pos+.003*force)
                viewer.extra_geoms.append(geom);self._owned.add(id(geom))

class TerrainSender(plugin.SimPlugin):
    def __init__(self):
        super().__init__()
        self._last_field=None
        self._model=None
        self._revision=0
        self._acked=False
        self._requested=False

    @messages.handler
    def on_request(self,event:TerrainRequest):
        self._requested=True

    @messages.handler
    def on_ack(self,event:TerrainAck):
        if event.revision==self._revision:self._acked=True

    @messages.handler
    def on_post_step(self,event:messages.PostStepEvent):
        model=event.model
        from multiworld_viewer import world_count
        count=world_count(model)
        if count:
            fields=[]
            for world in range(count):
                prefix=f'world_{world}/'
                if world:
                    fields.append(fields[0])
                else:
                    fields.append(next(i for i in range(model.nhfield) if model.geom_group[model.geom(prefix+('terrain' if i==0 else f'display_terrain_{i:02d}')).id]==0))
            snapshot=TerrainSnapshot(fields[0],tuple(fields))
        else:
            selected=np.flatnonzero((model.geom_type==int(mujoco.mjtGeom.mjGEOM_HFIELD))&(model.geom_group==0))
            if len(selected)!=1:raise RuntimeError('Expected exactly one visible display terrain')
            snapshot=TerrainSnapshot(active_field=int(model.geom_dataid[selected[0]]))
        key=(snapshot.active_field,snapshot.active_fields)
        if key!=self._last_field or model is not self._model or self._requested:
            self._last_field=key;self._model=model;self._requested=False
            self._revision+=1;self._acked=False
        if not self._acked:
            # Snapshot channels are latest-wins and Studio Reset can drain them.
            # Keep publishing until the viewer has applied this revision.
            snapshot.revision=self._revision
            self.handle.send_to_viewer(snapshot)

class TerrainReceiver(plugin.ViewerPlugin):
    def __init__(self):
        super().__init__()
        self._terrain_snapshot=None
        self._applied=None

    @messages.handler
    def on_state(self,event:messages.StateSnapshot):
        model=self.viewer.model
        if model.nq>=3 and len(event.state)==mujoco.mj_stateSize(model,event.state_sig):
            self.viewer._moon_camera_state_model_id=id(model)

    @messages.handler
    def on_model(self,event:messages.PostModelEvent):
        # Model replacement resets groups; the active field may not change and
        # therefore the sender need not publish another terrain snapshot.
        self._applied=None
        if self._terrain_snapshot is not None:self.on_terrain(self._terrain_snapshot)
        if getattr(self.viewer,'send_to_sim',None):self.viewer.send_to_sim(TerrainRequest())

    @messages.handler
    def on_terrain(self,event:TerrainSnapshot):
        self._terrain_snapshot=event
        model=self.viewer.model
        fields=event.active_fields or (event.active_field,)
        key=(id(model),event.active_field,event.active_fields,event.revision)
        # Events and snapshots have separate channels: terrain can arrive while
        # the model event is still queued. Cache it until PostModelEvent.
        from multiworld_viewer import world_count
        if bool(event.active_fields)!=bool(world_count(model)):return
        if event.active_fields and len(fields)!=world_count(model):return
        try:
            for world in range(1):
                prefix=f'world_{world}/' if event.active_fields else ''
                model.geom(prefix+'terrain')
                if model.nhfield>1:model.geom(prefix+f'display_terrain_{model.nhfield-1:02d}')
        except KeyError:return
        if self._applied==key:
            if getattr(self.viewer,'send_to_sim',None):self.viewer.send_to_sim(TerrainAck(event.revision))
            return
        if event.active_fields and any(active!=event.active_field for active in fields):
            raise ValueError('Shared display received different terrain fields')
        for world,active in enumerate(fields[:1]):
            prefix=f'world_{world}/' if event.active_fields else ''
            for i in range(model.nhfield):
                gid=model.geom(prefix+('terrain' if i==0 else f'display_terrain_{i:02d}')).id
                model.geom_group[gid]=0 if world==0 and i==active else 5
                if event.active_fields:model.geom_rgba[gid,3]=float(world==0)
        # Terrain lives on world and requires the static-geometry flag too.
        self.viewer.vis_options.flags[mujoco.mjtVisFlag.mjVIS_STATIC]=1
        self.viewer.vis_options.geomgroup[0]=1
        self.viewer.vis_options.geomgroup[5]=0
        self.viewer._moon_terrain_visibility_changed=True
        self._applied=key
        if getattr(self.viewer,'send_to_sim',None):self.viewer.send_to_sim(TerrainAck(event.revision))
