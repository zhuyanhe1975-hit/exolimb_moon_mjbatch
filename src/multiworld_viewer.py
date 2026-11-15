"""Display independent people on one shared terrain; never step the CPU copy."""
import mujoco
import numpy as np
import os,time


def offsets(count):
    return np.zeros((count,3))


def world_count(model):
    try:return int(model.numeric('display_world_count').data[0])
    except KeyError:return 0


def share_actor_assets(spec):
    """Canonicalise actor-instance references before compiling/uploading assets."""
    spec.dedup_assets()
    def canonical(name):
        if name.startswith('world_') and '/' in name:
            return 'world_0/'+name.split('/',1)[1]
        return name
    for collection in ('geoms','sites','skins','tendons'):
        for element in getattr(spec,collection):
            for attr in ('meshname','material'):
                if hasattr(element,attr):
                    name=getattr(element,attr)
                    if name:setattr(element,attr,canonical(name))
    for material in spec.materials:
        for role,name in enumerate(list(material.textures)):
            if name:material.textures[role]=canonical(name)
    for collection in ('meshes','materials','textures'):
        assets=list(getattr(spec,collection))
        names={asset.name for asset in assets}
        for asset in assets:
            target=canonical(asset.name)
            if target!=asset.name:
                if target not in names:raise ValueError(f'Missing shared asset: {target}')
                spec.delete(asset)


def build_grid(simulator):
    from moon_viewer import display_spec
    spec=mujoco.MjSpec()
    spec.add_numeric(name='display_world_count',data=[simulator.nworld])
    spec.add_numeric(name='display_shared_terrain',data=[1])
    # Build the bank once. Other attachments contain actors only, with no
    # hfield assets or hidden terrain geoms to upload into Filament.
    profile=os.environ.get('MOON_STARTUP_PROFILE')=='1'
    started=time.monotonic();stages={'copy':0.,'attach':0.}
    shared=display_spec(simulator,single=True)
    if profile:print('[Display profile] shared spec',time.monotonic()-started,flush=True)
    actors=shared.copy()
    # Renderer options are scene-wide names, not prefixed actor metadata.
    for numeric in list(shared.numerics):
        if numeric.name.startswith('filament.'):
            spec.add_numeric(name=numeric.name,data=list(numeric.data))
            shared.delete(numeric)
    for numeric in list(actors.numerics):
        if numeric.name.startswith('filament.'):actors.delete(numeric)
    spec.visual.headlight.active=shared.visual.headlight.active
    for channel in ('ambient','diffuse','specular'):
        setattr(spec.visual.headlight,channel,getattr(shared.visual.headlight,channel))
    # Lighting belongs to the scene, not to an actor instance. Multiple copies
    # of the directional sun multiply illumination and shadow contributions.
    for light in list(actors.lights):actors.delete(light)
    terrain_names={geom.name for geom in actors.geoms
                   if geom.name=='terrain' or geom.name.startswith('display_terrain_')}
    # CPU display sensors are not authoritative. Remove only sensors whose
    # geom references disappear with the terrain; robot sensors stay intact.
    for sensor in list(actors.sensors):
        if ((sensor.objtype==mujoco.mjtObj.mjOBJ_GEOM and sensor.objname in terrain_names)
            or (sensor.reftype==mujoco.mjtObj.mjOBJ_GEOM and sensor.refname in terrain_names)):
            actors.delete(sensor)
    for geom in list(actors.geoms):
        if geom.name in terrain_names:actors.delete(geom)
    for hfield in list(actors.hfields):actors.delete(hfield)
    # Actor-only attachments may reference the parent's shared assets after
    # namespacing. Remove declarations here, while the template is small,
    # rather than deleting thousands of duplicate assets from a 64-actor spec.
    for collection in ('materials','textures','meshes'):
        for asset in list(getattr(actors,collection)):actors.delete(asset)
    spec.option.timestep=shared.option.timestep
    spec.option.gravity=shared.option.gravity
    spec.option.integrator=shared.option.integrator
    spec.option.ccd_iterations=shared.option.ccd_iterations
    for i,offset in enumerate(offsets(simulator.nworld)):
        tick=time.monotonic()
        child=shared if i==0 else actors.copy()
        stages['copy']+=time.monotonic()-tick
        for key in list(child.keys):child.delete(key)
        # Removing key objects does not clear the source XML's explicit
        # <size nkey="1"> allocation. Display poses come from live state.
        child.nkey=0
        frame=spec.worldbody.add_frame(pos=offset)
        tick=time.monotonic()
        spec.attach(child,prefix=f'world_{i}/',frame=frame)
        stages['attach']+=time.monotonic()-tick
        if i:
            prefix=f'world_{i}/'
            for collection in ('geoms','sites','skins','tendons'):
                for element in getattr(spec,collection):
                    if not element.name.startswith(prefix):continue
                    for attr in ('meshname','material'):
                        if hasattr(element,attr):
                            value=getattr(element,attr)
                            if value.startswith(prefix):setattr(element,attr,'world_0/'+value[len(prefix):])
        if profile and (i+1)%8==0:print('[Display profile]',i+1,stages,flush=True)
        elif not profile and simulator.nworld>=16 and ((i+1)%16==0 or i+1==simulator.nworld):
            print(f'[Startup] Attached {i+1}/{simulator.nworld} people; textures/materials remain shared',flush=True)
    tick=time.monotonic()
    share_actor_assets(spec)
    if profile:print('[Display profile] dedup',time.monotonic()-tick,'total',time.monotonic()-started,flush=True)
    spec.stat.center=[0,0,1.];spec.stat.extent=8.
    return spec


def sync_grid(backend,model,data):
    sim=backend._simulator;source=sim.mj_model;count=sim.nworld
    if getattr(backend,'_grid_model',None) is not model:
        backend._grid_model=model;backend._grid_maps=[]
        for i in range(count):
            prefix=f'world_{i}/'
            joint=model.joint(prefix+source.joint(0).name)
            bodies=[(b,model.body(prefix+source.body(b).name).id) for b in range(1,source.nbody)]
            geoms=[(g,model.geom(prefix+source.geom(g).name).id) for g in range(source.ngeom) if source.geom(g).name!='terrain']
            backend._grid_maps.append((int(joint.qposadr[0]),int(joint.dofadr[0]),bodies,geoms))
    arrays={name:getattr(sim.data,name)[:].cpu().numpy() for name in ('qpos','qvel','ctrl','xfrc_applied','mocap_pos','mocap_quat')}
    fields={name:getattr(sim.model,name)[:].cpu().numpy() for name in ('geom_pos','geom_quat','geom_size','geom_rgba','geom_dataid')}
    layout=offsets(count);ground=source.geom('terrain').id
    active_ids=fields['geom_dataid'][:,ground]
    if not np.all(active_ids==active_ids[0]):
        raise ValueError('Shared display requires shared_terrain=true: collision fields differ across worlds')
    for i,(qa,va,bodies,geoms) in enumerate(backend._grid_maps):
        data.qpos[qa:qa+source.nq]=arrays['qpos'][i];data.qpos[qa:qa+3]+=layout[i]
        data.qvel[va:va+source.nv]=arrays['qvel'][i]
        start=model.actuator(f'world_{i}/'+source.actuator(0).name).id
        data.ctrl[start:start+source.nu]=arrays['ctrl'][i]
        for src,dst in bodies:
            data.xfrc_applied[dst]=arrays['xfrc_applied'][i,src]
            a=int(source.body_mocapid[src]);b=int(model.body_mocapid[dst])
            if a>=0:
                data.mocap_pos[b]=arrays['mocap_pos'][i,a]+layout[i]
                data.mocap_quat[b]=arrays['mocap_quat'][i,a]
        for src,dst in geoms:
            for name,values in fields.items():
                # Asset indices belong to the compiled display model. Static
                # meshes/hfields are shared, never rebound from source indices.
                if name!='geom_dataid':getattr(model,name)[dst]=values[i,src]
    active=int(active_ids[0])
    for patch in range(model.nhfield):
        gid=model.geom('world_0/'+('terrain' if patch==0 else f'display_terrain_{patch:02d}')).id
        model.geom_group[gid]=0 if patch==active else 5
        model.geom_rgba[gid,3]=1.
    data.time=float(sim.data.time[backend._world_id].cpu())
    mujoco.mj_forward(model,data)


def grid_forces(backend,forces):
    sim=backend._simulator
    result=np.zeros((sim.nworld,sim.mj_model.nbody,6))
    for i,(_,_,bodies,_) in enumerate(backend._grid_maps):
        for src,dst in bodies:result[i,src]=forces[dst]
    return result


def apply_grid_force(backend,forces):
    import torch
    sim=backend._simulator
    sim.data.xfrc_applied[:]=torch.as_tensor(grid_forces(backend,forces),device=sim.xpu_device,dtype=sim.data.xfrc_applied.dtype)
