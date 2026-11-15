"""Independent CPU display scene; the GPU simulation stays unchanged."""
import mujoco
import numpy as np
import os
from pathlib import Path


def display_spec(simulator,*,single=False):
    if not single and os.environ.get('MOON_VIEWER') in ('studio','classic') and simulator.nworld>1:
        from multiworld_viewer import build_grid
        return build_grid(simulator)
    spec=simulator.mj_spec.copy()
    geom=next(g for g in spec.geoms if g.name=='terrain')
    # Filament dev fails to display this hfield on its zero-DOF mocap parent.
    # Keep the unused mocap body so state layout is unchanged, but put the
    # same hfield on world in the CPU display scene. Its parent is identity.
    body=spec.body('site20_ground')
    np.testing.assert_allclose(body.pos,[0,0,0],atol=1e-12)
    np.testing.assert_allclose(body.quat,[1,0,0,0],atol=1e-12)
    params={k:getattr(geom,k) for k in ('name','type','hfieldname','pos','quat','size','rgba','group')}
    params={k:(np.array(v).copy().tolist() if hasattr(v,'tolist') else v) for k,v in params.items()}
    spec.delete(geom)
    spec.worldbody.add_geom(**params,contype=0,conaffinity=0)
    with np.load(Path(__file__).resolve().parents[1]/'assets/playground/state.npz') as archive:
        zmin=archive['zmin'].copy()
    for i in range(1,len(spec.hfields)):
        spec.worldbody.add_geom(name=f'display_terrain_{i:02d}',type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=f'site20_local_{i:02d}',pos=[0,0,float(zmin[i])],
            contype=0,conaffinity=0,rgba=[.42,.43,.45,1],group=5)
    spec.stat.center=[0,0,1.]
    spec.stat.extent=3.
    if os.environ.get('MOON_FILAMENT_SCENE')=='1':
        from filament_scene import add_astronaut
        add_astronaut(spec)
        if os.environ.get('MOON_VIEWER')=='classic':
            from classic_materials import adapt
            adapt(spec)
    return spec


def install():
    from reaf.simulation.torch.viewer._src import sim_backend
    from reaf.simulation.torch import viewer
    base=sim_backend.TorchSimBackend
    selection={'world':0}

    class MoonViewerBackend(base):
        def sync_to_viewer(self,model,data):
            sim=self._simulator;world=self._world_id
            selection['world']=world
            if model is sim.mj_model:raise RuntimeError('Display model must be independent')
            source=sim.mj_model
            from multiworld_viewer import world_count,sync_grid,offsets
            if world_count(model):
                selection['offsets']=offsets(sim.nworld)
                sync_grid(self,model,data)
                return
            key=id(model)
            if getattr(self,'_moon_model_key',None)!=key:
                self._moon_geom_map={i:source.geom(model.geom(i).name).id for i in range(model.ngeom)
                    if not model.geom(i).name.startswith('display_terrain_')}
                self._moon_model_key=key
                if (model.nq,model.nv,model.nbody,model.nmocap)!=(source.nq,source.nv,source.nbody,source.nmocap):
                    raise ValueError('Display state layout differs from GPU model')
            # Copy only integration state. Render-only geoms do not exist in GPU data.
            for name in ('qpos','qvel','act','mocap_pos','mocap_quat','ctrl','qfrc_applied','xfrc_applied'):
                getattr(data,name)[:]=getattr(sim.data,name)[world].cpu().numpy()
            data.time=float(sim.data.time[world].cpu())
            for name in ('geom_size','geom_rgba','geom_pos','geom_quat','geom_dataid'):
                values=getattr(sim.model,name)[world].cpu().numpy()
                for dst,src in self._moon_geom_map.items():
                    if model.geom(dst).name!='terrain':getattr(model,name)[dst]=values[src]
            gid=source.geom('terrain').id
            active=int(sim.model.geom_dataid[world,gid].cpu())
            for i in range(model.nhfield):
                dst=model.geom('terrain' if i==0 else f'display_terrain_{i:02d}').id
                # Every geom keeps a fixed native asset; a Snapshot syncs groups.
                model.geom_rgba[dst]=[1,1,1,1.] if os.environ.get('MOON_FILAMENT_SCENE')=='1' and os.environ.get('MOON_FILAMENT_TERRAIN_PBR','1')=='1' else [.42,.43,.45,1.]
                model.geom_group[dst]=0 if model.geom_dataid[dst]==active else 5
            for name in ('light_pos','light_dir','light_diffuse','light_ambient','cam_pos','cam_quat','cam_fovy','cam_intrinsic','site_pos','site_quat'):
                if name.startswith('light_') and model.nskin and os.environ.get('MOON_FILAMENT_TERRAIN_PBR','1')=='1':continue
                getattr(model,name)[:]=getattr(sim.model,name)[world].cpu().numpy()
            # CPU only, for display transforms. Never steps the CPU scene.
            mujoco.mj_forward(model,data)
            data.sensordata[:]=sim.data.sensordata[world].cpu().numpy()
        def apply_perturbation(self,forces):
            if hasattr(self,'_grid_maps'):
                from multiworld_viewer import apply_grid_force
                return apply_grid_force(self,forces)
            return super().apply_perturbation(forces)
    sim_backend.TorchSimBackend=MoonViewerBackend

    launch=viewer.launch
    class DisplaySimulator:
        def __init__(self,sim):
            self._sim=sim;self.mj_spec=display_spec(sim)
            print(f'[Playback worlds] simulated={sim.nworld}; displayed={sim.nworld}',flush=True)
        def __getattr__(self,name):return getattr(self._sim,name)
    def launch_moon(environment,*args,simulator=None,**kwargs):
        sim=simulator or environment.device_layer.sim
        from moon_studio_plugins import TerrainSender,TerrainReceiver,ContactSender,ContactReceiver
        selection['world']=kwargs.get('main_world_id',0)
        from playback_clock import studio_clock_plugin
        from terrain_controls import TerrainControlSender,TerrainControlReceiver
        kwargs['plugins']=[*kwargs.get('plugins',()),TerrainSender(),TerrainReceiver(),ContactSender(sim,selection),ContactReceiver(),studio_clock_plugin(),TerrainControlSender(),TerrainControlReceiver(environment)]
        return launch(environment,*args,simulator=DisplaySimulator(sim),**kwargs)
    viewer.launch=launch_moon
