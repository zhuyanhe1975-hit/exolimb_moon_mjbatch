"""Classic passive MuJoCo window for the Playground simulation and policy."""
from queue import SimpleQueue
from contextlib import contextmanager
import threading
import mujoco
import torch
from moon_viewer import display_spec


def launch(environment,*,simulator=None,policy=None,main_world_id=0,**kwargs):
    import mujoco.viewer
    from reaf.simulation.torch.viewer._src import sim_backend
    sim=simulator or environment.device_layer.sim
    if not 0<=main_world_id<sim.nworld:
        raise ValueError('env_id must select an existing world')
    if policy is None:
        class ZeroPolicy:
            def initial_state(self):return ()
            def step(self,timestep,state):
                actions={k:torch.zeros(s.shape,device=sim.xpu_device)
                    for k,s in environment.action_spec().items()}
                return (actions,{}),state
        policy=ZeroPolicy()
    # MoonViewerBackend (installed before this launcher) copies state to an
    # independent CPU display model; it never steps that model.
    backend=sim_backend.TorchSimBackend(environment,policy,sim,str(sim.xpu_device),main_world_id)
    model=display_spec(sim).compile();data=mujoco.MjData(model)
    # Apply GUI perturbations after task devices set their own external forces.
    # This keeps random task pushes additive and avoids changing the CPU display
    # pose directly (the authoritative physics remains in Playground).
    force=torch.zeros((sim.nworld,sim.mj_model.nbody,6),device=sim.xpu_device)
    last_device=environment.device_layer._devices[-1]
    original_commands=last_device.set_substep_commands
    original_substep=last_device.on_substep
    applied=torch.zeros_like(force)
    def commands_with_mouse(simulator,commands):
        original_commands(simulator,commands)
        applied.copy_(force)
        simulator.data.xfrc_applied[:]+=applied
    def remove_mouse_after_step(simulator):
        simulator.data.xfrc_applied[:]-=applied
        original_substep(simulator)
    last_device.set_substep_commands=commands_with_mouse
    last_device.on_substep=remove_mouse_after_step
    keys=SimpleQueue()
    print(f'[Playback worlds] simulated={sim.nworld}; displayed={sim.nworld if sim.nworld>1 else 1}',flush=True)
    print('[Classic viewer] Double-click a body, Ctrl + mouse drag: apply force. '
          'Space: pause; R: reset; [/]: previous/next terrain + reset; N: next world; Esc: close.',flush=True)
    try:
        backend.reset();backend.sync_to_viewer(model,data)
        _run_window(environment,backend,model,data,force,keys)
    finally:
        last_device.set_substep_commands=original_commands
        last_device.on_substep=original_substep


@contextmanager
def passive_window(model,data,**kwargs):
    """Keep model/data alive until the classic render thread fully exits."""
    import mujoco.viewer
    before=set(threading.enumerate())
    window=mujoco.viewer.launch_passive(model,data,**kwargs)
    # The installed Linux launcher starts _launch_internal on a daemon thread.
    # close() signals render exit; join also waits for its destroy/GLFW cleanup.
    threads=[thread for thread in threading.enumerate() if thread not in before
        and getattr(thread,'_target',None) is mujoco.viewer._launch_internal
        and getattr(thread,'_args',())[0:2]==(model,data)]
    try:
        yield window
    finally:
        window.close()
        for thread in threads:thread.join()


def _run_window(environment,backend,model,data,force,keys):
    import mujoco.viewer
    from contact_display import ContactDisplay
    contacts=ContactDisplay(backend._simulator)
    from playback_clock import PlaybackClock
    clock=PlaybackClock()
    paused=False
    with passive_window(model,data,key_callback=keys.put) as window:
        with window.lock():
            from multiworld_viewer import world_count
            count=world_count(model)
            if count:
                from large_terrain import half_extent
                window.cam.distance=150. if half_extent(model)>8 else 18.
                window.cam.lookat[:]=[0,0,1.];window.cam.elevation=-50.
            else:
                window.cam.distance=5.;window.cam.lookat[:]=data.qpos[:3]
            window.opt.geomgroup[5]=0
            if model.nskin:
                window.opt.geomgroup[4]=0
                window.opt.skingroup[1]=1
                window.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN]=1
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=1
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=1
        clock.reset()
        while window.is_running():
            previous_time=float(data.time)
            while not keys.empty():
                key=keys.get()
                if key==32:paused=not paused
                elif key in (82,114):
                    force.zero_();backend.reset()
                elif key in (91,93):
                    from terrain_controls import select_relative_terrain
                    if select_relative_terrain(environment,-1 if key==91 else 1):
                        force.zero_();backend.reset();clock.reset()
                elif key in (78,110):
                    force.zero_();backend._world_id=(backend._world_id+1)%backend._simulator.nworld
            if not paused:backend.step()
            with window.lock():
                backend.sync_to_viewer(model,data)
                window.opt.geomgroup[5]=0
                contacts.update(window,backend._world_id)
                if count:
                    from multiworld_viewer import offsets
                    shift=offsets(count)[backend._world_id]
                    for geom in window.user_scn.geoms[:window.user_scn.ngeom]:geom.pos[:]+=shift
            window.sync()
            with window.lock():
                data.xfrc_applied[:]=0
                mujoco.mjv_applyPerturbForce(model,data,window.perturb)
                if count:
                    from multiworld_viewer import grid_forces
                    applied=grid_forces(backend,data.xfrc_applied)
                    force.copy_(torch.as_tensor(applied,device=force.device,dtype=force.dtype))
                else:
                    force.zero_()
                    force[backend._world_id].copy_(torch.as_tensor(data.xfrc_applied,device=force.device,dtype=force.dtype))
            if not paused:
                clock.wait(float(data.time)-previous_time)
            if paused:
                clock.reset()
                # No simulation advances while paused; avoid a busy UI loop.
                import time
                time.sleep(.01)


def install():
    from reaf.simulation.torch import viewer
    viewer.launch=launch
