"""CPU-owned physics and independent classic/Studio presentation models."""
from contextlib import nullcontext
from types import SimpleNamespace
from queue import SimpleQueue
import os,sys,time
import mujoco


def launch(spec, model, data, viewer, viz, key, state, reset_requested, *, simulator=None):
    os.environ['MOON_VIEWER'] = viewer
    os.environ['MOON_FILAMENT_SCENE'] = '1' if viz == 'skin' else '0'
    from moon_viewer import display_spec
    count=simulator.nworld if simulator is not None else 1
    started=time.monotonic()
    print(f'[Startup] Building {viewer}/{viz} display for {count} people...',flush=True)
    shown = display_spec(simulator or SimpleNamespace(mj_spec=spec, nworld=1)).compile()
    print(f'[Startup] Display model ready in {time.monotonic()-started:.2f}s; skins={shown.nskin}; shared textures={shown.ntex}',flush=True)
    return CpuDisplay(model, data, shown, viewer, viz, key, state, reset_requested, simulator)


class CpuDisplay:
    def __init__(self, model, data, shown, kind, viz, key, state, reset_requested, simulator=None):
        self.source_model, self.source_data = model, data
        self.model, self.data = shown, mujoco.MjData(shown)
        self.kind, self.viz = kind, viz
        self.key = key
        self._keys = SimpleQueue()
        self.state, self.reset_requested = state, reset_requested
        self._active = None
        self.simulator = simulator
        if simulator is not None and simulator.nworld > 1:
            self._grid_backend = SimpleNamespace(_simulator=simulator, _world_id=0)
        self._copy_state()
        print(f'[Startup] Creating {kind} window...',flush=True)
        if kind == 'classic':
            from mujoco import viewer
            self.native = viewer.launch_passive(shown, self.data, key_callback=self._keys.put)
            self.cam, self.opt = self.native.cam, self.native.opt
        else:
            from skin_picking import install as install_skin_picking
            install_skin_picking()
            from studio_native import install
            install(ui_bridge='local' if sys.platform.startswith('linux') else 'native')
            from mujoco.studio import native_viewer, viewer_protocol, launch_thread
            from mujoco.studio.app import App,inputs
            from studio_input import configure_mac_input,StudioFocus,mac_control_or_command
            input_api=configure_mac_input()
            owner = self
            class CpuApp(App):
                def handle_mouse_events(app):
                    from mujoco.studio.dear_imgui import dear_imgui as imgui
                    io=imgui.GetIO()
                    using_modifier=io.KeyCtrl or (sys.platform=='darwin' and io.KeySuper)
                    with mac_control_or_command(imgui):
                        return inputs.handle_mouse_events(app.model,app.data,app.viewer.camera,
                            app.viewer.vis_options,app.viewer.perturb,app.ux_state,
                            move_camera=app.viewer.client_camera is None and not using_modifier)
                def handle_keyboard_events(app):
                    if owner.handle_studio_keys(app): return
                    super().handle_keyboard_events()
                    app.status = owner.controls.publish()
            self._studio_app = CpuApp()
            endpoint, self.sim_endpoint = launch_thread.make_thread_endpoints()
            self.native = native_viewer.NativeViewer(
                viewer_protocol.ViewerConfig(title='Moon CPU — Studio', gfx='opengl'),
                endpoint, model=shown, plugins=[self._studio_app])
            self.model, self.data = self.native.model, self.native.data
            self._copy_state()
            self.native._moon_camera_state_model_id = id(self.model)
            self._input_focus=StudioFocus(input_api,'Moon CPU — Studio')
            self._input_focus_requested=False
            self._input_focus_last=None
            self.cam, self.opt = self.native.camera, self.native.vis_options
        self.opt.geomgroup[4] = 0 if viz == 'skin' else 1
        self.opt.geomgroup[5] = 0
        self.opt.skingroup[1] = 1
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = viz == 'skin'
        from cpu_camera import CameraControls
        count = simulator.nworld if simulator is not None else 1
        self.controls = CameraControls(self.model,self.data,self.cam,count,state,self._select_body)
        self._classic_reserved = {
            ',':mujoco.mjtVisFlag.mjVIS_ACTIVATION,
            'A':mujoco.mjtVisFlag.mjVIS_AUTOCONNECT}
        self._classic_flag_values = {key:int(self.opt.flags[flag]) for key,flag in self._classic_reserved.items()}

    def _copy_state(self):
        if hasattr(self,'_grid_backend'):
            from multiworld_viewer import sync_grid
            active = int(self.source_model.geom('terrain').dataid[0])
            changed = active != self._active
            sync_grid(self._grid_backend,self.model,self.data)
            self._active = active
            if changed and self.kind == 'studio' and hasattr(self,'native'):
                self.native._moon_terrain_visibility_changed = True
            return
        for name in ('qpos','qvel','act','mocap_pos','mocap_quat','ctrl'):
            getattr(self.data,name)[:] = getattr(self.source_data,name)
        self.data.time = self.source_data.time
        for gname in getattr(self, 'state', {}).get('sync_geom_rgba', ()):
            src_id = mujoco.mj_name2id(self.source_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            dst_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if src_id >= 0 and dst_id >= 0:
                self.model.geom_rgba[dst_id] = self.source_model.geom_rgba[src_id]
        active = int(self.source_model.geom('terrain').dataid[0])
        changed = active != self._active
        for i in range(self.model.nhfield):
            gid = self.model.geom('terrain' if i == 0 else f'display_terrain_{i:02d}').id
            self.model.geom_group[gid] = 0 if int(self.model.geom_dataid[gid]) == active else 5
        self._active = active
        mujoco.mj_forward(self.model,self.data)
        self.data.sensordata[:] = self.source_data.sensordata
        if changed and self.kind == 'studio' and hasattr(self,'native'):
            self.native._moon_terrain_visibility_changed = True

    def _select_body(self, body):
        import numpy as np
        perturb = self.native.perturb
        perturb.active = 0
        perturb.select = body
        perturb.localpos[:] = 0
        self.data.xfrc_applied[:] = 0
        self.state['mouse_forces'] = np.zeros((self.controls.count,self.source_model.nbody,6))
        if hasattr(self,'_grid_backend'):
            self._grid_backend._world_id = self.controls.current

    def _sync_camera_mode(self, app):
        from mujoco.studio import ux
        app.ux_state.camera_index = ux.TRACKING_CAMERA_IDX if self.controls.tracking else ux.TUMBLE_CAMERA_IDX
        self.native.camera_index = app.ux_state.camera_index
        app.status = self.controls.publish()

    def handle_key(self, key):
        """Display style never changes the meaning of an interaction key."""
        handled = self.controls.handle(key)
        extra = {k for k, _ in getattr(self, 'state', {}).get('extra_keys', ())}
        if not handled and (key in ('[',']','R') or key in extra):
            self.key(ord(key))
            handled = True
        if handled and self.kind == 'studio':
            self._sync_camera_mode(self._studio_app)
        return handled

    def update_mouse_interaction(self):
        # Both backends use the same native perturbation and body/world mapping.
        # Do not rely on a backend-specific sync side effect to compute force.
        self.controls.adopt_mouse_selection(self.native.perturb.select)
        if hasattr(self,'_grid_backend'):
            self._grid_backend._world_id = self.controls.current
        self.data.xfrc_applied[:] = 0
        if not self.state.get('paused',False):
            mujoco.mjv_applyPerturbForce(self.model,self.data,self.native.perturb)
        self.capture_mouse_forces()
        label = self.controls.publish()
        if self.kind == 'studio':
            self._studio_app.status = label
        return label

    def handle_studio_keys(self, app):
        from mujoco.studio.dear_imgui import dear_imgui as imgui
        io = imgui.GetIO()
        if getattr(io,'WantTextInput',False) or imgui.IsAnyItemActive(): return False
        pressed = imgui.IsKeyChordPressed
        keys = [(key,getattr(imgui.Key,name)) for key,name in
                ((',', 'Comma'),('.', 'Period'),('/', 'Slash'),('A','A'),('Escape','Escape'))]
        for key,chord in keys:
            if pressed(chord):
                self.handle_key(key)
                return True
        extra_studio = [(ord(k), name) for k, name in getattr(self, 'state', {}).get('extra_keys', ())]
        for code,name in ((ord('['),'LeftBracket'),(ord(']'),'RightBracket'),(ord('R'),'R'), *extra_studio):
            if pressed(getattr(imgui.Key,name)):
                self.handle_key(chr(code))
                return True
        return False

    def _classic_keys(self):
        while not self._keys.empty():
            code = self._keys.get()
            key = chr(code).upper() if 0<=code<256 else ('Escape' if code==256 else '')
            if self.handle_key(key):
                if key in self._classic_reserved:
                    self.opt.flags[self._classic_reserved[key]] = self._classic_flag_values[key]
            else:
                self.key(code)
        self._classic_flag_values = {key:int(self.opt.flags[flag]) for key,flag in self._classic_reserved.items()}

    def capture_mouse_forces(self):
        """Map native viewer body forces back to the corresponding CPU world."""
        import numpy as np
        if hasattr(self,'_grid_backend'):
            from multiworld_viewer import grid_forces
            forces = grid_forces(self._grid_backend,self.data.xfrc_applied)
        else:
            forces = self.data.xfrc_applied[None].copy()
        if self.state.get('paused',False):
            forces[:] = 0
        self.state['mouse_forces'] = np.array(forces,copy=True)

    def update_contacts(self):
        """Display collisions from the solver, never the collision-free skin model."""
        from contact_display import draw_contacts
        points=bool(self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT])
        forces=bool(self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE])
        sources=getattr(self.simulator,'native_data',[self.source_data])
        if self.kind == 'classic':
            scene=self.native.user_scn
        else:
            if not hasattr(self,'_contact_scene'):
                self._contact_scene=mujoco.MjvScene(self.source_model,maxgeom=10000)
            scene=self._contact_scene
        scene.ngeom=0
        if points or forces:
            for data in sources:
                draw_contacts(self.source_model,data,scene,points=points,forces=forces,append=True)
        if self.kind == 'studio':
            self.native.extra_geoms[:]=list(scene.geoms[:scene.ngeom])

    def lock(self):
        return self.native.lock() if self.kind == 'classic' else nullcontext()

    def is_running(self):
        return self.native.is_running()

    def sync(self):
        with self.lock():
            self._copy_state()
            if self.kind == 'classic': self._classic_keys()
            if self.kind == 'classic': self.update_contacts()
        if self.kind == 'classic':
            self.native.sync()
            with self.lock():
                label = self.update_mouse_interaction()
            if self.native.is_running() and label != getattr(self,'_last_status',None):
                self.native.set_texts((None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,'CPU',label))
                self._last_status = label
            return
        from mujoco.studio import messages, sim
        from mujoco.studio.dear_imgui import dear_imgui as imgui
        frame = self.native.prepare_next_frame()
        if not self.native.is_running(): return
        for event in self.sim_endpoint.get_viewer_events():
            if isinstance(event, messages.ResetEvent): self.reset_requested.set()
            elif isinstance(event, messages.RequestPauseEvent):
                self.state['paused'] = event.pause_state != sim.PauseState.UNPAUSED
        for event in self.sim_endpoint.get_viewer_snapshots():
            if isinstance(event,messages.StepControlSnapshot):
                self.state['paused'] = event.pause_state != sim.PauseState.UNPAUSED
                self.state['speed'] = max(.01, float(event.speed) / 100)
        if frame:
            focused=self._input_focus.keyboard_focused()
            if focused is not None and focused!=self._input_focus_last:
                print('[Studio input] keyboard_focus='+str(focused)+
                      '; Mac mouse modifier: Control (Command compatible)',flush=True)
                self._input_focus_last=focused
                self.state['keyboard_focused']=focused
            self.native.dispatch(messages.UpdateEvent())
            self.update_mouse_interaction()
            self.update_contacts()
            self.native.dispatch(messages.BuildGuiEvent())
            self.native.sync()
            if not self._input_focus_requested:
                # Model/texture creation can change native contexts. Activate
                # only after the first complete, visible Studio frame exists.
                self._input_focus.acquire()
                self._input_focus_requested=True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        if self.kind == 'classic': self.native.close()
        else: self.native.close(); self.sim_endpoint.close()
