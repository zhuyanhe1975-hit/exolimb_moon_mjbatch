"""Camera and selected-world controls shared by CPU classic and Studio."""
import numpy as np
import mujoco


class CameraControls:
    def __init__(self, model, data, camera, count, state, on_select=None):
        self.model, self.data, self.camera = model, data, camera
        self.count, self.state, self.on_select = count, state, on_select
        self.current = 0
        self.overview = False
        self._saved = None
        self.roots = [model.body((f'world_{i}/' if count > 1 else '')+'robot/Pelvis').id
                      for i in range(count)]
        self.body_world = {}
        for body in range(1,model.nbody):
            name = model.body(body).name
            if count == 1:
                if name.startswith('robot/'): self.body_world[body] = 0
            elif name.startswith('world_'):
                prefix, suffix = name.split('/',1)
                if suffix.startswith('robot/'): self.body_world[body] = int(prefix[6:])
        self.publish()

    @property
    def tracking(self):
        return self.camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING

    def publish(self, *, log=False):
        mode = 'overview' if self.overview else ('tracking' if self.tracking else 'free')
        self.state.update(selected_world=self.current, camera_mode=mode)
        label = f'Person {self.current+1}/{self.count} | Camera: {mode} | /: follow | , .: person | A: overview'
        if log: print('[Camera] '+label,flush=True)
        return label

    def set_tracking(self, enabled):
        self.overview = False
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING if enabled else mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.trackbodyid = self.roots[self.current] if enabled else -1
        self.camera.fixedcamid = -1
        if enabled:
            self.camera.lookat[:] = self.data.xpos[self.roots[self.current]]
            self.camera.distance = 5.
        self.publish(log=True)

    def select(self, index):
        was_tracking = self.tracking
        if self.overview and self._saved:
            was_tracking = self._saved['type'] == mujoco.mjtCamera.mjCAMERA_TRACKING
        self.current = index % self.count
        self.overview = False
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING if was_tracking else mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.trackbodyid = self.roots[self.current] if was_tracking else -1
        self.camera.fixedcamid = -1
        self.camera.lookat[:] = self.data.xpos[self.roots[self.current]]
        self.camera.distance = 5.
        if self.on_select: self.on_select(self.roots[self.current])
        self.publish(log=True)

    def toggle_overview(self):
        fields = ('type','trackbodyid','fixedcamid','distance','azimuth','elevation','lookat')
        if not self.overview:
            self._saved = {name:np.array(getattr(self.camera,name),copy=True) if name=='lookat'
                           else getattr(self.camera,name) for name in fields}
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.trackbodyid = self.camera.fixedcamid = -1
            positions = self.data.xpos[self.roots]
            self.camera.lookat[:] = positions.mean(axis=0)
            self.camera.distance = max(5.,24. if self.count>1 else 5.,float(np.linalg.norm(np.ptp(positions,axis=0)))*1.8)
            self.camera.elevation = -50.
            self.overview = True
        else:
            for name,value in self._saved.items():
                if name=='lookat': self.camera.lookat[:] = value
                else: setattr(self.camera,name,value)
            self.overview = False
        self.publish(log=True)

    def handle(self, key):
        if key=='/': self.set_tracking(not self.tracking)
        elif key=='.':
            self.select(self.current+1)
            self.set_tracking(True)
        elif key==',':
            self.select(self.current-1)
            self.set_tracking(True)
        elif key=='A': self.toggle_overview()
        elif key=='Escape': self.set_tracking(False)
        else: return False
        return True

    def adopt_mouse_selection(self, body):
        if self.tracking: self.overview = False
        index = self.body_world.get(body)
        if index is not None and index != self.current:
            self.current = index
            if self.tracking: self.camera.trackbodyid = body
            self.publish(log=True)
