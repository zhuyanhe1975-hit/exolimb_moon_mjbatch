"""Independent native CPU worlds with one immutable shared terrain/model."""
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from moon_device_core import MoonDeviceCore


class CpuWorlds:
    def __init__(self, spec, count, terrain_idx):
        self.model = spec.compile()
        self.data = [mujoco.MjData(self.model) for _ in range(count)]
        self.count = count
        self.mouse_forces = np.zeros((count,self.model.nbody,6))
        def batch(array, dtype=torch.float32):
            return torch.tensor(np.repeat(np.asarray(array)[None], count, axis=0), dtype=dtype)
        self.sim = SimpleNamespace(mj_spec=spec, mj_model=self.model, nworld=count,
            model=SimpleNamespace(**{name:batch(getattr(self.model,name), torch.long if name=='geom_dataid' else torch.float32)
                for name in ('geom_dataid','geom_pos','geom_quat','geom_size','geom_rgba')}),
            data=SimpleNamespace(**{name:batch(getattr(self.data[0],name)) for name in
                ('qpos','qvel','ctrl','act','xfrc_applied','sensordata','actuator_force','mocap_pos','mocap_quat')}))
        self.sim.data.time = torch.zeros(count)
        self.device = MoonDeviceCore(pushes=False, shared_terrain=True, resample_terrain=False, terrain_idx=terrain_idx)
        self.device.initialize(self.sim)
        self.reset()

    def read_native(self):
        for i, data in enumerate(self.data):
            for name in ('qpos','qvel','sensordata','actuator_force','mocap_pos','mocap_quat'):
                getattr(self.sim.data,name)[i].copy_(torch.from_numpy(getattr(data,name)))
            self.sim.data.time[i] = data.time

    def reset(self, mask=None):
        self.device.reset(self.sim, mask)
        if mask is None:
            self.mouse_forces[:] = 0
        else:
            self.mouse_forces[mask.numpy()] = 0
        ids = range(self.count) if mask is None else mask.nonzero(as_tuple=True)[0].tolist()
        # Shared model is valid only because every world uses the same geometry.
        fields = self.sim.model
        if not torch.equal(fields.geom_dataid, fields.geom_dataid[:1].expand_as(fields.geom_dataid)):
            raise ValueError('CPU worlds must share the same terrain')
        if not torch.equal(fields.geom_pos, fields.geom_pos[:1].expand_as(fields.geom_pos)):
            raise ValueError('CPU worlds must share terrain placement')
        self.model.geom_dataid[:] = fields.geom_dataid[0].numpy()
        self.model.geom_pos[:] = fields.geom_pos[0].numpy()
        for i in ids:
            data = self.data[i]
            mujoco.mj_resetData(self.model,data)
            for name in ('qpos','qvel','ctrl','xfrc_applied'):
                getattr(data,name)[:] = getattr(self.sim.data,name)[i].numpy()
            mujoco.mj_forward(self.model,data)
        self.read_native()

    def step(self, action):
        for _ in range(2):
            self.device.set_substep_commands(self.sim, {'wangyu/action':action})
            for i, data in enumerate(self.data):
                data.ctrl[:] = self.sim.data.ctrl[i].numpy()
                data.xfrc_applied[:] = self.sim.data.xfrc_applied[i].numpy() + self.mouse_forces[i]
                mujoco.mj_step(self.model,data)
        self.device.on_step(self.sim)
        for data in self.data:
            mujoco.mj_forward(self.model,data)
        self.read_native()
        return self.device.get_measurements(self.sim)
