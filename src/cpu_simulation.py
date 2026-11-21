"""Independent native CPU worlds with one shared model per active terrain."""
from copy import copy
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from moon_device_core import MoonDeviceCore
from moon_scene import SOLE_TERRAIN_CONTACTS, vertical_load_columns


class CpuWorlds:
    def __init__(self, spec, count, terrain_idx, *, pushes=False,
                 use_mjbatch=False, workers=1, optimize_cpu=False, inward_heading=False,
                 terrain_indices=None, spawn_seed=42, assist_force=None, edge_start=False):
        self.model = spec.compile()
        load_sensor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                        'contacts/robot/left_sole_terrain_load')
        self._sole_terrain_columns = None
        if load_sensor != -1:
            self._sole_terrain_columns = []
            self._sole_load_columns = []
            for side in ('left', 'right'):
                source = int(self.model.sensor(f'contacts/robot/{side}_sole_terrain_load').adr[0])
                self._sole_terrain_columns.extend(range(source, source + 4 * SOLE_TERRAIN_CONTACTS))
                self._sole_load_columns.append(int(self.model.sensor(f'robot/balance_{side}_load').adr[0]))
        self._vertical_columns = vertical_load_columns(self.model)
        template = mujoco.MjData(self.model)
        self.data = None if use_mjbatch else [template] + [
            mujoco.MjData(self.model) for _ in range(count - 1)]
        self._batch = None
        self._workers = workers
        self.count = count
        self.mouse_forces = np.zeros((count,self.model.nbody,6))
        def batch(array, dtype=torch.float32):
            return torch.tensor(np.repeat(np.asarray(array)[None], count, axis=0), dtype=dtype)
        self.sim = SimpleNamespace(mj_spec=spec, mj_model=self.model, nworld=count,
            model=SimpleNamespace(**{name:batch(getattr(self.model,name), torch.long if name=='geom_dataid' else torch.float32)
                for name in ('geom_dataid','geom_pos','geom_quat','geom_size','geom_rgba')}),
            data=SimpleNamespace(**{name:batch(getattr(template,name)) for name in
                ('qpos','qvel','ctrl','act','xfrc_applied','sensordata','actuator_force',
                 'actuator_velocity','mocap_pos','mocap_quat')}))
        self.sim.data.time = torch.zeros(count)
        self.sim.native_data = self.data
        self.device = MoonDeviceCore(pushes=pushes, shared_terrain=True, resample_terrain=False,
                                     terrain_idx=terrain_idx, optimize_cpu=optimize_cpu,
                                     inward_heading=inward_heading, edge_start=edge_start)
        self.device.initialize(self.sim)
        self.exoskeleton = None
        if assist_force is not None:
            from exoskeleton import TelescopingAssist
            self.exoskeleton = TelescopingAssist(self, assist_force)
        self._terrain_models = None
        if terrain_indices is not None:
            from local_lunar_terrain import sole_sampling
            from playback_spawns import IndependentSpawns

            if (not terrain_indices or len(set(terrain_indices)) != len(terrain_indices)
                    or any(not 1 <= index <= 64 for index in terrain_indices)
                    or len(terrain_indices) > count):
                raise ValueError('Use unique terrain indices in 1–64 with at least one world per terrain')
            group_sizes = np.full(len(terrain_indices), count // len(terrain_indices))
            group_sizes[:count % len(terrain_indices)] += 1
            patches = np.repeat(np.asarray(terrain_indices) - 1, group_sizes)
            ground = self.device.ground
            _, cover = sole_sampling(self.model)
            self.device.spawn_sampler = IndependentSpawns(
                ground.meta, patches, ground.sole_points.numpy(), cover, spawn_seed)
            models = {}
            for patch in np.unique(patches):
                model = copy(self.model)
                model.geom_dataid[ground.geom_id] = int(ground.native_ids[patch])
                model.geom_pos[ground.geom_id, 2] = float(ground.zmin[patch])
                models[patch] = model
            self._terrain_models = [models[patch] for patch in patches]
        self.reset()

    def read_native(self):
        if self._batch is not None:
            for name, values in self._batch_outputs.items():
                np.copyto(getattr(self.sim.data, name).numpy(), values)
        else:
            for i, data in enumerate(self.data):
                for name in ('qpos','qvel','sensordata','actuator_force','actuator_velocity','mocap_pos','mocap_quat'):
                    getattr(self.sim.data,name)[i].copy_(torch.from_numpy(getattr(data,name)))
                self.sim.data.time[i] = data.time
        if self._vertical_columns is not None:
            sensors = self.sim.data.sensordata
            sensors[:, self._sole_load_columns] = sensors[:, self._vertical_columns].clamp_min(0.)
        elif self._sole_terrain_columns is not None:
            sensors = self.sim.data.sensordata
            contacts = sensors[:, self._sole_terrain_columns].reshape(self.count, 2, SOLE_TERRAIN_CONTACTS, 4)
            loads = (contacts[..., 1].abs() * (contacts[..., 0] > 0)).sum(-1)
            sensors[:, self._sole_load_columns] = loads

    def reset(self, mask=None):
        self.device.reset(self.sim, mask)
        if self.exoskeleton is not None:
            self.exoskeleton.reset_controls(mask)
        if mask is None:
            self.mouse_forces[:] = 0
        else:
            self.mouse_forces[mask.numpy()] = 0
        terrain_changed = False
        if self._terrain_models is None:
            # Shared model is valid only because every world uses the same geometry.
            fields = self.sim.model
            if not torch.equal(fields.geom_dataid, fields.geom_dataid[:1].expand_as(fields.geom_dataid)):
                raise ValueError('CPU worlds must share the same terrain')
            if not torch.equal(fields.geom_pos, fields.geom_pos[:1].expand_as(fields.geom_pos)):
                raise ValueError('CPU worlds must share terrain placement')
            geom_ids = fields.geom_dataid[0].numpy()
            geom_pos = fields.geom_pos[0].numpy()
            terrain_changed = (not np.array_equal(self.model.geom_dataid, geom_ids)
                               or not np.array_equal(self.model.geom_pos, geom_pos))
            self.model.geom_dataid[:] = geom_ids
            self.model.geom_pos[:] = geom_pos
        if self.data is None:
            self._reset_batch(mask, terrain_changed)
        else:
            ids = range(self.count) if mask is None else mask.nonzero(as_tuple=True)[0].tolist()
            for i in ids:
                data = self.data[i]
                model = self._model_for(i)
                mujoco.mj_resetData(model,data)
                for name in ('qpos','qvel','ctrl','xfrc_applied'):
                    getattr(data,name)[:] = getattr(self.sim.data,name)[i].numpy()
                mujoco.mj_forward(model,data)
        self.read_native()

    def _model_for(self, index):
        return self.model if self._terrain_models is None else self._terrain_models[index]

    def _reset_batch(self, mask, terrain_changed):
        if self._batch is None:
            from mujoco.batch import Batch
            self._batch = Batch(self.model, nsim=self.count, nthread=self._workers)
            if self._terrain_models is not None:
                self._batch.set_models(self._terrain_models)
            self._batch_inputs = {
                name: self._batch.bind(name) for name in ('qpos', 'qvel', 'ctrl', 'xfrc_applied')
            }
            self._batch_outputs = {
                name: self._batch.bind(name) for name in
                ('qpos', 'qvel', 'sensordata', 'actuator_force', 'actuator_velocity',
                 'mocap_pos', 'mocap_quat', 'time')
            }
        elif terrain_changed:
            self._batch.set_models(self.model)
        ids = None if mask is None else mask.numpy()
        self._batch.reset(ids, stage=mujoco.mjtStage.mjSTAGE_NONE)
        rows = slice(None) if ids is None else ids
        for name, values in self._batch_inputs.items():
            values[rows] = getattr(self.sim.data, name).numpy()[rows]
        self._batch.forward(ids)

    def _step_range(self, start, end, controls, forces):
        for i in range(start, end):
            data = self.data[i]
            data.ctrl[:] = controls[i]
            data.xfrc_applied[:] = forces[i] + self.mouse_forces[i]
            mujoco.mj_step(self._model_for(i), data)

    def step(self, action, *, pool=None, workers=1):
        if self.exoskeleton is not None:
            assist = self.exoskeleton
            strut_actions = None
            if assist.policy_control:
                width = len(assist.actuators)
                action, strut_actions = action[:, :-width], action[:, -width:]
            self.sim.data.ctrl[:, assist.actuators] = assist.controls(strut_actions)
        for substep in range(2):
            self.device.set_substep_commands(self.sim, {'wangyu/action':action})
            controls = self.sim.data.ctrl.numpy()
            forces = self.sim.data.xfrc_applied.numpy()
            if self._batch is not None:
                self._batch_inputs['ctrl'][:] = controls
                np.add(forces, self.mouse_forces, out=self._batch_inputs['xfrc_applied'])
                # Match mj_forward after the second substep, including force sensors.
                stage = (mujoco.mjtStage.mjSTAGE_ACC if substep == 1
                         else mujoco.mjtStage.mjSTAGE_NONE)
                self._batch.step(stage=stage)
            elif pool is None:
                self._step_range(0, self.count, controls, forces)
            else:
                # One read-only model, disjoint MjData, and a barrier between substeps.
                chunk = (self.count + workers - 1) // workers
                jobs = [pool.submit(self._step_range, start, min(start + chunk, self.count), controls, forces)
                        for start in range(0, self.count, chunk)]
                for job in jobs:
                    job.result()
        self.device.on_step(self.sim)
        if self.data is not None:
            for i, data in enumerate(self.data):
                mujoco.mj_forward(self._model_for(i), data)
        self.read_native()
        if self.exoskeleton is not None:
            self.exoskeleton.observe(2. * self.model.opt.timestep)
        return self.device.get_measurements(self.sim)
