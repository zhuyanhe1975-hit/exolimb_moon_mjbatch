"""Batched MuJoCo Warp worlds for the external-assist jump task."""
import os
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from exoskeleton import TelescopingAssist
from local_lunar_terrain import sole_sampling
from moon_device_core import MoonDeviceCore
from moon_scene import SOLE_TERRAIN_CONTACTS, vertical_load_columns
from playback_spawns import IndependentSpawns


class WarpSpawnTable:
    def __init__(self, model, ground, count, terrain_indices, seed):
        patches = np.asarray(terrain_indices, dtype=np.int64) - 1
        sizes = np.full(len(patches), count // len(patches))
        sizes[:count % len(patches)] += 1
        sole, cover = sole_sampling(model)
        table = IndependentSpawns(ground.meta, patches, sole, cover, seed)
        counts = [len(table.positions[patch][0]) for patch in patches]
        candidates = np.zeros((len(patches), max(counts), 3), dtype=np.float32)
        for row, patch in enumerate(patches):
            xy, padding = table.positions[patch]
            candidates[row, :len(xy), :2] = xy
            candidates[row, :len(xy), 2] = padding
        device = ground.sole_points.device
        self.patches = torch.as_tensor(np.repeat(patches, sizes), device=device)
        self.rows = torch.as_tensor(np.repeat(np.arange(len(patches)), sizes), device=device)
        self.counts = torch.as_tensor(np.repeat(counts, sizes), device=device)
        self.candidates = torch.as_tensor(candidates, device=device)
        self.headings = torch.as_tensor(table.headings, device=device, dtype=torch.float32)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def sample(self):
        device = self.rows.device
        index = (torch.rand(self.rows.shape, device=device, generator=self.generator) * self.counts).long()
        heading = torch.randint(len(self.headings), self.rows.shape, device=device, generator=self.generator)
        samples = self.candidates[self.rows, index]
        return samples[:, :2], self.headings[heading], samples[:, 2]


class WarpExoWorlds:
    def __init__(self, spec, count, terrain_idx, *, pushes, terrain_indices, seed, assist_force):
        self.model = spec.compile()
        self.count = count
        self._vertical_columns = vertical_load_columns(self.model)
        self.cuda_device = torch.device('cuda', torch.cuda.current_device())
        self.use_cuda_graphs = os.environ.get('MOON_WARP_GRAPH', '1') == '1'
        wp.init()
        with wp.ScopedStream(self._stream()):
            self._model = mjw.put_model(self.model, batch_sizes={'geom_pos': count, 'geom_dataid': count})
            if self._vertical_columns is not None:
                self._model.opt.contact_sensor_maxmatch = 128
            self._data = mjw.put_data(self.model, mujoco.MjData(self.model), nworld=count,
                                      nconmax=128, njmax=512)
            self._reset_mask = torch.empty(count, dtype=torch.bool, device=self.cuda_device)
            self._warp_reset_mask = wp.from_torch(self._reset_mask, dtype=wp.bool)
        if self.use_cuda_graphs:
            # The Torch default stream cannot begin capture. Use a Warp-owned stream
            # for capture, then replay in order on the caller's Torch stream.
            self._capture_stream = wp.Stream(wp.get_device(str(self.cuda_device)))
            with wp.ScopedStream(self._capture_stream):
                with wp.ScopedCapture() as step_capture:
                    mjw.step(self._model, self._data)
                with wp.ScopedCapture() as forward_capture:
                    mjw.forward(self._model, self._data)
                with wp.ScopedCapture() as reset_capture:
                    mjw.reset_data(self._model, self._data, self._warp_reset_mask)
                self._step_graph = step_capture.graph
                self._forward_graph = forward_capture.graph
                self._reset_graph = reset_capture.graph
        model = SimpleNamespace(**{name: wp.to_torch(getattr(self._model, name))
                                   for name in ('geom_pos', 'geom_dataid')})
        data = SimpleNamespace(**{name: wp.to_torch(getattr(self._data, name))
                                  for name in ('qpos', 'qvel', 'ctrl', 'xfrc_applied', 'sensordata',
                                               'actuator_force', 'actuator_velocity', 'time')})
        self.sim = SimpleNamespace(mj_model=self.model, model=model, data=data)
        self.device = MoonDeviceCore(pushes=pushes, resample_terrain=False)
        self.device.initialize(self.sim)
        self.device.use_measurement_graph = os.environ.get(
            'MOON_MEASUREMENT_GRAPH', '1' if self.use_cuda_graphs else '0') == '1'
        self.device.batch_measurement_terrain = os.environ.get(
            'MOON_BATCHED_TERRAIN', '1' if self.use_cuda_graphs else '0') == '1'
        self.device.device_spawns = WarpSpawnTable(
            self.model, self.device.ground, count, terrain_indices or (terrain_idx,), seed)
        self.exoskeleton = TelescopingAssist(self, assist_force)
        # Force accounting uses terrain-only contacts, excluding boot-to-boot contacts.
        sensor_columns = []
        load_columns = []
        for side in ('left', 'right'):
            start = int(self.model.sensor(f'contacts/robot/{side}_sole_terrain_load').adr[0])
            sensor_columns.append(slice(start, start + 4 * SOLE_TERRAIN_CONTACTS))
            load_columns.append(int(self.model.sensor(f'robot/balance_{side}_load').adr[0]))
        self._load_sensors = tuple(zip(sensor_columns, load_columns))
        self._load_columns = load_columns
        self.reset()

    def _stream(self):
        return wp.stream_from_torch(torch.cuda.current_stream(self.cuda_device))

    def _refresh_sensors(self):
        if self.use_cuda_graphs:
            wp.capture_launch(self._forward_graph)
        else:
            mjw.forward(self._model, self._data)
        sensors = self.sim.data.sensordata
        if self._vertical_columns is not None:
            sensors[:, self._load_columns] = sensors[:, self._vertical_columns].clamp_min(0.)
            return
        for columns, load in self._load_sensors:
            contacts = sensors[:, columns].reshape(self.count, SOLE_TERRAIN_CONTACTS, 4)
            sensors[:, load] = (contacts[..., 1].abs() * (contacts[..., 0] > 0)).sum(-1)

    def reset(self, mask=None):
        with wp.ScopedStream(self._stream()):
            if mask is None:
                self._reset_mask.fill_(True)
            else:
                self._reset_mask.copy_(mask)
            if self.use_cuda_graphs:
                wp.capture_launch(self._reset_graph)
            else:
                mjw.reset_data(self._model, self._data, self._warp_reset_mask)
            self.device.reset(self.sim, mask)
            self.exoskeleton.reset_controls(mask)
            self._refresh_sensors()

    def step(self, action, *, pool=None, workers=1):
        assist = self.exoskeleton
        width = len(assist.actuators)
        human, struts = action[:, :-width], action[:, -width:]
        self.sim.data.ctrl[:, assist.actuators] = assist.controls(struts)
        with wp.ScopedStream(self._stream()):
            for _ in range(2):
                self.device.set_substep_commands(self.sim, {'wangyu/action': human})
                if self.use_cuda_graphs:
                    wp.capture_launch(self._step_graph)
                else:
                    mjw.step(self._model, self._data)
            self.device.on_step(self.sim)
            self._refresh_sensors()
        assist.observe(2 * self.model.opt.timestep)
        return self.device.get_measurements(self.sim)
