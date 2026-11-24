"""Batched MuJoCo Warp worlds for the external-assist jump task."""
import os
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
from mujoco_warp._src import constraint as mjw_con
from mujoco_warp._src import forward as mjw_fwd
from mujoco_warp._src import sensor as mjw_sensor
from mujoco_warp._src import smooth as mjw_smooth
from mujoco_warp._src import solver as mjw_solver
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
        # Interleave terrain patches across env IDs so any subset of terminating worlds
        # spans all patches uniformly instead of clustering high-index tiles at the tail.
        row_ids = np.arange(count, dtype=np.int64) % len(patches)
        self.patches = torch.as_tensor(patches[row_ids], device=device)
        self.rows = torch.as_tensor(row_ids, device=device)
        self.counts = torch.as_tensor(np.asarray(counts, dtype=np.int64)[row_ids], device=device)
        self.candidates = torch.as_tensor(candidates, device=device)
        self.headings = torch.as_tensor(table.headings, device=device, dtype=torch.float32)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        # Precompute & validate exact whole-sole spawn z for every (patch, candidate, heading)
        # once at startup so per-step GPU resets need zero 208-point terrain queries or CPU syncs.
        foot = ground.sole_points  # (S, 3)
        n_rows, max_c = candidates.shape[:2]
        n_head = len(self.headings)
        c_h = self.headings.cos()  # (H,)
        s_h = self.headings.sin()  # (H,)
        z_table = torch.zeros((n_rows, max_c, n_head), device=device, dtype=torch.float32)
        orig_patch_ids = ground.patch_ids
        for row, patch in enumerate(patches):
            cnt = counts[row]
            xy_r = self.candidates[row, :cnt, :2]  # (C, 2)
            pad_r = self.candidates[row, :cnt, 2]  # (C,)
            # Rotate sole points for all headings: (H, S, 2)
            rot_x = c_h[:, None] * foot[None, :, 0] - s_h[:, None] * foot[None, :, 1]
            rot_y = s_h[:, None] * foot[None, :, 0] + c_h[:, None] * foot[None, :, 1]
            # (C, H, S, 2) -> flatten to (C*H, S, 2)
            pts_x = xy_r[:, None, None, 0] + rot_x[None, :, :]
            pts_y = xy_r[:, None, None, 1] + rot_y[None, :, :]
            pts = torch.stack((pts_x, pts_y), dim=-1).reshape(cnt * n_head, foot.shape[0], 2)
            ground.patch_ids = torch.full((cnt * n_head,), int(patch), dtype=torch.long, device=device)
            heights = ground.height(pts)  # (C*H, S)
            z_ch = (heights - foot[None, :, 2]).max(-1).values + .015 + pad_r[:, None].expand(cnt, n_head).reshape(-1)
            gaps = z_ch[:, None] + foot[None, :, 2] - heights
            valid = (torch.isfinite(gaps).all(-1)
                     & (pts.abs() <= ground.half - 2.25).all((1, 2))
                     & (gaps.min(-1).values >= .0145)
                     & (gaps.max(-1).values <= .20))
            if not bool(valid.all()):
                raise ValueError(f'Invalid whole-sole terrain spawn in WarpSpawnTable patch {patch}')
            z_table[row, :cnt] = z_ch.reshape(cnt, n_head)
        ground.patch_ids = orig_patch_ids
        self.candidates_z = z_table

    def sample(self):
        device = self.rows.device
        index = (torch.rand(self.rows.shape, device=device, generator=self.generator) * self.counts).long()
        heading = torch.randint(len(self.headings), self.rows.shape, device=device, generator=self.generator)
        samples = self.candidates[self.rows, index]
        z = self.candidates_z[self.rows, index, heading]
        return samples[:, :2], self.headings[heading], samples[:, 2], z


class WarpExoWorlds:
    def __init__(self, spec, count, terrain_idx, *, pushes, terrain_indices, seed, assist_force,
                 terrain_size=16, hinge_torque=None):
        self.model = spec.compile()
        self.count = count
        self.terrain_size = terrain_size
        self._vertical_columns = vertical_load_columns(self.model)
        self.cuda_device = torch.device('cuda', torch.cuda.current_device())
        self.use_cuda_graphs = os.environ.get('MOON_WARP_GRAPH', '1') == '1'
        self.use_step2_reuse = os.environ.get('MOON_WARP_STEP2_REUSE', '1') == '1'
        self.use_mesh_nograph = os.environ.get('MOON_WARP_MESH_NOGRAPH', '1') == '1'
        wp.init()
        nconmax = int(os.environ.get('MOON_WARP_NCONMAX', '32'))
        has_multi_sole = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'robot/left_sole_box_0') >= 0
        default_njmax = '160' if has_multi_sole else '128'
        njmax = int(os.environ.get('MOON_WARP_NJMAX', default_njmax))
        contact_maxmatch = int(os.environ.get('MOON_WARP_CONTACT_MAXMATCH', '32'))
        with wp.ScopedStream(self._stream()):
            self._model = mjw.put_model(self.model, batch_sizes={'geom_pos': count, 'geom_dataid': count})
            if self.use_mesh_nograph and self.model.nmesh > 0:
                # For small convex hulls (<=64 verts, e.g. 32-vert bodyfit_v2_hull), a coalesced
                # L1-cached linear scan in collision_gjk.support() avoids warp-divergent
                # mesh_graph pointer chasing (~22% faster physics with exact support).
                small = torch.as_tensor(self.model.mesh_vertnum <= 64, device=self.cuda_device)
                graphadr = wp.to_torch(self._model.mesh_graphadr)
                graphadr[small] = -1
            if self._vertical_columns is not None:
                self._model.opt.contact_sensor_maxmatch = contact_maxmatch
            self._data = mjw.put_data(self.model, mujoco.MjData(self.model), nworld=count,
                                      nconmax=nconmax, njmax=njmax)
            self._reset_mask = torch.empty(count, dtype=torch.bool, device=self.cuda_device)
            self._warp_reset_mask = wp.from_torch(self._reset_mask, dtype=wp.bool)
            self._nacon_torch = wp.to_torch(self._data.nacon)
            self._saved_nacon = torch.zeros_like(self._nacon_torch)
        self._use_implicit = self.model.opt.integrator in (
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST, mujoco.mjtIntegrator.mjINT_IMPLICIT)
        if self.use_cuda_graphs:
            # The Torch default stream cannot begin capture. Use a Warp-owned stream
            # for capture, then replay in order on the caller's Torch stream.
            self._capture_stream = wp.Stream(wp.get_device(str(self.cuda_device)))
            with wp.ScopedStream(self._capture_stream):
                with wp.ScopedCapture() as step_capture:
                    mjw.step(self._model, self._data)
                with wp.ScopedCapture() as step1_reuse_capture:
                    self._run_substep1_reuse()
                with wp.ScopedCapture() as forward_capture:
                    mjw.forward(self._model, self._data)
                with wp.ScopedCapture() as reset_capture:
                    mjw.reset_data(self._model, self._data, self._warp_reset_mask)
                with wp.ScopedCapture() as reset_kin_capture:
                    self._run_reset_refresh()
                self._step_graph = step_capture.graph
                self._step_substep1_graph = step1_reuse_capture.graph
                self._forward_graph = forward_capture.graph
                self._reset_graph = reset_capture.graph
                self._reset_kin_graph = reset_kin_capture.graph
        model = SimpleNamespace(**{name: wp.to_torch(getattr(self._model, name))
                                   for name in ('geom_pos', 'geom_dataid')})
        data = SimpleNamespace(**{name: wp.to_torch(getattr(self._data, name))
                                  for name in ('qpos', 'qvel', 'ctrl', 'xfrc_applied', 'sensordata',
                                               'actuator_force', 'actuator_velocity', 'time')})
        self.sim = SimpleNamespace(mj_model=self.model, model=model, data=data)
        self.device = MoonDeviceCore(pushes=pushes, resample_terrain=False, terrain_size=terrain_size,
                                     tiled_100m=(terrain_size == 100))
        self.device.initialize(self.sim)
        self.device.use_measurement_graph = os.environ.get(
            'MOON_MEASUREMENT_GRAPH', '1' if self.use_cuda_graphs else '0') == '1'
        self.device.batch_measurement_terrain = os.environ.get(
            'MOON_BATCHED_TERRAIN', '1' if self.use_cuda_graphs else '0') == '1'
        spawn_patches = (tuple(range(1, len(self.device.ground.zmin) + 1))
                         if terrain_size == 100 else (terrain_indices or (terrain_idx,)))
        self.device.device_spawns = WarpSpawnTable(
            self.model, self.device.ground, count, spawn_patches, seed)
        self.exoskeleton = TelescopingAssist(self, assist_force, hinge_torque=hinge_torque)
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

    def _run_substep1_reuse(self):
        # Reuse fwd_position (kinematics, collision, constraints) + fwd_velocity already
        # computed at the end of the previous control step (or reset) since qpos/qvel
        # are unchanged and only ctrl/xfrc_applied updated.
        mjw_fwd.fwd_actuation(self._model, self._data)
        mjw_fwd.fwd_acceleration(self._model, self._data, factorize=True)
        mjw_solver.solve(self._model, self._data)
        if self._use_implicit:
            mjw_fwd.implicit(self._model, self._data)
        else:
            mjw_fwd.euler(self._model, self._data)

    def _run_reset_refresh(self):
        mjw_smooth.kinematics(self._model, self._data)
        mjw_smooth.com_pos(self._model, self._data)
        if self.use_step2_reuse:
            mjw_smooth.crb(self._model, self._data)
            mjw_con.make_constraint(self._model, self._data)
            mjw_smooth.transmission(self._model, self._data)
        mjw_sensor.sensor_pos(self._model, self._data)
        mjw_fwd.fwd_velocity(self._model, self._data)
        mjw_sensor.sensor_vel(self._model, self._data)

    def _refresh_reset_sensors(self, mask):
        # Reset worlds spawn 1.5 cm clear of terrain with qvel=0 and zero contact force;
        # updating kinematics + pos/vel sensors (and crb/constraints/transmission when
        # step2 reuse is active) preserves non-reset worlds' contacts while skipping
        # collision detection and the Newton constraint solver.
        if mask is None:
            self.sim.data.sensordata.zero_()
        else:
            self.sim.data.sensordata.masked_fill_(mask[:, None], 0.)
        if self.use_cuda_graphs:
            wp.capture_launch(self._reset_kin_graph)
        else:
            self._run_reset_refresh()

    def reset(self, mask=None):
        with wp.ScopedStream(self._stream()):
            if mask is None:
                self._reset_mask.fill_(True)
            else:
                if self.use_step2_reuse:
                    # Preserve global active contact count for non-reset worlds when world 0 resets.
                    self._saved_nacon.copy_(self._nacon_torch)
                self._reset_mask.copy_(mask)
            if self.use_cuda_graphs:
                wp.capture_launch(self._reset_graph)
            else:
                mjw.reset_data(self._model, self._data, self._warp_reset_mask)
            if mask is not None and self.use_step2_reuse:
                self._nacon_torch.copy_(self._saved_nacon)
            self.device.reset(self.sim, mask)
            self.exoskeleton.reset_controls(mask)
            self._refresh_reset_sensors(mask)

    def step(self, action, *, pool=None, workers=1):
        assist = self.exoskeleton
        width = len(assist.actuators)
        human, struts = action[:, :-width], action[:, -width:]
        self.sim.data.ctrl[:, assist.actuators] = assist.controls(struts)
        with wp.ScopedStream(self._stream()):
            for substep in range(2):
                self.device.set_substep_commands(self.sim, {'wangyu/action': human})
                if substep == 0 and self.use_step2_reuse:
                    if self.use_cuda_graphs:
                        wp.capture_launch(self._step_substep1_graph)
                    else:
                        self._run_substep1_reuse()
                else:
                    if self.use_cuda_graphs:
                        wp.capture_launch(self._step_graph)
                    else:
                        mjw.step(self._model, self._data)
            self.device.on_step(self.sim)
            self._refresh_sensors()
        assist.observe(2 * self.model.opt.timestep)
        return self.device.get_measurements(self.sim)
