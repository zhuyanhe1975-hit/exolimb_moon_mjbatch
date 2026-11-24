"""Device-resident MuJoCo Warp rollout for Wangyu-Replant-v12 (DeepMimic tracking + hip/knee/ankle peak-torque & EVA ROM polish)."""
import os

import mujoco_warp as mjw
from mujoco_warp._src import smooth as mjw_smooth
import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from wangyu_replant import (CONTACT_GEOMS, DT, EPISODE_STEPS, GEOMETRY,
                            LOOKAHEAD_STEPS, MECHANISM, RESIDUAL_ARM_RAD,
                            RESIDUAL_LEG_RAD, RESIDUAL_STRUT_RAD,
                            RESIDUAL_STRUT_ROLL_RAD, RESIDUAL_STRUT_SLIDE_M,
                            TASK, Recovery, adapt_reference_to_model,
                            load_replant_reference)


@wp.kernel
def _support_polygon(slots: wp.array4d[float], com: wp.array2d[float],
                     candidate: wp.array[bool], inside: wp.array[bool]):
    """Test every supporting line of the loaded contacts, with CPU's 1 mm tolerance."""
    w = wp.tid()
    inside[w] = False
    if not candidate[w]:
        return
    noncollinear = bool(False)
    for i in range(16):
        if slots[w, i // 4, i % 4, 0] == 0.0 or wp.abs(slots[w, i // 4, i % 4, 1]) <= 1.0:
            continue
        x = slots[w, i // 4, i % 4, 4]
        y = slots[w, i // 4, i % 4, 5]
        for j in range(i + 1, 16):
            if slots[w, j // 4, j % 4, 0] == 0.0 or wp.abs(slots[w, j // 4, j % 4, 1]) <= 1.0:
                continue
            dx = slots[w, j // 4, j % 4, 4] - x
            dy = slots[w, j // 4, j % 4, 5] - y
            length = wp.sqrt(dx * dx + dy * dy)
            if length < 1.e-8:
                continue
            low = float(0)
            high = float(0)
            for k in range(16):
                if slots[w, k // 4, k % 4, 0] > 0.0 and wp.abs(slots[w, k // 4, k % 4, 1]) > 1.0:
                    cross = dx * (slots[w, k // 4, k % 4, 5] - y) - dy * (slots[w, k // 4, k % 4, 4] - x)
                    low = wp.min(low, cross)
                    high = wp.max(high, cross)
            eps = 1.e-7 * length
            if high > eps or low < -eps:
                noncollinear = True
            signed = dx * (com[w, 1] - y) - dy * (com[w, 0] - x)
            if (low >= -eps and signed < -.001 * length) or (high <= eps and signed > .001 * length):
                return
    inside[w] = noncollinear


def quat_rot_error_torch(q_cur: torch.Tensor, q_ref: torch.Tensor):
    """Vectorized PyTorch mirror of quat_rot_error_np (wxyz -> signed 3D rotvec & angle)."""
    w0, x0, y0, z0 = q_cur[:, 0], -q_cur[:, 1], -q_cur[:, 2], -q_cur[:, 3]
    w1, x1, y1, z1 = q_ref[:, 0], q_ref[:, 1], q_ref[:, 2], q_ref[:, 3]
    ew = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    ex = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    ey = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    ez = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    sign = torch.where(ew < 0., -1., 1.)
    ew = ew * sign
    vec = torch.stack((ex, ey, ez), dim=-1) * sign[:, None]
    sin_half = vec.norm(dim=-1)
    angle = 2.0 * torch.atan2(sin_half, ew.clamp_min(1e-9))
    scale = torch.where(sin_half > 1e-7, angle / sin_half.clamp_min(1e-7), torch.full_like(angle, 2.0))
    return vec * scale[:, None], angle


def mimic_reward_torch(q_act, q_ref, qd_act, qd_ref,
                       root_pos, root_pos_ref, root_rot_angle, root_vel, root_vel_ref,
                       key_pos, key_pos_ref, load_bw, tip_contact_exp,
                       act_rate_sq, leg_tau_rms_sq, leg_tau_peak_sq, rom_excess_sq,
                       hoof_slip_sq=0.0):
    """Vectorized PyTorch mirror of canonical DeepMimic mimic_reward_np in [0, 1]."""
    pose_err = (q_act - q_ref).square().mean(-1)
    r_pose = torch.exp(-pose_err / 0.04)

    vel_err = (qd_act - qd_ref).square().mean(-1)
    r_vel = torch.exp(-vel_err / 2.0)

    rpos_err = (root_pos - root_pos_ref).square().sum(-1)
    rvel_err = ((root_vel[:, :3] - root_vel_ref[:, :3]).square().sum(-1)
                + 0.25 * (root_vel[:, 3:6] - root_vel_ref[:, 3:6]).square().sum(-1))
    r_root = torch.exp(-(rpos_err / 0.04 + root_rot_angle.square() / 0.08 + rvel_err / 1.5))

    key_err = (key_pos - key_pos_ref).square().mean(dim=(1, 2))
    r_key = torch.exp(-key_err / 0.012)

    leg_grounded = (load_bw[:, :4].amax(-1) >= 0.02).float()
    tip_bw = load_bw[:, 4:]
    in_c = 0.5 * ((tip_bw[:, 0] / 0.06).clamp(0., 1.) + (tip_bw[:, 1] / 0.06).clamp(0., 1.))
    in_s = torch.exp(-(tip_bw[:, 0] + tip_bw[:, 1]) / 0.05)
    tip_match = torch.where(tip_contact_exp, in_c, in_s)
    r_contact = 0.65 * leg_grounded + 0.35 * tip_match

    r_track = 0.35 * r_pose + 0.10 * r_vel + 0.25 * r_root + 0.22 * r_key + 0.08 * r_contact
    reg = (0.03 * act_rate_sq + 0.06 * leg_tau_rms_sq + 0.06 * leg_tau_peak_sq
           + 0.40 * rom_excess_sq + 0.08 * hoof_slip_sq).clamp_max(0.18)
    return (r_track - reg).clamp_min(0.0)


class WarpReplantEnv(VecEnv):
    max_episode_length = EPISODE_STEPS

    def __init__(self, model, count, *, reference=None, rsi_prob=0.5, terminate_on_ref_div=True):
        self.device = torch.device('cuda', torch.cuda.current_device())
        raw_ref = reference if reference is not None else load_replant_reference()
        self.ref = adapt_reference_to_model(model, raw_ref)
        self.n_ref = int(self.ref['n_frames'])
        self.max_rsi = max(1, self.n_ref - 30)
        self.rsi_prob = float(rsi_prob)
        self.terminate_on_ref_div = bool(terminate_on_ref_div)
        self.num_envs, self.num_actions = count, model.nu
        self.cfg = dict(
            task=TASK, geometry=GEOMETRY, exo_mechanism=MECHANISM,
            physics_backend='mujoco_warp', device=str(self.device),
            reference_revision=self.ref['revision'],
            reference_sha256=self.ref['sha256'],
            rsi_prob=self.rsi_prob,
            residual_action=True,
            lookahead_steps=LOOKAHEAD_STEPS,
            residual_leg_rad=RESIDUAL_LEG_RAD,
            residual_arm_rad=RESIDUAL_ARM_RAD,
            residual_strut_rad=RESIDUAL_STRUT_RAD,
            residual_strut_roll_rad=RESIDUAL_STRUT_ROLL_RAD,
            residual_strut_slide_m=RESIDUAL_STRUT_SLIDE_M,
            terminate_on_ref_div=self.terminate_on_ref_div,
        )
        initial = Recovery(model, reference=self.ref, terminate_on_ref_div=self.terminate_on_ref_div)
        def upload(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=self.device)

        self.root, self.torso = initial.root, initial.torso
        self.soles = upload(initial.support[:2], dtype=torch.long)
        self.shins = upload(initial.support[2:4], dtype=torch.long)
        self.tips = upload(initial.tips, dtype=torch.long)
        self.tip_contacts = upload(initial.tip_contacts, dtype=torch.long)
        self.key_geoms = upload(initial.key_geoms, dtype=torch.long)
        self.angmom_adr = initial.angmom_adr
        self.weight = float(initial.weight)

        self.act_qadr = upload(initial.addr, dtype=torch.long)
        self.act_dofadr = upload(initial.dof_addr, dtype=torch.long)
        self.lower_aids = upload(initial.lower_aids, dtype=torch.long)
        self.lower_qadr = upload(initial.lower_qadr, dtype=torch.long)
        self.eva_soft_lo = upload(initial.eva_soft_lo)
        self.eva_soft_hi = upload(initial.eva_soft_hi)
        self.lo, self.hi = upload(initial.limits).T
        self.residual_scale = upload(initial.residual_scale)
        self.tau_max = upload(initial.tau_max)
        self.lower_tau_max = self.tau_max[self.lower_aids]

        self.ref_qpos = upload(self.ref['qpos'])
        ref_qvel = self.ref['qvel'].copy()
        ref_qvel[0] = 0.
        self.ref_qvel = upload(ref_qvel)
        self.ref_qd_act = upload(self.ref['qvel'][:, initial.dof_addr])
        self.ref_ctrl = upload(self.ref['ctrl'])
        self.ref_tips = upload(self.ref['tip_targets'])
        self.ref_keys = upload(self.ref['key_targets'])
        self.ref_tip_contact = upload(self.ref['tip_contact_mask'][:, 0], dtype=torch.bool)
        self.ref_phase = upload(self.ref['phase'], dtype=torch.long)

        anchor_table = np.zeros((self.n_ref, 2, 2), dtype=np.float32)
        for k in range(self.n_ref):
            ph = int(self.ref['phase'][k])
            if ph == 0:
                anchor_table[k] = self.ref['tip_targets'][k, :, :2]
            elif ph in (1, 2):
                anchor_table[k] = initial.anchor_prone_pair
            else:
                anchor_table[k] = initial.anchor_stand_pair
        self.ref_anchor = upload(anchor_table)

        k_all = torch.arange(self.n_ref, device=self.device, dtype=torch.long)
        k_next = (k_all + 1).clamp_max(self.n_ref - 1)
        k_look = (k_all + LOOKAHEAD_STEPS).clamp_max(self.n_ref - 1)
        rpos_err0 = self.ref_qpos[k_next, :3] - self.ref_qpos[k_all, :3]
        rrot_err0, _ = quat_rot_error_torch(self.ref_qpos[k_all, 3:7], self.ref_qpos[k_next, 3:7])
        rvel_err0 = self.ref_qvel[k_next, :6] - self.ref_qvel[k_all, :6]
        rpos_fut0 = self.ref_qpos[k_look, :3] - self.ref_qpos[k_all, :3]
        rrot_fut0, _ = quat_rot_error_torch(self.ref_qpos[k_all, 3:7], self.ref_qpos[k_look, 3:7])
        q_err_now0 = self.ref_ctrl[k_next] - self.ref_ctrl[k_all]
        q_err_fut0 = self.ref_ctrl[k_look] - self.ref_ctrl[k_all]
        tip_err0 = (self.ref_tips[k_next] - self.ref_tips[k_all]).flatten(1)
        zero_load = torch.zeros(self.n_ref, 6, device=self.device)
        zero_act = torch.zeros(self.n_ref, model.nu, device=self.device)
        prog0 = (k_all.float() / max(self.n_ref - 1, 1))[:, None]
        self.ref_obs = torch.cat((
            self.ref_qpos[:, 2:7],
            self.ref_qvel[:, :6],
            self.ref_ctrl,
            self.ref_qvel[:, self.act_dofadr],
            zero_load,
            zero_act,
            rpos_err0, rrot_err0, rvel_err0,
            rpos_fut0, rrot_fut0,
            q_err_now0, q_err_fut0, tip_err0, prog0,
        ), dim=-1)

        init_k = torch.zeros(count, dtype=torch.long, device=self.device)
        if self.rsi_prob > 0. and self.max_rsi > 1:
            n_rsi = int(round(count * self.rsi_prob))
            if n_rsi > 0:
                init_k[count - n_rsi:] = torch.randint(1, self.max_rsi, (n_rsi,), device=self.device)
        self.ref_idx = init_k.clone()
        self.from_prone = (init_k == 0)
        self.phase = self.ref_phase[init_k].clone()
        self.hold = torch.zeros(count, dtype=torch.long, device=self.device)
        self.episode_length_buf = torch.zeros(count, dtype=torch.long, device=self.device)
        import mujoco as _mj
        ref_hoof_xy = np.zeros((self.n_ref, 2, 2), dtype=np.float32)
        d_kin = _mj.MjData(model)
        tip_contact_cpu = list(initial.tip_contacts)
        for k in range(self.n_ref):
            d_kin.qpos[:] = self.ref['qpos'][k]
            _mj.mj_kinematics(model, d_kin)
            ref_hoof_xy[k] = d_kin.geom_xpos[tip_contact_cpu, :2]
        self.ref_hoof_xy = upload(ref_hoof_xy)

        self.anchor = self.ref_anchor[init_k].clone()
        self.prev_hoof_xy = self.ref_hoof_xy[init_k].clone()
        self.prev_action = torch.zeros(count, model.nu, device=self.device)
        self.inside = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.done = torch.ones_like(self.inside)
        self.obs = self.ref_obs[init_k].clone()
        self._step_counter = 0
        self._log_stride = max(1, int(os.environ.get('MOON_REPLANT_LOG_STRIDE', '32')))

        for i, name in enumerate(CONTACT_GEOMS):
            sensor = model.sensor(f'replant/{name}')
            if int(sensor.adr[0]) != i * 28 or int(sensor.dim[0]) != 28:
                raise ValueError('Unexpected replant contact sensor layout')
        ang_s = model.sensor('replant/angmom')
        if int(ang_s.adr[0]) != 168 or int(ang_s.dim[0]) != 3:
            raise ValueError('Unexpected replant/angmom sensor layout')

        wp.init()
        self.warp_device = wp.get_device(str(self.device))
        self.mask = wp.from_torch(self.done, dtype=wp.bool)
        self.graphs = os.environ.get('MOON_WARP_GRAPH', '1') == '1'
        nconmax = int(os.environ.get('MOON_REPLANT_NCONMAX', '48'))
        njmax = int(os.environ.get('MOON_REPLANT_NJMAX', '160'))
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(self.stream()):
            self.model = mjw.put_model(model)
            if os.environ.get('MOON_WARP_MESH_NOGRAPH', '1') == '1' and model.nmesh > 0:
                small = torch.as_tensor(model.mesh_vertnum <= 64, device=self.device)
                graphadr = wp.to_torch(self.model.mesh_graphadr)
                graphadr[small] = -1
            self.data = mjw.put_data(model, initial.d, nworld=count, nconmax=nconmax, njmax=njmax)
            for name in ('qpos', 'qvel', 'ctrl', 'actuator_force',
                         'geom_xpos', 'subtree_com', 'xmat', 'xpos', 'sensordata'):
                setattr(self, name, wp.to_torch(getattr(self.data, name)))
            self.qpos.copy_(self.ref_qpos[init_k])
            self.qvel.copy_(self.ref_qvel[init_k])
            self.ctrl.copy_(self.ref_ctrl[init_k])
            mjw.forward(self.model, self.data)
            if self.graphs:
                stream = wp.Stream(self.warp_device)
                stream.wait_stream(self.stream())
                with wp.ScopedStream(stream):
                    with wp.ScopedCapture() as physics:
                        self.physics()
                    with wp.ScopedCapture() as reset:
                        mjw.reset_data(self.model, self.data, self.mask)
                self.stream().wait_stream(stream)
                self.physics_graph, self.reset_graph = physics.graph, reset.graph

    def stream(self):
        return wp.stream_from_torch(torch.cuda.current_stream(self.device))

    def physics(self):
        # 2 physics substeps (120 Hz -> 60 Hz) + lightweight kinematics/COM refresh only.
        # Avoids a redundant 3rd full collision + constraint + Newton solver pass (~-35% physics cost).
        mjw.step(self.model, self.data)
        mjw.step(self.model, self.data)
        mjw_smooth.kinematics(self.model, self.data)
        mjw_smooth.com_pos(self.model, self.data)

    def get_observations(self):
        return TensorDict({'policy': self.obs}, batch_size=[self.num_envs])

    def step(self, actions):
        clipped = actions.clamp(-1., 1.)
        next_ref = (self.ref_idx + 1).clamp_max(self.n_ref - 1)
        target_ctrl = (self.ref_ctrl[next_ref] + clipped * self.residual_scale).clamp(self.lo, self.hi)
        self.ctrl.copy_(target_ctrl)
        with wp.ScopedStream(self.stream()):
            if self.graphs:
                wp.capture_launch(self.physics_graph)
            else:
                self.physics()
        self.episode_length_buf += 1
        self.ref_idx = next_ref

        slots = self.sensordata[:, :168].reshape(self.num_envs, 6, 4, 7)
        angmom = self.sensordata[:, self.angmom_adr:self.angmom_adr + 3]
        load = (slots[..., 1].abs() * (slots[..., 0] > 0)).sum(-1)
        load_bw = load / self.weight
        tips = self.geom_xpos[:, self.tips]
        soles = self.geom_xpos[:, self.soles]
        key_pos = self.geom_xpos[:, self.key_geoms]
        com = self.subtree_com[:, self.root, :2]
        span = tips[:, 1, :2] - tips[:, 0, :2]
        fraction = ((com - tips[:, 0, :2]) * span).sum(-1) / span.square().sum(-1).clamp_min(1e-9)
        tip_line_dist = (com - tips[:, 0, :2] - fraction.clamp(0, 1)[:, None] * span).norm(dim=-1)
        near = tip_line_dist <= .15
        loaded = load[:, 4:].amin(-1) >= 10.
        upright = self.xmat[:, self.torso, 2, 2]
        height = self.xpos[:, self.root, 2]
        speed = self.qvel[:, :6].norm(dim=-1)
        foot_mid = .5 * (soles[:, 0, :2] + soles[:, 1, :2])
        shins = self.geom_xpos[:, self.shins, :2]
        shin_mid = .5 * (shins[:, 0] + shins[:, 1])
        leg_xy_dist = (com - foot_mid).norm(dim=-1)
        shin_xy_dist = (com - shin_mid).norm(dim=-1)
        tip_shift = (tips[:, :, :2] - self.anchor).norm(dim=-1).amin(-1)
        both_feet = load[:, :2].amin(-1) >= 10.
        both_shins = load[:, 2:4].amin(-1) >= 10.
        sole_dx = (soles[:, 0, 0] - soles[:, 1, 0]).abs()

        base_p0 = ((self.phase == 0) & (upright > .72) & (height > .36)
                   & (load[:, :4].sum(-1) > .55 * self.weight) & (speed < .45))
        direct_ready = base_p0 & ((both_feet & (leg_xy_dist <= .25)) | (both_shins & (shin_xy_dist <= .22)))
        kneel_cand = base_p0 & ~direct_ready & ((load[:, :4] > 5.).sum(-1) >= 3)
        wp.launch(_support_polygon, self.num_envs,
                  inputs=[wp.from_torch(slots), wp.from_torch(com), wp.from_torch(kneel_cand),
                          wp.from_torch(self.inside)], device=self.warp_device, stream=self.stream())
        p0_stable = direct_ready | (kneel_cand & self.inside)
        standing = (loaded & near
                    & (load[:, :2].amin(-1) >= .20 * self.weight)
                    & (load[:, :2].sum(-1) >= .65 * self.weight)
                    & ((load[:, 0] - load[:, 1]).abs() <= .30 * self.weight)
                    & (sole_dx <= .12)
                    & (load[:, 2:4].amax(-1) < 5.)
                    & (height > .74) & (upright > .94) & (speed < .30))
        self.hold = torch.where(p0_stable | ((self.phase == 3) & standing), self.hold + 1, 0)
        knelt = (self.phase == 0) & (self.hold >= 10)
        lift = (self.phase == 1) & (load[:, 4:].amax(-1) < 1.) & (tips[:, :, 2].amin(-1) > .06)
        replant = (self.phase == 2) & loaded & near & (tip_shift >= .12)
        update_anchor = ((self.phase == 0) & loaded) | ((self.phase == 1) & loaded)
        self.anchor = torch.where(update_anchor[:, None, None], tips[:, :, :2], self.anchor)
        advanced = knelt | lift | replant
        self.phase += advanced.long()
        self.hold = torch.where(advanced, 0, self.hold)
        success = (self.phase == 3) & (self.hold >= 60)

        ref_k = self.ref_idx
        q_act = self.qpos[:, self.act_qadr]
        qd_act = self.qvel[:, self.act_dofadr]
        q_ref = self.ref_ctrl[ref_k]
        qd_ref = self.ref_qd_act[ref_k]
        _, root_rot_angle = quat_rot_error_torch(self.qpos[:, 3:7], self.ref_qpos[ref_k, 3:7])
        act_rate_sq = (clipped - self.prev_action).square().mean(-1)
        self.prev_action.copy_(clipped)
        lower_tau = self.actuator_force[:, self.lower_aids]
        leg_tau_norm_sq = (lower_tau / self.lower_tau_max).square()
        leg_tau_rms_sq = leg_tau_norm_sq.mean(-1)
        leg_tau_peak_sq = leg_tau_norm_sq.amax(-1)
        q_lower = self.qpos[:, self.lower_qadr]
        rom_exc = (q_lower - self.eva_soft_hi).clamp_min(0.) + (self.eva_soft_lo - q_lower).clamp_min(0.)
        rom_excess_sq = rom_exc.square().mean(-1)
        hoof_xy = self.geom_xpos[:, self.tip_contacts, :2]
        hoof_speed = (hoof_xy - self.prev_hoof_xy).norm(dim=-1) / DT
        hoof_gate = (load_bw[:, 4:] / 0.03).clamp(0., 1.)
        hoof_slip_mps = (hoof_gate * hoof_speed).mean(-1)
        hoof_slip_sq = (hoof_gate * (hoof_speed / 0.15).square()).mean(-1)
        reward = mimic_reward_torch(
            q_act, q_ref, qd_act, qd_ref,
            self.qpos[:, :3], self.ref_qpos[ref_k, :3],
            root_rot_angle, self.qvel[:, :6], self.ref_qvel[ref_k, :6],
            key_pos, self.ref_keys[ref_k],
            load_bw, self.ref_tip_contact[ref_k],
            act_rate_sq, leg_tau_rms_sq, leg_tau_peak_sq, rom_excess_sq,
            hoof_slip_sq,
        )

        finite = self.qpos.isfinite().all(-1) & self.qvel.isfinite().all(-1)
        root_pos_err = (self.qpos[:, :3] - self.ref_qpos[ref_k, :3]).norm(dim=-1)
        key_err_mean = (key_pos - self.ref_keys[ref_k]).norm(dim=-1).mean(-1)
        after_grace = self.episode_length_buf >= 15
        if self.terminate_on_ref_div:
            failed = ~finite | (after_grace & ((root_pos_err > .35) | (root_rot_angle > .75) | (key_err_mean > .30)))
        else:
            failed = ~finite | (self.qpos[:, :2].norm(dim=-1) > 3.) | (after_grace & (upright < -.20))
        reward = torch.where(finite, reward, 0.)
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failed
        done = failed | timeout

        self._step_counter += 1
        if (self._step_counter % self._log_stride) == 0:
            prone_mask = self.from_prone
            prone_done = prone_mask & done
            prone_denom = prone_done.float().sum().clamp_min(1.)
            prone_cnt = prone_mask.float().sum().clamp_min(1.)
            human_q_rmse = (q_act[:, :23] - q_ref[:, :23]).square().mean(-1).sqrt()
            tip_rmse = (tips - self.ref_tips[ref_k]).square().mean(dim=(1, 2)).sqrt()
            all_airborne = load_bw[:, :4].amax(-1) < .02
            lower_abs = lower_tau.abs()
            log = {
                'Replant/phase': self.phase.float().mean(),
                'Replant/max_phase': self.phase.float().max(),
                'Replant/upright': upright.mean(),
                'Replant/height': height.mean(),
                'Replant/root_rot_deg': (root_rot_angle * (180. / np.pi)).mean(),
                'Replant/root_pos_err': root_pos_err.mean(),
                'Replant/key_err_mean': key_err_mean.mean(),
                'Replant/human_q_rmse': human_q_rmse.mean(),
                'Replant/tip_rmse': tip_rmse.mean(),
                'Replant/sole_dx': sole_dx.mean(),
                'Replant/sole_load_bw': load_bw[:, :2].sum(-1).mean(),
                'Replant/shin_load_bw': load_bw[:, 2:4].sum(-1).mean(),
                'Replant/tip_load_bw': load_bw[:, 4:].sum(-1).mean(),
                'Replant/hoof_slip_mps': hoof_slip_mps.mean(),
                'Replant/airborne_frac': all_airborne.float().mean(),
                'Replant/angmom_nms': angmom.norm(dim=-1).mean(),
                'Replant/hip_peak_nm': lower_abs[:, 0:2].amax(-1).mean(),
                'Replant/knee_peak_nm': lower_abs[:, 2:4].amax(-1).mean(),
                'Replant/ankle_peak_nm': lower_abs[:, 4:6].amax(-1).mean(),
                'Replant/eva_rom_excess_deg': (rom_exc.amax(-1) * (180. / np.pi)).mean(),
                'Replant/success_rate': success.float().mean(),
                'Replant/prone_phase': (self.phase.float() * prone_mask).sum() / prone_cnt,
                'Replant/prone_height': (height * prone_mask).sum() / prone_cnt,
                'Replant/prone_upright': (upright * prone_mask).sum() / prone_cnt,
                'Replant/prone_success_rate': (success & prone_mask).float().sum() / prone_denom,
            }
        else:
            log = {}

        k_next = (ref_k + 1).clamp_max(self.n_ref - 1)
        k_look = (ref_k + LOOKAHEAD_STEPS).clamp_max(self.n_ref - 1)
        rpos_err = self.ref_qpos[k_next, :3] - self.qpos[:, :3]
        rrot_err, _ = quat_rot_error_torch(self.qpos[:, 3:7], self.ref_qpos[k_next, 3:7])
        rvel_err = self.ref_qvel[k_next, :6] - self.qvel[:, :6]
        rpos_fut = self.ref_qpos[k_look, :3] - self.qpos[:, :3]
        rrot_fut, _ = quat_rot_error_torch(self.qpos[:, 3:7], self.ref_qpos[k_look, 3:7])
        q_err_now = self.ref_ctrl[k_next] - q_act
        q_err_fut = self.ref_ctrl[k_look] - q_act
        tip_err_vec = (self.ref_tips[k_next] - tips).flatten(1)
        prog = (ref_k.float() / max(self.n_ref - 1, 1))[:, None]
        obs = torch.cat((
            self.qpos[:, 2:7],
            self.qvel[:, :6],
            q_act, qd_act,
            load_bw,
            self.prev_action,
            rpos_err, rrot_err, rvel_err,
            rpos_fut, rrot_fut,
            q_err_now, q_err_fut, tip_err_vec, prog,
        ), dim=-1)

        if self.rsi_prob > 0. and self.max_rsi > 1:
            use_rsi = torch.rand(self.num_envs, device=self.device) < self.rsi_prob
            rand_k = torch.randint(1, self.max_rsi, (self.num_envs,), device=self.device)
            next_k = torch.where(use_rsi, rand_k, 0)
        else:
            next_k = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.obs = torch.where(done[:, None], self.ref_obs[next_k], obs)
        self.done.copy_(done)
        with wp.ScopedStream(self.stream()):
            if self.graphs:
                wp.capture_launch(self.reset_graph)
            else:
                mjw.reset_data(self.model, self.data, self.mask)
        self.qpos.copy_(torch.where(done[:, None], self.ref_qpos[next_k], self.qpos))
        self.qvel.copy_(torch.where(done[:, None], self.ref_qvel[next_k], self.qvel))
        self.ctrl.copy_(torch.where(done[:, None], self.ref_ctrl[next_k], self.ctrl))
        self.ref_idx = torch.where(done, next_k, self.ref_idx)
        self.from_prone = torch.where(done, next_k == 0, self.from_prone)
        self.phase = torch.where(done, self.ref_phase[next_k], self.phase)
        self.hold.masked_fill_(done, 0)
        self.anchor = torch.where(done[:, None, None], self.ref_anchor[next_k], self.anchor)
        self.prev_hoof_xy = torch.where(done[:, None, None], self.ref_hoof_xy[next_k], hoof_xy)
        self.prev_action.masked_fill_(done[:, None], 0.)
        self.episode_length_buf.masked_fill_(done, 0)
        return self.get_observations(), reward, done, {'time_outs': timeout, 'log': log}
