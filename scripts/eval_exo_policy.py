#!/usr/bin/env python3
"""Multi-terrain quantitative benchmark for exoskeleton forward-jump policies.

Evaluates speed tracking (0.0 / 0.5 / 1.0 / 1.5 m/s), assist ON vs OFF energy savings,
bilateral strut symmetry, stability (failure rate), and first-hop startup kinematics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpu_stand_env import CpuExoJumpEnv
from rsl_rl.models import MLPModel


def load_actor_and_env(ckpt_path: Path, terrain_idx: int, assist_force: float, seed: int):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_path = ckpt_path.parent / "config.json"
    full_cfg = json.loads(cfg_path.read_text())
    cfg = full_cfg["env"]
    env = CpuExoJumpEnv(
        num_envs=1,
        terrain_idx=terrain_idx,
        pushes=False,
        seed=seed,
        workers=1,
        geometry=cfg["geometry"],
        assist_force=assist_force,
        exo_mechanism=cfg["exo_mechanism"],
        exo_policy_io=cfg["exo_policy_io"],
        suit_ankle_pitch=bool(cfg.get("ankle_pitch_profile")),
        slope_adaptive_posture=bool(cfg.get("slope_adaptive_posture")),
        suit_major_joints=bool(cfg.get("suit_major_joints_profile")),
        relax_propulsion=bool(cfg.get("propulsion_profile")),
        mass_profile=cfg.get("apollo_mass_profile"),
        posture_balance=True,
        economical_hops=True,
        long_hop_period=cfg.get("long_hop_period_s", 1.7),
        physics_backend="mjbatch",
    )
    obs = env.get_observations()
    actor_cfg = dict(full_cfg["runner"]["actor"])
    actor_cfg.pop("class_name", None)
    actor = MLPModel(obs, {"actor": ["policy"]}, "actor", env.num_actions, **actor_cfg)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return env, actor


def run_episode(env: CpuExoJumpEnv, actor: MLPModel, target_speed: float, seconds: float):
    env.jump_commands.fixed_speed = target_speed
    env.worlds.reset()
    env.reset_task(None)
    env.episode_length_buf.zero_()
    exo = env.worlds.exoskeleton
    exo.positive_work_j.zero_()
    exo.supported_work_j.zero_()
    exo.human_positive_work_j.zero_()
    exo.human_absolute_work_j.zero_()
    exo.human_metabolic_proxy_j.zero_()
    exo.peak_force_n.zero_()
    exo.sample_seconds = 0.0
    obs = env.get_observations()

    steps = int(round(seconds * 60))
    cmd = env.jump_commands
    data = env.worlds.sim.data
    fails = 0
    edge_resets = 0
    max_foot_back_1st = -1.0
    max_tilt_1st = -99.0
    first_ep = True
    speed_sum = 0.0
    cruise_speed_sum = 0.0
    cruise_steps = 0
    ep_steps = 0

    for _ in range(steps):
        with torch.no_grad():
            act = actor(obs)
        obs, _, done, info = env.step(act)
        ep_steps += 1
        com_vel = data.sensordata[0, cmd.com_velocity_xy]
        vx = float((com_vel * cmd.heading[0]).sum())
        speed_sum += vx
        if ep_steps >= 120:
            cruise_speed_sum += vx
            cruise_steps += 1
        if first_ep and not bool(cmd.first_landing_done[0]):
            probes = cmd.sole_probes()
            foot_mid = probes.mean(dim=(1, 2))[0]
            fb = float(-((foot_mid[:2] - cmd.startup_foot_origin[0]) * cmd.heading[0]).sum())
            com_xyz = data.sensordata[0, cmd.com_xy.start:cmd.com_xy.start + 3]
            ahead = float(((com_xyz[:2] - foot_mid[:2]) * cmd.heading[0]).sum())
            tilt = math.degrees(math.atan2(ahead, max(float(com_xyz[2] - foot_mid[2]), 0.25)))
            max_foot_back_1st = max(max_foot_back_1st, fb)
            max_tilt_1st = max(max_tilt_1st, tilt)
        if bool(done[0]):
            if not bool(info["time_outs"][0]):
                fails += 1
            elif ep_steps < steps - 5:
                edge_resets += 1
            first_ep = False
            ep_steps = 0

    mean_spd = speed_sum / steps
    cruise_spd = cruise_speed_sum / max(cruise_steps, 1) if cruise_steps > 0 else mean_spd
    dist = mean_spd * seconds
    h_pos = float(exo.human_positive_work_j[0])
    h_met = float(exo.human_metabolic_proxy_j[0])
    s_pos = exo.positive_work_j[0].tolist()
    s_tot = sum(s_pos)
    share = s_tot / max(s_tot + h_pos, 1e-6)
    sym = min(s_pos) / max(max(s_pos), 1e-6) if max(s_pos) > 1.0 else 0.0
    peak_f = exo.peak_force_n[0].tolist()
    return {
        "speed_mps": mean_spd,
        "cruise_speed_mps": cruise_spd,
        "edge_resets": edge_resets,
        "dist_m": dist,
        "fails": fails,
        "h_pos_j": h_pos,
        "h_met_j": h_met,
        "met_j_per_m": h_met / max(dist, 0.1),
        "strut_pos_j": s_pos,
        "strut_share": share,
        "strut_sym": sym,
        "peak_force_n": peak_f,
        "max_foot_back_m": max_foot_back_1st,
        "max_tilt_1st_deg": max_tilt_1st,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--terrains", nargs="+", type=int, default=[12, 3, 24, 47, 8, 35])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=Path)
    args = p.parse_args()

    speeds = [0.0, 0.5, 1.0, 1.5]
    results = {"checkpoint": str(args.checkpoint), "terrains": args.terrains, "on": {}, "off": {}}

    for spd in speeds:
        per_t = []
        for tidx in args.terrains:
            env, actor = load_actor_and_env(args.checkpoint, tidx, 500.0, args.seed)
            per_t.append(run_episode(env, actor, spd, args.seconds))
        results["on"][str(spd)] = per_t

    for spd in (1.0, 1.5):
        per_t = []
        for tidx in args.terrains:
            env, actor = load_actor_and_env(args.checkpoint, tidx, 0.0, args.seed)
            per_t.append(run_episode(env, actor, spd, args.seconds))
        results["off"][str(spd)] = per_t

    print(f"\n=== Multi-Terrain Benchmark: {args.checkpoint.name} ({len(args.terrains)} terrains x {args.seconds:.0f}s) ===")
    for spd in speeds:
        rows = results["on"][str(spd)]
        v = sum(r["speed_mps"] for r in rows) / len(rows)
        vc = sum(r["cruise_speed_mps"] for r in rows) / len(rows)
        fails = sum(r["fails"] for r in rows)
        edges = sum(r["edge_resets"] for r in rows)
        share = sum(r["strut_share"] for r in rows) / len(rows) * 100.0
        sym = sum(r["strut_sym"] for r in rows) / len(rows)
        sL = sum(r["strut_pos_j"][0] for r in rows) / len(rows)
        sR = sum(r["strut_pos_j"][1] for r in rows) / len(rows)
        fb = sum(r["max_foot_back_m"] for r in rows) / len(rows)
        tlt = sum(r["max_tilt_1st_deg"] for r in rows) / len(rows)
        jm = sum(r["met_j_per_m"] for r in rows) / len(rows)
        pct_c = (vc / spd * 100.0) if spd > 0 else 0.0
        print(f"  ON  v={spd:3.1f} | cruise={vc:5.3f} m/s ({pct_c:5.1f}%) [raw={v:.3f}, edge={edges}] | fails={fails:2d}/{len(rows)} | "
              f"J/m={jm:6.1f} | sPos={sL:5.1f}/{sR:5.1f}J (share={share:4.1f}%, sym={sym:.2f}) | "
              f"1stHop: back={fb:+.3f}m tilt={tlt:+5.1f}°")

    for spd in (1.0, 1.5):
        on_r = results["on"][str(spd)]
        off_r = results["off"][str(spd)]
        on_jm = sum(r["met_j_per_m"] for r in on_r) / len(on_r)
        off_jm = sum(r["met_j_per_m"] for r in off_r) / len(off_r)
        on_hp = sum(r["h_pos_j"] for r in on_r) / len(on_r)
        off_hp = sum(r["h_pos_j"] for r in off_r) / len(off_r)
        on_v = sum(r["speed_mps"] for r in on_r) / len(on_r)
        off_v = sum(r["speed_mps"] for r in off_r) / len(off_r)
        off_fails = sum(r["fails"] for r in off_r)
        save_jm = (on_jm - off_jm) / off_jm * 100.0
        save_hp = (on_hp - off_hp) / off_hp * 100.0
        print(f"  CMP v={spd:3.1f} | ON v={on_v:.3f} vs OFF v={off_v:.3f} (OFF fails={off_fails}) | "
              f"ΔJ/m={save_jm:+.1f}% ({off_jm:.1f}->{on_jm:.1f}) | ΔhPos={save_hp:+.1f}%")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2))
        print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
