#!/usr/bin/env python3
"""Visualize the old vs. new event-driven 3-stage startup logic in ForwardFacingBootCommands."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_curves(
    t: np.ndarray,
    command_speed: float = 1.5,
    long_hop_period: float = 1.7,
    t_takeoff: float = 0.55,
    t_landing: float = 1.15,
) -> dict[str, np.ndarray]:
    first_takeoff = t >= t_takeoff
    first_landing = t >= t_landing

    # --- Old time-window startup logic ---
    old_v = np.minimum(command_speed, 0.32 + 0.25 * t)
    old_opening = (~first_takeoff) | (t < 1.8)
    old_transition = (~old_opening) & (t < 3.0)
    old_mirror = np.where(old_opening, 1.0, np.where(old_transition, 0.55, 0.0))
    old_launch_w = np.clip((5.0 - t) / 1.5, 0.0, 1.0)
    old_tilt_deg = np.rad2deg(np.interp(old_launch_w, [0.0, 1.0], [0.26, 0.18]))  # EXTEND allow
    old_period = 0.75 + (long_hop_period - 0.75) * np.clip(old_v / 1.0, 0.0, 1.0)
    old_hip_cap_deg = np.where((~first_landing) | (t < 3.0), np.rad2deg(0.22), 45.0)

    # --- New event-driven 3-stage startup logic ---
    ramp_s = 1.5
    new_v = np.minimum(command_speed, 0.75 + 0.75 * t)
    new_opening = ~first_takeoff
    new_transition = first_takeoff & (~first_landing) & (t < ramp_s)
    new_mirror = np.where(new_opening, 1.0, np.where(new_transition, 0.55, 0.0))
    time_cap = np.clip((ramp_s - t) / 0.5, 0.0, 1.0)
    new_launch_w = np.where(~first_takeoff, 1.0, np.where(~first_landing, 0.5 * time_cap, 0.0))
    new_tilt_deg = np.rad2deg(np.interp(new_launch_w, [0.0, 1.0], [0.26, 0.18]))  # EXTEND allow
    new_cruise_p = 0.95 + (long_hop_period - 0.95) * np.clip(new_v / 1.0, 0.0, 1.0)
    new_period = np.where(first_landing, new_cruise_p, 0.95)
    new_hip_cap_deg = np.where((~first_landing) & (t < ramp_s), np.rad2deg(0.26), 45.0)

    return {
        "old_v": old_v, "new_v": new_v,
        "old_mirror": old_mirror, "new_mirror": new_mirror,
        "old_launch_w": old_launch_w, "new_launch_w": new_launch_w,
        "old_tilt_deg": old_tilt_deg, "new_tilt_deg": new_tilt_deg,
        "old_period": old_period, "new_period": new_period,
        "old_hip_cap_deg": old_hip_cap_deg, "new_hip_cap_deg": new_hip_cap_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize startup logic comparison.")
    parser.add_argument("--command-speed", type=float, default=1.5)
    parser.add_argument("--long-hop-period", type=float, default=1.7)
    parser.add_argument("--t-takeoff", type=float, default=0.55, help="Simulated 1st takeoff time (s)")
    parser.add_argument("--t-landing", type=float, default=1.15, help="Simulated 1st landing time (s)")
    parser.add_argument("--output", type=Path, default=Path("outputs/startup_logic_visualization.png"))
    args = parser.parse_args()

    t = np.linspace(0.0, 6.0, 601)
    c = compute_curves(t, args.command_speed, args.long_hop_period, args.t_takeoff, args.t_landing)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    fig.suptitle(
        f"Startup Logic Comparison (command_speed={args.command_speed:.1f} m/s, "
        f"1st takeoff={args.t_takeoff:.2f}s, 1st landing={args.t_landing:.2f}s)",
        fontsize=13, fontweight="bold",
    )

    def mark_phases(ax):
        ax.axvspan(0.0, args.t_takeoff, color="#e8f4fd", alpha=0.55, label="_nolegend_")
        ax.axvspan(args.t_takeoff, args.t_landing, color="#fef6e4", alpha=0.65, label="_nolegend_")
        ax.axvspan(args.t_landing, 6.0, color="#edf7ed", alpha=0.45, label="_nolegend_")
        ax.axvline(args.t_takeoff, color="#2b6cb0", ls=":", lw=1.2)
        ax.axvline(args.t_landing, color="#2f855a", ls=":", lw=1.2)
        ax.grid(True, alpha=0.3)

    # 1. Target Speed Ramp
    ax = axes[0, 0]
    mark_phases(ax)
    ax.plot(t, c["old_v"], "--", color="#c53030", lw=2.0, label="Old (0.32 + 0.25t, full at 4.72s)")
    ax.plot(t, c["new_v"], "-", color="#2b6cb0", lw=2.5, label="New (0.75 + 0.75t, full at 1.00s)")
    ax.axhline(args.command_speed, color="gray", ls="-.", lw=1.0, alpha=0.7)
    ax.set_ylabel("Target Speed (m/s)")
    ax.set_title("1. Target Speed Ramp")
    ax.legend(loc="lower right", fontsize=9)

    # 2. Startup Guard Weights (launch_w & mirror)
    ax = axes[0, 1]
    mark_phases(ax)
    ax.plot(t, c["old_launch_w"], "--", color="#c53030", lw=2.0, label="Old launch_w (0..5.0s)")
    ax.plot(t, c["new_launch_w"], "-", color="#2b6cb0", lw=2.5, label="New launch_w (event-driven)")
    ax.plot(t, c["old_mirror"], ":", color="#dd6b20", lw=1.8, label="Old mirror lock (0..3.0s)")
    ax.plot(t, c["new_mirror"], "-.", color="#2f855a", lw=2.0, label="New mirror lock (exits at landing)")
    ax.set_ylabel("Weight / Strength")
    ax.set_title("2. Startup Guard & Symmetry Weights")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=8.5)

    # 3. Effective Long Hop Period
    ax = axes[1, 0]
    mark_phases(ax)
    ax.plot(t, c["old_period"], "--", color="#c53030", lw=2.0, label="Old period (0.75s -> 1.70s)")
    ax.plot(t, c["new_period"], "-", color="#2b6cb0", lw=2.5, label="New period (0.95s -> 1.70s at landing)")
    ax.set_xlabel("Episode Time (s)")
    ax.set_ylabel("Target Hop Period (s)")
    ax.set_title("3. Effective Long-Hop Period")
    ax.legend(loc="lower right", fontsize=9)

    # 4. Forward Lean Allowance (EXTEND) & HipPitch Cap
    ax = axes[1, 1]
    mark_phases(ax)
    ax.plot(t, c["old_tilt_deg"], "--", color="#c53030", lw=2.0, label="Old EXTEND tilt allow (deg)")
    ax.plot(t, c["new_tilt_deg"], "-", color="#2b6cb0", lw=2.5, label="New EXTEND tilt allow (deg)")
    ax.set_xlabel("Episode Time (s)")
    ax.set_ylabel("Allowed Forward Tilt (deg)")
    ax.set_title("4. EXTEND Forward Tilt Allowance (Cruise = 14.9°)")
    ax.set_ylim(9.5, 16.0)
    ax.legend(loc="lower right", fontsize=9)

    # Phase annotations on top of first row
    for ax in axes[0]:
        ax.text(args.t_takeoff * 0.5, 1.02, "A: Crouch\nStore", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color="#2b6cb0", fontweight="bold")
        ax.text((args.t_takeoff + args.t_landing) * 0.5, 1.02, "B: 1st\nHop", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color="#b7791f", fontweight="bold")
        ax.text((args.t_landing + 6.0) * 0.5, 1.02, "C: Periodic Long-Stride Cruise",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5,
                color="#2f855a", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved startup logic visualization to {args.output}")


if __name__ == "__main__":
    main()
