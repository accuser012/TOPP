from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from codroid_topp.planning.rolling_topp import RetimedTrajectory


def _ensure(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "figure.dpi": 120,
        "savefig.dpi": 280,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=320)
    plt.close(fig)
    return out.with_suffix(".png")


def plot_speed_profiles(trajs: Mapping[str, RetimedTrajectory], out_dir: str | Path, name: str = "fig_path_speed_profiles") -> Path:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for label, tr in trajs.items():
        ax.plot(tr.s, tr.speed, linewidth=2.0, label=label)
    ax.set_xlabel("Path parameter s")
    ax.set_ylabel("Path speed ds/dt")
    ax.set_title("Path-speed profiles")
    ax.legend()
    return _save(fig, _ensure(out_dir) / name)


def plot_torque_utilization(trajs: Mapping[str, RetimedTrajectory], tau_max: np.ndarray, out_dir: str | Path, name: str = "fig_torque_utilization_over_s") -> Path:
    apply_paper_style()
    tau_max = np.asarray(tau_max, dtype=float).reshape(7)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for label, tr in trajs.items():
        util = np.max(np.abs(tr.tau) / np.maximum(tau_max.reshape(1, 7), 1e-9), axis=1)
        ax.plot(tr.s, util, linewidth=2.0, label=label)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="limit")
    ax.set_xlabel("Path parameter s")
    ax.set_ylabel("Max joint torque utilization")
    ax.set_title("Torque utilization")
    ax.legend()
    return _save(fig, _ensure(out_dir) / name)


def plot_joint_profiles(tr: RetimedTrajectory, out_dir: str | Path, name: str = "fig_joint_profiles_q_qd_qdd_tau") -> Path:
    apply_paper_style()
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8.0), sharex=True, constrained_layout=True)
    for j in range(tr.q.shape[1]):
        axes[0].plot(tr.s, tr.q[:, j], linewidth=1.1)
        axes[1].plot(tr.s, tr.qd[:, j], linewidth=1.1)
        axes[2].plot(tr.s, tr.qdd[:, j], linewidth=1.1)
        axes[3].plot(tr.s, tr.tau[:, j], linewidth=1.1)
    axes[0].set_ylabel("q / rad")
    axes[1].set_ylabel("qd / rad/s")
    axes[2].set_ylabel("qdd / rad/s²")
    axes[3].set_ylabel("tau / Nm")
    axes[3].set_xlabel("Path parameter s")
    axes[0].set_title("Joint position, velocity, acceleration, and torque")
    return _save(fig, _ensure(out_dir) / name)


def plot_duration_compute_bars(rows: Sequence[Mapping[str, object]], out_dir: str | Path, name: str = "fig_main_4x2_duration_compute") -> Path:
    apply_paper_style()
    labels = [f"{r.get('method_label', r.get('method', ''))}\n{r.get('constraint_mode', '')}" for r in rows]
    duration = np.asarray([float(r.get('duration_mean', r.get('duration', np.nan))) for r in rows], dtype=float)
    compute = np.asarray([float(r.get('p95_compute_time_ms_mean', r.get('p95_compute_time_ms', np.nan))) for r in rows], dtype=float)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(9, 0.48 * len(labels)), 5.0), constrained_layout=True)
    ax1.bar(x - 0.18, duration, width=0.36, label="Duration / s")
    ax1.set_ylabel("Duration / s")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, compute, marker="o", linewidth=2.0, label="P95 compute / ms")
    ax2.axhline(50.0, linestyle="--", linewidth=1.2, label="50 ms")
    ax2.set_ylabel("Compute time / ms")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
    ax1.set_title("Duration and computation time")
    return _save(fig, _ensure(out_dir) / name)


def plot_arm_path(model, q_path: np.ndarray, out_dir: str | Path, name: str = "fig_cartesian_arm_path") -> Path:
    apply_paper_style()
    pts = []
    elbows = []
    for q in q_path:
        _, e, w = model.joint_positions(q)
        elbows.append(e)
        pts.append(w)
    pts = np.asarray(pts)
    elbows = np.asarray(elbows)
    fig = plt.figure(figsize=(6.2, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, label="wrist path")
    ax.plot(elbows[:, 0], elbows[:, 1], elbows[:, 2], linewidth=1.4, label="elbow path")
    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=40, label="start")
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=40, label="goal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    ax.set_title("Virtual SRS arm path")
    ax.legend()
    return _save(fig, _ensure(out_dir) / name)
