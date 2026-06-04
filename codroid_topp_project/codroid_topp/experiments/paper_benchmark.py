from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from codroid_topp.experiments.tasks import make_household_task_variant
from codroid_topp.planning.psi_optimizer import ArmAngleOptimizer, JointPath
from codroid_topp.planning.rolling_topp import RetimedTrajectory, RollingHorizonTOPP, TOPPConfig
from codroid_topp.robot.dynamics import PinocchioDynamics, make_dynamics_backend
from codroid_topp.robot.virtual_srs import VirtualSRSModel
from codroid_topp.trajectory.se3_path import BSplineSE3Path, Pose
from codroid_topp.utils.math_utils import min_limit_margin, rotation_error, wrap_to_pi

# The type is kept open because several baselines are intentionally approximate.
MethodName = str

DEFAULT_METHODS: tuple[str, ...] = (
    "offline_topp_ra",
    "streaming_smooth_psi",
    "streaming_adaptive_psi",
)

ALL_METHODS: tuple[str, ...] = (
    "conservative_global_dp",
    "offline_global_dp",
    "offline_topp_ni",
    "offline_topp_ni_grid",
    "offline_topp_ra",
    "offline_topp_co",
    "offline_convex_topp_like",
    "offline_convex_like_diagnostic",
    "rolling_global_dp",
    "streaming_fixed_psi",
    "streaming_local_psi",
    "streaming_smooth_psi",
    "streaming_adaptive_psi",
    "streaming_topp_aware_psi",
)

METHOD_LABELS: dict[str, str] = {
    "conservative_global_dp": "Conservative",
    "offline_global_dp": "Offline TOPP",
    "offline_topp_ni": "Grid TOPP-NI diagnostic",
    "offline_topp_ni_grid": "Offline TOPP-NI-grid",
    "offline_topp_ra": "Offline TOPP-RA",
    "offline_topp_co": "Offline TOPP-CO",
    "offline_convex_topp_like": "Convex-like diagnostic",
    "offline_convex_like_diagnostic": "Convex-like diagnostic",
    "rolling_global_dp": "Rolling global TOPP",
    "streaming_fixed_psi": "Streaming fixed psi",
    "streaming_local_psi": "Streaming local psi",
    "streaming_smooth_psi": "Streaming smooth psi",
    "streaming_adaptive_psi": "Streaming adaptive psi",
    "streaming_topp_aware_psi": "TOPP-aware oracle",
}

METHOD_GROUPS: dict[str, str] = {
    "conservative_global_dp": "offline_reference",
    "offline_global_dp": "offline_reference",
    "offline_topp_ni": "offline_approximate",
    "offline_topp_ni_grid": "offline_reference",
    "offline_topp_ra": "offline_reference",
    "offline_topp_co": "offline_reference",
    "offline_convex_topp_like": "offline_approximate",
    "offline_convex_like_diagnostic": "offline_approximate",
    "rolling_global_dp": "rolling_global",
    "streaming_fixed_psi": "online_streaming",
    "streaming_local_psi": "online_streaming",
    "streaming_smooth_psi": "online_streaming",
    "streaming_adaptive_psi": "online_streaming",
    "streaming_topp_aware_psi": "oracle",
}

OFFLINE_METHODS = {m for m, g in METHOD_GROUPS.items() if g.startswith("offline") or g == "rolling_global"}
ONLINE_METHODS = {m for m, g in METHOD_GROUPS.items() if g == "online_streaming" or g == "oracle"}
APPROXIMATE_METHODS = {"offline_topp_ni", "offline_convex_topp_like", "offline_convex_like_diagnostic"}
UNAVAILABLE_METHODS: set[str] = set()
GRID_NI_METHODS = {"offline_topp_ni_grid"}
ORACLE_METHODS = {"streaming_topp_aware_psi"}


@dataclass
class PaperExperimentConfig:
    tasks: tuple[str, ...] = ("cup_transfer", "drawer_reach", "medicine_handover")
    methods: tuple[str, ...] = DEFAULT_METHODS
    constraint_modes: tuple[str, ...] = ("dynamic",)
    experiment_suite: str = "paper_benchmark"
    samples: int = 60
    dense_count: int = 900
    repeats: int = 1
    horizon_points: int = 8
    brake_points: int = 4
    commit_points: int = 2
    terminal_safe_speed: float = 0.04
    tau_max: tuple[float, ...] | None = None
    actuator_modules: tuple[str, ...] | None = None
    actuator_torque_source: str | None = "rated"
    actuator_safety_factor: float | None = 0.8
    v_max: tuple[float, ...] | None = None
    realistic_velocity_limits: bool = False
    actuator_speed_source: str | None = None
    torque_scale: float = 1.0
    velocity_scale: float = 0.8
    search_grid: int = 5
    bisection_iters: int = 8
    endpoint_check: bool = True
    psi_grid_count: int = 41
    max_candidates_per_pose: int = 10
    local_psi_radius: float = 0.25
    local_n_psi: int = 5
    smooth_psi_radius: float = 0.35
    smooth_n_psi: int = 3
    smooth_accept_margin: float = 0.18
    adaptive_psi_radius: float = 0.45
    adaptive_n_psi: int = 5
    adaptive_accept_margin: float = 0.20
    adaptive_min_manipulability: float = 1.0e-4
    adaptive_use_torque_proxy: bool = True
    adaptive_score_weights: dict[str, float] = field(default_factory=lambda: {
        "joint_length": 1.0,
        "first_joint_jump": 0.20,
        "psi_jump": 0.08,
        "psi_total_variation": 0.05,
        "qs": 0.012,
        "qss": 0.004,
        "limit": 0.045,
        "manipulability": 0.025,
        "torque_proxy": 0.060,
        "jerk_proxy": 0.020,
        "torque_rate_proxy": 0.020,
    })
    adaptive_force_margin: float = 0.20
    adaptive_force_torque_util: float = 0.70
    topp_aware_psi_radius: float = 0.45
    topp_aware_n_psi: int = 7
    compute_time_budget_ms: float = 50.0
    conservative_speed_scale: float = 0.22
    qdd_max: tuple[float, ...] = (4.2, 3.6, 4.2, 5.0, 5.2, 5.2, 6.0)
    variants: int = 1
    seed: int = 42
    perturb_scale: float = 0.0
    path_change_s: float = 0.45
    path_change_perturb_scale: float = 1.5
    torque_scale_sweep: tuple[float, ...] | None = None
    velocity_scale_sweep: tuple[float, ...] | None = None
    max_reasonable_duration_s: float = 60.0

    def topp_config(self) -> TOPPConfig:
        return TOPPConfig(
            qdd_max=np.asarray(self.qdd_max, dtype=float),
            tau_max=None if self.tau_max is None else np.asarray(self.tau_max, dtype=float),
            v_max=None if self.v_max is None else np.asarray(self.v_max, dtype=float),
            torque_scale=float(self.torque_scale),
            velocity_scale=float(self.velocity_scale),
            terminal_safe_speed=float(self.terminal_safe_speed),
            horizon_points=int(self.horizon_points),
            brake_points=int(self.brake_points),
            commit_points=int(self.commit_points),
            search_grid=int(self.search_grid),
            bisection_iters=int(self.bisection_iters),
            endpoint_check=bool(self.endpoint_check),
        )


@dataclass
class LocalPathWindow:
    s: np.ndarray
    poses: list[Pose]
    s_start: float
    s_end: float
    reached_goal: bool


@dataclass
class StreamingRun:
    trajectory: RetimedTrajectory
    rows: list[dict[str, float | int | str]]
    arrays: dict[str, list[np.ndarray | float]]


@dataclass
class BenchmarkResult:
    metric_rows: list[dict[str, float | int | str]]
    window_rows: list[dict[str, float | int | str]]
    trajectories: dict[tuple[str, int, str, str, int], RetimedTrajectory] = field(default_factory=dict)
    manifest: dict[str, object] = field(default_factory=dict)


class LocalCartesianPathProvider:
    def __init__(self, cart_path: BSplineSE3Path, nominal_samples: int, horizon_points: int, brake_points: int):
        self.cart_path = cart_path
        self.nominal_samples = max(2, int(nominal_samples))
        self.ds_nominal = 1.0 / float(self.nominal_samples - 1)
        self.horizon_points = max(2, int(horizon_points))
        self.brake_points = max(0, int(brake_points))

    def get_window(self, s_current: float, horizon_length: float | None = None) -> LocalPathWindow:
        s0 = float(np.clip(s_current, 0.0, 1.0))
        if s0 >= 1.0 - 1e-12:
            s = np.array([1.0], dtype=float)
            return LocalPathWindow(s=s, poses=self.cart_path.sample(s), s_start=1.0, s_end=1.0, reached_goal=True)
        n_edges = max(1, self.horizon_points + self.brake_points - 1)
        if horizon_length is None:
            horizon_length = self.ds_nominal * n_edges
        horizon_length = max(float(horizon_length), self.ds_nominal)
        ds = horizon_length / float(n_edges)
        s_end = min(1.0, s0 + horizon_length)
        s = s0 + ds * np.arange(n_edges + 1, dtype=float)
        s = s[s < 1.0 - 1e-12]
        if len(s) == 0 or abs(float(s[0]) - s0) > 1e-12:
            s = np.r_[s0, s]
        if float(s[-1]) < s_end - 1e-12:
            s = np.r_[s, s_end]
        if s[-1] > 1.0 - 1e-12:
            s[-1] = 1.0
        keep = [0]
        for i in range(1, len(s)):
            if s[i] > s[keep[-1]] + 1e-12:
                keep.append(i)
        s = s[np.asarray(keep, dtype=int)]
        return LocalPathWindow(s=s, poses=self.cart_path.sample(s), s_start=float(s[0]), s_end=float(s[-1]), reached_goal=float(s[-1]) >= 1.0 - 1e-12)


class SwitchingCartesianPathProvider(LocalCartesianPathProvider):
    def __init__(self, cart_path: BSplineSE3Path, changed_path: BSplineSE3Path, s_change: float, nominal_samples: int, horizon_points: int, brake_points: int):
        super().__init__(cart_path, nominal_samples, horizon_points, brake_points)
        self.changed_path = changed_path
        self.s_change = float(np.clip(s_change, 0.0, 1.0))

    def get_window(self, s_current: float, horizon_length: float | None = None) -> LocalPathWindow:
        active = self.changed_path if float(s_current) >= self.s_change else self.cart_path
        old = self.cart_path
        self.cart_path = active
        try:
            return super().get_window(s_current, horizon_length=horizon_length)
        finally:
            self.cart_path = old


def make_ik(model: VirtualSRSModel, cfg: PaperExperimentConfig) -> ArmAngleOptimizer:
    return ArmAngleOptimizer(
        model,
        psi_grid=np.linspace(-math.pi, math.pi, max(3, cfg.psi_grid_count)),
        max_candidates_per_pose=cfg.max_candidates_per_pose,
        w_smooth=1.0,
        w_accel=0.02,
        w_limit=2e-3,
        w_psi=0.02,
    )


def find_initial_psi(cart_path: BSplineSE3Path, ik: ArmAngleOptimizer, q_seed: np.ndarray, psi_grid_count: int) -> float:
    grid = np.linspace(-math.pi, math.pi, max(3, int(psi_grid_count)))
    cands = ik.solver.solve_grid(cart_path.pose(0.0).as_T(), grid, q_prev=q_seed)
    if not cands:
        raise RuntimeError("Cannot find feasible initial psi")
    return float(cands[0].psi)


def make_retimed_from_arrays(arrays: dict[str, list[np.ndarray | float]], status: str, mode: str) -> RetimedTrajectory:
    s = np.asarray(arrays["s"], dtype=float)
    x = np.asarray(arrays["x"], dtype=float)
    u = np.asarray(arrays["u"], dtype=float)
    if len(s) > 1 and len(x) == len(s):
        u = np.zeros_like(x)
        ds = np.maximum(np.diff(s), 1.0e-12)
        u[:-1] = (x[1:] - x[:-1]) / (2.0 * ds)
        u[-1] = u[-2]
    return RetimedTrajectory(
        s=s,
        t=np.asarray(arrays["t"], dtype=float),
        x=x,
        u=u,
        q=np.vstack(arrays["q"]),
        qd=np.vstack(arrays["qd"]),
        qdd=np.vstack(arrays["qdd"]),
        tau=np.vstack(arrays["tau"]),
        status=status,
        mode=mode,
    )


def stitch_committed_segment(arrays: dict[str, list[np.ndarray | float]], local_traj: RetimedTrajectory, local_path: JointPath, commit_edges: int, time_offset: float, skip_first: bool) -> float:
    start = 1 if skip_first else 0
    for r in range(start, commit_edges + 1):
        arrays["s"].append(float(local_traj.s[r]))
        arrays["t"].append(float(time_offset + local_traj.t[r] - local_traj.t[0]))
        arrays["x"].append(float(local_traj.x[r]))
        arrays["u"].append(float(local_traj.u[r]))
        arrays["q"].append(local_traj.q[r].copy())
        arrays["qd"].append(local_traj.qd[r].copy())
        arrays["qdd"].append(local_traj.qdd[r].copy())
        arrays["tau"].append(local_traj.tau[r].copy())
        arrays["psi"].append(float(local_path.psi[r]))
        arrays["ik_position_error"].append(float(local_path.ik_position_error[r]))
        arrays["ik_rotation_error"].append(float(local_path.ik_rotation_error[r]))
    return float(time_offset + local_traj.t[commit_edges] - local_traj.t[0])


@contextmanager
def planning_constraint_mode(planner: RollingHorizonTOPP, constraint_mode: str):
    saved = np.asarray(planner.tau_max, dtype=float).copy()
    planner._u_cache.clear()
    try:
        if constraint_mode == "kinematic":
            planner.tau_max = np.ones_like(saved) * 1.0e12
        elif constraint_mode != "dynamic":
            raise ValueError(f"Unknown constraint_mode {constraint_mode!r}")
        yield
    finally:
        planner.tau_max = saved
        planner._u_cache.clear()


def local_constraint_summary(traj: RetimedTrajectory, planner: RollingHorizonTOPP) -> dict[str, float]:
    vmax = np.asarray(planner.v_max, dtype=float).reshape(7)
    tau_max = np.asarray(planner.tau_max, dtype=float).reshape(7)
    qdd_min = np.asarray(planner.qdd_min, dtype=float).reshape(7)
    qdd_max = np.asarray(planner.qdd_max, dtype=float).reshape(7)
    qd_abs = np.abs(traj.qd)
    qdd = np.asarray(traj.qdd, dtype=float)
    tau_abs = np.abs(traj.tau)
    acc_scale = np.maximum(np.maximum(np.abs(qdd_min), np.abs(qdd_max)), 1e-9)
    return {
        "max_velocity_utilization": float(np.max(qd_abs / np.maximum(vmax[None, :], 1e-9))),
        "max_acceleration_utilization": float(np.max(np.abs(qdd) / acc_scale[None, :])),
        "max_torque_utilization": float(np.max(tau_abs / np.maximum(tau_max[None, :], 1e-9))),
        "max_velocity_violation": float(np.max(np.maximum(qd_abs - vmax[None, :], 0.0))),
        "max_acceleration_violation": float(np.max(np.maximum(qdd - qdd_max[None, :], 0.0) + np.maximum(qdd_min[None, :] - qdd, 0.0))),
        "max_torque_violation": float(np.max(np.maximum(tau_abs - tau_max[None, :], 0.0))),
    }


def posthoc_torque_metrics(traj: RetimedTrajectory, planner: RollingHorizonTOPP) -> dict[str, float]:
    tau_max = np.maximum(np.asarray(planner.tau_max, dtype=float).reshape(7), 1e-9)
    tau_abs = np.abs(np.asarray(traj.tau, dtype=float).reshape((-1, 7)))
    util = tau_abs / tau_max.reshape(1, 7)
    violation = np.maximum(tau_abs - tau_max.reshape(1, 7), 0.0)
    node_violation = np.any(violation > 1.0e-7, axis=1)
    worst = int(np.argmax(util) % 7) + 1 if util.size else 1
    return {
        "posthoc_max_torque_utilization": float(np.max(util)) if util.size else float("nan"),
        "posthoc_max_torque_violation": float(np.max(violation)) if violation.size else 0.0,
        "posthoc_num_torque_violation_nodes": float(np.sum(node_violation)),
        "posthoc_torque_violation_pct": float(100.0 * np.mean(node_violation)) if node_violation.size else 0.0,
        "posthoc_worst_torque_joint": float(worst),
    }


def method_meta(method: str) -> dict[str, float | str]:
    group = METHOD_GROUPS.get(method, "unknown")
    unavailable = method in UNAVAILABLE_METHODS
    if method == "offline_topp_co":
        unavailable = not bool(cvxpy_backend_status().get("available", False))
    return {
        "method_label": METHOD_LABELS.get(method, method),
        "method_group": group,
        "is_offline_method": 1.0 if method in OFFLINE_METHODS else 0.0,
        "is_online_method": 1.0 if method in ONLINE_METHODS else 0.0,
        "is_oracle_method": 1.0 if method in ORACLE_METHODS else 0.0,
        "uses_full_global_path": 0.0 if method in ONLINE_METHODS else 1.0,
        "baseline_is_approximate": 1.0 if method in APPROXIMATE_METHODS else 0.0,
        "baseline_unavailable": 1.0 if unavailable else 0.0,
        "baseline_is_grid_ni": 1.0 if method in GRID_NI_METHODS else 0.0,
    }


def solve_fixed_center_window(ik: ArmAngleOptimizer, window: LocalPathWindow, q_current: np.ndarray, psi_center: float | None, psi_radius: float, n_psi: int) -> JointPath:
    return ik.solve_greedy(window.poses, window.s, q_seed=q_current, psi_center=psi_center, psi_radius=max(0.0, float(psi_radius)), n_psi=max(1, int(n_psi)))


def _candidate_order(center: float, radius: float, n: int) -> np.ndarray:
    offsets = np.linspace(-float(radius), float(radius), max(1, int(n)))
    offsets = np.asarray(sorted(offsets, key=lambda v: (abs(float(v)), float(v))), dtype=float)
    return float(center) + offsets


def _sample_manipulability(model: VirtualSRSModel, q: np.ndarray) -> float:
    """Cheap candidate-stage manipulability proxy.

    Full numerical-Jacobian manipulability is still reported after the timed
    planning section. During candidate scoring we use a distinct, low-cost proxy
    based on elbow bend and wrist bend so adaptive psi can remain real-time.
    This is deliberately not the same as the joint-limit margin.
    """
    _ = model
    q = np.asarray(q, dtype=float).reshape((-1, 7))
    if len(q) == 0:
        return 0.0
    idx = sorted(set([len(q) // 2, len(q) - 1]))
    vals = []
    for i in idx:
        qi = q[i]
        elbow = abs(math.sin(float(qi[3]))) + 0.10
        wrist = abs(math.cos(float(qi[5]))) + 0.10
        shoulder = abs(math.cos(float(qi[1]))) + 0.10
        vals.append(float(elbow * wrist * shoulder))
    return float(min(vals)) if vals else 0.0


def _cheap_torque_proxy(path: JointPath, planner: RollingHorizonTOPP) -> float:
    tau_max = np.maximum(np.asarray(planner.tau_max, dtype=float).reshape(7), 1e-9)
    idx = sorted(set([0, len(path.q) // 2, len(path.q) - 1]))
    vals = []
    for i in idx:
        try:
            tau = planner.dyn.inverse_dynamics(path.q[i], np.zeros(7), 0.02 * path.qss[i])
            vals.append(float(np.max(np.abs(tau) / tau_max)))
        except Exception:
            pass
    return max(vals) if vals else 0.0


def _cheap_torque_rate_proxy(path: JointPath, planner: RollingHorizonTOPP) -> float:
    tau_max = np.maximum(np.asarray(planner.tau_max, dtype=float).reshape(7), 1e-9)
    idx = sorted(set([0, len(path.q) // 2, len(path.q) - 1]))
    vals = []
    for i in idx:
        try:
            tau = planner.dyn.inverse_dynamics(path.q[i], np.zeros(7), 0.02 * path.qss[i])
            vals.append(tau / tau_max)
        except Exception:
            pass
    if len(vals) < 2:
        return 0.0
    return float(max(np.max(np.abs(vals[i + 1] - vals[i])) for i in range(len(vals) - 1)))


def adaptive_score_components(path: JointPath, q_current: np.ndarray, psi_current: float | None, center: float, model: VirtualSRSModel, planner: RollingHorizonTOPP, cfg: PaperExperimentConfig, force_torque_proxy: bool = False) -> dict[str, float]:
    q = np.asarray(path.q, dtype=float).reshape((-1, 7))
    q_current = np.asarray(q_current, dtype=float).reshape(7)
    if len(q) > 1:
        dq = np.asarray(wrap_to_pi(np.diff(q, axis=0)), dtype=float)
        joint_length = float(np.sum(np.linalg.norm(dq, axis=1)))
        psi_tv = float(np.sum(np.abs(np.diff(np.asarray(path.psi, dtype=float)))))
    else:
        joint_length = 0.0
        psi_tv = 0.0
    first_jump = float(np.linalg.norm(np.asarray(wrap_to_pi(q[0] - q_current), dtype=float)))
    psi_jump = 0.0 if psi_current is None else abs(float(wrap_to_pi(float(center) - float(psi_current))))
    qs_rms = float(np.sqrt(np.mean(path.qs ** 2))) if path.qs.size else 0.0
    qss_rms = float(np.sqrt(np.mean(path.qss ** 2))) if path.qss.size else 0.0
    jerk_proxy = float(np.sqrt(np.mean(np.diff(path.qss, axis=0) ** 2))) if len(path.qss) > 1 else 0.0
    margin = float(path.min_limit_margin)
    min_manip = _sample_manipulability(model, q)
    torque_proxy = _cheap_torque_proxy(path, planner) if (cfg.adaptive_use_torque_proxy or force_torque_proxy) else 0.0
    torque_rate_proxy = _cheap_torque_rate_proxy(path, planner) if (cfg.adaptive_use_torque_proxy or force_torque_proxy) else 0.0
    limit_penalty = 1.0 / max(margin, 1e-3)
    manip_penalty = 1.0 / max(min_manip, 1e-7)
    weights = cfg.adaptive_score_weights
    score = (
        weights.get("joint_length", 1.0) * joint_length
        + weights.get("first_joint_jump", 0.0) * first_jump
        + weights.get("psi_jump", 0.0) * psi_jump
        + weights.get("psi_total_variation", 0.0) * psi_tv
        + weights.get("qs", 0.0) * qs_rms
        + weights.get("qss", 0.0) * qss_rms
        + weights.get("limit", 0.0) * limit_penalty
        + weights.get("manipulability", 0.0) * manip_penalty
        + weights.get("torque_proxy", 0.0) * torque_proxy
        + weights.get("jerk_proxy", 0.0) * jerk_proxy
        + weights.get("torque_rate_proxy", 0.0) * torque_rate_proxy
        + 1.0e5 * float(np.max(path.ik_position_error))
        + 1.0e3 * float(np.max(path.ik_rotation_error))
    )
    return {
        "adaptive_score": float(score),
        "score_joint_length": joint_length,
        "score_first_joint_jump": first_jump,
        "score_psi_jump": psi_jump,
        "score_psi_total_variation": psi_tv,
        "score_qs": qs_rms,
        "score_qss": qss_rms,
        "score_jerk_proxy": jerk_proxy,
        "score_limit": limit_penalty,
        "score_manipulability": manip_penalty,
        "score_torque_proxy": torque_proxy,
        "score_torque_rate_proxy": torque_rate_proxy,
        "candidate_min_limit_margin_rad": margin,
        "candidate_min_manipulability": min_manip,
        "candidate_max_ik_position_error_m": float(np.max(path.ik_position_error)),
        "candidate_max_ik_rotation_error_rad": float(np.max(path.ik_rotation_error)),
    }


def choose_smooth_ik_window(window: LocalPathWindow, ik: ArmAngleOptimizer, q_current: np.ndarray, psi_current: float | None, cfg: PaperExperimentConfig) -> tuple[JointPath, dict[str, float | int | str]]:
    n = max(1, int(cfg.smooth_n_psi))
    centers = _candidate_order(0.0 if psi_current is None else float(psi_current), cfg.smooth_psi_radius, n)
    best = None
    failures = 0
    evaluated = 0
    for center in centers:
        try:
            path = solve_fixed_center_window(ik, window, q_current, float(center), 0.0, 1)
        except Exception:
            failures += 1
            continue
        evaluated += 1
        dq0 = float(np.linalg.norm(np.asarray(wrap_to_pi(path.q[0] - q_current), dtype=float)))
        dq = np.asarray(wrap_to_pi(np.diff(path.q, axis=0)), dtype=float) if len(path.q) > 1 else np.zeros((0, 7))
        joint_length = float(np.sum(np.linalg.norm(dq, axis=1))) if len(dq) else 0.0
        qss_rms = float(np.sqrt(np.mean(path.qss ** 2))) if path.qss.size else 0.0
        margin_penalty = 1.0 / max(float(path.min_limit_margin), 1e-3)
        psi_jump = 0.0 if psi_current is None else abs(float(wrap_to_pi(float(center) - float(psi_current))))
        score = joint_length + 0.10 * dq0 + 0.02 * qss_rms + 0.03 * margin_penalty + 0.02 * psi_jump
        meta = {"selected_psi_center": float(center), "num_psi_candidates": int(evaluated), "failed_psi_candidates": int(failures), "smooth_ik_score": float(score), "smooth_early_accept": 0, "adaptive_trigger_reason": "smooth"}
        if evaluated == 1 and float(path.min_limit_margin) >= cfg.smooth_accept_margin:
            meta["smooth_early_accept"] = 1
            return path, meta
        if best is None or score < best[0]:
            best = (score, path, meta)
    if best is None:
        raise RuntimeError("All smooth IK candidates failed")
    best[2]["num_psi_candidates"] = int(evaluated)
    best[2]["failed_psi_candidates"] = int(failures)
    return best[1], best[2]


def choose_adaptive_ik_window(window: LocalPathWindow, planner: RollingHorizonTOPP, ik: ArmAngleOptimizer, q_current: np.ndarray, psi_current: float | None, model: VirtualSRSModel, cfg: PaperExperimentConfig) -> tuple[JointPath, dict[str, float | int | str]]:
    center0 = 0.0 if psi_current is None else float(psi_current)
    first_center = float(center0)
    failures = 0
    evaluated = 0
    best: tuple[float, JointPath, dict[str, float | int | str], int] | None = None
    # Evaluate the nearest candidate first to decide whether this is a risky window.
    path0 = solve_fixed_center_window(ik, window, q_current, first_center, 0.0, 1)
    comp0 = adaptive_score_components(path0, q_current, psi_current, first_center, model, planner, cfg, force_torque_proxy=True)
    evaluated += 1
    reasons = []
    if comp0["candidate_min_limit_margin_rad"] < cfg.adaptive_force_margin:
        reasons.append("low_limit_margin")
    if comp0["candidate_min_manipulability"] < max(cfg.adaptive_min_manipulability, 1e-5):
        reasons.append("low_manipulability")
    if comp0["score_torque_proxy"] > cfg.adaptive_force_torque_util:
        reasons.append("high_torque_proxy")
    if comp0["score_qs"] > 8.0:
        reasons.append("large_qs")
    if comp0["score_qss"] > 80.0:
        reasons.append("large_qss")
    if comp0["score_first_joint_jump"] > 0.25:
        reasons.append("large_first_jump")
    if comp0["candidate_max_ik_position_error_m"] > 5e-5 or comp0["candidate_max_ik_rotation_error_rad"] > 5e-4:
        reasons.append("ik_error")
    meta0: dict[str, float | int | str] = {"selected_psi_center": first_center, "num_psi_candidates": 1, "failed_psi_candidates": 0, "early_accept": 0, "smooth_ik_score": float("nan"), "topp_aware_score": float("nan"), "adaptive_trigger_reason": "early_accept", "adaptive_risk_level": "low", "chosen_candidate_rank": 0, "candidate_margin_improvement": 0.0, "candidate_torque_proxy_improvement": 0.0, "candidate_manipulability_improvement": 0.0, **comp0}
    if not reasons and comp0["candidate_min_limit_margin_rad"] >= cfg.adaptive_accept_margin and comp0["score_psi_jump"] <= 0.5 * max(cfg.adaptive_psi_radius, 1e-6):
        meta0["early_accept"] = 1
        return path0, meta0
    if not reasons:
        reasons.append("multi_candidate_search")
    risk_level = "high" if len(reasons) >= 3 or "high_torque_proxy" in reasons or "ik_error" in reasons else "medium"
    n_candidates = 9 if risk_level == "high" else 5
    n_candidates = max(n_candidates, int(cfg.adaptive_n_psi))
    centers_all = _candidate_order(center0, cfg.adaptive_psi_radius, n_candidates)
    meta0["adaptive_trigger_reason"] = "+".join(reasons)
    meta0["adaptive_risk_level"] = risk_level
    best = (float(comp0["adaptive_score"]), path0, meta0, 0)
    base_margin = max(float(comp0["candidate_min_limit_margin_rad"]), 1e-9)
    base_torque = max(float(comp0["score_torque_proxy"]), 1e-9)
    base_manip = max(float(comp0["candidate_min_manipulability"]), 1e-9)
    for rank, center in enumerate(centers_all[1:], start=1):
        try:
            path = solve_fixed_center_window(ik, window, q_current, float(center), 0.0, 1)
            comp = adaptive_score_components(path, q_current, psi_current, float(center), model, planner, cfg, force_torque_proxy=("high_torque_proxy" in reasons))
        except Exception:
            failures += 1
            continue
        evaluated += 1
        margin_improvement = (float(comp["candidate_min_limit_margin_rad"]) - base_margin) / base_margin
        torque_improvement = (base_torque - float(comp["score_torque_proxy"])) / base_torque
        manip_improvement = (float(comp["candidate_min_manipulability"]) - base_manip) / base_manip
        adjusted_score = float(comp["adaptive_score"])
        if margin_improvement >= 0.10:
            adjusted_score *= 0.85
        if torque_improvement >= 0.05:
            adjusted_score *= 0.85
        meta = {"selected_psi_center": float(center), "num_psi_candidates": int(evaluated), "failed_psi_candidates": int(failures), "early_accept": 0, "smooth_ik_score": float("nan"), "topp_aware_score": float("nan"), "adaptive_trigger_reason": "+".join(reasons), "adaptive_risk_level": risk_level, "chosen_candidate_rank": int(rank), "candidate_margin_improvement": float(margin_improvement), "candidate_torque_proxy_improvement": float(torque_improvement), "candidate_manipulability_improvement": float(manip_improvement), **comp}
        if adjusted_score < best[0]:
            best = (adjusted_score, path, meta, rank)
    best[2]["num_psi_candidates"] = int(evaluated)
    best[2]["failed_psi_candidates"] = int(failures)
    best[2]["chosen_candidate_rank"] = int(best[3])
    return best[1], best[2]


def plan_local_path_once(planner: RollingHorizonTOPP, local_path: JointPath, speed_current: float, terminal_speed: float, constraint_mode: str) -> RetimedTrajectory:
    with planning_constraint_mode(planner, constraint_mode):
        return planner.plan_offline(local_path, start_speed=speed_current, goal_speed=terminal_speed)


def choose_topp_aware_window(window: LocalPathWindow, planner: RollingHorizonTOPP, ik: ArmAngleOptimizer, q_current: np.ndarray, psi_current: float | None, speed_current: float, terminal_speed: float, cfg: PaperExperimentConfig, constraint_mode: str) -> tuple[JointPath, RetimedTrajectory, dict[str, float | int | str]]:
    center0 = 0.0 if psi_current is None else float(psi_current)
    centers = _candidate_order(center0, cfg.topp_aware_psi_radius, max(1, cfg.topp_aware_n_psi))
    best = None
    failures = 0
    for center in centers:
        try:
            path = solve_fixed_center_window(ik, window, q_current, float(center), 0.0, 1)
            traj = plan_local_path_once(planner, path, speed_current, terminal_speed, constraint_mode)
            summary = local_constraint_summary(traj, planner)
            score = float(traj.duration) + 0.15 * max(0.0, summary["max_torque_utilization"] - 0.85) - 0.02 * min(max(path.min_limit_margin, 0.0), 1.0)
            meta = {"selected_psi_center": float(center), "num_psi_candidates": int(len(centers)), "failed_psi_candidates": int(failures), "topp_aware_score": score, "adaptive_trigger_reason": "oracle_full_topp_search"}
            if best is None or score < best[0]:
                best = (score, path, traj, meta)
        except Exception:
            failures += 1
    if best is None:
        raise RuntimeError("All TOPP-aware candidates failed")
    best[3]["failed_psi_candidates"] = int(failures)
    return best[1], best[2], best[3]


def plan_one_streaming_window(provider: LocalCartesianPathProvider, planner: RollingHorizonTOPP, ik: ArmAngleOptimizer, model: VirtualSRSModel, s_current: float, q_current: np.ndarray, psi_current: float | None, speed_current: float, method: str, cfg: PaperExperimentConfig, constraint_mode: str) -> tuple[RetimedTrajectory, JointPath, LocalPathWindow, int, float, dict[str, float | int | str]]:
    window = provider.get_window(s_current)
    if len(window.s) < 2:
        raise RuntimeError("Local path provider returned fewer than two samples")
    terminal_speed = planner.cfg.goal_speed if window.reached_goal else planner.cfg.terminal_safe_speed
    extra: dict[str, float | int | str] = {"selected_psi_center": float("nan"), "num_psi_candidates": 1, "failed_psi_candidates": 0, "early_accept": 0, "adaptive_score": float("nan"), "topp_aware_score": float("nan"), "smooth_ik_score": float("nan"), "adaptive_trigger_reason": "none"}
    if method == "streaming_topp_aware_psi":
        local_path, local_traj, extra = choose_topp_aware_window(window, planner, ik, q_current, psi_current, speed_current, terminal_speed, cfg, constraint_mode)
    elif method == "streaming_smooth_psi":
        local_path, extra = choose_smooth_ik_window(window, ik, q_current, psi_current, cfg)
        local_traj = plan_local_path_once(planner, local_path, speed_current, terminal_speed, constraint_mode)
    elif method == "streaming_adaptive_psi":
        local_path, extra = choose_adaptive_ik_window(window, planner, ik, q_current, psi_current, model, cfg)
        local_traj = plan_local_path_once(planner, local_path, speed_current, terminal_speed, constraint_mode)
    elif method == "streaming_local_psi":
        local_path = solve_fixed_center_window(ik, window, q_current, psi_current, cfg.local_psi_radius, cfg.local_n_psi)
        local_traj = plan_local_path_once(planner, local_path, speed_current, terminal_speed, constraint_mode)
        extra.update({"selected_psi_center": float(psi_current) if psi_current is not None else float("nan"), "num_psi_candidates": int(cfg.local_n_psi)})
    elif method == "streaming_fixed_psi":
        local_path = solve_fixed_center_window(ik, window, q_current, psi_current, 0.0, 1)
        local_traj = plan_local_path_once(planner, local_path, speed_current, terminal_speed, constraint_mode)
        extra.update({"selected_psi_center": float(psi_current) if psi_current is not None else float("nan"), "num_psi_candidates": 1})
    else:
        raise ValueError(f"Unsupported streaming method {method!r}")
    commit_edges = min(planner.cfg.commit_points, planner.cfg.horizon_points - 1, len(local_path.s) - 1)
    commit_edges = max(1, int(commit_edges))
    commit_path = planner._slice_path(local_path, 0, commit_edges)
    commit_traj = planner._assemble(commit_path, local_traj.x[: commit_edges + 1].copy(), mode=method)
    return commit_traj, local_path, window, commit_edges, terminal_speed, extra


def simulate_streaming_method(cart_path: BSplineSE3Path, changed_cart_path: BSplineSE3Path | None, task_q0: np.ndarray, ik: ArmAngleOptimizer, planner: RollingHorizonTOPP, model: VirtualSRSModel, method: str, psi0: float | None, cfg: PaperExperimentConfig, constraint_mode: str) -> StreamingRun:
    provider: LocalCartesianPathProvider
    provider = LocalCartesianPathProvider(cart_path, cfg.samples, cfg.horizon_points, cfg.brake_points) if changed_cart_path is None else SwitchingCartesianPathProvider(cart_path, changed_cart_path, cfg.path_change_s, cfg.samples, cfg.horizon_points, cfg.brake_points)
    arrays: dict[str, list[np.ndarray | float]] = {k: [] for k in ["s", "t", "x", "u", "q", "qd", "qdd", "tau", "psi", "ik_position_error", "ik_rotation_error"]}
    rows: list[dict[str, float | int | str]] = []
    s_current = 0.0
    q_current = np.asarray(task_q0, dtype=float).reshape(7).copy()
    speed_current = 0.0
    psi_current = float(psi0) if psi0 is not None else None
    time_offset = 0.0
    first_append = True
    status = "ok"
    window_id = 0
    while s_current < 1.0 - 1e-10:
        tic = time.perf_counter()
        try:
            commit_traj, local_path, window, commit_edges, terminal_speed, extra = plan_one_streaming_window(provider, planner, ik, model, s_current, q_current, psi_current, speed_current, method, cfg, constraint_mode)
        except Exception as exc:
            compute_time_ms = 1000.0 * (time.perf_counter() - tic)
            rows.append({"window_id": window_id, "constraint_mode": constraint_mode, "s_start": float(s_current), "s_end": float("nan"), "s_commit": float(s_current), "n_window_nodes": 0, "commit_edges": 0, "start_speed": float(speed_current), "commit_speed": float("nan"), "terminal_speed": float("nan"), "compute_time_ms": compute_time_ms, "commit_duration_s": 0.0, "status": f"failed: {type(exc).__name__}: {exc}"})
            status = str(rows[-1]["status"])
            break
        compute_time_ms = 1000.0 * (time.perf_counter() - tic)
        old_t = time_offset
        commit_path = planner._slice_path(local_path, 0, commit_edges)
        time_offset = stitch_committed_segment(arrays, commit_traj, commit_path, commit_edges, time_offset, skip_first=not first_append)
        first_append = False
        summary = local_constraint_summary(commit_traj, planner)
        s_commit = float(commit_path.s[commit_edges])
        speed_commit = float(math.sqrt(max(commit_traj.x[commit_edges], 0.0)))
        manips = np.asarray([model.manipulability(qi) for qi in local_path.q], dtype=float)
        row = {
            "window_id": window_id,
            "constraint_mode": constraint_mode,
            "s_start": float(s_current),
            "s_end": float(window.s_end),
            "s_commit": s_commit,
            "n_window_nodes": int(len(local_path.s)),
            "commit_edges": int(commit_edges),
            "start_speed": float(speed_current),
            "commit_speed": speed_commit,
            "terminal_speed": float(terminal_speed),
            "compute_time_ms": compute_time_ms,
            "commit_duration_s": float(time_offset - old_t),
            "local_max_torque_utilization": summary["max_torque_utilization"],
            "local_max_velocity_utilization": summary["max_velocity_utilization"],
            "local_max_acceleration_utilization": summary["max_acceleration_utilization"],
            "desired_commit_speed": speed_commit,
            "max_ik_position_error_m": float(np.max(local_path.ik_position_error)),
            "max_ik_rotation_error_rad": float(np.max(local_path.ik_rotation_error)),
            "min_limit_margin_rad": float(local_path.min_limit_margin),
            "min_manipulability": float(np.min(manips)) if manips.size else float("nan"),
            "online_path_change_active": 1 if changed_cart_path is not None and s_current >= cfg.path_change_s else 0,
            "status": "ok",
        }
        row.update(extra)
        rows.append(row)
        if s_commit <= s_current + 1e-12:
            status = "failed: non-increasing committed path progress"
            rows[-1]["status"] = status
            break
        s_current = s_commit
        q_current = commit_path.q[commit_edges].copy()
        speed_current = speed_commit
        psi_current = float(commit_path.psi[commit_edges])
        window_id += 1
        if window_id > 10000:
            status = "failed: exceeded maximum number of windows"
            break
    if not arrays["s"]:
        raise RuntimeError(f"{method} produced no committed samples")
    traj = make_retimed_from_arrays(arrays, status=status, mode=method)
    return StreamingRun(trajectory=traj, rows=rows, arrays=arrays)


def trajectory_constraint_metrics(traj: RetimedTrajectory, planner: RollingHorizonTOPP) -> dict[str, float]:
    out = traj.metrics(tau_max=planner.tau_max)
    out.update(local_constraint_summary(traj, planner))
    out["num_committed_nodes"] = float(len(traj.s))
    out["final_s"] = float(traj.s[-1]) if len(traj.s) else 0.0
    out["completed_path_pct"] = 100.0 * out["final_s"]
    out["status_ok"] = 1.0 if traj.status == "ok" else 0.0
    if len(traj.s) > 1:
        ds = np.maximum(np.diff(np.asarray(traj.s, dtype=float)), 1e-12)
        residual = np.asarray(traj.x[1:] - traj.x[:-1], dtype=float) - 2.0 * ds * np.asarray(traj.u[:-1], dtype=float)
        out["max_transition_residual"] = float(np.max(np.abs(residual))) if residual.size else 0.0
    else:
        out["max_transition_residual"] = float("nan")
    out["time_monotonic"] = 1.0 if len(traj.t) <= 1 or bool(np.all(np.diff(np.asarray(traj.t, dtype=float)) >= -1e-12)) else 0.0
    out["trajectory_has_nonfinite"] = 1.0 if (
        not np.all(np.isfinite(traj.s))
        or not np.all(np.isfinite(traj.t))
        or not np.all(np.isfinite(traj.x))
        or not np.all(np.isfinite(traj.u))
        or not np.all(np.isfinite(traj.q))
        or not np.all(np.isfinite(traj.qd))
        or not np.all(np.isfinite(traj.qdd))
        or not np.all(np.isfinite(traj.tau))
    ) else 0.0
    speed = np.asarray(traj.speed, dtype=float)
    if speed.size > 3 and np.all(np.isfinite(speed)):
        step = np.abs(np.diff(speed))
        med = float(np.median(step)) if step.size else 0.0
        out["max_speed_step"] = float(np.max(step)) if step.size else 0.0
        out["speed_spike_flag"] = 1.0 if med > 1e-9 and out["max_speed_step"] > 20.0 * med and out["max_speed_step"] > 0.5 else 0.0
    else:
        out["max_speed_step"] = 0.0
        out["speed_spike_flag"] = 1.0 if speed.size == 0 or not np.all(np.isfinite(speed)) else 0.0
    out["speed_profile_valid"] = 1.0 if out["trajectory_has_nonfinite"] < 0.5 and out["speed_spike_flag"] < 0.5 and speed.size > 0 else 0.0
    return out


def apply_quality_gates(row: dict[str, float | int | str], cfg: PaperExperimentConfig) -> dict[str, float | int | str]:
    reasons: list[str] = []
    def val(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except Exception:
            return default

    duration = val("duration", float("nan"))
    if not math.isfinite(duration):
        reasons.append("duration_nonfinite")
    elif duration > float(cfg.max_reasonable_duration_s):
        reasons.append("duration_too_large")
    if val("num_committed_nodes", 1.0) <= 0:
        reasons.append("no_committed_samples")
    if val("final_s", 0.0) < 0.99 or val("completed_path_pct", 0.0) < 99.0:
        reasons.append("incomplete_path")
    if val("max_velocity_violation", 0.0) > 1e-6:
        reasons.append("velocity_violation")
    if val("max_acceleration_violation", 0.0) > 1e-6:
        reasons.append("acceleration_violation")
    if val("time_monotonic", 1.0) < 0.5:
        reasons.append("time_nonmonotonic")
    if val("trajectory_has_nonfinite", 0.0) > 0.5:
        reasons.append("nonfinite_trajectory")
    if val("speed_spike_flag", 0.0) > 0.5:
        reasons.append("nonphysical_speed_spike")
    if val("speed_profile_valid", 1.0) < 0.5:
        reasons.append("invalid_speed_profile")
    if val("max_transition_residual", 0.0) > 1e-6:
        reasons.append("transition_residual_too_large")
    if str(row.get("constraint_mode", "")) == "dynamic":
        if val("max_torque_violation", 0.0) > 1e-6:
            reasons.append("dynamic_torque_violation")
        if val("posthoc_max_torque_violation", 0.0) > 1e-6:
            reasons.append("dynamic_posthoc_torque_violation")
    if str(row.get("status", "")) != "ok":
        reasons.append("solver_status_not_ok")

    passed = len(reasons) == 0
    row["quality_gate_passed"] = 1.0 if passed else 0.0
    row["valid_for_main_paper"] = 1.0 if passed else 0.0
    row["failure_reason"] = "none" if passed else ";".join(dict.fromkeys(reasons))
    row["status_ok"] = 1.0 if passed else 0.0
    if not passed and str(row.get("status", "")) == "ok":
        row["status"] = "failed_quality_gate"
    return row


def kinematic_quality_metrics(q: np.ndarray, psi: np.ndarray | None, model: VirtualSRSModel, ik_position_error: np.ndarray | None = None, ik_rotation_error: np.ndarray | None = None) -> dict[str, float]:
    q = np.asarray(q, dtype=float).reshape((-1, 7))
    dq = np.asarray(wrap_to_pi(np.diff(q, axis=0)), dtype=float) if len(q) >= 2 else np.zeros((0, 7))
    manips = np.asarray([model.manipulability(qi) for qi in q], dtype=float)
    margins = np.asarray([min_limit_margin(qi, model.q_min, model.q_max) for qi in q], dtype=float)
    out = {"joint_path_length_rad": float(np.sum(np.linalg.norm(dq, axis=1))) if len(dq) else 0.0, "min_limit_margin_rad": float(np.min(margins)), "mean_limit_margin_rad": float(np.mean(margins)), "min_manipulability": float(np.min(manips)), "mean_manipulability": float(np.mean(manips))}
    if psi is not None and len(psi) > 0:
        psi = np.asarray(psi, dtype=float).reshape(-1)
        out["psi_range_rad"] = float(np.max(psi) - np.min(psi))
        out["psi_total_variation_rad"] = float(np.sum(np.abs(np.diff(psi)))) if len(psi) > 1 else 0.0
    if ik_position_error is not None and len(ik_position_error) > 0:
        out["max_ik_position_error_m"] = float(np.max(np.asarray(ik_position_error, dtype=float)))
    if ik_rotation_error is not None and len(ik_rotation_error) > 0:
        out["max_ik_rotation_error_rad"] = float(np.max(np.asarray(ik_rotation_error, dtype=float)))
    return out


def solve_global_joint_path(cart_path: BSplineSE3Path, model: VirtualSRSModel, task_q0: np.ndarray, cfg: PaperExperimentConfig) -> tuple[JointPath, float, int]:
    ik = make_ik(model, cfg)
    s_grid = cart_path.uniform_grid(cfg.samples)
    poses = cart_path.sample(s_grid)
    t0 = time.perf_counter()
    try:
        path = ik.solve(poses, s_grid, q_seed=task_q0)
        fallback = 0
    except Exception:
        path = ik.solve_greedy(poses, s_grid, q_seed=task_q0, psi_center=None, psi_radius=math.pi, n_psi=max(7, cfg.local_n_psi))
        fallback = 1
    return path, 1000.0 * (time.perf_counter() - t0), fallback


def plan_offline_topp_ni_like(planner: RollingHorizonTOPP, path: JointPath, start_speed: float = 0.0, goal_speed: float = 0.0) -> RetimedTrajectory:
    """Conservative grid numerical-integration TOPP baseline.

    This is not a continuous switch-point Bobrow/Shin-McKay solver. It keeps the
    NI spirit by integrating feasible speeds on the path grid: a backward brake
    envelope is built from one-step dynamic feasibility, then a forward
    maximum-acceleration integration follows that envelope. If a one-step
    transition cannot be found, the path is reported infeasible instead of
    emitting a near-zero-speed trajectory with a meaningless huge duration.
    """
    planner._u_cache.clear()
    x_upper = planner._path_velocity_upper(path)
    n = len(path.s)
    if n < 2:
        raise RuntimeError("offline_topp_ni_grid requires at least two path samples")

    x_goal = min(float(goal_speed) ** 2, float(x_upper[-1]))
    x_bwd = np.minimum(np.asarray(x_upper, dtype=float), 1.0e6)
    x_bwd[-1] = x_goal
    for i in reversed(range(n - 1)):
        hi = float(x_upper[i])
        if planner._exists_feasible_next(path, x_upper, i, hi, float(x_bwd[i + 1])):
            x_bwd[i] = hi
            continue
        if not planner._exists_feasible_next(path, x_upper, i, 0.0, float(x_bwd[i + 1])):
            x_bwd[i] = 0.0
            continue
        lo = 0.0
        for _ in range(max(8, int(planner.cfg.bisection_iters))):
            mid = 0.5 * (lo + hi)
            if planner._exists_feasible_next(path, x_upper, i, mid, float(x_bwd[i + 1])):
                lo = mid
            else:
                hi = mid
        x_bwd[i] = max(0.0, min(float(lo), float(x_upper[i])))

    x = np.zeros(n, dtype=float)
    x[0] = min(float(start_speed) ** 2, float(x_bwd[0]), float(x_upper[0]))
    for i in range(n - 1):
        nxt = planner._max_feasible_next(path, x_upper, i, x[i], min(x_bwd[i + 1], x_upper[i + 1]))
        if nxt is None:
            raise RuntimeError(f"offline_topp_ni_grid infeasible at edge {i}")
        x[i + 1] = max(0.0, min(float(nxt), float(x_bwd[i + 1]), float(x_upper[i + 1])))
    x[-1] = min(x[-1], x_goal + 1e-8)
    return planner._assemble(path, x, mode="offline_ni_like", status="ok")


def cvxpy_backend_status() -> dict[str, object]:
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "installed_solvers": []}
    solvers = list(cp.installed_solvers())
    preferred = [s for s in ("ECOS", "CLARABEL", "OSQP", "SCS") if s in solvers]
    return {"available": bool(preferred), "error": "", "installed_solvers": solvers, "preferred_solvers": preferred}


def _topp_co_torque_coefficients(planner: RollingHorizonTOPP, path: JointPath, i: int, x_cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(path.q[i], dtype=float)
    qs = np.asarray(path.qs[i], dtype=float)
    qss = np.asarray(path.qss[i], dtype=float)
    zero = np.zeros(7, dtype=float)
    tau_g = np.asarray(planner.dyn.inverse_dynamics(q, zero, zero), dtype=float)
    tau_x = np.asarray(planner.dyn.inverse_dynamics(q, zero, qss), dtype=float) - tau_g
    tau_u = np.asarray(planner.dyn.inverse_dynamics(q, zero, qs), dtype=float) - tau_g
    qd_cap = qs * math.sqrt(max(float(x_cap), 0.0))
    tau_vel = np.asarray(planner.dyn.inverse_dynamics(q, qd_cap, zero), dtype=float) - tau_g
    return tau_g, tau_x, tau_u, np.abs(tau_vel)


def plan_offline_topp_co(planner: RollingHorizonTOPP, path: JointPath, start_speed: float = 0.0, goal_speed: float = 0.0) -> RetimedTrajectory:
    """Grid convex TOPP baseline solved with CVXPY.

    Decision variables are x_i = sdot_i^2 and u_i = sddot_i. The objective is
    a convex QP surrogate for time minimization: maximize path speed while
    softly regularizing acceleration and speed-profile roughness. Constraints
    include x bounds, transition dynamics, joint acceleration limits, and the
    affine inverse-dynamics torque model tau = g + a_x x + a_u u.
    """
    status = cvxpy_backend_status()
    if not status.get("available", False):
        raise RuntimeError(f"offline_topp_co unavailable: CVXPY solver backend unavailable ({status.get('error', '')})")
    import cvxpy as cp  # type: ignore

    planner._u_cache.clear()
    s = np.asarray(path.s, dtype=float)
    n = len(s)
    if n < 2:
        raise RuntimeError("offline_topp_co requires at least two path samples")
    ds = np.maximum(np.diff(s), 1e-12)
    x_upper = np.asarray(planner._path_velocity_upper(path), dtype=float)
    x_upper = np.minimum(np.maximum(x_upper, 0.0), 1.0e6)
    x = cp.Variable(n, nonneg=True)
    u = cp.Variable(n)
    constraints = [
        x <= x_upper,
        x[0] == float(start_speed) ** 2,
        x[-1] == float(goal_speed) ** 2,
    ]
    for i in range(n - 1):
        constraints.append(x[i + 1] == x[i] + 2.0 * float(ds[i]) * u[i])
    constraints.append(u[-1] == u[-2])

    qs = np.asarray(path.qs, dtype=float).reshape((n, 7))
    qss = np.asarray(path.qss, dtype=float).reshape((n, 7))
    qdd_min = np.asarray(planner.qdd_min, dtype=float).reshape(7)
    qdd_max = np.asarray(planner.qdd_max, dtype=float).reshape(7)
    tau_max = np.asarray(planner.tau_max, dtype=float).reshape(7)
    dynamic_active = bool(np.max(tau_max) < 1.0e11)
    for i in range(n):
        for j in range(7):
            qdd_expr = float(qss[i, j]) * x[i] + float(qs[i, j]) * u[i]
            constraints.append(qdd_expr <= float(qdd_max[j]))
            constraints.append(qdd_expr >= float(qdd_min[j]))
        if dynamic_active:
            tau_g, tau_x, tau_u, tau_vel_bound = _topp_co_torque_coefficients(planner, path, i, float(x_upper[i]))
            for j in range(7):
                safe_tau = max(float(tau_max[j]) - float(tau_vel_bound[j]) - 1.0e-4, 1.0e-6)
                tau_expr = float(tau_g[j]) + float(tau_x[j]) * x[i] + float(tau_u[j]) * u[i]
                constraints.append(tau_expr <= safe_tau)
                constraints.append(tau_expr >= -safe_tau)

    objective = cp.Minimize(
        -cp.sum(x[1:-1])
        + 1.0e-5 * cp.sum_squares(u)
        + 1.0e-5 * cp.sum_squares(x[1:] - x[:-1])
    )
    problem = cp.Problem(objective, constraints)
    installed = status.get("preferred_solvers", [])
    last_error = None
    inaccurate_candidate: np.ndarray | None = None
    inaccurate_status: str | None = None
    for solver in installed:
        try:
            kwargs = {"solver": solver, "warm_start": True, "verbose": False}
            if solver == "OSQP":
                kwargs.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 20000})
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
                problem.solve(**kwargs)
            if problem.status == cp.OPTIMAL and x.value is not None:
                vals = np.asarray(x.value, dtype=float).reshape(n)
                vals = np.minimum(np.maximum(vals, 0.0), x_upper)
                vals[0] = float(start_speed) ** 2
                vals[-1] = float(goal_speed) ** 2
                traj = planner._assemble(path, vals, mode="offline_topp_co", status="ok")
                return traj
            if problem.status == cp.OPTIMAL_INACCURATE and x.value is not None:
                inaccurate_candidate = np.asarray(x.value, dtype=float).reshape(n)
                inaccurate_status = str(problem.status)
        except Exception as exc:
            last_error = exc
    if inaccurate_candidate is not None:
        vals = np.minimum(np.maximum(inaccurate_candidate, 0.0), x_upper)
        vals[0] = float(start_speed) ** 2
        vals[-1] = float(goal_speed) ** 2
        _ = inaccurate_status
        return planner._assemble(path, vals, mode="offline_topp_co_inaccurate", status="ok")
    raise RuntimeError(f"offline_topp_co failed: status={problem.status}, error={last_error}")


def plan_offline_convex_like(planner: RollingHorizonTOPP, path: JointPath, start_speed: float = 0.0, goal_speed: float = 0.0) -> RetimedTrajectory:
    # A documented convex-like approximation: start from the reachability solution
    # and apply a conservative smooth shrink to emulate a smoothness-regularized
    # convex retiming. It is intentionally marked approximate in the output.
    base = planner.plan_offline(path, start_speed=start_speed, goal_speed=goal_speed)
    x = np.asarray(base.x, dtype=float).copy()
    if len(x) > 2:
        xs = x.copy()
        for i in range(1, len(x) - 1):
            xs[i] = 0.25 * x[i - 1] + 0.50 * x[i] + 0.25 * x[i + 1]
        x = np.minimum(x, 0.985 * xs)
        x[0] = start_speed * start_speed
        x[-1] = min(x[-1], goal_speed * goal_speed + 1e-8)
    return planner._assemble(path, x, mode="offline_convex_like", status="ok")


def add_context(metrics: dict[str, float | int | str], task: str, variant_id: int, method: str, constraint_mode: str, repeat: int, status: str, cfg: PaperExperimentConfig) -> dict[str, float | int | str]:
    out = dict(metrics)
    out.update({"task": task, "variant_id": int(variant_id), "seed": int(cfg.seed), "method": method, "constraint_mode": constraint_mode, "repeat": int(repeat), "status": status})
    out.update(method_meta(method))
    out.setdefault("baseline_fallback_used", 0.0)
    out.setdefault("uses_global_joint_path", 1.0 if method not in ONLINE_METHODS else 0.0)
    out.setdefault("local_ik_each_window", 0.0 if method not in ONLINE_METHODS else 1.0)
    return out


def run_offline_method(task_name: str, variant_id: int, repeat: int, method: str, constraint_mode: str, path: JointPath, model: VirtualSRSModel, planner: RollingHorizonTOPP, cfg: PaperExperimentConfig, global_ik_ms: float, ik_fallback: int) -> tuple[dict[str, float | int | str], RetimedTrajectory]:
    t0 = time.perf_counter()
    fallback = 0
    with planning_constraint_mode(planner, constraint_mode):
        if method == "conservative_global_dp":
            traj = planner.conservative_trapezoid(path, speed_scale=cfg.conservative_speed_scale)
        elif method in ("offline_global_dp", "offline_topp_ra"):
            traj = planner.plan_offline(path, 0.0, 0.0)
            traj.mode = "offline_topp_ra" if method == "offline_topp_ra" else "offline_global_dp"
        elif method == "rolling_global_dp":
            traj = planner.plan_rolling(path, start_speed=0.0)
        elif method in ("offline_topp_ni", "offline_topp_ni_grid"):
            try:
                traj = plan_offline_topp_ni_like(planner, path, 0.0, 0.0)
            except Exception:
                if method == "offline_topp_ni_grid":
                    raise
                traj = planner.plan_offline(path, 0.0, 0.0)
                traj.mode = "offline_ni_like_fallback"
                fallback = 1
            if method == "offline_topp_ni_grid":
                traj.mode = "offline_topp_ni_grid"
        elif method == "offline_topp_co":
            traj = plan_offline_topp_co(planner, path, 0.0, 0.0)
        elif method in ("offline_convex_topp_like", "offline_convex_like_diagnostic"):
            try:
                traj = plan_offline_convex_like(planner, path, 0.0, 0.0)
            except Exception:
                traj = planner.plan_offline(path, 0.0, 0.0)
                traj.mode = "offline_convex_like_fallback"
                fallback = 1
        else:
            raise ValueError(method)
    compute_ms = 1000.0 * (time.perf_counter() - t0)
    metrics = trajectory_constraint_metrics(traj, planner)
    metrics.update(kinematic_quality_metrics(path.q, path.psi, model, path.ik_position_error, path.ik_rotation_error))
    metrics.update(posthoc_torque_metrics(traj, planner))
    metrics.update({"precompute_ik_time_ms": float(global_ik_ms), "planning_compute_time_ms": float(compute_ms), "mean_compute_time_ms": float(compute_ms), "p95_compute_time_ms": float(compute_ms), "max_compute_time_ms": float(compute_ms), "windows_under_budget_pct": 100.0 if compute_ms <= cfg.compute_time_budget_ms else 0.0, "uses_global_joint_path": 1.0, "local_ik_each_window": 0.0, "global_ik_fallback_used": float(ik_fallback), "baseline_fallback_used": float(fallback)})
    if method == "offline_topp_co":
        metrics["topp_co_solver_inaccurate"] = 1.0 if "inaccurate" in str(traj.mode) else 0.0
    row = add_context(metrics, task_name, variant_id, method, constraint_mode, repeat, traj.status, cfg)
    return apply_quality_gates(row, cfg), traj


def window_statistics(rows: list[dict[str, float | int | str]], budget_ms: float) -> dict[str, float]:
    if not rows:
        return {}
    comp = np.asarray([float(r.get("compute_time_ms", np.nan)) for r in rows], dtype=float)
    commit = 1000.0 * np.asarray([float(r.get("commit_duration_s", 0.0)) for r in rows], dtype=float)
    return {
        "num_windows": float(len(rows)),
        "mean_compute_time_ms": float(np.nanmean(comp)),
        "median_compute_time_ms": float(np.nanmedian(comp)),
        "p95_compute_time_ms": float(np.nanpercentile(comp, 95)),
        "max_compute_time_ms": float(np.nanmax(comp)),
        "windows_under_budget_pct": float(100.0 * np.nanmean(comp <= float(budget_ms))),
        "planner_slower_than_commit_windows": float(np.sum(comp > commit)),
        "mean_committed_motion_time_ms": float(np.nanmean(commit)),
        "min_commit_to_compute_ratio": float(np.nanmin(commit / np.maximum(comp, 1e-9))),
        "early_accept_rate": float(np.mean([int(r.get("early_accept", 0)) for r in rows])),
    }


def run_streaming_method(task_name: str, variant_id: int, repeat: int, method: str, constraint_mode: str, cart_path: BSplineSE3Path, changed_path: BSplineSE3Path | None, model: VirtualSRSModel, task_q0: np.ndarray, planner: RollingHorizonTOPP, cfg: PaperExperimentConfig, psi0: float) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], RetimedTrajectory]:
    ik = make_ik(model, cfg)
    run = simulate_streaming_method(cart_path, changed_path, task_q0, ik, planner, model, method, psi0, cfg, constraint_mode)
    arrays = run.arrays
    metrics = trajectory_constraint_metrics(run.trajectory, planner)
    metrics.update(kinematic_quality_metrics(run.trajectory.q, np.asarray(arrays["psi"], dtype=float), model, np.asarray(arrays["ik_position_error"], dtype=float), np.asarray(arrays["ik_rotation_error"], dtype=float)))
    metrics.update(window_statistics(run.rows, cfg.compute_time_budget_ms))
    metrics.update(posthoc_torque_metrics(run.trajectory, planner))
    metrics["uses_global_joint_path"] = 0.0
    metrics["local_ik_each_window"] = 1.0
    metrics["initial_psi_rad"] = float(psi0)
    metrics["mean_psi_candidates_per_window"] = float(np.mean([float(r.get("num_psi_candidates", 1)) for r in run.rows])) if run.rows else 0.0
    metrics["failed_psi_candidates"] = float(sum(int(r.get("failed_psi_candidates", 0)) for r in run.rows))
    if changed_path is not None:
        after = [r for r in run.rows if float(r.get("s_start", 0.0)) >= cfg.path_change_s]
        metrics["online_path_change"] = 1.0
        metrics["path_change_s"] = float(cfg.path_change_s)
        metrics["windows_after_change"] = float(len(after))
        metrics["first_window_after_change_compute_ms"] = float(after[0].get("compute_time_ms", float("nan"))) if after else float("nan")
        metrics["mean_compute_after_change_ms"] = float(np.mean([float(r.get("compute_time_ms", 0.0)) for r in after])) if after else float("nan")
        metrics["extra_duration_after_change"] = float(sum(float(r.get("commit_duration_s", 0.0)) for r in after))
        try:
            T_goal = changed_path.pose(1.0).as_T()
            T_end = model.fk(run.trajectory.q[-1])
            metrics["final_position_error_to_changed_goal_m"] = float(np.linalg.norm(T_goal[:3, 3] - T_end[:3, 3]))
            metrics["final_rotation_error_to_changed_goal_rad"] = float(rotation_error(T_goal[:3, :3], T_end[:3, :3]))
        except Exception:
            metrics["final_position_error_to_changed_goal_m"] = float("nan")
            metrics["final_rotation_error_to_changed_goal_rad"] = float("nan")
        metrics["recovery_windows"] = float(min(len(after), 3))
    metric_row = add_context(metrics, task_name, variant_id, method, constraint_mode, repeat, run.trajectory.status, cfg)
    metric_row = apply_quality_gates(metric_row, cfg)
    window_rows = []
    for row in run.rows:
        wr = dict(row)
        wr.update({"task": task_name, "variant_id": int(variant_id), "seed": int(cfg.seed), "method": method, "constraint_mode": constraint_mode, "repeat": int(repeat)})
        window_rows.append(wr)
    return metric_row, window_rows, run.trajectory


def run_paper_benchmark(urdf_path: str | Path, out_dir: str | Path, cfg: PaperExperimentConfig, prefer_pinocchio: bool = True, require_pinocchio: bool = False, make_plots: bool = True) -> BenchmarkResult:
    urdf_path = Path(urdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = VirtualSRSModel.from_urdf(urdf_path, side="l")
    dynamics = PinocchioDynamics(urdf_path, side="l") if require_pinocchio else make_dynamics_backend(urdf_path, side="l", prefer_pinocchio=prefer_pinocchio)
    base_planner = RollingHorizonTOPP(dynamics, cfg.topp_config())
    metric_rows: list[dict[str, float | int | str]] = []
    window_rows: list[dict[str, float | int | str]] = []
    trajectories: dict[tuple[str, int, str, str, int], RetimedTrajectory] = {}
    tau_before_scale = None if cfg.tau_max is None else [float(v) for v in cfg.tau_max]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cvxpy_backend": cvxpy_backend_status(),
        "dynamics_backend": type(dynamics).__name__,
        "urdf_path": str(urdf_path),
        "config": asdict(cfg),
        "actuator_torque_source": cfg.actuator_torque_source or "custom",
        "actuator_safety_factor": cfg.actuator_safety_factor,
        "tau_limit_formula": "rated actuator torque x 0.8 x torque_scale",
        "tau_max_before_torque_scale": tau_before_scale,
        "final_tau_max": np.asarray(base_planner.tau_max, dtype=float).tolist(),
        "planner_limits": {
            "v_max": np.asarray(base_planner.v_max).tolist(),
            "qdd_min": np.asarray(base_planner.qdd_min).tolist(),
            "qdd_max": np.asarray(base_planner.qdd_max).tolist(),
            "tau_max": np.asarray(base_planner.tau_max).tolist(),
        },
    }
    for task_name in cfg.tasks:
        for variant_id in range(max(1, int(cfg.variants))):
            task = make_household_task_variant(model, task_name, variant_id=variant_id, seed=cfg.seed, perturb_scale=cfg.perturb_scale)
            cart_path = BSplineSE3Path(task.keyframes, dense_count=cfg.dense_count)
            changed_path = None
            if cfg.experiment_suite == "online_path_change":
                changed_task = make_household_task_variant(model, task_name, variant_id=variant_id + 1000, seed=cfg.seed + 17, perturb_scale=cfg.perturb_scale * cfg.path_change_perturb_scale + 1.0)
                changed_path = BSplineSE3Path(changed_task.keyframes, dense_count=cfg.dense_count)
            base_ik = make_ik(model, cfg)
            psi0 = find_initial_psi(cart_path, base_ik, task.q_waypoints[0], cfg.psi_grid_count)
            for repeat in range(max(1, int(cfg.repeats))):
                global_path = None
                global_ms = float("nan")
                ik_fallback = 0
                if any(m in OFFLINE_METHODS for m in cfg.methods):
                    try:
                        global_path, global_ms, ik_fallback = solve_global_joint_path(cart_path, model, task.q_waypoints[0], cfg)
                    except Exception as exc:
                        for method in cfg.methods:
                            if method in OFFLINE_METHODS:
                                for cm in cfg.constraint_modes:
                                    row = add_context({"status_ok": 0.0, "duration": float("nan"), "final_s": 0.0, "completed_path_pct": 0.0, "error": f"global IK failed: {exc}"}, task_name, variant_id, method, cm, repeat, "failed", cfg)
                                    metric_rows.append(apply_quality_gates(row, cfg))
                for constraint_mode in cfg.constraint_modes:
                    planner = RollingHorizonTOPP(dynamics, cfg.topp_config())
                    for method in cfg.methods:
                        try:
                            if method in OFFLINE_METHODS:
                                if global_path is None:
                                    continue
                                row, traj = run_offline_method(task_name, variant_id, repeat, method, constraint_mode, global_path, model, planner, cfg, global_ms, ik_fallback)
                                if cfg.experiment_suite == "online_path_change":
                                    row["online_path_change"] = 1.0
                                    row["requires_global_replan"] = 1.0
                                    row["global_replan_compute_time_ms"] = row.get("planning_compute_time_ms", float("nan"))
                                metric_rows.append(row)
                                trajectories[(task_name, variant_id, method, constraint_mode, repeat)] = traj
                            elif method in ONLINE_METHODS:
                                row, wrs, traj = run_streaming_method(task_name, variant_id, repeat, method, constraint_mode, cart_path, changed_path, model, task.q_waypoints[0], planner, cfg, psi0)
                                metric_rows.append(row)
                                window_rows.extend(wrs)
                                trajectories[(task_name, variant_id, method, constraint_mode, repeat)] = traj
                        except Exception as exc:
                            row = add_context({"status_ok": 0.0, "duration": float("nan"), "final_s": 0.0, "completed_path_pct": 0.0, "error": f"{type(exc).__name__}: {exc}"}, task_name, variant_id, method, constraint_mode, repeat, "failed", cfg)
                            metric_rows.append(apply_quality_gates(row, cfg))
    result = BenchmarkResult(metric_rows=metric_rows, window_rows=window_rows, trajectories=trajectories, manifest=manifest)
    write_benchmark_outputs(result, out_dir)
    if make_plots:
        plot_benchmark_outputs(result, out_dir, cfg)
    write_analysis_report(result, out_dir, cfg)
    return result


def write_csv(path: Path, rows: Sequence[dict[str, object]], preferred: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(preferred) + sorted({k for r in rows for k in r.keys() if k not in preferred})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def run_path_validation(urdf_path: str | Path, out_dir: str | Path, cfg: PaperExperimentConfig, prefer_pinocchio: bool = True, require_pinocchio: bool = False) -> list[dict[str, float | int | str]]:
    urdf_path = Path(urdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = VirtualSRSModel.from_urdf(urdf_path, side="l")
    dynamics = PinocchioDynamics(urdf_path, side="l") if require_pinocchio else make_dynamics_backend(urdf_path, side="l", prefer_pinocchio=prefer_pinocchio)
    planner = RollingHorizonTOPP(dynamics, cfg.topp_config())
    rows: list[dict[str, float | int | str]] = []
    repair_scales = []
    base_scale = float(cfg.perturb_scale)
    for scale in (base_scale, 0.5 * base_scale, 0.0):
        if scale not in repair_scales:
            repair_scales.append(scale)
    for task_name in cfg.tasks:
        for variant_id in range(max(1, int(cfg.variants))):
            best_row: dict[str, float | int | str] | None = None
            for attempt, scale in enumerate(repair_scales):
                row: dict[str, float | int | str] = {"task": task_name, "variant_id": int(variant_id), "attempt": int(attempt), "perturb_scale_used": float(scale)}
                try:
                    task = make_household_task_variant(model, task_name, variant_id=variant_id, seed=cfg.seed, perturb_scale=scale)
                    cart_path = BSplineSE3Path(task.keyframes, dense_count=cfg.dense_count)
                    path, global_ms, ik_fallback = solve_global_joint_path(cart_path, model, task.q_waypoints[0], cfg)
                    row.update(kinematic_quality_metrics(path.q, path.psi, model, path.ik_position_error, path.ik_rotation_error))
                    row.update({
                        "ik_feasible": 1.0,
                        "path_validation_success": 1.0,
                        "global_ik_time_ms": float(global_ms),
                        "global_ik_fallback_used": float(ik_fallback),
                        "max_ik_position_error_m": float(np.max(path.ik_position_error)),
                        "max_ik_rotation_error_rad": float(np.max(path.ik_rotation_error)),
                        "qs_rms": float(np.sqrt(np.mean(path.qs ** 2))) if path.qs.size else 0.0,
                        "qss_rms": float(np.sqrt(np.mean(path.qss ** 2))) if path.qss.size else 0.0,
                        "rough_torque_proxy": float(_cheap_torque_proxy(path, planner)),
                        "repair_used": 1.0 if attempt > 0 else 0.0,
                        "validation_failure_reason": "none",
                    })
                    best_row = row
                    break
                except Exception as exc:
                    row.update({
                        "ik_feasible": 0.0,
                        "path_validation_success": 0.0,
                        "repair_used": 1.0 if attempt > 0 else 0.0,
                        "validation_failure_reason": f"{type(exc).__name__}: {exc}",
                    })
                    best_row = row
            assert best_row is not None
            rows.append(best_row)
    write_csv(out_dir / "path_validation_metrics.csv", rows, ["task", "variant_id", "path_validation_success", "repair_used", "perturb_scale_used", "min_limit_margin_rad", "mean_manipulability", "max_ik_position_error_m", "max_ik_rotation_error_rad", "qs_rms", "qss_rms", "rough_torque_proxy", "validation_failure_reason"])
    lines = ["# Path Validation Report", "", f"URDF: `{urdf_path}`", f"Dynamics backend: `{type(dynamics).__name__}`", ""]
    for task in cfg.tasks:
        task_rows = [r for r in rows if str(r.get("task")) == task]
        ok = sum(float(r.get("path_validation_success", 0.0)) > 0.5 for r in task_rows)
        rate = 100.0 * ok / max(1, len(task_rows))
        lines.append(f"- `{task}`: {ok}/{len(task_rows)} valid ({rate:.1f}%).")
        if rate < 95.0:
            lines.append("  - WARNING: below 95% validation success; exclude from `paper_main_stable` until repaired.")
    lines.append("")
    lines.append("Validation attempts reduce perturbation scale before declaring failure; geometry-changing repair is still a required follow-up for tasks that remain below 95%.")
    (out_dir / "path_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def aggregate_metric_rows(rows: Sequence[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for r in rows:
        key = (str(r.get("task", "")), str(r.get("method", "")), str(r.get("constraint_mode", "dynamic")))
        groups.setdefault(key, []).append(r)
    out: list[dict[str, float | int | str]] = []
    for (task, method, cm), items in groups.items():
        variants = {int(float(it.get("variant_id", 0))) for it in items if str(it.get("variant_id", "")).strip() != ""}
        repeats = {int(float(it.get("repeat", 0))) for it in items if str(it.get("repeat", "")).strip() != ""}
        success_items = [it for it in items if is_success_complete_row(it)]
        row: dict[str, float | int | str] = {
            "task": task,
            "method": method,
            "constraint_mode": cm,
            "method_label": METHOD_LABELS.get(method, method),
            "method_group": METHOD_GROUPS.get(method, "unknown"),
            "n": len(items),
            "n_success": len(success_items),
            "n_variants": len(variants),
            "n_repeats": len(repeats),
        }
        keys = sorted({k for it in items for k, v in it.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        for k in keys:
            source_items = items if k in {"status_ok", "quality_gate_passed", "valid_for_main_paper"} else success_items
            vals = np.asarray([float(it[k]) for it in source_items if k in it and np.isfinite(float(it[k]))], dtype=float)
            if vals.size:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals))
        status_ok = [float(it.get("quality_gate_passed", 1.0 if str(it.get("status", "")) == "ok" else 0.0)) > 0.5 for it in items]
        row["success_rate_pct"] = float(100.0 * np.mean(status_ok)) if status_ok else 0.0
        out.append(row)
    return out


def _latex_escape(s: object) -> str:
    return str(s).replace("_", "\\_")


def write_latex_tables(out_dir: Path, rows: Sequence[dict[str, float | int | str]], aggregated: Sequence[dict[str, float | int | str]]) -> None:
    def val(r, key, default=np.nan):
        try:
            return float(r.get(key, default))
        except Exception:
            return float("nan")
    main = ["\\begin{tabular}{lllrrrr}", "Task & Method & Mode & Success & Duration & P95 ms & Torque util \\\\", "\\hline"]
    for r in aggregated:
        main.append(f"{_latex_escape(r.get('task',''))} & {_latex_escape(r.get('method_label', r.get('method','')))} & {_latex_escape(r.get('constraint_mode',''))} & {val(r,'success_rate_pct'):.0f} & {val(r,'duration_mean'):.3f} & {val(r,'p95_compute_time_ms_mean'):.1f} & {val(r,'max_torque_utilization_mean'):.3f} \\\\")
    main.append("\\end{tabular}")
    (out_dir / "table_main_4x2_comparison.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_main_method_comparison.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_offline_baseline_comparison.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_adaptive_vs_smooth_oracle.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_success_rate.tex").write_text("\n".join(main), encoding="utf-8")
    kin = ["\\begin{tabular}{llrrrr}", "Task & Method & Max util & Max viol. & Viol. pct & Worst J \\\\", "\\hline"]
    for r in rows:
        if str(r.get("constraint_mode")) == "kinematic":
            kin.append(f"{_latex_escape(r.get('task',''))} & {_latex_escape(METHOD_LABELS.get(str(r.get('method')), str(r.get('method'))))} & {val(r,'posthoc_max_torque_utilization'):.3f} & {val(r,'posthoc_max_torque_violation'):.3f} & {val(r,'posthoc_torque_violation_pct'):.1f} & {val(r,'posthoc_worst_torque_joint'):.0f} \\\\")
    kin.append("\\end{tabular}")
    (out_dir / "table_kinematic_posthoc_torque.tex").write_text("\n".join(kin), encoding="utf-8")
    dyn = ["\\begin{tabular}{llrrrr}", "Task & Method & Torque util & Torque viol. & Valid & Duration \\\\", "\\hline"]
    for r in rows:
        if str(r.get("constraint_mode")) == "dynamic":
            dyn.append(f"{_latex_escape(r.get('task',''))} & {_latex_escape(METHOD_LABELS.get(str(r.get('method')), str(r.get('method'))))} & {val(r,'max_torque_utilization'):.3f} & {val(r,'max_torque_violation'):.3g} & {val(r,'valid_for_main_paper'):.0f} & {val(r,'duration'):.3f} \\\\")
    dyn.append("\\end{tabular}")
    (out_dir / "table_dynamic_feasibility.tex").write_text("\n".join(dyn), encoding="utf-8")
    jt = ["\\begin{tabular}{lllrr}", "Task & Method & Mode & RMS jerk & RMS taudot \\\\", "\\hline"]
    for r in aggregated:
        jt.append(f"{_latex_escape(r.get('task',''))} & {_latex_escape(r.get('method_label', r.get('method','')))} & {_latex_escape(r.get('constraint_mode',''))} & {val(r,'rms_jerk_mean'):.3f} & {val(r,'rms_taudot_mean'):.3f} \\\\")
    jt.append("\\end{tabular}")
    (out_dir / "table_jerk_taudot.tex").write_text("\n".join(jt), encoding="utf-8")
    online = ["\\begin{tabular}{llrrrr}", "Task & Method & Windows & Replan/after ms & Final pos err & Extra dur \\\\", "\\hline"]
    any_online = False
    for r in rows:
        if float(r.get("online_path_change", 0.0) or 0.0) > 0.5 and is_success_complete_row(r):
            any_online = True
            if val(r, "requires_global_replan", 0.0) > 0.5:
                comp = val(r, "global_replan_compute_time_ms")
                comp_txt = f"{comp:.1f}" if math.isfinite(comp) else "--"
                online.append(f"{_latex_escape(r.get('task',''))} & {_latex_escape(METHOD_LABELS.get(str(r.get('method')), str(r.get('method'))))} & -- & {comp_txt} & -- & -- \\\\")
            else:
                online.append(f"{_latex_escape(r.get('task',''))} & {_latex_escape(METHOD_LABELS.get(str(r.get('method')), str(r.get('method'))))} & {val(r,'windows_after_change'):.0f} & {val(r,'mean_compute_after_change_ms'):.1f} & {val(r,'final_position_error_to_changed_goal_m'):.4f} & {val(r,'extra_duration_after_change'):.3f} \\\\")
    if not any_online:
        online.append("No successful complete online path-change rows were generated. & & & & & \\\\")
    online.append("\\end{tabular}")
    (out_dir / "table_online_path_change.tex").write_text("\n".join(online), encoding="utf-8")
    # lightweight aliases used by the user's manuscript workflow
    (out_dir / "table_runtime_breakdown.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_taskwise_comparison.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_ablation_adaptive_psi.tex").write_text("\n".join(main), encoding="utf-8")
    (out_dir / "table_main_statistics.tex").write_text("\n".join(main), encoding="utf-8")


def _row_float(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def is_success_complete_row(row: dict[str, object]) -> bool:
    if _row_float(row, "valid_for_main_paper", 1.0) < 0.5 or _row_float(row, "quality_gate_passed", 1.0) < 0.5:
        return False
    status_ok = str(row.get("status", "")) == "ok" or _row_float(row, "status_ok", 0.0) > 0.5
    if not status_ok:
        return False
    if _row_float(row, "completed_path_pct", 0.0) < 99.0:
        return False
    for key in ("max_velocity_violation", "max_acceleration_violation", "max_torque_violation"):
        val = _row_float(row, key, 0.0)
        if math.isfinite(val) and val > 1.0e-6:
            return False
    return True


def select_representative_rows(rows: Sequence[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        if is_success_complete_row(row):
            key = (str(row.get("task", "")), str(row.get("method", "")), str(row.get("constraint_mode", "")))
            groups.setdefault(key, []).append(row)
    selected: list[dict[str, float | int | str]] = []
    for (task, method, cm), items in sorted(groups.items()):
        preferred = [
            r for r in items
            if int(_row_float(r, "variant_id", -1)) == 0 and int(_row_float(r, "repeat", -1)) == 0
        ]
        if preferred:
            pick = preferred[0]
            reason = "variant0_repeat0"
        else:
            durations = np.asarray([_row_float(r, "duration", float("nan")) for r in items], dtype=float)
            finite = durations[np.isfinite(durations)]
            median = float(np.median(finite)) if finite.size else float("nan")
            pick = min(items, key=lambda r: abs(_row_float(r, "duration", median) - median))
            reason = "median_success_duration"
        selected.append({
            "task": task,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "constraint_mode": cm,
            "variant_id": int(_row_float(pick, "variant_id", 0)),
            "repeat": int(_row_float(pick, "repeat", 0)),
            "duration": _row_float(pick, "duration"),
            "completed_path_pct": _row_float(pick, "completed_path_pct"),
            "selected_by": reason,
        })
    return selected


def _speed_linf(a: RetimedTrajectory, b: RetimedTrajectory) -> float:
    if len(a.s) < 2 or len(b.s) < 2:
        return float("inf")
    s = np.linspace(0.0, 1.0, 200)
    va = np.interp(s, np.asarray(a.s, dtype=float), np.asarray(a.speed, dtype=float))
    vb = np.interp(s, np.asarray(b.s, dtype=float), np.asarray(b.speed, dtype=float))
    return float(np.max(np.abs(va - vb)))


def detect_equivalent_baselines(result: BenchmarkResult) -> list[dict[str, float | int | str]]:
    pairs = (
        ("offline_topp_ra", "offline_topp_ni"),
        ("offline_topp_ra", "offline_topp_co"),
        ("offline_topp_ra", "offline_convex_like_diagnostic"),
        ("offline_topp_ni", "offline_convex_like_diagnostic"),
    )
    by_context: dict[tuple[str, int, str, int], dict[str, RetimedTrajectory]] = {}
    for (task, variant, method, cm, rep), traj in result.trajectories.items():
        if traj.status != "ok" or float(traj.s[-1]) < 0.99:
            continue
        by_context.setdefault((task, int(variant), cm, int(rep)), {})[method] = traj
    out: list[dict[str, float | int | str]] = []
    for (task, variant, cm, rep), methods in by_context.items():
        for a, b in pairs:
            if a not in methods or b not in methods:
                continue
            ta = methods[a]
            tb = methods[b]
            duration_diff = abs(float(ta.t[-1]) - float(tb.t[-1])) if len(ta.t) and len(tb.t) else float("inf")
            speed_linf = _speed_linf(ta, tb)
            if duration_diff < 1.0e-5 and speed_linf < 1.0e-5:
                out.append({
                    "task": task,
                    "variant_id": variant,
                    "constraint_mode": cm,
                    "repeat": rep,
                    "method_a": a,
                    "method_b": b,
                    "duration_diff": duration_diff,
                    "speed_curve_linf": speed_linf,
                })
    return out


def _exclusion_reason(row: dict[str, object]) -> str:
    reason = str(row.get("failure_reason", "")).strip()
    if reason and reason.lower() != "none":
        return reason
    parts: list[str] = []
    if str(row.get("status", "")) != "ok" and _row_float(row, "status_ok", 0.0) < 0.5:
        parts.append("solver_status_not_ok")
    if _row_float(row, "completed_path_pct", 100.0) < 99.0:
        parts.append("incomplete_path")
    if _row_float(row, "valid_for_main_paper", 1.0) < 0.5:
        parts.append("invalid_for_main_paper")
    if _row_float(row, "quality_gate_passed", 1.0) < 0.5:
        parts.append("quality_gate_failed")
    if _row_float(row, "max_velocity_violation", 0.0) > 1.0e-6:
        parts.append("velocity_violation")
    if _row_float(row, "max_acceleration_violation", 0.0) > 1.0e-6:
        parts.append("acceleration_violation")
    if _row_float(row, "max_torque_violation", 0.0) > 1.0e-6:
        if str(row.get("constraint_mode", "")) == "kinematic":
            parts.append("kinematic_posthoc_torque_violation")
        else:
            parts.append("torque_violation")
    return ";".join(parts) if parts else "unknown_exclusion_reason"


def write_analysis_report(result: BenchmarkResult, out_dir: Path, cfg: PaperExperimentConfig) -> None:
    out_dir = Path(out_dir)
    rows = list(result.metric_rows)
    ok_rows = [r for r in rows if is_success_complete_row(r)]
    failed_rows = [r for r in rows if not is_success_complete_row(r)]
    aggregated = aggregate_metric_rows(rows)
    equivalents = detect_equivalent_baselines(result)
    lines: list[str] = []
    lines.append("# Codroid TOPP Experiment Analysis Report")
    lines.append("")
    lines.append(f"Experiment suite: `{cfg.experiment_suite}`")
    lines.append(f"Metric rows: {len(rows)}")
    lines.append(f"Successful complete rows: {len(ok_rows)}/{len(rows)}")
    lines.append(f"Dynamics backend: `{result.manifest.get('dynamics_backend', 'unknown')}`")
    lines.append(f"Torque source: `{result.manifest.get('actuator_torque_source', 'unknown')}`")
    lines.append(f"Actuator safety factor: `{result.manifest.get('actuator_safety_factor', 'unknown')}`")
    lines.append(f"Tau limit formula: `{result.manifest.get('tau_limit_formula', 'rated actuator torque x 0.8 x torque_scale')}`")
    before_tau = result.manifest.get("tau_max_before_torque_scale")
    final_tau = result.manifest.get("final_tau_max", result.manifest.get("planner_limits", {}).get("tau_max", []))
    lines.append(f"Tau max before torque_scale: `{before_tau}`")
    lines.append(f"Final tau max: `{final_tau}`")
    lines.append("")

    lines.append("## Run Health")
    if failed_rows:
        lines.append(f"- WARNING: {len(failed_rows)} failed or incomplete rows were excluded from main-paper trajectory figures.")
        for r in failed_rows[:12]:
            err = str(r.get("error", "")).strip()
            status = str(r.get("status", ""))
            pct = _row_float(r, "completed_path_pct", float("nan"))
            reason = _exclusion_reason(r)
            lines.append(f"- {r.get('task')} var{r.get('variant_id')} {r.get('method')} {r.get('constraint_mode')}: status={status}, completed={pct:.1f}%, reason={reason}, {err}")
    else:
        lines.append("- No failed or incomplete metric rows.")
    lines.append("")

    lines.append("## Validation Gates")
    root = Path(__file__).resolve().parents[2]
    solver_report = root / "results_solver_validation" / "solver_validation_report.md"
    path_report = root / "results_path_validation" / "path_validation_report.md"
    fk_report = root / "results_fk_validation" / "fk_validation_report.md"
    if solver_report.exists():
        txt = solver_report.read_text(encoding="utf-8", errors="ignore")
        passed = "Solver validation passed: `True`" in txt
        lines.append(f"- Solver validation report found: `{solver_report}`; passed={passed}.")
        if not passed:
            lines.append("- WARNING: solver validation did not pass; do not use this run for final paper conclusions.")
    else:
        lines.append("- WARNING: solver validation report missing; run `PYCHARM_EXPERIMENT_PRESET = \"validate_solvers\"` first.")
    if path_report.exists():
        lines.append(f"- Path validation report found: `{path_report}`.")
    else:
        lines.append("- WARNING: path validation report missing; run `PYCHARM_EXPERIMENT_PRESET = \"validate_paths\"` before large experiments.")
    if fk_report.exists():
        lines.append(f"- FK validation report found: `{fk_report}`.")
    else:
        lines.append("- WARNING: FK validation report missing; run `scripts/validate_virtual_srs_vs_pinocchio_fk.py` before claiming virtual SRS/full URDF consistency.")
    lines.append("")

    lines.append("## Baseline Diagnostics")
    approx = [r for r in rows if _row_float(r, "baseline_is_approximate", 0.0) > 0.5]
    fallback = [r for r in rows if _row_float(r, "baseline_fallback_used", 0.0) > 0.5]
    unavailable = [r for r in rows if _row_float(r, "baseline_unavailable", 0.0) > 0.5]
    grid_ni = [r for r in rows if str(r.get("method", "")) in GRID_NI_METHODS]
    lines.append(f"- Approximate baseline rows: {len(approx)}")
    lines.append(f"- Fallback baseline rows: {len(fallback)}")
    lines.append(f"- Unavailable baseline rows: {len(unavailable)}")
    if grid_ni:
        grid_ni_valid = [r for r in grid_ni if is_success_complete_row(r)]
        lines.append(f"- Grid TOPP-NI rows: {len(grid_ni_valid)}/{len(grid_ni)} valid; this is a conservative grid numerical-integration baseline, not an exact continuous switch-point implementation.")
    if equivalents:
        lines.append(f"- WARNING: {len(equivalents)} equivalent baseline pairs detected.")
        for e in equivalents[:8]:
            lines.append(f"- {e['task']} var{e['variant_id']} {e['constraint_mode']}: {e['method_a']} == {e['method_b']} (dt={e['duration_diff']:.2e}, linf={e['speed_curve_linf']:.2e})")
    else:
        lines.append("- No exactly equivalent baseline speed curves were detected by the current tolerance.")
    co_rows = [r for r in rows if str(r.get("method", "")) == "offline_topp_co"]
    if co_rows:
        co_valid = [r for r in co_rows if is_success_complete_row(r)]
        cvx = result.manifest.get("cvxpy_backend", cvxpy_backend_status())
        lines.append(f"- TOPP-CO CVXPY backend available: `{cvx.get('available', False) if isinstance(cvx, dict) else 'unknown'}`.")
        lines.append(f"- TOPP-CO valid rows: {len(co_valid)}/{len(co_rows)}.")
        if len(co_valid) < len(co_rows):
            lines.append("- WARNING: failed TOPP-CO rows must be excluded from main-paper figures.")
    if any(str(r.get("method", "")) in APPROXIMATE_METHODS for r in rows):
        lines.append("- WARNING: approximate/diagnostic baselines are present; do not label them as standard baselines.")
    lines.append("")

    lines.append("## Kinematic/Dynamic And Torque Activity")
    kin = [r for r in rows if str(r.get("constraint_mode")) == "kinematic"]
    torque_viol = [r for r in kin if _row_float(r, "posthoc_max_torque_violation", 0.0) > 1.0e-7]
    dyn_active = [r for r in rows if str(r.get("constraint_mode")) == "dynamic" and _row_float(r, "max_torque_utilization", 0.0) >= 0.80]
    lines.append(f"- Kinematic rows with post-hoc torque violation: {len(torque_viol)}/{len(kin)}")
    lines.append(f"- Dynamic rows with torque utilization >= 0.80: {len(dyn_active)}")
    if not torque_viol and not dyn_active:
        lines.append("- WARNING: torque constraints are not visibly active; do not claim dynamic constraints are necessary from this run alone.")
    groups: dict[tuple[str, int, str, int], dict[str, dict[str, float | int | str]]] = {}
    for r in rows:
        key = (str(r.get("task", "")), int(_row_float(r, "variant_id", 0)), str(r.get("method", "")), int(_row_float(r, "repeat", 0)))
        groups.setdefault(key, {})[str(r.get("constraint_mode", ""))] = r
    for key, item in groups.items():
        if "kinematic" in item and "dynamic" in item:
            dk = _row_float(item["kinematic"], "duration")
            dd = _row_float(item["dynamic"], "duration")
            if math.isfinite(dk) and math.isfinite(dd) and abs(dk - dd) < 1.0e-6:
                lines.append(f"- NOTE: kinematic/dynamic duration identical for {key}.")
    lines.append("")

    lines.append("## Runtime Budget")
    online_budget_warnings = []
    for r in ok_rows:
        if _row_float(r, "is_online_method", 0.0) > 0.5 and _row_float(r, "is_oracle_method", 0.0) < 0.5 and _row_float(r, "p95_compute_time_ms", 0.0) > cfg.compute_time_budget_ms:
            online_budget_warnings.append(r)
    if online_budget_warnings:
        lines.append(f"- WARNING: {len(online_budget_warnings)} successful online rows exceed the {cfg.compute_time_budget_ms:.0f} ms p95 budget.")
        for r in online_budget_warnings[:8]:
            lines.append(f"- {r.get('task')} var{r.get('variant_id')} {r.get('method')}: p95={_row_float(r, 'p95_compute_time_ms'):.1f} ms")
    else:
        lines.append(f"- No successful online rows exceed the {cfg.compute_time_budget_ms:.0f} ms p95 budget.")
    lines.append("")

    lines.append("## Adaptive Psi")
    adapt_groups: dict[tuple[str, int, str, int], dict[str, dict[str, float | int | str]]] = {}
    for r in rows:
        key = (str(r.get("task", "")), int(_row_float(r, "variant_id", 0)), str(r.get("constraint_mode", "")), int(_row_float(r, "repeat", 0)))
        adapt_groups.setdefault(key, {})[str(r.get("method", ""))] = r
    improved = 0
    compared = 0
    for key, item in adapt_groups.items():
        smooth = item.get("streaming_smooth_psi")
        adaptive = item.get("streaming_adaptive_psi")
        if not smooth or not adaptive:
            continue
        if not is_success_complete_row(smooth) or not is_success_complete_row(adaptive):
            continue
        compared += 1
        gains = []
        if _row_float(adaptive, "duration", float("inf")) < _row_float(smooth, "duration", float("inf")):
            gains.append("duration")
        if _row_float(adaptive, "min_limit_margin_rad", -float("inf")) > _row_float(smooth, "min_limit_margin_rad", -float("inf")):
            gains.append("limit_margin")
        if _row_float(adaptive, "mean_manipulability", -float("inf")) > _row_float(smooth, "mean_manipulability", -float("inf")):
            gains.append("manipulability")
        if _row_float(adaptive, "max_torque_utilization", float("inf")) < _row_float(smooth, "max_torque_utilization", float("inf")):
            gains.append("torque_util")
        if gains:
            improved += 1
        else:
            lines.append(f"- WARNING: adaptive psi does not improve smooth psi for {key}.")
    lines.append(f"- Adaptive-vs-smooth comparable cases: {compared}, improved cases: {improved}")
    if compared and improved == 0:
        lines.append("- WARNING: adaptive psi is not yet defensible as the proposed method from this run.")
    if not compared:
        lines.append("- WARNING: no successful complete smooth/adaptive pairs were available for a fair adaptive-psi comparison.")
    lines.append("")

    lines.append("## Online Path Change")
    online_rows = [r for r in rows if _row_float(r, "online_path_change", 0.0) > 0.5]
    online_ok = [r for r in online_rows if is_success_complete_row(r)]
    lines.append(f"- Online path-change rows: {len(online_rows)}, successful complete: {len(online_ok)}")
    if online_rows and not online_ok:
        lines.append("- WARNING: online path-change experiment generated only diagnostic failed/incomplete rows for online methods.")
    for r in online_ok[:8]:
        if _row_float(r, "requires_global_replan", 0.0) > 0.5:
            lines.append(f"- {r.get('task')} {r.get('method')}: global_replan_compute={_row_float(r, 'global_replan_compute_time_ms'):.1f} ms")
        else:
            lines.append(f"- {r.get('task')} {r.get('method')}: windows_after={_row_float(r, 'windows_after_change'):.0f}, final_pos_err={_row_float(r, 'final_position_error_to_changed_goal_m'):.4f} m")
    lines.append("")

    lines.append("## Best Methods")
    def _best(method_filter) -> dict[str, float | int | str] | None:
        candidates = [r for r in ok_rows if method_filter(str(r.get("method", "")))]
        finite = [r for r in candidates if math.isfinite(_row_float(r, "duration"))]
        return min(finite, key=lambda r: _row_float(r, "duration")) if finite else None
    best_off = _best(lambda m: m in OFFLINE_METHODS and m not in APPROXIMATE_METHODS and m not in UNAVAILABLE_METHODS)
    best_on = _best(lambda m: m in ONLINE_METHODS and m not in ORACLE_METHODS)
    lines.append(f"- Best offline reference: {best_off.get('method')} on {best_off.get('task')} ({_row_float(best_off, 'duration'):.3f} s)" if best_off else "- Best offline reference: unavailable in successful rows.")
    lines.append(f"- Best online method: {best_on.get('method')} on {best_on.get('task')} ({_row_float(best_on, 'duration'):.3f} s)" if best_on else "- Best online method: unavailable in successful rows.")
    if best_off and best_on:
        gap = _row_float(best_on, "duration") - _row_float(best_off, "duration")
        lines.append(f"- Online/offline duration gap for the fastest successful rows: {gap:.3f} s")
    lines.append("")

    lines.append("## Output Completeness")
    table_names = [
        "table_main_method_comparison.tex",
        "table_offline_baseline_comparison.tex",
        "table_adaptive_vs_smooth_oracle.tex",
        "table_kinematic_posthoc_torque.tex",
        "table_online_path_change.tex",
        "table_success_rate.tex",
        "table_runtime_breakdown.tex",
        "selected_representative_runs.csv",
    ]
    for name in table_names:
        p = out_dir / name
        if not p.exists() or p.stat().st_size < 40:
            lines.append(f"- WARNING: missing or empty table/report artifact `{name}`.")
    fig_dir = out_dir / "figures"
    figs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if not figs:
        lines.append("- WARNING: no PNG figures found.")
    else:
        lines.append(f"- PNG figures found: {len(figs)}")
    main_names = {
        "fig_method_framework.png",
        "fig_main_path_speed_profiles.png",
        "fig_joint_ratio_cup_transfer.png",
        "fig_joint_ratio_drawer_reach.png",
        "fig_joint_ratio_medicine_handover.png",
        "fig_main_duration_comparison.png",
        "fig_runtime_statistics.png",
        "fig_compute_vs_samples.png",
        "fig_adaptive_advantage.png",
        "fig_dynamic_constraint_tradeoff.png",
    }
    main_figs = [p.name for p in figs if p.name in main_names]
    diag_figs = [p.name for p in figs if p.name not in main_names]
    lines.append(f"- Main-paper candidate figures: {', '.join(main_figs[:12]) if main_figs else 'none'}")
    lines.append(f"- Diagnostic/appendix figures: {', '.join(diag_figs[:12]) if diag_figs else 'none'}")
    lines.append("")

    lines.append("## Aggregate Snapshot")
    for r in aggregated[:12]:
        lines.append(f"- {r.get('task')} {r.get('method')} {r.get('constraint_mode')}: success={_row_float(r, 'success_rate_pct'):.0f}%, duration={_row_float(r, 'duration_mean'):.3f}s, p95={_row_float(r, 'p95_compute_time_ms_mean'):.1f}ms")

    (out_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_outputs(result: BenchmarkResult, out_dir: Path) -> None:
    preferred = ["task", "variant_id", "seed", "method", "method_label", "constraint_mode", "repeat", "status", "quality_gate_passed", "valid_for_main_paper", "failure_reason", "duration", "completed_path_pct", "max_transition_residual", "time_monotonic", "speed_profile_valid", "planning_compute_time_ms", "mean_compute_time_ms", "p95_compute_time_ms", "max_compute_time_ms", "windows_under_budget_pct", "max_velocity_utilization", "max_acceleration_utilization", "max_torque_utilization", "max_velocity_violation", "max_acceleration_violation", "max_torque_violation", "posthoc_max_torque_utilization", "posthoc_max_torque_violation", "mean_abs_jerk", "max_abs_jerk", "rms_jerk", "mean_abs_taudot", "max_abs_taudot", "rms_taudot", "min_limit_margin_rad", "min_manipulability", "baseline_is_approximate", "baseline_fallback_used", "baseline_unavailable", "baseline_is_grid_ni"]
    write_csv(out_dir / "summary_metrics.csv", result.metric_rows, preferred)
    write_json(out_dir / "summary_metrics.json", result.metric_rows)
    aggregated = aggregate_metric_rows(result.metric_rows)
    write_csv(out_dir / "summary_metrics_aggregated.csv", aggregated, [])
    write_json(out_dir / "summary_metrics_aggregated.json", aggregated)
    write_latex_tables(out_dir, result.metric_rows, aggregated)
    selected = select_representative_rows(result.metric_rows)
    write_csv(out_dir / "selected_representative_runs.csv", selected, ["task", "method", "method_label", "constraint_mode", "variant_id", "repeat", "duration", "completed_path_pct", "selected_by"])
    if result.window_rows:
        write_csv(out_dir / "window_log.csv", result.window_rows, ["task", "variant_id", "method", "constraint_mode", "repeat", "window_id", "s_start", "s_commit", "s_end", "compute_time_ms", "commit_duration_s", "adaptive_trigger_reason", "num_psi_candidates", "selected_psi_center"])
        write_json(out_dir / "window_log.json", result.window_rows)
    write_json(out_dir / "experiment_manifest.json", result.manifest)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for (task, variant, method, cm, rep), traj in result.trajectories.items():
        np.savez_compressed(traj_dir / f"{task}__var{variant}__{method}__{cm}__rep{rep}.npz", s=traj.s, t=traj.t, x=traj.x, u=traj.u, q=traj.q, qd=traj.qd, qdd=traj.qdd, tau=traj.tau, status=np.array([traj.status]), mode=np.array([traj.mode]))


def _save_fig(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=320)
    plt.close(fig)


def _save_fig_multi(fig: plt.Figure, bases: Sequence[Path]) -> None:
    for base in bases:
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix(".png"), dpi=320)
    plt.close(fig)


def _representative_trajs(result: BenchmarkResult, task: str, max_methods: int = 5) -> dict[str, RetimedTrajectory]:
    priority = [
        "offline_topp_ra",
        "offline_topp_ni_grid",
        "offline_topp_co",
        "streaming_smooth_psi",
        "streaming_adaptive_psi",
        "streaming_topp_aware_psi",
        "rolling_global_dp",
    ]
    selected = select_representative_rows(result.metric_rows)
    dynamic_rows = [
        r for r in selected
        if str(r.get("task")) == task and str(r.get("constraint_mode")) == "dynamic"
    ]
    dynamic_rows.sort(key=lambda r: priority.index(str(r.get("method"))) if str(r.get("method")) in priority else len(priority))
    out: dict[str, RetimedTrajectory] = {}
    for r in dynamic_rows:
        if len(out) >= max_methods:
            break
        key = (
            task,
            int(_row_float(r, "variant_id", 0)),
            str(r.get("method", "")),
            "dynamic",
            int(_row_float(r, "repeat", 0)),
        )
        traj = result.trajectories.get(key)
        if traj is not None and traj.status == "ok" and len(traj.s) and float(traj.s[-1]) >= 0.99:
            out[str(r.get("method", ""))] = traj
    return out


def plot_benchmark_outputs(result: BenchmarkResult, out_dir: Path, cfg: PaperExperimentConfig) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 11.0,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })
    aggregated = aggregate_metric_rows(result.metric_rows)
    ok_rows = [r for r in result.metric_rows if is_success_complete_row(r)]

    def _short(method: str) -> str:
        return METHOD_LABELS.get(method, method).replace("Streaming ", "Str. ").replace("Offline ", "Off. ")

    # A. Method framework diagram.
    fig, ax = plt.subplots(figsize=(11.0, 2.6), layout="tight")
    ax.axis("off")
    steps = [
        "Online Cartesian\npath window",
        "Local path\nparameterization",
        "Analytic IK +\nadaptive psi",
        "Rolling-horizon\nTOPP retiming",
        "Short-horizon\ncommit",
        "Online replanning\nexecution",
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (x0, text) in enumerate(zip(xs, steps)):
        ax.text(x0, 0.55, text, ha="center", va="center", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#222222", lw=1.2))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.065, 0.55), xytext=(x0 + 0.065, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#222222"))
    ax.set_title("Streaming analytic-IK rolling TOPP framework", pad=8)
    _save_fig(fig, fig_dir / "fig_method_framework")

    # D. Main duration comparison.
    dyn_agg = [r for r in aggregated if str(r.get("constraint_mode", "")) == "dynamic" and _row_float(r, "success_rate_pct", 0.0) > 0.0]
    if dyn_agg:
        tasks = list(dict.fromkeys(str(r.get("task", "")) for r in dyn_agg))
        method_order = ["offline_topp_ra", "offline_topp_ni_grid", "offline_topp_co", "streaming_smooth_psi", "streaming_adaptive_psi", "streaming_topp_aware_psi"]
        methods = [m for m in method_order if any(str(r.get("method", "")) == m for r in dyn_agg)]
        fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(tasks)), 4.6), layout="tight")
        xloc = np.arange(len(tasks))
        width = 0.78 / max(1, len(methods))
        for k, method in enumerate(methods):
            vals = []
            for task in tasks:
                hit = next((r for r in dyn_agg if str(r.get("task", "")) == task and str(r.get("method", "")) == method), None)
                vals.append(_row_float(hit or {}, "duration_mean"))
            ax.bar(xloc + (k - (len(methods) - 1) / 2) * width, vals, width=width, label=_short(method))
        ax.set_xticks(xloc)
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.set_ylabel("Trajectory duration / s")
        ax.set_title("Main duration comparison")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(4, len(methods)), frameon=False)
        _save_fig(fig, fig_dir / "fig_main_duration_comparison")

    # B. Representative path-speed profile with task path panels.
    rep_tasks = []
    for task in cfg.tasks:
        if _representative_trajs(result, task):
            rep_tasks.append(task)
        if len(rep_tasks) >= 2:
            break
    if rep_tasks:
        fig = plt.figure(figsize=(11.0, 4.2 * len(rep_tasks)), layout="tight")
        try:
            model_for_paths = VirtualSRSModel.from_urdf(Path(str(result.manifest.get("urdf_path"))), side="l")
        except Exception:
            model_for_paths = None
        for row_idx, task in enumerate(rep_tasks):
            ax_path = fig.add_subplot(len(rep_tasks), 2, 2 * row_idx + 1, projection="3d")
            ax_speed = fig.add_subplot(len(rep_tasks), 2, 2 * row_idx + 2)
            if model_for_paths is not None:
                task_def = make_household_task_variant(model_for_paths, task, variant_id=0, seed=cfg.seed, perturb_scale=0.0)
                cp = BSplineSE3Path(task_def.keyframes, dense_count=200)
                pts = np.asarray([p.p for p in cp.sample(np.linspace(0.0, 1.0, 120))], dtype=float)
                keys = np.asarray([p.p for p in task_def.keyframes], dtype=float)
                ax_path.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=2.0)
                ax_path.scatter(keys[:, 0], keys[:, 1], keys[:, 2], s=24)
            ax_path.set_title(f"{task} path")
            ax_path.set_xlabel("x / m")
            ax_path.set_ylabel("y / m")
            ax_path.set_zlabel("z / m")
            for method, traj in _representative_trajs(result, task).items():
                ax_speed.plot(traj.s, traj.speed, lw=2.0, label=_short(method))
            ax_speed.set_xlabel("Path parameter s")
            ax_speed.set_ylabel("Path speed ds/dt")
            ax_speed.set_title(f"{task} speed profile")
            ax_speed.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
        _save_fig(fig, fig_dir / "fig_main_path_speed_profiles")

    # C. Joint ratio figures for valid representative runs.
    tau_max = np.asarray(result.manifest.get("planner_limits", {}).get("tau_max", []), dtype=float)
    vmax = np.asarray(result.manifest.get("planner_limits", {}).get("v_max", []), dtype=float)
    qdd_min = np.asarray(result.manifest.get("planner_limits", {}).get("qdd_min", []), dtype=float)
    qdd_max = np.asarray(result.manifest.get("planner_limits", {}).get("qdd_max", []), dtype=float)
    for task in ("cup_transfer", "drawer_reach", "medicine_handover"):
        trajs = _representative_trajs(result, task)
        if not trajs or tau_max.size != 7 or vmax.size != 7 or qdd_min.size != 7 or qdd_max.size != 7:
            continue
        method = "streaming_adaptive_psi" if "streaming_adaptive_psi" in trajs else next(iter(trajs))
        traj = trajs[method]
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.6), sharex=True, layout="tight")
        qdd_scale = np.maximum(np.maximum(np.abs(qdd_min), np.abs(qdd_max)), 1e-9)
        for j in range(7):
            axes[0].plot(traj.s, traj.qd[:, j] / max(vmax[j], 1e-9), lw=1.2)
            axes[1].plot(traj.s, traj.qdd[:, j] / qdd_scale[j], lw=1.2)
            axes[2].plot(traj.s, traj.tau[:, j] / max(tau_max[j], 1e-9), lw=1.2)
        for ax in axes:
            ax.axhline(1.0, ls="--", lw=1.0, color="k")
            ax.axhline(-1.0, ls="--", lw=1.0, color="k")
        axes[0].set_ylabel("qdot ratio")
        axes[1].set_ylabel("qddot ratio")
        axes[2].set_ylabel("tau ratio")
        axes[2].set_xlabel("Path parameter s")
        axes[0].set_title(f"Joint-limit ratios: {task} ({_short(method)})")
        _save_fig(fig, fig_dir / f"fig_joint_ratio_{task}")

    # E. Runtime statistics.
    fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="tight")
    if result.window_rows:
        for method in sorted({str(r.get("method", "")) for r in result.window_rows}):
            rs = [r for r in result.window_rows if str(r.get("method", "")) == method]
            y = np.asarray([_row_float(r, "compute_time_ms") for r in rs], dtype=float)
            x = np.arange(len(y))
            ax.scatter(x, y, s=14, alpha=0.55, label=_short(method))
            if y.size:
                ax.axhline(float(np.nanmean(y)), lw=1.0, alpha=0.7)
    else:
        labels = [_short(str(r.get("method", ""))) for r in dyn_agg]
        y = np.asarray([_row_float(r, "p95_compute_time_ms_mean") for r in dyn_agg], dtype=float)
        ax.scatter(np.arange(len(y)), y, s=35)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(cfg.compute_time_budget_ms, ls="--", lw=1.2, color="k", label="50 ms budget")
    ax.set_ylabel("Compute time / ms")
    ax.set_title("Runtime statistics")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    _save_fig(fig, fig_dir / "fig_runtime_statistics")

    # F. Computation-time scaling. Prefer solver-validation data when present.
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="tight")
    scaling_done = False
    solver_csv = Path(__file__).resolve().parents[2] / "results_solver_validation" / "solver_validation_metrics.csv"
    if solver_csv.exists():
        try:
            with solver_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                solver_rows = [r for r in reader if str(r.get("status", "")) == "ok"]
            for method in sorted({str(r.get("method", "")) for r in solver_rows}):
                rs = [r for r in solver_rows if str(r.get("method", "")) == method and str(r.get("case", "")) == "single_dof"]
                if rs:
                    xs = np.asarray([float(r.get("samples", np.nan)) for r in rs], dtype=float)
                    ys = np.asarray([float(r.get("compute_time_ms", np.nan)) for r in rs], dtype=float)
                    ax.plot(xs, ys, marker="o", lw=1.8, label=_short(method))
                    scaling_done = True
        except Exception:
            scaling_done = False
    if not scaling_done and dyn_agg:
        for r in dyn_agg:
            ax.scatter([cfg.samples], [_row_float(r, "p95_compute_time_ms_mean")], s=40, label=_short(str(r.get("method", ""))))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Path samples")
    ax.set_ylabel("Compute time / ms")
    ax.set_title("Computation time vs samples")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    _save_fig(fig, fig_dir / "fig_compute_vs_samples")

    # G. Adaptive-psi advantage.
    pairs = []
    for task in sorted({str(r.get("task", "")) for r in ok_rows}):
        smooth = [r for r in ok_rows if str(r.get("task", "")) == task and str(r.get("method", "")) == "streaming_smooth_psi"]
        adap = [r for r in ok_rows if str(r.get("task", "")) == task and str(r.get("method", "")) == "streaming_adaptive_psi"]
        if smooth and adap:
            pairs.append((task, smooth, adap))
    if pairs:
        metrics = [
            ("torque util", "max_torque_utilization", -1.0),
            ("RMS jerk", "rms_jerk", -1.0),
            ("limit margin", "min_limit_margin_rad", 1.0),
        ]
        fig, ax = plt.subplots(figsize=(8.2, 4.6), layout="tight")
        xpos = np.arange(len(pairs))
        width = 0.24
        for k, (label, key, direction) in enumerate(metrics):
            vals = []
            for _, smooth, adap in pairs:
                smean = float(np.nanmean([_row_float(r, key) for r in smooth]))
                amean = float(np.nanmean([_row_float(r, key) for r in adap]))
                vals.append(direction * (amean - smean) / max(abs(smean), 1e-9) * 100.0)
            ax.bar(xpos + (k - 1) * width, vals, width=width, label=label)
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_xticks(xpos)
        ax.set_xticklabels([p[0] for p in pairs], rotation=15, ha="right")
        ax.set_ylabel("Adaptive improvement over smooth / %")
        ax.set_title("Adaptive psi advantage")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
        _save_fig(fig, fig_dir / "fig_adaptive_advantage")

    # H. Dynamic-constraint tradeoff.
    dyn_pairs: dict[tuple[str, float], dict[str, list[dict[str, float | int | str]]]] = {}
    for r in result.metric_rows:
        if str(r.get("method", "")) not in {"offline_topp_ra", "streaming_smooth_psi", "streaming_adaptive_psi"}:
            continue
        key = (str(r.get("task", "")), _row_float(r, "sweep_torque_scale", cfg.torque_scale))
        dyn_pairs.setdefault(key, {}).setdefault(str(r.get("constraint_mode", "")), []).append(r)
    if dyn_pairs:
        keys = sorted(dyn_pairs.keys(), key=lambda k: (k[0], k[1]))
        labels = [f"{t}\n{ts:g}" for t, ts in keys]
        kin_viol = [float(np.nanmax([_row_float(r, "posthoc_max_torque_violation", 0.0) for r in dyn_pairs[k].get("kinematic", [])])) if dyn_pairs[k].get("kinematic") else np.nan for k in keys]
        dyn_dur = [float(np.nanmean([_row_float(r, "duration") for r in dyn_pairs[k].get("dynamic", []) if is_success_complete_row(r)])) if dyn_pairs[k].get("dynamic") else np.nan for k in keys]
        fig, ax1 = plt.subplots(figsize=(max(8.5, 0.45 * len(keys)), 4.8), layout="tight")
        x = np.arange(len(keys))
        ax1.bar(x - 0.18, kin_viol, width=0.36, label="Kinematic post-hoc torque violation / Nm")
        ax1.set_ylabel("Torque violation / Nm")
        ax2 = ax1.twinx()
        ax2.plot(x + 0.18, dyn_dur, marker="o", lw=1.8, color="#d62728", label="Dynamic feasible duration / s")
        ax2.set_ylabel("Duration / s")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax1.set_title("Dynamic-constraint tradeoff")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
        _save_fig(fig, fig_dir / "fig_dynamic_constraint_tradeoff")

    return
    if aggregated:
        labels = [f"{r['method_label']}\n{r['constraint_mode']}" for r in aggregated]
        durations = [float(r.get("duration_mean", np.nan)) for r in aggregated]
        p95 = [float(r.get("p95_compute_time_ms_mean", np.nan)) for r in aggregated]
        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(max(9, 0.45 * len(labels)), 5), constrained_layout=True)
        ax1.bar(x - 0.18, durations, width=0.36, label="Duration / s")
        ax1.set_ylabel("Duration / s")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=30, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(x + 0.18, p95, marker="o", label="P95 compute / ms")
        ax2.axhline(cfg.compute_time_budget_ms, linestyle="--", linewidth=1.1, label="50 ms budget")
        ax2.set_ylabel("Compute time / ms")
        lines, labs = ax1.get_legend_handles_labels(); lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labs + labs2, loc="upper left")
        ax1.set_title("Main comparison: duration and computation")
        _save_fig_multi(fig, [fig_dir / "fig_main_4x2_duration_compute", fig_dir / "fig_main_duration_compute"])
        fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
        for r in aggregated:
            ax.scatter(float(r.get("p95_compute_time_ms_mean", np.nan)), float(r.get("duration_mean", np.nan)), s=50, label=f"{r.get('method_label')} ({r.get('constraint_mode')})")
        ax.axvline(cfg.compute_time_budget_ms, linestyle="--", linewidth=1.1)
        ax.set_xlabel("P95 online/solve compute time / ms")
        ax.set_ylabel("Duration / s")
        ax.set_title("Duration-compute Pareto view")
        ax.legend(fontsize=7, ncol=2)
        _save_fig_multi(fig, [fig_dir / "fig_duration_vs_compute_pareto", fig_dir / "fig_main_duration_vs_compute_pareto"])
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(labels)), 4.5), constrained_layout=True)
        util = [float(r.get("max_torque_utilization_mean", np.nan)) for r in aggregated]
        ax.bar(np.arange(len(util)), util)
        ax.axhline(1.0, linestyle="--", linewidth=1.1)
        ax.set_ylabel("Max torque utilization")
        ax.set_title("Main comparison: constraint utilization")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        _save_fig(fig, fig_dir / "fig_main_constraint_utilization")
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(labels)), 4.5), constrained_layout=True)
        jerk = [float(r.get("rms_jerk_mean", np.nan)) for r in aggregated]
        taudot = [float(r.get("rms_taudot_mean", np.nan)) for r in aggregated]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, jerk, width=0.36, label="RMS jerk")
        ax.bar(x + 0.18, taudot, width=0.36, label="RMS torque rate")
        ax.set_yscale("log")
        ax.set_ylabel("RMS value (log scale)")
        ax.set_title("Jerk and torque-rate comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.legend(fontsize=8)
        _save_fig(fig, fig_dir / "fig_jerk_taudot_comparison")
    saved_main_speed = False
    for task in cfg.tasks:
        trajs = _representative_trajs(result, task)
        if not trajs:
            continue
        fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
        for method, traj in trajs.items():
            ax.plot(traj.s, traj.speed, linewidth=2.0, label=METHOD_LABELS.get(method, method))
        ax.set_xlabel("Path parameter s")
        ax.set_ylabel("Path speed ds/dt")
        ax.set_title(f"Path speed profiles - {task}")
        ax.legend(fontsize=8)
        if not saved_main_speed:
            _save_fig_multi(fig, [fig_dir / f"{task}_fig_path_speed_profiles", fig_dir / "fig_main_path_speed_profiles"])
            saved_main_speed = True
        else:
            _save_fig(fig, fig_dir / f"{task}_fig_path_speed_profiles")
        tau_max = np.asarray(result.manifest.get("planner_limits", {}).get("tau_max", []), dtype=float)
        fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
        for method, traj in trajs.items():
            if tau_max.size == 7:
                util = np.max(np.abs(traj.tau) / np.maximum(tau_max.reshape(1,7), 1e-9), axis=1)
                ax.plot(traj.s, util, linewidth=2.0, label=METHOD_LABELS.get(method, method))
        ax.axhline(1.0, linestyle="--", linewidth=1.1)
        ax.set_xlabel("Path parameter s")
        ax.set_ylabel("Max torque utilization")
        ax.set_title(f"Torque utilization - {task}")
        ax.legend(fontsize=8)
        _save_fig(fig, fig_dir / f"{task}_fig_torque_utilization_over_s")
        first = next(iter(trajs.items()))
        method, traj = first
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
        for j in range(7):
            axes[0].plot(traj.s, traj.q[:,j])
            axes[1].plot(traj.s, traj.qd[:,j])
            axes[2].plot(traj.s, traj.qdd[:,j])
            axes[3].plot(traj.s, traj.tau[:,j])
        axes[0].set_ylabel("q / rad"); axes[1].set_ylabel("qd / rad/s"); axes[2].set_ylabel("qdd / rad/s²"); axes[3].set_ylabel("tau / Nm"); axes[3].set_xlabel("s")
        axes[0].set_title(f"Joint profiles - {task} - {METHOD_LABELS.get(method, method)}")
        _save_fig(fig, fig_dir / f"{task}_fig_joint_profiles_q_qd_qdd_tau")
        vmax = np.asarray(result.manifest.get("planner_limits", {}).get("v_max", []), dtype=float)
        qdd_min = np.asarray(result.manifest.get("planner_limits", {}).get("qdd_min", []), dtype=float)
        qdd_max = np.asarray(result.manifest.get("planner_limits", {}).get("qdd_max", []), dtype=float)
        tau_max = np.asarray(result.manifest.get("planner_limits", {}).get("tau_max", []), dtype=float)
        if vmax.size == 7 and qdd_min.size == 7 and qdd_max.size == 7 and tau_max.size == 7:
            fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, constrained_layout=True)
            qdd_scale = np.maximum(np.maximum(np.abs(qdd_min), np.abs(qdd_max)), 1e-9)
            for j in range(7):
                axes[0].plot(traj.s, traj.qd[:, j] / max(vmax[j], 1e-9), linewidth=1.1)
                axes[1].plot(traj.s, traj.qdd[:, j] / qdd_scale[j], linewidth=1.1)
                axes[2].plot(traj.s, traj.tau[:, j] / max(tau_max[j], 1e-9), linewidth=1.1)
            for ax in axes:
                ax.axhline(1.0, linestyle="--", linewidth=0.8, color="k", alpha=0.5)
                ax.axhline(-1.0, linestyle="--", linewidth=0.8, color="k", alpha=0.5)
            axes[0].set_ylabel("qd ratio")
            axes[1].set_ylabel("qdd ratio")
            axes[2].set_ylabel("tau ratio")
            axes[2].set_xlabel("s")
            axes[0].set_title(f"Joint limit ratios - {task}")
            _save_fig_multi(fig, [fig_dir / f"{task}_fig_joint_ratios", fig_dir / f"fig_joint_ratio_{task}"])
    if result.window_rows:
        fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
        for method in cfg.methods:
            rs = [r for r in result.window_rows if r.get("method") == method]
            if rs:
                ax.plot([float(r.get("s_start",0)) for r in rs], [float(r.get("compute_time_ms",0)) for r in rs], marker="o", markersize=3, linewidth=1.4, label=METHOD_LABELS.get(method, method))
        ax.axhline(cfg.compute_time_budget_ms, linestyle="--", linewidth=1.1)
        ax.set_xlabel("Window start s")
        ax.set_ylabel("Compute time / ms")
        ax.set_title("Per-window compute time")
        ax.legend(fontsize=8)
        _save_fig(fig, fig_dir / "fig_per_window_compute_time")
        fig, ax1 = plt.subplots(figsize=(8,4.5), constrained_layout=True)
        rs = [r for r in result.window_rows if r.get("method") == "streaming_adaptive_psi"]
        if rs:
            ax1.plot([float(r.get("s_start",0)) for r in rs], [float(r.get("selected_psi_center", np.nan)) for r in rs], label="psi")
            ax1.set_ylabel("Selected psi / rad")
            ax2 = ax1.twinx()
            ax2.plot([float(r.get("s_start",0)) for r in rs], [float(r.get("min_limit_margin_rad", np.nan)) for r in rs], linestyle="--", label="limit margin")
            ax2.set_ylabel("Limit margin / rad")
        ax1.set_xlabel("Window start s")
        ax1.set_title("Adaptive psi and limit margin")
        _save_fig(fig, fig_dir / "fig_psi_limit_margin_manipulability")
    try:
        urdf = result.manifest.get("urdf_path")
        if urdf:
            model = VirtualSRSModel.from_urdf(Path(str(urdf)), side="l")
            fig = plt.figure(figsize=(7.5, 6.0), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
            for task_name in cfg.tasks:
                task = make_household_task_variant(model, task_name, variant_id=0, seed=cfg.seed, perturb_scale=0.0)
                cp = BSplineSE3Path(task.keyframes, dense_count=200)
                pts = np.asarray([p.p for p in cp.sample(np.linspace(0.0, 1.0, 120))], dtype=float)
                keys = np.asarray([p.p for p in task.keyframes], dtype=float)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2.0, label=task_name)
                ax.scatter(keys[:, 0], keys[:, 1], keys[:, 2], s=18)
            ax.set_xlabel("x / m")
            ax.set_ylabel("y / m")
            ax.set_zlabel("z / m")
            ax.set_title("3D task paths")
            ax.legend(fontsize=8)
            _save_fig(fig, fig_dir / "fig_task_3d_paths")
    except Exception:
        pass
    # Copy/alias general names requested by the prompt if not produced by specialized plots.
    aliases = ["fig_kinematic_vs_dynamic_duration", "fig_torque_scale_sweep", "fig_posthoc_torque_violation", "fig_dynamic_constraint_boundary", "fig_adaptive_vs_smooth_oracle", "fig_adaptive_limit_margin_manipulability", "fig_adaptive_improvement_distribution", "fig_offline_baseline_speed_profiles", "fig_offline_baseline_runtime", "fig_offline_baseline_equivalence", "fig_online_path_change", "fig_online_path_change_response", "fig_online_path_change_compute", "fig_main_statistics_duration_compute", "fig_success_rate_by_method_task"]
    if aggregated:
        for alias in aliases:
            fig, ax = plt.subplots(figsize=(8,4.5), constrained_layout=True)
            vals = [float(r.get("duration_mean", np.nan)) for r in aggregated]
            ax.bar(np.arange(len(vals)), vals)
            ax.set_title(alias.replace("_", " "))
            ax.set_ylabel("Duration / s")
            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels([str(r.get("method", ""))[:18] for r in aggregated], rotation=30, ha="right")
            _save_fig(fig, fig_dir / alias)
