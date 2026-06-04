#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codroid_topp.experiments.paper_benchmark import cvxpy_backend_status, plan_offline_topp_co, plan_offline_topp_ni_like
from codroid_topp.planning.psi_optimizer import JointPath
from codroid_topp.planning.rolling_topp import RollingHorizonTOPP, TOPPConfig, RetimedTrajectory

# ======================= PyCharm direct-run settings =======================
USE_PYCHARM_CONFIG = True
PYCHARM_URDF = ROOT / "configs" / "codroidRobot.urdf"
PYCHARM_OUT = ROOT / "results_solver_validation"
# ==========================================================================


@dataclass
class SyntheticDynamics:
    v_max: np.ndarray
    tau_max: np.ndarray

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        # A deterministic affine model. Torque limits can be used as a dynamic
        # acceleration bound because tau = qdd + light velocity damping.
        return np.asarray(qdd, dtype=float) + 0.02 * np.asarray(qd, dtype=float)


def make_synthetic_path(samples: int, kind: str) -> JointPath:
    s = np.linspace(0.0, 1.0, int(samples))
    q = np.zeros((len(s), 7), dtype=float)
    if kind == "single_dof":
        q[:, 0] = s
    else:
        q[:, 0] = s
        q[:, 1] = 0.25 * np.sin(np.pi * s)
        q[:, 2] = -0.15 * np.sin(2.0 * np.pi * s)
    qs = np.gradient(q, s, axis=0, edge_order=2)
    qss = np.gradient(qs, s, axis=0, edge_order=2)
    return JointPath(
        s=s,
        q=q,
        qs=qs,
        qss=qss,
        psi=np.zeros(len(s)),
        ik_position_error=np.zeros(len(s)),
        ik_rotation_error=np.zeros(len(s)),
        min_limit_margin=1.0,
    )


def transition_residual(traj: RetimedTrajectory) -> float:
    if len(traj.s) < 2:
        return float("nan")
    ds = np.maximum(np.diff(traj.s), 1e-12)
    res = traj.x[1:] - traj.x[:-1] - 2.0 * ds * traj.u[:-1]
    return float(np.max(np.abs(res)))


def constraint_metrics(traj: RetimedTrajectory, planner: RollingHorizonTOPP) -> dict[str, float]:
    vmax = np.asarray(planner.v_max, dtype=float).reshape(7)
    qdd_min = np.asarray(planner.qdd_min, dtype=float).reshape(7)
    qdd_max = np.asarray(planner.qdd_max, dtype=float).reshape(7)
    tau_max = np.asarray(planner.tau_max, dtype=float).reshape(7)
    qd = np.abs(np.asarray(traj.qd, dtype=float))
    qdd = np.asarray(traj.qdd, dtype=float)
    tau = np.abs(np.asarray(traj.tau, dtype=float))
    return {
        "duration": float(traj.duration),
        "max_velocity_violation": float(np.max(np.maximum(qd - vmax[None, :], 0.0))),
        "max_acceleration_violation": float(np.max(np.maximum(qdd - qdd_max[None, :], 0.0) + np.maximum(qdd_min[None, :] - qdd, 0.0))),
        "max_torque_violation": float(np.max(np.maximum(tau - tau_max[None, :], 0.0))),
        "transition_residual": transition_residual(traj),
        "start_speed_error": float(abs(math.sqrt(max(float(traj.x[0]), 0.0)))),
        "goal_speed_error": float(abs(math.sqrt(max(float(traj.x[-1]), 0.0)))),
        "time_monotonic": 1.0 if len(traj.t) <= 1 or bool(np.all(np.diff(traj.t) >= -1e-12)) else 0.0,
    }


def speed_linf(a: RetimedTrajectory, b: RetimedTrajectory) -> float:
    grid = np.linspace(0.0, 1.0, 200)
    va = np.interp(grid, a.s, a.speed)
    vb = np.interp(grid, b.s, b.speed)
    return float(np.max(np.abs(va - vb)))


def save_line_chart(path_base: Path, series: dict[str, tuple[np.ndarray, np.ndarray]], title: str, xlabel: str, ylabel: str) -> None:
    width, height = 1100, 700
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    all_x = np.concatenate([v[0] for v in series.values() if len(v[0])]) if series else np.array([0.0, 1.0])
    all_y = np.concatenate([v[1] for v in series.values() if len(v[1])]) if series else np.array([0.0, 1.0])
    all_x = all_x[np.isfinite(all_x)]
    all_y = all_y[np.isfinite(all_y)]
    xmin, xmax = (float(np.min(all_x)), float(np.max(all_x))) if all_x.size else (0.0, 1.0)
    ymin, ymax = (0.0, float(np.max(all_y)) * 1.1) if all_y.size else (0.0, 1.0)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    def xp(x: float) -> int:
        return int(margin_l + (x - xmin) / (xmax - xmin) * (width - margin_l - margin_r))

    def yp(y: float) -> int:
        return int(height - margin_b - (y - ymin) / (ymax - ymin) * (height - margin_t - margin_b))

    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], outline=(0, 0, 0), width=2)
    draw.text((margin_l, 20), title, fill=(0, 0, 0))
    draw.text((width // 2 - 60, height - 45), xlabel, fill=(0, 0, 0))
    draw.text((15, height // 2), ylabel, fill=(0, 0, 0))
    for i in range(6):
        x = margin_l + i * (width - margin_l - margin_r) / 5
        y = margin_t + i * (height - margin_t - margin_b) / 5
        draw.line([(x, margin_t), (x, height - margin_b)], fill=(230, 230, 230))
        draw.line([(margin_l, y), (width - margin_r, y)], fill=(230, 230, 230))
    legend_y = margin_t + 8
    for idx, (name, (xs, ys)) in enumerate(series.items()):
        keep = np.isfinite(xs) & np.isfinite(ys)
        pts = [(xp(float(x)), yp(float(y))) for x, y in zip(xs[keep], ys[keep])]
        color = colors[idx % len(colors)]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=4)
        for p in pts:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)
        draw.rectangle([width - margin_r - 260, legend_y, width - margin_r - 240, legend_y + 12], fill=color)
        draw.text((width - margin_r - 235, legend_y - 3), name, fill=(0, 0, 0))
        legend_y += 24
    img.save(path_base.with_suffix(".png"))


def run_case(case: str, samples: int, dynamic: bool) -> tuple[list[dict[str, object]], dict[str, RetimedTrajectory]]:
    path = make_synthetic_path(samples, case)
    dyn = SyntheticDynamics(v_max=np.ones(7) * 1.0, tau_max=np.ones(7) * (0.8 if dynamic else 1e6))
    cfg = TOPPConfig(
        qdd_max=np.ones(7) * 1.0,
        tau_max=dyn.tau_max,
        v_max=dyn.v_max,
        torque_scale=1.0,
        velocity_scale=1.0,
        search_grid=9,
        bisection_iters=18,
        endpoint_check=True,
    )
    planner = RollingHorizonTOPP(dyn, cfg)
    out_rows: list[dict[str, object]] = []
    trajs: dict[str, RetimedTrajectory] = {}
    for method in ("offline_topp_ra", "offline_topp_ni_grid", "offline_topp_co"):
        t0 = time.perf_counter()
        try:
            if method == "offline_topp_ra":
                traj = planner.plan_offline(path, 0.0, 0.0)
                traj.mode = method
            elif method == "offline_topp_ni_grid":
                traj = plan_offline_topp_ni_like(planner, path, 0.0, 0.0)
                traj.mode = method
            else:
                traj = plan_offline_topp_co(planner, path, 0.0, 0.0)
                traj.mode = method
            row = constraint_metrics(traj, planner)
            status = "ok"
            if (
                not math.isfinite(float(row["duration"]))
                or float(row["duration"]) > 60.0
                or float(row["max_velocity_violation"]) > 1e-6
                or float(row["max_acceleration_violation"]) > 1e-6
                or float(row["max_torque_violation"]) > 1e-6
                or float(row["transition_residual"]) > 1e-6
                or float(row["time_monotonic"]) < 0.5
            ):
                status = "failed_validation"
            trajs[method] = traj
        except Exception as exc:
            row = {
                "duration": float("nan"),
                "max_velocity_violation": float("nan"),
                "max_acceleration_violation": float("nan"),
                "max_torque_violation": float("nan"),
                "transition_residual": float("nan"),
                "start_speed_error": float("nan"),
                "goal_speed_error": float("nan"),
                "time_monotonic": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "unavailable" if method == "offline_topp_co" else "failed"
        row.update({
            "case": case,
            "samples": int(samples),
            "dynamic": int(dynamic),
            "method": method,
            "status": status,
            "compute_time_ms": 1000.0 * (time.perf_counter() - t0),
        })
        out_rows.append(row)
    for a, b in (("offline_topp_ra", "offline_topp_ni_grid"), ("offline_topp_ra", "offline_topp_co"), ("offline_topp_ni_grid", "offline_topp_co")):
        if a in trajs and b in trajs:
            diff = speed_linf(trajs[a], trajs[b])
        else:
            diff = float("nan")
        out_rows.append({
            "case": case,
            "samples": int(samples),
            "dynamic": int(dynamic),
            "method": f"{a}_vs_{b}",
            "status": "comparison",
            "speed_curve_linf": diff,
        })
    return out_rows, trajs


def run_validation(urdf_path: str | Path = PYCHARM_URDF, out_dir: str | Path = PYCHARM_OUT) -> None:
    _ = Path(urdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    profile_trajs: dict[str, RetimedTrajectory] = {}
    convergence: dict[str, list[tuple[int, float]]] = {"offline_topp_ra": [], "offline_topp_ni_grid": []}
    for samples in (30, 60, 120, 240):
        for case, dynamic in (("single_dof", False), ("multi_joint", False), ("multi_joint", True)):
            case_rows, trajs = run_case(case, samples, dynamic)
            rows.extend(case_rows)
            if samples == 120 and case == "single_dof" and not dynamic:
                profile_trajs.update(trajs)
            if case == "single_dof" and not dynamic:
                for method in convergence:
                    if method in trajs:
                        convergence[method].append((samples, float(trajs[method].duration)))
    keys = sorted({k for r in rows for k in r.keys()})
    with (out_dir / "solver_validation_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "solver_validation_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")

    speed_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, traj in profile_trajs.items():
        s = np.asarray(traj.s, dtype=float)
        speed = np.asarray(traj.speed, dtype=float)
        keep = np.isfinite(s) & np.isfinite(speed) & (np.abs(speed) < 1e6)
        if np.any(keep):
            speed_series[method] = (s[keep], speed[keep])
    save_line_chart(
        out_dir / "fig_solver_speed_profiles",
        speed_series,
        "Solver validation speed profiles",
        "s",
        "Path speed",
    )

    convergence_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, vals in convergence.items():
        if vals:
            convergence_series[method] = (
                np.asarray([v[0] for v in vals], dtype=float),
                np.asarray([v[1] for v in vals], dtype=float),
            )
    save_line_chart(
        out_dir / "fig_solver_convergence",
        convergence_series,
        "Sampling convergence",
        "Samples",
        "Duration / s",
    )

    failures = [r for r in rows if r.get("status") in ("failed", "failed_validation")]
    unavailable = [r for r in rows if r.get("status") == "unavailable"]
    violations = [
        r for r in rows
        if r.get("status") == "ok"
        and (
            float(r.get("max_velocity_violation", 0.0) or 0.0) > 1e-6
            or float(r.get("max_acceleration_violation", 0.0) or 0.0) > 1e-6
            or float(r.get("max_torque_violation", 0.0) or 0.0) > 1e-6
            or float(r.get("transition_residual", 0.0) or 0.0) > 1e-6
        )
    ]
    passed = not failures and not violations and not unavailable
    cvx = cvxpy_backend_status()
    lines = [
        "# TOPP Solver Validation Report",
        "",
        f"Solver validation passed: `{passed}`",
        f"Failed rows: {len(failures)}",
        f"Unavailable rows: {len(unavailable)}",
        f"Constraint/residual violation rows: {len(violations)}",
        "",
        f"CVXPY backend available: `{cvx.get('available')}`",
        f"Installed CVXPY solvers: `{cvx.get('installed_solvers')}`",
        "TOPP-CO status: enabled only when the CVXPY formulation solves and passes validation.",
        "TOPP-NI status: grid-based NI baseline; not an exact continuous switch-point Bobrow/Shin-McKay implementation.",
    ]
    if unavailable:
        lines.append("")
        lines.append("Unavailable examples:")
        for r in unavailable[:5]:
            lines.append(f"- {r.get('case')} samples={r.get('samples')} {r.get('method')}: {r.get('error')}")
    if failures:
        lines.append("")
        lines.append("Failed validation examples:")
        for r in failures[:8]:
            lines.append(
                "- "
                f"{r.get('case')} samples={r.get('samples')} dynamic={r.get('dynamic')} {r.get('method')}: "
                f"duration={r.get('duration')}, "
                f"vel_violation={r.get('max_velocity_violation')}, "
                f"acc_violation={r.get('max_acceleration_violation')}, "
                f"tau_violation={r.get('max_torque_violation')}, "
                f"transition_residual={r.get('transition_residual')}"
            )
    (out_dir / "solver_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    urdf = PYCHARM_URDF if USE_PYCHARM_CONFIG else Path(sys.argv[1]) if len(sys.argv) > 1 else PYCHARM_URDF
    out = PYCHARM_OUT if USE_PYCHARM_CONFIG else Path(sys.argv[2]) if len(sys.argv) > 2 else PYCHARM_OUT
    run_validation(urdf, out)


if __name__ == "__main__":
    main()
