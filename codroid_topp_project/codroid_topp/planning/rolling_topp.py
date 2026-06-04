from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from codroid_topp.planning.psi_optimizer import JointPath
from codroid_topp.robot.dynamics import DynamicsBackend


@dataclass
class TOPPConfig:
    qdd_max: np.ndarray | None = None
    qdd_min: np.ndarray | None = None
    tau_max: np.ndarray | None = None
    v_max: np.ndarray | None = None
    torque_scale: float = 0.85
    velocity_scale: float = 0.85
    terminal_safe_speed: float = 0.03
    goal_speed: float = 0.0
    search_grid: int = 17
    bisection_iters: int = 32
    eps: float = 1e-10
    horizon_points: int = 45
    brake_points: int = 25
    commit_points: int = 6
    endpoint_check: bool = True

    def resolved_acc_limits(self) -> tuple[np.ndarray, np.ndarray]:
        if self.qdd_max is None:
            qdd_max = np.ones(7) * 5.0
        else:
            qdd_max = np.asarray(self.qdd_max, dtype=float).reshape(7)
        if self.qdd_min is None:
            qdd_min = -qdd_max
        else:
            qdd_min = np.asarray(self.qdd_min, dtype=float).reshape(7)
        return qdd_min, qdd_max


@dataclass
class RetimedTrajectory:
    s: np.ndarray
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray
    status: str
    mode: str

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.x, 0.0))

    def metrics(self, tau_max: np.ndarray | None = None) -> dict[str, float]:
        duration = self.duration
        out = {
            "duration": duration,
            "max_speed": float(np.max(self.speed)),
            "max_abs_qd": float(np.max(np.abs(self.qd))),
            "max_abs_qdd": float(np.max(np.abs(self.qdd))),
            "max_abs_tau": float(np.max(np.abs(self.tau))),
        }
        if len(self.t) > 2 and np.all(np.isfinite(self.t)) and np.all(np.diff(self.t) > 1e-12):
            qddd = np.gradient(self.qdd, self.t, axis=0, edge_order=1)
            taud = np.gradient(self.tau, self.t, axis=0, edge_order=1)
            out["mean_abs_jerk"] = float(np.mean(np.abs(qddd)))
            out["max_abs_jerk"] = float(np.max(np.abs(qddd)))
            out["rms_jerk"] = float(np.sqrt(np.mean(qddd * qddd)))
            out["mean_abs_taudot"] = float(np.mean(np.abs(taud)))
            out["max_abs_taudot"] = float(np.max(np.abs(taud)))
            out["rms_taudot"] = float(np.sqrt(np.mean(taud * taud)))
        else:
            out["mean_abs_jerk"] = 0.0
            out["max_abs_jerk"] = 0.0
            out["rms_jerk"] = 0.0
            out["mean_abs_taudot"] = 0.0
            out["max_abs_taudot"] = 0.0
            out["rms_taudot"] = 0.0
        if tau_max is not None:
            tau_max = np.asarray(tau_max, dtype=float).reshape(7)
            out["max_torque_utilization"] = float(np.max(np.abs(self.tau) / np.maximum(tau_max, 1e-9)))
        return out


class RollingHorizonTOPP:
    """Reachability-analysis TOPP with terminal feasible set.

    The implementation uses the same x=s_dot^2 and u=s_ddot variables as
    TOPP-RA. Dynamics constraints are evaluated through inverse dynamics and are
    linear in u for fixed (q, qdot), so a scalar interval is enough at each node.
    """

    def __init__(self, dynamics: DynamicsBackend, config: TOPPConfig | None = None):
        self.dyn = dynamics
        self.cfg = config if config is not None else TOPPConfig()
        self.qdd_min, self.qdd_max = self.cfg.resolved_acc_limits()
        v_source = dynamics.v_max if self.cfg.v_max is None else self.cfg.v_max
        tau_source = dynamics.tau_max if self.cfg.tau_max is None else self.cfg.tau_max
        self.v_max = np.asarray(v_source, dtype=float).reshape(7) * self.cfg.velocity_scale
        self.tau_max = np.asarray(tau_source, dtype=float).reshape(7) * self.cfg.torque_scale
        self._u_cache: dict[tuple[int, int, float], tuple[float, float]] = {}

    def _path_velocity_upper(self, path: JointPath) -> np.ndarray:
        xs = np.full(path.s.size, np.inf, dtype=float)
        for i, qs in enumerate(path.qs):
            vals = []
            for j in range(7):
                a = abs(float(qs[j]))
                if a > 1e-9:
                    vals.append((self.v_max[j] / a) ** 2)
            xs[i] = min(vals) if vals else 1e6
        return np.minimum(xs, 1e6)

    def _u_bounds_single(self, path: JointPath, i: int, x: float) -> tuple[float, float]:
        q = path.q[i]
        qs = path.qs[i]
        qss = path.qss[i]
        sdot = float(np.sqrt(max(x, 0.0)))
        qd = qs * sdot
        lo = -np.inf
        hi = np.inf
        # Joint acceleration interval.
        for j in range(7):
            a = float(qs[j])
            b = float(qss[j] * x)
            if abs(a) < 1e-12:
                if b < self.qdd_min[j] - 1e-9 or b > self.qdd_max[j] + 1e-9:
                    return 1.0, 0.0
                continue
            uj1 = (self.qdd_min[j] - b) / a
            uj2 = (self.qdd_max[j] - b) / a
            lo = max(lo, min(uj1, uj2))
            hi = min(hi, max(uj1, uj2))
            if lo > hi:
                return 1.0, 0.0
        # Rigid-body inverse dynamics is affine in qdd, hence affine in u.
        qdd0 = qss * x
        qdd1 = qs + qdd0
        tau0 = self.dyn.inverse_dynamics(q, qd, qdd0)
        tau1 = self.dyn.inverse_dynamics(q, qd, qdd1)
        a_tau = tau1 - tau0
        for j in range(7):
            a = float(a_tau[j])
            b = float(tau0[j])
            lower = -self.tau_max[j]
            upper = self.tau_max[j]
            if abs(a) < 1e-12:
                if b < lower - 1e-9 or b > upper + 1e-9:
                    return 1.0, 0.0
                continue
            uj1 = (lower - b) / a
            uj2 = (upper - b) / a
            lo = max(lo, min(uj1, uj2))
            hi = min(hi, max(uj1, uj2))
            if lo > hi:
                return 1.0, 0.0
        return float(lo), float(hi)

    def _u_bounds(self, path: JointPath, i: int, x: float) -> tuple[float, float]:
        # Reachability checks repeatedly query identical node/speed pairs during
        # grid search and bisection. Caching is especially helpful when inverse
        # dynamics is backed by Pinocchio.
        key = (id(path.s), int(i), round(float(x), 12))
        val = self._u_cache.get(key)
        if val is None:
            val = self._u_bounds_single(path, i, x)
            self._u_cache[key] = val
        return val

    def _transition_feasible(self, path: JointPath, x_upper: np.ndarray, i: int, xi: float, xj: float) -> bool:
        if xi < -self.cfg.eps or xj < -self.cfg.eps:
            return False
        if xi > x_upper[i] + 1e-8 or xj > x_upper[i + 1] + 1e-8:
            return False
        ds = float(path.s[i + 1] - path.s[i])
        if ds <= 0.0:
            return False
        u = (xj - xi) / (2.0 * ds)
        lo, hi = self._u_bounds(path, i, xi)
        if u < lo - 1e-8 or u > hi + 1e-8:
            return False
        if self.cfg.endpoint_check:
            lo2, hi2 = self._u_bounds(path, i + 1, xj)
            if u < lo2 - 1e-8 or u > hi2 + 1e-8:
                return False
        return True

    def _max_feasible_next(self, path: JointPath, x_upper: np.ndarray, i: int, xi: float, x_next_cap: float) -> float | None:
        ds = float(path.s[i + 1] - path.s[i])
        lo_u, hi_u = self._u_bounds(path, i, xi)
        if lo_u > hi_u:
            return None
        lower = max(0.0, xi + 2.0 * ds * lo_u)
        upper = min(float(x_next_cap), float(x_upper[i + 1]), xi + 2.0 * ds * hi_u)
        if lower > upper + 1e-10:
            return None
        if self._transition_feasible(path, x_upper, i, xi, upper):
            return float(upper)
        # Find the highest feasible point in the interval, then refine upward.
        grid = np.linspace(upper, lower, max(3, self.cfg.search_grid))
        best = None
        high_bad = upper
        for val in grid:
            if self._transition_feasible(path, x_upper, i, xi, float(val)):
                best = float(val)
                break
            high_bad = float(val)
        if best is None:
            return None
        lo = best
        hi = high_bad
        if hi < lo:
            hi = upper
        for _ in range(self.cfg.bisection_iters):
            mid = 0.5 * (lo + hi)
            if self._transition_feasible(path, x_upper, i, xi, mid):
                lo = mid
            else:
                hi = mid
        return float(lo)

    def _exists_feasible_next(self, path: JointPath, x_upper: np.ndarray, i: int, xi: float, x_next_cap: float) -> bool:
        return self._max_feasible_next(path, x_upper, i, xi, x_next_cap) is not None

    def _backward_controllable_sets(self, path: JointPath, x_upper: np.ndarray, terminal_x: float) -> np.ndarray:
        n = len(path.s)
        K = np.zeros(n, dtype=float)
        K[-1] = min(float(terminal_x), float(x_upper[-1]))
        for i in reversed(range(n - 1)):
            hi = float(x_upper[i])
            ds = max(float(path.s[i + 1] - path.s[i]), 1e-12)
            lo_u0, _ = self._u_bounds(path, i, 0.0)
            if lo_u0 <= 0.0:
                # Numerical guard: with torque constraints, the true controllable
                # set can be tiny compared with the velocity upper bound. Bracket
                # the binary search by the zero-speed braking capability so a small
                # feasible interval is not missed when bisection_iters is low.
                hi = min(hi, float(K[i + 1]) + 2.0 * ds * max(0.0, -float(lo_u0)))
            if self._exists_feasible_next(path, x_upper, i, hi, K[i + 1]):
                K[i] = hi
                continue
            lo = 0.0
            if not self._exists_feasible_next(path, x_upper, i, lo, K[i + 1]):
                K[i] = 0.0
                continue
            for _ in range(self.cfg.bisection_iters):
                mid = 0.5 * (lo + hi)
                if self._exists_feasible_next(path, x_upper, i, mid, K[i + 1]):
                    lo = mid
                else:
                    hi = mid
            K[i] = lo
        return K

    def _forward_pass(self, path: JointPath, x_upper: np.ndarray, K: np.ndarray, start_x: float) -> np.ndarray:
        n = len(path.s)
        x = np.zeros(n, dtype=float)
        x[0] = min(float(start_x), float(K[0]), float(x_upper[0]))
        for i in range(n - 1):
            cap = min(float(K[i + 1]), float(x_upper[i + 1]))
            xnext = self._max_feasible_next(path, x_upper, i, x[i], cap)
            if xnext is None:
                # Fall back to a full stop if the current speed is too aggressive.
                xnext = 0.0 if self._transition_feasible(path, x_upper, i, x[i], 0.0) else min(cap, x[i])
            x[i + 1] = max(0.0, min(float(xnext), cap))
        return x

    def _assemble(self, path: JointPath, x: np.ndarray, mode: str, status: str = "ok") -> RetimedTrajectory:
        s = path.s
        n = len(s)
        u = np.zeros(n, dtype=float)
        for i in range(n - 1):
            ds = max(float(s[i + 1] - s[i]), 1e-12)
            u[i] = (x[i + 1] - x[i]) / (2.0 * ds)
        if n > 1:
            u[-1] = u[-2]
        qd = path.qs * np.sqrt(np.maximum(x, 0.0))[:, None]
        qdd = path.qs * u[:, None] + path.qss * x[:, None]
        tau = np.vstack([self.dyn.inverse_dynamics(path.q[i], qd[i], qdd[i]) for i in range(n)])
        t = np.zeros(n, dtype=float)
        for i in range(n - 1):
            ds = max(float(s[i + 1] - s[i]), 1e-12)
            v0 = float(np.sqrt(max(x[i], 0.0)))
            v1 = float(np.sqrt(max(x[i + 1], 0.0)))
            denom = max(v0 + v1, 1e-9)
            t[i + 1] = t[i] + 2.0 * ds / denom
        return RetimedTrajectory(s=s.copy(), t=t, x=x.copy(), u=u, q=path.q.copy(), qd=qd, qdd=qdd, tau=tau, status=status, mode=mode)

    def plan_offline(self, path: JointPath, start_speed: float = 0.0, goal_speed: float | None = None) -> RetimedTrajectory:
        self._u_cache.clear()
        x_upper = self._path_velocity_upper(path)
        terminal = self.cfg.goal_speed if goal_speed is None else float(goal_speed)
        K = self._backward_controllable_sets(path, x_upper, terminal * terminal)
        x = self._forward_pass(path, x_upper, K, start_speed * start_speed)
        x[-1] = min(x[-1], terminal * terminal + 1e-8)
        return self._assemble(path, x, mode="offline")

    def plan_rolling(self, path: JointPath, start_speed: float = 0.0) -> RetimedTrajectory:
        self._u_cache.clear()
        n = len(path.s)
        h = max(2, int(self.cfg.horizon_points))
        b = max(1, int(self.cfg.brake_points))
        c = max(1, int(self.cfg.commit_points))
        idx_all: list[int] = [0]
        x_all: list[float] = [start_speed * start_speed]
        k = 0
        xk = start_speed * start_speed
        while k < n - 1:
            end = min(n - 1, k + h + b)
            sub = self._slice_path(path, k, end)
            terminal_speed = self.cfg.goal_speed if end == n - 1 else self.cfg.terminal_safe_speed
            x_upper = self._path_velocity_upper(sub)
            K = self._backward_controllable_sets(sub, x_upper, terminal_speed * terminal_speed)
            xs = self._forward_pass(sub, x_upper, K, xk)
            commit_edges = min(c, end - k, n - 1 - k)
            for r in range(1, commit_edges + 1):
                idx_all.append(k + r)
                x_all.append(float(xs[r]))
            k += commit_edges
            xk = x_all[-1]
        # Re-assemble on the full path using the stitched x profile.
        x_full = np.zeros(n, dtype=float)
        for idx, val in zip(idx_all, x_all):
            x_full[idx] = val
        # Fill any numerical holes by interpolation.
        known = np.array(idx_all, dtype=int)
        vals = np.array(x_all, dtype=float)
        x_full = np.interp(np.arange(n), known, vals)
        return self._assemble(path, x_full, mode="rolling")

    def _slice_path(self, path: JointPath, i0: int, i1: int) -> JointPath:
        from codroid_topp.planning.psi_optimizer import JointPath

        sl = slice(i0, i1 + 1)
        s = path.s[sl].copy()
        # Normalize local s only for numerical conditioning; ds values are unchanged up to scale if we keep original.
        return JointPath(
            s=s,
            q=path.q[sl].copy(),
            qs=path.qs[sl].copy(),
            qss=path.qss[sl].copy(),
            psi=path.psi[sl].copy(),
            ik_position_error=path.ik_position_error[sl].copy(),
            ik_rotation_error=path.ik_rotation_error[sl].copy(),
            min_limit_margin=path.min_limit_margin,
        )

    def conservative_trapezoid(self, path: JointPath, speed_scale: float = 0.25) -> RetimedTrajectory:
        x_upper = self._path_velocity_upper(path)
        x = x_upper * float(speed_scale) ** 2
        x[0] = 0.0
        x[-1] = 0.0
        # Enforce acceleration approximately with one backward and one forward clamp.
        for i in range(len(path.s) - 1):
            ds = max(float(path.s[i + 1] - path.s[i]), 1e-12)
            lo, hi = self._u_bounds(path, i, x[i])
            x[i + 1] = min(x[i + 1], max(0.0, x[i] + 2.0 * ds * max(0.0, hi)))
        for i in reversed(range(len(path.s) - 1)):
            ds = max(float(path.s[i + 1] - path.s[i]), 1e-12)
            lo, hi = self._u_bounds(path, i, x[i + 1])
            x[i] = min(x[i], max(0.0, x[i + 1] - 2.0 * ds * min(0.0, lo)))
        return self._assemble(path, x, mode="conservative")

    def kinematic_only(self, path: JointPath, start_speed: float = 0.0, goal_speed: float = 0.0) -> RetimedTrajectory:
        saved_tau = self.tau_max.copy()
        self.tau_max = np.ones_like(self.tau_max) * 1e12
        try:
            out = self.plan_offline(path, start_speed=start_speed, goal_speed=goal_speed)
            out.mode = "kinematic_only"
            return out
        finally:
            self.tau_max = saved_tau
