from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from codroid_topp.robot.virtual_srs import AnalyticIKSolver, IKResult, VirtualSRSModel
from codroid_topp.trajectory.se3_path import Pose
from codroid_topp.utils.math_utils import finite_difference, limit_barrier, min_limit_margin, wrap_to_pi


@dataclass
class JointPath:
    s: np.ndarray
    q: np.ndarray
    qs: np.ndarray
    qss: np.ndarray
    psi: np.ndarray
    ik_position_error: np.ndarray
    ik_rotation_error: np.ndarray
    min_limit_margin: float


class ArmAngleOptimizer:
    """Select a continuous IK branch and arm-angle sequence.

    The default solver uses a dynamic-programming pass over analytic IK
    candidates generated on a psi grid. It optimizes smoothness, limit margin,
    and optional psi regularization. This is fast enough for a short rolling
    window; for a hard real-time controller, call `solve_greedy` with a small
    grid around the previous psi.
    """

    def __init__(
        self,
        model: VirtualSRSModel,
        psi_grid: np.ndarray | None = None,
        w_smooth: float = 1.0,
        w_accel: float = 0.03,
        w_limit: float = 1e-3,
        w_psi: float = 0.02,
        max_candidates_per_pose: int = 24,
    ):
        self.model = model
        self.solver = AnalyticIKSolver(model)
        self.psi_grid = np.asarray(psi_grid if psi_grid is not None else np.linspace(-np.pi, np.pi, 37), dtype=float)
        self.w_smooth = float(w_smooth)
        self.w_accel = float(w_accel)
        self.w_limit = float(w_limit)
        self.w_psi = float(w_psi)
        self.max_candidates_per_pose = int(max_candidates_per_pose)

    def _pose_candidates(self, pose: Pose, q_seed: np.ndarray | None) -> list[IKResult]:
        cands = self.solver.solve_grid(pose.as_T(), self.psi_grid, q_prev=q_seed)
        # Deduplicate nearly identical q vectors while preserving branch diversity.
        uniq: list[IKResult] = []
        for c in cands:
            if not any(np.linalg.norm(wrap_to_pi(c.q - u.q)) < 1e-5 for u in uniq):
                uniq.append(c)
            if len(uniq) >= self.max_candidates_per_pose:
                break
        if not uniq:
            raise ValueError("IK failed for a path sample. Check reachability or expand psi_grid.")
        return uniq

    def solve(self, poses: Iterable[Pose], s: np.ndarray, q_seed: np.ndarray | None = None) -> JointPath:
        poses = list(poses)
        s = np.asarray(s, dtype=float).reshape(-1)
        if len(poses) != len(s):
            raise ValueError("poses and s must have the same length")
        if len(poses) < 2:
            raise ValueError("At least two samples are required")

        all_cands: list[list[IKResult]] = []
        seed = q_seed
        for pose in poses:
            cands = self._pose_candidates(pose, seed)
            all_cands.append(cands)
            seed = cands[0].q

        costs: list[np.ndarray] = []
        parents: list[np.ndarray] = []
        c0 = np.array([self.w_limit * limit_barrier(c.q, self.model.q_min, self.model.q_max) for c in all_cands[0]], dtype=float)
        if q_seed is not None:
            c0 += self.w_smooth * np.array([np.sum(wrap_to_pi(c.q - q_seed) ** 2) for c in all_cands[0]])
        costs.append(c0)
        parents.append(-np.ones(len(all_cands[0]), dtype=int))

        for i in range(1, len(all_cands)):
            prev = all_cands[i - 1]
            cur = all_cands[i]
            ci = np.full(len(cur), np.inf)
            pi = np.full(len(cur), -1, dtype=int)
            ds = max(float(s[i] - s[i - 1]), 1e-9)
            for j, cj in enumerate(cur):
                local = self.w_limit * limit_barrier(cj.q, self.model.q_min, self.model.q_max)
                for k, pk in enumerate(prev):
                    dq = wrap_to_pi(cj.q - pk.q)
                    dpsi = float(wrap_to_pi(cj.psi - pk.psi))
                    trans = self.w_smooth * float(dq @ dq) / ds + self.w_psi * dpsi * dpsi / ds
                    # A small acceleration-like term once we have a grandparent.
                    accel_cost = 0.0
                    if i >= 2 and parents[-1][k] >= 0:
                        g = all_cands[i - 2][parents[-1][k]]
                        ddq = wrap_to_pi(cj.q - 2.0 * pk.q + g.q)
                        accel_cost = self.w_accel * float(ddq @ ddq) / (ds * ds)
                    val = costs[-1][k] + local + trans + accel_cost + 1e3 * cj.pos_err + 10.0 * cj.rot_err
                    if val < ci[j]:
                        ci[j] = val
                        pi[j] = k
            costs.append(ci)
            parents.append(pi)

        idx = int(np.argmin(costs[-1]))
        chosen: list[IKResult] = []
        for i in reversed(range(len(all_cands))):
            chosen.append(all_cands[i][idx])
            idx = int(parents[i][idx]) if i > 0 else -1
        chosen.reverse()

        q = np.vstack([c.q for c in chosen])
        psi = np.array([c.psi for c in chosen], dtype=float)
        pos_err = np.array([c.pos_err for c in chosen], dtype=float)
        rot_err = np.array([c.rot_err for c in chosen], dtype=float)
        qs, qss = finite_difference(q, s)
        margin = min(min_limit_margin(qi, self.model.q_min, self.model.q_max) for qi in q)
        return JointPath(s=s, q=q, qs=qs, qss=qss, psi=psi, ik_position_error=pos_err, ik_rotation_error=rot_err, min_limit_margin=float(margin))

    def solve_greedy(self, poses: Iterable[Pose], s: np.ndarray, q_seed: np.ndarray | None = None, psi_center: float | None = None, psi_radius: float = 0.35, n_psi: int = 9) -> JointPath:
        old_grid = self.psi_grid.copy()
        if psi_center is not None:
            self.psi_grid = np.linspace(psi_center - psi_radius, psi_center + psi_radius, n_psi)
        out_q = []
        out_psi = []
        out_pos = []
        out_rot = []
        seed = q_seed
        for pose in poses:
            c = self._pose_candidates(pose, seed)[0]
            out_q.append(c.q)
            out_psi.append(c.psi)
            out_pos.append(c.pos_err)
            out_rot.append(c.rot_err)
            seed = c.q
            self.psi_grid = np.linspace(c.psi - psi_radius, c.psi + psi_radius, n_psi)
        self.psi_grid = old_grid
        q = np.vstack(out_q)
        s = np.asarray(s, dtype=float)
        qs, qss = finite_difference(q, s)
        margin = min(min_limit_margin(qi, self.model.q_min, self.model.q_max) for qi in q)
        return JointPath(s=s, q=q, qs=qs, qss=qss, psi=np.asarray(out_psi), ik_position_error=np.asarray(out_pos), ik_rotation_error=np.asarray(out_rot), min_limit_margin=float(margin))
