from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d, make_interp_spline, splprep, splev
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


@dataclass
class Pose:
    p: np.ndarray
    R: np.ndarray

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=float).reshape(3)
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)

    @classmethod
    def from_T(cls, T: np.ndarray) -> "Pose":
        T = np.asarray(T, dtype=float)
        return cls(T[:3, 3], T[:3, :3])

    def as_T(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        return T


class BSplineSE3Path:
    """C2-ish Cartesian pose path with arc-length reparameterization.

    Position is fit by a cubic B-spline/PCHIP depending on keyframe count.
    Orientation uses SciPy RotationSpline when possible; otherwise Slerp.
    The public parameter s is normalized arc length in [0, 1].
    """

    def __init__(
        self,
        keyframes: Iterable[Pose],
        smoothing: float = 0.0,
        dense_count: int = 2000,
        force_interpolate: bool = True,
        position_mode: str = "pchip",
    ):
        self.keyframes = list(keyframes)
        if len(self.keyframes) < 2:
            raise ValueError("At least two keyframes are required")
        self.smoothing = float(smoothing)
        self.force_interpolate = bool(force_interpolate)
        self.position_mode = position_mode
        self._build(dense_count=dense_count)

    @staticmethod
    def _key_parameter(points: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.r_[0.0, np.cumsum(np.maximum(d, 1e-9))]
        if u[-1] <= 1e-12:
            u = np.linspace(0.0, 1.0, len(points))
        else:
            u = u / u[-1]
        # Ensure strictly increasing values for repeated Cartesian points.
        for i in range(1, len(u)):
            if u[i] <= u[i - 1]:
                u[i] = u[i - 1] + 1e-6
        return u / u[-1]

    def _build(self, dense_count: int) -> None:
        self.points = np.vstack([p.p for p in self.keyframes])
        self.rotations = Rotation.from_matrix(np.stack([p.R for p in self.keyframes]))
        self.u_key = self._key_parameter(self.points)
        n = len(self.keyframes)
        if self.smoothing > 0.0 and n >= 4 and not self.force_interpolate:
            tck, _ = splprep(self.points.T, u=self.u_key, s=self.smoothing, k=min(3, n - 1))
            self._pos_kind = "splprep"
            self._tck = tck
        elif self.position_mode == "bspline" and n >= 4:
            self._pos_kind = "bspline"
            self._spl = make_interp_spline(self.u_key, self.points, k=3, axis=0)
        else:
            # PCHIP is shape-preserving component-wise and avoids the workspace
            # overshoot that global cubic splines can create near reach limits.
            self._pos_kind = "pchip"
            self._spl = PchipInterpolator(self.u_key, self.points, axis=0)
        try:
            self._rot_kind = "rotation_spline"
            self._rot_spline = RotationSpline(self.u_key, self.rotations)
        except Exception:
            self._rot_kind = "slerp"
            self._slerp = Slerp(self.u_key, self.rotations)
        dense_count = max(int(dense_count), 20)
        self._u_dense = np.linspace(0.0, 1.0, dense_count)
        p_dense = self._eval_pos_u(self._u_dense)
        seg = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
        ell = np.r_[0.0, np.cumsum(seg)]
        self.length = float(ell[-1])
        if self.length <= 1e-12:
            self._s_dense = self._u_dense.copy()
        else:
            self._s_dense = ell / self.length
        # Repair duplicates in near-zero segments.
        for i in range(1, len(self._s_dense)):
            if self._s_dense[i] <= self._s_dense[i - 1]:
                self._s_dense[i] = self._s_dense[i - 1] + 1e-12
        self._s_dense[-1] = 1.0
        self._u_of_s = interp1d(self._s_dense, self._u_dense, kind="linear", bounds_error=False, fill_value=(0.0, 1.0), assume_sorted=True)

    def _eval_pos_u(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if self._pos_kind == "splprep":
            vals = np.asarray(splev(u, self._tck)).T
        else:
            vals = np.asarray(self._spl(u), dtype=float)
        return vals.reshape((-1, 3))

    def _eval_rot_u(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        if self._rot_kind == "rotation_spline":
            return self._rot_spline(u).as_matrix()
        return self._slerp(u).as_matrix()

    def u_of_s(self, s: np.ndarray | float) -> np.ndarray:
        return np.asarray(self._u_of_s(np.clip(s, 0.0, 1.0)), dtype=float)

    def position(self, s: np.ndarray | float) -> np.ndarray:
        u = self.u_of_s(s)
        p = self._eval_pos_u(u)
        if np.ndim(s) == 0:
            return p[0]
        return p

    def rotation(self, s: np.ndarray | float) -> np.ndarray:
        u = self.u_of_s(s)
        R = self._eval_rot_u(u)
        if np.ndim(s) == 0:
            return R[0]
        return R

    def pose(self, s: float) -> Pose:
        return Pose(self.position(float(s)), self.rotation(float(s)))

    def sample(self, s_values: Iterable[float]) -> list[Pose]:
        return [self.pose(float(s)) for s in s_values]

    def transforms(self, s_values: Iterable[float]) -> np.ndarray:
        poses = self.sample(s_values)
        return np.stack([p.as_T() for p in poses])

    def curvature(self, s_values: Iterable[float]) -> np.ndarray:
        s = np.asarray(list(s_values), dtype=float)
        p = self.position(s)
        if len(s) < 3:
            return np.zeros_like(s)
        dp = np.gradient(p, s, axis=0, edge_order=2)
        ddp = np.gradient(dp, s, axis=0, edge_order=2)
        cross = np.linalg.norm(np.cross(dp, ddp), axis=1)
        den = np.linalg.norm(dp, axis=1) ** 3 + 1e-12
        return cross / den

    def uniform_grid(self, n: int) -> np.ndarray:
        if n < 2:
            raise ValueError("n must be >= 2")
        return np.linspace(0.0, 1.0, int(n))

    def adaptive_grid(self, ds_max: float = 0.015, ds_min: float = 0.002, curvature_gain: float = 0.08) -> np.ndarray:
        probe = np.linspace(0.0, 1.0, 250)
        kappa = self.curvature(probe)
        kappa_i = interp1d(probe, kappa, kind="linear", bounds_error=False, fill_value="extrapolate")
        s_vals = [0.0]
        s = 0.0
        while s < 1.0 - 1e-12:
            local_k = float(max(kappa_i(s), 0.0))
            ds = ds_max / (1.0 + curvature_gain * local_k)
            ds = float(np.clip(ds, ds_min, ds_max))
            s = min(1.0, s + ds)
            s_vals.append(s)
        return np.asarray(s_vals, dtype=float)
