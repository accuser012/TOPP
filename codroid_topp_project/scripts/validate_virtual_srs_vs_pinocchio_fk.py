#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codroid_topp.robot.virtual_srs import VirtualSRSModel
from codroid_topp.robot.dynamics import PinocchioDynamics
from codroid_topp.utils.math_utils import rotation_error

# ======================= PyCharm direct-run settings =======================
USE_PYCHARM_CONFIG = True
PYCHARM_URDF = ROOT / "configs" / "codroidRobot.urdf"
PYCHARM_SAMPLES = 1000
PYCHARM_SEED = 42
PYCHARM_OUT = ROOT / "results_fk_validation"
MATLAB_REFERENCE = Path("D:/design/TOPP/Dynamic_Identification/运动学一致性/left_arm_mdh_analytic_ik_v2.m")
# ==========================================================================


def _fk_pin(dyn: PinocchioDynamics, pin, q: np.ndarray, frame_id: int) -> np.ndarray:
    q_full = dyn.q_neutral.copy()
    q_full[dyn.q_indices] = np.asarray(q, dtype=float).reshape(7)
    pin.forwardKinematics(dyn.model, dyn.data, q_full)
    pin.updateFramePlacements(dyn.model, dyn.data)
    return np.asarray(dyn.data.oMf[frame_id].homogeneous, dtype=float)


def _candidate_frames(dyn: PinocchioDynamics) -> list[str]:
    names = []
    for frame in dyn.model.frames:
        name = str(frame.name)
        if "arm_l_07" in name or "joint_arm_l_07" in name:
            names.append(name)
    names.extend(["link_arm_l_07", dyn.info.arm_joint_names("l")[-1]])
    out: list[str] = []
    for name in names:
        if name not in out and dyn.model.getFrameId(name) < dyn.model.nframes:
            out.append(name)
    return out


def _pose_errors(Tv: np.ndarray, Tp: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(Tv[:3, 3] - Tp[:3, 3])),
        float(rotation_error(Tv[:3, :3], Tp[:3, :3])),
    )


def run_validation(urdf_path: str | Path = PYCHARM_URDF, out_dir: str | Path = PYCHARM_OUT, samples: int = PYCHARM_SAMPLES, seed: int = PYCHARM_SEED) -> None:
    urdf = Path(urdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = VirtualSRSModel.from_urdf(urdf, side="l")
    matlab_reference = MATLAB_REFERENCE if MATLAB_REFERENCE.exists() else ROOT / "reference_left_arm_mdh_analytic_ik.m"
    try:
        import pinocchio as pin  # type: ignore
        dyn = PinocchioDynamics(urdf, side="l")
    except Exception as exc:
        msg = f"Pinocchio unavailable, cannot compare full URDF FK: {exc}"
        print(msg)
        (out_dir / "fk_validation_report.md").write_text("# FK validation\n\n" + msg + "\n", encoding="utf-8")
        return

    rng = np.random.default_rng(seed)
    q_min = np.asarray(model.q_min, dtype=float)
    q_max = np.asarray(model.q_max, dtype=float)
    qs = rng.uniform(q_min + 0.05, q_max - 0.05, size=(int(samples), 7))
    q_home = np.zeros(7, dtype=float)
    frames = _candidate_frames(dyn)
    if not frames:
        raise RuntimeError("No candidate Pinocchio frame found for left-arm end frame.")

    frame_scores: list[dict[str, object]] = []
    for frame_name in frames:
        frame_id = dyn.model.getFrameId(frame_name)
        pos_err = []
        rot_err = []
        for q in qs[: min(len(qs), 200)]:
            Tv = model.fk(q)
            Tp = _fk_pin(dyn, pin, q, frame_id)
            p, r = _pose_errors(Tv, Tp)
            pos_err.append(p)
            rot_err.append(r)
        frame_scores.append({
            "frame": frame_name,
            "mean_position_error_m": float(np.mean(pos_err)),
            "mean_rotation_error_rad": float(np.mean(rot_err)),
        })
    frame_scores.sort(key=lambda r: (float(r["mean_position_error_m"]), float(r["mean_rotation_error_rad"])))
    frame_name = str(frame_scores[0]["frame"])
    frame_id = dyn.model.getFrameId(frame_name)

    Tv_home = model.fk(q_home)
    Tp_home = _fk_pin(dyn, pin, q_home, frame_id)
    right_fixed_transform = np.linalg.inv(Tp_home) @ Tv_home
    left_fixed_transform = Tv_home @ np.linalg.inv(Tp_home)

    raw_pos: list[float] = []
    raw_rot: list[float] = []
    right_pos: list[float] = []
    right_rot: list[float] = []
    left_pos: list[float] = []
    left_rot: list[float] = []
    for q in qs:
        Tv = model.fk(q)
        Tp = _fk_pin(dyn, pin, q, frame_id)
        p, r = _pose_errors(Tv, Tp)
        raw_pos.append(p)
        raw_rot.append(r)
        p, r = _pose_errors(Tv, Tp @ right_fixed_transform)
        right_pos.append(p)
        right_rot.append(r)
        p, r = _pose_errors(Tv, left_fixed_transform @ Tp)
        left_pos.append(p)
        left_rot.append(r)

    best_fixed = "right_tcp" if float(np.max(right_rot)) <= float(np.max(left_rot)) else "left_base"
    best_pos = right_pos if best_fixed == "right_tcp" else left_pos
    best_rot = right_rot if best_fixed == "right_tcp" else left_rot
    raw_already_consistent = bool(float(np.max(raw_pos)) < 1e-6 and float(np.max(raw_rot)) < 1e-6)
    fixed_transform_resolves = bool(float(np.max(best_pos)) < 1e-3 and float(np.max(best_rot)) < 1e-3)
    if raw_already_consistent:
        mismatch_type = "selected Pinocchio frame is consistent with the virtual SRS remodel"
    elif fixed_transform_resolves:
        mismatch_type = "fixed frame/tool transform mismatch"
    elif float(np.max(raw_pos)) < 1e-4 and float(np.max(raw_rot)) > 1e-3:
        mismatch_type = "orientation convention mismatch or virtual SRS remodel orientation approximation"
    else:
        mismatch_type = "virtual model approximation or wrong frame selection"

    payload = {
        "urdf": str(urdf),
        "matlab_reference_file": str(matlab_reference),
        "comparison_target": "virtual SRS remodel wrist-center FK vs selected full-URDF Pinocchio frame",
        "pinocchio_frame": frame_name,
        "candidate_frame_scores": frame_scores,
        "samples": int(samples),
        "position_error_mean_m": float(np.mean(raw_pos)),
        "position_error_max_m": float(np.max(raw_pos)),
        "rotation_error_mean_rad": float(np.mean(raw_rot)),
        "rotation_error_max_rad": float(np.max(raw_rot)),
        "right_tcp_fixed_transform": right_fixed_transform.tolist(),
        "right_tcp_position_error_mean_m": float(np.mean(right_pos)),
        "right_tcp_position_error_max_m": float(np.max(right_pos)),
        "right_tcp_rotation_error_mean_rad": float(np.mean(right_rot)),
        "right_tcp_rotation_error_max_rad": float(np.max(right_rot)),
        "left_base_fixed_transform": left_fixed_transform.tolist(),
        "left_base_position_error_mean_m": float(np.mean(left_pos)),
        "left_base_position_error_max_m": float(np.max(left_pos)),
        "left_base_rotation_error_mean_rad": float(np.mean(left_rot)),
        "left_base_rotation_error_max_rad": float(np.max(left_rot)),
        "best_fixed_transform": best_fixed,
        "fixed_transform_resolves_error": fixed_transform_resolves,
        "raw_already_consistent": raw_already_consistent,
        "diagnosis": mismatch_type,
        "virtual_model_wording": "virtual SRS remodel for analytic IK",
    }
    (out_dir / "fk_validation_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="tight")
    axes[0].hist(raw_pos, bins=40, alpha=0.75, label="raw")
    axes[0].hist(best_pos, bins=40, alpha=0.55, label=f"fixed {best_fixed}")
    axes[0].set_xlabel("Position error / m")
    axes[0].set_ylabel("Count")
    axes[0].set_title("FK position error")
    axes[0].legend()
    axes[1].hist(raw_rot, bins=40, alpha=0.75, label="raw")
    axes[1].hist(best_rot, bins=40, alpha=0.55, label=f"fixed {best_fixed}")
    axes[1].set_xlabel("Rotation error / rad")
    axes[1].set_ylabel("Count")
    axes[1].set_title("FK rotation error")
    axes[1].legend()
    fig.savefig(out_dir / "fig_fk_error_distribution.png", dpi=320)
    plt.close(fig)

    report = [
        "# Virtual SRS Remodel vs Pinocchio FK Validation",
        "",
        f"URDF: `{urdf}`",
        f"MATLAB reference: `{matlab_reference}`",
        "Comparison: virtual SRS remodel wrist-center FK against full-URDF Pinocchio frame.",
        f"Pinocchio frame compared: `{frame_name}`",
        f"Samples: {samples}",
        "",
        "## Raw Error",
        f"- Mean position error: {payload['position_error_mean_m']:.6f} m",
        f"- Max position error: {payload['position_error_max_m']:.6f} m",
        f"- Mean rotation error: {payload['rotation_error_mean_rad']:.6f} rad",
        f"- Max rotation error: {payload['rotation_error_max_rad']:.6f} rad",
        "",
        "## Fixed-Transform Diagnosis",
        f"- Best fixed transform model: `{best_fixed}`",
        f"- Mean position error after fixed transform: {float(np.mean(best_pos)):.6f} m",
        f"- Max position error after fixed transform: {float(np.max(best_pos)):.6f} m",
        f"- Mean rotation error after fixed transform: {float(np.mean(best_rot)):.6f} rad",
        f"- Max rotation error after fixed transform: {float(np.max(best_rot)):.6f} rad",
        f"- Fixed transform resolves error: `{fixed_transform_resolves}`",
        "",
        "## Conclusion",
        f"- Diagnosis: {mismatch_type}.",
        "- Use wording: `virtual SRS remodel for analytic IK`.",
        "- Do not claim exact full-URDF analytic IK unless this validation passes with the selected frame and fixed transform.",
    ]
    (out_dir / "fk_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main() -> None:
    urdf = PYCHARM_URDF if USE_PYCHARM_CONFIG else Path(sys.argv[1]) if len(sys.argv) > 1 else PYCHARM_URDF
    n = PYCHARM_SAMPLES if USE_PYCHARM_CONFIG else int(sys.argv[2]) if len(sys.argv) > 2 else PYCHARM_SAMPLES
    out = PYCHARM_OUT if USE_PYCHARM_CONFIG else Path(sys.argv[3]) if len(sys.argv) > 3 else PYCHARM_OUT
    run_validation(urdf, out, samples=n, seed=PYCHARM_SEED)


if __name__ == "__main__":
    main()
