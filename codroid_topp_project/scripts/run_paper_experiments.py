#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codroid_topp.experiments.paper_benchmark import (
    ALL_METHODS,
    DEFAULT_METHODS,
    BenchmarkResult,
    PaperExperimentConfig,
    aggregate_metric_rows,
    run_path_validation,
    run_paper_benchmark,
    select_representative_rows,
    write_csv,
    write_json,
    write_latex_tables,
    write_analysis_report,
    plot_benchmark_outputs,
)

SA_MODULE_TORQUE_NM = {
    "SA14": {"rated": 10.0, "startup": 36.0, "instantaneous": 70.0, "average_load": 14.0},
    "SA17": {"rated": 31.0, "startup": 70.0, "instantaneous": 143.0, "average_load": 51.0},
    "SA20": {"rated": 52.0, "startup": 107.0, "instantaneous": 191.0, "average_load": 64.0},
}
SA_MODULE_SPEED_RPM = {
    "SA14": {"rated": 49.5, "max": 56.0},
    "SA17": {"rated": 34.6, "max": 44.0},
    "SA20": {"rated": 34.6, "max": 39.8},
}
LEFT_ARM_MODULES = ("SA20", "SA20", "SA17", "SA17", "SA17", "SA14", "SA14")
TORQUE_SOURCE_CHOICES = ("rated", "average_load", "startup", "instantaneous")
SPEED_SOURCE_CHOICES = ("rated", "max")


def safety_factor_arg(raw: str) -> float:
    value = float(raw)
    if abs(value - 0.8) > 1e-12:
        raise argparse.ArgumentTypeError("actuator safety factor is fixed to 0.8; sweep torque_scale instead")
    return value


def actuator_tau_limits(source: str, safety_factor: float) -> tuple[float, ...]:
    return tuple(SA_MODULE_TORQUE_NM[m][source] * safety_factor for m in LEFT_ARM_MODULES)


def actuator_v_limits(source: str) -> tuple[float, ...]:
    return tuple(SA_MODULE_SPEED_RPM[m][source] * 2.0 * 3.141592653589793 / 60.0 for m in LEFT_ARM_MODULES)


# ======================= PyCharm direct-run settings =======================
# I normally run this file directly from PyCharm. Change only this variable.
USE_PYCHARM_CONFIG = True
PYCHARM_EXPERIMENT_PRESET = "torque_constraint_sweep"
# Available presets:
#   smoke_fast
#   validate_solvers
#   validate_paths
#   validate_fk
#   paper_main_stable
#   paper_main_statistics
#   adaptive_vs_smooth_oracle
#   torque_constraint_sweep
#   online_path_change
#   appendix_failure_diagnosis
#   paper_quality_full
# Legacy alias:
#   main_4x2


def pycharm_preset_argv(name: str) -> list[str]:
    common = [
        "--require-pinocchio",
        "--actuator-torque-source", "rated",
        "--actuator-safety-factor", "0.8",
        "--realistic-velocity-limits",
        "--actuator-speed-source", "rated",
        "--velocity-scale", "0.8",
        "--torque-scale", "1.0",
        "--seed", "42",
    ]
    presets: dict[str, list[str]] = {
        "smoke_fast": [
            "--preset", "fast",
            "--experiment-suite", "smoke_fast",
            "--tasks", "cup_transfer",
            "--methods", "streaming_smooth_psi", "streaming_adaptive_psi",
            "--constraint-modes", "dynamic",
            "--variants", "1",
            "--perturb-scale", "0.2",
            "--out", str(ROOT / "results_smoke_fast"),
            *common,
        ],
        "validate_solvers": [
            "--preset", "fast",
            "--experiment-suite", "validate_solvers",
            "--out", str(ROOT / "results_solver_validation"),
            *common,
        ],
        "validate_paths": [
            "--preset", "fast",
            "--experiment-suite", "validate_paths",
            "--tasks", "cup_transfer", "drawer_reach", "medicine_handover", "fast_lift_transfer", "near_limit_reach",
            "--variants", "5",
            "--perturb-scale", "0.8",
            "--out", str(ROOT / "results_path_validation"),
            *common,
        ],
        "validate_fk": [
            "--preset", "fast",
            "--experiment-suite", "validate_fk",
            "--out", str(ROOT / "results_fk_validation"),
            *common,
        ],
        "paper_main_stable": [
            "--preset", "fast",
            "--experiment-suite", "paper_main_stable",
            "--tasks", "cup_transfer", "drawer_reach", "medicine_handover",
            "--methods", "offline_topp_ni_grid", "offline_topp_ra", "offline_topp_co", "streaming_smooth_psi", "streaming_adaptive_psi",
            "--constraint-modes", "dynamic",
            "--variants", "2",
            "--perturb-scale", "0.8",
            "--adaptive-n-psi", "3",
            "--adaptive-psi-radius", "0.50",
            "--adaptive-use-torque-proxy",
            "--out", str(ROOT / "results_paper_main_stable"),
            *common,
        ],
        "paper_main_statistics": [
            "--preset", "quality",
            "--experiment-suite", "paper_main_statistics",
            "--tasks", "cup_transfer", "drawer_reach", "medicine_handover",
            "--methods", "offline_topp_ra", "offline_topp_co", "streaming_fixed_psi", "streaming_smooth_psi", "streaming_adaptive_psi", "streaming_topp_aware_psi",
            "--constraint-modes", "dynamic",
            "--variants", "10",
            "--perturb-scale", "0.8",
            "--adaptive-n-psi", "3",
            "--adaptive-psi-radius", "0.50",
            "--adaptive-use-torque-proxy",
            "--out", str(ROOT / "results_paper_main_statistics"),
            *common,
        ],
        "main_4x2": [
            "--preset", "fast",
            "--experiment-suite", "offline_vs_online_kinematic_dynamic",
            "--methods", "offline_topp_ni_grid", "offline_topp_ra", "offline_topp_co", "streaming_adaptive_psi",
            "--constraint-modes", "kinematic", "dynamic",
            "--variants", "2",
            "--perturb-scale", "1.0",
            "--out", str(ROOT / "results_offline_vs_online_kinodynamic"),
            *common,
        ],
        "adaptive_vs_smooth_oracle": [
            "--preset", "fast",
            "--experiment-suite", "adaptive_vs_smooth_oracle",
            "--tasks", "cup_transfer", "drawer_reach", "medicine_handover", "near_limit_reach",
            "--methods", "streaming_fixed_psi", "streaming_smooth_psi", "streaming_adaptive_psi", "streaming_topp_aware_psi",
            "--constraint-modes", "dynamic",
            "--variants", "10",
            "--perturb-scale", "0.8",
            "--adaptive-n-psi", "3",
            "--adaptive-psi-radius", "0.50",
            "--adaptive-use-torque-proxy",
            "--out", str(ROOT / "results_adaptive_vs_smooth_oracle"),
            *common,
        ],
        "torque_constraint_sweep": [
            "--preset", "fast",
            "--experiment-suite", "torque_constraint_sweep",
            "--tasks", "cup_transfer", "near_limit_reach",
            "--methods", "offline_topp_ra", "streaming_smooth_psi", "streaming_adaptive_psi",
            "--constraint-modes", "kinematic", "dynamic",
            "--variants", "2",
            "--perturb-scale", "0.8",
            "--torque-scale-sweep", "1.0", "0.8", "0.6", "0.55", "0.5", "0.45",
            "--velocity-scale-sweep", "0.8", "1.0",
            "--adaptive-use-torque-proxy",
            "--out", str(ROOT / "results_torque_constraint_sweep"),
            *common,
        ],
        "online_path_change": [
            "--preset", "fast",
            "--experiment-suite", "online_path_change",
            "--tasks", "cup_transfer", "drawer_reach", "medicine_handover",
            "--methods", "offline_topp_ra", "streaming_smooth_psi", "streaming_adaptive_psi",
            "--constraint-modes", "dynamic",
            "--variants", "5",
            "--perturb-scale", "0.8",
            "--path-change-s", "0.45",
            "--path-change-perturb-scale", "1.8",
            "--out", str(ROOT / "results_online_path_change"),
            *common,
        ],
        "appendix_failure_diagnosis": [
            "--preset", "fast",
            "--experiment-suite", "appendix_failure_diagnosis",
            "--tasks", "medicine_handover", "fast_lift_transfer", "near_limit_reach",
            "--methods", "offline_topp_ni_grid", "offline_topp_ra", "offline_convex_like_diagnostic", "streaming_fixed_psi", "streaming_smooth_psi", "streaming_adaptive_psi",
            "--constraint-modes", "kinematic", "dynamic",
            "--variants", "3",
            "--perturb-scale", "1.0",
            "--adaptive-n-psi", "3",
            "--adaptive-psi-radius", "0.50",
            "--adaptive-use-torque-proxy",
            "--out", str(ROOT / "results_appendix_failure_diagnosis"),
            *common,
        ],
        "paper_quality_full": [
            "--preset", "quality",
            "--experiment-suite", "paper_quality_full",
            "--tasks", "cup_transfer", "drawer_reach", "medicine_handover", "fast_lift_transfer", "near_limit_reach",
            "--methods", "offline_topp_ni_grid", "offline_topp_ra", "offline_convex_like_diagnostic", "rolling_global_dp", "streaming_fixed_psi", "streaming_smooth_psi", "streaming_adaptive_psi", "streaming_topp_aware_psi",
            "--constraint-modes", "kinematic", "dynamic",
            "--variants", "8",
            "--perturb-scale", "1.0",
            "--adaptive-n-psi", "3",
            "--adaptive-psi-radius", "0.50",
            "--adaptive-use-torque-proxy",
            "--out", str(ROOT / "results_paper_quality_full"),
            *common,
        ],
    }
    if name not in presets:
        raise ValueError(f"Unknown PYCHARM_EXPERIMENT_PRESET={name!r}. Choose from {sorted(presets)}")
    return presets[name]


PYCHARM_ARGV = pycharm_preset_argv(PYCHARM_EXPERIMENT_PRESET)
# ========================================================================== 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-grade Codroid rolling TOPP benchmark and ablations.")
    parser.add_argument("--urdf", type=Path, default=ROOT / "configs" / "codroidRobot.urdf")
    parser.add_argument("--out", type=Path, default=ROOT / "results_paper")
    parser.add_argument("--tasks", nargs="+", default=["cup_transfer", "drawer_reach", "medicine_handover"], choices=["cup_transfer", "drawer_reach", "medicine_handover", "fast_lift_transfer", "near_limit_reach"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(ALL_METHODS))
    parser.add_argument("--preset", choices=["fast", "quality", "paper"], default="quality")
    parser.add_argument("--experiment-suite", type=str, default="paper_benchmark")
    parser.add_argument("--constraint-modes", nargs="+", default=["dynamic"], choices=["kinematic", "dynamic"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--dense-count", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--horizon-points", type=int, default=None)
    parser.add_argument("--brake-points", type=int, default=None)
    parser.add_argument("--commit-points", type=int, default=None)
    parser.add_argument("--terminal-safe-speed", type=float, default=0.04)
    parser.add_argument("--tau-max", type=float, nargs=7, default=None)
    parser.add_argument("--actuator-torque-source", choices=TORQUE_SOURCE_CHOICES, default="rated")
    parser.add_argument("--actuator-safety-factor", type=safety_factor_arg, default=0.8)
    parser.add_argument("--v-max", type=float, nargs=7, default=None)
    parser.add_argument("--realistic-velocity-limits", "--realistic-vmax", action="store_true", dest="realistic_velocity_limits")
    parser.add_argument("--actuator-speed-source", choices=SPEED_SOURCE_CHOICES, default="rated")
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument("--velocity-scale", type=float, default=0.8)
    parser.add_argument("--torque-scale-sweep", type=float, nargs="+", default=None)
    parser.add_argument("--velocity-scale-sweep", type=float, nargs="+", default=None)
    parser.add_argument("--search-grid", type=int, default=None)
    parser.add_argument("--bisection-iters", type=int, default=None)
    parser.add_argument("--psi-grid-count", type=int, default=41)
    parser.add_argument("--local-psi-radius", type=float, default=0.25)
    parser.add_argument("--local-n-psi", type=int, default=5)
    parser.add_argument("--smooth-psi-radius", type=float, default=0.35)
    parser.add_argument("--smooth-n-psi", type=int, default=3)
    parser.add_argument("--smooth-accept-margin", type=float, default=0.18)
    parser.add_argument("--adaptive-psi-radius", type=float, default=0.45)
    parser.add_argument("--adaptive-n-psi", type=int, default=5)
    parser.add_argument("--adaptive-accept-margin", type=float, default=0.20)
    parser.add_argument("--adaptive-min-manipulability", type=float, default=1.0e-4)
    parser.add_argument("--adaptive-use-torque-proxy", action="store_true")
    parser.add_argument("--topp-aware-psi-radius", type=float, default=0.45)
    parser.add_argument("--topp-aware-n-psi", type=int, default=7)
    parser.add_argument("--compute-time-budget-ms", type=float, default=50.0)
    parser.add_argument("--max-reasonable-duration-s", type=float, default=60.0)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturb-scale", type=float, default=0.0)
    parser.add_argument("--path-change-s", type=float, default=0.45)
    parser.add_argument("--path-change-perturb-scale", type=float, default=1.5)
    parser.add_argument("--no-endpoint-check", action="store_true")
    parser.add_argument("--no-pinocchio", action="store_true")
    parser.add_argument("--require-pinocchio", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    # PyCharm direct-run remains the primary workflow: with no command-line
    # arguments, use the preset selected near the top of this file. Explicit
    # CLI arguments are still honored for automated checks.
    argv = PYCHARM_ARGV if USE_PYCHARM_CONFIG and len(sys.argv) == 1 else None
    return parser.parse_args(argv)


def _preset_defaults(preset: str) -> dict[str, int]:
    if preset == "fast":
        return {"samples": 32, "dense_count": 520, "repeats": 1, "horizon_points": 5, "brake_points": 2, "commit_points": 2, "search_grid": 3, "bisection_iters": 5}
    if preset == "paper":
        return {"samples": 90, "dense_count": 1400, "repeats": 2, "horizon_points": 10, "brake_points": 5, "commit_points": 2, "search_grid": 7, "bisection_iters": 10}
    return {"samples": 60, "dense_count": 900, "repeats": 1, "horizon_points": 8, "brake_points": 4, "commit_points": 2, "search_grid": 5, "bisection_iters": 8}


def build_config(args: argparse.Namespace, torque_scale: float | None = None, velocity_scale: float | None = None) -> PaperExperimentConfig:
    tau_max = actuator_tau_limits(args.actuator_torque_source, float(args.actuator_safety_factor)) if args.tau_max is None else tuple(float(v) for v in args.tau_max)
    v_max = tuple(float(v) for v in args.v_max) if args.v_max is not None else (actuator_v_limits(args.actuator_speed_source) if args.realistic_velocity_limits else None)
    d = _preset_defaults(args.preset)
    return PaperExperimentConfig(
        tasks=tuple(args.tasks),
        methods=tuple(args.methods),
        constraint_modes=tuple(args.constraint_modes),
        experiment_suite=str(args.experiment_suite),
        samples=int(args.samples if args.samples is not None else d["samples"]),
        dense_count=int(args.dense_count if args.dense_count is not None else d["dense_count"]),
        repeats=int(args.repeats if args.repeats is not None else d["repeats"]),
        horizon_points=int(args.horizon_points if args.horizon_points is not None else d["horizon_points"]),
        brake_points=int(args.brake_points if args.brake_points is not None else d["brake_points"]),
        commit_points=int(args.commit_points if args.commit_points is not None else d["commit_points"]),
        terminal_safe_speed=float(args.terminal_safe_speed),
        tau_max=tau_max,
        actuator_modules=LEFT_ARM_MODULES,
        actuator_torque_source=None if args.tau_max is not None else "rated",
        actuator_safety_factor=None if args.tau_max is not None else 0.8,
        v_max=v_max,
        realistic_velocity_limits=bool(args.realistic_velocity_limits),
        actuator_speed_source=str(args.actuator_speed_source),
        torque_scale=float(args.torque_scale if torque_scale is None else torque_scale),
        velocity_scale=float(args.velocity_scale if velocity_scale is None else velocity_scale),
        search_grid=int(args.search_grid if args.search_grid is not None else d["search_grid"]),
        bisection_iters=int(args.bisection_iters if args.bisection_iters is not None else d["bisection_iters"]),
        endpoint_check=not args.no_endpoint_check,
        psi_grid_count=int(args.psi_grid_count),
        local_psi_radius=float(args.local_psi_radius),
        local_n_psi=int(args.local_n_psi),
        smooth_psi_radius=float(args.smooth_psi_radius),
        smooth_n_psi=int(args.smooth_n_psi),
        smooth_accept_margin=float(args.smooth_accept_margin),
        adaptive_psi_radius=float(args.adaptive_psi_radius),
        adaptive_n_psi=int(args.adaptive_n_psi),
        adaptive_accept_margin=float(args.adaptive_accept_margin),
        adaptive_min_manipulability=float(args.adaptive_min_manipulability),
        adaptive_use_torque_proxy=bool(args.adaptive_use_torque_proxy),
        topp_aware_psi_radius=float(args.topp_aware_psi_radius),
        topp_aware_n_psi=int(args.topp_aware_n_psi),
        compute_time_budget_ms=float(args.compute_time_budget_ms),
        max_reasonable_duration_s=float(args.max_reasonable_duration_s),
        variants=int(args.variants),
        seed=int(args.seed),
        perturb_scale=float(args.perturb_scale),
        path_change_s=float(args.path_change_s),
        path_change_perturb_scale=float(args.path_change_perturb_scale),
        torque_scale_sweep=None if args.torque_scale_sweep is None else tuple(float(v) for v in args.torque_scale_sweep),
        velocity_scale_sweep=None if args.velocity_scale_sweep is None else tuple(float(v) for v in args.velocity_scale_sweep),
    )


def write_sweep_root_outputs(root_out: Path, rows: list[dict[str, float | int | str]], window_rows: list[dict[str, float | int | str]], args: argparse.Namespace) -> None:
    root_out.mkdir(parents=True, exist_ok=True)
    write_csv(root_out / "summary_metrics.csv", rows, ["sweep_torque_scale", "sweep_velocity_scale", "task", "variant_id", "method", "constraint_mode", "status", "duration"])
    write_json(root_out / "summary_metrics.json", rows)
    aggregated = aggregate_metric_rows(rows)
    write_csv(root_out / "summary_metrics_aggregated.csv", aggregated, [])
    write_json(root_out / "summary_metrics_aggregated.json", aggregated)
    write_latex_tables(root_out, rows, aggregated)
    selected = select_representative_rows(rows)
    write_csv(root_out / "selected_representative_runs.csv", selected, ["task", "method", "method_label", "constraint_mode", "variant_id", "repeat", "duration", "completed_path_pct", "selected_by"])
    if window_rows:
        write_csv(root_out / "window_log.csv", window_rows, ["sweep_torque_scale", "sweep_velocity_scale", "task", "variant_id", "method", "constraint_mode", "repeat", "window_id", "s_start", "compute_time_ms"])
        write_json(root_out / "window_log.json", window_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_suite": "torque_constraint_sweep",
        "torque_scale_sweep": [float(v) for v in args.torque_scale_sweep or []],
        "velocity_scale_sweep": [float(v) for v in args.velocity_scale_sweep or []],
        "subdirectories": sorted({str(r.get("sweep_dir", "")) for r in rows if r.get("sweep_dir")}),
        "dynamics_backend": ", ".join(sorted({str(r.get("dynamics_backend", "")) for r in rows if str(r.get("dynamics_backend", "")).strip()})) or "unknown",
        "actuator_torque_source": "rated",
        "actuator_safety_factor": 0.8,
        "tau_limit_formula": "rated actuator torque x 0.8 x torque_scale",
        "tau_max_before_torque_scale": list(actuator_tau_limits("rated", 0.8)),
        "final_tau_max": list(actuator_tau_limits("rated", 0.8)),
    }
    write_json(root_out / "experiment_manifest.json", manifest)
    result = BenchmarkResult(metric_rows=rows, window_rows=window_rows, trajectories={}, manifest=manifest)
    cfg = build_config(args)
    try:
        plot_benchmark_outputs(result, root_out, cfg)
    except Exception as exc:
        (root_out / "plot_generation_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    write_analysis_report(result, root_out, cfg)


def main() -> None:
    args = parse_args()
    active_source = PYCHARM_EXPERIMENT_PRESET if USE_PYCHARM_CONFIG and len(sys.argv) == 1 else "CLI"
    print(f"[paper] PyCharm preset: {active_source}")
    print(f"[paper] left-arm modules: {' '.join(LEFT_ARM_MODULES)}")
    print("[paper] tau source: rated")
    print("[paper] safety factor: 0.8")
    print("[paper] final tau formula: rated x 0.8 x torque_scale")
    print(f"[paper] tau before torque_scale: {[round(v,3) for v in actuator_tau_limits('rated', 0.8)]} Nm")
    if args.experiment_suite == "validate_solvers":
        from scripts.validate_topp_solvers import run_validation
        run_validation(args.urdf, args.out)
        return
    if args.experiment_suite == "validate_paths":
        cfg = build_config(args)
        rows = run_path_validation(args.urdf, args.out, cfg, prefer_pinocchio=not args.no_pinocchio, require_pinocchio=args.require_pinocchio)
        ok = sum(1 for r in rows if float(r.get("path_validation_success", 0.0)) > 0.5)
        print(f"[paper] path validation: {ok}/{len(rows)} task variants valid")
        print(f"[paper] outputs written to: {args.out}")
        return
    if args.experiment_suite == "validate_fk":
        from scripts.validate_virtual_srs_vs_pinocchio_fk import run_validation
        run_validation(args.urdf, args.out)
        return
    if args.experiment_suite == "torque_constraint_sweep" and args.torque_scale_sweep and args.velocity_scale_sweep:
        root_out = Path(args.out)
        all_ok = 0; all_total = 0
        sweep_rows: list[dict[str, float | int | str]] = []
        sweep_window_rows: list[dict[str, float | int | str]] = []
        for ts in args.torque_scale_sweep:
            for vs in args.velocity_scale_sweep:
                cfg = build_config(args, torque_scale=ts, velocity_scale=vs)
                out = root_out / f"torque{ts:g}_vel{vs:g}".replace('.', 'p')
                print(f"[paper] sweep torque_scale={ts:g}, velocity_scale={vs:g} -> {out}")
                result = run_paper_benchmark(args.urdf, out, cfg, prefer_pinocchio=not args.no_pinocchio, require_pinocchio=args.require_pinocchio, make_plots=not args.no_plots)
                all_ok += sum(1 for r in result.metric_rows if str(r.get("status")) == "ok")
                all_total += len(result.metric_rows)
                for row in result.metric_rows:
                    rr = dict(row)
                    rr["sweep_torque_scale"] = float(ts)
                    rr["sweep_velocity_scale"] = float(vs)
                    rr["sweep_dir"] = str(out)
                    rr["dynamics_backend"] = str(result.manifest.get("dynamics_backend", "unknown"))
                    sweep_rows.append(rr)
                for row in result.window_rows:
                    rr = dict(row)
                    rr["sweep_torque_scale"] = float(ts)
                    rr["sweep_velocity_scale"] = float(vs)
                    rr["sweep_dir"] = str(out)
                    rr["dynamics_backend"] = str(result.manifest.get("dynamics_backend", "unknown"))
                    sweep_window_rows.append(rr)
        write_sweep_root_outputs(root_out, sweep_rows, sweep_window_rows, args)
        print(f"[paper] sweep complete: {all_ok}/{all_total} runs ok")
        print(f"[paper] sweep root: {root_out}")
        return
    cfg = build_config(args)
    result = run_paper_benchmark(args.urdf, args.out, cfg, prefer_pinocchio=not args.no_pinocchio, require_pinocchio=args.require_pinocchio, make_plots=not args.no_plots)
    ok = sum(1 for row in result.metric_rows if str(row.get("status", "")) == "ok")
    print(f"[paper] metrics: {ok}/{len(result.metric_rows)} method runs completed")
    print(f"[paper] dynamics backend: {result.manifest.get('dynamics_backend')}")
    print(f"[paper] outputs written to: {args.out}")
    print(f"[paper] summary: {args.out / 'summary_metrics_aggregated.csv'}")


if __name__ == "__main__":
    main()
