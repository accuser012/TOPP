#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from collections import defaultdict
from math import isfinite

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ======================= PyCharm direct-run settings =======================
USE_PYCHARM_CONFIG = True
PYCHARM_RESULTS_DIR = ROOT / "results_offline_vs_online_kinodynamic"
# ========================================================================== 


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _finite(x) -> bool:
    try:
        return isfinite(float(x))
    except Exception:
        return False


def _group(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(str(r.get(k, "")) for k in keys)].append(r)
    return out


def analyze(results_dir: Path) -> str:
    rows = _read_json(results_dir / "summary_metrics.json")
    if rows is None:
        rows = _read_csv(results_dir / "summary_metrics.csv")
    agg = _read_json(results_dir / "summary_metrics_aggregated.json") or _read_csv(results_dir / "summary_metrics_aggregated.csv")
    win = _read_json(results_dir / "window_log.json") or _read_csv(results_dir / "window_log.csv")
    manifest = _read_json(results_dir / "experiment_manifest.json") or {}

    lines: list[str] = []
    lines.append("# Codroid TOPP Experiment Analysis Report")
    lines.append("")
    lines.append(f"Results directory: `{results_dir}`")
    lines.append(f"Runs: {len(rows)} metric rows, {len(win)} window rows")
    lines.append(f"Dynamics backend: `{manifest.get('dynamics_backend', 'unknown')}`")
    lines.append(f"Torque source: `{manifest.get('actuator_torque_source', 'unknown')}`")
    lines.append(f"Actuator safety factor: `{manifest.get('actuator_safety_factor', 'unknown')}`")
    lines.append(f"Tau limit formula: `{manifest.get('tau_limit_formula', 'rated actuator torque x 0.8 x torque_scale')}`")
    lines.append(f"Final tau max: `{manifest.get('final_tau_max', manifest.get('planner_limits', {}).get('tau_max', []))}`")
    lines.append("")

    lines.append("## Validation gates")
    root = ROOT
    solver_passed = False
    fk_explained = False
    path_present = False
    for label, rel in [
        ("Solver validation", "results_solver_validation/solver_validation_report.md"),
        ("Path validation", "results_path_validation/path_validation_report.md"),
        ("FK validation", "results_fk_validation/fk_validation_report.md"),
    ]:
        p = root / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            status = "present"
            if label == "Solver validation" and "Solver validation passed: `True`" not in txt:
                status = "present but not passed"
            if label == "Solver validation":
                solver_passed = "Solver validation passed: `True`" in txt
            if label == "Path validation":
                path_present = True
            if label == "FK validation":
                fk_explained = "Diagnosis:" in txt or "Fixed transform resolves error:" in txt
            lines.append(f"- {label}: {status} (`{p}`)")
        else:
            lines.append(f"- WARNING: {label} missing (`{p}`).")
    lines.append("")

    if not rows:
        lines.append("## Fatal")
        lines.append("No summary metric rows were found.")
        return "\n".join(lines)

    ok = [r for r in rows if _num(r.get("valid_for_main_paper"), _num(r.get("quality_gate_passed"), 0)) > 0.5]
    lines.append("## Completion")
    lines.append(f"Successful rows: {len(ok)}/{len(rows)}")
    failed = [r for r in rows if r not in ok]
    if failed:
        lines.append("Failed or incomplete examples:")
        for r in failed[:10]:
            lines.append(f"- {r.get('task')} {r.get('method')} {r.get('constraint_mode')}: {r.get('status')} {r.get('error','')}")
    lines.append("")

    lines.append("## Runtime budget")
    for r in rows:
        p95 = _num(r.get("p95_compute_time_ms", r.get("p95_compute_time_ms_mean")))
        if _finite(p95) and p95 > 50.0 and _num(r.get("is_online_method"), 0) > 0.5:
            lines.append(f"- WARNING: {r.get('method')} on {r.get('task')} exceeds 50 ms p95: {p95:.1f} ms")
    if not any("WARNING" in s for s in lines[-10:]):
        lines.append("No online p95 runtime warning found in direct metric rows.")
    lines.append("")

    lines.append("## Offline baseline equivalence check")
    off_methods = {"offline_topp_ni_grid", "offline_topp_ni", "offline_topp_ra", "offline_topp_co", "offline_convex_topp_like", "offline_global_dp"}
    for key, items in _group([r for r in rows if str(r.get("method")) in off_methods], ["task", "variant_id", "constraint_mode", "repeat"]).items():
        by_m = {str(r.get("method")): r for r in items}
        if len(by_m) >= 2:
            durations = {m: round(_num(r.get("duration")), 6) for m, r in by_m.items() if _finite(_num(r.get("duration")))}
            if len(set(durations.values())) <= 1 and len(durations) >= 2:
                lines.append(f"- WARNING: offline baselines have identical duration for {key}: {durations}")
    approx = [r for r in rows if _num(r.get("baseline_is_approximate"), 0) > 0.5]
    fallback = [r for r in rows if _num(r.get("baseline_fallback_used"), 0) > 0.5]
    unavailable = [r for r in rows if _num(r.get("baseline_unavailable"), 0) > 0.5 or str(r.get("method", "")) == "offline_topp_co" and str(r.get("status", "")) != "ok"]
    lines.append(f"Approximate baseline rows: {len(approx)}")
    lines.append(f"Fallback baseline rows: {len(fallback)}")
    lines.append(f"Unavailable/failed TOPP-CO rows: {len(unavailable)}")
    lines.append("")

    lines.append("## Kinematic vs dynamic and torque activity")
    kin = [r for r in rows if str(r.get("constraint_mode")) == "kinematic"]
    dyn = [r for r in rows if str(r.get("constraint_mode")) == "dynamic"]
    torque_viol = [r for r in kin if _num(r.get("posthoc_max_torque_violation"), 0) > 1e-7]
    torque_util_gt1 = [r for r in kin if _num(r.get("posthoc_max_torque_utilization"), 0) > 1.0]
    lines.append(f"Kinematic rows with post-hoc torque violation: {len(torque_viol)}/{len(kin)}")
    lines.append(f"Kinematic rows with post-hoc torque utilization > 1: {len(torque_util_gt1)}/{len(kin)}")
    if not torque_viol:
        lines.append("- NOTE: current tasks/settings did not make the torque constraint active enough to prove dynamic-constraint necessity.")
    for key, items in _group(rows, ["task", "variant_id", "method", "repeat"]).items():
        kk = [r for r in items if str(r.get("constraint_mode")) == "kinematic"]
        dd = [r for r in items if str(r.get("constraint_mode")) == "dynamic"]
        if kk and dd:
            if abs(_num(kk[0].get("duration")) - _num(dd[0].get("duration"))) < 1e-6:
                lines.append(f"- NOTE: kinematic/dynamic duration identical for {key}")
    lines.append("")

    lines.append("## Adaptive psi diagnostic")
    for key, items in _group(rows, ["task", "variant_id", "constraint_mode", "repeat"]).items():
        smooth = next((r for r in items if str(r.get("method")) == "streaming_smooth_psi"), None)
        adapt = next((r for r in items if str(r.get("method")) == "streaming_adaptive_psi"), None)
        if smooth and adapt:
            better = []
            if _num(adapt.get("duration"), 1e9) < _num(smooth.get("duration"), 1e9):
                better.append("duration")
            if _num(adapt.get("min_limit_margin_rad"), -1) > _num(smooth.get("min_limit_margin_rad"), -1):
                better.append("limit_margin")
            if _num(adapt.get("mean_manipulability"), -1) > _num(smooth.get("mean_manipulability"), -1):
                better.append("manipulability")
            if _num(adapt.get("max_torque_utilization"), 1e9) < _num(smooth.get("max_torque_utilization"), 1e9):
                better.append("torque_util")
            if not better:
                lines.append(f"- WARNING: adaptive psi does not beat smooth psi for {key}")
            else:
                lines.append(f"- Adaptive improves {', '.join(better)} for {key}")
    if win:
        triggers = defaultdict(int)
        for r in win:
            if str(r.get("method")) == "streaming_adaptive_psi":
                triggers[str(r.get("adaptive_trigger_reason", "missing"))] += 1
        if triggers:
            lines.append(f"Adaptive trigger counts: {dict(triggers)}")
    lines.append("")

    lines.append("## Online path-change")
    online = [r for r in rows if _num(r.get("online_path_change"), 0) > 0.5]
    if not online:
        lines.append("- WARNING: no online path-change metric rows were generated.")
    else:
        lines.append(f"Online path-change rows: {len(online)}")
        for r in online[:12]:
            lines.append(f"- {r.get('task')} {r.get('method')}: windows_after={r.get('windows_after_change')}, final_pos_err={r.get('final_position_error_to_changed_goal_m')}")
    lines.append("")

    lines.append("## Empty table / figure check")
    for name in ["table_online_path_change.tex", "table_main_method_comparison.tex", "table_kinematic_posthoc_torque.tex"]:
        p = results_dir / name
        if not p.exists():
            lines.append(f"- WARNING: missing {name}")
        elif len(p.read_text(encoding="utf-8", errors="ignore").strip()) < 80:
            lines.append(f"- WARNING: {name} appears empty or too short")
    fig_dir = results_dir / "figures"
    png_figs = list(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if not png_figs:
        lines.append("- WARNING: no PNG figures found")
    else:
        lines.append(f"PNG figures: {[p.name for p in png_figs]}")
    lines.append("")

    co_rows = [r for r in rows if str(r.get("method", "")) == "offline_topp_co"]
    co_valid = bool(co_rows) and all(_num(r.get("valid_for_main_paper"), 0) > 0.5 for r in co_rows if str(r.get("status", "")) == "ok")
    ni_rows = [r for r in rows if str(r.get("method", "")) in {"offline_topp_ni", "offline_topp_ni_grid"}]
    torque_formula_ok = str(manifest.get("actuator_torque_source", "")).lower() == "rated" and abs(_num(manifest.get("actuator_safety_factor"), -1) - 0.8) < 1e-12
    smooth_adapt_pairs = []
    adaptive_better = 0
    for key, items in _group(rows, ["task", "variant_id", "constraint_mode", "repeat"]).items():
        smooth = next((r for r in items if str(r.get("method")) == "streaming_smooth_psi" and _num(r.get("valid_for_main_paper"), 0) > 0.5), None)
        adapt = next((r for r in items if str(r.get("method")) == "streaming_adaptive_psi" and _num(r.get("valid_for_main_paper"), 0) > 0.5), None)
        if smooth and adapt:
            smooth_adapt_pairs.append(key)
            if (
                _num(adapt.get("duration"), 1e9) < _num(smooth.get("duration"), 1e9)
                or _num(adapt.get("max_torque_utilization"), 1e9) < _num(smooth.get("max_torque_utilization"), 1e9)
                or _num(adapt.get("rms_jerk"), 1e9) < _num(smooth.get("rms_jerk"), 1e9)
                or _num(adapt.get("min_limit_margin_rad"), -1) > _num(smooth.get("min_limit_margin_rad"), -1)
            ):
                adaptive_better += 1
    excluded_tasks = sorted({str(r.get("task", "")) for r in rows if _num(r.get("valid_for_main_paper"), 0) < 0.5})
    key_figs = {
        "fig_method_framework.png",
        "fig_main_path_speed_profiles.png",
        "fig_main_duration_comparison.png",
        "fig_runtime_statistics.png",
        "fig_compute_vs_samples.png",
        "fig_adaptive_advantage.png",
        "fig_dynamic_constraint_tradeoff.png",
    }
    paper_ready_figs = sorted({p.name for p in png_figs if p.name in key_figs or p.name.startswith("fig_joint_ratio_")})
    lines.append("## Final Experiment Conclusion")
    lines.append(f"1. Offline baselines validated: {'yes' if solver_passed else 'no'}; solver validation must pass before final paper claims.")
    lines.append(f"2. TOPP-CO available and valid: {'yes' if co_valid else 'no'}; valid rows={sum(1 for r in co_rows if _num(r.get('valid_for_main_paper'), 0) > 0.5)}/{len(co_rows)}.")
    lines.append("3. TOPP-NI is grid-based NI, not exact continuous switch-point Bobrow/Shin-McKay." if ni_rows else "3. TOPP-NI was not run in this result directory.")
    lines.append(f"4. Adaptive psi better than smooth psi: {adaptive_better}/{len(smooth_adapt_pairs)} comparable valid pairs show at least one improvement metric.")
    lines.append(f"5. Dynamic constraints matter: {'yes, in the tested sweep' if torque_viol else 'not proven in this result'}; kinematic post-hoc torque violations={len(torque_viol)}.")
    lines.append(f"6. Torque limits based on rated x 0.8: {'yes' if torque_formula_ok else 'no or not recorded'}.")
    lines.append(f"7. Paper-ready PNG figures: {paper_ready_figs if paper_ready_figs else 'none yet'}.")
    lines.append(f"8. Tasks/methods to exclude from main figures: {excluded_tasks if excluded_tasks else 'none from this directory by quality gate'}; exclude any method with valid_for_main_paper != 1.")
    lines.append("9. Supported contribution: streaming/receding-horizon TOPP with analytic IK and adaptive psi is supported only where runtime budget, completion, and adaptive advantage all hold; otherwise treat as diagnostic.")
    lines.append(f"10. Virtual SRS / URDF consistency: {'explained by FK validation report' if fk_explained else 'not yet explained; run validate_fk'}; use wording `virtual SRS remodel for analytic IK` unless exact-frame validation passes.")
    return "\n".join(lines)


def main() -> None:
    results_dir = PYCHARM_RESULTS_DIR if USE_PYCHARM_CONFIG else Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results_offline_vs_online_kinodynamic"
    report = analyze(Path(results_dir))
    out = Path(results_dir) / "analysis_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    final = Path(results_dir) / "final_experiment_conclusion.md"
    final.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[analysis] wrote {out}")
    print(f"[analysis] wrote {final}")


if __name__ == "__main__":
    main()
