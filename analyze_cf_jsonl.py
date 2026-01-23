import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Record:
    win_idx: int
    success: bool
    unit: str
    base_delta: float | None
    cf_delta: float | None
    num_changed_segments: int | None
    mean_abs_change_scaled: float | None
    max_abs_change_scaled: float | None
    last_obs_scaled: float | None
    last_obs_units: float | None
    base_pred_scaled: float | None
    base_pred_units: float | None
    query_ts_units: list[float] | None
    cf_ts_units: list[float] | None
    delta_mv_units: list[list[float]] | None
    details: dict[str, Any] | None

    # MASCOTS paper metrics (computed in scaled space, unitless)
    proximity_l1_scaled: float | None
    proximity_l2_scaled: float | None
    sparsity_count: int | None
    sparsity_ratio: float | None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _as_record(row: dict[str, Any]) -> Record:
    def g(key: str, default=None):
        return row.get(key, default)

    unit = "°C" if ("base_delta_degC" in row or "cf_delta_degC" in row) else "units"

    base_delta = g("base_delta_degC")
    if base_delta is None:
        base_delta = g("base_delta_units")
    cf_delta = g("cf_delta_degC")
    if cf_delta is None:
        cf_delta = g("cf_delta_units")

    last_obs_units = g("last_obs_degC")
    if last_obs_units is None:
        last_obs_units = g("last_obs_units")
    base_pred_units = g("base_pred_degC")
    if base_pred_units is None:
        base_pred_units = g("base_pred_units")

    query_ts_units = g("query_ts_degC")
    if query_ts_units is None:
        query_ts_units = g("query_ts_units")
    cf_ts_units = g("cf_ts_degC")
    if cf_ts_units is None:
        cf_ts_units = g("cf_ts_units")

    # Paper metrics may be absent in older JSONL; try to infer from stored scaled sequences.
    prox_l1 = g("proximity_l1_scaled")
    prox_l2 = g("proximity_l2_scaled")
    spar_count = g("sparsity_count")
    spar_ratio = g("sparsity_ratio")

    if prox_l1 is None or prox_l2 is None or spar_count is None or spar_ratio is None:
        try:
            eps = 1e-6
            d = None
            if g("cf_mv_scaled") is not None and g("query_mv_scaled") is not None:
                c = np.asarray(g("cf_mv_scaled"), dtype=np.float32)
                q = np.asarray(g("query_mv_scaled"), dtype=np.float32)
                d = (c - q).astype(np.float32)
            elif g("cf_ts_scaled") is not None and g("query_ts_scaled") is not None:
                c = np.asarray(g("cf_ts_scaled"), dtype=np.float32)
                q = np.asarray(g("query_ts_scaled"), dtype=np.float32)
                d = (c - q).astype(np.float32)
            if d is not None and d.size:
                if prox_l1 is None:
                    prox_l1 = float(np.mean(np.abs(d)))
                if prox_l2 is None:
                    prox_l2 = float(np.sqrt(np.mean(d * d)))
                if spar_count is None:
                    spar_count = int(np.sum(np.abs(d) > eps))
                if spar_ratio is None:
                    spar_ratio = float(spar_count / int(d.size))
        except Exception:
            pass

    return Record(
        win_idx=int(g("win_idx")),
        success=bool(g("success")),
        unit=unit,
        base_delta=base_delta,
        cf_delta=cf_delta,
        num_changed_segments=g("num_changed_segments"),
        mean_abs_change_scaled=g("mean_abs_change_scaled"),
        max_abs_change_scaled=g("max_abs_change_scaled"),
        last_obs_scaled=g("last_obs_scaled"),
        last_obs_units=last_obs_units,
        base_pred_scaled=g("base_pred_scaled"),
        base_pred_units=base_pred_units,
        query_ts_units=query_ts_units,
        cf_ts_units=cf_ts_units,
        delta_mv_units=g("delta_mv_units"),
        details=g("details"),

        proximity_l1_scaled=prox_l1,
        proximity_l2_scaled=prox_l2,
        sparsity_count=spar_count,
        sparsity_ratio=spar_ratio,
    )


def _infer_output_std(r: Record) -> float | None:
    """Infer scaler std for the output channel from two (scaled, units) pairs.

    Uses last_obs and base_pred, which are present in JSONL versions.
    """
    if (
        r.last_obs_scaled is None
        or r.last_obs_units is None
        or r.base_pred_scaled is None
        or r.base_pred_units is None
    ):
        return None
    denom = float(r.base_pred_scaled) - float(r.last_obs_scaled)
    if abs(denom) < 1e-9:
        return None
    return (float(r.base_pred_units) - float(r.last_obs_units)) / denom


def _change_vs_input_units(
    r: Record,
) -> tuple[float | None, float | None, list[tuple[int, float, float, float]] | None]:
    """Return (mean_abs, max_abs, top_changes) in output units.

    Priority:
    1) If multivariate deltas exist (delta_mv_units): compute mean/max across all channels+timesteps.
    2) Else if we have full 1D series (query_ts_units + cf_ts_units): compute exact.
    3) Else: derive from scaled metrics using inferred output std.
    """

    if r.delta_mv_units is not None:
        try:
            flat = [abs(float(v)) for row in r.delta_mv_units for v in row]
            if not flat:
                raise ValueError("empty delta_mv_units")
            mean_abs = sum(flat) / len(flat)
            max_abs = max(flat)
            return mean_abs, max_abs, None
        except Exception:
            pass

    if r.query_ts_units is not None and r.cf_ts_units is not None:
        try:
            q = [float(x) for x in r.query_ts_units]
            c = [float(x) for x in r.cf_ts_units]
            if len(q) != len(c) or len(q) == 0:
                raise ValueError("series length mismatch")
            diffs = [abs(ci - qi) for qi, ci in zip(q, c)]
            mean_abs = sum(diffs) / len(diffs)
            max_abs = max(diffs)
            # Top-3 indices by absolute change
            topk = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:3]
            top_changes = [(i, q[i], c[i], c[i] - q[i]) for i in topk]
            return mean_abs, max_abs, top_changes
        except Exception:
            pass

    std = _infer_output_std(r)
    if std is None:
        return None, None, None

    mean_abs = None
    max_abs = None
    if r.mean_abs_change_scaled is not None:
        mean_abs = abs(float(r.mean_abs_change_scaled) * std)
    if r.max_abs_change_scaled is not None:
        max_abs = abs(float(r.max_abs_change_scaled) * std)
    return mean_abs, max_abs, None


def _fmt_example(r: Record) -> str:
    d = r.details or {}
    changed = d.get("changed_segments", [])

    lines: list[str] = []
    if r.base_delta is not None and r.cf_delta is not None:
        lines.append(
            f"- win_idx={r.win_idx} | Δy base={float(r.base_delta):.3f}{r.unit} -> cf={float(r.cf_delta):.3f}{r.unit}"
        )
    else:
        lines.append(f"- win_idx={r.win_idx} | success={r.success}")
    if r.num_changed_segments is not None:
        lines.append(f"  - changed_segments={r.num_changed_segments}, mean|Δ|_scaled={r.mean_abs_change_scaled:.4f}, max|Δ|_scaled={r.max_abs_change_scaled:.4f}")

    if r.proximity_l1_scaled is not None or r.sparsity_ratio is not None:
        parts = []
        if r.proximity_l1_scaled is not None:
            parts.append(f"proximity_l1_scaled={float(r.proximity_l1_scaled):.5f}")
        if r.proximity_l2_scaled is not None:
            parts.append(f"proximity_l2_scaled={float(r.proximity_l2_scaled):.5f}")
        if r.sparsity_count is not None:
            parts.append(f"sparsity_count={int(r.sparsity_count)}")
        if r.sparsity_ratio is not None:
            parts.append(f"sparsity_ratio={float(r.sparsity_ratio):.3%}")
        lines.append("  - paper metrics: " + ", ".join(parts))

    mean_abs_u, max_abs_u, top_changes = _change_vs_input_units(r)
    if mean_abs_u is not None or max_abs_u is not None:
        parts = []
        if mean_abs_u is not None:
            parts.append(f"mean|Δ|≈{mean_abs_u:.3f}{r.unit}")
        if max_abs_u is not None:
            parts.append(f"max|Δ|≈{max_abs_u:.3f}{r.unit}")
        lines.append("  - change vs input: " + ", ".join(parts))
    if top_changes:
        # Only available when we have full ambient series in JSONL
        desc = "; ".join([f"t={i}: {q:.2f}→{c:.2f} (Δ={d:+.2f})" for i, q, c, d in top_changes])
        lines.append(f"  - biggest point changes: {desc}")

    # Intentionally omit SAX symbols/grams here: report should stay in ambient space.
    for ch in changed:
        s = ch.get("start")
        e = ch.get("end")
        m = ch.get("original_segment_mean_degC")
        v = ch.get("counterfactual_value_degC")
        if m is None or v is None:
            # Fallback for older jsonl
            m = ch.get("original_segment_mean")
            v = ch.get("counterfactual_value")
        # If values are still in scaled space, label without °C
        if m is None or v is None:
            continue
        try:
            delta = float(v) - float(m)
            delta_s = f" (Δ≈{delta:+.3f}{r.unit})"
        except Exception:
            delta_s = ""
        lines.append(f"  - seg#{ch.get('segment_index')} [{s}:{e}] mean {m:.3f}{r.unit} -> set {v:.3f}{r.unit}{delta_s}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize weather CF JSONL into quantitative + qualitative report")
    ap.add_argument("jsonl", type=str)
    ap.add_argument("--topk", type=int, default=8, help="How many examples to print")
    ap.add_argument("--out", type=str, default="", help="Optional markdown output")
    args = ap.parse_args()

    path = Path(args.jsonl)
    rows = _load_jsonl(path)
    recs = [_as_record(r) for r in rows]

    attempted = len(recs)
    successes = [r for r in recs if r.success]
    failures = [r for r in recs if not r.success]

    # Ranking: most explainable = fewest changed segments, then smallest mean abs change,
    # then smallest max change, then smallest |cf_delta - threshold| (not always present)
    def key_explainable(r: Record):
        return (
            r.num_changed_segments if r.num_changed_segments is not None else 10**9,
            r.mean_abs_change_scaled if r.mean_abs_change_scaled is not None else 10**9,
            r.max_abs_change_scaled if r.max_abs_change_scaled is not None else 10**9,
        )

    explainable = sorted(successes, key=key_explainable)[: max(0, args.topk)]

    # Also show "strong" effects: largest delta
    strong = sorted(
        [r for r in successes if r.cf_delta is not None],
        key=lambda r: float(r.cf_delta),
        reverse=True,
    )[: max(0, min(args.topk, 5))]

    lines: list[str] = []
    lines.append(f"# Counterfactual batch summary")
    lines.append(f"- file: {path}")
    lines.append(f"- attempted: {attempted}")
    lines.append(f"- successes: {len(successes)}")
    lines.append(f"- success rate: {len(successes)/attempted:.1%}" if attempted else "- success rate: n/a")

    if successes:
        changed = [r.num_changed_segments for r in successes if r.num_changed_segments is not None]
        if changed:
            lines.append(f"- changed_segments (successes): min={min(changed)}, median={sorted(changed)[len(changed)//2]}, max={max(changed)}")

        units = sorted({r.unit for r in successes})
        unit_s = units[0] if len(units) == 1 else "mixed"
        deltas = [float(r.cf_delta) for r in successes if r.cf_delta is not None]
        if deltas:
            lines.append(
                f"- cf Δy range: min={min(deltas):.3f}{unit_s}, median={sorted(deltas)[len(deltas)//2]:.3f}{unit_s}, max={max(deltas):.3f}{unit_s}"
            )

        mean_abs_list = []
        max_abs_list = []
        for r in successes:
            m, mx, _ = _change_vs_input_units(r)
            if m is not None:
                mean_abs_list.append(m)
            if mx is not None:
                max_abs_list.append(mx)
        prox_l1 = [float(r.proximity_l1_scaled) for r in successes if r.proximity_l1_scaled is not None]
        prox_l2 = [float(r.proximity_l2_scaled) for r in successes if r.proximity_l2_scaled is not None]
        spar = [float(r.sparsity_ratio) for r in successes if r.sparsity_ratio is not None]

        if mean_abs_list:
            srt = sorted(mean_abs_list)
            lines.append(f"- mean|Δ| vs input ({unit_s}): min={srt[0]:.3f}, median={srt[len(srt)//2]:.3f}, max={srt[-1]:.3f}")
        if max_abs_list:
            srt = sorted(max_abs_list)
            lines.append(f"- max|Δ| vs input ({unit_s}): min={srt[0]:.3f}, median={srt[len(srt)//2]:.3f}, max={srt[-1]:.3f}")

        if prox_l1:
            srt = sorted(prox_l1)
            lines.append(f"- proximity_l1_scaled (successes): min={srt[0]:.5f}, median={srt[len(srt)//2]:.5f}, max={srt[-1]:.5f}")
        if prox_l2:
            srt = sorted(prox_l2)
            lines.append(f"- proximity_l2_scaled (successes): min={srt[0]:.5f}, median={srt[len(srt)//2]:.5f}, max={srt[-1]:.5f}")
        if spar:
            srt = sorted(spar)
            lines.append(f"- sparsity_ratio (successes): min={srt[0]:.3%}, median={srt[len(srt)//2]:.3%}, max={srt[-1]:.3%}")

    lines.append("\n## Most explainable examples")
    for r in explainable:
        lines.append(_fmt_example(r))
    lines.append("\n## Strongest ΔT examples")
    for r in strong:
        lines.append(_fmt_example(r))

    if failures:
        lines.append("\n## Failures (window indices)")
        lines.append("- " + ", ".join(str(r.win_idx) for r in failures[:50]) + (" ..." if len(failures) > 50 else ""))

    report = "\n".join(lines) + "\n"

    print(report)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
