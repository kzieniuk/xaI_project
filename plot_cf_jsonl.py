import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend (batch/CI friendly)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _infer_figsize_from_template(template_png: Path, dpi: int) -> tuple[float, float]:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Template support requires Pillow. Install it with: pip install pillow"
        ) from e

    img = Image.open(template_png)
    w_px, h_px = img.size
    return (float(w_px) / float(dpi), float(h_px) / float(dpi))


@dataclass
class Row:
    win_idx: int
    success: bool

    target_col: str | None
    horizon_step: int | None

    # Target (1D) time series in units (e.g. °C)
    query_ts: np.ndarray | None
    cf_ts: np.ndarray | None

    # Multivariate selected columns
    cf_cols: list[str] | None
    query_mv: np.ndarray | None  # (d, T)
    cf_mv: np.ndarray | None  # (d, T)

    # Forecast/true values in units
    last_obs: float | None
    base_pred: float | None
    cf_pred: float | None
    true_next: float | None

    base_delta: float | None
    cf_delta: float | None
    true_delta: float | None

    details: dict[str, Any] | None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _to_1d(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 1 and a.size:
            return a
    except Exception:
        return None
    return None


def _to_mv(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 2 and a.shape[0] > 0 and a.shape[1] > 0:
            return a
    except Exception:
        return None
    return None


def _unit_from_row(r: dict[str, Any]) -> str:
    if "base_delta_degC" in r or "cf_delta_degC" in r or "true_next_degC" in r:
        return "°C"
    return "units"


def _parse_row(r: dict[str, Any]) -> Row:
    # Target series
    query_ts = _to_1d(r.get("query_ts_degC") or r.get("query_ts_units"))
    cf_ts = _to_1d(r.get("cf_ts_degC") or r.get("cf_ts_units"))

    # Multivariate series
    cf_cols = r.get("cf_cols") if isinstance(r.get("cf_cols"), list) else None
    query_mv = _to_mv(r.get("query_mv_units"))
    cf_mv = _to_mv(r.get("cf_mv_units"))

    return Row(
        win_idx=int(r.get("win_idx", -1)),
        success=bool(r.get("success", False)),
        target_col=r.get("target_col"),
        horizon_step=_to_int(r.get("horizon_step")),
        query_ts=query_ts,
        cf_ts=cf_ts,
        cf_cols=cf_cols,
        query_mv=query_mv,
        cf_mv=cf_mv,
        last_obs=_to_float(r.get("last_obs_degC") or r.get("last_obs_units")),
        base_pred=_to_float(r.get("base_pred_degC") or r.get("base_pred_units")),
        cf_pred=_to_float(r.get("cf_pred_degC") or r.get("cf_pred_units")),
        true_next=_to_float(r.get("true_next_degC") or r.get("true_next_units")),
        base_delta=_to_float(r.get("base_delta_degC") or r.get("base_delta_units")),
        cf_delta=_to_float(r.get("cf_delta_degC") or r.get("cf_delta_units")),
        true_delta=_to_float(r.get("true_delta_degC") or r.get("true_delta_units")),
        details=r.get("details"),
    )


def _extract_changed_spans(details: dict[str, Any] | None, T: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not isinstance(details, dict):
        return spans
    changed = details.get("changed_segments")
    if not isinstance(changed, list):
        return spans
    for seg in changed:
        if not isinstance(seg, dict):
            continue
        s = _to_int(seg.get("start"))
        e = _to_int(seg.get("end"))
        if s is None or e is None:
            continue
        s = max(0, min(s, T))
        e = max(0, min(e, T))
        if e > s:
            spans.append((s, e))
    return spans


def _plot_panel(ax, t: np.ndarray, q: np.ndarray, c: np.ndarray | None, label: str, spans: list[tuple[int, int]]):
    ax.plot(t, q, label="query", linewidth=1.5)
    if c is not None:
        ax.plot(t, c, label="cf", linewidth=1.5)
    for s, e in spans:
        ax.axvspan(s, e, alpha=0.12)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)


def plot_row(r: Row, out_png: Path, *, fig_width: float = 6.8, dpi: int = 160) -> None:
    unit = "°C" if (r.base_delta is not None or r.true_next is not None) else "units"

    # Build panels: always include target series if present
    panels: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    if r.query_ts is not None:
        panels.append(((r.target_col or "target") + f" [{unit}]", r.query_ts, r.cf_ts))

    # Add multivariate panels for cf_cols (excluding target col to avoid duplication)
    if r.cf_cols is not None and r.query_mv is not None:
        for i, col in enumerate(r.cf_cols):
            if r.target_col is not None and col == r.target_col and r.query_ts is not None:
                continue
            q = r.query_mv[i]
            c = r.cf_mv[i] if (r.cf_mv is not None and i < r.cf_mv.shape[0]) else None
            panels.append((col, q, c))

    if not panels:
        raise ValueError("No series found to plot")

    T = panels[0][1].shape[0]
    t = np.arange(T)
    spans = _extract_changed_spans(r.details, T)

    fig_h = 2.1 * len(panels) + 1.0
    fig, axes = plt.subplots(len(panels), 1, figsize=(float(fig_width), fig_h), dpi=int(dpi), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, q, c) in zip(axes, panels):
        _plot_panel(ax, t, q, c, label=label, spans=spans)

    # Add forecast markers to the first (target) panel if we have the necessary numbers.
    # Place markers at x = T + horizon_step (0 -> next step) so it’s visually separate from history.
    hx = None
    if r.horizon_step is not None:
        hx = int(T + r.horizon_step)
    elif r.query_ts is not None:
        hx = int(T)

    if hx is not None and r.query_ts is not None:
        ax0 = axes[0]
        if r.true_next is not None:
            ax0.scatter([hx], [r.true_next], marker="o", s=40, label="true")
        if r.base_pred is not None:
            ax0.scatter([hx], [r.base_pred], marker="x", s=50, label="base_pred")
        if r.cf_pred is not None:
            ax0.scatter([hx], [r.cf_pred], marker="^", s=45, label="cf_pred")
        ax0.axvline(x=hx, color="k", alpha=0.15, linewidth=1.0)

    title_parts = [f"win_idx={r.win_idx}"]
    if r.base_delta is not None and r.cf_delta is not None:
        title_parts.append(f"Δy base={r.base_delta:.3f}{unit} → cf={r.cf_delta:.3f}{unit}")
    if r.true_delta is not None:
        title_parts.append(f"true Δy={r.true_delta:.3f}{unit}")

    fig.suptitle(" | ".join(title_parts), fontsize=10)
    axes[-1].set_xlabel("t (history index)")

    # Put a single legend on the first panel
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-window CF plots (query vs cf) + true/pred markers")
    ap.add_argument("jsonl", type=str)
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory for PNGs (default: <jsonl_stem>_plots next to jsonl)",
    )
    ap.add_argument(
        "--index",
        type=str,
        default="",
        help="Optional markdown index output (default: <outdir>/index.md)",
    )
    ap.add_argument("--max", type=int, default=0, help="If >0, limit number of plotted successes")
    ap.add_argument(
        "--template",
        type=str,
        default="",
        help="Optional template PNG path; if set, match its output pixel size/aspect as closely as possible",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI (default: 160). If --template is set, DPI is used to match the template pixel size.",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    rows = _load_jsonl(jsonl_path)
    parsed = [_parse_row(r) for r in rows]

    successes = [r for r in parsed if r.success]

    fig_width = 6.8
    dpi = int(args.dpi)
    if args.template:
        template_path = Path(args.template)
        if not template_path.is_file():
            raise SystemExit(f"template not found: {template_path}")
        fig_width, _template_h = _infer_figsize_from_template(template_path, dpi=dpi)

    if args.max and args.max > 0:
        successes = successes[: int(args.max)]

    if not args.outdir:
        outdir = (jsonl_path.parent / f"{jsonl_path.stem}_plots").resolve()
    else:
        outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pngs: list[Path] = []
    for r in successes:
        out_png = outdir / f"win_{r.win_idx:06d}.png"
        plot_row(r, out_png, fig_width=fig_width, dpi=dpi)
        pngs.append(out_png)

    if not args.index:
        index_path = outdir / "index.md"
    else:
        index_path = Path(args.index).resolve()

    base_dir = index_path.parent
    lines: list[str] = []
    lines.append("# Counterfactual plots")
    lines.append(f"- jsonl: {jsonl_path}")
    lines.append(f"- plotted successes: {len(pngs)}")
    lines.append("")

    for p in pngs:
        rel = p.relative_to(base_dir).as_posix() if p.is_relative_to(base_dir) else p.as_posix()
        lines.append(f"## {p.stem}")
        lines.append(f"![]({rel})")
        lines.append("")

    index_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved {len(pngs)} pngs to {outdir}")
    print(f"index_md={index_path}")


if __name__ == "__main__":
    main()
