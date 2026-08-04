"""Comprehensive HTML report for a Monte Carlo batch.

Produces a single self-contained ``report.html`` (figures embedded as base64
PNG) so it can be emailed, archived, or opened offline. PDF export via fpdf2
is planned for phase 2 and will reuse the same figure functions.
"""

from __future__ import annotations

import base64
import io
from importlib import resources
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works in CLI, GUI thread, and frozen exe
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FunctionLoader, select_autoescape

from ..sim.montecarlo import BatchResult

# NFP palette
INK = "#0B1D3A"       # deep space navy
ACCENT = "#D9A441"    # mission gold
GOOD = "#3E8E5A"
BAD = "#B4453A"
GRID = "#C9CFD9"


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def fig_completion_hist(result: BatchResult) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(result.per_run["completion_rate"], bins=20, color=INK, alpha=0.85)
    ax.set_xlabel("Task completion rate")
    ax.set_ylabel("Replications")
    ax.set_title("Distribution of mission task completion")
    _style_axes(ax)
    return _fig_to_base64(fig)


def fig_cohesion_band(result: BatchResult) -> str:
    """Mean cohesion trajectory with a 10-90% band across replications."""
    n_days = min(len(ts["day"]) for ts in result.timeseries)
    days = result.timeseries[0]["day"][:n_days]
    mat = np.array([ts["cohesion"][:n_days] for ts in result.timeseries])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(days, np.percentile(mat, 10, axis=0),
                    np.percentile(mat, 90, axis=0),
                    color=ACCENT, alpha=0.30, label="10-90% band")
    ax.plot(days, mat.mean(axis=0), color=INK, linewidth=2, label="mean")
    ax.axhline(0.0, color=GRID, linewidth=1)
    ax.set_xlabel("Mission day")
    ax.set_ylabel("Crew cohesion (-1 to 1)")
    ax.set_title("Crew cohesion trajectory across replications")
    ax.legend(frameon=False)
    _style_axes(ax)
    return _fig_to_base64(fig)


def fig_stress_by_agent(result: BatchResult) -> str:
    """Mean nightly stress per crew member, averaged across replications."""
    n_days = min(len(ts["day"]) for ts in result.timeseries)
    days = result.timeseries[0]["day"][:n_days]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(result.crew)))
    for color, cand in zip(cmap, result.crew):
        mat = np.array([ts["stress"][cand.id][:n_days] for ts in result.timeseries])
        ax.plot(days, mat.mean(axis=0), linewidth=2, color=color, label=cand.name)
    ax.axhline(55, color=BAD, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(days[-1], 56.5, "stressed threshold", color=BAD,
            fontsize=8, ha="right")
    ax.set_xlabel("Mission day")
    ax.set_ylabel("Stress (0-100)")
    ax.set_title("Mean stress trajectory by crew member")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    return _fig_to_base64(fig)


def fig_conflicts_hist(result: BatchResult) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    data = result.per_run["conflict_events"]
    bins = np.arange(-0.5, data.max() + 1.5, 1)
    ax.hist(data, bins=bins, color=BAD, alpha=0.8)
    ax.set_xlabel("Interpersonal friction incidents per mission")
    ax.set_ylabel("Replications")
    ax.set_title("Distribution of conflict events")
    _style_axes(ax)
    return _fig_to_base64(fig)


def _load_template():
    def _loader(name: str):
        return (
            resources.files("nfp.report").joinpath("templates").joinpath(name)
            .read_text(encoding="utf-8")
        )

    env = Environment(loader=FunctionLoader(_loader),
                      autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda x: f"{100.0 * x:.1f}%"
    env.filters["f2"] = lambda x: f"{x:.2f}"
    return env.get_template("report.html.j2")


def build_report(result: BatchResult, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    summary = result.summary()
    context = {
        "provenance": result.provenance(),
        "scenario": result.scenario,
        "crew": result.crew,
        "archetypes": result.archetypes,
        "summary": summary,
        "p_cascade_free": summary["p_cascade_free"],
        "archetype_probs": result.archetype_probs(),
        "figures": {
            "completion": fig_completion_hist(result),
            "cohesion": fig_cohesion_band(result),
            "stress": fig_stress_by_agent(result),
            "conflicts": fig_conflicts_hist(result),
        },
    }
    html = _load_template().render(**context)
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
