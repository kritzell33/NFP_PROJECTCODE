"""NFP Assessor command-line interface.

    nfp run   --crew ... --scenario ... --archetypes ... [--runs N] [--seed S]
    nfp demo                    # run the bundled sample end-to-end
    nfp gui                     # launch the PySide6 desktop shell

Run these from the repository root so the default data paths resolve
(the .bat scripts in scripts/ handle that for you).
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

DEFAULT_CREW = Path("data/crew/sample_crew.json")
DEFAULT_SCENARIO = Path("scenarios/shakedown_14d.json")
DEFAULT_ARCHETYPES = Path("data/archetypes/archetypes_sample.json")
DEFAULT_OUT = Path("reports")


def _run(crew: Path, scenario: Path, archetypes: Path, runs: int, seed: int,
         out: Path, open_report: bool) -> Path:
    # Imports are deferred so `nfp --help` stays snappy.
    from .report.builder import build_report
    from .sim.montecarlo import run_batch, save_batch

    for p, label in [(crew, "crew"), (scenario, "scenario"),
                     (archetypes, "archetypes")]:
        if not Path(p).exists():
            sys.exit(
                f"error: {label} file not found: {p}\n"
                "hint: run from the repository root, or pass an explicit path."
            )

    print(f"Running {runs} replications of '{scenario}' (seed {seed})...")
    result = run_batch(crew, scenario, archetypes, n_runs=runs, seed=seed)
    run_dir = save_batch(result, out)
    report = build_report(result, run_dir)

    s = result.summary()
    print(f"  cascade-free missions : {100 * s['p_cascade_free']:.1f}%")
    print(f"  mean task completion  : {100 * s['completion_rate']['mean']:.1f}%")
    print(f"  mean final cohesion   : {s['final_cohesion']['mean']:.2f}")
    print(f"  report               -> {report}")

    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nfp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a Monte Carlo mission batch")
    p_run.add_argument("--crew", type=Path, default=DEFAULT_CREW)
    p_run.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    p_run.add_argument("--archetypes", type=Path, default=DEFAULT_ARCHETYPES)
    p_run.add_argument("--runs", type=int, default=200)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_run.add_argument("--open", action="store_true", dest="open_report",
                       help="open the HTML report when finished")

    sub.add_parser("demo", help="run the bundled sample scenario (100 runs)")
    sub.add_parser("gui", help="launch the desktop GUI (requires PySide6)")

    args = parser.parse_args(argv)

    if args.command == "run":
        _run(args.crew, args.scenario, args.archetypes, args.runs, args.seed,
             args.out, args.open_report)
    elif args.command == "demo":
        _run(DEFAULT_CREW, DEFAULT_SCENARIO, DEFAULT_ARCHETYPES,
             runs=100, seed=42, out=DEFAULT_OUT, open_report=False)
    elif args.command == "gui":
        try:
            from .gui.app import launch
        except ImportError:
            sys.exit(
                "PySide6 is not installed in this environment.\n"
                'Install the GUI extras first:  pip install -e ".[gui]"'
            )
        launch()


if __name__ == "__main__":
    main()
