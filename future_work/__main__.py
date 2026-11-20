"""Run and verify the follow-up studies without writing to presentation artifacts."""

import argparse

from .freeze import verify as verify_baseline
from .storage import RESULT


def render():
    from . import figures, reporting

    if (RESULT / "season_faces.csv").exists():
        figures.season_figures()
    if (RESULT / "heat_effects.csv").exists():
        figures.heat_figure()
    if (RESULT / "figure_03b_vegetation.csv").exists():
        figures.vegetation_figure()
    if (RESULT / "history_severity.csv").exists():
        figures.history_figure()
    from .district import render_if_ready

    render_if_ready()
    reporting.report()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["all", "season", "heat", "history", "india", "report", "verify"])
    p.add_argument("--workers", default=12, type=int, help="Concurrent ERA5 download workers")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    verify_baseline()
    if args.stage == "verify":
        from .provenance import OUTPUT_PINS, verify_files
        from .validation import validate_results

        count = verify_files(OUTPUT_PINS)
        validate_results()
        print(f"Verified {count} follow-up artifacts.")
        return 0
    district_available = True
    if args.stage in ("season", "all"):
        from .seasons import main as seasons

        seasons()
    if args.stage in ("heat", "all"):
        from . import era5, heat, vegetation, vegetation_grid

        if not (RESULT / "season_crop_year.parquet").exists():
            from .seasons import main as seasons

            seasons()
        era5.grid_check()
        era5.download(workers=args.workers)
        vegetation.weekly()
        heat.main()
        vegetation_grid.run()
        vegetation_grid.comparison()
        from .measurement_checks import run as audit_sources

        audit_sources()
        from .provenance import write_sources

        write_sources()
    if args.stage in ("history", "all"):
        from .history import analyze

        analyze()
    if args.stage in ("india", "all"):
        from .district import run

        district_available = run()
    if args.stage == "india":
        from . import reporting
        from .district import render_if_ready

        render_if_ready()
        reporting.report()
    else:
        render()
    from .provenance import write_outputs

    write_outputs()
    verify_baseline()
    return 0 if district_available else 2


if __name__ == "__main__":
    raise SystemExit(main())
