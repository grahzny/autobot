"""CLI entry point for Autobot world simulation."""

from __future__ import annotations

import argparse
import sys

from autobot.scenarios import ALL_SCENARIOS
from autobot.simulation import SimulationConfig, run_simulation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="autobot",
        description="Autobot — AI World Model: Logic & Language Environment",
    )
    parser.add_argument(
        "scenario",
        choices=list(ALL_SCENARIOS.keys()) + ["all"],
        help="Which scenario to run (or 'all' for every scenario).",
    )
    parser.add_argument(
        "-c", "--cycles",
        type=int,
        default=20,
        help="Max cycles to simulate (default: 20).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-cycle output.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Write JSON log to this file.",
    )

    args = parser.parse_args(argv)
    config = SimulationConfig(
        max_cycles=args.cycles,
        verbose=not args.quiet,
        log_file=args.output,
    )

    if args.scenario == "all":
        for name, builder in ALL_SCENARIOS.items():
            print(f"\n{'#' * 70}")
            print(f"#  SCENARIO: {name}")
            print(f"{'#' * 70}\n")
            engine = builder()
            run_simulation(engine, config)
            print()
    else:
        engine = ALL_SCENARIOS[args.scenario]()
        run_simulation(engine, config)


if __name__ == "__main__":
    main()
