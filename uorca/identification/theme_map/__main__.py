"""Terminal entry point for the theme map (spec section 8).

::

    uv run python -m uorca.identification.theme_map --run_dir <run> [--dry-run]

``--dry-run`` prints the dataset count, the token count and the estimated cost, then
exits without making a single API call.

This entry point **raises**. Only the automatic call site inside
``dataset_identification.py`` catches, and it does so for the one reason spec decision 4
gives: a theme-map failure must not cost a completed 60-minute identification run. Here
there is no such run to protect, so a failure is a non-zero exit and a traceback.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import build_theme_map, estimate_theme_map_build


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m uorca.identification.theme_map",
        description=(
            "Build the theme map for an identification run directory. Embeds every "
            "valid dataset, clusters it, names the clusters with one LLM call and "
            "writes four artifacts to <run_dir>/theme_map/."
        ),
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        type=Path,
        help="Identification run directory holding Dataset_identification_result.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dataset count, token count and estimated cost, then exit "
        "without calling any API.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log every pipeline step at INFO level.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a build or a dry run. Returns 0; every failure raises."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # The API key normally arrives through .env, exactly as `uorca identify` loads it.
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(), override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        pass

    run_dir: Path = args.run_dir

    if args.dry_run:
        estimate = estimate_theme_map_build(run_dir)
        print(f"Theme map dry run for {run_dir}")
        print(f"  datasets to embed : {estimate.n_datasets}")
        print(f"  tokens            : {estimate.n_tokens}")
        print(f"  estimated cost    : ${estimate.cost_usd:.4f}")
        print("  (embedding only; the one theme-naming call is not priced here)")
        print("No API call was made.")
        return 0

    result = build_theme_map(run_dir)
    provenance = result.provenance
    print(f"Theme map written to {run_dir / 'theme_map'}")
    print(f"  datasets : {provenance.n_datasets}")
    print(f"  themes   : {provenance.n_themes}")
    print(f"  unthemed : {provenance.noise_fraction:.1%}")
    print(f"  cost     : ${provenance.estimated_cost_usd:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
