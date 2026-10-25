"""Command-line entry points. Run any of them with ``python -m av_contradiction.cli.<name>``."""

from __future__ import annotations

import argparse


def add_config_args(parser: argparse.ArgumentParser, default: str | None = None) -> None:
    parser.add_argument("--config", default=default, help="Optional YAML file overriding the built-in Kaggle GPU defaults")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set model.hidden_dim=128 (repeatable)",
    )
