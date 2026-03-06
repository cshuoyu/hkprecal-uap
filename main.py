#!/usr/bin/env python3
"""Unified CLI entrypoint for UAP pipelines."""

import logging
import sys
from argparse import Namespace

import hydra
from omegaconf import OmegaConf

import uap.engine.csv2plot as engine_csv2plot
import uap.engine.root2csv as engine_root2csv


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )


def _run_root2csv_legacy(argv):
    _setup_logging()
    args = engine_root2csv.parse_args(argv)
    engine_root2csv.run(args)


def _extract_config_name(argv, default_name):
    config_name = str(default_name)
    overrides = []
    i = 0
    while i < len(argv):
        token = str(argv[i])
        if token == "--config-name":
            if i + 1 >= len(argv):
                raise SystemExit("--config-name requires a value")
            config_name = str(argv[i + 1])
            i += 2
            continue
        if token.startswith("--config-name="):
            config_name = token.split("=", 1)[1].strip()
            i += 1
            continue
        overrides.append(token)
        i += 1
    return config_name, overrides


def _run_hydra_with_config(config_name, overrides, runner_func):
    orig_argv = sys.argv[:]
    try:
        sys.argv = [orig_argv[0]] + list(overrides)

        @hydra.main(config_path="config", config_name=config_name, version_base=None)
        def _run(cfg):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            args = Namespace(**cfg_dict)
            runner_func(args)

        _run()
    finally:
        sys.argv = orig_argv


def _run_root2csv_hydra(argv):
    config_name, overrides = _extract_config_name(argv, "aus_root2csv_default")
    _run_hydra_with_config(config_name, overrides, engine_root2csv.run)


def _run_csv2plot_legacy(argv):
    _setup_logging()
    args = engine_csv2plot.parse_args(argv)
    engine_csv2plot.run(args)


def _run_csv2plot_hydra(argv):
    config_name, overrides = _extract_config_name(argv, "csv2plot_default")
    _run_hydra_with_config(config_name, overrides, engine_csv2plot.run)


def _looks_like_csv2plot_legacy(argv):
    # Legacy mode is explicit argparse-style flags.
    flags = {"--out-dir", "--out-base", "--out-name", "--points-csv", "--series-json"}
    for token in argv:
        if token in flags:
            return True
    return False


def _print_help():
    print("Usage:")
    print("  python3 main.py root2csv [aus|kor ...]")
    print("  python3 main.py csv2plot [options]")
    print("")
    print("Shortcuts:")
    print("  python3 main.py aus ...    # same as: root2csv aus ...")
    print("  python3 main.py kor ...    # same as: root2csv kor ...")
    print("")
    print("Hydra mode:")
    print("  python3 main.py root2csv   # use config/aus_root2csv_default.yaml by default")
    print("  python3 main.py csv2plot   # use config/csv2plot_default.yaml by default")
    print("  Add --config-name <yaml_without_ext> to switch profile")


def main():
    argv = sys.argv[1:]
    if not argv:
        _print_help()
        raise SystemExit(2)

    cmd = argv[0].strip().lower()
    rest = argv[1:]

    # Keep old short commands available.
    if cmd in ("aus", "kor"):
        _run_root2csv_legacy([cmd] + rest)
        return

    if cmd == "root2csv":
        # root2csv legacy CLI mode: explicit aus/kor subcommand
        if rest and rest[0].strip().lower() in ("aus", "kor"):
            _run_root2csv_legacy(rest)
            return
        # root2csv hydra mode: config-driven
        _run_root2csv_hydra(rest)
        return

    if cmd == "csv2plot":
        if _looks_like_csv2plot_legacy(rest):
            _run_csv2plot_legacy(rest)
            return
        _run_csv2plot_hydra(rest)
        return

    _print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
