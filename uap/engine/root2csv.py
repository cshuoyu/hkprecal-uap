#!/usr/bin/env python3
"""ROOT -> CSV pipeline for AUS/KOR scan analyses."""

import argparse
import logging
from pathlib import Path

from uap.fit.emg_timing_offset_fit import TimingEMGFitter


logger = logging.getLogger(__name__)


def build_fitter(args, out_csv_path):
    fig_dir_arg = getattr(args, "fig_dir", "")
    fig_dir = (
        Path(fig_dir_arg).resolve()
        if fig_dir_arg
        else Path(out_csv_path).resolve().parent / "figures"
    )
    return TimingEMGFitter(
        method_name=args.fit_method,
        fig_dir=fig_dir,
    )


def build_parser():
    ap = argparse.ArgumentParser(description="ROOT -> CSV pipeline for AUS/KOR.")
    sub = ap.add_subparsers(dest="system")

    ap_aus = sub.add_parser("aus", help="AUS timing fit from output_theta*_phi*.root")
    ap_aus.add_argument("--input-dir", required=True)
    ap_aus.add_argument("--out-csv", default="", help="Output fit CSV path")
    ap_aus.add_argument("--out-base", default="", help="Run output root (default: <UAP_HOME>/outputs)")
    ap_aus.add_argument("--serial", default="")
    ap_aus.add_argument("--fit-method", default="fitandplot_emg", help="Timing fitter method")
    ap_aus.add_argument("--fig-dir", default="", help="Directory to store diagnostic fit figures")
    ap_aus.add_argument("--pmt-ch", default="auto")
    ap_aus.add_argument("--trigger-ch", default="auto")
    ap_aus.add_argument("--laser-ch", default=None, help="Deprecated alias of --trigger-ch")
    ap_aus.add_argument("--sipm-ch", default="auto")
    ap_aus.add_argument("--no-sipm", action="store_true")
    ap_aus.add_argument("--tmin", type=float, default=304.0)
    ap_aus.add_argument("--tmax", type=float, default=314.0)
    ap_aus.add_argument("--window-half-width", type=float, default=5.0)
    ap_aus.add_argument("--window-bin-width", type=float, default=0.0)
    ap_aus.add_argument("--sipm-tmin", type=float, default=95.0)
    ap_aus.add_argument("--sipm-tmax", type=float, default=105.0)
    ap_aus.add_argument("--sipm-window-half-width", type=float, default=5.0)
    ap_aus.add_argument("--sipm-window-bin-width", type=float, default=0.0)
    ap_aus.add_argument("--inc-bkg", dest="inc_bkg", action="store_true")
    ap_aus.add_argument("--no-inc-bkg", dest="inc_bkg", action="store_false")
    ap_aus.set_defaults(inc_bkg=True)
    ap_aus.add_argument("--max-files", type=int, default=0)

    ap_kor = sub.add_parser("kor", help="KOR diff fit from prd_*.root")
    ap_kor.add_argument("--input-dir", required=True)
    ap_kor.add_argument("--out-csv", default="", help="Output fit CSV path")
    ap_kor.add_argument("--out-base", default="", help="Run output root (default: <UAP_HOME>/outputs)")
    ap_kor.add_argument("--serial", default="AA0000A")
    ap_kor.add_argument("--channel", default="auto")
    ap_kor.add_argument("--trigger-ch", default="auto")
    ap_kor.add_argument("--fit-method", default="fitandplot_emg", help="Timing fitter method")
    ap_kor.add_argument("--fig-dir", default="", help="Directory to store diagnostic fit figures")
    ap_kor.add_argument("--inc-bkg", dest="inc_bkg", action="store_true")
    ap_kor.add_argument("--no-inc-bkg", dest="inc_bkg", action="store_false")
    ap_kor.set_defaults(inc_bkg=True)
    ap_kor.add_argument("--window-half-width", type=float, default=30.0)
    ap_kor.add_argument("--window-bin-width", type=float, default=2.0)
    ap_kor.add_argument("--tmin", type=float, default=None)
    ap_kor.add_argument("--tmax", type=float, default=None)
    ap_kor.add_argument("--max-files", type=int, default=0)
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.system:
        ap.print_help()
        raise SystemExit(2)
    return args


def _prepare_output_layout(args):
    out_base_arg = getattr(args, "out_base", "")
    out_csv_arg = getattr(args, "out_csv", "")
    fig_dir_arg = getattr(args, "fig_dir", "")

    out_base = Path(out_base_arg).resolve() if out_base_arg else None
    if not out_csv_arg:
        if out_base:
            out_csv_arg = str(out_base / "csv" / "{}_results.csv".format(args.system))
        else:
            out_csv_arg = str(Path("csv") / "{}_results.csv".format(args.system))
    if not fig_dir_arg:
        if out_base:
            fig_dir_arg = str(out_base / "figures")
        else:
            fig_dir_arg = str(Path("figures"))

    args.out_csv = out_csv_arg
    args.fig_dir = fig_dir_arg
    args.out_base = out_base_arg

    Path(args.out_csv).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).resolve().mkdir(parents=True, exist_ok=True)

def _ensure_main_log_file(args):
    out_csv_path = Path(args.out_csv).resolve()
    run_dir = out_csv_path.parent.parent
    log_path = run_dir / "main.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    target = str(log_path.resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if str(Path(handler.baseFilename).resolve()) == target:
                    return
            except Exception:
                continue

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    file_handler = logging.FileHandler(target)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _prepend_hydra_config_to_main_log(args):
    out_csv_path = Path(args.out_csv).resolve()
    run_dir = out_csv_path.parent.parent
    main_log_path = run_dir / "main.log"
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    marker_begin = "=============================="
    marker_end = "=============================="

    if not hydra_cfg_path.is_file():
        return

    cfg_text = hydra_cfg_path.read_text(encoding="utf-8").strip()
    if not cfg_text:
        return

    existing = main_log_path.read_text(encoding="utf-8") if main_log_path.exists() else ""
    if marker_begin in existing:
        return

    header = "{}\n{}\n{}\n\n".format(marker_begin, cfg_text, marker_end)
    main_log_path.write_text(header + existing, encoding="utf-8")


def run(args):
    _prepare_output_layout(args)
    _ensure_main_log_file(args)
    _prepend_hydra_config_to_main_log(args)
    logger.info(
        "[START] system=%s input_dir=%s out_csv=%s fig_dir=%s max_files=%s",
        args.system,
        Path(args.input_dir).resolve(),
        Path(args.out_csv).resolve(),
        Path(args.fig_dir).resolve(),
        args.max_files,
    )

    fitter = build_fitter(args, args.out_csv)
    raw_df, out_df = fitter.run_scan_to_csv(system=args.system, args=args)

    logger.info("[DONE] system=%s rows_raw=%s rows_out=%s", args.system, len(raw_df), len(out_df))
    logger.info("[DONE] out csv -> %s", Path(args.out_csv).resolve())
