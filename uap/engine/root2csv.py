#!/usr/bin/env python3
"""ROOT -> CSV pipeline for AUS/KOR scan analyses."""

import argparse
import logging
from pathlib import Path

import yaml

from uap.fit.charge_spectrum_fit import (
    CHARGE_METHOD_NAME,
    CHARGE_METHOD_NAMES,
    ChargeSpectrumFitter,
)
from uap.fit.emg_timing_offset_fit import TimingEMGFitter
from uap.fit.kor_relative_qe_fit import (
    METHOD_NAME as KOR_RELQE_METHOD_NAME,
    KorRelativeQEFitter,
)


logger = logging.getLogger(__name__)


def _is_charge_method(method_name):
    return str(method_name or "").strip() in CHARGE_METHOD_NAMES


def _is_kor_relqe_method(method_name):
    return str(method_name or "").strip() == KOR_RELQE_METHOD_NAME


def _add_charge_args(parser):
    parser.add_argument("--charge-branch", default="auto")
    parser.add_argument(
        "--charge-fit-range",
        nargs=2,
        type=float,
        metavar=("QMIN", "QMAX"),
        default=None,
    )
    parser.add_argument("--npe", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--poisson-gaussian", action="store_true",
                        help="Tie Gaussian yields to Poisson weights (default: independent yields)")
    parser.add_argument("--charge-fit-statistic", choices=["unbinned_nll", "chi2", "kor_chi2"],
                        default="unbinned_nll", help="Legacy option; prefer the four-section fit config")
    parser.add_argument("--charge-fit-nbins", type=int, default=300)
    parser.add_argument("--spe-mu-min-pc", type=float, default=1.0,
                        help="Lower bound on SPE peak position (pC). Default 1.0")
    parser.add_argument("--spe-mu-max-pc", type=float, default=2.5,
                        help="Upper bound on SPE peak position (pC). Default 2.5")


def build_fitter(args, out_csv_path):
    fit_config = getattr(args, "fit", None)
    if fit_config is not None:
        args.fit_method = "timing" if fit_config["model"]["name"] == "emg" else "charge"
    fig_dir_arg = getattr(args, "fig_dir", "")
    fig_dir = (
        Path(fig_dir_arg).resolve()
        if fig_dir_arg
        else Path(out_csv_path).resolve().parent / "figures"
    )
    if _is_kor_relqe_method(args.fit_method):
        return KorRelativeQEFitter(
            diff_lo_samples=int(getattr(args, "diff_lo_smp", 195)),
            diff_hi_samples=int(getattr(args, "diff_hi_smp", 215)),
            pulse_thr_mv=float(getattr(args, "pulse_thr_mv", 5.0)),
            t_signal_eff_ns=getattr(args, "t_signal_eff_ns", None),
            default_dark_window_ns=float(getattr(args, "default_dark_window_ns", 600.0)),
            fig_dir=fig_dir,
        )
    if _is_charge_method(args.fit_method):
        qmin, qmax = (None, None)
        fit_range = getattr(args, "charge_fit_range", None)
        if fit_range:
            qmin, qmax = [float(x) for x in fit_range]
        def _prior_tuple(val):
            if val is None:
                return None
            seq = list(val)
            if len(seq) != 2:
                raise ValueError("Prior must be a [mean, sigma] pair, got {!r}".format(val))
            return (float(seq[0]), float(seq[1]))
        return ChargeSpectrumFitter(
            method_name=str(args.fit_method).strip(),
            fig_dir=fig_dir,
            inc_backscatter=bool(
                getattr(args, "inc_backscatter", getattr(args, "inc_bkg", True))
            ),
            npe=int(getattr(args, "npe", 2)),
            poisson_gaussian=bool(getattr(args, "poisson_gaussian", False)),
            charge_fit_statistic=getattr(args, "charge_fit_statistic", "unbinned_nll"),
            charge_fit_nbins=int(getattr(args, "charge_fit_nbins", 300)),
            charge_branch=getattr(args, "charge_branch", "auto"),
            charge_qmin=qmin,
            charge_qmax=qmax,
            spe_mu_min_pc=getattr(args, "spe_mu_min_pc", 1.0),
            spe_mu_max_pc=getattr(args, "spe_mu_max_pc", 2.5),
            spe_mu_prior=_prior_tuple(getattr(args, "spe_mu_prior", None)),
            spe_sigma_prior=_prior_tuple(getattr(args, "spe_sigma_prior", None)),
            fit_config=fit_config,
        )
    nbins_arg = getattr(args, "plot_nbins", None)
    emg_kwargs = {"method_name": args.fit_method, "fig_dir": fig_dir}
    if nbins_arg:
        emg_kwargs["nbins"] = int(nbins_arg)
    if fit_config is not None:
        emg_kwargs["fit_config"] = fit_config
    return TimingEMGFitter(**emg_kwargs)


def build_parser():
    ap = argparse.ArgumentParser(description="ROOT -> CSV pipeline for AUS/KOR.")
    sub = ap.add_subparsers(dest="system")

    ap_aus = sub.add_parser("aus", help="AUS timing fit from output_theta*_phi*.root")
    ap_aus.add_argument("--input-dir", required=True)
    ap_aus.add_argument("--out-csv", default="", help="Output fit CSV path")
    ap_aus.add_argument("--out-base", default="", help="Run output root (default: <UAP_HOME>/outputs)")
    ap_aus.add_argument("--serial", default="")
    ap_aus.add_argument("--fit-method", default="fitandplot_emg", help="Fit method")
    ap_aus.add_argument("--fig-dir", default="", help="Directory to store diagnostic fit figures")
    _add_charge_args(ap_aus)
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
    # Charge-fit specific cuts (only consulted when fit_method is a charge method).
    ap_aus.add_argument("--require-pulsestart", dest="require_pulsestart",
                        action="store_true",
                        help="Drop events with NaN PMT PulseStart "
                             "(reproduces AUS implicit LED discriminator cut)")
    ap_aus.set_defaults(require_pulsestart=False)
    ap_aus.add_argument("--apply-timing-cut", dest="apply_timing_cut",
                        action="store_true",
                        help="Apply delta_PMT_trigger timing cut on charge fit input")
    ap_aus.set_defaults(apply_timing_cut=False)
    ap_aus.add_argument("--timing-cut", nargs=2, type=float,
                        metavar=("TMIN_NS", "TMAX_NS"),
                        default=[300.0, 320.0],
                        help="Timing cut window in ns for charge fit "
                             "(used only with --apply-timing-cut)")

    ap_kor = sub.add_parser("kor", help="KOR diff fit from prd_*.root")
    ap_kor.add_argument("--input-dir", required=True)
    ap_kor.add_argument("--out-csv", default="", help="Output fit CSV path")
    ap_kor.add_argument("--out-base", default="", help="Run output root (default: <UAP_HOME>/outputs)")
    ap_kor.add_argument("--serial", default="AA0000A")
    ap_kor.add_argument("--channel", default="auto")
    ap_kor.add_argument("--trigger-ch", default="auto")
    ap_kor.add_argument("--fit-method", default="fitandplot_emg", help="Fit method")
    ap_kor.add_argument("--fig-dir", default="", help="Directory to store diagnostic fit figures")
    _add_charge_args(ap_kor)
    ap_kor.add_argument("--inc-bkg", dest="inc_bkg", action="store_true")
    ap_kor.add_argument("--no-inc-bkg", dest="inc_bkg", action="store_false")
    ap_kor.set_defaults(inc_bkg=True)
    ap_kor.add_argument("--window-half-width", type=float, default=30.0)
    ap_kor.add_argument("--window-bin-width", type=float, default=2.0)
    ap_kor.add_argument("--tmin", type=float, default=None)
    ap_kor.add_argument("--tmax", type=float, default=None)
    ap_kor.add_argument("--max-files", type=int, default=0)
    # Relative-QE (cut-based) options
    ap_kor.add_argument("--diff-lo-smp", type=int, default=195,
                        help="Lower bound of timing cut in ADC samples (default 195)")
    ap_kor.add_argument("--diff-hi-smp", type=int, default=215,
                        help="Upper bound of timing cut in ADC samples (default 215)")
    ap_kor.add_argument("--pulse-thr-mv", type=float, default=5.0,
                        help="Pulse-height threshold in mV (default 5.0)")
    ap_kor.add_argument("--t-signal-eff-ns", type=float, default=None,
                        help="Effective signal window for dark scaling (ns). "
                             "Default = diff cut width.")
    ap_kor.add_argument("--default-dark-window-ns", type=float, default=600.0,
                        help="Fallback T_dark (ns) if prd file lacks Config_DarkWindow_ns")
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
        stem = "{}_results.csv".format(args.system)
        if _is_charge_method(getattr(args, "fit_method", "")):
            stem = "{}_charge_results.csv".format(args.system)
        if out_base:
            out_csv_arg = str(out_base / "csv" / stem)
        else:
            out_csv_arg = str(Path("csv") / stem)
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
    if getattr(args, "fit", None) is not None:
        args.fit_method = "timing" if args.fit["model"]["name"] == "emg" else "charge"
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
    if hasattr(fitter, "fit_config"):
        config_path = Path(args.out_csv).resolve().parent.parent / "fit_config.yaml"
        config_path.write_text(yaml.safe_dump(fitter.fit_config, sort_keys=False), encoding="utf-8")
        logger.info("[FIT CONFIG] %s", fitter.fit_config)
    raw_df, out_df = fitter.run_scan_to_csv(system=args.system, args=args)

    logger.info("[DONE] system=%s rows_raw=%s rows_out=%s", args.system, len(raw_df), len(out_df))
    logger.info("[DONE] out csv -> %s", Path(args.out_csv).resolve())
