"""
KOR Relative QE calculator (cut-based, no fitting).

Implements the Korean group's relative-QE analysis exactly as described in
kor2.pdf (2026-04-08):

  Step 1. Event time window     — signal window fixed in NTP (100 ns)
  Step 2. Pulse-height cut      — max > 5 mV
  Step 3. Timing cut            — diff ∈ [diff_lo, diff_hi] samples (40 ns default)
  Step 4. Dark subtraction      — N_dark_raw * (T_sig_eff / T_dark)
  Step 5. Relative QE           — (N_signal - N_dark_exp) / N_total

No fitting is performed. Each prd_*.root file produces one row.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from uap.scan_reader import kor_reader
from uap.tool import scan_prepare


logger = logging.getLogger(__name__)

METHOD_NAME = "kor_relative_qe"

SAMPLE_NS = 2.0
ADC_TO_MV = 0.1220703125


def _read_tparam(root_file, name, default=None):
    """Return TParameter fVal, or default if missing / unreadable."""
    try:
        obj = root_file[name]
    except (KeyError, Exception):
        return default
    try:
        return obj.member("fVal")
    except Exception:
        return default


class KorRelativeQEFitter(object):
    """
    Cut-based KOR relative-QE calculator.

    Not a true fitter — kept under uap/fit/ for pipeline consistency.
    Exposes run_scan_to_csv(system, args) matching BaseScanFitter.
    """

    def __init__(
        self,
        diff_lo_samples=195,
        diff_hi_samples=215,
        pulse_thr_mv=5.0,
        t_signal_eff_ns=None,
        default_dark_window_ns=600.0,
        fig_dir=None,
    ):
        self.diff_lo = int(diff_lo_samples)
        self.diff_hi = int(diff_hi_samples)
        self.pulse_thr_mv = float(pulse_thr_mv)
        # If not set, effective signal window = diff cut width in ns.
        self.t_signal_eff_ns = (
            float(t_signal_eff_ns)
            if t_signal_eff_ns is not None
            else float((self.diff_hi - self.diff_lo) * SAMPLE_NS)
        )
        self.default_dark_window_ns = float(default_dark_window_ns)
        self.fig_dir = Path(fig_dir).resolve() if fig_dir else None
        if self.fig_dir:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline entry point (matches BaseScanFitter).
    def run_scan_to_csv(self, system, args):
        system = str(system or "").strip().lower()
        if system != "kor":
            raise RuntimeError(
                "KorRelativeQEFitter only supports system=kor, got: {}".format(system)
            )
        return self._run_kor(args)

    # Resolve inputs, compute per-file QE, write CSV.
    def _run_kor(self, args):
        input_dir = Path(args.input_dir).resolve()
        files = sorted(input_dir.glob("prd_*.root"))
        if getattr(args, "max_files", 0):
            files = files[: int(args.max_files)]
        if not files:
            raise SystemExit("No prd_*.root found in {}".format(input_dir))

        # Filter files by requested serial block (phi/theta encoding).
        selected = []
        for fp in files:
            parsed = kor_reader.extract_serial_block_angles(fp.name, args.serial)
            if parsed is None:
                continue
            phi, theta_raw = parsed
            selected.append((fp, int(phi), int(theta_raw)))
        if not selected:
            raise SystemExit("No files matched serial={}.".format(args.serial))

        files_for_auto = [x[0] for x in selected]
        ref_order, mismatches = kor_reader.check_serial_order_consistency(files_for_auto)
        if mismatches:
            logger.warning(
                "[WARN] serial order mismatch across files. reference=%s mismatched=%s first=%s",
                ref_order,
                len(mismatches),
                mismatches[0][0],
            )

        ctx = scan_prepare.resolve_kor_channels(
            args=args,
            files_for_auto=files_for_auto,
            parse_auto_or_int_fn=_parse_auto_or_int,
            auto_pick_trigger_channel_fn=kor_reader.auto_pick_trigger_channel,
            auto_pick_channel_fn=kor_reader.auto_pick_channel,
        )

        logger.info(
            "[KOR][RELQE] serial=%s channel=%s trigger_ch=%s  "
            "ph_thr=%s mV  diff_cut=[%s,%s] smp (%.1f ns)  default_dark_ns=%.1f",
            args.serial,
            ctx["channel"],
            ctx["trigger_ch"],
            self.pulse_thr_mv,
            self.diff_lo,
            self.diff_hi,
            self.t_signal_eff_ns,
            self.default_dark_window_ns,
        )

        rows = []
        total = len(selected)
        for idx, (fp, phi, theta_raw) in enumerate(selected, start=1):
            try:
                row = self._process_one_file(fp, phi, theta_raw, ctx, args.serial)
            except Exception as exc:
                logger.warning(
                    "[RELQE][%s/%s][FAIL] %s: %s: %s",
                    idx, total, fp.name, type(exc).__name__, exc,
                )
                continue
            self._log_row(idx, total, row)
            rows.append(row)

        if not rows:
            raise SystemExit("No valid KOR relative-QE results produced.")

        raw_df = pd.DataFrame(rows).sort_values(
            ["phi_raw", "theta_raw", "file"]
        ).reset_index(drop=True)
        out_df = self._postprocess(raw_df)

        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        logger.info("[RELQE] wrote %d rows -> %s", len(out_df), out_csv)
        return raw_df, out_df

    # Read one prd file and compute cut-based QE.
    def _process_one_file(self, fp, phi, theta_raw, ctx, serial):
        ch = int(ctx["channel"])
        tree_name = "tree_ch{}".format(ch)

        with uproot.open(str(fp)) as f:
            if tree_name not in f:
                raise KeyError("missing tree {}".format(tree_name))
            tree = f[tree_name]
            max_adc = np.asarray(tree["max"].array(library="np")).reshape(-1)
            diff = np.asarray(tree["diff"].array(library="np")).reshape(-1)

            n_dark_raw = _read_tparam(f, "NoiseCount_ch{}".format(ch), default=0)
            thr_mv_stored = _read_tparam(f, "Config_Threshold_mV_ch{}".format(ch))
            dark_ns_stored = _read_tparam(f, "Config_DarkWindow_ns_ch{}".format(ch))
            sig_ns_stored = _read_tparam(f, "Config_SigWindow_ns_ch{}".format(ch))

        n_total = int(max_adc.size)

        # If the NTP didn't store the dark window size, fall back to the
        # nominal PDF value. Warn so the user notices stale prd files.
        if dark_ns_stored is None:
            logger.warning(
                "[RELQE] %s: Config_DarkWindow_ns_ch%d missing — "
                "fall back to %.1f ns (regenerate prd with updated NTP).",
                fp.name, ch, self.default_dark_window_ns,
            )
            t_dark_ns = self.default_dark_window_ns
        else:
            t_dark_ns = float(dark_ns_stored)

        # Cuts
        max_mv = max_adc.astype(float) * ADC_TO_MV
        pass_ph = max_mv > self.pulse_thr_mv
        pass_timing = (diff >= self.diff_lo) & (diff <= self.diff_hi)
        n_signal_raw = int(np.sum(pass_ph & pass_timing))

        # Dark subtraction
        f_scale = self.t_signal_eff_ns / t_dark_ns if t_dark_ns > 0 else 0.0
        n_dark_raw = int(n_dark_raw)
        n_dark_exp = n_dark_raw * f_scale
        n_sig_corr = n_signal_raw - n_dark_exp

        # Poisson-propagated error on N_sig_corr (then divided by N_total)
        var_sig_corr = float(n_signal_raw) + (f_scale ** 2) * float(n_dark_raw)
        err_sig_corr = float(np.sqrt(var_sig_corr))
        rel_qe = n_sig_corr / n_total if n_total > 0 else np.nan
        rel_qe_err = err_sig_corr / n_total if n_total > 0 else np.nan

        row = scan_prepare.build_kor_row(
            fp_name=fp.name,
            serial=serial,
            ctx=ctx,
            phi=phi,
            theta_raw=theta_raw,
            window_method="cut_based",
            peak=float(0.5 * (self.diff_lo + self.diff_hi)),
        )
        row.update({
            "n_total": n_total,
            "n_signal_raw": n_signal_raw,
            "n_dark_raw": n_dark_raw,
            "n_dark_expected": float(n_dark_exp),
            "n_signal_corrected": float(n_sig_corr),
            "rel_qe": float(rel_qe),
            "rel_qe_err": float(rel_qe_err),
            "pulse_thr_mv": self.pulse_thr_mv,
            "diff_lo_smp": self.diff_lo,
            "diff_hi_smp": self.diff_hi,
            "t_signal_eff_ns": self.t_signal_eff_ns,
            "t_dark_ns": t_dark_ns,
            "f_scale": f_scale,
            "thr_mv_stored": (
                float(thr_mv_stored) if thr_mv_stored is not None else np.nan
            ),
            "sig_window_ns_stored": (
                float(sig_ns_stored) if sig_ns_stored is not None else np.nan
            ),
            # Compatibility with BaseScanFitter CSV columns (downstream tools).
            "sig_yield": float(n_sig_corr),
            "sig_err": float(err_sig_corr),
            "n_in_window": n_signal_raw,
            "window_min": float(self.diff_lo * SAMPLE_NS),
            "window_max": float(self.diff_hi * SAMPLE_NS),
        })
        return row

    # Per-phi normalization (theta_raw == 0 as reference).
    def _postprocess(self, df):
        out = df.copy()
        out["rel_qe_norm"] = np.nan
        out["rel_qe_norm_err"] = np.nan
        out["relative_de"] = np.nan
        out["relative_de_err"] = np.nan

        for phi_value, grp in out.groupby("phi_raw"):
            idx = grp.index
            center = grp[grp["theta_raw"] == 0]
            if center.empty:
                logger.warning(
                    "[RELQE] phi_raw=%s missing theta_raw=0, skip normalization.",
                    phi_value,
                )
                continue
            qe0 = float(np.nanmean(center["rel_qe"].values))
            qe0_err_vals = center["rel_qe_err"].values
            n_ok = int(np.sum(np.isfinite(qe0_err_vals)))
            qe0_err = (
                float(np.sqrt(np.nansum(qe0_err_vals ** 2)) / n_ok) if n_ok > 0 else np.nan
            )
            if not np.isfinite(qe0) or qe0 == 0:
                continue
            qe = out.loc[idx, "rel_qe"].values
            qe_err = out.loc[idx, "rel_qe_err"].values
            norm = qe / qe0
            with np.errstate(invalid="ignore"):
                norm_err = np.abs(norm) * np.sqrt(
                    (qe_err / np.where(qe != 0, qe, np.nan)) ** 2
                    + (qe0_err / qe0) ** 2
                )
            out.loc[idx, "rel_qe_norm"] = norm
            out.loc[idx, "rel_qe_norm_err"] = norm_err
            out.loc[idx, "relative_de"] = norm
            out.loc[idx, "relative_de_err"] = norm_err
        return out

    @staticmethod
    def _log_row(idx, total, row):
        logger.info(
            "[RELQE][%d/%d] file=%s phi=%d theta=%d  "
            "N_tot=%d N_sig=%d N_dark=%d f=%.4f  "
            "N_sig_corr=%.1f  rel_qe=%.4f±%.4f",
            idx, total, row["file"], row["phi_raw"], row["theta_raw"],
            row["n_total"], row["n_signal_raw"], row["n_dark_raw"],
            row["f_scale"], row["n_signal_corrected"],
            row["rel_qe"], row["rel_qe_err"],
        )


def _parse_auto_or_int(value, arg_name):
    if value is None:
        return "auto"
    s = str(value).strip().lower()
    if s in ("", "auto"):
        return "auto"
    try:
        return int(s)
    except ValueError:
        raise ValueError(
            "Invalid value for {}: {!r} (expected 'auto' or int)".format(arg_name, value)
        )
