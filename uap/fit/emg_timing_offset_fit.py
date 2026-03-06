"""
zfit EMG fitter for timing-based analyses.
The analysis method is rewritten from https://github.com/wihann00/HKAus_precal_analysis/tree/main. (Author: Wi Han Ng)
"""

import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .fitter_interface import BaseScanFitter
from .common_math import safe_ratio, safe_ratio_err
from .plot_utils import save_fit_with_pull_plot
from uap.scan_reader.aus_reader import (
    DEFAULT_AUS_CHANNELS,
    load_branch,
    parse_theta_phi_aus,
    resolve_aus_channel,
)
from uap.scan_reader.kor_reader import (
    auto_pick_channel,
    auto_pick_trigger_channel,
    check_serial_order_consistency,
    extract_serial_block_angles,
    read_tree_branch,
)
from uap.tool.scan_prepare import (
    build_aus_row,
    build_kor_row,
    init_aus_prep_stats,
    init_kor_prep_stats,
    resolve_aus_context,
    resolve_kor_channels,
)
from uap.tool.window import select_window


logger = logging.getLogger(__name__)
plt = None
tensorflow = None
zfit = None
z = None
UAPEMG = None
_RUNTIME_READY = False
SAMPLE_NS = 2.0


# Keep third-party runtime output quiet by default.
def _configure_runtime_verbosity():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ZFIT_DISABLE_TF_WARNINGS", "1")
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"google\.api_core\..*"
    )
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"google\.auth(\..*)?$"
    )
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"google\.oauth2(\..*)?$"
    )


# Load heavy deps (zfit/tensorflow/matplotlib).
def _ensure_runtime():
    global plt, tensorflow, zfit, z, UAPEMG, _RUNTIME_READY
    if _RUNTIME_READY:
        return
    _configure_runtime_verbosity()
    try:
        import matplotlib.pyplot as _plt
        import tensorflow as _tensorflow
        import zfit as _zfit
        from zfit import z as _z
    except Exception as exc:
        raise RuntimeError(
            "TimingEMGFitter requires zfit/tensorflow/matplotlib: {}".format(exc)
        )

    plt = _plt
    tensorflow = _tensorflow
    zfit = _zfit
    z = _z
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    try:
        zfit.settings.changed_warnings.hesse_name = False
    except Exception:
        pass
    # EMG PDF used for timing fit.
    class _UAPEMG(zfit.pdf.BasePDF):

        def __init__(self, obs, mu, lambd, sigma, extended=None, norm=None, name=None):
            params = {"mu": mu, "lambd": lambd, "sigma": sigma}
            super().__init__(
                obs=obs, params=params, extended=extended, norm=norm, name=name
            )

        # Unnormalized EMG formula.
        def _unnormalized_pdf(self, x):
            x = z.unstack_x(x)
            mu = self.params["mu"]
            lambd = self.params["lambd"]
            sigma = self.params["sigma"]
            a = (mu + (lambd * (sigma**2)) - x) / (z.sqrt(2.0) * sigma)
            b = 2 * mu + (lambd * (sigma**2)) - 2 * x
            return (lambd / 2.0) * z.exp((lambd / 2.0) * b) * tensorflow.math.erfc(a)

    UAPEMG = _UAPEMG
    _RUNTIME_READY = True

# Concrete fitter.
class TimingEMGFitter(BaseScanFitter):

    # Store fitter options and create figure directory.
    def __init__(self, method_name="fitandplot_emg", fig_dir=None, nbins=30):
        _ensure_runtime()

        self.method_name = method_name
        self.nbins = int(nbins)
        self.fig_dir = Path(fig_dir).resolve() if fig_dir else None
        if self.fig_dir:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

    # Select fit backend by method_name.
    def fit(self, request):
        method = self.method_name
        if method == "fitandplot_emg":
            return self._fit_emg(request)
        raise RuntimeError("Unsupported built-in fit method: {}".format(method))

    # Convert coord into a safe suffix for zfit parameter names.
    @staticmethod
    def _coord_key(coord):
        if isinstance(coord, (tuple, list)):
            txt = "_".join([str(x) for x in coord])
        else:
            txt = str(coord)
        txt = re.sub(r"[^0-9A-Za-z_]+", "_", txt)
        txt = txt.strip("_")
        return txt or "coord"

    # Read one parameter error from hesse() output.
    @staticmethod
    def _extract_err(hesse, param):
        if hesse is None:
            return np.nan
        try:
            entry = hesse[param]
            if isinstance(entry, dict):
                return float(entry.get("error", np.nan))
            return float(getattr(entry, "error", np.nan))
        except Exception:
            return np.nan

    # Estimate FWHM from sampled fitted model curve.
    def _compute_fwhm(self, model, data, size, xr):
        try:
            x = np.linspace(float(xr[0]), float(xr[1]), 1000)
            y_model = np.asarray(zfit.run(model.pdf(x)), dtype=float)
            area = float(zfit.run(data.data_range.area()))
            y = y_model * float(size) / float(self.nbins) * area

            if y.size == 0 or not np.isfinite(y).any():
                logger.warning("[FWHM][WARN] invalid sampled model values, return NaN.")
                return np.nan
            ymax = float(np.nanmax(y))
            if ymax <= 0:
                logger.warning("[FWHM][WARN] non-positive model maximum, return NaN.")
                return np.nan

            y_half = y - (ymax / 2.0)
            peak_x = float(x[int(np.nanargmax(y))])

            signs = np.sign(y_half)
            cross_idx = np.where(np.diff(signs) != 0)[0]
            roots = []
            for idx in cross_idx:
                x1, x2 = x[idx], x[idx + 1]
                y1, y2 = y_half[idx], y_half[idx + 1]
                if y2 == y1:
                    continue
                roots.append(float(x1 - y1 * (x2 - x1) / (y2 - y1)))

            if len(roots) < 2:
                logger.warning("[FWHM][WARN] insufficient half-maximum crossings, return NaN.")
                return np.nan

            roots = np.asarray(roots, dtype=float)
            left = roots[roots < peak_x]
            right = roots[roots > peak_x]
            if left.size > 0 and right.size > 0:
                r1 = float(np.max(left))
                r2 = float(np.min(right))
            else:
                r1 = float(roots[0])
                r2 = float(roots[-1])
            return abs(r2 - r1)
        except Exception as exc:
            logger.warning("[FWHM][WARN] computation failed: %s: %s. return NaN.", type(exc).__name__, exc)
            return np.nan

    # Save one diagnostic fit plot (data/model + pull).
    def _make_plot(self, model, data_np, xr, size, out, plotname):
        if self.fig_dir is None:
            return

        x = np.linspace(float(xr[0]), float(xr[1]), 1000)
        y_model = np.asarray(zfit.run(model.pdf(x)), dtype=float)
        area = float(xr[1] - xr[0])
        y = y_model * float(size) / float(self.nbins) * area

        counts, edges = np.histogram(
            data_np, bins=self.nbins, range=(float(xr[0]), float(xr[1]))
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        y_exp = (
            np.asarray(zfit.run(model.pdf(centers)), dtype=float)
            * float(size)
            / float(self.nbins)
            * area
        )
        pull = (counts - y_exp) / np.sqrt(np.clip(counts, 1, None))

        text = "mu={:.3g}".format(
            out.get("mean", np.nan),
        )
        text_lines = [
            text,
            "lambda={:.3g}".format(out.get("lambda", np.nan)),
            "sigma={:.3g}".format(out.get("sigma", np.nan)),
            "sig={:.4g}".format(out.get("sig_yield", np.nan)),
            "FWHM={:.3g}".format(out.get("FWHM", np.nan)),
        ]

        name = (plotname or "fit").strip()
        target = self.fig_dir / "{}.png".format(name)
        save_fit_with_pull_plot(
            plt=plt,
            out_png=target,
            centers=centers,
            counts=counts,
            yerr=np.sqrt(np.clip(counts, 1, None)),
            x_model=x,
            y_model=y,
            pull=pull,
            xlim=(float(xr[0]), float(xr[1])),
            text_lines=text_lines,
            y_label="Events",
            x_label="Time (ns)",
            pull_ylim=(-5, 5),
        )
        logger.info("[PLOT] saved {}".format(target))

    # Run one EMG fit and return standard fit-output dict.
    def _fit_emg(self, request):
        data_np = np.asarray(request.data).reshape(-1)
        data_np = data_np[np.isfinite(data_np)]
        if data_np.size == 0:
            raise RuntimeError("No finite data points for fit.")

        fit_kwargs = dict(request.fit_kwargs or {})
        inc_bkg = bool(fit_kwargs.get("inc_bkg", True))

        xr = (
            request.xr
            if request.xr is not None
            else [float(np.min(data_np)), float(np.max(data_np))]
        )
        if xr[1] <= xr[0]:
            raise RuntimeError(
                "Invalid fit range xr={}, data size={}".format(xr, data_np.size)
            )

        coord_key = self._coord_key(request.coord)
        obs = zfit.Space("x", limits=(float(xr[0]), float(xr[1])))
        data = zfit.Data.from_numpy(obs=obs, array=data_np)
        size = int(data_np.shape[0])

        mu_guess = float(np.mean(data_np))
        if not np.isfinite(mu_guess):
            mu_guess = 0.5 * (float(xr[0]) + float(xr[1]))
        sigma_guess = float(np.std(data_np))
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            sigma_guess = max((float(xr[1]) - float(xr[0])) * 0.1, 0.2)

        mu_lo = min(mu_guess * 0.3, mu_guess * 1.5)
        mu_hi = max(mu_guess * 0.3, mu_guess * 1.5)
        if mu_lo == mu_hi:
            mu_lo = mu_guess - 1.0
            mu_hi = mu_guess + 1.0

        mu = zfit.Parameter("mu_{}".format(coord_key), mu_guess, mu_lo, mu_hi)
        lambd = zfit.Parameter("lambd_{}".format(coord_key), 0.1, 0.005, 5.0)
        sigma = zfit.Parameter(
            "sigma_{}".format(coord_key), np.clip(sigma_guess, 0.2, 20.0), 0.2, 50.0
        )

        emg = UAPEMG(obs=obs, mu=mu, lambd=lambd, sigma=sigma)
        emg_yield_naive = float(size) * 0.8
        bkg_yield_naive = float(size) - emg_yield_naive

        bkg_yield = None
        coeffs = None
        if inc_bkg:
            coeffs = [zfit.Parameter("coeff_0_{}".format(coord_key), 0.0, -2.0, 1.0)]
            chebyshev = zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)

            emg_yield = zfit.Parameter(
                "emg_yield_{}".format(coord_key),
                emg_yield_naive,
                max(emg_yield_naive * 0.01, 1.0),
                max(emg_yield_naive * 1.5, 2.0),
                step_size=1,
            )
            emg_ext = emg.create_extended(emg_yield)

            bkg_yield = zfit.Parameter(
                "comb_bkg_yield_{}".format(coord_key),
                max(bkg_yield_naive, 1.0),
                0.0,
                max(bkg_yield_naive * 2.5, 2.0),
                step_size=1,
            )
            bkg_ext = chebyshev.create_extended(bkg_yield)
            model = zfit.pdf.SumPDF([emg_ext, bkg_ext])
        else:
            emg_yield = zfit.Parameter(
                "emg_yield_{}".format(coord_key),
                float(size),
                0.0,
                max(float(size) * 1.2, 2.0),
                step_size=1,
            )
            model = emg.create_extended(emg_yield)

        nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
        minimizer = zfit.minimize.Minuit()
        result = minimizer.minimize(nll)
        try:
            hesse = result.hesse()
        except Exception:
            hesse = None

        out = {}
        out["mean"] = float(zfit.run(mu.value()))
        out["lambda"] = float(zfit.run(lambd.value()))
        out["sigma"] = float(zfit.run(sigma.value()))
        out["mu_err"] = self._extract_err(hesse, mu)
        out["std_err"] = self._extract_err(hesse, sigma)
        out["lambd_err"] = self._extract_err(hesse, lambd)
        out["sig_yield"] = float(zfit.run(emg_yield.value()))
        out["sig_err"] = self._extract_err(hesse, emg_yield)
        out["bkg_yield"] = (
            float(zfit.run(bkg_yield.value())) if bkg_yield is not None else np.nan
        )
        out["bkg_err"] = (
            self._extract_err(hesse, bkg_yield) if bkg_yield is not None else np.nan
        )
        if coeffs:
            out["coeff"] = float(zfit.run(coeffs[0].value()))

        out["FWHM"] = self._compute_fwhm(model, data, size, xr)
        self._make_plot(model, data_np, xr, size, out, request.plotname)
        return out

    # Build AUS relative columns (with optional SiPM normalization).
    def _apply_aus_relative_columns(self, df, use_sipm):
        out_df = df.copy()
        out_df["rel_yield"] = np.nan
        out_df["rel_yield_err"] = np.nan
        out_df["rel_sipm_yield"] = np.nan
        out_df["rel_sipm_yield_err"] = np.nan
        out_df["relative_norm"] = np.nan
        out_df["relative_norm_err"] = np.nan
        out_df["relative_de"] = np.nan
        out_df["relative_de_err"] = np.nan

        center_rows = out_df[(out_df["theta"] == 0) & (out_df["phi"] == 0)]
        if center_rows.empty:
            center_row = out_df.iloc[0]
            logger.warning(
                "[AUS] center point (theta=0,phi=0) missing, fallback to first row: %s",
                center_row.get("file", ""),
            )
        else:
            center_row = center_rows.iloc[0]

        sig0 = float(center_row.get("sig_yield", np.nan))
        sig0_err = float(center_row.get("sig_err", np.nan))

        out_df["rel_yield"] = safe_ratio(out_df["sig_yield"].values, sig0, logger=logger)
        out_df["rel_yield_err"] = safe_ratio_err(
            out_df["sig_yield"].values,
            out_df["sig_err"].values,
            sig0,
            sig0_err,
            logger=logger,
        )

        if use_sipm:
            sipm0 = float(center_row.get("sipm_sig_yield", np.nan))
            sipm0_err = float(center_row.get("sipm_sig_err", np.nan))

            out_df["rel_sipm_yield"] = safe_ratio(
                out_df["sipm_sig_yield"].values, sipm0, logger=logger
            )
            out_df["rel_sipm_yield_err"] = safe_ratio_err(
                out_df["sipm_sig_yield"].values,
                out_df["sipm_sig_err"].values,
                sipm0,
                sipm0_err,
                logger=logger,
            )

            point_ratio = safe_ratio(
                out_df["sig_yield"].values, out_df["sipm_sig_yield"].values
                , logger=logger
            )
            point_ratio_err = safe_ratio_err(
                out_df["sig_yield"].values,
                out_df["sig_err"].values,
                out_df["sipm_sig_yield"].values,
                out_df["sipm_sig_err"].values,
                logger=logger,
            )
            center_ratio = safe_ratio(sig0, sipm0, logger=logger)
            center_ratio_err = safe_ratio_err(sig0, sig0_err, sipm0, sipm0_err, logger=logger)

            out_df["relative_norm"] = safe_ratio(point_ratio, center_ratio, logger=logger)
            out_df["relative_norm_err"] = safe_ratio_err(
                point_ratio,
                point_ratio_err,
                center_ratio,
                center_ratio_err,
                logger=logger,
            )
            out_df["relative_de"] = out_df["relative_norm"]
            out_df["relative_de_err"] = out_df["relative_norm_err"]
        else:
            out_df["relative_de"] = out_df["rel_yield"]
            out_df["relative_de_err"] = out_df["rel_yield_err"]

        return out_df

    # Build KOR relative columns within each phi_raw group.
    def _apply_kor_relative_columns(self, df):
        out_df = df.copy()
        out_df["relative_qe"] = np.nan
        out_df["relative_qe_err"] = np.nan
        out_df["relative_de"] = np.nan
        out_df["relative_de_err"] = np.nan

        for phi_value, grp in out_df.groupby("phi_raw"):
            idx = grp.index
            center_rows = grp[grp["theta_raw"] == 0]
            if center_rows.empty:
                center_row = grp.iloc[0]
                logger.warning(
                    "[KOR] center point (theta_raw=0) missing for phi_raw=%s, fallback to first row: %s",
                    phi_value,
                    center_row.get("file", ""),
                )
                sig0 = float(center_row.get("sig_yield", np.nan))
                sig0_err = float(center_row.get("sig_err", np.nan))
            else:
                sig0_vals = pd.to_numeric(center_rows["sig_yield"], errors="coerce").values
                sig0 = float(np.nanmean(sig0_vals))

                sig0_err_vals = pd.to_numeric(center_rows["sig_err"], errors="coerce").values
                n_err = int(np.sum(np.isfinite(sig0_err_vals)))
                if n_err > 0:
                    sig0_err = float(np.sqrt(np.nansum(sig0_err_vals ** 2)) / n_err)
                else:
                    sig0_err = np.nan

                if len(center_rows) > 1:
                    logger.info(
                        "[KOR] phi_raw=%s has %s theta_raw=0 rows, normalize by mean center.",
                        phi_value,
                        len(center_rows),
                    )

            rel = safe_ratio(out_df.loc[idx, "sig_yield"].values, sig0, logger=logger)
            rel_err = safe_ratio_err(
                out_df.loc[idx, "sig_yield"].values,
                out_df.loc[idx, "sig_err"].values,
                sig0,
                sig0_err,
                logger=logger,
            )
            out_df.loc[idx, "relative_qe"] = rel
            out_df.loc[idx, "relative_qe_err"] = rel_err
            out_df.loc[idx, "relative_de"] = rel
            out_df.loc[idx, "relative_de_err"] = rel_err

        return out_df

    # Mark obviously invalid fit outputs.
    @staticmethod
    def _is_abnormal_main_fit(fit_out):
        sig = fit_out.get("sig_yield", np.nan)
        sigma = fit_out.get("sigma", np.nan)
        lambd = fit_out.get("lambda", np.nan)
        checks = [sig, sigma, lambd]
        if not np.all(np.isfinite(np.asarray(checks, dtype=float))):
            return True
        if float(sig) < 0 or float(sigma) <= 0 or float(lambd) <= 0:
            return True
        return False

    # Prepare AUS points:
    # - load PMT/trigger PulseStart
    # - build delta timing
    # - choose fit window
    # - attach optional SiPM aux fit input
    def _prepare_aus_input(self, args):
        _input_dir, out_csv, files = self.resolve_inputs(
            args=args,
            default_out_csv="csv/aus_results.csv",
            file_pattern="output_theta*_phi*.root",
            empty_msg="No output_theta*_phi*.root found in: {}",
        )

        defaults = dict(DEFAULT_AUS_CHANNELS)
        ctx = resolve_aus_context(args, defaults, resolve_aus_channel)

        self.log_channel_config(
            system="aus",
            serial=ctx["serial"],
            resolved={
                "pmt_ch": int(ctx["pmt_ch"]),
                "trigger_ch": int(ctx["trigger_ch"]),
                "sipm_ch": int(ctx["sipm_ch"]),
            },
            cfg={
                "pmt_ch": str(ctx["pmt_ch_cfg"]),
                "trigger_ch": str(ctx["trigger_ch_cfg"]),
                "sipm_ch": str(ctx["sipm_ch_cfg"]),
            },
            defaults=defaults,
        )

        tbranch = "PulseStart"
        pmt_tree = "Tree_CH{}".format(ctx["pmt_ch"])
        trigger_tree = "Tree_CH{}".format(ctx["trigger_ch"])
        sipm_tree = "Tree_CH{}".format(ctx["sipm_ch"])
        use_sipm = not args.no_sipm

        # Preparation counters for logging.
        prep_stats = init_aus_prep_stats(len(files))
        points = []
        for fp in files:
            parsed = parse_theta_phi_aus(fp)
            if parsed is None:
                continue
            theta, phi = parsed

            # Read PMT/trigger timing arrays.
            read_pair = self.run_step(
                lambda: (
                    load_branch(fp, pmt_tree, tbranch) * SAMPLE_NS,
                    load_branch(fp, trigger_tree, tbranch) * SAMPLE_NS,
                ),
                stats=prep_stats,
                fail_key="pmt_read_fail",
                file_name=fp.name,
                fail_msg="read PMT/trigger branch failed",
            )
            if read_pair is None:
                continue
            pmt_t, trigger_t = read_pair

            delta_pmt = pmt_t - trigger_t
            d_all = delta_pmt[np.isfinite(delta_pmt)]
            if d_all.size == 0:
                self.inc_stat(prep_stats, "pmt_empty_window")
                continue

            # Select PMT fit window around peak.
            pmt_window = self.run_step(
                lambda: select_window(
                    values=d_all,
                    method="peak_center",
                    tmin=args.tmin,
                    tmax=args.tmax,
                    half_width=args.window_half_width,
                    bin_width=args.window_bin_width,
                    positive_only=False,
                    drop_zero=False,
                ),
                stats=prep_stats,
                fail_key="window_fail",
                file_name=fp.name,
                fail_msg="PMT window selection failed",
            )
            if pmt_window is None:
                continue
            pmt_tmin, pmt_tmax, pmt_peak = pmt_window

            d = self.cut_interval(d_all, pmt_tmin, pmt_tmax)
            if d.size == 0:
                self.inc_stat(prep_stats, "pmt_empty_window")
                continue

            row = build_aus_row(
                fp_name=fp.name,
                ctx=ctx,
                defaults=defaults,
                theta=theta,
                phi=phi,
                pmt_tmin=pmt_tmin,
                pmt_tmax=pmt_tmax,
                pmt_peak=pmt_peak,
            )

            # Main fit point (PMT branch).
            point = self.make_point(
                row=row,
                main_fit_input=self.make_fit_input(
                    data=d,
                    coord=(theta, phi),
                    plotname="theta{}_phi{}".format(theta, phi),
                    xr=(pmt_tmin, pmt_tmax),
                    meta={"n_in_window": int(d.size)},
                ),
                aux_blocks=[],
            )

            if use_sipm:
                # Optional SiPM auxiliary fit for normalization.
                sipm = {
                    "prefix": "sipm",
                    "fit_input": None,
                }
                sipm_t = self.run_step(
                    lambda: load_branch(fp, sipm_tree, tbranch) * SAMPLE_NS,
                    stats=prep_stats,
                    fail_key="sipm_read_fail",
                    file_name=fp.name,
                    fail_msg="read SiPM branch failed",
                )
                if sipm_t is not None:
                    delta_sipm = sipm_t - trigger_t
                    ds_all = delta_sipm[np.isfinite(delta_sipm)]
                    if ds_all.size == 0:
                        self.inc_stat(prep_stats, "sipm_empty_window")
                    else:
                        sipm_window = self.run_step(
                            lambda: select_window(
                                values=ds_all,
                                method="peak_center",
                                tmin=args.sipm_tmin,
                                tmax=args.sipm_tmax,
                                half_width=args.sipm_window_half_width,
                                bin_width=args.sipm_window_bin_width,
                                positive_only=False,
                                drop_zero=False,
                            ),
                            stats=prep_stats,
                            fail_key="sipm_window_fail",
                            file_name=fp.name,
                            fail_msg="SiPM window selection failed",
                        )
                        if sipm_window is not None:
                            sipm_tmin, sipm_tmax, _ = sipm_window
                            row["sipm_window_min"] = sipm_tmin
                            row["sipm_window_max"] = sipm_tmax

                            ds = self.cut_interval(ds_all, sipm_tmin, sipm_tmax)
                            row["sipm_n_in_window"] = int(ds.size)
                            if ds.size > 0:
                                sipm["fit_input"] = self.make_fit_input(
                                    data=ds,
                                    coord=(theta, "sipm_{}".format(phi)),
                                    plotname="sipm_theta{}_phi{}".format(theta, phi),
                                    xr=(sipm_tmin, sipm_tmax),
                                )
                            else:
                                self.inc_stat(prep_stats, "sipm_empty_window")
                point["aux_blocks"].append(sipm)

            points.append(point)
            prep_stats["files_kept"] += 1

        return {
            "out_csv": out_csv,
            "use_sipm": use_sipm,
            "points": points,
            "prep_stats": prep_stats,
        }

    # Prepare KOR points:
    # - pick files by serial block
    # - resolve channel/trigger
    # - read diff branch
    # - choose fit window and build fit input
    def _prepare_kor_input(self, args):
        _input_dir, out_csv, files = self.resolve_inputs(
            args=args,
            default_out_csv="csv/kor_results.csv",
            file_pattern="prd_*.root",
            empty_msg="No prd_*.root found in {}",
        )

        # Keep only files containing the requested serial block.
        selected = []
        for fp in files:
            parsed = extract_serial_block_angles(fp.name, args.serial)
            if parsed is None:
                continue
            phi, theta_raw = parsed
            selected.append((fp, int(phi), int(theta_raw)))

        if not selected:
            raise SystemExit("No files matched serial={}.".format(args.serial))

        files_for_auto = [x[0] for x in selected]
        ref_order, mismatches = check_serial_order_consistency(files_for_auto)
        if mismatches:
            logger.warning(
                "[WARN] serial order mismatch across files. reference=%s mismatched=%s first=%s",
                ref_order,
                len(mismatches),
                mismatches[0][0],
            )

        # Resolve channel and trigger from config/auto mapping.
        ctx = resolve_kor_channels(
            args=args,
            files_for_auto=files_for_auto,
            parse_auto_or_int_fn=self.parse_auto_or_int,
            auto_pick_trigger_channel_fn=auto_pick_trigger_channel,
            auto_pick_channel_fn=auto_pick_channel,
        )

        self.log_channel_config(
            system="kor",
            serial=str(args.serial),
            resolved={
                "channel": int(ctx["channel"]),
                "trigger_ch": int(ctx["trigger_ch"]),
            },
            cfg={
                "channel": str(ctx["channel_cfg"]),
                "trigger_ch": str(ctx["trigger_ch_cfg"]),
            },
        )

        window_method = "peak_center"
        # Preparation counters for logging.
        prep_stats = init_kor_prep_stats(len(selected))
        points = []

        for idx, (fp, phi, theta_raw) in enumerate(selected):
            # Read KOR diff and convert to ns.
            diff = self.run_step(
                lambda: read_tree_branch(fp, ctx["channel"], "diff") * SAMPLE_NS,
                stats=prep_stats,
                fail_key="diff_read_fail",
                file_name=fp.name,
                fail_msg="read diff branch failed",
            )
            if diff is None:
                continue

            # Select KOR fit window around peak.
            kor_window = self.run_step(
                lambda: select_window(
                    values=diff,
                    method=window_method,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    half_width=args.window_half_width,
                    bin_width=args.window_bin_width,
                    positive_only=False,
                    drop_zero=True,
                ),
                stats=prep_stats,
                fail_key="window_fail",
                file_name=fp.name,
                fail_msg="peak search failure",
            )
            if kor_window is None:
                continue
            use_tmin, use_tmax, peak = kor_window

            diff_valid = diff[np.isfinite(diff)]
            diff_valid = diff_valid[diff_valid != 0]
            d = self.cut_interval(diff_valid, use_tmin, use_tmax)

            if d.size == 0:
                self.inc_stat(prep_stats, "empty_window")
                self.log_skip(fp.name, "no entries in selected fixed window.")
                continue

            # Main fit input for this KOR point.
            fit_input = self.make_fit_input(
                data=d,
                coord=("kor", args.serial, int(phi), int(theta_raw), idx),
                plotname="kor_{}_phi{}_theta{}_{}".format(
                    args.serial, phi, theta_raw, idx
                ),
                xr=(use_tmin, use_tmax),
                meta={
                    "window_min": float(use_tmin),
                    "window_max": float(use_tmax),
                    "peak": float(peak),
                    "n_in_window": int(d.size),
                },
            )

            row = build_kor_row(
                fp_name=fp.name,
                serial=args.serial,
                ctx=ctx,
                phi=phi,
                theta_raw=theta_raw,
                window_method=window_method,
                peak=peak,
            )

            points.append(
                self.make_point(
                    row=row,
                    main_fit_input=fit_input,
                    main_skip_msg="[SKIP] {}: fit failed in selected fixed window.".format(
                        fp.name
                    ),
                )
            )
            prep_stats["files_kept"] += 1

        return {"out_csv": out_csv, "points": points, "prep_stats": prep_stats}

    # Build system-specific config.
    def prepare_scan(self, system, args):
        if system == "aus":
            inputs = self._prepare_aus_input(args)
            return {
                "out_csv": inputs["out_csv"],
                "points": inputs["points"],
                "prep_stats": inputs.get("prep_stats", {}),
                "sort_cols": ["phi", "theta", "file"],
                "empty_msg": "No valid AUS fit results produced.",
                "postprocess": lambda df: self._apply_aus_relative_columns(
                    df, use_sipm=inputs["use_sipm"]
                ),
            }

        if system == "kor":
            inputs = self._prepare_kor_input(args)
            return {
                "out_csv": inputs["out_csv"],
                "points": inputs["points"],
                "prep_stats": inputs.get("prep_stats", {}),
                "sort_cols": ["phi_raw", "theta_raw", "file"],
                "empty_msg": "No valid KOR fits produced.",
                "postprocess": self._apply_kor_relative_columns,
            }

        raise RuntimeError("Unsupported system: {}".format(system))
