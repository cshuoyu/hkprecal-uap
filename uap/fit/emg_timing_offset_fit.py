"""EMG timing fits using NumPy/SciPy and iminuit.

The analysis method follows https://github.com/wihann00/HKAus_precal_analysis
(Author: Wi Han Ng). Signal/background yields refer to the selected time window.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import exponnorm

from . import common_math, fitter_interface, plot_utils
from .fit_config import timing_configuration
from uap.scan_reader import aus_reader, kor_reader
from uap.tool import scan_prepare, window


logger = logging.getLogger(__name__)
SAMPLE_NS = 2.0


def emg_pdf(x, mu, lambd, sigma, limits):
    """EMG normalized inside limits; mu is the Gaussian location, not its mean."""
    shape = 1. / (sigma * lambd)
    lo, hi = limits
    cdf = exponnorm.cdf([lo, hi], shape, loc=mu, scale=sigma)
    if cdf[0] > .5:
        sf = exponnorm.sf([lo, hi], shape, loc=mu, scale=sigma)
        acceptance = sf[0] - sf[1]
    else:
        acceptance = cdf[1] - cdf[0]
    if not np.isfinite(acceptance) or acceptance <= 0:
        return np.full_like(np.asarray(x, dtype=float), np.nan)
    # SciPy evaluates the EMG in log space, avoiding exp * erfc overflow.
    return np.exp(exponnorm.logpdf(x, shape, loc=mu, scale=sigma) - np.log(acceptance))


def timing_density(x, parameters, limits):
    """Extended EMG plus the existing first-order Chebyshev background."""
    p = parameters
    density = p["sig_yield"] * emg_pdf(x, p["mu"], p["lambd"], p["sigma"], limits)
    if "bkg_yield" in p:
        lo, hi = limits
        scaled_x = 2. * (np.asarray(x) - lo) / (hi - lo) - 1.
        density = density + p["bkg_yield"] * (1. + p["coeff"] * scaled_x) / (hi - lo)
    return density


# Concrete fitter.
class TimingEMGFitter(fitter_interface.BaseScanFitter):
    FIT_FIELD_MAP = fitter_interface.BaseScanFitter.FIT_FIELD_MAP + [
        ("fit_model", "fit_model"),
        ("fit_statistic", "fit_statistic"),
        ("fit_optimizer", "fit_optimizer"),
        ("chi2", "chi2"),
        ("ndf", "ndf"),
        ("objective_value", "objective_value"),
        ("prior_penalty", "prior_penalty"),
        ("fit_converged", "fit_converged"),
        ("fit_covariance_accurate", "fit_covariance_accurate"),
        ("fit_parameters_at_limit", "fit_parameters_at_limit"),
    ]

    # Store fitter options and create figure directory.
    def __init__(self, method_name="timing", fig_dir=None, nbins=30, fit_config=None):
        self.method_name = "timing" if method_name == "fitandplot_emg" else method_name
        self._explicit_config = fit_config is not None
        self.fit_config = timing_configuration(fit_config)
        self.nbins = int(nbins)
        self.fig_dir = Path(fig_dir).resolve() if fig_dir else None
        if self.fig_dir:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

    # Select fit backend by method_name.
    def fit(self, request):
        method = self.method_name
        if method == "timing":
            return self._fit_emg(request)
        raise RuntimeError("Unsupported built-in fit method: {}".format(method))

    # Estimate FWHM from sampled fitted model curve.
    def _compute_fwhm(self, model, xr):
        try:
            x = np.linspace(float(xr[0]), float(xr[1]), 1000)
            y = np.asarray(model(x), dtype=float)

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
                logger.warning(
                    "[FWHM][WARN] insufficient half-maximum crossings, return NaN."
                )
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
            logger.warning(
                "[FWHM][WARN] computation failed: %s: %s. return NaN.",
                type(exc).__name__,
                exc,
            )
            return np.nan

    # Save one diagnostic fit plot (data/model + pull).
    def _make_plot(self, model, data_np, xr, size, out, plotname):
        if self.fig_dir is None:
            return

        x = np.linspace(float(xr[0]), float(xr[1]), 1000)
        y_model = np.asarray(model(x), dtype=float)
        area = float(xr[1] - xr[0])
        y = y_model * float(size) / float(self.nbins) * area

        counts, edges = np.histogram(
            data_np, bins=self.nbins, range=(float(xr[0]), float(xr[1]))
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        y_exp = (
            np.asarray(model(centers), dtype=float)
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
        plot_utils.save_fit_with_pull_plot(
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
        if self._explicit_config:
            inc_bkg = bool(self.fit_config["model"]["background"])

        xr = (
            request.xr
            if request.xr is not None
            else [float(np.min(data_np)), float(np.max(data_np))]
        )
        if xr[1] <= xr[0]:
            raise RuntimeError(
                "Invalid fit range xr={}, data size={}".format(xr, data_np.size)
            )

        data_np = data_np[(data_np >= xr[0]) & (data_np <= xr[1])]
        if data_np.size == 0:
            raise RuntimeError("No finite timing data inside fit range.")
        size = int(data_np.size)

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

        initial = dict(mu=mu_guess, lambd=.1, sigma=float(np.clip(sigma_guess, .2, 20.)))
        limits = dict(mu=(mu_lo, mu_hi), lambd=(.005, 5.), sigma=(.2, 50.))
        if inc_bkg:
            initial.update(sig_yield=size * .8, bkg_yield=max(size * .2, 1.), coeff=0.)
            limits.update(sig_yield=(max(size * .008, 1.), max(size * 1.2, 2.)),
                          bkg_yield=(0., max(size * .5, 2.)), coeff=(-2., 1.))
        else:
            initial["sig_yield"] = float(size)
            limits["sig_yield"] = (0., max(size * 1.2, 2.))

        constraints, statistic = self.fit_config["constraints"], self.fit_config["statistic"]
        physical = dict(mu=(None, None), lambd=(1e-8, None), sigma=(1e-8, None), sig_yield=(0., None))
        if inc_bkg:
            physical.update(bkg_yield=(0., None), coeff=(-1., 1.))
        initial, limits = common_math.apply_constraints(initial, limits, constraints, physical)
        cost = common_math.build_objective(
            data_np, xr, tuple(initial), lambda x, p: timing_density(x, p, xr),
            lambda p: p["sig_yield"] + p.get("bkg_yield", 0.), statistic, constraints["priors"])
        result = common_math.minimize(cost, initial, limits, self.fit_config["optimizer"], constraints["fixed"])
        p = result.values.to_dict()
        errors = common_math.parameter_errors(result)
        out = {
            "mean": p["mu"], "lambda": p["lambd"], "sigma": p["sigma"],
            "mu_err": errors["mu"], "std_err": errors["sigma"],
            "lambd_err": errors["lambd"], "sig_yield": p["sig_yield"],
            "sig_err": errors["sig_yield"], "bkg_yield": p.get("bkg_yield", np.nan),
            "bkg_err": errors["bkg_yield"] if inc_bkg else np.nan,
            **common_math.fit_quality(result),
        }
        if inc_bkg:
            out["coeff"] = p["coeff"]
        penalty = sum(cost.errordef * ((p[name] - mean) / sigma) ** 2
                      for name, (mean, sigma) in constraints["priors"].items())
        out.update(fit_model="emg", fit_statistic=statistic["name"], fit_optimizer="iminuit",
                   objective_value=result.fval, prior_penalty=penalty,
                   chi2=result.fval - penalty if statistic["name"] == "chi2" else np.nan,
                   ndf=cost.ndata - result.nfit if statistic["name"] == "chi2" else np.nan)

        def model(x):
            return timing_density(x, p, xr) / (p["sig_yield"] + p.get("bkg_yield", 0.))

        # Preserve the legacy full-model FWHM, including background.
        out["FWHM"] = self._compute_fwhm(model, xr)
        self._make_plot(model, data_np, xr, size, out, request.plotname)
        return out

    def _attach_main_fit_fields(self, row, fit_out):
        super()._attach_main_fit_fields(row, fit_out)
        if fit_out.get("fit_converged") is False:
            row["fit_status"] = "warning_not_converged"
        elif fit_out.get("fit_covariance_accurate") is False:
            row["fit_status"] = "warning_covariance"

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

        out_df["rel_yield"] = common_math.safe_ratio(
            out_df["sig_yield"].values, sig0, logger=logger
        )
        out_df["rel_yield_err"] = common_math.safe_ratio_err(
            out_df["sig_yield"].values,
            out_df["sig_err"].values,
            sig0,
            sig0_err,
            logger=logger,
        )

        if use_sipm:
            sipm0 = float(center_row.get("sipm_sig_yield", np.nan))
            sipm0_err = float(center_row.get("sipm_sig_err", np.nan))

            out_df["rel_sipm_yield"] = common_math.safe_ratio(
                out_df["sipm_sig_yield"].values, sipm0, logger=logger
            )
            out_df["rel_sipm_yield_err"] = common_math.safe_ratio_err(
                out_df["sipm_sig_yield"].values,
                out_df["sipm_sig_err"].values,
                sipm0,
                sipm0_err,
                logger=logger,
            )

            point_ratio = common_math.safe_ratio(
                out_df["sig_yield"].values,
                out_df["sipm_sig_yield"].values,
                logger=logger,
            )
            point_ratio_err = common_math.safe_ratio_err(
                out_df["sig_yield"].values,
                out_df["sig_err"].values,
                out_df["sipm_sig_yield"].values,
                out_df["sipm_sig_err"].values,
                logger=logger,
            )
            center_ratio = common_math.safe_ratio(sig0, sipm0, logger=logger)
            center_ratio_err = common_math.safe_ratio_err(
                sig0, sig0_err, sipm0, sipm0_err, logger=logger
            )

            out_df["relative_norm"] = common_math.safe_ratio(
                point_ratio, center_ratio, logger=logger
            )
            out_df["relative_norm_err"] = common_math.safe_ratio_err(
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
                sig0_vals = pd.to_numeric(
                    center_rows["sig_yield"], errors="coerce"
                ).values
                sig0 = float(np.nanmean(sig0_vals))

                sig0_err_vals = pd.to_numeric(
                    center_rows["sig_err"], errors="coerce"
                ).values
                n_err = int(np.sum(np.isfinite(sig0_err_vals)))
                if n_err > 0:
                    sig0_err = float(np.sqrt(np.nansum(sig0_err_vals**2)) / n_err)
                else:
                    sig0_err = np.nan

                if len(center_rows) > 1:
                    logger.info(
                        "[KOR] phi_raw=%s has %s theta_raw=0 rows, normalize by mean center.",
                        phi_value,
                        len(center_rows),
                    )

            rel = common_math.safe_ratio(
                out_df.loc[idx, "sig_yield"].values, sig0, logger=logger
            )
            rel_err = common_math.safe_ratio_err(
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

        defaults = dict(aus_reader.DEFAULT_AUS_CHANNELS)
        ctx = scan_prepare.resolve_aus_context(
            args, defaults, aus_reader.resolve_aus_channel
        )

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
        prep_stats = scan_prepare.init_aus_prep_stats(len(files))
        points = []
        for fp in files:
            parsed = aus_reader.parse_theta_phi_aus(fp)
            if parsed is None:
                continue
            theta, phi = parsed

            # Read PMT/trigger timing arrays.
            read_pair = self.run_step(
                lambda: (
                    aus_reader.load_branch(fp, pmt_tree, tbranch) * SAMPLE_NS,
                    aus_reader.load_branch(fp, trigger_tree, tbranch) * SAMPLE_NS,
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
                lambda: window.select_window(
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

            row = scan_prepare.build_aus_row(
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
                    lambda: aus_reader.load_branch(fp, sipm_tree, tbranch) * SAMPLE_NS,
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
                            lambda: window.select_window(
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
            parsed = kor_reader.extract_serial_block_angles(fp.name, args.serial)
            if parsed is None:
                continue
            phi, theta_raw = parsed
            selected.append((fp, int(phi), int(theta_raw)))

        if not selected:
            raise SystemExit("No files matched serial={}.".format(args.serial))

        files_for_auto = [x[0] for x in selected]
        ref_order, mismatches = kor_reader.check_serial_order_consistency(
            files_for_auto
        )
        if mismatches:
            logger.warning(
                "[WARN] serial order mismatch across files. reference=%s mismatched=%s first=%s",
                ref_order,
                len(mismatches),
                mismatches[0][0],
            )

        # Resolve channel and trigger from config/auto mapping.
        ctx = scan_prepare.resolve_kor_channels(
            args=args,
            files_for_auto=files_for_auto,
            parse_auto_or_int_fn=self.parse_auto_or_int,
            auto_pick_trigger_channel_fn=kor_reader.auto_pick_trigger_channel,
            auto_pick_channel_fn=kor_reader.auto_pick_channel,
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
        prep_stats = scan_prepare.init_kor_prep_stats(len(selected))
        points = []

        for idx, (fp, phi, theta_raw) in enumerate(selected):
            # Read KOR diff and convert to ns.
            diff = self.run_step(
                lambda: (
                    kor_reader.read_tree_branch(fp, ctx["channel"], "diff") * SAMPLE_NS
                ),
                stats=prep_stats,
                fail_key="diff_read_fail",
                file_name=fp.name,
                fail_msg="read diff branch failed",
            )
            if diff is None:
                continue

            # Select KOR fit window around peak.
            kor_window = self.run_step(
                lambda: window.select_window(
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

            row = scan_prepare.build_kor_row(
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
