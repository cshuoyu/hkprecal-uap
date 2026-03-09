"""
zfit charge-spectrum fitter for charge-based analyses.

Current implementation:
- KOR fixed method: pedestal Gaussian + SPE Gaussian + optional backscatter
- AUS: interface reserved, not implemented yet
"""

import logging
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ZFIT_DISABLE_TF_WARNINGS", "1")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
import zfit
from zfit import z

from .common_math import safe_ratio, safe_ratio_err
from .fitter_interface import BaseScanFitter
from .plot_utils import save_fit_with_pull_plot
from uap.scan_reader.kor_reader import (
    auto_pick_channel,
    auto_pick_trigger_channel,
    check_serial_order_consistency,
    extract_serial_block_angles,
    read_tree_branch,
)
from uap.tool.scan_prepare import resolve_kor_channels


logger = logging.getLogger(__name__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    zfit.settings.changed_warnings.hesse_name = False
except Exception:
    pass


class KORBackscatterPDF(zfit.pdf.BasePDF):
    def __init__(
        self,
        obs,
        mu_ped,
        sigma_ped,
        mu_spe,
        sigma_spe,
        extended=None,
        norm=None,
        name=None,
    ):
        params = {
            "mu_ped": mu_ped,
            "sigma_ped": sigma_ped,
            "mu_spe": mu_spe,
            "sigma_spe": sigma_spe,
        }
        super().__init__(obs=obs, params=params, extended=extended, norm=norm, name=name)

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        mu_ped = self.params["mu_ped"]
        sigma_ped = self.params["sigma_ped"]
        mu_spe = self.params["mu_spe"]
        sigma_spe = self.params["sigma_spe"]

        eps = tensorflow.constant(1e-6, dtype=x.dtype)
        sigma_ped = tensorflow.maximum(sigma_ped, eps)
        sigma_spe = tensorflow.maximum(sigma_spe, eps)

        shape = tensorflow.math.erf((x - mu_ped) / sigma_ped) - tensorflow.math.erf(
            (x - mu_spe) / sigma_spe
        )
        return tensorflow.maximum(shape, tensorflow.zeros_like(shape))


E_CHARGE_PC = 1.602176634e-7
CHARGE_METHOD_NAME = "fitandplot_kor_charge"


class ChargeSpectrumFitter(BaseScanFitter):
    FIT_FIELD_MAP = [
        ("ped_mean", "ped_mean"),
        ("ped_mean_err", "ped_mean_err"),
        ("ped_sigma", "ped_sigma"),
        ("ped_sigma_err", "ped_sigma_err"),
        ("ped_yield", "ped_yield"),
        ("ped_yield_err", "ped_yield_err"),
        ("spe_mean", "spe_mean"),
        ("spe_mean_err", "spe_mean_err"),
        ("spe_sigma", "spe_sigma"),
        ("spe_sigma_err", "spe_sigma_err"),
        ("spe_yield", "spe_yield"),
        ("spe_yield_err", "spe_yield_err"),
        ("backscatter_yield", "backscatter_yield"),
        ("backscatter_yield_err", "backscatter_yield_err"),
        ("total_yield", "total_yield"),
        ("gain", "gain"),
        ("gain_err", "gain_err"),
        ("resolution", "resolution"),
        ("peak_to_valley", "peak_to_valley"),
    ]

    def __init__(
        self,
        method_name=CHARGE_METHOD_NAME,
        fig_dir=None,
        nbins=50,
        inc_backscatter=True,
        charge_branch="auto",
        charge_qmin=None,
        charge_qmax=None,
        auto_qmin=0.0005,
        auto_qmax=0.9995,
        min_events=50,
    ):
        self.method_name = str(method_name or CHARGE_METHOD_NAME).strip()
        self.nbins = int(nbins)
        self.inc_backscatter = bool(inc_backscatter)
        self.charge_branch = str(charge_branch or "auto")
        self.charge_qmin = charge_qmin
        self.charge_qmax = charge_qmax
        self.auto_qmin = float(np.clip(auto_qmin, 0.0, 1.0))
        self.auto_qmax = float(np.clip(auto_qmax, 0.0, 1.0))
        self.min_events = max(int(min_events), 1)
        self.fig_dir = Path(fig_dir).resolve() if fig_dir else None
        if self.fig_dir:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, request):
        if self.method_name == CHARGE_METHOD_NAME:
            return self._fit_kor_charge(request)
        raise RuntimeError(
            "Unsupported built-in charge fit method: {}".format(self.method_name)
        )

    @staticmethod
    def _coord_key(coord):
        if isinstance(coord, (tuple, list)):
            txt = "_".join([str(x) for x in coord])
        else:
            txt = str(coord)
        txt = re.sub(r"[^0-9A-Za-z_]+", "_", txt).strip("_")
        return txt or "coord"

    @staticmethod
    def _extract_err(hesse, param):
        if hesse is None or param is None:
            return np.nan
        try:
            entry = hesse[param]
            if isinstance(entry, dict):
                return float(entry.get("error", np.nan))
            return float(getattr(entry, "error", np.nan))
        except Exception:
            return np.nan

    @staticmethod
    def _finite_1d(values):
        arr = np.asarray(values).reshape(-1)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _clip_to_range(values, xmin, xmax):
        arr = ChargeSpectrumFitter._finite_1d(values)
        return arr[(arr >= float(xmin)) & (arr <= float(xmax))]

    @staticmethod
    def _histogram(values, xmin, xmax, nbins):
        counts, edges = np.histogram(
            values, bins=max(int(nbins), 20), range=(float(xmin), float(xmax))
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        return counts.astype(float), centers.astype(float), edges.astype(float)

    @staticmethod
    def _hist_peak(values, xmin, xmax, nbins=120):
        arr = ChargeSpectrumFitter._clip_to_range(values, xmin, xmax)
        if arr.size == 0:
            return np.nan
        counts, centers, _ = ChargeSpectrumFitter._histogram(arr, xmin, xmax, nbins)
        if counts.size == 0 or np.max(counts) <= 0:
            return float(np.nanmean(arr))
        return float(centers[int(np.argmax(counts))])

    def _resolve_charge_branch(self, system):
        branch = str(self.charge_branch or "auto").strip()
        if branch and branch.lower() != "auto":
            return branch
        if str(system).lower() == "kor":
            return "pico"
        if str(system).lower() == "aus":
            return "PulseCharge"
        return branch or "value"

    def _resolve_fit_range(self, values):
        arr = self._finite_1d(values)
        if arr.size == 0:
            raise RuntimeError("No finite charge values found.")

        xmin = self.charge_qmin
        xmax = self.charge_qmax
        if xmin is None or xmax is None:
            q_lo = min(self.auto_qmin, self.auto_qmax)
            q_hi = max(self.auto_qmin, self.auto_qmax)
            try:
                auto_lo = float(np.nanquantile(arr, q_lo))
                auto_hi = float(np.nanquantile(arr, q_hi))
            except Exception:
                auto_lo = float(np.nanmin(arr))
                auto_hi = float(np.nanmax(arr))
            if xmin is None:
                xmin = auto_lo
            if xmax is None:
                xmax = auto_hi

        xmin = float(xmin)
        xmax = float(xmax)
        if xmax <= xmin:
            xmin = float(np.nanmin(arr))
            xmax = float(np.nanmax(arr))

        span = float(xmax - xmin)
        if span <= 0:
            center = float(np.nanmean(arr))
            span = max(abs(center) * 0.25, 1.0)
            xmin = center - 0.5 * span
            xmax = center + 0.5 * span

        peak = self._hist_peak(arr, xmin, xmax, nbins=max(self.nbins * 2, 80))
        return float(xmin), float(xmax), float(peak)

    def _kor_seed_windows(self, xmin, xmax):
        ped_hi = min(float(xmax), 0.9)
        if ped_hi <= xmin:
            ped_hi = xmin + max(0.20 * (xmax - xmin), 0.2)

        spe_lo = max(float(xmin), 1.1)
        spe_hi = min(float(xmax), 2.4)
        if spe_hi <= spe_lo:
            spe_lo = xmin + max(0.35 * (xmax - xmin), 0.3)
            spe_hi = xmin + max(0.70 * (xmax - xmin), 0.8)
        spe_hi = min(spe_hi, xmax)
        return float(xmin), float(ped_hi), float(spe_lo), float(spe_hi)

    def _initial_guesses(self, data_np, xr):
        xmin, xmax = float(xr[0]), float(xr[1])
        span = max(xmax - xmin, 1e-6)
        ped_min, ped_max, spe_min, spe_max = self._kor_seed_windows(xmin, xmax)

        ped_arr = self._clip_to_range(data_np, ped_min, ped_max)
        if ped_arr.size == 0:
            ped_arr = self._clip_to_range(data_np, xmin, xmin + 0.35 * span)
        mu_ped = self._hist_peak(ped_arr, ped_min, ped_max, nbins=max(self.nbins * 2, 80))
        if not np.isfinite(mu_ped):
            mu_ped = float(np.nanmedian(ped_arr)) if ped_arr.size > 0 else 0.0
        sigma_ped = float(np.nanstd(ped_arr)) if ped_arr.size > 2 else np.nan
        if not np.isfinite(sigma_ped) or sigma_ped <= 0:
            sigma_ped = max(0.03 * span, 0.08)

        spe_arr = self._clip_to_range(data_np, spe_min, spe_max)
        if spe_arr.size == 0:
            spe_arr = self._clip_to_range(data_np, mu_ped + 0.15 * span, xmax)
        mu_spe = self._hist_peak(spe_arr, spe_min, spe_max, nbins=max(self.nbins * 2, 80))
        if not np.isfinite(mu_spe):
            mu_spe = float(np.nanmedian(spe_arr)) if spe_arr.size > 0 else mu_ped + 0.5 * span
        sigma_spe = float(np.nanstd(spe_arr)) if spe_arr.size > 2 else np.nan
        if not np.isfinite(sigma_spe) or sigma_spe <= 0:
            sigma_spe = max(0.05 * span, 0.12)

        return {
            "mu_ped": float(np.clip(mu_ped, xmin, xmax)),
            "sigma_ped": float(np.clip(sigma_ped, 1e-4, max(0.4 * span, 0.2))),
            "mu_spe": float(np.clip(mu_spe, xmin, xmax)),
            "sigma_spe": float(np.clip(sigma_spe, 1e-4, max(span, 0.4))),
        }

    def _compute_peak_to_valley(self, values, xr, mu_ped, mu_spe):
        counts, centers, _ = self._histogram(values, xr[0], xr[1], max(self.nbins * 2, 80))
        if counts.size == 0:
            return np.nan
        left = min(float(mu_ped), float(mu_spe))
        right = max(float(mu_ped), float(mu_spe))
        peak_mask = centers >= float(mu_spe - 0.15 * (xr[1] - xr[0]))
        peak_mask &= centers <= float(mu_spe + 0.15 * (xr[1] - xr[0]))
        valley_mask = (centers >= left) & (centers <= right)
        if not np.any(peak_mask) or not np.any(valley_mask):
            return np.nan
        peak_height = float(np.nanmax(counts[peak_mask]))
        valley_positive = counts[valley_mask]
        valley_positive = valley_positive[valley_positive > 0]
        if valley_positive.size == 0:
            return np.nan
        valley_height = float(np.nanmin(valley_positive))
        if valley_height <= 0:
            return np.nan
        return peak_height / valley_height

    def _make_log_plot(self, out_png, centers, counts, yerr, x_model, y_model, xr):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.errorbar(centers, counts, yerr=yerr, fmt="ok", label="data")
        ax.plot(x_model, y_model, linewidth=2, label="model")
        ax.set_xlim([float(xr[0]), float(xr[1])])
        ax.set_xlabel("Charge (pC)")
        ax.set_ylabel("Events")
        ax.set_yscale("log")
        positive = np.asarray(counts, dtype=float)
        positive = positive[positive > 0]
        ymin = float(np.min(positive)) * 0.5 if positive.size > 0 else 0.5
        ymax = max(
            float(np.max(counts)) if counts.size > 0 else 1.0,
            float(np.max(y_model)) if y_model.size > 0 else 1.0,
        )
        ax.set_ylim(max(ymin, 0.5), max(ymax * 1.5, 2.0))
        ax.legend(loc="best")
        fig.tight_layout()
        target = Path(out_png).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(target))
        plt.close(fig)

    def _make_plot(self, model, data_np, xr, out, plotname):
        if self.fig_dir is None:
            return

        total_yield = float(out.get("total_yield", np.nan))
        if not np.isfinite(total_yield) or total_yield <= 0:
            total_yield = float(np.asarray(data_np).size)

        x = np.linspace(float(xr[0]), float(xr[1]), 1200)
        y_model = np.asarray(zfit.run(model.pdf(x)), dtype=float)
        area = float(xr[1] - xr[0])
        y = y_model * total_yield / float(self.nbins) * area

        counts, edges = np.histogram(
            data_np, bins=self.nbins, range=(float(xr[0]), float(xr[1]))
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        y_exp = (
            np.asarray(zfit.run(model.pdf(centers)), dtype=float)
            * total_yield
            / float(self.nbins)
            * area
        )
        pull = (counts - y_exp) / np.sqrt(np.clip(counts, 1, None))

        text_lines = [
            "mu_spe={:.4g}".format(out.get("spe_mean", np.nan)),
            "sigma_spe={:.4g}".format(out.get("spe_sigma", np.nan)),
            "gain={:.4g}".format(out.get("gain", np.nan)),
            "res={:.3g}%".format(out.get("resolution", np.nan)),
        ]
        if np.isfinite(out.get("backscatter_yield", np.nan)):
            text_lines.append("bs={:.4g}".format(out.get("backscatter_yield", np.nan)))
        if np.isfinite(out.get("peak_to_valley", np.nan)):
            text_lines.append("P/V={:.3g}".format(out.get("peak_to_valley", np.nan)))

        name = (plotname or "kor_charge_fit").strip()
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
            x_label="Charge (pC)",
            pull_ylim=(-5, 5),
        )
        logger.info("[PLOT] saved %s", target)
        if self.inc_backscatter:
            log_target = self.fig_dir / "{}_log.png".format(name)
            self._make_log_plot(
                log_target,
                centers,
                counts,
                np.sqrt(np.clip(counts, 1, None)),
                x,
                y,
                xr,
            )
            logger.info("[PLOT] saved %s", log_target)

    def _fit_kor_charge(self, request):
        data_np = self._finite_1d(request.data)
        xr = request.xr
        if xr is None:
            raise RuntimeError("Charge fit requires explicit fit range.")
        data_np = self._clip_to_range(data_np, xr[0], xr[1])
        if data_np.size == 0:
            raise RuntimeError("No finite charge values inside fit range.")

        coord_key = self._coord_key(request.coord)
        xmin = float(xr[0])
        xmax = float(xr[1])
        span = float(xmax - xmin)
        if xmax <= xmin:
            raise RuntimeError("Invalid fit range xr={}".format(xr))

        guesses = self._initial_guesses(data_np, xr)
        obs = zfit.Space("x", limits=(xmin, xmax))
        data = zfit.Data.from_numpy(obs=obs, array=data_np)
        size = int(data_np.shape[0])

        mu_ped_guess = guesses["mu_ped"]
        sigma_ped_guess = guesses["sigma_ped"]
        mu_spe_guess = guesses["mu_spe"]
        sigma_spe_guess = guesses["sigma_spe"]

        mu_ped = zfit.Parameter(
            "mu_ped_{}".format(coord_key),
            mu_ped_guess,
            max(xmin, mu_ped_guess - 0.25 * span),
            min(xmax, mu_ped_guess + 0.25 * span),
        )
        sigma_ped = zfit.Parameter(
            "sigma_ped_{}".format(coord_key),
            sigma_ped_guess,
            1e-4,
            max(0.5 * span, 0.3),
        )
        mu_spe = zfit.Parameter(
            "mu_spe_{}".format(coord_key),
            max(mu_spe_guess, mu_ped_guess + 0.05),
            max(mu_ped_guess, xmin),
            xmax,
        )
        sigma_spe = zfit.Parameter(
            "sigma_spe_{}".format(coord_key),
            sigma_spe_guess,
            1e-4,
            max(span, 0.5),
        )

        ped_pdf = zfit.pdf.Gauss(obs=obs, mu=mu_ped, sigma=sigma_ped)
        ped_yield = zfit.Parameter(
            "ped_yield_{}".format(coord_key),
            max(size * 0.5, 1.0),
            0.0,
            max(size * 1.2, 2.0),
            step_size=1,
        )
        ped_ext = ped_pdf.create_extended(ped_yield)

        spe_pdf = zfit.pdf.Gauss(obs=obs, mu=mu_spe, sigma=sigma_spe)
        spe_yield = zfit.Parameter(
            "spe_yield_{}".format(coord_key),
            max(size * 0.3, 1.0),
            0.0,
            max(size * 1.2, 2.0),
            step_size=1,
        )
        spe_ext = spe_pdf.create_extended(spe_yield)

        components = [ped_ext, spe_ext]
        bs_yield = None
        if self.inc_backscatter:
            bs_pdf = KORBackscatterPDF(
                obs=obs,
                mu_ped=mu_ped,
                sigma_ped=sigma_ped,
                mu_spe=mu_spe,
                sigma_spe=sigma_spe,
            )
            bs_yield = zfit.Parameter(
                "bs_yield_{}".format(coord_key),
                max(size * 0.05, 0.0),
                0.0,
                max(size * 0.8, 2.0),
                step_size=1,
            )
            components.append(bs_pdf.create_extended(bs_yield))

        model = zfit.pdf.SumPDF(components)
        nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
        minimizer = zfit.minimize.Minuit()
        result = minimizer.minimize(nll)
        try:
            hesse = result.hesse()
        except Exception:
            hesse = None

        out = {
            "ped_mean": float(zfit.run(mu_ped.value())),
            "ped_mean_err": self._extract_err(hesse, mu_ped),
            "ped_sigma": float(zfit.run(sigma_ped.value())),
            "ped_sigma_err": self._extract_err(hesse, sigma_ped),
            "ped_yield": float(zfit.run(ped_yield.value())),
            "ped_yield_err": self._extract_err(hesse, ped_yield),
            "spe_mean": float(zfit.run(mu_spe.value())),
            "spe_mean_err": self._extract_err(hesse, mu_spe),
            "spe_sigma": float(zfit.run(sigma_spe.value())),
            "spe_sigma_err": self._extract_err(hesse, sigma_spe),
            "spe_yield": float(zfit.run(spe_yield.value())),
            "spe_yield_err": self._extract_err(hesse, spe_yield),
            "backscatter_yield": (
                float(zfit.run(bs_yield.value())) if bs_yield is not None else np.nan
            ),
            "backscatter_yield_err": (
                self._extract_err(hesse, bs_yield) if bs_yield is not None else np.nan
            ),
        }
        out["total_yield"] = float(
            np.nansum([out["ped_yield"], out["spe_yield"], out["backscatter_yield"]])
        )
        out["gain"] = float(out["spe_mean"] / E_CHARGE_PC)
        out["gain_err"] = (
            float(out["spe_mean_err"] / E_CHARGE_PC)
            if np.isfinite(out["spe_mean_err"])
            else np.nan
        )
        out["resolution"] = (
            float(out["spe_sigma"] / out["spe_mean"] * 100.0)
            if np.isfinite(out["spe_mean"]) and out["spe_mean"] != 0
            else np.nan
        )
        out["peak_to_valley"] = self._compute_peak_to_valley(
            data_np, xr, out["ped_mean"], out["spe_mean"]
        )

        self._make_plot(model, data_np, xr, out, request.plotname)
        return out

    @staticmethod
    def _is_abnormal_main_fit(fit_out):
        checks = [
            fit_out.get("ped_mean", np.nan),
            fit_out.get("ped_sigma", np.nan),
            fit_out.get("spe_mean", np.nan),
            fit_out.get("spe_sigma", np.nan),
            fit_out.get("gain", np.nan),
        ]
        if not np.all(np.isfinite(np.asarray(checks, dtype=float))):
            return True
        if float(fit_out.get("ped_sigma", 0.0)) <= 0:
            return True
        if float(fit_out.get("spe_mean", 0.0)) <= 0:
            return True
        if float(fit_out.get("spe_sigma", 0.0)) <= 0:
            return True
        if float(fit_out.get("spe_mean", 0.0)) <= float(fit_out.get("ped_mean", 0.0)):
            return True
        return False

    def _apply_kor_relative_columns(self, df):
        out_df = df.copy()
        out_df["relative_gain"] = np.nan
        out_df["relative_gain_err"] = np.nan

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
                g0 = float(center_row.get("gain", np.nan))
                g0_err = float(center_row.get("gain_err", np.nan))
            else:
                g0_vals = pd.to_numeric(center_rows["gain"], errors="coerce").values
                g0 = float(np.nanmean(g0_vals))
                g0_err_vals = pd.to_numeric(
                    center_rows["gain_err"], errors="coerce"
                ).values
                n_err = int(np.sum(np.isfinite(g0_err_vals)))
                g0_err = (
                    float(np.sqrt(np.nansum(g0_err_vals ** 2)) / n_err)
                    if n_err > 0
                    else np.nan
                )

            out_df.loc[idx, "relative_gain"] = safe_ratio(
                out_df.loc[idx, "gain"].values, g0, logger=logger
            )
            out_df.loc[idx, "relative_gain_err"] = safe_ratio_err(
                out_df.loc[idx, "gain"].values,
                out_df.loc[idx, "gain_err"].values,
                g0,
                g0_err,
                logger=logger,
            )
        return out_df

    def _prepare_aus_input(self, args):
        raise RuntimeError(
            "ChargeSpectrumFitter AUS interface is reserved but not implemented yet."
        )

    def _prepare_kor_input(self, args):
        _input_dir, out_csv, files = self.resolve_inputs(
            args=args,
            default_out_csv="csv/kor_charge_results.csv",
            file_pattern="prd_*.root",
            empty_msg="No prd_*.root found in {}",
        )

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

        charge_branch = self._resolve_charge_branch("kor")
        prep_stats = {
            "files_scanned": int(len(selected)),
            "files_kept": 0,
            "charge_read_fail": 0,
            "empty_fit_range": 0,
            "too_few_events": 0,
        }
        points = []

        for idx, (fp, phi, theta_raw) in enumerate(selected):
            charge = self.run_step(
                lambda: read_tree_branch(fp, ctx["channel"], charge_branch),
                stats=prep_stats,
                fail_key="charge_read_fail",
                file_name=fp.name,
                fail_msg="read KOR charge branch failed",
            )
            if charge is None:
                continue

            charge = self._finite_1d(charge)
            if charge.size == 0:
                self.inc_stat(prep_stats, "empty_fit_range")
                continue

            try:
                use_qmin, use_qmax, peak = self._resolve_fit_range(charge)
            except Exception:
                self.inc_stat(prep_stats, "empty_fit_range")
                continue

            fit_values = self._clip_to_range(charge, use_qmin, use_qmax)
            if fit_values.size < self.min_events:
                self.inc_stat(prep_stats, "too_few_events")
                self.log_skip(
                    fp.name,
                    "charge fit skipped after range selection: n={} < min_events={}".format(
                        fit_values.size, self.min_events
                    ),
                )
                continue

            row = {
                "system": "kor",
                "file": fp.name,
                "serial": str(args.serial),
                "channel": int(ctx["channel"]),
                "trigger_ch": int(ctx["trigger_ch"]),
                "channel_cfg": str(ctx["channel_cfg"]),
                "trigger_ch_cfg": str(ctx["trigger_ch_cfg"]),
                "phi_raw": int(phi),
                "theta_raw": int(theta_raw),
                "charge_method": CHARGE_METHOD_NAME,
                "charge_branch": str(charge_branch),
                "include_backscatter": bool(self.inc_backscatter),
                "charge_range_min": float(use_qmin),
                "charge_range_max": float(use_qmax),
                "charge_peak": float(peak),
            }

            points.append(
                self.make_point(
                    row=row,
                    main_fit_input=self.make_fit_input(
                        data=fit_values,
                        coord=("kor_charge", args.serial, int(phi), int(theta_raw), idx),
                        plotname="kor_charge_{}_phi{}_theta{}_{}".format(
                            args.serial, phi, theta_raw, idx
                        ),
                        xr=(use_qmin, use_qmax),
                        meta={"n_in_window": int(fit_values.size)},
                    ),
                    main_skip_msg="[SKIP] {}: KOR charge fit failed.".format(fp.name),
                )
            )
            prep_stats["files_kept"] += 1

        return {"out_csv": out_csv, "points": points, "prep_stats": prep_stats}

    def prepare_scan(self, system, args):
        if system == "aus":
            return self._prepare_aus_input(args)

        if system == "kor":
            inputs = self._prepare_kor_input(args)
            return {
                "out_csv": inputs["out_csv"],
                "points": inputs["points"],
                "prep_stats": inputs.get("prep_stats", {}),
                "sort_cols": ["phi_raw", "theta_raw", "file"],
                "empty_msg": "No valid KOR charge-fit results produced.",
                "postprocess": self._apply_kor_relative_columns,
            }

        raise RuntimeError("Unsupported system: {}".format(system))
