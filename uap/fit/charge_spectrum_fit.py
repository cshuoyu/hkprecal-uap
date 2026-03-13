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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
import zfit
from zfit import z

from . import common_math, fitter_interface
from uap.scan_reader import kor_reader
from uap.tool import scan_prepare


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
        super().__init__(
            obs=obs, params=params, extended=extended, norm=norm, name=name
        )

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
LEGACY_KOR_PED_WINDOW_WIDTH = 2.9
LEGACY_KOR_SPE_WINDOW_WIDTH = 1.3
LEGACY_KOR_PEAK_SEPARATION = 2.3


class ChargeSpectrumFitter(fitter_interface.BaseScanFitter):
    FIT_FIELD_MAP = [
        ("npe", "npe"),
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
        ("pe2_mean", "pe2_mean"),
        ("pe2_sigma", "pe2_sigma"),
        ("pe2_yield", "pe2_yield"),
        ("pe2_yield_err", "pe2_yield_err"),
        ("pe3_mean", "pe3_mean"),
        ("pe3_sigma", "pe3_sigma"),
        ("pe3_yield", "pe3_yield"),
        ("pe3_yield_err", "pe3_yield_err"),
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
        npe=2,
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
        self.npe = 3 if int(npe) == 3 else 2
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

    @staticmethod
    def _window_with_fixed_width(center, width, xmin, xmax):
        xmin = float(xmin)
        xmax = float(xmax)
        span = max(float(xmax - xmin), 1e-6)
        use_width = min(float(width), span)
        half = 0.5 * use_width

        lo = float(center) - half
        hi = float(center) + half
        if lo < xmin:
            hi += xmin - lo
            lo = xmin
        if hi > xmax:
            lo -= hi - xmax
            hi = xmax

        lo = max(lo, xmin)
        hi = min(hi, xmax)
        if hi <= lo:
            return float(xmin), float(xmax)
        return float(lo), float(hi)

    def _estimate_kor_seed_centers(self, data_np, xr):
        xmin, xmax = float(xr[0]), float(xr[1])
        span = max(float(xmax - xmin), 1e-6)

        ped_center = self._hist_peak(data_np, xmin, xmax, nbins=max(self.nbins * 2, 80))
        if not np.isfinite(ped_center):
            ped_center = float(np.nanmedian(data_np))

        spe_search_lo = max(xmin, ped_center + 0.3)
        spe_center = self._hist_peak(
            data_np, spe_search_lo, xmax, nbins=max(self.nbins * 2, 80)
        )

        if not np.isfinite(spe_center) or spe_center <= ped_center:
            fallback_center = ped_center + LEGACY_KOR_PEAK_SEPARATION
            search_lo, search_hi = self._window_with_fixed_width(
                fallback_center,
                max(2.0 * LEGACY_KOR_SPE_WINDOW_WIDTH, 0.35 * span),
                xmin,
                xmax,
            )
            spe_center = self._hist_peak(
                data_np, search_lo, search_hi, nbins=max(self.nbins * 2, 80)
            )

        if not np.isfinite(spe_center) or spe_center <= ped_center:
            upper = data_np[data_np > (ped_center + 0.2)]
            if upper.size > 0:
                spe_center = float(np.nanmedian(upper))
            else:
                spe_center = ped_center + max(0.5, 0.25 * span)

        ped_center = float(np.clip(ped_center, xmin, xmax))
        spe_center = float(np.clip(spe_center, xmin, xmax))
        if spe_center <= ped_center:
            spe_center = float(min(xmax, ped_center + max(0.5, 0.25 * span)))
        return ped_center, spe_center

    def _kor_seed_windows(self, data_np, xr):
        xmin, xmax = float(xr[0]), float(xr[1])
        ped_center, spe_center = self._estimate_kor_seed_centers(data_np, xr)
        ped_min, ped_max = self._window_with_fixed_width(
            ped_center, LEGACY_KOR_PED_WINDOW_WIDTH, xmin, xmax
        )
        spe_min, spe_max = self._window_with_fixed_width(
            spe_center, LEGACY_KOR_SPE_WINDOW_WIDTH, xmin, xmax
        )
        return ped_min, ped_max, spe_min, spe_max

    def _prefit_gaussian_window(self, values, xmin, xmax, coord_key, label):
        arr = self._clip_to_range(values, xmin, xmax)
        span = max(float(xmax - xmin), 1e-6)
        if arr.size == 0:
            center = float(0.5 * (xmin + xmax))
            sigma = max(0.08, 0.15 * span)
            return {
                "mean": center,
                "sigma": sigma,
                "success": False,
                "n_used": 0,
            }

        mu_guess = self._hist_peak(arr, xmin, xmax, nbins=max(self.nbins * 2, 80))
        if not np.isfinite(mu_guess):
            mu_guess = float(np.nanmedian(arr))
        sigma_guess = float(np.nanstd(arr)) if arr.size > 2 else np.nan
        if not np.isfinite(sigma_guess) or sigma_guess <= 0:
            sigma_guess = max(0.08, 0.12 * span)

        mu_lo = max(float(xmin), float(mu_guess - 0.5 * span))
        mu_hi = min(float(xmax), float(mu_guess + 0.5 * span))
        if mu_hi <= mu_lo:
            mu_lo, mu_hi = float(xmin), float(xmax)

        sigma_lo = 1e-4
        sigma_hi = max(float(sigma_guess * 3.0), float(0.5 * span), sigma_lo * 10.0)

        try:
            obs = zfit.Space("x", limits=(float(xmin), float(xmax)))
            data = zfit.Data.from_numpy(obs=obs, array=arr)
            mu = zfit.Parameter(
                "prefit_mu_{}_{}".format(label, coord_key),
                float(np.clip(mu_guess, mu_lo, mu_hi)),
                mu_lo,
                mu_hi,
            )
            sigma = zfit.Parameter(
                "prefit_sigma_{}_{}".format(label, coord_key),
                float(np.clip(sigma_guess, sigma_lo, sigma_hi)),
                sigma_lo,
                sigma_hi,
            )
            model = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
            nll = zfit.loss.UnbinnedNLL(model, data)
            zfit.minimize.Minuit().minimize(nll)
            mu_val = float(zfit.run(mu.value()))
            sigma_val = float(zfit.run(sigma.value()))
            if not np.isfinite(mu_val) or not np.isfinite(sigma_val) or sigma_val <= 0:
                raise RuntimeError("non-finite gaussian prefit result")
            return {
                "mean": mu_val,
                "sigma": sigma_val,
                "success": True,
                "n_used": int(arr.size),
            }
        except Exception:
            return {
                "mean": float(np.clip(mu_guess, xmin, xmax)),
                "sigma": float(np.clip(sigma_guess, 1e-4, max(span, 0.4))),
                "success": False,
                "n_used": int(arr.size),
            }

    def _initial_guesses(self, data_np, xr, coord_key):
        xmin, xmax = float(xr[0]), float(xr[1])
        span = max(xmax - xmin, 1e-6)
        ped_min, ped_max, spe_min, spe_max = self._kor_seed_windows(data_np, xr)

        ped_prefit = self._prefit_gaussian_window(
            data_np, ped_min, ped_max, coord_key, "ped"
        )
        mu_ped = float(ped_prefit["mean"])
        sigma_ped = float(ped_prefit["sigma"])

        spe_prefit = self._prefit_gaussian_window(
            data_np, spe_min, spe_max, coord_key, "spe"
        )
        mu_spe = float(spe_prefit["mean"])
        sigma_spe = float(spe_prefit["sigma"])

        if not np.isfinite(mu_ped):
            mu_ped = float(max(xmin, min(0.0, xmax)))
        if not np.isfinite(mu_spe) or mu_spe <= mu_ped:
            mu_spe = float(min(xmax, max(mu_ped + 0.5, mu_ped + 0.2 * span)))
        if not np.isfinite(sigma_ped) or sigma_ped <= 0:
            sigma_ped = max(0.03 * span, 0.08)
        if not np.isfinite(sigma_spe) or sigma_spe <= 0:
            sigma_spe = max(0.05 * span, 0.12)

        return {
            "mu_ped": float(np.clip(mu_ped, xmin, xmax)),
            "sigma_ped": float(np.clip(sigma_ped, 1e-4, max(0.4 * span, 0.2))),
            "mu_spe": float(np.clip(mu_spe, xmin, xmax)),
            "sigma_spe": float(np.clip(sigma_spe, 1e-4, max(span, 0.4))),
            "ped_window": (float(ped_min), float(ped_max)),
            "spe_window": (float(spe_min), float(spe_max)),
            "ped_prefit_success": bool(ped_prefit["success"]),
            "spe_prefit_success": bool(spe_prefit["success"]),
        }

    def _compute_peak_to_valley(self, values, xr, mu_ped, mu_spe):
        counts, centers, _ = self._histogram(
            values, xr[0], xr[1], max(self.nbins * 2, 80)
        )
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

    @staticmethod
    def _build_npe_shape_params(coord_key, mu_ped, sigma_ped, mu_spe, sigma_spe):
        def pe_charge_step(mu_ped_val, mu_spe_val):
            return mu_spe_val - mu_ped_val

        def sigma_step(sigma_ped_val, sigma_spe_val):
            return z.sqrt(
                tensorflow.maximum(
                    sigma_spe_val * sigma_spe_val - sigma_ped_val * sigma_ped_val,
                    1e-8,
                )
            )

        def pe2_mean(mu_ped_val, mu_spe_val):
            return mu_ped_val + 2.0 * pe_charge_step(mu_ped_val, mu_spe_val)

        def pe3_mean(mu_ped_val, mu_spe_val):
            return mu_ped_val + 3.0 * pe_charge_step(mu_ped_val, mu_spe_val)

        def pe2_sigma(sigma_ped_val, sigma_spe_val):
            sig1 = sigma_step(sigma_ped_val, sigma_spe_val)
            return z.sqrt(
                tensorflow.maximum(
                    sigma_ped_val * sigma_ped_val + 2.0 * sig1 * sig1,
                    1e-8,
                )
            )

        def pe3_sigma(sigma_ped_val, sigma_spe_val):
            sig1 = sigma_step(sigma_ped_val, sigma_spe_val)
            return z.sqrt(
                tensorflow.maximum(
                    sigma_ped_val * sigma_ped_val + 3.0 * sig1 * sig1,
                    1e-8,
                )
            )
        return {
            "pe2_mean": zfit.ComposedParameter(
                "pe2_mean_{}".format(coord_key),
                pe2_mean,
                params=[mu_ped, mu_spe],
            ),
            "pe3_mean": zfit.ComposedParameter(
                "pe3_mean_{}".format(coord_key),
                pe3_mean,
                params=[mu_ped, mu_spe],
            ),
            "pe2_sigma": zfit.ComposedParameter(
                "pe2_sigma_{}".format(coord_key),
                pe2_sigma,
                params=[sigma_ped, sigma_spe],
            ),
            "pe3_sigma": zfit.ComposedParameter(
                "pe3_sigma_{}".format(coord_key),
                pe3_sigma,
                params=[sigma_ped, sigma_spe],
            ),
        }

    @staticmethod
    def _canonicalize_gaussian_roles(out):
        ped_mean = float(out.get("ped_mean", np.nan))
        spe_mean = float(out.get("spe_mean", np.nan))
        if not np.isfinite(ped_mean) or not np.isfinite(spe_mean):
            return False
        if spe_mean >= ped_mean:
            return False

        for suffix in ("mean", "mean_err", "sigma", "sigma_err", "yield", "yield_err"):
            ped_key = "ped_{}".format(suffix)
            spe_key = "spe_{}".format(suffix)
            out[ped_key], out[spe_key] = out.get(spe_key, np.nan), out.get(
                ped_key, np.nan
            )
        return True

    def _make_log_plot(
        self, out_png, centers, counts, yerr, x_model, curves, xr, text_lines
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.errorbar(centers, counts, yerr=yerr, fmt="ok", label="data")
        for curve in curves:
            ax.plot(
                x_model,
                curve["y"],
                linewidth=curve.get("linewidth", 2),
                color=curve.get("color"),
                linestyle=curve.get("linestyle", "-"),
                label=curve["label"],
            )
        ax.set_xlim([float(xr[0]), float(xr[1])])
        ax.set_xlabel("Charge (pC)")
        ax.set_ylabel("Events")
        ax.set_yscale("log")
        positive = np.asarray(counts, dtype=float)
        positive = positive[positive > 0]
        ymin = float(np.min(positive)) * 0.5 if positive.size > 0 else 0.5
        ymax = float(np.max(counts)) if counts.size > 0 else 1.0
        for curve in curves:
            y_vals = np.asarray(curve["y"], dtype=float)
            if y_vals.size > 0:
                ymax = max(ymax, float(np.nanmax(y_vals)))
        ax.set_ylim(max(ymin, 0.5), max(ymax * 1.5, 2.0))
        if text_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )
        ax.legend(loc="best")
        fig.tight_layout()
        target = Path(out_png).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(target))
        plt.close(fig)

    def _make_plot(self, model, data_np, xr, out, plotname, component_specs=None):
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
        text_lines = [
            "npe={}".format(int(out.get("npe", self.npe))),
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
        curves = [
            {
                "label": "total",
                "y": y,
                "color": "tab:blue",
                "linewidth": 2.2,
            }
        ]
        for spec in component_specs or []:
            comp_yield = float(spec.get("yield", np.nan))
            if not np.isfinite(comp_yield) or comp_yield <= 0:
                continue
            comp_pdf = spec.get("pdf")
            if comp_pdf is None:
                continue
            comp_y = (
                np.asarray(zfit.run(comp_pdf.pdf(x)), dtype=float)
                * comp_yield
                / float(self.nbins)
                * area
            )
            curves.append(
                {
                    "label": spec.get("label", "component"),
                    "y": comp_y,
                    "color": spec.get("color"),
                    "linewidth": spec.get("linewidth", 1.8),
                    "linestyle": spec.get("linestyle", "--"),
                }
            )

        log_target = self.fig_dir / "{}_log.png".format(name)
        self._make_log_plot(
            log_target,
            centers,
            counts,
            np.sqrt(np.clip(counts, 1, None)),
            x,
            curves,
            xr,
            text_lines,
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

        guesses = self._initial_guesses(data_np, xr, coord_key)
        obs = zfit.Space("x", limits=(xmin, xmax))
        data = zfit.Data.from_numpy(obs=obs, array=data_np)
        size = int(data_np.shape[0])

        mu_ped_guess = guesses["mu_ped"]
        sigma_ped_guess = guesses["sigma_ped"]
        mu_spe_guess = guesses["mu_spe"]
        sigma_spe_guess = guesses["sigma_spe"]
        ped_window = guesses.get("ped_window", (xmin, xmax))
        spe_window = guesses.get("spe_window", (xmin, xmax))

        ped_mu_lo = max(float(xmin), float(mu_ped_guess - 2.0 * sigma_ped_guess))
        ped_mu_hi = min(float(xmax), float(mu_ped_guess + 2.0 * sigma_ped_guess))
        if ped_mu_hi <= ped_mu_lo:
            ped_mu_lo = max(float(xmin), float(mu_ped_guess - 0.25 * span))
            ped_mu_hi = min(float(xmax), float(mu_ped_guess + 0.25 * span))

        spe_mu_lo = max(
            float(spe_window[0]),
            float(mu_spe_guess - 3.0 * sigma_spe_guess),
            float(mu_ped_guess + 0.05),
        )
        spe_mu_hi = min(float(xmax), float(mu_spe_guess + 3.0 * sigma_spe_guess))
        if spe_mu_hi <= spe_mu_lo:
            spe_mu_lo = max(float(mu_ped_guess + 0.05), float(spe_window[0]))
            spe_mu_hi = min(float(xmax), float(spe_window[1]))
        if spe_mu_hi <= spe_mu_lo:
            spe_mu_hi = min(float(xmax), float(spe_mu_lo + max(0.3, 0.15 * span)))

        sigma_ped_lo = max(1e-4, float(0.5 * sigma_ped_guess))
        sigma_ped_hi = max(
            sigma_ped_lo * 1.2, float(np.sqrt(2.0) * sigma_ped_guess), 0.1
        )
        sigma_spe_lo = max(1e-4, float(0.5 * sigma_spe_guess))
        sigma_spe_hi = max(sigma_spe_lo * 1.2, float(2.0 * sigma_spe_guess), 0.15)

        mu_ped = zfit.Parameter(
            "mu_ped_{}".format(coord_key),
            mu_ped_guess,
            ped_mu_lo,
            ped_mu_hi,
        )
        sigma_ped = zfit.Parameter(
            "sigma_ped_{}".format(coord_key),
            sigma_ped_guess,
            sigma_ped_lo,
            sigma_ped_hi,
        )
        mu_spe = zfit.Parameter(
            "mu_spe_{}".format(coord_key),
            float(np.clip(mu_spe_guess, spe_mu_lo, spe_mu_hi)),
            spe_mu_lo,
            spe_mu_hi,
        )
        sigma_spe = zfit.Parameter(
            "sigma_spe_{}".format(coord_key),
            sigma_spe_guess,
            sigma_spe_lo,
            sigma_spe_hi,
        )
        npe_shapes = self._build_npe_shape_params(
            coord_key, mu_ped, sigma_ped, mu_spe, sigma_spe
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
        pe2_yield = None
        pe2_pdf = None
        pe3_yield = None
        pe3_pdf = None
        if self.npe >= 2:
            pe2_pdf = zfit.pdf.Gauss(
                obs=obs,
                mu=npe_shapes["pe2_mean"],
                sigma=npe_shapes["pe2_sigma"],
            )
            pe2_yield = zfit.Parameter(
                "pe2_yield_{}".format(coord_key),
                max(size * 0.08, 1.0),
                0.0,
                max(size * 0.8, 2.0),
                step_size=1,
            )
            components.append(pe2_pdf.create_extended(pe2_yield))
        if self.npe >= 3:
            pe3_pdf = zfit.pdf.Gauss(
                obs=obs,
                mu=npe_shapes["pe3_mean"],
                sigma=npe_shapes["pe3_sigma"],
            )
            pe3_yield = zfit.Parameter(
                "pe3_yield_{}".format(coord_key),
                max(size * 0.03, 1.0),
                0.0,
                max(size * 0.6, 2.0),
                step_size=1,
            )
            components.append(pe3_pdf.create_extended(pe3_yield))
        bs_yield = None
        bs_pdf = None
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
            "npe": int(self.npe),
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
            "pe2_mean": (
                float(zfit.run(npe_shapes["pe2_mean"].value()))
                if self.npe >= 2
                else np.nan
            ),
            "pe2_sigma": (
                float(zfit.run(npe_shapes["pe2_sigma"].value()))
                if self.npe >= 2
                else np.nan
            ),
            "pe2_yield": (
                float(zfit.run(pe2_yield.value())) if pe2_yield is not None else np.nan
            ),
            "pe2_yield_err": (
                self._extract_err(hesse, pe2_yield) if pe2_yield is not None else np.nan
            ),
            "pe3_mean": (
                float(zfit.run(npe_shapes["pe3_mean"].value()))
                if self.npe >= 3
                else np.nan
            ),
            "pe3_sigma": (
                float(zfit.run(npe_shapes["pe3_sigma"].value()))
                if self.npe >= 3
                else np.nan
            ),
            "pe3_yield": (
                float(zfit.run(pe3_yield.value())) if pe3_yield is not None else np.nan
            ),
            "pe3_yield_err": (
                self._extract_err(hesse, pe3_yield) if pe3_yield is not None else np.nan
            ),
            "backscatter_yield": (
                float(zfit.run(bs_yield.value())) if bs_yield is not None else np.nan
            ),
            "backscatter_yield_err": (
                self._extract_err(hesse, bs_yield) if bs_yield is not None else np.nan
            ),
        }
        swapped_roles = self._canonicalize_gaussian_roles(out)
        out["total_yield"] = float(
            np.nansum(
                [
                    out["ped_yield"],
                    out["spe_yield"],
                    out["pe2_yield"],
                    out["pe3_yield"],
                    out["backscatter_yield"],
                ]
            )
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

        component_specs = [
            {
                "label": "pedestal",
                "pdf": spe_pdf if swapped_roles else ped_pdf,
                "yield": out["ped_yield"],
                "color": "tab:orange",
            },
            {
                "label": "SPE",
                "pdf": ped_pdf if swapped_roles else spe_pdf,
                "yield": out["spe_yield"],
                "color": "tab:green",
            },
        ]
        if pe2_pdf is not None:
            component_specs.append(
                {
                    "label": "2PE",
                    "pdf": pe2_pdf,
                    "yield": out["pe2_yield"],
                    "color": "tab:purple",
                }
            )
        if pe3_pdf is not None:
            component_specs.append(
                {
                    "label": "3PE",
                    "pdf": pe3_pdf,
                    "yield": out["pe3_yield"],
                    "color": "tab:brown",
                }
            )
        if bs_pdf is not None:
            component_specs.append(
                {
                    "label": "backscatter",
                    "pdf": bs_pdf,
                    "yield": out["backscatter_yield"],
                    "color": "tab:red",
                }
            )

        self._make_plot(model, data_np, xr, out, request.plotname, component_specs)
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
                finite_mask = np.isfinite(g0_vals)
                if not np.any(finite_mask):
                    logger.warning(
                        "[KOR] no finite center gain for phi_raw=%s, keep relative_gain as NaN",
                        phi_value,
                    )
                    continue
                g0 = float(np.nanmean(g0_vals[finite_mask]))
                g0_err_vals = pd.to_numeric(
                    center_rows["gain_err"], errors="coerce"
                ).values
                n_err = int(np.sum(np.isfinite(g0_err_vals)))
                g0_err = (
                    float(np.sqrt(np.nansum(g0_err_vals**2)) / n_err)
                    if n_err > 0
                    else np.nan
                )

            out_df.loc[idx, "relative_gain"] = common_math.safe_ratio(
                out_df.loc[idx, "gain"].values, g0, logger=logger
            )
            out_df.loc[idx, "relative_gain_err"] = common_math.safe_ratio_err(
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
                lambda: kor_reader.read_tree_branch(fp, ctx["channel"], charge_branch),
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
                "npe": int(self.npe),
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
                        coord=(
                            "kor_charge",
                            args.serial,
                            int(phi),
                            int(theta_raw),
                            idx,
                        ),
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
