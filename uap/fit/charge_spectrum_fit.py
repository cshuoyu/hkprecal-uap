"""Charge-spectrum models and fits using NumPy/SciPy and iminuit.

The Gaussian-peak and SPE-response models support free or Poisson weights.
Models, constraints, data statistics and optimizers are configured separately.
"""

import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf, ndtr

from . import common_math, fitter_interface, plot_utils
from .fit_config import charge_configuration
from uap.scan_reader import aus_reader, kor_reader
from uap.tool import scan_prepare


logger = logging.getLogger(__name__)


def gaussian_shapes(mu_ped, sigma_ped, mu_spe, sigma_spe, npe):
    """Keep the legacy SPE width and tied 2/3-PE means and widths."""
    n = np.arange(npe + 1)
    means = mu_ped + n * (mu_spe - mu_ped)
    sigmas = np.sqrt(sigma_ped ** 2 + n * max(sigma_spe ** 2 - sigma_ped ** 2, 1e-8))
    sigmas[:2] = sigma_ped, sigma_spe
    return means, sigmas


def poisson_gaussian_yields(amplitude, mu, means, sigmas, limits):
    """Poisson yields inside the fit window, not over the full real line."""
    weights = np.array([np.exp(-mu) * mu ** n / math.factorial(n) for n in range(len(means))])
    return amplitude * weights * common_math.gaussian_acceptance(limits, means, sigmas)


def gaussian_pdf(x, mean, sigma):
    return np.exp(-0.5 * ((np.asarray(x) - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def backscatter_pdf(x, mu_ped, sigma_ped, mu_spe, sigma_spe, limits):
    """Legacy clipped erf-difference, normalized in the selected window."""
    lo, hi = limits
    # erf is monotonic: its difference changes sign at this single crossing.
    if sigma_spe != sigma_ped:
        crossing = (mu_ped * sigma_spe - mu_spe * sigma_ped) / (sigma_spe - sigma_ped)
        if sigma_spe > sigma_ped:
            lo = max(lo, crossing)
        else:
            hi = min(hi, crossing)

    def primitive(q, mean, sigma):
        u = (q - mean) / sigma
        return (q - mean) * erf(u) + sigma / np.sqrt(np.pi) * np.exp(-u * u)

    integral = (primitive(hi, mu_ped, sigma_ped) - primitive(lo, mu_ped, sigma_ped)
                - primitive(hi, mu_spe, sigma_spe) + primitive(lo, mu_spe, sigma_spe))
    if hi <= lo or integral <= 0:
        return np.full_like(np.asarray(x, dtype=float), np.nan)
    shape = erf((np.asarray(x) - mu_ped) / sigma_ped) - erf((np.asarray(x) - mu_spe) / sigma_spe)
    return np.maximum(shape, 0.) / integral


RESPONSE_PARAMETERS = ("amplitude", "mu", "pedestal", "spe_gain", "ped_sigma", "spe_sigma", "back_fraction")


def response_parameters(parameters, npe, poisson=True):
    """Free weights replace amplitude/occupancy without changing any peak shape."""
    if poisson:
        amplitude, mu, ped, gain, sig0, sig1, fraction = parameters
        weights = [amplitude * np.exp(-mu) * mu ** n / math.factorial(n)
                   for n in range(npe + 1)]
    else:
        weights = parameters[:npe + 1]
        ped, gain, sig0, sig1, fraction = parameters[npe + 1:]
    return weights, ped, gain, sig0, sig1, fraction


def spe_response_components(x, parameters, npe, poisson=True):
    """Return full-line Gaussian and backscatter densities, including yields."""
    weights, ped, gain, sig0, sig1, back_fraction = response_parameters(parameters, npe, poisson)
    x = np.asarray(x, dtype=float)
    gaussians, backscatters = [], []
    for n in range(npe + 1):
        mean = ped + n * gain
        sigma = np.sqrt(sig0 ** 2 + n * sig1 ** 2)
        weight = weights[n]
        gaussian = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        gaussians.append(weight * gaussian * (1 - back_fraction if n else 1))
        if n:
            back = (erf((x - ped) / sig0) - erf((x - mean) / sigma)) / (2 * n * gain)
            backscatters.append(weight * back_fraction * back)
    return gaussians, backscatters


def spe_response_model(x, parameters, npe, poisson=True):
    gaussians, backscatters = spe_response_components(x, parameters, npe, poisson)
    return np.sum(gaussians + backscatters, axis=0)


def gaussian_histogram_prefit(counts, centers, limits):
    keep = (centers >= limits[0]) & (centers <= limits[1]) & (counts > 0)
    x, y = centers[keep], counts[keep].astype(float)
    if len(x) < 4:
        raise ValueError("Too few occupied bins for Gaussian histogram prefit")
    mean = np.average(x, weights=y)
    sigma = np.sqrt(np.average((x - mean) ** 2, weights=y))

    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    parameters, _ = curve_fit(
        gaussian, x, y, p0=(y.max(), mean, sigma), sigma=np.sqrt(y),
        absolute_sigma=True, maxfev=20000,
    )
    return float(parameters[1]), float(abs(parameters[2]))


def pulse_height_initial_parameters(counts, selected_counts, centers):
    ped, sig0 = gaussian_histogram_prefit(counts, centers, (-2.0, 0.9))
    peak = centers[np.argmax(selected_counts)]
    width = max(peak * 0.5, 2.5)
    spe, sigma = gaussian_histogram_prefit(selected_counts, centers, (peak - width, peak + width))
    width = sigma * 0.5
    if width < 2.0:
        width = 2.5
    spe, sigma = gaussian_histogram_prefit(selected_counts, centers, (spe - width, spe + width))
    gain = spe - ped
    if gain <= 0 or sig0 <= 0:
        raise ValueError("Charge prefit did not resolve a positive SPE charge step")
    if sigma > gain * 0.4:
        sigma = gain * 0.35
    return ped, sig0, gain, sigma


def spe_response_yields(parameters, limits, npe, poisson=True):
    """Integrals in the selected charge range; not full-spectrum event counts."""
    weights, ped, gain, sig0, sig1, fraction = response_parameters(parameters, npe, poisson)
    lo, hi = limits
    gaussians, back = [], 0.0

    def erf_integral(x, mean, sigma):
        u = (x - mean) / sigma
        return (x - mean) * erf(u) + sigma / np.sqrt(np.pi) * np.exp(-u * u)

    for n in range(npe + 1):
        mean, sigma = ped + n * gain, np.sqrt(sig0 ** 2 + n * sig1 ** 2)
        weight = weights[n]
        acceptance = ndtr((hi - mean) / sigma) - ndtr((lo - mean) / sigma)
        gaussians.append(weight * acceptance * (1 - fraction if n else 1))
        if n:
            integral = (erf_integral(hi, ped, sig0) - erf_integral(lo, ped, sig0)
                        - erf_integral(hi, mean, sigma) + erf_integral(lo, mean, sigma)) / (2 * n * gain)
            back += weight * fraction * integral
    return np.array(gaussians + [back])


E_CHARGE_PC = 1.602176634e-7
CHARGE_METHOD_NAME = "charge"
# Accepted only for historical configs; models never depend on acquisition system.
CHARGE_METHOD_NAMES = (CHARGE_METHOD_NAME, "fitandplot_kor_charge", "fitandplot_aus_charge")
SAMPLE_NS = 2.0
SPECTRUM_PED_WINDOW_WIDTH = 2.9
SPECTRUM_SPE_WINDOW_WIDTH = 1.3
SPECTRUM_PEAK_SEPARATION = 2.3


class ChargeSpectrumFitter(fitter_interface.BaseScanFitter):
    FIT_FIELD_MAP = [
        ("fit_model", "fit_model"),
        ("fit_weights", "fit_weights"),
        ("fit_statistic", "fit_statistic"),
        ("fit_optimizer", "fit_optimizer"),
        ("npe", "npe"),
        ("poisson_gaussian", "poisson_gaussian"),
        ("poisson_mu", "poisson_mu"),
        ("poisson_mu_err", "poisson_mu_err"),
        ("charge_fit_statistic", "charge_fit_statistic"),
        ("spe_charge_step", "spe_charge_step"),
        ("spe_charge_step_err", "spe_charge_step_err"),
        ("spe_sigma_intrinsic", "spe_sigma_intrinsic"),
        ("spe_sigma_intrinsic_err", "spe_sigma_intrinsic_err"),
        ("gain_pedestal_subtracted", "gain_pedestal_subtracted"),
        ("gain_pedestal_subtracted_err", "gain_pedestal_subtracted_err"),
        ("backscatter_fraction", "backscatter_fraction"),
        ("backscatter_fraction_err", "backscatter_fraction_err"),
        ("chi2", "chi2"),
        ("ndf", "ndf"),
        ("objective_value", "objective_value"),
        ("prior_penalty", "prior_penalty"),
        ("fit_converged", "fit_converged"),
        ("fit_covariance_accurate", "fit_covariance_accurate"),
        ("fit_parameters_at_limit", "fit_parameters_at_limit"),
        ("seed_ped_mean", "seed_ped_mean"),
        ("seed_ped_sigma", "seed_ped_sigma"),
        ("seed_charge_step", "seed_charge_step"),
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
        spe_mu_min_pc=1.0,
        spe_mu_max_pc=2.5,
        spe_mu_prior=None,
        spe_sigma_prior=None,
        poisson_gaussian=False,
        charge_fit_statistic="unbinned_nll",
        charge_fit_nbins=300,
        fit_config=None,
    ):
        method = str(method_name or CHARGE_METHOD_NAME).strip()
        self.method_name = CHARGE_METHOD_NAME if method in CHARGE_METHOD_NAMES else method
        self.nbins = int(nbins)
        self.fit_config = charge_configuration(fit_config, charge_fit_statistic,
                                               poisson_gaussian, npe, inc_backscatter,
                                               spe_mu_prior, spe_sigma_prior)
        if fit_config is None:
            self.fit_config["statistic"]["nbins"] = int(charge_fit_nbins)
            self.fit_config["optimizer"]["prefit_nbins"] = int(charge_fit_nbins)
        else:
            # The four-section config owns its bounds; do not add old flat defaults.
            spe_mu_min_pc = spe_mu_max_pc = None
        self.model_name = self.fit_config["model"]["name"]
        self.inc_backscatter = bool(self.fit_config["model"]["backscatter"])
        self.npe = int(self.fit_config["model"]["npe"])
        self.constraints = self.fit_config["constraints"]
        self.poisson_gaussian = self.constraints["weights"] == "poisson"
        self.statistic = self.fit_config["statistic"]
        self.optimizer = self.fit_config["optimizer"]
        self.charge_fit_statistic = self.statistic["name"]
        self.charge_fit_nbins = int(self.statistic["nbins"])
        if self.charge_fit_nbins < 20:
            raise ValueError("fit.statistic.nbins must be at least 20")
        self.charge_branch = str(charge_branch or "auto")
        self.charge_qmin = charge_qmin
        self.charge_qmax = charge_qmax
        self.auto_qmin = float(np.clip(auto_qmin, 0.0, 1.0))
        self.auto_qmax = float(np.clip(auto_qmax, 0.0, 1.0))
        self.min_events = max(int(min_events), 1)
        self.spe_mu_min_pc = (
            float(spe_mu_min_pc) if spe_mu_min_pc is not None else None
        )
        self.spe_mu_max_pc = (
            float(spe_mu_max_pc) if spe_mu_max_pc is not None else None
        )
        self.spe_mu_prior = self._validate_prior(spe_mu_prior)
        self.spe_sigma_prior = self._validate_prior(spe_sigma_prior)
        self.fig_dir = Path(fig_dir).resolve() if fig_dir else None
        if self.fig_dir:
            self.fig_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, request):
        if self.method_name != CHARGE_METHOD_NAME:
            raise RuntimeError("Unsupported charge method: " + self.method_name)
        if self.model_name == "spe_response":
            out = self._fit_spe_response(request)
        else:
            out = self._fit_gaussian_peaks(request)
        out.update(fit_model=self.model_name, fit_weights=self.constraints["weights"],
                   fit_statistic=self.statistic["name"], fit_optimizer=self.optimizer["name"])
        return out

    @staticmethod
    def _validate_prior(prior):
        if prior is None:
            return None
        try:
            mean, sigma = prior
        except (TypeError, ValueError):
            raise ValueError(
                "Prior must be a (mean, sigma) pair, got {!r}".format(prior)
            )
        mean = float(mean)
        sigma = float(sigma)
        if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "Prior (mean={}, sigma={}) must be finite with sigma>0".format(mean, sigma)
            )
        return (mean, sigma)

    @staticmethod
    def _coord_key(coord):
        if isinstance(coord, (tuple, list)):
            txt = "_".join([str(x) for x in coord])
        else:
            txt = str(coord)
        txt = re.sub(r"[^0-9A-Za-z_]+", "_", txt).strip("_")
        return txt or "coord"

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

    def _estimate_seed_centers(self, data_np, xr):
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
            fallback_center = ped_center + SPECTRUM_PEAK_SEPARATION
            search_lo, search_hi = self._window_with_fixed_width(
                fallback_center,
                max(2.0 * SPECTRUM_SPE_WINDOW_WIDTH, 0.35 * span),
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

    def _seed_windows(self, data_np, xr):
        xmin, xmax = float(xr[0]), float(xr[1])
        ped_center, spe_center = self._estimate_seed_centers(data_np, xr)
        ped_min, ped_max = self._window_with_fixed_width(
            ped_center, SPECTRUM_PED_WINDOW_WIDTH, xmin, xmax
        )
        spe_min, spe_max = self._window_with_fixed_width(
            spe_center, SPECTRUM_SPE_WINDOW_WIDTH, xmin, xmax
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
            mean, variance = float(np.mean(arr)), float(np.var(arr))

            def nll(mu, sigma):
                acceptance = common_math.gaussian_acceptance((xmin, xmax), mu, sigma)
                if acceptance <= 0:
                    return np.inf
                return arr.size * (np.log(sigma) + np.log(acceptance)
                                   + 0.5 * (variance + (mean - mu) ** 2) / sigma ** 2)

            result = common_math.minimize_nll(
                nll,
                dict(mu=float(np.clip(mu_guess, mu_lo, mu_hi)),
                     sigma=float(np.clip(sigma_guess, sigma_lo, sigma_hi))),
                dict(mu=(mu_lo, mu_hi), sigma=(sigma_lo, sigma_hi)),
            )
            mu_val, sigma_val = result.values["mu"], result.values["sigma"]
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
        ped_min, ped_max, spe_min, spe_max = self._seed_windows(data_np, xr)

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

    def _gaussian_components(self, x, parameters, xr):
        """Return event intensities and window yields for the unbinned model."""
        p = parameters
        means, sigmas = gaussian_shapes(
            p["mu_ped"], p["sigma_ped"], p["mu_spe"], p["sigma_spe"], self.npe
        )
        acceptance = common_math.gaussian_acceptance(xr, means, sigmas)
        if self.poisson_gaussian:
            yields = poisson_gaussian_yields(p["amplitude"], p["poisson_mu"], means, sigmas, xr)
        else:
            yields = np.array([p[name + "_yield"] for name in ("ped", "spe", "pe2", "pe3")[:self.npe + 1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            curves = [y * gaussian_pdf(x, mean, sigma) / acc
                      for y, mean, sigma, acc in zip(yields, means, sigmas, acceptance)]
        if self.inc_backscatter:
            curves.append(p["bs_yield"] * backscatter_pdf(
                x, p["mu_ped"], p["sigma_ped"], p["mu_spe"], p["sigma_spe"], xr))
            yields = np.r_[yields, p["bs_yield"]]
        return curves, yields

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

    @plt.rc_context(plot_utils.FIT_STYLE)
    def _make_log_plot(
        self, out_png, centers, counts, yerr, x_model, curves, xr, text_lines
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.errorbar(centers, counts, yerr=yerr, fmt="ok", label="Data")
        for curve in curves:
            ax.plot(
                x_model,
                curve["y"],
                linewidth=curve.get("linewidth", 2),
                color=curve.get("color"),
                linestyle=curve.get("linestyle", "-"),
                label=curve["label"][:1].upper() + curve["label"][1:],
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
                0.78,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
            )
        ax.legend(loc="upper right")
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
        y_model = np.asarray(model(x), dtype=float)
        area = float(xr[1] - xr[0])
        y = y_model * total_yield / float(self.nbins) * area

        counts, edges = np.histogram(
            data_np, bins=self.nbins, range=(float(xr[0]), float(xr[1]))
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        text_lines = plot_utils.charge_fit_text(out)

        name = (plotname or "charge_fit").strip()
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
                np.asarray(comp_pdf(x), dtype=float)
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

    def _fit_gaussian_peaks(self, request):
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

        if self.optimizer["initialization"] == "spectrum":
            guesses = self._initial_guesses(data_np, xr, coord_key)
        else:
            selected = self._finite_1d(request.fit_kwargs.get("seed_charge", []))
            if not len(selected):
                raise ValueError("pulse_height initialization requires seed_charge")
            counts, edges = np.histogram(data_np, bins=self.optimizer["prefit_nbins"], range=xr)
            selected_counts, _ = np.histogram(selected, bins=edges)
            ped, sig0, step, sig1 = pulse_height_initial_parameters(counts, selected_counts, .5 * (edges[1:] + edges[:-1]))
            guesses = dict(mu_ped=ped, sigma_ped=sig0, mu_spe=ped + step,
                           sigma_spe=np.sqrt(sig0 ** 2 + sig1 ** 2),
                           ped_window=xr, spe_window=xr)
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

        # User-level SPE peak constraint (e.g. 1.0–2.5 pC), intersected with
        # the data range for safety. Falls back to the loose bounds above if
        # the requested window does not overlap the data range.
        if self.spe_mu_min_pc is not None or self.spe_mu_max_pc is not None:
            user_lo = float(self.spe_mu_min_pc) if self.spe_mu_min_pc is not None else -np.inf
            user_hi = float(self.spe_mu_max_pc) if self.spe_mu_max_pc is not None else np.inf
            cand_lo = max(float(xmin), float(mu_ped_guess + 0.05), user_lo)
            cand_hi = min(float(xmax), user_hi)
            if cand_hi > cand_lo:
                spe_mu_lo = cand_lo
                spe_mu_hi = cand_hi

        sigma_ped_lo = max(1e-4, float(0.5 * sigma_ped_guess))
        sigma_ped_hi = max(
            sigma_ped_lo * 1.2, float(np.sqrt(2.0) * sigma_ped_guess), 0.1
        )
        sigma_spe_lo = max(1e-4, float(0.5 * sigma_spe_guess))
        sigma_spe_hi = max(sigma_spe_lo * 1.2, float(2.0 * sigma_spe_guess), 0.15)

        initial = dict(mu_ped=mu_ped_guess, sigma_ped=sigma_ped_guess,
                       mu_spe=float(np.clip(mu_spe_guess, spe_mu_lo, spe_mu_hi)),
                       sigma_spe=sigma_spe_guess)
        limits = dict(mu_ped=(ped_mu_lo, ped_mu_hi), sigma_ped=(sigma_ped_lo, sigma_ped_hi),
                      mu_spe=(spe_mu_lo, spe_mu_hi), sigma_spe=(sigma_spe_lo, sigma_spe_hi))
        prefixes = ("ped", "spe", "pe2", "pe3")[:self.npe + 1]
        if self.poisson_gaussian:
            initial.update(amplitude=float(size), poisson_mu=0.1)
            limits.update(amplitude=(0., None), poisson_mu=(0.001, 2.5))
        else:
            for prefix, fraction, upper in zip(prefixes, (.5, .3, .08, .03), (1.2, 1.2, .8, .6)):
                initial[prefix + "_yield"] = max(size * fraction, 1.)
                limits[prefix + "_yield"] = (0., max(size * upper, 2.))
        if self.inc_backscatter:
            initial["bs_yield"] = max(size * .05, 0.)
            limits["bs_yield"] = (0., max(size * .8, 2.))

        physical = {name: (0., None) for name in initial}
        physical.update(mu_ped=xr, mu_spe=xr, sigma_ped=(1e-8, None), sigma_spe=(1e-8, None))
        if self.poisson_gaussian:
            physical["poisson_mu"] = (1e-8, None)
        initial, limits = common_math.apply_constraints(initial, limits, self.constraints, physical)
        names = tuple(initial)

        def density(x, p):
            return np.sum(self._gaussian_components(x, p, xr)[0], axis=0)

        cost = common_math.build_objective(
            data_np, xr, names, density, lambda p: self._gaussian_components(np.array([]), p, xr)[1].sum(),
            self.statistic, self.constraints["priors"])
        starts = None
        if self.poisson_gaussian and self.optimizer["multistart"] and "poisson_mu" not in self.constraints["fixed"]:
            occupancy = np.mean(data_np > .5 * (mu_ped_guess + mu_spe_guess))
            starts = [dict(initial, poisson_mu=mu) for mu in (.1, float(np.clip(occupancy, *limits["poisson_mu"])))]
        result = common_math.minimize(cost, initial, limits, self.optimizer, self.constraints["fixed"], starts)
        p = result.values.to_dict()
        errors = common_math.parameter_errors(result)
        means, sigmas = gaussian_shapes(p["mu_ped"], p["sigma_ped"], p["mu_spe"], p["sigma_spe"], self.npe)
        yields = self._gaussian_components(np.array([]), p, xr)[1]

        def window_yields(values):
            return self._gaussian_components(np.array([]), dict(zip(initial, values)), xr)[1]

        yield_errors = common_math.propagated_errors(window_yields, list(result.values), result.covariance)
        out = dict(npe=self.npe, poisson_gaussian=self.poisson_gaussian,
                   poisson_mu=p.get("poisson_mu", np.nan),
                   poisson_mu_err=errors["poisson_mu"] if self.poisson_gaussian else np.nan,
                   charge_fit_statistic=self.charge_fit_statistic, **common_math.fit_quality(result))
        out["objective_value"] = result.fval
        out["prior_penalty"] = sum(cost.errordef * ((p[name] - mean) / sigma) ** 2
                                   for name, (mean, sigma) in self.constraints["priors"].items())
        out["chi2"] = result.fval - out["prior_penalty"] if self.charge_fit_statistic == "chi2" else np.nan
        out["ndf"] = cost.ndata - result.nfit if self.charge_fit_statistic == "chi2" else np.nan
        for i, prefix in enumerate(("ped", "spe", "pe2", "pe3")):
            for suffix, values in (("mean", means), ("sigma", sigmas),
                                   ("yield", yields), ("yield_err", yield_errors)):
                out[prefix + "_" + suffix] = float(values[i]) if i <= self.npe else np.nan
        for prefix in ("ped", "spe"):
            out[prefix + "_mean_err"] = errors["mu_" + prefix]
            out[prefix + "_sigma_err"] = errors["sigma_" + prefix]
        out["backscatter_yield"] = float(yields[-1]) if self.inc_backscatter else np.nan
        out["backscatter_yield_err"] = float(yield_errors[-1]) if self.inc_backscatter else np.nan

        # Preserve the existing CSV definitions, including legacy absolute SPE gain.
        swapped_roles = False if self.poisson_gaussian else self._canonicalize_gaussian_roles(out)
        out["total_yield"] = float(np.sum(yields))
        out["gain"] = out["spe_mean"] / E_CHARGE_PC
        out["gain_err"] = out["spe_mean_err"] / E_CHARGE_PC
        step_gradient = np.array([1. if name == "mu_spe" else -1. if name == "mu_ped" else 0.
                                  for name in result.parameters])
        variance = (step_gradient @ np.asarray(result.covariance) @ step_gradient
                    if result.covariance is not None else np.nan)
        out["spe_charge_step"] = out["spe_mean"] - out["ped_mean"]
        out["spe_charge_step_err"] = np.sqrt(variance) if variance >= 0 else np.nan
        out["gain_pedestal_subtracted"] = out["spe_charge_step"] / E_CHARGE_PC
        out["gain_pedestal_subtracted_err"] = out["spe_charge_step_err"] / E_CHARGE_PC
        out["resolution"] = out["spe_sigma"] / out["spe_mean"] * 100. if out["spe_mean"] else np.nan
        out["peak_to_valley"] = self._compute_peak_to_valley(data_np, xr, out["ped_mean"], out["spe_mean"])

        def model(x):
            curves, _ = self._gaussian_components(x, p, xr)
            return np.sum(curves, axis=0) / out["total_yield"]

        specs = []
        labels = ["pedestal", "SPE", "2PE", "3PE"][:self.npe + 1]
        if swapped_roles:
            labels[:2] = labels[1], labels[0]
        if self.inc_backscatter:
            labels.append("backscatter")
        for i, (label, color) in enumerate(zip(labels, ["tab:orange", "tab:green", "tab:purple", "tab:brown"][:self.npe + 1]
                                              + (["tab:red"] if self.inc_backscatter else []))):
            def component_pdf(x, i=i):
                curves, _ = self._gaussian_components(x, p, xr)
                return curves[i] / yields[i]
            specs.append({"label": label, "pdf": component_pdf, "yield": yields[i], "color": color})
        self._make_plot(model, data_np, xr, out, request.plotname, specs)
        return out

    def _fit_spe_response(self, request):
        charge = self._clip_to_range(request.data, *request.xr)
        if len(charge) < self.min_events:
            raise ValueError("Too few events in the charge fit range")
        counts, edges = np.histogram(charge, bins=self.charge_fit_nbins, range=request.xr)
        centers, bin_width = .5 * (edges[:-1] + edges[1:]), edges[1] - edges[0]
        selected = self._finite_1d(request.fit_kwargs.get("seed_charge", []))
        if self.optimizer["initialization"] == "pulse_height":
            if selected.size == 0:
                raise ValueError("pulse_height initialization requires seed_charge from the same run")
            seed_counts, seed_edges = np.histogram(charge, bins=self.optimizer["prefit_nbins"], range=request.xr)
            selected_counts, _ = np.histogram(selected, bins=seed_edges)
            ped, sig0, gain, sig1 = pulse_height_initial_parameters(
                seed_counts, selected_counts, .5 * (seed_edges[:-1] + seed_edges[1:]))
        else:
            guesses = self._initial_guesses(charge, request.xr, self._coord_key(request.coord))
            ped, sig0 = guesses["mu_ped"], guesses["sigma_ped"]
            gain = guesses["mu_spe"] - ped
            sig1 = np.sqrt(max(guesses["sigma_spe"] ** 2 - sig0 ** 2, 1e-8))
        if gain <= 0:
            raise ValueError("Initialization did not resolve a positive charge step")
        low_gain = gain < 1.6
        initial = dict(zip(RESPONSE_PARAMETERS,
                           [len(charge) * bin_width, .1, ped, gain, sig0, sig1,
                            .03 if low_gain else .08]))
        limits = dict(zip(RESPONSE_PARAMETERS, [
            (0, None), (.001, 2.5), (ped - sig0, ped + sig0),
            (gain * (.85 if low_gain else .75), gain * (1.15 if low_gain else 1.25)),
            (sig0 * .5, sig0 * 1.3),
            (sig0 * (.8 if low_gain else .5), gain * (.45 if low_gain else .55)),
            (0, .15 if low_gain else .35),
        ]))
        gain_lo, gain_hi = limits["spe_gain"]
        if self.spe_mu_min_pc is not None:
            gain_lo = max(gain_lo, self.spe_mu_min_pc - ped)
        if self.spe_mu_max_pc is not None:
            gain_hi = min(gain_hi, self.spe_mu_max_pc - ped)
        if gain_lo >= gain_hi:
            raise ValueError("SPE bounds do not overlap the prefit charge-step interval")
        limits["spe_gain"] = (gain_lo, gain_hi)
        initial["spe_gain"] = float(np.clip(gain, gain_lo, gain_hi))
        amplitude_names = ["amplitude"]
        if not self.poisson_gaussian:
            amplitude_names = [name + "_amplitude" for name in ("ped", "spe", "pe2", "pe3")[:self.npe + 1]]
            weights = {name: initial["amplitude"] * np.exp(-.1) * .1 ** n / math.factorial(n)
                       for n, name in enumerate(amplitude_names)}
            initial = dict(weights, **{name: initial[name] for name in RESPONSE_PARAMETERS[2:]})
            limits = dict({name: (0., None) for name in amplitude_names},
                          **{name: limits[name] for name in RESPONSE_PARAMETERS[2:]})
        physical = {name: (0., None) for name in initial}
        physical.update(pedestal=tuple(request.xr), spe_gain=(1e-8, None),
                        ped_sigma=(1e-8, None), spe_sigma=(1e-8, None), back_fraction=(0., 1.))
        if self.poisson_gaussian:
            physical["mu"] = (1e-8, None)
        initial, limits = common_math.apply_constraints(initial, limits, self.constraints, physical)
        fixed = dict(self.constraints["fixed"])
        if not self.inc_backscatter:
            if any("back_fraction" in self.constraints[section] for section in ("bounds", "priors")):
                raise ValueError("back_fraction is inactive when model.backscatter=false")
            if fixed.get("back_fraction", 0.) != 0.:
                raise ValueError("model.backscatter=false requires back_fraction=0")
            initial["back_fraction"] = 0.
            fixed["back_fraction"] = 0.
        names = tuple(initial)

        def vector(p):
            return np.array([p[name] for name in names])

        def density_parameters(values):
            values = np.asarray(values, dtype=float).copy()
            for name in amplitude_names:
                values[names.index(name)] /= bin_width
            return values

        def bin_prediction(x, p):
            return spe_response_model(x, vector(p), self.npe, self.poisson_gaussian)

        def density(x, p):
            return spe_response_model(x, density_parameters(vector(p)), self.npe, self.poisson_gaussian)

        def window_yields(values):
            return spe_response_yields(density_parameters(values), request.xr, self.npe, self.poisson_gaussian)

        cost = common_math.build_objective(
            charge, request.xr, names, density, lambda p: window_yields(vector(p)).sum(),
            self.statistic, self.constraints["priors"], bin_prediction=bin_prediction)
        starts = [initial]
        if self.poisson_gaussian and self.optimizer["multistart"] and "mu" not in fixed:
            occupancy = len(selected) / len(charge) if len(selected) else .1
            starts = [dict(initial, mu=mu) for mu in
                      (.1, float(np.clip(occupancy, *limits["mu"])))]
        result = common_math.minimize(cost, initial, limits, self.optimizer, fixed, starts)
        parameters = np.array(list(result.values))
        p = result.values.to_dict()
        covariance = np.asarray(result.covariance) if result.covariance is not None else np.full((len(names), len(names)), np.nan)

        def propagated(gradient):
            variance = np.asarray(gradient) @ covariance @ np.asarray(gradient)
            return float(np.sqrt(variance)) if variance >= 0 else np.nan

        def gradient(**terms):
            return [terms.get(name, 0.) for name in names]

        ped, gain, sig0, sig1 = p["pedestal"], p["spe_gain"], p["ped_sigma"], p["spe_sigma"]
        sigma_total = np.sqrt(sig0 ** 2 + sig1 ** 2)
        mean_error = propagated(gradient(pedestal=1., spe_gain=1.))
        sigma_error = propagated(gradient(ped_sigma=sig0 / sigma_total, spe_sigma=sig1 / sigma_total))
        prior_penalty = sum(cost.errordef * ((p[name] - mean) / sigma) ** 2
                            for name, (mean, sigma) in self.constraints["priors"].items())
        out = {
            "npe": self.npe, "poisson_gaussian": self.poisson_gaussian,
            "charge_fit_statistic": self.charge_fit_statistic,
            "poisson_mu": p.get("mu", np.nan),
            "poisson_mu_err": result.errors["mu"] if self.poisson_gaussian else np.nan,
            "ped_mean": ped, "ped_mean_err": result.errors["pedestal"],
            "ped_sigma": sig0, "ped_sigma_err": result.errors["ped_sigma"],
            "spe_mean": ped + gain, "spe_mean_err": mean_error,
            "spe_sigma": sigma_total, "spe_sigma_err": sigma_error,
            "spe_sigma_intrinsic": sig1, "spe_sigma_intrinsic_err": result.errors["spe_sigma"],
            "spe_charge_step": gain, "spe_charge_step_err": result.errors["spe_gain"],
            "gain": (ped + gain) / E_CHARGE_PC, "gain_err": mean_error / E_CHARGE_PC,
            "gain_pedestal_subtracted": gain / E_CHARGE_PC,
            "gain_pedestal_subtracted_err": result.errors["spe_gain"] / E_CHARGE_PC,
            "backscatter_fraction": p["back_fraction"],
            "backscatter_fraction_err": result.errors["back_fraction"],
            "resolution": sig1 / gain * 100.,
            "chi2": result.fval - prior_penalty if self.charge_fit_statistic == "chi2" else np.nan,
            "ndf": cost.ndata - result.nfit if self.charge_fit_statistic == "chi2" else np.nan,
            "objective_value": result.fval, "prior_penalty": prior_penalty,
            "seed_ped_mean": initial["pedestal"], "seed_ped_sigma": initial["ped_sigma"],
            "seed_charge_step": initial["spe_gain"], **common_math.fit_quality(result),
        }
        yields = window_yields(parameters)
        jacobian = np.empty((len(yields), len(parameters)))
        for i, value in enumerate(parameters):
            step = 1e-5 * max(abs(value), 1e-3)
            plus, minus = parameters.copy(), parameters.copy()
            plus[i] += step
            minus[i] -= step
            jacobian[:, i] = (window_yields(plus) - window_yields(minus)) / (2 * step)
        for n, prefix in enumerate(("ped", "spe", "pe2", "pe3")[:self.npe + 1]):
            out[prefix + "_yield"] = yields[n]
            out[prefix + "_yield_err"] = propagated(jacobian[n])
            if n >= 2:
                out[prefix + "_mean"] = ped + n * gain
                out[prefix + "_sigma"] = np.sqrt(sig0 ** 2 + n * sig1 ** 2)
        out["backscatter_yield"], out["backscatter_yield_err"] = yields[-1], propagated(jacobian[-1])
        out["total_yield"] = float(yields.sum())
        out["peak_to_valley"] = np.nan

        if self.fig_dir is not None:
            xx = np.linspace(*request.xr, 1500)
            gaussians, backscatters = spe_response_components(xx, parameters, self.npe, self.poisson_gaussian)
            curves = [{"label": "total", "y": np.sum(gaussians + backscatters, axis=0), "color": "tab:blue"}]
            for name, curve, color in zip(("pedestal", "SPE", "2PE", "3PE"), gaussians, ("tab:orange", "tab:green", "tab:purple", "tab:brown")):
                curves.append({"label": name, "y": curve, "color": color, "linestyle": "--"})
            if self.inc_backscatter:
                curves.append({"label": "backscatter", "y": np.sum(backscatters, axis=0), "color": "tab:red", "linestyle": "--"})
            notes = plot_utils.charge_fit_text(out)
            target = self.fig_dir / ((request.plotname or "spe_response") + "_log.png")
            self._make_log_plot(target, centers, counts, np.sqrt(np.maximum(counts, 1)), xx, curves, request.xr, notes)
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

    def _attach_main_fit_fields(self, row, fit_out):
        super()._attach_main_fit_fields(row, fit_out)
        if fit_out.get("fit_converged") is False:
            row["fit_status"] = "warning_not_converged"
        elif fit_out.get("fit_covariance_accurate") is False:
            row["fit_status"] = "warning_covariance"
        elif {"spe_gain", "mu_spe"} & set(fit_out.get("fit_parameters_at_limit", "").split(";")):
            row["fit_status"] = "warning_gain_at_limit"

    def _apply_kor_relative_columns(self, df):
        out_df = df.copy()
        out_df["relative_gain"] = np.nan
        out_df["relative_gain_err"] = np.nan
        # Charge-based relative QE proxy: SPE-peak yield / events in fit window,
        # normalized per phi to the theta_raw=0 reference.
        spe_y = pd.to_numeric(out_df.get("spe_yield"), errors="coerce").astype(float)
        spe_y_err = pd.to_numeric(out_df.get("spe_yield_err"), errors="coerce").astype(float)
        n_in = pd.to_numeric(out_df.get("n_in_window"), errors="coerce").astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out_df["rel_qe_charge"] = np.where(n_in > 0, spe_y / n_in, np.nan)
            out_df["rel_qe_charge_err"] = np.where(
                n_in > 0, spe_y_err / n_in, np.nan
            )
        out_df["rel_qe_charge_norm"] = np.nan
        out_df["rel_qe_charge_norm_err"] = np.nan

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
        if self.optimizer["initialization"] == "pulse_height":
            raise ValueError("This reader has no saved pulse-height seed threshold; "
                             "use fit.optimizer.initialization=spectrum")

        _input_dir, out_csv, files = self.resolve_inputs(
            args=args,
            default_out_csv="csv/aus_charge_results.csv",
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

        pmt_tree = "Tree_CH{}".format(ctx["pmt_ch"])
        trigger_tree = "Tree_CH{}".format(ctx["trigger_ch"])
        charge_branch = self._resolve_charge_branch("aus")
        require_ps = bool(getattr(args, "require_pulsestart", False))
        apply_tcut = bool(getattr(args, "apply_timing_cut", False))
        timing_cut = getattr(args, "timing_cut", (300.0, 320.0)) or (300.0, 320.0)
        tmin_ns, tmax_ns = float(timing_cut[0]), float(timing_cut[1])

        prep_stats = {
            "files_scanned": int(len(files)),
            "files_kept": 0,
            "charge_read_fail": 0,
            "empty_fit_range": 0,
            "too_few_events": 0,
        }
        points = []

        for idx, fp in enumerate(files):
            parsed = aus_reader.parse_theta_phi_aus(fp)
            if parsed is None:
                continue
            theta, phi = parsed

            charge = self.run_step(
                lambda: aus_reader.load_branch(fp, pmt_tree, charge_branch),
                stats=prep_stats,
                fail_key="charge_read_fail",
                file_name=fp.name,
                fail_msg="read AUS charge branch failed",
            )
            if charge is None:
                continue

            if require_ps or apply_tcut:
                ps_pmt = self.run_step(
                    lambda: aus_reader.load_branch(fp, pmt_tree, "PulseStart") * SAMPLE_NS,
                    stats=prep_stats,
                    fail_key="charge_read_fail",
                    file_name=fp.name,
                    fail_msg="read PMT PulseStart failed",
                )
                if ps_pmt is None:
                    continue
                mask = np.isfinite(ps_pmt)
                if apply_tcut:
                    ps_las = self.run_step(
                        lambda: aus_reader.load_branch(fp, trigger_tree, "PulseStart")
                        * SAMPLE_NS,
                        stats=prep_stats,
                        fail_key="charge_read_fail",
                        file_name=fp.name,
                        fail_msg="read trigger PulseStart failed",
                    )
                    if ps_las is None:
                        continue
                    delta = ps_pmt - ps_las
                    mask &= np.isfinite(delta) & (delta > tmin_ns) & (delta < tmax_ns)
                charge = np.asarray(charge)[mask]

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
                "system": "aus",
                "file": fp.name,
                "serial": str(ctx["serial"]),
                "pmt_ch": int(ctx["pmt_ch"]),
                "trigger_ch": int(ctx["trigger_ch"]),
                "sipm_ch": int(ctx["sipm_ch"]),
                "pmt_ch_cfg": str(ctx["pmt_ch_cfg"]),
                "trigger_ch_cfg": str(ctx["trigger_ch_cfg"]),
                "sipm_ch_cfg": str(ctx["sipm_ch_cfg"]),
                "theta": int(theta),
                "phi": int(phi),
                "charge_method": self.method_name,
                "npe": int(self.npe),
                "charge_branch": str(charge_branch),
                "include_backscatter": bool(self.inc_backscatter),
                "charge_range_min": float(use_qmin),
                "charge_range_max": float(use_qmax),
                "charge_peak": float(peak),
                "require_pulsestart": bool(require_ps),
                "apply_timing_cut": bool(apply_tcut),
                "timing_cut_min_ns": float(tmin_ns) if apply_tcut else np.nan,
                "timing_cut_max_ns": float(tmax_ns) if apply_tcut else np.nan,
            }

            points.append(
                self.make_point(
                    row=row,
                    main_fit_input=self.make_fit_input(
                        data=fit_values,
                        coord=("aus_charge", ctx["serial"], int(theta), int(phi), idx),
                        plotname="aus_charge_{}_theta{}_phi{}_{}".format(
                            ctx["serial"] or "SNX", theta, phi, idx
                        ),
                        xr=(use_qmin, use_qmax),
                        meta={"n_in_window": int(fit_values.size)},
                    ),
                    main_skip_msg="[SKIP] {}: AUS charge fit failed.".format(fp.name),
                )
            )
            prep_stats["files_kept"] += 1

        return {"out_csv": out_csv, "points": points, "prep_stats": prep_stats}

    def _prepare_kor_input(self, args):
        _input_dir, out_csv, files = self.resolve_inputs(
            args=args,
            default_out_csv="csv/kor_charge_results.csv",
            file_pattern="*prd_*.root",
            empty_msg="No *prd_*.root found in {}",
        )

        selected = []
        for fp in files:
            parsed = kor_reader.extract_serial_block_angles(fp, args.serial)
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
            fit_kwargs = {}
            seed_threshold = np.nan
            if self.optimizer["initialization"] == "pulse_height":
                if charge_branch != "pico":
                    raise ValueError("Pulse-height initialization expects the pico branch in pC")
                seed_charge, seed_threshold = kor_reader.read_charge_seed(fp, ctx["channel"])
                fit_kwargs["seed_charge"] = seed_charge
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
                "seed_threshold_mv": seed_threshold,
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
                        fit_kwargs=fit_kwargs,
                    ),
                    main_skip_msg="[SKIP] {}: KOR charge fit failed.".format(fp.name),
                )
            )
            prep_stats["files_kept"] += 1

        return {"out_csv": out_csv, "points": points, "prep_stats": prep_stats}

    def prepare_scan(self, system, args):
        if system == "aus":
            inputs = self._prepare_aus_input(args)
            return {
                "out_csv": inputs["out_csv"],
                "points": inputs["points"],
                "prep_stats": inputs.get("prep_stats", {}),
                "sort_cols": ["phi", "theta", "file"],
                "empty_msg": "No valid AUS charge-fit results produced.",
            }

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

# Deprecated names for external validation scripts; all fitting uses generic functions.
kor_charge_components = spe_response_components
kor_charge_model = spe_response_model
kor_component_yields = spe_response_yields
