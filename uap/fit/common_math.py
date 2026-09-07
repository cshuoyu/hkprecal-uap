"""Shared numeric helpers for fit pipelines."""

import logging

import numpy as np
from iminuit import Minuit
from scipy.special import ndtr


_logger = logging.getLogger(__name__)


def gaussian_acceptance(limits, mean, sigma):
    """Gaussian probability inside a finite window, also in the upper tail."""
    lo = (limits[0] - np.asarray(mean)) / sigma
    hi = (limits[1] - np.asarray(mean)) / sigma
    return np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))


def extended_nll(expected, density, counts=None):
    """Extended unbinned -log L, dropping only parameter-independent terms.

    density is the event intensity (yield * normalized PDF), not a PDF alone.
    counts may combine exactly repeated observations; this is not binning.
    """
    if not np.isfinite(expected) or expected <= 0:
        return np.inf
    if not np.all(np.isfinite(density)) or np.any(density <= 0):
        return np.inf
    log_density = np.log(density)
    return float(expected - np.sum(log_density if counts is None else counts * log_density))


def build_objective(data, limits, names, density, integral, statistic, priors=None,
                    bin_prediction=None):
    """Compose a data cost and Gaussian penalties, independently of the model.

    density/integral take (x, parameters)/(parameters) in event units. Optional
    bin_prediction preserves an existing bin-height parameterization exactly.
    Chi2 uses observed-count variance and excludes empty bins, as before.
    """
    name = statistic["name"]
    if name == "chi2":
        counts, edges = np.histogram(data, bins=int(statistic["nbins"]), range=limits)
        keep = counts > 0
        x, counts = (.5 * (edges[1:] + edges[:-1]))[keep], counts[keep].astype(float)
        width = edges[1] - edges[0]
        errordef = 1.0
    elif name == "unbinned_nll":
        x, counts = np.unique(data, return_counts=True)
        errordef = .5
    else:
        raise ValueError("Unknown fit statistic: " + str(name))

    def cost(*values):
        parameters = dict(zip(names, values))
        if name == "chi2":
            expected = (bin_prediction(x, parameters) if bin_prediction is not None else
                        density(x, parameters) * width)
            if not np.all(np.isfinite(expected)) or np.any(expected < 0):
                return np.inf
            value = np.sum((expected - counts) ** 2 / counts)
        else:
            value = extended_nll(integral(parameters), density(x, parameters), counts)
        for parameter, (mean, sigma) in (priors or {}).items():
            value += errordef * ((parameters[parameter] - mean) / sigma) ** 2
        return float(value)

    cost.errordef = errordef
    cost.ndata = len(x) if name == "chi2" else len(data)
    return cost


def apply_constraints(initial, limits, constraints, physical_bounds):
    """Apply explicit bounds/fixed values; priors are added by build_objective."""
    initial, limits = dict(initial), dict(limits)
    for section in ("bounds", "fixed", "priors"):
        unknown = set(constraints.get(section, {})) - set(initial)
        if unknown:
            raise ValueError("Unknown or inactive parameters in constraints.{}: {}".format(section, sorted(unknown)))
    if not constraints.get("prefit_bounds", True):
        limits = dict(physical_bounds)
    for name, interval in constraints.get("bounds", {}).items():
        lo, hi = interval
        physical_lo, physical_hi = physical_bounds[name]
        lo = max(v for v in (lo, physical_lo) if v is not None) if lo is not None or physical_lo is not None else None
        hi = min(v for v in (hi, physical_hi) if v is not None) if hi is not None or physical_hi is not None else None
        limits[name] = (lo, hi)
    for name, (lo, hi) in limits.items():
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("Invalid bounds for {}: {}".format(name, (lo, hi)))
        if not constraints.get("prefit_bounds", True) or name in constraints.get("bounds", {}):
            initial[name] = float(np.clip(initial[name], -np.inf if lo is None else lo, np.inf if hi is None else hi))
    for name, value in constraints.get("fixed", {}).items():
        lo, hi = limits[name]
        if not np.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError("Fixed value outside bounds for " + name)
        initial[name] = float(value)
    for name, (mean, sigma) in constraints.get("priors", {}).items():
        if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Invalid Gaussian prior for " + name)
    return initial, limits


def minimize(cost, initial, limits, options=None, fixed=None, starts=None):
    """The optimizer only sees a scalar objective and parameter constraints."""
    options = options or {}
    if options.get("name", "iminuit") != "iminuit":
        raise ValueError("Only the iminuit optimizer is implemented")
    candidates = []
    for start in starts or [initial]:
        result = Minuit(cost, *[start[name] for name in initial], name=tuple(initial))
        # Minuit reads cost.errordef; setting it again emits a warning.
        result.limits = [limits[name] for name in initial]
        for name in fixed or {}:
            result.fixed[name] = True
        step = options.get("yield_step", 1.0)
        if step is not None:
            for name in initial:
                if "yield" in name or "amplitude" in name:
                    result.errors[name] = step
        result.strategy = int(options.get("strategy", 1))
        result.tol = float(options.get("tolerance", .1))
        result.migrad(ncall=int(options.get("max_calls", 20000)))
        result.hesse()
        if not result.valid:
            result.migrad(ncall=int(options.get("max_calls", 20000)))
            result.hesse()
        candidates.append(result)
    valid = [result for result in candidates if result.valid]
    return min(valid or candidates, key=lambda result: result.fval)


def minimize_nll(cost, initial, limits):
    """Compatibility helper for Gaussian prefits and existing callers."""
    cost.errordef = Minuit.LIKELIHOOD
    return minimize(cost, initial, limits)


def parameter_errors(result):
    """Symmetric errors from covariance, matching the former NLL HESSE output.

    Minuit.errors includes a bounded-parameter transformation near limits.
    Neither convention is a reliable confidence interval at a hard boundary.
    """
    if result.covariance is None:
        return dict.fromkeys(result.parameters, np.nan)
    diagonal = np.diag(np.asarray(result.covariance))
    return dict(zip(result.parameters, np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))))


def propagated_errors(function, values, covariance):
    """Propagate the full covariance through a small vector-valued function."""
    values = np.asarray(values, dtype=float)
    outputs = np.atleast_1d(function(values))
    if covariance is None:
        return np.full(outputs.shape, np.nan)
    jacobian = np.empty((outputs.size, values.size))
    for i, value in enumerate(values):
        step = 1e-5 * max(abs(value), 1e-3)
        plus, minus = values.copy(), values.copy()
        plus[i] += step
        minus[i] -= step
        jacobian[:, i] = (np.asarray(function(plus)) - np.asarray(function(minus))) / (2 * step)
    variance = np.einsum("ij,jk,ik->i", jacobian, np.asarray(covariance), jacobian)
    return np.sqrt(np.where(variance >= 0, variance, np.nan))


def fit_quality(result):
    """Preserve convergence/covariance failures instead of declaring every fit OK."""
    at_limit = []
    for name in result.parameters:
        if result.fixed[name]:
            continue
        value = result.values[name]
        lo, hi = result.limits[name]
        scale = hi - lo if np.isfinite(lo) and np.isfinite(hi) else max(1., abs(value))
        if any(np.isfinite(bound) and abs(value - bound) <= 1e-3 * scale for bound in (lo, hi)):
            at_limit.append(name)
    return dict(fit_converged=bool(result.valid),
                fit_covariance_accurate=bool(result.fmin.has_accurate_covar),
                fit_parameters_at_limit=";".join(at_limit))


def safe_ratio(num, den, logger=None, warn_tag="RATIO"):
    logger = logger or _logger
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(num, den)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    invalid_count = int(np.sum(~mask))
    if invalid_count > 0:
        logger.warning(
            "[%s][WARN] invalid ratio entries: %s/%s (den==0 or non-finite).",
            warn_tag,
            invalid_count,
            int(mask.size),
        )
    return np.where(mask, ratio, np.nan)


def safe_ratio_err(num, num_err, den, den_err, logger=None, warn_tag="RATIO_ERR"):
    logger = logger or _logger
    num = np.asarray(num, dtype=float)
    num_err = np.asarray(num_err, dtype=float)
    den = np.asarray(den, dtype=float)
    den_err = np.asarray(den_err, dtype=float)

    ratio = safe_ratio(num, den, logger=logger, warn_tag="RATIO")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_num = np.divide(num_err, num)
        rel_den = np.divide(den_err, den)
        err = np.abs(ratio) * np.sqrt(rel_num ** 2 + rel_den ** 2)

    mask = (
        np.isfinite(ratio)
        & np.isfinite(num)
        & np.isfinite(num_err)
        & np.isfinite(den)
        & np.isfinite(den_err)
        & (num != 0)
        & (den != 0)
    )
    invalid_count = int(np.sum(~mask))
    if invalid_count > 0:
        logger.warning(
            "[%s][WARN] invalid propagated-error entries: %s/%s.",
            warn_tag,
            invalid_count,
            int(mask.size),
        )
    return np.where(mask, err, np.nan)
