"""Shared numeric helpers for fit pipelines."""

import logging

import numpy as np


_logger = logging.getLogger(__name__)


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
