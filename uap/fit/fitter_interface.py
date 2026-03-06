"""Shared fitter building blocks for ROOT->CSV pipelines."""

import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from uap.tool.scan_prepare import (
    cut_interval as _cut_interval,
    inc_stat as _inc_stat,
    log_channel_config as _log_channel_config,
    log_skip as _log_skip,
    make_fit_input as _make_fit_input,
    make_point as _make_point,
    parse_auto_or_int as _parse_auto_or_int,
    resolve_inputs as _resolve_inputs,
    run_step as _run_step,
)


logger = logging.getLogger(__name__)


# One fit task passed into self.fit.
class FitRequest(object):
    def __init__(self, coord, data, plotname=None, xr=None, fit_kwargs=None):
        self.coord = coord
        self.data = data
        self.plotname = plotname
        self.xr = xr
        self.fit_kwargs = fit_kwargs or {}


# Minimal fitter interface.
class BaseFitter(object, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, request):
        """Return a dict with fit outputs."""
        raise NotImplementedError


# Shared root->csv scan workflow used by concrete fitters.
class BaseScanFitter(BaseFitter, metaclass=ABCMeta):
    # Map output CSV columns to keys returned by fit().
    FIT_FIELD_MAP = [
        ("sig_yield", "sig_yield"),
        ("sig_err", "sig_err"),
        ("bkg_yield", "bkg_yield"),
        ("bkg_err", "bkg_err"),
        ("mu", "mean"),
        ("mu_err", "mu_err"),
        ("sigma", "sigma"),
        ("lambda", "lambda"),
        ("FWHM", "FWHM"),
    ]

    # Engine entrypoint.
    def run_scan_to_csv(self, system, args):
        return self.run_root_to_csv(args, system=system)

    # Parse channel config.
    @staticmethod
    def parse_auto_or_int(value, arg_name):
        return _parse_auto_or_int(value, arg_name)

    # Resolve input_dir/out_csv and discover input ROOT files.
    def resolve_inputs(self, args, default_out_csv, file_pattern, empty_msg):
        return _resolve_inputs(args, default_out_csv, file_pattern, empty_msg)

    # Create one fit-input record used by _fit_from_input().
    @staticmethod
    def make_fit_input(data, coord, plotname, xr, meta=None):
        return _make_fit_input(data, coord, plotname, xr, meta=meta)

    # Create one point record.
    @staticmethod
    def make_point(row, main_fit_input, main_skip_msg=None, aux_blocks=None):
        return _make_point(
            row,
            main_fit_input,
            main_skip_msg=main_skip_msg,
            aux_blocks=aux_blocks,
        )

    # Increase a counter in prep_stats.
    @staticmethod
    def inc_stat(stats, key, inc=1):
        return _inc_stat(stats, key, inc=inc)

    # Unified skip/failure log format.
    @staticmethod
    def log_skip(file_name, message, exc=None, level="info"):
        return _log_skip(logger, file_name, message, exc=exc, level=level)

    # Run one callable and handle failures in one place.
    # On failure: update stats, log, return None.
    def run_step(
        self,
        step_fn,
        stats=None,
        fail_key=None,
        file_name="",
        fail_msg="step failed",
        log_level="info",
    ):
        return _run_step(
            step_fn,
            logger=logger,
            stats=stats,
            fail_key=fail_key,
            file_name=file_name,
            fail_msg=fail_msg,
            log_level=log_level,
        )

    # Log resolved channels and raw config values.
    @staticmethod
    def log_channel_config(system, serial, resolved=None, cfg=None, defaults=None):
        return _log_channel_config(
            logger,
            system=system,
            serial=serial,
            resolved=resolved,
            cfg=cfg,
            defaults=defaults,
        )

    # Keep finite values inside (tmin, tmax).
    @staticmethod
    def cut_interval(values, tmin, tmax):
        return _cut_interval(values, tmin, tmax)

    # Full shared flow:
    # 1) prepare_scan -> points
    # 2) run fits
    # 3) postprocess columns
    # 4) write CSV
    def run_root_to_csv(self, args, system=None):
        system = (
            str(system if system is not None else getattr(args, "system", ""))
            .strip()
            .lower()
        )
        logger.info("[{}][STEP] prepare_scan".format(system.upper()))
        scan = self.prepare_scan(system, args)

        points = list(scan.get("points", []))
        logger.info(
            "[{}][STEP] run_fits total_points={}".format(system.upper(), len(points))
        )
        raw_df, ana_stats = self._analyze_points(
            points, inc_bkg=bool(getattr(args, "inc_bkg", True))
        )
        if raw_df.empty:
            raise SystemExit(scan.get("empty_msg", "No valid fit results produced."))

        sort_cols = [c for c in list(scan.get("sort_cols", [])) if c in raw_df.columns]
        if sort_cols:
            raw_df = raw_df.sort_values(sort_cols).reset_index(drop=True)

        logger.info("[{}][STEP] postprocess".format(system.upper()))
        postprocess = scan.get("postprocess")
        out_df = postprocess(raw_df) if callable(postprocess) else raw_df

        prep_stats = scan.get("prep_stats", {})
        prep_target = int(prep_stats.get("files_scanned", len(points)))
        prep_success = int(prep_stats.get("files_kept", len(points)))
        prep_fail = max(0, prep_target - prep_success)

        fit_target = int(ana_stats.get("points_total", len(points)))
        fit_success = int(ana_stats.get("points_fitted", 0))
        fit_fail = max(0, fit_target - fit_success)

        logger.info(
            "[{}][PREP] target={} success={} fail={}".format(
                system.upper(), prep_target, prep_success, prep_fail
            )
        )
        logger.info(
            "[{}][FIT ] target={} success={} fail={}".format(
                system.upper(), fit_target, fit_success, fit_fail
            )
        )
        logger.info("[{}][STEP] save_csv".format(system.upper()))
        return self._save_scan_outputs(
            raw_df, out_df, scan.get("out_csv", "csv/results.csv")
        )

    # Subclass must provide points and postprocess setup.
    @abstractmethod
    def prepare_scan(self, system, args):
        """
        Return dict with at least:
        - points: list of point dict
        - out_csv: output CSV path
        Optional:
        - sort_cols: list of sort keys
        - empty_msg: message when no valid rows
        - postprocess: callable(raw_df) -> out_df
        - prep_stats: dict for logging
        """
        raise NotImplementedError

    # Optional abnormal-fit check hook.
    def _is_abnormal_main_fit(self, fit_out):
        return False

    # Run one fit input and normalize return format:
    # success -> (out, meta, 0), failure -> (None, None, 1)
    def _fit_from_input(self, fit_input, inc_bkg):
        if not isinstance(fit_input, dict):
            return None, None, 1

        data = np.asarray(fit_input.get("data", [])).reshape(-1)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return None, None, 1

        out = self.run_step(
            lambda: self.fit(
                FitRequest(
                    coord=fit_input["coord"],
                    data=data,
                    plotname=fit_input.get("plotname"),
                    xr=fit_input.get("xr"),
                    fit_kwargs={"inc_bkg": inc_bkg},
                )
            ),
            file_name=str(fit_input.get("plotname", "fit_input")),
            fail_msg="fit failed",
            log_level="info",
        )
        if out is None:
            return None, None, 1

        meta = dict(fit_input.get("meta", {}))
        if "n_in_window" not in meta:
            meta["n_in_window"] = int(data.size)
        return out, meta, 0

    # Copy fit output fields into one row (optionally with prefix).
    def _attach_fit_fields(self, row, fit_out, prefix=""):
        for out_key, fit_key in self.FIT_FIELD_MAP:
            col = "{}_{}".format(prefix, out_key) if prefix else out_key
            row[col] = fit_out.get(fit_key, np.nan)

    # Print one fit result line (main or aux).
    def _log_fit_result(self, index, total, row, prefix=""):
        file_name = row.get("file", "unknown")
        status = row.get("fit_status", "ok")
        n_key = "{}_n_in_window".format(prefix) if prefix else "n_in_window"
        n_used = row.get(n_key, np.nan)
        label = prefix.upper() if prefix else "MAIN"

        parts = []
        for out_key, _ in self.FIT_FIELD_MAP:
            col = "{}_{}".format(prefix, out_key) if prefix else out_key
            val = row.get(col, np.nan)
            try:
                parts.append("{}={:.6g}".format(col, float(val)))
            except Exception:
                parts.append("{}={}".format(col, val))

        logger.info(
            "[FIT][{}/{}][{}] file={} status={} n={} {}".format(
                index,
                total,
                label,
                file_name,
                status,
                n_used,
                " ".join(parts),
            )
        )

    # Write main fit fields.
    def _attach_main_fit_fields(self, row, fit_out):
        self._attach_fit_fields(row, fit_out, prefix="")

    # Write aux fit fields (for blocks like sipm_*).
    def _attach_aux_fit_fields(self, row, prefix, fit_out):
        self._attach_fit_fields(row, fit_out, prefix=prefix)

    # Iterate all points and run main/aux fits.
    def _analyze_points(self, points, inc_bkg):
        rows = []
        total = int(len(points))
        stats = {
            "points_total": total,
            "points_fitted": 0,
            "points_skipped_main": 0,
            "points_warn_nan": 0,
            "main_fit_errors": 0,
            "aux_fit_errors": 0,
            "aux_fit_failed": 0,
        }

        for index, point in enumerate(points, start=1):
            row = dict(point.get("row", {}))
            row.setdefault("fit_status", "ok")
            logger.info(
                "[FIT][{}/{}] start file={}".format(
                    index, total, row.get("file", "unknown")
                )
            )

            out_main, main_meta, main_errs = self._fit_from_input(
                point.get("main_fit_input"),
                inc_bkg=inc_bkg,
            )
            stats["main_fit_errors"] += int(main_errs)
            if out_main is None:
                msg = point.get("main_skip_msg") or "main fit failed"
                logger.warning(msg)
                stats["points_skipped_main"] += 1
                stats["points_warn_nan"] += 1
                row["fit_status"] = "warning_main_fit_failed"
                if isinstance(main_meta, dict):
                    row.update(main_meta)
                self._attach_main_fit_fields(row, {})
                self._log_fit_result(index, total, row, prefix="")
                rows.append(row)
                continue

            if isinstance(main_meta, dict):
                row.update(main_meta)
            if self._is_abnormal_main_fit(out_main):
                warn_msg = "abnormal fit output at {} -> set main fit fields to NaN".format(
                    row.get("file", "unknown")
                )
                logger.warning(warn_msg)
                row["fit_status"] = "warning_abnormal_fit"
                self._attach_main_fit_fields(row, {})
                stats["points_warn_nan"] += 1
            else:
                self._attach_main_fit_fields(row, out_main)
                stats["points_fitted"] += 1
            self._log_fit_result(index, total, row, prefix="")

            aux_blocks = []
            if isinstance(point.get("aux_blocks"), (list, tuple)):
                aux_blocks.extend([x for x in point.get("aux_blocks") if isinstance(x, dict)])

            for aux in aux_blocks:
                prefix = str(aux.get("prefix", "aux"))
                self._attach_aux_fit_fields(row, prefix, {})

                out_aux, aux_meta, aux_errs = self._fit_from_input(
                    aux.get("fit_input"),
                    inc_bkg=inc_bkg,
                )
                stats["aux_fit_errors"] += int(aux_errs)

                if out_aux is not None:
                    if isinstance(aux_meta, dict) and "n_in_window" in aux_meta:
                        row["{}_n_in_window".format(prefix)] = int(
                            aux_meta.get("n_in_window", 0)
                        )
                    self._attach_aux_fit_fields(row, prefix, out_aux)
                    self._log_fit_result(index, total, row, prefix=prefix)
                elif aux.get("fit_fail_msg"):
                    logger.warning(aux["fit_fail_msg"])
                    stats["aux_fit_failed"] += 1

            rows.append(row)

        return pd.DataFrame(rows), stats

    # Shared CSV writer.
    @staticmethod
    def _save_scan_outputs(raw_df, out_df, out_csv):
        out_csv = Path(out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        return raw_df, out_df
