"""Scan preparation and pipeline."""

from pathlib import Path

import numpy as np


def parse_auto_or_int(value, arg_name):
    # Parse channel config value: "auto" or integer.
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        return int(value)
    except Exception:
        raise SystemExit("{} must be integer or 'auto'. got: {}".format(arg_name, value))


def resolve_inputs(args, default_out_csv, file_pattern, empty_msg):
    # Resolve input_dir/out_csv and discover matching ROOT files.
    input_dir = Path(getattr(args, "input_dir", "")).resolve()
    out_csv = Path(getattr(args, "out_csv", default_out_csv)).resolve()

    files = sorted(input_dir.glob(file_pattern))
    max_files = int(getattr(args, "max_files", 0) or 0)
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise SystemExit(empty_msg.format(input_dir))
    return input_dir, out_csv, files


def make_fit_input(data, coord, plotname, xr, meta=None):
    # Build one standard fit-input record.
    out = {
        "data": data,
        "coord": coord,
        "plotname": plotname,
        "xr": xr,
    }
    if isinstance(meta, dict) and meta:
        out["meta"] = dict(meta)
    return out


def make_point(row, main_fit_input, main_skip_msg=None, aux_blocks=None):
    # Build one standard scan point record.
    point = {
        "row": dict(row),
        "main_fit_input": dict(main_fit_input),
        "aux_blocks": list(aux_blocks) if isinstance(aux_blocks, (list, tuple)) else [],
    }
    if main_skip_msg:
        point["main_skip_msg"] = str(main_skip_msg)
    return point


def inc_stat(stats, key, inc=1):
    # Increase one counter in stats dict.
    if isinstance(stats, dict) and key:
        stats[key] = int(stats.get(key, 0)) + int(inc)


def log_skip(logger, file_name, message, exc=None, level="info"):
    # Unified skip/failure logging format.
    text = "[SKIP] {}: {}".format(file_name, message)
    if exc is not None:
        text = "{}: {}: {}".format(text, type(exc).__name__, exc)
    if str(level).strip().lower() == "warning":
        logger.warning(text)
    else:
        logger.info(text)


def run_step(
    step_fn,
    logger,
    stats=None,
    fail_key=None,
    file_name="",
    fail_msg="step failed",
    log_level="info",
):
    # Run one callable and normalize failure behavior.
    try:
        return step_fn()
    except Exception as exc:
        if fail_key:
            inc_stat(stats, fail_key)
        if file_name:
            log_skip(logger, file_name, fail_msg, exc=exc, level=log_level)
        return None


def log_channel_config(logger, system, serial, resolved=None, cfg=None, defaults=None):
    # Log resolved channels and raw config values.
    resolved = dict(resolved or {})
    cfg = dict(cfg or {})
    defaults = dict(defaults or {})
    logger.info(
        "[%s] serial=%s resolved=%s cfg=%s defaults=%s",
        str(system).upper(),
        serial,
        resolved,
        cfg,
        defaults,
    )


def cut_interval(values, tmin, tmax):
    # Keep finite values inside (tmin, tmax).
    arr = np.asarray(values).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr[(arr > tmin) & (arr < tmax)]


def init_aus_prep_stats(n_files):
    # Initialize AUS preparation counters.
    return {
        "files_scanned": int(n_files),
        "files_kept": 0,
        "window_fail": 0,
        "pmt_read_fail": 0,
        "pmt_empty_window": 0,
        "sipm_read_fail": 0,
        "sipm_window_fail": 0,
        "sipm_empty_window": 0,
    }


def init_kor_prep_stats(n_files):
    # Initialize KOR preparation counters.
    return {
        "files_scanned": int(n_files),
        "files_kept": 0,
        "diff_read_fail": 0,
        "window_fail": 0,
        "empty_window": 0,
    }


def resolve_aus_context(args, defaults, resolve_channel_fn):
    # Resolve AUS serial/channels from args + defaults.
    serial = str(getattr(args, "serial", "") or "")
    trigger_src = getattr(args, "trigger_ch", None)
    if trigger_src is None:
        trigger_src = getattr(args, "laser_ch", None)
    if trigger_src is None:
        trigger_src = "auto"

    pmt_ch_cfg = getattr(args, "pmt_ch", "auto")
    sipm_ch_cfg = getattr(args, "sipm_ch", "auto")
    trigger_ch_cfg = trigger_src

    pmt_ch = resolve_channel_fn(pmt_ch_cfg, "pmt_ch", defaults["pmt"])
    sipm_ch = resolve_channel_fn(sipm_ch_cfg, "sipm_ch", defaults["sipm"])
    trigger_ch = resolve_channel_fn(trigger_ch_cfg, "trigger_ch", defaults["trigger"])

    return {
        "serial": serial,
        "pmt_ch": int(pmt_ch),
        "sipm_ch": int(sipm_ch),
        "trigger_ch": int(trigger_ch),
        "pmt_ch_cfg": str(pmt_ch_cfg),
        "sipm_ch_cfg": str(sipm_ch_cfg),
        "trigger_ch_cfg": str(trigger_ch_cfg),
    }


def build_aus_row(fp_name, ctx, defaults, theta, phi, pmt_tmin, pmt_tmax, pmt_peak):
    # Build one AUS output row for one angle point.
    return {
        "system": "aus",
        "file": fp_name,
        "serial": ctx["serial"],
        "pmt_ch": int(ctx["pmt_ch"]),
        "trigger_ch": int(ctx["trigger_ch"]),
        "sipm_ch": int(ctx["sipm_ch"]),
        "pmt_ch_cfg": str(ctx["pmt_ch_cfg"]),
        "trigger_ch_cfg": str(ctx["trigger_ch_cfg"]),
        "sipm_ch_cfg": str(ctx["sipm_ch_cfg"]),
        "default_trigger_ch": int(defaults["trigger"]),
        "default_sipm_ch": int(defaults["sipm"]),
        "default_pmt_ch": int(defaults["pmt"]),
        "theta": int(theta),
        "phi": int(phi),
        "window_min": float(pmt_tmin),
        "window_max": float(pmt_tmax),
        "peak": float(pmt_peak),
        "sipm_window_min": np.nan,
        "sipm_window_max": np.nan,
        "sipm_n_in_window": 0,
    }


def resolve_kor_channels(
    args,
    files_for_auto,
    parse_auto_or_int_fn,
    auto_pick_trigger_channel_fn,
    auto_pick_channel_fn,
):
    # Resolve KOR channel/trigger from args + auto mapping.
    trigger_ch_cfg = parse_auto_or_int_fn(args.trigger_ch, "trigger_ch")
    if trigger_ch_cfg == "auto":
        trigger_ch = auto_pick_trigger_channel_fn()
    else:
        trigger_ch = int(trigger_ch_cfg)

    channel_cfg = parse_auto_or_int_fn(args.channel, "channel")
    if channel_cfg == "auto":
        ch = auto_pick_channel_fn(files_for_auto, serial=args.serial, trigger_ch=trigger_ch)
    else:
        ch = int(channel_cfg)

    return {
        "channel": int(ch),
        "trigger_ch": int(trigger_ch),
        "channel_cfg": str(args.channel),
        "trigger_ch_cfg": str(args.trigger_ch),
    }


def build_kor_row(fp_name, serial, ctx, phi, theta_raw, window_method, peak):
    # Build one KOR output row for one angle point.
    return {
        "system": "kor",
        "file": fp_name,
        "serial": serial,
        "channel": int(ctx["channel"]),
        "trigger_ch": int(ctx["trigger_ch"]),
        "channel_cfg": str(ctx["channel_cfg"]),
        "trigger_ch_cfg": str(ctx["trigger_ch_cfg"]),
        "phi_raw": int(phi),
        "theta_raw": int(theta_raw),
        "window_method": str(window_method),
        "window_min": np.nan,
        "window_max": np.nan,
        "peak": float(peak),
        "n_in_window": 0,
    }
