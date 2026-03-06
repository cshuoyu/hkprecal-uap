#!/usr/bin/env python3
"""CSV -> plot pipeline for AUS/KOR scan results."""

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uap.tool.draw import build_x_values
from uap.tool.draw import build_line_from_series_cfg
from uap.tool.draw import plot_multi_lines
from uap.tool.draw import parse_series_json
from uap.tool.draw import plot_line
from uap.tool.draw import resolve_angle_columns
from uap.tool.draw import select_points_by_angles
from uap.tool.draw import parse_angle_pairs

logger = logging.getLogger(__name__)


def make_signed_line(df, value_col, phi_pos, phi_neg, center_value):
    sub = df[df["phi"].isin([phi_pos, phi_neg])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["theta_signed", value_col])
    sub["theta_signed"] = np.where(sub["phi"] == phi_pos, sub["theta"].abs(), -sub["theta"].abs())
    sub = sub[["theta_signed", value_col]].dropna()
    sub = pd.concat(
        [sub, pd.DataFrame({"theta_signed": [0.0], value_col: [center_value]})],
        ignore_index=True,
    )
    sub = sub.sort_values("theta_signed").drop_duplicates(subset=["theta_signed"], keep="first")
    return sub


def plot_single_line(data, value_col, ylabel, legend_label, title, out_png):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(data["theta_signed"], data[value_col], "o-", lw=1.3, ms=4, label=legend_label)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def ensure_kor_rel(df):
    if "rel_yield" in df.columns:
        out = df.copy()
        if "rel_err" not in out.columns:
            out["rel_err"] = np.nan
        return out

    req = {"axis", "theta_signed", "sig_yield"}
    if not req.issubset(set(df.columns)):
        raise SystemExit("KOR CSV must contain rel_yield or (axis, theta_signed, sig_yield).")

    grp = (
        df.groupby(["axis", "theta_signed"], as_index=False)
        .agg(
            sig_yield=("sig_yield", "mean"),
            sig_std=("sig_yield", "std"),
            sig_count=("sig_yield", "count"),
            sig_err_mean=("sig_err", "mean") if "sig_err" in df.columns else ("sig_yield", "std"),
        )
        .sort_values(["axis", "theta_signed"])
        .reset_index(drop=True)
    )
    grp["sig_std"] = grp["sig_std"].fillna(0.0)
    grp["sig_sem"] = np.where(grp["sig_count"] > 1, grp["sig_std"] / np.sqrt(grp["sig_count"]), grp["sig_err_mean"])

    rel_rows = []
    for axis_name in ["x", "y"]:
        sub = grp[grp["axis"] == axis_name].copy()
        if sub.empty:
            continue
        center = sub[sub["theta_signed"] == 0]
        if center.empty:
            continue
        y0 = float(center.iloc[0]["sig_yield"])
        y0err = float(center.iloc[0]["sig_sem"]) if np.isfinite(center.iloc[0]["sig_sem"]) else np.nan
        if not np.isfinite(y0) or y0 == 0:
            continue
        sub["rel_yield"] = sub["sig_yield"] / y0
        if np.isfinite(y0err) and y0err >= 0:
            sub["rel_err"] = sub["rel_yield"] * np.sqrt(
                (sub["sig_sem"] / sub["sig_yield"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) ** 2 + (y0err / y0) ** 2
            )
        else:
            sub["rel_err"] = np.nan
        rel_rows.append(sub)

    if not rel_rows:
        raise SystemExit("No KOR axis has theta=0 to normalize.")
    out = pd.concat(rel_rows, ignore_index=True).sort_values(["axis", "theta_signed"]).reset_index(drop=True)
    return out


def plot_aus_standard(args):
    csv = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)

    if args.value_col == "auto":
        if "rel_norm" in df.columns and df["rel_norm"].notna().any():
            value_col = "rel_norm"
        elif "rel_sig" in df.columns:
            value_col = "rel_sig"
        else:
            raise SystemExit("AUS CSV has neither rel_norm nor rel_sig.")
    else:
        value_col = args.value_col
        if value_col not in df.columns:
            raise SystemExit("Requested value_col not in CSV: {}".format(value_col))

    center_value = 1.0
    ylabel = "relative DE (SiPM normalized)" if value_col == "rel_norm" else "relative signal yield"
    legend_head = "PMT (SiPM norm)" if value_col == "rel_norm" else "PMT (signal yield)"

    x_line = make_signed_line(df, value_col, args.x_phi_plus, args.x_phi_minus, center_value)
    y_line = make_signed_line(df, value_col, args.y_phi_plus, args.y_phi_minus, center_value)
    if x_line.empty or y_line.empty:
        raise SystemExit("Not enough AUS points for standard plotting.")

    plot_single_line(
        x_line,
        value_col,
        ylabel,
        "{} x axis".format(legend_head),
        "x axis",
        out_dir / "{}_phi{}_{}.png".format(value_col, args.x_phi_plus, args.x_phi_minus),
    )
    plot_single_line(
        y_line,
        value_col,
        ylabel,
        "{} y axis".format(legend_head),
        "y axis",
        out_dir / "{}_phi{}_{}.png".format(value_col, args.y_phi_plus, args.y_phi_minus),
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=170)
    ax.plot(x_line["theta_signed"], x_line[value_col], "o-", lw=1.4, ms=4, label="{} x axis".format(legend_head))
    ax.plot(y_line["theta_signed"], y_line[value_col], "s-", lw=1.4, ms=4, label="{} y axis".format(legend_head))
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title("{}: x and y axes".format(legend_head))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / args.xy_out_name)
    plt.close(fig)

    logger.info("[DONE] system=aus style=standard")
    logger.info("[DONE] out dir -> %s", out_dir)


def plot_kor_standard(args):
    csv = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    rel_df = ensure_kor_rel(df)

    serial = args.serial
    if not serial:
        if "serial" in df.columns and df["serial"].dropna().nunique() == 1:
            serial = str(df["serial"].dropna().iloc[0])
        else:
            serial = "PMT"

    xdf = rel_df[rel_df["axis"] == "x"].copy()
    ydf = rel_df[rel_df["axis"] == "y"].copy()
    if xdf.empty or ydf.empty:
        raise SystemExit("KOR CSV must contain both x and y axis rows.")

    def plot_line(df_axis, out_png, title):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.errorbar(
            df_axis["theta_signed"],
            df_axis["rel_yield"],
            yerr=df_axis["rel_err"] if ("rel_err" in df_axis.columns and df_axis["rel_err"].notna().any()) else None,
            fmt="o-",
            lw=1.3,
            ms=4,
            capsize=2,
            label=title,
        )
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("relative yield (diff fit)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)

    plot_line(xdf, out_dir / "kor_diff_rel_x_{}.png".format(serial), "PMT (diff zfit) x axis")
    plot_line(ydf, out_dir / "kor_diff_rel_y_{}.png".format(serial), "PMT (diff zfit) y axis")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=170)
    ax.errorbar(
        xdf["theta_signed"],
        xdf["rel_yield"],
        yerr=xdf["rel_err"] if ("rel_err" in xdf.columns and xdf["rel_err"].notna().any()) else None,
        fmt="o-",
        lw=1.4,
        ms=4,
        capsize=2,
        label="PMT (diff zfit) x axis",
    )
    ax.errorbar(
        ydf["theta_signed"],
        ydf["rel_yield"],
        yerr=ydf["rel_err"] if ("rel_err" in ydf.columns and ydf["rel_err"].notna().any()) else None,
        fmt="s-",
        lw=1.4,
        ms=4,
        capsize=2,
        label="PMT (diff zfit) y axis",
    )
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("relative yield (diff fit)")
    ax.set_title("{}: x and y axes".format(serial))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "kor_diff_rel_xy_{}.png".format(serial))
    plt.close(fig)

    rel_df.to_csv(out_dir / "kor_diff_rel_{}.csv".format(serial), index=False)
    logger.info("[DONE] system=kor style=standard")
    logger.info("[DONE] out dir -> %s", out_dir)


def plot_manual(args):
    csv = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    if not args.angle_pairs:
        raise SystemExit("manual style requires --angle-pairs, e.g. '[[270,10],[270,20],[90,20]]'")
    if not args.y_col:
        raise SystemExit("manual style requires --y-col")

    angle_pairs = parse_angle_pairs(args.angle_pairs)
    phi_col, theta_col = resolve_angle_columns(df, args.system, args.phi_col, args.theta_col)
    points = select_points_by_angles(
        df=df,
        angle_pairs=angle_pairs,
        phi_col=phi_col,
        theta_col=theta_col,
        y_col=args.y_col,
        yerr_col=args.yerr_col,
    )
    x = build_x_values(points, x_source=args.x_source, x_values=args.x_values)
    y = points["y"].values
    yerr = points["yerr"].values if args.yerr_col else None

    out_png = out_dir / args.out_name
    plot_line(
        x=x,
        y=y,
        yerr=yerr,
        out_png=out_png,
        line_color=args.line_color,
        marker=args.marker,
        line_style=args.line_style,
        line_width=args.line_width,
        marker_size=args.marker_size,
        x_label=args.x_label if args.x_label else ("theta (deg)" if args.x_source == "theta" else "x"),
        y_label=args.y_label if args.y_label else args.y_col,
        title=args.title,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        legend_label=args.legend_label,
    )
    points.to_csv(out_dir / args.points_csv, index=False)
    logger.info("[DONE] system=%s style=manual", args.system)
    logger.info("[DONE] out plot -> %s", out_png)
    logger.info("[DONE] out selected points -> %s", out_dir / args.points_csv)


def _load_overlay_series(args):
    lines = getattr(args, "lines", None)
    line_order = getattr(args, "line_order", None)
    if isinstance(lines, dict) and len(lines) > 0:
        if isinstance(line_order, (list, tuple)) and len(line_order) > 0:
            keys = [str(k) for k in line_order]
        else:
            keys = list(lines.keys())

        out = []
        for key in keys:
            if key not in lines:
                raise SystemExit("line_order contains unknown key: {}".format(key))
            cfg = dict(lines[key] or {})
            cfg.setdefault("label", str(key))
            out.append(cfg)
        if out:
            return out

    series = getattr(args, "series", None)
    if isinstance(series, (list, tuple)) and len(series) > 0:
        return [dict(x) for x in series]
    series_json = str(getattr(args, "series_json", "") or "").strip()
    if series_json:
        return parse_series_json(series_json)
    raise SystemExit("overlay style requires lines+line_order (or series/--series-json).")


def plot_overlay(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    series_list = _load_overlay_series(args)

    lines = []
    merged_rows = []
    for idx, cfg in enumerate(series_list):
        data, style = build_line_from_series_cfg(cfg)
        lines.append({"data": data, "style": style})
        tmp = data.copy()
        tmp["series_index"] = int(idx)
        tmp["series_label"] = style.get("label", "series_{}".format(idx))
        merged_rows.append(tmp)

    out_png = out_dir / args.out_name
    plot_multi_lines(
        lines=lines,
        out_png=out_png,
        x_label=args.x_label if args.x_label else "Position angle (deg)",
        y_label=args.y_label if args.y_label else "value",
        title=args.title,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        hline=args.hline,
    )

    merged = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
    merged.to_csv(out_dir / args.points_csv, index=False)
    logger.info("[DONE] style=overlay")
    logger.info("[DONE] out plot -> %s", out_png)
    logger.info("[DONE] out selected points -> %s", out_dir / args.points_csv)


def build_parser():
    ap = argparse.ArgumentParser(description="CSV -> plot pipeline (overlay only).")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--out-base", default="", help="Run output root (default: <UAP_HOME>/outputs)")
    ap.add_argument("--x-label", default="")
    ap.add_argument("--y-label", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--x-min", type=float, default=None)
    ap.add_argument("--x-max", type=float, default=None)
    ap.add_argument("--y-min", type=float, default=None)
    ap.add_argument("--y-max", type=float, default=None)
    ap.add_argument("--hline", type=float, default=None)
    ap.add_argument("--out-name", default="manual_plot.png")
    ap.add_argument("--points-csv", default="manual_points.csv")
    ap.add_argument("--series-json", default="", help="Overlay series list string")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    return args


def _prepare_output_layout(args):
    out_dir = str(getattr(args, "out_dir", "") or "").strip()
    out_base = str(getattr(args, "out_base", "") or "").strip()
    if not out_dir:
        if out_base:
            out_dir = str(Path(out_base).resolve() / "figures")
        else:
            out_dir = str(Path("figures"))
    args.out_dir = out_dir
    args.out_base = out_base
    Path(args.out_dir).resolve().mkdir(parents=True, exist_ok=True)


def _ensure_main_log_file(args):
    out_dir = Path(args.out_dir).resolve()
    run_dir = out_dir.parent
    log_path = run_dir / "main.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    target = str(log_path.resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and str(
            Path(handler.baseFilename).resolve()
        ) == target:
            return

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    file_handler = logging.FileHandler(target)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def _prepend_hydra_config_to_main_log(args):
    out_dir = Path(args.out_dir).resolve()
    run_dir = out_dir.parent
    main_log_path = run_dir / "main.log"
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    marker_begin = "===== HYDRA CONFIG (resolved) ====="
    marker_end = "===== END HYDRA CONFIG ====="

    if not hydra_cfg_path.is_file():
        return

    cfg_text = hydra_cfg_path.read_text(encoding="utf-8").strip()
    if not cfg_text:
        return

    existing = main_log_path.read_text(encoding="utf-8") if main_log_path.exists() else ""
    if marker_begin in existing:
        return

    header = "{}\n{}\n{}\n\n".format(marker_begin, cfg_text, marker_end)
    main_log_path.write_text(header + existing, encoding="utf-8")


def run(args):
    _prepare_output_layout(args)
    _ensure_main_log_file(args)
    _prepend_hydra_config_to_main_log(args)
    logger.info(
        "[START] style=overlay out_dir=%s",
        Path(args.out_dir).resolve(),
    )
    plot_overlay(args)
