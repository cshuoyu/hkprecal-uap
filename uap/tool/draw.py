"""CSV point selection and line plotting helpers."""

import ast
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def parse_angle_pairs(text):
    # Parse angle list like: [[270,10],[270,20],[90,20]] (phi, theta).
    try:
        obj = ast.literal_eval(str(text))
    except Exception as exc:
        raise SystemExit("Invalid --angle-pairs: {} ({})".format(text, exc))

    if not isinstance(obj, (list, tuple)):
        raise SystemExit("--angle-pairs must be a list of [phi, theta] pairs.")

    out = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SystemExit("Each angle entry must be [phi, theta], got: {}".format(item))
        out.append((float(item[0]), float(item[1])))
    return out


def parse_num_list(text):
    # Parse numeric list like: [-50,-40,-30,-20,-10].
    try:
        obj = ast.literal_eval(str(text))
    except Exception as exc:
        raise SystemExit("Invalid numeric list: {} ({})".format(text, exc))
    if not isinstance(obj, (list, tuple)):
        raise SystemExit("Numeric list must be list/tuple.")
    out = []
    for item in obj:
        out.append(float(item))
    return out


def parse_series_json(text):
    # Parse overlay series spec from JSON/Python-literal string.
    try:
        obj = ast.literal_eval(str(text))
    except Exception as exc:
        raise SystemExit("Invalid --series-json: {} ({})".format(text, exc))
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, (list, tuple)):
        raise SystemExit("--series-json must be list[dict] or dict.")
    out = []
    for item in obj:
        if not isinstance(item, dict):
            raise SystemExit("Each series entry must be dict, got: {}".format(type(item)))
        out.append(dict(item))
    return out


def as_bool(value, default=False):
    # Convert bool-like values from config/CLI.
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    if txt in ("1", "true", "yes", "y", "on"):
        return True
    if txt in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def default_angle_columns(system):
    # Default angle columns per system.
    if str(system).strip().lower() == "kor":
        return "phi_raw", "theta_raw"
    return "phi", "theta"


def resolve_angle_columns(df, system, phi_col="", theta_col=""):
    # Resolve usable angle columns from explicit args or system defaults.
    p_col = str(phi_col).strip()
    t_col = str(theta_col).strip()

    if not p_col or not t_col:
        p_def, t_def = default_angle_columns(system)
        if not p_col:
            p_col = p_def if p_def in df.columns else "phi"
        if not t_col:
            t_col = t_def if t_def in df.columns else "theta"

    if p_col not in df.columns:
        raise SystemExit("phi column not found: {}".format(p_col))
    if t_col not in df.columns:
        raise SystemExit("theta column not found: {}".format(t_col))
    return p_col, t_col


def load_csv(csv_path):
    # Read one CSV file.
    p = Path(csv_path).resolve()
    if not p.is_file():
        raise SystemExit("CSV not found: {}".format(p))
    return pd.read_csv(p), p


def resolve_value_column(df, column_name):
    # Resolve old/new compatible value-column names.
    col = str(column_name).strip()
    if not col:
        return ""
    if col in df.columns:
        return col

    alias = {
        "rel_norm": "relative_norm",
        "relative_norm": "rel_norm",
        "rel_norm_err": "relative_norm_err",
        "relative_norm_err": "rel_norm_err",
    }
    alt = alias.get(col, "")
    if alt and alt in df.columns:
        logger.info("[CSV2PLOT] column '%s' not found, use alias '%s'.", col, alt)
        return alt
    return col


def select_points_by_angles(df, angle_pairs, phi_col, theta_col, y_col, yerr_col=""):
    # Select y values by user-specified (phi, theta) list, preserving order.
    if y_col not in df.columns:
        raise SystemExit("y column not found: {}".format(y_col))
    has_yerr = bool(yerr_col) and yerr_col in df.columns

    phi_vals = pd.to_numeric(df[phi_col], errors="coerce").values
    theta_vals = pd.to_numeric(df[theta_col], errors="coerce").values
    y_vals = pd.to_numeric(df[y_col], errors="coerce").values
    yerr_vals = pd.to_numeric(df[yerr_col], errors="coerce").values if has_yerr else None

    rows = []
    for phi, theta in angle_pairs:
        mask = np.isfinite(phi_vals) & np.isfinite(theta_vals)
        mask &= np.isclose(phi_vals, float(phi), atol=1e-8)
        mask &= np.isclose(theta_vals, float(theta), atol=1e-8)

        if np.any(mask):
            y = np.nanmean(y_vals[mask])
            yerr = np.nanmean(yerr_vals[mask]) if has_yerr else np.nan
            found = True
        else:
            y = np.nan
            yerr = np.nan
            found = False

        rows.append(
            {
                "phi": float(phi),
                "theta": float(theta),
                "y": float(y) if np.isfinite(y) else np.nan,
                "yerr": float(yerr) if np.isfinite(yerr) else np.nan,
                "found": bool(found),
            }
        )

    out = pd.DataFrame(rows)
    missing = int((~out["found"]).sum())
    if missing > 0:
        logger.warning("Missing %s/%s requested angle points in CSV.", missing, len(out))
    return out


def _aggregate_xy_rows(rows):
    # Aggregate duplicate x values with mean (and mean yerr if provided).
    if not rows:
        return pd.DataFrame(columns=["x", "y", "yerr"])
    out = pd.DataFrame(rows)
    if "yerr" not in out.columns:
        out["yerr"] = np.nan
    out = (
        out.groupby("x", as_index=False)
        .agg(
            y=("y", "mean"),
            yerr=("yerr", "mean"),
        )
        .sort_values("x")
        .reset_index(drop=True)
    )
    return out


def build_phi_pair_profile(
    df,
    phi_col,
    theta_col,
    y_col,
    yerr_col,
    phi_pos,
    phi_neg,
    center_phi=None,
    center_theta=None,
    center_value=None,
):
    # Build signed-theta profile from two azimuth branches:
    # +theta from phi_pos, -theta from phi_neg.
    phi_vals = pd.to_numeric(df[phi_col], errors="coerce").values
    theta_vals = pd.to_numeric(df[theta_col], errors="coerce").values
    y_vals = pd.to_numeric(df[y_col], errors="coerce").values
    has_yerr = bool(yerr_col) and yerr_col in df.columns
    yerr_vals = pd.to_numeric(df[yerr_col], errors="coerce").values if has_yerr else None

    rows = []
    for i in range(len(df)):
        phi = phi_vals[i]
        theta = theta_vals[i]
        y = y_vals[i]
        if not (np.isfinite(phi) and np.isfinite(theta) and np.isfinite(y)):
            continue
        if np.isclose(phi, float(phi_pos), atol=1e-8):
            x = abs(float(theta))
        elif np.isclose(phi, float(phi_neg), atol=1e-8):
            x = -abs(float(theta))
        else:
            continue
        row = {"x": x, "y": float(y)}
        row["yerr"] = float(yerr_vals[i]) if has_yerr and np.isfinite(yerr_vals[i]) else np.nan
        rows.append(row)

    # Optional center point injection.
    if center_phi is not None and center_theta is not None:
        center_mask = np.isfinite(phi_vals) & np.isfinite(theta_vals)
        center_mask &= np.isclose(phi_vals, float(center_phi), atol=1e-8)
        center_mask &= np.isclose(theta_vals, float(center_theta), atol=1e-8)
        if np.any(center_mask):
            yc = float(np.nanmean(y_vals[center_mask]))
            yec = (
                float(np.nanmean(yerr_vals[center_mask]))
                if has_yerr and np.isfinite(yerr_vals[center_mask]).any()
                else np.nan
            )
            rows.append({"x": 0.0, "y": yc, "yerr": yec})
    elif center_value is not None:
        rows.append({"x": 0.0, "y": float(center_value), "yerr": np.nan})

    return _aggregate_xy_rows(rows)


def build_single_phi_profile(
    df,
    phi_col,
    theta_col,
    y_col,
    yerr_col,
    phi_value,
    flip_sign=False,
    use_abs_theta=False,
):
    # Build profile from one azimuth branch:
    # x = theta (optionally abs/flip).
    phi_vals = pd.to_numeric(df[phi_col], errors="coerce").values
    theta_vals = pd.to_numeric(df[theta_col], errors="coerce").values
    y_vals = pd.to_numeric(df[y_col], errors="coerce").values
    has_yerr = bool(yerr_col) and yerr_col in df.columns
    yerr_vals = pd.to_numeric(df[yerr_col], errors="coerce").values if has_yerr else None

    rows = []
    for i in range(len(df)):
        phi = phi_vals[i]
        theta = theta_vals[i]
        y = y_vals[i]
        if not (np.isfinite(phi) and np.isfinite(theta) and np.isfinite(y)):
            continue
        if not np.isclose(phi, float(phi_value), atol=1e-8):
            continue
        x = abs(float(theta)) if use_abs_theta else float(theta)
        if as_bool(flip_sign, False):
            x = -x
        row = {"x": x, "y": float(y)}
        row["yerr"] = float(yerr_vals[i]) if has_yerr and np.isfinite(yerr_vals[i]) else np.nan
        rows.append(row)

    return _aggregate_xy_rows(rows)


def build_x_values(points_df, x_source="theta", x_values=""):
    # Build x values from theta/custom/index strategy.
    x_mode = str(x_source).strip().lower()
    if x_values:
        custom = parse_num_list(x_values)
        if len(custom) != len(points_df):
            raise SystemExit("Length mismatch: x_values={} points={}".format(len(custom), len(points_df)))
        return np.asarray(custom, dtype=float)

    if x_mode == "theta":
        return points_df["theta"].astype(float).values
    if x_mode == "index":
        return np.arange(len(points_df), dtype=float)
    if x_mode == "custom":
        raise SystemExit("x_source=custom requires --x-values.")
    raise SystemExit("Unsupported x_source: {}".format(x_source))


def apply_hamamatsu_angle(x_values, use_hamamatsu_angle=False):
    # Optional KR->Hamamatsu conversion with fixed formula.
    x = np.asarray(x_values, dtype=float)
    if not as_bool(use_hamamatsu_angle, False):
        return x
    x_abs = np.abs(x)
    ham_abs = -0.0049 * np.power(x_abs, 2) + 1.7515 * x_abs - 0.0402
    sign = np.where(x < 0.0, -1.0, 1.0)
    return sign * ham_abs


def plot_line(
    x,
    y,
    yerr=None,
    out_png="plot.png",
    line_color="C0",
    marker="o",
    line_style="-",
    line_width=1.4,
    marker_size=4.5,
    x_label="theta (deg)",
    y_label="",
    title="",
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    legend_label="",
):
    # Draw a single line (with optional error bars).
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=170)

    if yerr is not None and np.isfinite(np.asarray(yerr, dtype=float)).any():
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker + line_style,
            color=line_color,
            lw=float(line_width),
            ms=float(marker_size),
            capsize=2.5,
            label=legend_label if legend_label else None,
        )
    else:
        ax.plot(
            x,
            y,
            marker=marker,
            linestyle=line_style,
            color=line_color,
            lw=float(line_width),
            ms=float(marker_size),
            label=legend_label if legend_label else None,
        )

    ax.grid(alpha=0.30)
    ax.set_xlabel(x_label if x_label else "x")
    ax.set_ylabel(y_label if y_label else "y")
    if title:
        ax.set_title(title)
    if x_min is not None and x_max is not None and float(x_max) > float(x_min):
        ax.set_xlim(float(x_min), float(x_max))
    if y_min is not None and y_max is not None and float(y_max) > float(y_min):
        ax.set_ylim(float(y_min), float(y_max))
    if legend_label:
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(str(out_png))
    plt.close(fig)


def build_line_from_series_cfg(series_cfg):
    # Build one plotting line from one series config.
    cfg = dict(series_cfg or {})
    if "csv" not in cfg:
        raise SystemExit("series entry missing 'csv': {}".format(cfg))
    if "y_col" not in cfg:
        raise SystemExit("series entry missing 'y_col': {}".format(cfg))

    system = str(cfg.get("system", "aus")).strip().lower()
    mode = str(cfg.get("mode", "angle_pairs")).strip().lower()
    y_col = str(cfg.get("y_col"))
    yerr_col = str(cfg.get("yerr_col", "")).strip()

    df, csv_path = load_csv(cfg["csv"])
    y_col = resolve_value_column(df, y_col)
    yerr_col = resolve_value_column(df, yerr_col)
    phi_col, theta_col = resolve_angle_columns(
        df=df,
        system=system,
        phi_col=cfg.get("phi_col", ""),
        theta_col=cfg.get("theta_col", ""),
    )

    if mode in ("xy_columns", "direct_xy"):
        x_col = str(cfg.get("x_col", "")).strip()
        if not x_col:
            raise SystemExit("series mode=xy_columns requires x_col")
        if x_col not in df.columns:
            raise SystemExit("x_col not found: {}".format(x_col))
        if y_col not in df.columns:
            raise SystemExit("y_col not found: {}".format(y_col))
        out = pd.DataFrame(
            {
                "x": pd.to_numeric(df[x_col], errors="coerce"),
                "y": pd.to_numeric(df[y_col], errors="coerce"),
                "yerr": pd.to_numeric(df[yerr_col], errors="coerce") if yerr_col and yerr_col in df.columns else np.nan,
            }
        )
    elif mode in ("angle_pairs",):
        angle_pairs = cfg.get("angle_pairs")
        if not angle_pairs:
            raise SystemExit("series mode=angle_pairs requires angle_pairs")
        if isinstance(angle_pairs, str):
            angle_pairs = parse_angle_pairs(angle_pairs)
        points = select_points_by_angles(
            df=df,
            angle_pairs=angle_pairs,
            phi_col=phi_col,
            theta_col=theta_col,
            y_col=y_col,
            yerr_col=yerr_col,
        )
        x = build_x_values(
            points_df=points,
            x_source=cfg.get("x_source", "theta"),
            x_values=cfg.get("x_values", ""),
        )
        out = pd.DataFrame(
            {
                "x": np.asarray(x, dtype=float),
                "y": points["y"].astype(float).values,
                "yerr": points["yerr"].astype(float).values,
            }
        )
    elif mode in ("phi_pair", "hamamatsu_phi_pair"):
        out = build_phi_pair_profile(
            df=df,
            phi_col=phi_col,
            theta_col=theta_col,
            y_col=y_col,
            yerr_col=yerr_col,
            phi_pos=cfg.get("phi_pos"),
            phi_neg=cfg.get("phi_neg"),
            center_phi=cfg.get("center_phi", None),
            center_theta=cfg.get("center_theta", None),
            center_value=cfg.get("center_value", None),
        )
    elif mode in ("single_phi", "kor_hamamatsu_single_phi"):
        out = build_single_phi_profile(
            df=df,
            phi_col=phi_col,
            theta_col=theta_col,
            y_col=y_col,
            yerr_col=yerr_col,
            phi_value=cfg.get("phi"),
            flip_sign=cfg.get("flip_sign", False),
            use_abs_theta=cfg.get("use_abs_theta", False),
        )
    else:
        raise SystemExit("Unknown series mode: {}".format(mode))

    out = out.copy()
    out["x_raw"] = pd.to_numeric(out["x"], errors="coerce")
    out["x"] = apply_hamamatsu_angle(
        out["x_raw"].values, use_hamamatsu_angle=cfg.get("use_hamamatsu_angle", False)
    )
    out = out.dropna(subset=["x", "y"]).sort_values("x").reset_index(drop=True)
    label = str(cfg.get("label", "{}:{}".format(system, csv_path.name)))

    style = {
        "label": label,
        "color": cfg.get("line_color", cfg.get("color", "C0")),
        "marker": cfg.get("marker", "o"),
        "linestyle": cfg.get("line_style", cfg.get("linestyle", "-")),
        "linewidth": float(cfg.get("line_width", cfg.get("linewidth", 1.4))),
        "markersize": float(cfg.get("marker_size", cfg.get("markersize", 4.5))),
    }
    return out, style


def plot_multi_lines(
    lines,
    out_png,
    x_label="theta (deg)",
    y_label="",
    title="",
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    hline=None,
):
    # Draw multiple lines in one figure.
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=170)
    for line in lines:
        data = line["data"]
        style = line["style"]
        x = data["x"].values
        y = data["y"].values
        yerr = data["yerr"].values if "yerr" in data.columns else np.full_like(y, np.nan, dtype=float)
        if np.isfinite(yerr).any():
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=str(style.get("marker", "o")) + str(style.get("linestyle", "-")),
                color=style.get("color", "C0"),
                lw=float(style.get("linewidth", 1.4)),
                ms=float(style.get("markersize", 4.5)),
                capsize=2.5,
                label=style.get("label", ""),
            )
        else:
            ax.plot(
                x,
                y,
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                color=style.get("color", "C0"),
                lw=float(style.get("linewidth", 1.4)),
                ms=float(style.get("markersize", 4.5)),
                label=style.get("label", ""),
            )

    if hline is not None:
        ax.axhline(float(hline), ls="--", c="gray", lw=1.0)
    ax.grid(alpha=0.30)
    ax.set_xlabel(x_label if x_label else "x")
    ax.set_ylabel(y_label if y_label else "y")
    if title:
        ax.set_title(title)
    if x_min is not None and x_max is not None and float(x_max) > float(x_min):
        ax.set_xlim(float(x_min), float(x_max))
    if y_min is not None and y_max is not None and float(y_max) > float(y_min):
        ax.set_ylim(float(y_min), float(y_max))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(str(out_png))
    plt.close(fig)
