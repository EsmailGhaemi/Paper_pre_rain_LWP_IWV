"""
Data loading and summary figures
================================

    load_station                   WegenerNet L2 v8 station CSV -> xarray
    resample_to_2p5min             radiometer profiles on the 2.5-min L2a time grid
    plot_lwp_boxen, plot_iwv_boxen distributions of LWP and IWV (f05)
    plot_combined_lwp_iwv_summary  dLWP-dIWV scatter and trajectory (f09)

Note: plot_lwp_boxen / plot_iwv_boxen call seaborn.set_theme(), which changes the
global matplotlib style. Restore the style afterwards (see notebook 01, f05).
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from plot_functions import (
    _corr,
    _pstars,
    plot_lwp_iwv_anomaly_scatter,
    plot_lwp_iwv_trajectory,
)

# WegenerNet L2 v8 station files (one CSV per station), see data/README.md
STATION_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "stations"

COLUMN_MAP = {
    "Relative humidity [%]": "RH",
    "Relative humidity DP-Flag []": "RH_flag",
    "Precipitation [mm]": "precip",
    "Precipitation DP-Flag []": "precip_flag",
    "Air temperature [degC]": "temp",
    "Air temperature DP-Flag []": "temp_flag",
}
_REGIMES = ["Warm", "Cold", "No rain"]
_PALETTE = {"Warm": "#d62728", "Cold": "#1f77b4", "No rain": "#7f7f7f"}


def resample_to_2p5min(ds, bin_seconds=150):
    """Resample an xarray Dataset to fixed time bins using fast bincount."""
    ds = ds.load()

    time_vals = ds.time.values
    origin = time_vals[0].astype("datetime64[D]")
    offset_s = (time_vals - origin).astype("timedelta64[s]").astype(np.int64)
    bin_idx = offset_s // bin_seconds
    bin_idx -= bin_idx[0]
    n_bins = int(bin_idx[-1] + 1)

    first_bin_offset = (offset_s[0] // bin_seconds) * bin_seconds
    new_time = origin + (np.arange(n_bins) * bin_seconds + first_bin_offset).astype(
        "timedelta64[s]"
    )

    resampled_vars = {}
    extra_coords = {}

    for vname in ds.data_vars:
        da = ds[vname].values.astype(np.float64)
        valid = ~np.isnan(da)

        if da.ndim == 1:
            sums = np.bincount(
                bin_idx, weights=np.where(valid, da, 0.0), minlength=n_bins
            )
            counts = np.bincount(
                bin_idx, weights=valid.astype(np.float64), minlength=n_bins
            )
            with np.errstate(invalid="ignore"):
                resampled_vars[vname] = (
                    "time",
                    np.where(counts > 0, sums / counts, np.nan),
                )
        elif da.ndim == 2:
            n_h = da.shape[1]
            second_dim = [d for d in ds[vname].dims if d != "time"][0]
            means = np.empty((n_bins, n_h))
            da_filled = np.where(valid, da, 0.0)
            valid_f = valid.astype(np.float64)
            for j in range(n_h):
                s = np.bincount(bin_idx, weights=da_filled[:, j], minlength=n_bins)
                c = np.bincount(bin_idx, weights=valid_f[:, j], minlength=n_bins)
                with np.errstate(invalid="ignore"):
                    means[:, j] = np.where(c > 0, s / c, np.nan)
            resampled_vars[vname] = (("time", second_dim), means)
            extra_coords[second_dim] = ds[second_dim].values
        else:
            print(f"  Skipping {vname}: {da.ndim}-D not supported")

    return xr.Dataset(resampled_vars, coords={"time": new_time, **extra_coords})


def load_station(station_id, station_dir=STATION_DIR):
    """WegenerNet L2 v8 station file (5-min precipitation [mm], temperature, RH) as xarray."""
    path = (
        f"{station_dir}/WN_L2_V8_BD_St{station_id}_"
        "2021-01-01d00h00m_2025-09-23d00h00m_UTC.csv"
    )
    return (
        pd.read_csv(path, parse_dates=True, index_col=[0, 1])
        .rename_axis(["station", "time"])
        .rename(columns=COLUMN_MAP)
        .to_xarray()
    )


def plot_combined_lwp_iwv_summary(pre_warm_by_cloud, fig_dir):
    """Create the combined scatter and trajectory figure used by the notebook."""
    phase_labels = {
        "liquid": "warm rain",
        "ice": "cold rain",
        "mixed": "mixed rain",
    }
    phase_colors = {"liquid": "#1f77b4", "ice": "#d62728", "mixed": "#2ca02c"}
    regimes = ["convective", "non-convective"]
    phases = ("liquid", "ice", "mixed")

    scatter_kwargs = dict(
        warm_cloud_datasets=pre_warm_by_cloud,
        target_minutes=-30,
        xlim=(-300, 600),
        ylim=(-5, 10),
        show_corr=True,
        facet_regime=True,
        corr_scope="overall",
        title="",
        savefig=None,
        return_data=True,
    )
    trajectory_kwargs = dict(
        warm_cloud_datasets=pre_warm_by_cloud,
        nodes=(-180, -150, -120, -90, -60, -30, -15),
        anomaly_reference="long_tail",
        ellipse_method="parametric",
        ellipse_type="confidence",
        ellipse_nodes=(-180, -120, -60, -15),
        ci_level=0.95,
        xlim=(-300, 550),
        ylim=(-3, 5),
        suptitle="",
        savefig=None,
        return_data=True,
    )

    scatter_fig_tmp, _, scatter_data = plot_lwp_iwv_anomaly_scatter(**scatter_kwargs)
    plt.close(scatter_fig_tmp)

    trajectory_fig_tmp, _, trajectory_data = plot_lwp_iwv_trajectory(
        **trajectory_kwargs
    )
    plt.close(trajectory_fig_tmp)

    combined_xlim = (
        min(scatter_kwargs["xlim"][0], trajectory_kwargs["xlim"][0]),
        max(scatter_kwargs["xlim"][1], trajectory_kwargs["xlim"][1]),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    def draw_scatter(ax, regime):
        panel_lwp, panel_iwv = [], []
        for phase in phases:
            d = scatter_data[phase]
            mask = d["regime"] == regime
            if not mask.any():
                continue
            lwp = d["lwp"][mask]
            iwv = d["iwv"][mask]
            ax.scatter(
                lwp,
                iwv,
                s=55,
                c=phase_colors.get(phase, "gray"),
                marker="o",
                alpha=0.8,
                edgecolors="k",
                linewidths=0.4,
                label=f"{phase_labels.get(phase, phase)} (n={int(mask.sum())})",
            )
            panel_lwp.append(lwp)
            panel_iwv.append(iwv)

        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)

        if scatter_kwargs["show_corr"] and panel_lwp:
            lwp_all = np.concatenate(panel_lwp)
            iwv_all = np.concatenate(panel_iwv)
            if lwp_all.size > 2:
                r, p = _corr(lwp_all, iwv_all)
                ax.text(
                    0.03,
                    0.97,
                    f"All: r={r:.2f}{_pstars(p)} (n={lwp_all.size})",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
                )

        ax.set_xlim(combined_xlim)
        ax.set_ylim(scatter_kwargs["ylim"])
        ax.set_title(f"{regime.capitalize()} scatter at 30 min", fontweight="bold")
        ax.set_xlabel("Delta LWP (g$~$m$^{-2}$)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", framealpha=0.85, fontsize=9)

    def draw_trajectory(ax, regime):
        nodes = list(trajectory_kwargs["nodes"])
        ellipse_nodes = set(trajectory_kwargs["ellipse_nodes"])
        node_is_ellipse = [node in ellipse_nodes for node in nodes]

        for phase in phases:
            d = trajectory_data.get((regime, phase))
            if d is None or d.get("insufficient", True):
                continue
            color = phase_colors.get(phase, "gray")
            x = d["mean_lwp"]
            y = d["mean_iwv"]
            label = f"{phase_labels.get(phase, phase)} (n={d['n_kept']})"

            ax.plot(x, y, "-", color=color, lw=1.6, alpha=0.9, zorder=3)
            ax.scatter(
                x,
                y,
                s=45,
                c=color,
                edgecolors="k",
                linewidths=0.5,
                zorder=4,
                label=label,
            )
            ax.scatter(
                x[0],
                y[0],
                s=72,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                marker="s",
                zorder=5,
            )
            if len(x) >= 2:
                ax.annotate(
                    "",
                    xy=(x[-1], y[-1]),
                    xytext=(x[-2], y[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
                    zorder=4,
                )

            for j, ent in enumerate(d["ellipses"]):
                if not node_is_ellipse[j] or ent is None:
                    continue
                cx, cy, width, height, angle = ent
                ax.add_patch(
                    Ellipse(
                        (cx, cy),
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        alpha=0.06,
                        edgecolor=color,
                        lw=1.0,
                        zorder=2,
                    )
                )

        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlim(combined_xlim)
        ax.set_ylim(trajectory_kwargs["ylim"])
        ax.set_title(f"{regime.capitalize()} trajectory", fontweight="bold")
        ax.set_xlabel("Delta LWP (g$~$m$^{-2}$)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", framealpha=0.85, fontsize=9)

    for col, regime in enumerate(regimes):
        draw_scatter(axes[0, col], regime)
        draw_trajectory(axes[1, col], regime)

    axes[0, 0].set_ylabel("Delta IWV (kg$~$m$^{-2}$)")
    axes[1, 0].set_ylabel("Delta IWV (kg$~$m$^{-2}$)")
    fig.suptitle("")

    savefig = f"{fig_dir}/lwp_iwv_scatter_trajectory_10km.png"
    fig.savefig(savefig, dpi=300, bbox_inches="tight")
    print(f"Saved to {savefig}")
    return fig, axes


def _make_long_df(warm, cold, norain, drop_nonpositive=False):
    parts = []
    for arr, name in zip([warm, cold, norain], _REGIMES):
        a = np.asarray(arr)
        a = a[np.isfinite(a)]
        if drop_nonpositive:
            a = a[a > 0]
        parts.append(pd.DataFrame({"value": a, "regime": name}))
    return pd.concat(parts, ignore_index=True)


def _annotate_counts(ax, df):
    counts = df.groupby("regime", observed=True).size().reindex(_REGIMES)
    ax.set_xticks(range(len(_REGIMES)))
    ax.set_xticklabels([f"{r}\n(n={counts[r]:,})" for r in _REGIMES])


def _boxen(df, log_y=False, ax=None):
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.figure

    sns.boxenplot(
        data=df,
        x="regime",
        y="value",
        order=_REGIMES,
        hue="regime",
        palette=_PALETTE,
        legend=False,
        k_depth="trustworthy",
        width_method="linear",
        width=0.65,
        linecolor="black",
        linewidth=0.8,
        line_kws=dict(color="white", linewidth=2.0),
        flier_kws=dict(s=4, alpha=0.4, linewidth=0, facecolor="black"),
        log_scale=10 if log_y else None,
        ax=ax,
    )
    _annotate_counts(ax, df)
    ax.set_xlabel("")
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_lwp_boxen(warm, cold, norain, ax=None):
    df = _make_long_df(warm, cold, norain, drop_nonpositive=True)
    fig, ax = _boxen(df, log_y=True, ax=ax)
    ax.set_ylabel(r"LWP (g$~$m$^{-2}$)")
    ax.set_title("", fontweight="bold", pad=10)
    if ax is not None:
        return fig, ax
    plt.tight_layout()
    return fig, ax


def plot_iwv_boxen(warm, cold, norain, ax=None):
    df = _make_long_df(warm, cold, norain, drop_nonpositive=False)
    fig, ax = _boxen(df, log_y=False, ax=ax)
    ax.set_ylabel(r"IWV (kg$~$m$^{-2}$)")
    ax.set_title("", fontweight="bold", pad=10)
    if ax is not None:
        return fig, ax
    plt.tight_layout()
    return fig, ax
