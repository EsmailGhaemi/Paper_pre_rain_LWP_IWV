"""
Height-time plots of single rain events
=======================================

Radar variables above the radiometer (X-band L1 on the Cartesian grid, L2 rain
intensity) with the -5 degC isotherm from the microwave radiometer and an LWP or
IWV overlay (f10-f13, fA05-fA10).

    configure_event_plotting      event date, site location, output folder
    plot_radar_var_with_overlay   Zh or ZDR with LWP / IWV
    plot_precip_height_time       L2 rain intensity with cloud base and gauge rain
    plot_minus5_isotherm          -5 degC isotherm line and grid-cell band
"""

from __future__ import annotations
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerPathCollection
from figure_style import style_fonts
from event_based_functions import (
    calculate_minus5_isotherm_height,
)

EVENT_DATE = "2023-06-08"
PLOT_TLIM = ("00:00", "23:55")
RADIOMETER_X = 564565.44
RADIOMETER_Y = 5200868.51
SPATIAL_TOL_L1 = 100
FIGURE_DIR = "../results/Figs/"
PLOT_DEFAULTS = {
    "figsize": (8, 5),
    "height_ylim": (0, 6000),
    "event_span_color": "skyblue",
    "event_span_alpha": 0.5,
    "t0_color": "black",
    "t0_lw": 1.3,
    "t0_alpha": 1.0,
    "t0_smoothing_window": 5,
    "t0_band_color": "gray",
    "t0_band_alpha": 0.5,
    "grid_alpha": 0.5,
    "savefig_dpi": 300,
}


def configure_event_plotting(
    *,
    event_date=None,
    plot_tlim=None,
    radiometer_x=None,
    radiometer_y=None,
    spatial_tol_l1=None,
    figure_dir=None,
    plot_defaults=None,
):
    """Configure module-level defaults used by the plotting functions."""
    global EVENT_DATE, PLOT_TLIM, RADIOMETER_X, RADIOMETER_Y, SPATIAL_TOL_L1
    global FIGURE_DIR, PLOT_DEFAULTS

    if event_date is not None:
        EVENT_DATE = event_date
    if plot_tlim is not None:
        PLOT_TLIM = plot_tlim
    if radiometer_x is not None:
        RADIOMETER_X = radiometer_x
    if radiometer_y is not None:
        RADIOMETER_Y = radiometer_y
    if spatial_tol_l1 is not None:
        SPATIAL_TOL_L1 = spatial_tol_l1
    if figure_dir is not None:
        FIGURE_DIR = figure_dir
    if plot_defaults is not None:
        PLOT_DEFAULTS = {**PLOT_DEFAULTS, **plot_defaults}


def _make_modified_cmap(cmap_name="rainbow"):
    """Create a colormap with the first color set to white."""
    original_cmap = colormaps[cmap_name]
    new_colors = original_cmap(np.arange(original_cmap.N))
    new_colors[0, :] = [1, 1, 1, 1]
    return ListedColormap(new_colors)


def _get_plot_xlim(event_date, tlim):
    """Convert a plot time-limit tuple to a np.datetime64 pair."""
    return (
        np.datetime64(f"{event_date}T{tlim[0]}"),
        np.datetime64(f"{event_date}T{tlim[1]}"),
    )


def _extract_l1_at_radiometer(l1_ds, var_name, x=None, y=None, tol=None):
    """Extract an L1 radar variable at the radiometer pixel location."""
    x = x or RADIOMETER_X
    y = y or RADIOMETER_Y
    tol = tol or SPATIAL_TOL_L1
    return l1_ds.where(
        (abs(l1_ds.x_utm - x) <= tol) & (abs(l1_ds.y_utm - y) <= tol),
        drop=True,
    )[var_name].mean(dim=["x_utm", "y_utm"])


def plot_minus5_isotherm(
    ax,
    minus5_isotherm,
    *,
    show_label=False,
    show_resolution_band=True,
    color=None,
    lw=None,
    alpha=None,
    band_color=None,
    band_alpha=None,
):
    """Plot the -5 degC isotherm line and its native-grid resolution band."""
    color = color if color is not None else PLOT_DEFAULTS["t0_color"]
    lw = lw if lw is not None else PLOT_DEFAULTS["t0_lw"]
    alpha = alpha if alpha is not None else PLOT_DEFAULTS["t0_alpha"]
    band_color = band_color if band_color is not None else PLOT_DEFAULTS["t0_band_color"]
    band_alpha = band_alpha if band_alpha is not None else PLOT_DEFAULTS["t0_band_alpha"]

    plot_data = minus5_isotherm.where(
        minus5_isotherm["plot_height_m"].notnull(), drop=True
    )
    time = plot_data["time"].values
    line_height = plot_data["plot_height_m"].values
    qc_pass = plot_data["qc_pass"].values.astype(bool)

    if show_resolution_band:
        ax.fill_between(
            time,
            plot_data["band_lower_m"].values,
            plot_data["band_upper_m"].values,
            color=band_color,
            alpha=band_alpha,
            linewidth=0,
            zorder=4,
            label="-5 degC vertical grid cell" if show_label else "_nolegend_",
        )

    ax.plot(
        time,
        line_height,
        label="-5 degC isotherm (QC flagged)" if show_label else "_nolegend_",
        color=color,
        lw=lw,
        linestyle="--",
        zorder=5,
        alpha=alpha * 0.8,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    ax.plot(
        time,
        np.where(qc_pass, line_height, np.nan),
        label="-5 degC isotherm" if show_label else "_nolegend_",
        color=color,
        lw=lw,
        linestyle="-",
        zorder=6,
        alpha=alpha,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def _add_event_shading(
    ax, events_df, color=None, alpha=None, label=None, hatch=None, zorder=0, ymax=0.1
):
    """Shade event periods on an axis."""
    color = color or PLOT_DEFAULTS["event_span_color"]
    alpha = alpha or PLOT_DEFAULTS["event_span_alpha"]

    for i in range(len(events_df)):
        current_label = label if i == 0 else None

        ax.axvspan(
            events_df.start.iloc[i],
            events_df.end.iloc[i],
            ymin=0,
            ymax=ymax,
            alpha=alpha,
            color=color,
            hatch=hatch,
            zorder=zorder,
            label=current_label,
        )


def _format_height_time_axes(ax, time_ds, height_ylim=None, freq="30min"):
    """Standard formatting for height-time plots."""
    height_ylim = height_ylim or PLOT_DEFAULTS["height_ylim"]
    ax.set_ylim(*height_ylim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1000:g}")
    ax.set_ylabel("Height (km)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=PLOT_DEFAULTS["grid_alpha"])


def _add_legend_if_labeled(ax, *args, **kwargs):
    """Add a legend only when visible labels are present."""
    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, label) for h, label in zip(handles, labels) if not label.startswith("_")]
    if visible:
        handles, labels = zip(*visible)
        ax.legend(handles, labels, *args, **kwargs)


def plot_radar_var_with_overlay(
    l1_ds,
    var_name,
    tprof_ds,
    events_df,
    *,
    cmap_name="rainbow",
    vmin=None,
    vmax=None,
    levels=None,
    cbar_label=None,
    cbar_kwargs_extra=None,
    overlay_var="LWP",
    overlay_ds=None,
    overlay_flag_var=None,
    overlay_scale=1.0,
    overlay_label=None,
    overlay_color="saddlebrown",
    overlay_ylim=None,
    overlay_ylabel=None,
    show_t0=True,
    t0_use_dpg=True,
    t0_offset_K=268.15,
    t0_color=None,
    show_t0_label=False,
    t0_smoothing_window=None,
    show_t0_resolution_band=True,
    height_ylim=None,
    xlim=None,
    tick_freq="30min",
    savefig_name=None,
    title=None,
    figsize=None,
):
    """
    Generic height-time radar variable plot with optional twin-axis overlay.
    """
    figsize = figsize or PLOT_DEFAULTS["figsize"]
    height_ylim = height_ylim or PLOT_DEFAULTS["height_ylim"]
    t0_color = t0_color or PLOT_DEFAULTS["t0_color"]
    xlim = xlim or _get_plot_xlim(EVENT_DATE, PLOT_TLIM)

    fig, ax1 = plt.subplots(figsize=figsize)
    modified_cmap = _make_modified_cmap(cmap_name)

    radar_data = _extract_l1_at_radiometer(l1_ds, var_name)

    cbar_kw = {
        "orientation": "horizontal",
        "aspect": 40,
        "pad": 0.25,                   # room for the rotated time labels and "Time (UTC)"
        "label": cbar_label or var_name,
    }
    if cbar_kwargs_extra:
        cbar_kw.update(cbar_kwargs_extra)

    plot_kwargs = {
        "x": "time",
        "cmap": modified_cmap,
        "alpha": 0.8,
        "cbar_kwargs": cbar_kw,
    }
    if vmin is not None:
        plot_kwargs["vmin"] = vmin
    if vmax is not None:
        plot_kwargs["vmax"] = vmax
    if levels is not None:
        plot_kwargs["levels"] = levels

    radar_data.plot(**plot_kwargs)

    if show_t0:
        minus5_isotherm = calculate_minus5_isotherm_height(
            tprof_ds,
            use_dpg_flag=t0_use_dpg,
            isotherm_temperature_K=t0_offset_K,
            smoothing_window=t0_smoothing_window,
        )
        plot_minus5_isotherm(
            ax1,
            minus5_isotherm,
            show_label=show_t0_label,
            show_resolution_band=show_t0_resolution_band,
            color=t0_color,
        )

    if events_df is not None:
        _add_event_shading(ax1, events_df)

    if overlay_ds is not None and overlay_var is not None:
        ax2 = ax1.twinx()
        if overlay_flag_var:
            filtered = overlay_ds.where(
                (overlay_ds[overlay_flag_var] < 1).compute(), drop=True
            )
        else:
            filtered = overlay_ds
        overlay_data = filtered[overlay_var] * overlay_scale
        ax2.plot(
            filtered.time,
            overlay_data,
            c=overlay_color,
            label=overlay_label or overlay_var,
            markersize=3,
            marker="o",
            linewidth=0.7,
            alpha=0.8,
            ls="--",
        )
        if overlay_ylim:
            ax2.set_ylim(*overlay_ylim)
        if overlay_ylabel:
            ax2.set_ylabel(overlay_ylabel)
        ax2.legend(loc=2)

    _format_height_time_axes(ax1, l1_ds, height_ylim)
    _add_legend_if_labeled(ax1, loc=2)
    ax1.set_xlim(*xlim)
    ax1.set_xlabel("Time (UTC)")
    if title:
        ax1.set_title(title)

    style_fonts(fig, tight=True)     # common font family and sizes (figure_style.py)

    if savefig_name:
        filepath = os.path.join(FIGURE_DIR, savefig_name)
        plt.savefig(filepath, dpi=PLOT_DEFAULTS["savefig_dpi"], bbox_inches="tight")
        print(f"Saved: {filepath}")

    plt.show()


def plot_precip_height_time(
    precip_l2_ds,
    tprof_ds,
    cbh_ds,
    events_df,
    *,
    cmap_name="YlGnBu",
    vmin=0.1,
    vmax=40.1,
    levels=21,
    cbar_label="Precipitation intensity (mm/h)",
    precip_scale=24,
    spatial_tol=100,
    show_cbh=True,
    cbh_resample="10min",
    precip_var_name="precip_amount",
    show_t0_label=False,
    t0_smoothing_window=None,
    show_t0_resolution_band=True,
    overlay_ds=None,
    overlay_var=None,
    overlay_flag_var=None,
    overlay_scale=1.0,
    overlay_label=None,
    overlay_ylim=None,
    overlay_ylabel=None,
    height_ylim=None,
    xlim=None,
    savefig_name=None,
    title=None,
    g_label=None,
    loc_legend=2,
    event_color=None,
    event_alpha=None,
    event_hatch=None,
    event_zorder=0,
    event_band_height_m=None,
):
    """Plot precipitation rate height-time cross-section from L2 data."""
    figsize = PLOT_DEFAULTS["figsize"]
    height_ylim = height_ylim or PLOT_DEFAULTS["height_ylim"]
    xlim = xlim or _get_plot_xlim(EVENT_DATE, PLOT_TLIM)

    fig, ax1 = plt.subplots(figsize=figsize)
    modified_cmap = _make_modified_cmap(cmap_name)

    if show_cbh and cbh_ds is not None:
        measured_cbh = cbh_ds.where((cbh_ds.cbh_qcs_flag).compute() < 1, drop=True)
        measured_cbh.cbh.resample(time=cbh_resample).mean().plot(
            x="time",
            c="gray",
            ax=ax1,
            label="cloud base",
            lw=0.5,
            marker="s",
            markersize=3,
            alpha=0.8,
        )

    precip_pixel = precip_l2_ds.where(
        (abs(precip_l2_ds.X - RADIOMETER_X) <= spatial_tol)
        & (abs(precip_l2_ds.Y - RADIOMETER_Y) <= spatial_tol),
        drop=True,
    )
    (precip_pixel[precip_var_name] * precip_scale).mean(dim=["X", "Y"]).plot(
        x="time",
        cmap=modified_cmap,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        cbar_kwargs={
            "orientation": "horizontal",
            "aspect": 40,
            "pad": 0.25,                   # room for the rotated time labels and "Time (UTC)"
            "label": cbar_label,
        },
        alpha=0.9,
    )

    minus5_isotherm = calculate_minus5_isotherm_height(
        tprof_ds,
        use_dpg_flag=True,
        isotherm_temperature_K=268.15,
        smoothing_window=t0_smoothing_window,
    )
    plot_minus5_isotherm(
        ax1,
        minus5_isotherm,
        show_label=show_t0_label,
        show_resolution_band=show_t0_resolution_band,
    )

    _add_event_shading(
        ax1,
        events_df,
        color=event_color,
        alpha=event_alpha,
        label=g_label,
        hatch=event_hatch,
        zorder=event_zorder,
        ymax=(
            event_band_height_m / (height_ylim[1] - height_ylim[0])
            if event_band_height_m is not None
            else 0.1
        ),
    )

    _format_height_time_axes(ax1, precip_l2_ds, height_ylim)
    _add_legend_if_labeled(ax1, loc=loc_legend)
    ax1.set_xlim(*xlim)
    ax1.set_xlabel("Time (UTC)")
    if title:
        ax1.set_title(title)

    style_fonts(fig, tight=True)     # common font family and sizes (figure_style.py)

    if savefig_name:
        filepath = os.path.join(FIGURE_DIR, savefig_name)
        plt.savefig(filepath, dpi=PLOT_DEFAULTS["savefig_dpi"], bbox_inches="tight")
        print(f"Saved: {filepath}")

    plt.show()


__all__ = [
    "BiggerMarkerHandler",
    "configure_event_plotting",
    "plot_lwp_iwv_single_event_trajectory",
    "plot_lwp_timeseries",
    "plot_minus5_isotherm",
    "plot_precip_height_time",
    "plot_radar_var_with_overlay",
]
