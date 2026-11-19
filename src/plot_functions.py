"""
Pre-onset composites by rain regime and rain type
=================================================

3 x 2 panels (warm / cold / mixed rain x convective / non-convective) of LWP, IWV
and CBH before rain onset, the cold-season panels, and the dLWP-dIWV scatter and
trajectory. Composites give every event the same weight; shading is the 95 %
confidence interval of the mean.

    plot_warm_regime_cloud_grid            one variable (f06, f07, fA03)
    plot_warm_regime_cloud_grid_multivar   radiometer and GNSS IWV (f06, fA03)
    plot_cold_events_panel                 cold-season events (f06, f07, fA03)
    plot_lwp_iwv_anomaly_scatter           dLWP vs dIWV at one lead time (f09)
    plot_lwp_iwv_trajectory                dLWP-dIWV trajectory before onset (f09)
"""

from __future__ import annotations
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats as sp_stats
from figure_style import style_fonts

try:
    from scipy.stats import pearsonr

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
try:
    from scipy.stats import f as _f_dist

    _HAVE_F = True
except Exception:  # pragma: no cover
    _HAVE_F = False


def prepare_pre_event_data(
    ds,
    var,
    var_flag=None,
    scale_lwp=True,
    compute_stats=False,
    ci_level=0.95,
    bin_minutes=None,
):
    """
    Prepare pre-event data with minutes_before_rain as a proper dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 'time' dimension and 'minutes_before_rain' coordinate.
    var : str
        Variable name to process.
    var_flag : str, optional
        Flag variable for filtering (keeps where flag < 1).
    scale_lwp : bool
        Scale tclw by 1000 (to g.m^{-2}).
    compute_stats : bool
        If True, return grouped statistics (median, std, ci, count).
    ci_level : float
        Confidence interval level (default 0.95).
    bin_minutes : int, optional
        If provided, bin minutes_before_rain to this interval.

    Returns
    -------
    xr.Dataset or dict
    """
    da = ds[var] * 1000 if (var == "tclw" and scale_lwp) else ds[var]

    # Reduce spatial dims
    other_dims = [d for d in da.dims if d != "time"]
    if other_dims:
        da = da.mean(dim=other_dims, skipna=True)

    # Apply flag filter
    if var_flag and var_flag in ds:
        flag_da = ds[var_flag]
        if other_dims:
            flag_da = flag_da.any(dim=other_dims)
        da = da.where(flag_da < 1)

    # Handle minutes_before_rain grouping
    mbr_vals = da["minutes_before_rain"].values.astype(float)
    if bin_minutes is not None:
        mbr_binned = (np.round(mbr_vals / bin_minutes) * bin_minutes).astype(int)
    else:
        mbr_binned = np.round(mbr_vals).astype(int)

    da = da.assign_coords(mbr_bin=("time", mbr_binned))

    if not compute_stats:
        da_out = da.swap_dims({"time": "mbr_bin"}).rename(
            {"mbr_bin": "minutes_before_rain"}
        )
        da_out = da_out.drop_vars("time", errors="ignore")
        return da_out.to_dataset(name=var)

    # Grouped statistics
    grp = da.groupby("mbr_bin")
    median_val = grp.quantile(0.5)
    std_val = grp.std(skipna=True)
    count_val = grp.count()

    z = sp_stats.norm.ppf(1 - (1 - ci_level) / 2)
    ci_val = z * std_val / np.sqrt(count_val)

    x = median_val["mbr_bin"].values
    sort_idx = np.argsort(x)

    return {
        "minutes_before_rain": x[sort_idx],
        "median": median_val.values[sort_idx],
        "std": std_val.values[sort_idx],
        "ci": ci_val.values[sort_idx],
        "count": count_val.values[sort_idx],
        "ci_level": ci_level,
        "var": var,
        "binned": bin_minutes is not None,
        "bin_minutes": bin_minutes,
    }


"""
Pre-event composite plots — shared helpers
==========================================

Design decisions (manuscript discussion, May 2026):

- HATPRO cannot reliably retrieve LWP for ice-dominated cold-season events,
  so phase classification (warm rain / cold rain / mixed rain) is not
  physically defensible there.
- Sample sizes for mixed rain are workable (n=25 conv, n=33 non-conv), so
  mixed rain is retained as a separate row in the warm-season figure.

Public plotting functions (defined in the cells below) all share the
helpers in this cell (`_count_events`, `_add_last_steps_hatch`,
`_anomaly_offset`, `_plot_panel`) and `prepare_pre_event_data` from
section 2.5:

  * ``plot_warm_regime_cloud_grid``          — 3x2 grid (phase x regime),
        ONE variable, curves coloured by REGIME (convective / non-conv.).
  * ``plot_warm_regime_cloud_grid_multivar`` — same 3x2 grid, but SEVERAL
        variables overlaid per panel, curves coloured by VARIABLE
        (with a legend). Formerly ``plot_warm_regime_cloud_grid_v2``.
  * ``plot_cold_events_panel``               — cold events combined (no
        phase split), one panel per entry of `variables`; each entry may
        be a single variable or a tuple of variables to overlay.
        (The former ``_v2`` variant was a superset and has been merged in.)
"""


def _count_events(ds):
    """
    Return the number of independent events in a pre-event dataset.

    A dataset typically has many time-steps per event (one per minute of
    the pre-event window). For statistical reporting we want the number
    of *events*, not the number of time-steps.

    Strategy
    --------
    1. If the dataset has an ``event_id`` (or ``event_index``) coordinate
       on the time dimension, count its unique values.
    2. Otherwise fall back to counting transitions in
       ``minutes_before_rain``: each new event begins when MBR resets
       (jumps to a less-negative value than the previous step).
    3. As a last resort, return the size of ``time`` (i.e. the old
       behaviour, which is wrong but at least won't crash).

    The function works on the *time-step* form of the dataset (before
    binning), which is exactly what `plot_warm_regime_cloud_grid` and
    `plot_cold_events_panel` already pass in.
    """
    if ds is None or ds.sizes.get("time", 0) == 0:
        return 0

    # Strategy 1: explicit event coordinate
    for cand in ("event_id", "event_index", "event"):
        if cand in ds.coords:
            return int(np.unique(ds[cand].values).size)

    # Strategy 2: detect event boundaries via minutes_before_rain resets.
    # Within an event, MBR is ascending (e.g. -360, -359, ..., -1).
    # When a new event starts, MBR drops back (e.g. -1 -> -360), i.e. a
    # *negative* jump. Count those resets.
    if "minutes_before_rain" in ds.coords:
        mbr = np.asarray(ds["minutes_before_rain"].values)
        if mbr.size == 0:
            return 0
        diffs = np.diff(mbr)
        # Use a generous threshold so we catch resets but not normal +1 steps.
        n_resets = int((diffs < -1).sum())
        return n_resets + 1  # number of contiguous ascending runs

    # Strategy 3: fallback (this is what the original function did)
    return int(ds.sizes.get("time", 0))


def _default_phase_labels():
    return {
        "liquid": "warm rain",
        "ice": "cold rain",
        "mixed": "mixed rain",
    }


def _warm_regime_datasets(ds, conv_threshold, hail_threshold, require_coords=False):
    """Split one warm-season dataset into convective/non-convective subsets."""
    empty = {"convective": None, "non-convective": None}
    if ds is None or ds.sizes.get("time", 0) == 0:
        return empty

    missing = [coord for coord in ("prob_q95_q95", "prob_hail") if coord not in ds.coords]
    if missing:
        if require_coords:
            raise ValueError(
                "Warm-regime split requires 'prob_q95_q95' and 'prob_hail' coords; "
                f"missing {missing}."
            )
        return empty

    conv_ds = ds.where(
        (ds.prob_q95_q95 >= conv_threshold) & (ds.prob_hail < hail_threshold),
        drop=True,
    )
    nonconv_ds = ds.where(ds.prob_q95_q95 < conv_threshold, drop=True)
    return {
        "convective": conv_ds if conv_ds.sizes.get("time", 0) > 0 else None,
        "non-convective": nonconv_ds if nonconv_ds.sizes.get("time", 0) > 0 else None,
    }


def _add_last_steps_hatch(
    ax, hatching_last_steps, hatch_style="x", edge_color="gray", alpha=0.0
):
    """
    Overlay a hatched span on the last `hatching_last_steps` minutes
    before rain onset, i.e. between x = -hatching_last_steps and x = 0.

    Used to flag periods where a measurement is potentially unreliable
    (e.g. HATPRO LWP near rain onset due to wet-radome bias).
    """
    if hatching_last_steps is None or hatching_last_steps <= 0:
        return
    ax.axvspan(
        -abs(hatching_last_steps),
        0,
        facecolor="none",
        edgecolor=edge_color,
        alpha=1.0,
        hatch=hatch_style,
        linewidth=0.6,
        zorder=0,
    )


def _anomaly_offset(stats, anomaly_reference, anomaly_tail_minutes, raw_da=None):
    """Return scalar offset to subtract for anomaly mode."""
    x = stats["minutes_before_rain"]
    y = stats["median"]

    if anomaly_reference == "long_tail":
        mask = x < anomaly_tail_minutes
        if mask.any() and np.isfinite(y[mask]).any():
            return float(np.nanmean(y[mask]))
        # fallback if the long tail is empty
        return float(y[0]) if np.isfinite(y[0]) else 0.0

    if anomaly_reference == "first":
        return float(y[0]) if np.isfinite(y[0]) else 0.0

    if anomaly_reference == "overall_median":
        if raw_da is not None:
            return float(np.nanmedian(raw_da.values))
        return float(np.nanmedian(y))

    raise ValueError(f"Unknown anomaly_reference: {anomaly_reference!r}")


def _plot_panel(
    ax,
    stats,
    color,
    show_ci,
    show_counts,
    anomaly_offset,
    count_ylim=None,
    n_events=None,
    show_n_annotation=True,
    label=None,
    linestyle="-",
    marker="o",
):
    """Draw one composite curve (median line + CI band + optional sample-count
    bars + optional "n=" annotation) onto `ax`.

    Single shared implementation — used by both the single-variable and the
    multi-variable plotting functions (the two former near-identical copies
    have been merged; `label`, `linestyle` and `marker` simply default to the
    old single-variable behaviour).
    """
    x = stats["minutes_before_rain"]
    y = stats["median"] - anomaly_offset

    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        color=color,
        markersize=2,
        linewidth=1.2,
        label=label,
    )

    if show_ci:
        ci = stats["ci"]
        ax.fill_between(x, y - ci, y + ci, alpha=0.15, color=color, linewidth=0)

    if show_counts:
        ax_counts = ax.twinx()
        width = (
            stats["bin_minutes"] * 0.7
            if stats["binned"]
            else (np.median(np.diff(np.sort(x))) * 0.7 if len(x) > 1 else 5)
        )
        ax_counts.bar(
            x,
            stats["count"],
            width=width,
            alpha=0.2,
            color="gray",
            edgecolor="gray",
            zorder=0,
        )
        ax_counts.set_ylabel("n", color="gray")
        ax_counts.tick_params(axis="y", labelcolor="gray", labelsize=8)
        if count_ylim is not None:
            ax_counts.set_ylim(count_ylim)

    if show_n_annotation and n_events is not None:
        ax.annotate(
            f"n={n_events}",
            fontsize=10,
            xy=(0.2, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )


def plot_warm_regime_cloud_grid(
    warm_cloud_datasets,
    var,
    var_flag=None,
    phases=("liquid", "ice", "mixed"),
    phase_labels=None,
    anomaly=False,
    anomaly_reference="long_tail",
    anomaly_tail_minutes=-180,
    conv_threshold=0.5,
    hail_threshold=0.5,
    show_ci=True,
    show_counts=False,
    figsize=(13, 14),
    xlim=(-365, 0),
    ylim=None,
    xtick_step=30,
    ylabel=None,
    sharey=False,
    suptitle=None,
    savefig=None,
    count_ylim=None,
    hatching_last_steps=None,
    hatch_style="x",
    hatch_edge_color="gray",
    n_annotation="events",
):
    """
    Plot a 3×2 grid of pre-event composites for warm-season events only
    (ONE variable; curves coloured by regime: convective = darkred,
    non-convective = seagreen). For several variables overlaid per panel
    use `plot_warm_regime_cloud_grid_multivar` instead.

        rows    = cloud phase  (warm rain, cold rain, mixed rain by default)
        columns = regime       (convective, non-convective)

    Convective is defined as prob_q95_q95 >= conv_threshold AND
    prob_hail < hail_threshold. Non-convective is prob_q95_q95 < conv_threshold.

    Parameters
    ----------
    warm_cloud_datasets : dict
        {"liquid": xr.Dataset, "ice": xr.Dataset, "mixed": xr.Dataset}.
        Each Dataset must carry 'prob_q95_q95' and 'prob_hail' as coords.
    var, var_flag, anomaly, anomaly_reference, anomaly_tail_minutes,
    conv_threshold, hail_threshold, show_ci, show_counts, figsize, xlim,
    ylim, xtick_step, ylabel, suptitle, savefig, count_ylim :
        Same semantics as the original 3×3 function.
    phases : tuple of str
        Keys to look up in `warm_cloud_datasets`. Default ("liquid","ice","mixed").
    phase_labels : dict or None
        Optional mapping from phase key to display label, e.g.
        {"liquid": "warm rain", "ice": "cold rain", "mixed": "mixed rain"}.
    hatching_last_steps : int or None
        If given, hatch the span between -hatching_last_steps and 0 minutes
        (i.e. the final `hatching_last_steps` minutes before rain onset).
        Used to flag periods where the retrieval is unreliable, e.g. HATPRO
        LWP near rain onset due to wet-radome bias.
    hatch_style : str
        Matplotlib hatch pattern (default "x"). Common choices: "x", "//",
        "\\\\", "..", "++".
    hatch_edge_color : str
        Color of the hatch lines (default "gray").
    n_annotation : {"events", "obs"}
        What "n=" in each panel refers to. "events" (default and recommended)
        counts independent events, matching the convention in pre-event
        compositing studies (Holloway & Neelin 2009; Schiro et al. 2016).
        "obs" counts the raw number of valid time-steps (the original
        behaviour) — useful only for diagnostic plots.

    Returns
    -------
    fig, axes : matplotlib figure and 3×2 array of axes
    """
    regimes = ["convective", "non-convective"]
    regime_colors = {"convective": "darkred", "non-convective": "seagreen"}
    if phase_labels is None:
        phase_labels = _default_phase_labels()

    # ---- Build the 6 datasets (phase × regime) --------------------------
    grid_data = {}
    for phase in phases:
        split = _warm_regime_datasets(
            warm_cloud_datasets.get(phase), conv_threshold, hail_threshold
        )
        for regime in regimes:
            grid_data[(phase, regime)] = split[regime]

    # ---- Default y-label ------------------------------------------------
    if ylabel is None:
        ylabel = {
            "tclw": "LWP (g$~$m$^{-2}$)",
            "tcwv": "IWV (kg$~$m$^{-2}$)",
        }.get(var, var)
        if anomaly:
            ylabel = "Δ " + ylabel

    # ---- Figure (constrained_layout handles twin axes correctly) --------
    fig, axes = plt.subplots(
        len(phases),
        2,
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        constrained_layout=True,
    )

    for row_idx, phase in enumerate(phases):
        for col_idx, regime in enumerate(regimes):
            ax = axes[row_idx, col_idx]
            ds = grid_data[(phase, regime)]

            if ds is None or ds.sizes.get("time", 0) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                    color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            stats = prepare_pre_event_data(
                ds,
                var,
                var_flag=var_flag,
                compute_stats=True,
            )

            offset = (
                _anomaly_offset(
                    stats,
                    anomaly_reference,
                    anomaly_tail_minutes,
                    raw_da=ds[var] if anomaly_reference == "overall_median" else None,
                )
                if anomaly
                else 0.0
            )

            _plot_panel(
                ax,
                stats,
                color=regime_colors[regime],
                show_ci=show_ci,
                show_counts=show_counts,
                anomaly_offset=offset,
                count_ylim=count_ylim,
                n_events=(
                    _count_events(ds)
                    if n_annotation == "events"
                    else int(ds.sizes.get("time", 0))
                ),
            )

            _add_last_steps_hatch(
                ax,
                hatching_last_steps,
                hatch_style=hatch_style,
                edge_color=hatch_edge_color,
            )

            if anomaly:
                ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

            ax.grid(True, alpha=0.25)

            if row_idx == 0:
                ax.set_title(regime.capitalize(), fontsize=14, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{phase_labels.get(phase, phase.capitalize())}\n{ylabel}",
                    fontsize=11,
                )
            if row_idx == len(phases) - 1:
                ax.set_xlabel("Minutes before rain", fontsize=11)

    # ---- Shared axis formatting ----------------------------------------
    for ax in axes.flat:
        ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, xtick_step)))
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", labelrotation=90, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        if ylim is not None:
            ax.set_ylim(ylim)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    style_fonts(fig)                 # common font family and sizes (figure_style.py)
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Saved to {savefig}")

    return fig, axes


def plot_warm_regime_cloud_grid_multivar(
    warm_cloud_datasets,
    variables=("tcwv", "tcwv_gnss"),
    var_flags=None,
    phases=("liquid", "ice", "mixed"),
    phase_labels=None,
    variable_labels=None,
    variable_colors=None,
    variable_linestyles=None,
    variable_markers=None,
    anomaly=False,
    anomaly_reference="long_tail",
    anomaly_tail_minutes=-180,
    conv_threshold=0.5,
    hail_threshold=0.5,
    show_ci=True,
    show_counts=False,
    figsize=(8, 9),
    xlim=(-365, 0),
    ylim=None,
    xtick_step=30,
    ylabel=None,
    suptitle=None,
    savefig=None,
    count_ylim=None,
    hatching_last_steps=None,
    hatch_style="x",
    sharey=False,
    hatch_edge_color="gray",
    n_annotation="events",
    show_legend=True,
):
    """
    Plot a 3×2 warm-season grid with MULTIPLE variables overlaid in each
    subplot (formerly `plot_warm_regime_cloud_grid_v2`).

    Difference from `plot_warm_regime_cloud_grid`: there, curves are
    coloured by REGIME (column); here they are coloured by VARIABLE
    (same colours in both columns) and a legend is drawn.

    Example:
        variables=("tcwv", "tcwv_gnss")

    Rows    = cloud phase: (warm/cold/mixed rain)
    Columns = regime: convective, non-convective
    Curves  = variables
    """

    # Allow backward-compatible single-string input
    if isinstance(variables, str):
        variables = (variables,)

    var_flags = var_flags or {}

    regimes = ["convective", "non-convective"]

    if phase_labels is None:
        phase_labels = _default_phase_labels()

    if variable_labels is None:
        variable_labels = {
            "tclw": "LWP",
            "tcwv": "IWV (microwave radiometer)",
            "tcwv_gnss": "IWV (GNSS station)",
        }

    if variable_colors is None:
        variable_colors = {
            "tcwv": "darkred",
            "tcwv_gnss": "royalblue",
            "tclw": "darkorange",
        }

    if variable_linestyles is None:
        variable_linestyles = {
            "tcwv": "-",
            "tcwv_gnss": "--",
            "tclw": "-",
        }

    if variable_markers is None:
        variable_markers = {
            "tcwv": "o",
            "tcwv_gnss": "s",
            "tclw": "o",
        }

    # ------------------------------------------------------------------
    # Build the 6 datasets: phase × regime
    # ------------------------------------------------------------------
    grid_data = {}

    for phase in phases:
        split = _warm_regime_datasets(
            warm_cloud_datasets.get(phase), conv_threshold, hail_threshold
        )
        for regime in regimes:
            grid_data[(phase, regime)] = split[regime]

    # ------------------------------------------------------------------
    # Default y-label
    # ------------------------------------------------------------------
    if ylabel is None:
        # For tcwv and tcwv_gnss together
        if set(variables) == {"tcwv", "tcwv_gnss"}:
            ylabel = "IWV (kg$~$m$^{-2}$)"
        elif len(variables) == 1:
            ylabel = {
                "tclw": "LWP (g$~$m$^{-2}$)",
                "tcwv": "IWV (kg$~$m$^{-2}$)",
                "tcwv_gnss": "IWV (kg$~$m$^{-2}$)",
            }.get(variables[0], variables[0])
        else:
            ylabel = "Value"

        if anomaly:
            ylabel = "Δ " + ylabel

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        len(phases),
        2,
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        constrained_layout=True,
    )

    for row_idx, phase in enumerate(phases):
        for col_idx, regime in enumerate(regimes):
            ax = axes[row_idx, col_idx]
            ds = grid_data[(phase, regime)]

            if ds is None or ds.sizes.get("time", 0) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            n_events = (
                _count_events(ds)
                if n_annotation == "events"
                else int(ds.sizes.get("time", 0))
            )

            # ----------------------------------------------------------
            # Plot each variable in the same subplot
            # ----------------------------------------------------------
            for var_idx, var in enumerate(variables):

                if var not in ds:
                    continue

                stats = prepare_pre_event_data(
                    ds,
                    var,
                    var_flag=var_flags.get(var),
                    compute_stats=True,
                )

                offset = (
                    _anomaly_offset(
                        stats,
                        anomaly_reference,
                        anomaly_tail_minutes,
                        raw_da=(
                            ds[var] if anomaly_reference == "overall_median" else None
                        ),
                    )
                    if anomaly
                    else 0.0
                )

                _plot_panel(
                    ax,
                    stats,
                    color=variable_colors.get(var, f"C{var_idx}"),
                    show_ci=show_ci,
                    # only draw counts once per panel
                    show_counts=show_counts and var_idx == 0,
                    anomaly_offset=offset,
                    count_ylim=count_ylim,
                    # only annotate n once per panel
                    show_n_annotation=var_idx == 0,
                    n_events=n_events,
                    label=variable_labels.get(var, var),
                    linestyle=variable_linestyles.get(var, "-"),
                    marker=variable_markers.get(var, "o"),
                )

            _add_last_steps_hatch(
                ax,
                hatching_last_steps,
                hatch_style=hatch_style,
                edge_color=hatch_edge_color,
            )

            if anomaly:
                ax.axhline(
                    0,
                    color="k",
                    lw=0.8,
                    ls="--",
                    alpha=0.5,
                )

            ax.grid(True, alpha=0.25)

            if row_idx == 0:
                ax.set_title(
                    regime.capitalize(),
                    fontweight="bold",
                )

            if col_idx == 0:
                ax.set_ylabel(
                    f"{phase_labels.get(phase, phase.capitalize())}\n{ylabel}",
                    fontsize=11,
                )

            if row_idx == len(phases) - 1:
                ax.set_xlabel("Minutes before rain")

            if show_legend:
                ax.legend(loc="best", frameon=True, fontsize=9)

    # ------------------------------------------------------------------
    # Shared axis formatting
    # ------------------------------------------------------------------
    for ax in axes.flat:
        ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, xtick_step)))
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", labelrotation=90, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        if ylim is not None:
            ax.set_ylim(ylim)

    if suptitle:
        fig.suptitle(
            suptitle,
            fontweight="bold",
        )

    style_fonts(fig)                 # common font family and sizes (figure_style.py)
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Saved to {savefig}")

    return fig, axes


def plot_cold_events_panel(
    cold_dataset,
    variables=("tclw", ("tcwv", "tcwv_gnss")),
    var_flags=None,
    anomaly=False,
    anomaly_reference="long_tail",
    anomaly_tail_minutes=-180,
    show_ci=True,
    show_counts=False,
    figsize=None,
    xlim=(-365, 0),
    xtick_step=30,
    ylim=None,
    ylabels=None,
    variable_labels=None,
    variable_colors=None,
    variable_linestyles=None,
    variable_markers=None,
    color="steelblue",
    suptitle="Pre-event composites — cold-season events (combined)",
    savefig=None,
    count_ylim=None,
    hatching_last_steps=None,
    hatch_style="x",
    hatch_edge_color="gray",
    n_annotation="events",
    show_legend=True,
):
    """
    Plot pre-event composites for cold-season events with NO phase split
    (companion figure to `plot_warm_regime_cloud_grid`). Cold events are a
    single combined population because HATPRO cannot reliably distinguish
    liquid / ice / mixed phase in cold conditions.

    This is the merged version of the former `plot_cold_events_panel` and
    `plot_cold_events_panel_v2` (the v2 signature was a strict superset).
    Each element of `variables` defines one subplot; single-curve panels
    with no entry in `variable_colors` fall back to `color`, exactly as in
    the old single-variable function.

    Examples
    --------
    Original behavior, one variable per panel:

        variables=("tclw", "tcwv")

    New behavior, with GNSS IWV only in the right panel:

        variables=("tclw", ("tcwv", "tcwv_gnss"))

    This means:
        left panel  = tclw
        right panel = tcwv + tcwv_gnss

    Parameters
    ----------
    cold_dataset : xr.Dataset
        Combined cold-season pre-event dataset.

    variables : sequence
        Each item is either a string or a tuple/list of strings.
        A string gives one curve in that panel.
        A tuple/list gives multiple curves in that panel.

    var_flags : dict or None
        Optional mapping from variable name to flag name, e.g.
        {
            "tclw": "tclw_dpg_flag",
            "tcwv": "tcwv_dpg_flag",
            "tcwv_gnss": "tcwv_gnss_flag",
        }

    Returns
    -------
    fig, axes : matplotlib figure and 1D array of axes
    """

    var_flags = var_flags or {}

    if not isinstance(var_flags, dict):
        raise TypeError(
            f"`var_flags` must be a dict mapping variable names to flag "
            f"names, e.g. {{'tclw': 'tclw_dpg_flag'}}. Got "
            f"{type(var_flags).__name__}: {var_flags!r}. (A common mistake is "
            f"writing {{'tclw_dpg_flag'}} — that's a set, not a dict.)"
        )

    if ylabels is None:
        ylabels = {
            "tclw": "LWP (g$~$m$^{-2}$)",
            "tcwv": "IWV (kg$~$m$^{-2}$)",
            "tcwv_gnss": "IWV (kg$~$m$^{-2}$)",
            ("tcwv", "tcwv_gnss"): "IWV (kg$~$m$^{-2}$)",
        }

    if variable_labels is None:
        variable_labels = {
            "tclw": "LWP",
            "tcwv": "HATPRO IWV",
            "tcwv_gnss": "GNSS IWV",
        }

    if variable_colors is None:
        variable_colors = {
            "tclw": color,
            "tcwv": "darkred",
            "tcwv_gnss": "royalblue",
        }

    if variable_linestyles is None:
        variable_linestyles = {
            "tclw": "-",
            "tcwv": "-",
            "tcwv_gnss": "--",
        }

    if variable_markers is None:
        variable_markers = {
            "tclw": "o",
            "tcwv": "o",
            "tcwv_gnss": "s",
        }

    n_panels = len(variables)

    if figsize is None:
        figsize = (6.5 * n_panels, 4.5)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    n_events = (
        _count_events(cold_dataset)
        if n_annotation == "events"
        else int(cold_dataset.sizes.get("time", 0))
    )

    for i, panel_vars in enumerate(variables):
        ax = axes[i]

        if n_events == 0:
            ax.text(
                0.5,
                0.5,
                "no cold events",
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
                color="gray",
            )
            continue

        # --------------------------------------------------------------
        # Allow either:
        #   panel_vars = "tclw"
        # or:
        #   panel_vars = ("tcwv", "tcwv_gnss")
        # --------------------------------------------------------------
        if isinstance(panel_vars, str):
            panel_var_list = (panel_vars,)
        else:
            panel_var_list = tuple(panel_vars)

        for var_idx, var in enumerate(panel_var_list):

            if var not in cold_dataset:
                ax.text(
                    0.5,
                    0.5,
                    f"{var} not found",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                    color="gray",
                )
                continue

            stats = prepare_pre_event_data(
                cold_dataset,
                var,
                var_flag=var_flags.get(var),
                compute_stats=True,
            )

            offset = (
                _anomaly_offset(
                    stats,
                    anomaly_reference,
                    anomaly_tail_minutes,
                    raw_da=(
                        cold_dataset[var]
                        if anomaly_reference == "overall_median"
                        else None
                    ),
                )
                if anomaly
                else 0.0
            )

            fallback_color = color if len(panel_var_list) == 1 else f"C{var_idx}"
            _plot_panel(
                ax,
                stats,
                color=variable_colors.get(var, fallback_color),
                show_ci=show_ci,
                # show counts only once per subplot
                show_counts=show_counts and var_idx == 0,
                anomaly_offset=offset,
                count_ylim=count_ylim,
                # show n only once per subplot
                show_n_annotation=var_idx == 0,
                n_events=n_events,
                label=variable_labels.get(var, var),
                linestyle=variable_linestyles.get(var, "-"),
                marker=variable_markers.get(var, "o"),
            )

        if anomaly:
            ax.axhline(
                0,
                color="k",
                lw=0.8,
                ls="--",
                alpha=0.5,
            )

        _add_last_steps_hatch(
            ax,
            hatching_last_steps,
            hatch_style=hatch_style,
            edge_color=hatch_edge_color,
        )

        ax.grid(True, alpha=0.25)

        ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, xtick_step)))
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", labelrotation=90, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        # --------------------------------------------------------------
        # Per-variable ylim:
        #   ylim can be:
        #       None
        #       tuple, e.g. (-2, 2)
        #       dict, e.g. {"tclw": (-20, 20), "tcwv": (-2, 2)}
        # --------------------------------------------------------------
        if ylim is not None:
            if isinstance(ylim, dict):
                first_var = panel_var_list[0]
                if first_var in ylim and ylim[first_var] is not None:
                    ax.set_ylim(ylim[first_var])
            else:
                ax.set_ylim(ylim)

        # --------------------------------------------------------------
        # Y-label
        # --------------------------------------------------------------
        if len(panel_var_list) == 1:
            ylab_key = panel_var_list[0]
        else:
            ylab_key = tuple(panel_var_list)

        ylab = ylabels.get(ylab_key)

        if ylab is None:
            # fallback to first variable
            ylab = ylabels.get(panel_var_list[0], panel_var_list[0])

        if anomaly:
            ylab = "Δ " + ylab

        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xlabel("Minutes before rain", fontsize=11)

        # Keep empty title as in your current function
        ax.set_title("", fontweight="bold")

        if show_legend and len(panel_var_list) > 1:
            ax.legend(
                loc="best",
                fontsize=9,
                frameon=True,
            )

    if suptitle:
        fig.suptitle(
            suptitle,
            fontweight="bold",
        )

    style_fonts(fig)                 # common font family and sizes (figure_style.py)
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Saved to {savefig}")

    return fig, axes


"""Shared helpers for the Δ–Δ scatter and trajectory plots below.

These were previously defined (near-)identically in both cells;
they are kept once here.
"""


def _split_event_indices(ds):
    """
    Return a list of integer index arrays, one per event, indexing the
    'time' dimension.

    Strategy: use an explicit event coordinate if present
    (event_id / event_index / event), else detect event boundaries via
    negative jumps in minutes_before_rain (within an event MBR ascends;
    a new event resets it to a more-negative value).
    """
    n = ds.sizes.get("time", 0)
    if n == 0:
        return []

    for cand in ("event_id", "event_index", "event"):
        if cand in ds.coords:
            ev = np.asarray(ds[cand].values)
            return [np.where(ev == u)[0] for u in np.unique(ev)]

    if "minutes_before_rain" in ds.coords:
        mbr = np.asarray(ds["minutes_before_rain"].values)
        diffs = np.diff(mbr)
        reset_idx = np.where(diffs < -1)[0] + 1
        starts = np.r_[0, reset_idx]
        ends = np.r_[reset_idx, n]
        return [np.arange(s, e) for s, e in zip(starts, ends)]

    # Degenerate: whole dataset is one event
    return [np.arange(n)]


def _series(ds, var, var_flag=None, scale_lwp=True):
    """Return (values, mbr) as 1-D numpy arrays along 'time'."""
    da = ds[var] * 1000 if (var == "tclw" and scale_lwp) else ds[var]
    other = [d for d in da.dims if d != "time"]
    if other:
        da = da.mean(dim=other, skipna=True)
    if var_flag and var_flag in ds:
        flag = ds[var_flag]
        if other:
            flag = flag.any(dim=other)
        da = da.where(flag < 1)
    return np.asarray(da.values), np.round(
        np.asarray(da["minutes_before_rain"].values)
    ).astype(int)


def _event_baseline(ev_vals, ev_mbr, finite, anomaly_reference, anomaly_tail_minutes):
    """Return the per-event baseline; 'none' means values are already anomalies."""
    if anomaly_reference == "none":
        return 0.0
    if anomaly_reference == "long_tail":
        tail = (ev_mbr < anomaly_tail_minutes) & finite
        return np.nanmean(ev_vals[tail]) if tail.any() else np.nan

    if anomaly_reference == "first":
        order = np.argsort(ev_mbr)
        first_finite = ev_vals[order][np.isfinite(ev_vals[order])]
        return first_finite[0] if first_finite.size else np.nan

    if anomaly_reference == "overall_median":
        return np.nanmedian(ev_vals[finite])

    raise ValueError(f"Unknown anomaly_reference: {anomaly_reference!r}")


"""
plot_lwp_iwv_anomaly_scatter
============================

Scatter plot of per-event LWP anomaly vs IWV anomaly at a single chosen
time-step before rain onset (e.g. 30 minutes before rain).

Each marker is ONE event. For event *i*:

    anomaly_LWP_i = LWP_i(target) - baseline_LWP_i
    anomaly_IWV_i = IWV_i(target) - baseline_IWV_i

where ``target`` is the time-step nearest to ``target_minutes`` (within
``tolerance`` minutes) and the baseline is that event's own mean over its
long tail (mbr < anomaly_tail_minutes). This is the per-event analogue of
the "long_tail" reference used by the composite figures.

Takes the same ``warm_cloud_datasets`` dict as
``plot_warm_regime_cloud_grid``:

    {"liquid": xr.Dataset, "ice": xr.Dataset, "mixed": xr.Dataset}

Each Dataset must carry ``minutes_before_rain`` plus (for regime
splitting) ``prob_q95_q95`` and ``prob_hail`` as coords.
"""


# =====================================================================
# Helper: per-event value at target + baseline -> anomaly
# =====================================================================
def _extract_from_dataset(
    ds,
    lwp_var,
    iwv_var,
    lwp_flag,
    iwv_flag,
    anomaly,
    anomaly_reference,
    anomaly_tail_minutes,
    target_minutes,
    tolerance,
):
    """
    Extract per-event (LWP anomaly, IWV anomaly) from one dataset.

    Returns (lwp_arr, iwv_arr, n_total_events). An event contributes a
    point only if BOTH variables are valid at the target step.
    """
    if ds is None or ds.sizes.get("time", 0) == 0:
        return np.array([]), np.array([]), 0

    lwp_vals, lwp_mbr = _series(ds, lwp_var, lwp_flag)
    iwv_vals, iwv_mbr = _series(ds, iwv_var, iwv_flag)
    idx_list = _split_event_indices(ds)

    lwp_out, iwv_out = [], []
    for idx in idx_list:
        a_lwp, _ = _event_anomaly(
            lwp_vals,
            lwp_mbr,
            idx,
            target_minutes,
            tolerance,
            anomaly,
            anomaly_reference,
            anomaly_tail_minutes,
        )
        a_iwv, _ = _event_anomaly(
            iwv_vals,
            iwv_mbr,
            idx,
            target_minutes,
            tolerance,
            anomaly,
            anomaly_reference,
            anomaly_tail_minutes,
        )
        if np.isfinite(a_lwp) and np.isfinite(a_iwv):
            lwp_out.append(a_lwp)
            iwv_out.append(a_iwv)

    return np.array(lwp_out), np.array(iwv_out), len(idx_list)


def _event_anomaly(
    values,
    mbr,
    idx,
    target_minutes,
    tolerance,
    anomaly,
    anomaly_reference,
    anomaly_tail_minutes,
):
    """
    For one event (indices `idx`), return (anomaly_value, value_at_target).

    Returns (np.nan, np.nan) if no valid sample within `tolerance` of the
    target, or if the baseline cannot be computed.
    """
    ev_vals = values[idx]
    ev_mbr = mbr[idx]

    # --- value at the target time-step ---
    finite = np.isfinite(ev_vals)
    if not finite.any():
        return np.nan, np.nan
    dist = np.abs(ev_mbr - target_minutes)
    dist = np.where(finite, dist, np.inf)
    j = int(np.argmin(dist))
    if not np.isfinite(dist[j]) or dist[j] > tolerance:
        return np.nan, np.nan
    val_target = ev_vals[j]

    if not anomaly:
        return val_target, val_target

    baseline = _event_baseline(
        ev_vals, ev_mbr, finite, anomaly_reference, anomaly_tail_minutes
    )
    if not np.isfinite(baseline):
        return np.nan, val_target
    return val_target - baseline, val_target


# =====================================================================
# Main
# =====================================================================
def plot_lwp_iwv_anomaly_scatter(
    warm_cloud_datasets,
    target_minutes=-30,
    tolerance=2,
    lwp_var="tclw",
    iwv_var="tcwv",
    lwp_flag="tclw_dpg_flag",
    iwv_flag="tcwv_dpg_flag",
    anomaly=True,
    anomaly_reference="long_tail",
    anomaly_tail_minutes=-180,
    conv_threshold=0.5,
    hail_threshold=0.5,
    split_regime=False,
    facet_regime=False,
    phases=("liquid", "ice", "mixed"),
    phase_labels=None,
    phase_colors=None,
    regime_markers=None,
    figsize=(8, 8),
    point_size=55,
    alpha=0.8,
    swap_axes=False,
    show_zero_lines=True,
    show_one_to_one=False,
    show_corr=True,
    corr_scope="overall",
    xlim=None,
    ylim=None,
    title=None,
    savefig=None,
    return_data=False,
):
    """
    Scatter per-event LWP anomaly vs IWV anomaly at a chosen pre-rain step.

    Parameters
    ----------
    warm_cloud_datasets : dict
        {"liquid": ds, "ice": ds, "mixed": ds}. Each ds needs
        'minutes_before_rain', and (for regime markers) 'prob_q95_q95'
        and 'prob_hail' as coords on the time dim.
    target_minutes : int
        Time-step before rain to sample, e.g. -30 for 30 min before rain.
        Negative = before rain. Default -30.
    tolerance : int
        Max |mbr - target_minutes| (minutes) accepted when matching the
        nearest sample. Events with no sample inside this window are
        dropped. Default 2.
    lwp_var, iwv_var : str
        Variable names. Defaults 'tclw', 'tcwv'.
    lwp_flag, iwv_flag : str or None
        QC flag variable names; flagged samples become NaN and an event
        is dropped if either variable is invalid at the target.
    anomaly : bool
        If True (default), plot per-event anomalies relative to baseline.
        If False, plot raw values at the target step.
    anomaly_reference : {'long_tail', 'first', 'overall_median'}
        Per-event baseline definition. 'long_tail' = mean over the event's
        samples with mbr < anomaly_tail_minutes (default).
    anomaly_tail_minutes : int
        Cutoff for the long-tail baseline. Default -180.
    conv_threshold, hail_threshold : float
        Convective = prob_q95_q95 >= conv_threshold AND
        prob_hail < hail_threshold. Only used if split_regime=True.
    split_regime : bool
        If True, marker shape encodes convective vs non-convective.
    phases : tuple of str
        Keys into warm_cloud_datasets. Default ("liquid","ice","mixed").
    phase_labels : dict or None
        Display labels, default {"liquid":"warm rain","ice":"cold rain","mixed":"mixed rain"}.
    phase_colors : dict or None
        Per-phase colors. Sensible defaults provided.
    regime_markers : dict or None
        {"convective": "o", "non-convective": "^"} by default.
    swap_axes : bool
        If False (default), LWP anomaly on x, IWV anomaly on y.
        If True, the axes are swapped.
    show_zero_lines : bool
        Draw axhline/axvline at 0 (only meaningful for anomalies).
    show_one_to_one : bool
        Draw a 1:1 reference line (rarely meaningful since units differ;
        off by default).
    show_corr : bool
        Annotate Pearson correlation.
    corr_scope : {'overall', 'per_phase', 'both'}
        Whether the annotated correlation is computed across all points,
        within each phase, or both.
    xlim, ylim : tuple or None
        Axis limits.
    title : str or None
        Plot title; auto-generated if None.
    savefig : str or None
        Path to save (300 dpi).
    return_data : bool
        If True, also return a dict of the extracted per-event arrays.

    Returns
    -------
    fig, ax            (default)
    fig, ax, data      (if return_data=True)
    """
    if phase_labels is None:
        phase_labels = _default_phase_labels()
    if phase_colors is None:
        phase_colors = {"liquid": "#1f77b4", "ice": "#d62728", "mixed": "#2ca02c"}
    if regime_markers is None:
        regime_markers = {"convective": "o", "non-convective": "^"}

    # ---- Extract per-event anomalies, per phase ------------------------
    data = {}  # phase -> dict of arrays
    extract_kw = dict(
        lwp_var=lwp_var,
        iwv_var=iwv_var,
        lwp_flag=lwp_flag,
        iwv_flag=iwv_flag,
        anomaly=anomaly,
        anomaly_reference=anomaly_reference,
        anomaly_tail_minutes=anomaly_tail_minutes,
        target_minutes=target_minutes,
        tolerance=tolerance,
    )

    for phase in phases:
        ds = warm_cloud_datasets.get(phase)
        if ds is None or ds.sizes.get("time", 0) == 0:
            data[phase] = dict(
                lwp=np.array([]),
                iwv=np.array([]),
                regime=np.array([]),
                n_total=0,
                n_used=0,
            )
            continue

        if split_regime or facet_regime:
            try:
                split = _warm_regime_datasets(
                    ds, conv_threshold, hail_threshold, require_coords=True
                )
            except ValueError as exc:
                raise ValueError(
                    f"split_regime/facet_regime require 'prob_q95_q95' and "
                    f"'prob_hail' coords, missing in phase '{phase}'."
                ) from exc
            # Identical masking to plot_warm_regime_cloud_grid:
            #   convective    = prob_q95_q95 >= conv_threshold
            #                     AND prob_hail < hail_threshold
            #   non-convective = prob_q95_q95 < conv_threshold
            # Hail-dominant convective events (pq>=thr AND ph>=hail_thr)
            # fall into NEITHER group, exactly as in the grid functions.
            lwp_c, iwv_c, n_c = _extract_from_dataset(
                split["convective"], **extract_kw
            )
            lwp_n, iwv_n, n_n = _extract_from_dataset(
                split["non-convective"], **extract_kw
            )

            lwp_arr = np.concatenate([lwp_c, lwp_n])
            iwv_arr = np.concatenate([iwv_c, iwv_n])
            reg_arr = np.array(
                ["convective"] * lwp_c.size + ["non-convective"] * lwp_n.size
            )
            n_total = n_c + n_n
        else:
            lwp_arr, iwv_arr, n_total = _extract_from_dataset(ds, **extract_kw)
            reg_arr = np.array(["all"] * lwp_arr.size)

        data[phase] = dict(
            lwp=lwp_arr,
            iwv=iwv_arr,
            regime=reg_arr,
            n_total=n_total,
            n_used=lwp_arr.size,
        )

    # ---- Plot ----------------------------------------------------------
    def _xy(lwp, iwv):
        return (iwv, lwp) if swap_axes else (lwp, iwv)

    lwp_label = ("ΔLWP" if anomaly else "LWP") + " (g$~$m$^{-2}$)"
    iwv_label = ("ΔIWV" if anomaly else "IWV") + " (kg$~$m$^{-2}$)"

    def _draw_panel(ax, regime_filter, panel_title):
        """Draw one panel. regime_filter=None -> all events on this panel."""
        # collect for per-panel correlation
        panel_lwp, panel_iwv = [], []
        for phase in phases:
            d = data[phase]
            if d["lwp"].size == 0:
                continue
            color = phase_colors.get(phase, "gray")
            plabel = phase_labels.get(phase, phase)

            if regime_filter is None and split_regime:
                # single panel, marker-encoded regime (original behaviour)
                for regime, marker in regime_markers.items():
                    m = d["regime"] == regime
                    if not m.any():
                        continue
                    x, y = _xy(d["lwp"][m], d["iwv"][m])
                    ax.scatter(
                        x,
                        y,
                        s=point_size,
                        c=color,
                        marker=marker,
                        alpha=alpha,
                        edgecolors="k",
                        linewidths=0.4,
                        label=f"{plabel} – {regime} (n={int(m.sum())})",
                    )
                    panel_lwp.append(d["lwp"][m])
                    panel_iwv.append(d["iwv"][m])
            else:
                # filter to one regime (faceted) or take all (regime_filter None)
                if regime_filter is None:
                    m = np.ones(d["lwp"].size, dtype=bool)
                else:
                    m = d["regime"] == regime_filter
                if not m.any():
                    continue
                x, y = _xy(d["lwp"][m], d["iwv"][m])
                ax.scatter(
                    x,
                    y,
                    s=point_size,
                    c=color,
                    marker="o",
                    alpha=alpha,
                    edgecolors="k",
                    linewidths=0.4,
                    label=f"{plabel} (n={int(m.sum())})",
                )
                panel_lwp.append(d["lwp"][m])
                panel_iwv.append(d["iwv"][m])

        # reference lines
        if show_zero_lines and anomaly:
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        if show_one_to_one:
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8, ls=":", alpha=0.6)

        # correlation annotation
        if show_corr:
            pl = np.concatenate(panel_lwp) if panel_lwp else np.array([])
            pw = np.concatenate(panel_iwv) if panel_iwv else np.array([])
            lines = []
            if corr_scope in ("overall", "both") and pl.size > 2:
                r, p = _corr(pl, pw)
                lines.append(f"All: r={r:.2f}{_pstars(p)} (n={pl.size})")
            if corr_scope in ("per_phase", "both"):
                for phase in phases:
                    d = data[phase]
                    if regime_filter is None:
                        mm = np.ones(d["lwp"].size, dtype=bool)
                    else:
                        mm = d["regime"] == regime_filter
                    if mm.sum() > 2:
                        r, p = _corr(d["lwp"][mm], d["iwv"][mm])
                        lines.append(
                            f"{phase_labels.get(phase, phase)}: "
                            f"r={r:.2f}{_pstars(p)} (n={int(mm.sum())})"
                        )
            if lines:
                ax.text(
                    0.03,
                    0.97,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
                )

        # labels / limits
        if swap_axes:
            ax.set_xlabel(iwv_label)
            ax.set_ylabel(lwp_label)
        else:
            ax.set_xlabel(lwp_label)
            ax.set_ylabel(iwv_label)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", framealpha=0.8)
        ax.set_title(panel_title, fontweight="bold")

    when = f"{abs(target_minutes)} min before rain"
    base_title = (
        title
        if title is not None
        else f"Per-event {'anomaly' if anomaly else 'value'} at {when}"
    )

    if facet_regime:
        regimes = ["convective", "non-convective"]
        ff = figsize if figsize != (8, 8) else (15, 7)
        fig, axes = plt.subplots(
            1, 2, figsize=ff, sharex=True, sharey=True, constrained_layout=True
        )
        for ax, regime in zip(axes, regimes):
            _draw_panel(ax, regime_filter=regime, panel_title=regime.capitalize())
        fig.suptitle(base_title, fontweight="bold")
        ax_out = axes
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _draw_panel(ax, regime_filter=None, panel_title=base_title)
        ax_out = ax

    style_fonts(fig)                 # common font family and sizes (figure_style.py)
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Saved to {savefig}")

    # ---- Report --------------------------------------------------------
    for phase in phases:
        d = data[phase]
        print(
            f"{phase_labels.get(phase, phase):>4}: "
            f"{d['n_used']}/{d['n_total']} events had valid LWP & IWV "
            f"at {target_minutes}±{tolerance} min"
        )

    if return_data:
        return fig, ax_out, data
    return fig, ax_out


# =====================================================================
# Small stats helpers
# =====================================================================
def _corr(x, y):
    """Pearson r and p-value (falls back to numpy if scipy absent)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan, np.nan
    if _HAVE_SCIPY:
        r, p = pearsonr(x, y)
        return r, p
    return float(np.corrcoef(x, y)[0, 1]), np.nan


def _pstars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


"""
plot_lwp_iwv_trajectory
=======================

Phase-space trajectory of per-event mean (ΔLWP, ΔIWV) anomaly as a
function of time before rain, for warm-season events, split by regime
(convective | non-convective) and coloured by cloud phase (warm rain/cold rain/mixed rain).

Each trajectory is a path through the ΔLWP–ΔIWV plane: one node per lead
time (e.g. -180, -150, ... -15 min). Even time spacing means arc length
between nodes reads as rate of change. Bootstrap confidence ellipses at
selected nodes show where the *mean* trajectory sits (does it separate
between groups?), with the ellipse tilt encoding ΔLWP–ΔIWV coupling.

Design choices (see module-level discussion):
  * Anomaly space only (per-event long-tail baseline, as in the composites).
  * BALANCED event subset: only events valid at *every* node contribute,
    so a kink reflects evolution, not changing event membership.
  * Trajectory terminates at `nodes[-1]` (default -15 min), before the
    wet-radome LWP bias zone near onset.
  * Ellipses are bootstrap CONFIDENCE ellipses for the mean (shrink with
    n), drawn from the covariance of the bootstrap mean-cloud and scaled
    with the chi-square_2 quantile (NOT the "1-sigma = 68%" fallacy).

Regime masking is IDENTICAL to plot_warm_regime_cloud_grid:
    convective     = prob_q95_q95 >= conv_threshold AND prob_hail < hail_threshold
    non-convective = prob_q95_q95 < conv_threshold
(hail-dominant convective events fall into neither group).
"""
_CHI2_2 = {0.68: 2.279, 0.90: 4.605, 0.95: 5.991, 0.99: 9.210}


def _chi2_2_quantile(p):
    return -2.0 * np.log(1.0 - p)


def _cov_to_ellipse(cov, scale_c2):
    """
    Given a 2x2 covariance and a scalar c2 such that the ellipse is
    {Δ : Δ' cov^{-1} Δ <= c2}, return (width, height, angle_deg) where
    width/height are FULL axis lengths.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(vals, 0.0) * scale_c2)
    return width, height, angle


# =====================================================================
# Per-event: anomaly value at each node
# =====================================================================
def _event_node_anomalies(
    values, mbr, idx, nodes, half_width, anomaly_reference, anomaly_tail_minutes
):
    """
    For one event, return a 1-D array of anomalies (len == len(nodes)),
    with NaN where that node has no valid sample in its ±half_width bin.
    """
    ev_vals = values[idx]
    ev_mbr = mbr[idx]
    finite = np.isfinite(ev_vals)
    if not finite.any():
        return np.full(len(nodes), np.nan)

    baseline = _event_baseline(
        ev_vals, ev_mbr, finite, anomaly_reference, anomaly_tail_minutes
    )
    if not np.isfinite(baseline):
        return np.full(len(nodes), np.nan)

    out = np.full(len(nodes), np.nan)
    for k, node in enumerate(nodes):
        sel = (np.abs(ev_mbr - node) <= half_width) & finite
        if sel.any():
            out[k] = np.nanmean(ev_vals[sel]) - baseline
    return out


# =====================================================================
# Build the balanced per-event node matrices for one dataset
# =====================================================================
def _node_matrices(
    ds,
    nodes,
    half_width,
    lwp_var,
    iwv_var,
    lwp_flag,
    iwv_flag,
    anomaly_reference,
    anomaly_tail_minutes,
    balanced=True,
):
    """
    Return (LWP_mat, IWV_mat) each of shape (n_events_kept, n_nodes).

    If balanced=True, only events with finite ΔLWP and ΔIWV at *every*
    node are kept (rows with any NaN dropped).
    """
    if ds is None or ds.sizes.get("time", 0) == 0:
        return np.empty((0, len(nodes))), np.empty((0, len(nodes))), 0

    lwp_vals, lwp_mbr = _series(ds, lwp_var, lwp_flag)
    iwv_vals, iwv_mbr = _series(ds, iwv_var, iwv_flag)
    idx_list = _split_event_indices(ds)
    n_total = len(idx_list)

    lwp_rows, iwv_rows = [], []
    for idx in idx_list:
        l = _event_node_anomalies(
            lwp_vals,
            lwp_mbr,
            idx,
            nodes,
            half_width,
            anomaly_reference,
            anomaly_tail_minutes,
        )
        w = _event_node_anomalies(
            iwv_vals,
            iwv_mbr,
            idx,
            nodes,
            half_width,
            anomaly_reference,
            anomaly_tail_minutes,
        )
        lwp_rows.append(l)
        iwv_rows.append(w)

    L = np.array(lwp_rows) if lwp_rows else np.empty((0, len(nodes)))
    W = np.array(iwv_rows) if iwv_rows else np.empty((0, len(nodes)))

    if balanced and L.size:
        keep = np.all(np.isfinite(L), axis=1) & np.all(np.isfinite(W), axis=1)
        L, W = L[keep], W[keep]

    return L, W, n_total


# =====================================================================
# Bootstrap mean or median trajectory + per-node confidence ellipse params
# =====================================================================
def _bootstrap_trajectory(L, W, n_boot, ci_level, rng, statistic="mean"):
    """
    Given balanced matrices L, W of shape (n_events, n_nodes), return:
      summary_lwp, summary_iwv : (n_nodes,) point estimates
      ell : list of (cx, cy, width, height, angle_deg) per node for the
            bootstrap confidence ellipse of the chosen statistic.
    """
    n_events, n_nodes = L.shape
    summarize = np.nanmean if statistic == "mean" else np.nanmedian
    summary_lwp = summarize(L, axis=0)
    summary_iwv = summarize(W, axis=0)

    # Bootstrap cloud of estimates: (n_boot, n_nodes, 2).
    boot = np.empty((n_boot, n_nodes, 2))
    idx = np.arange(n_events)
    for b in range(n_boot):
        s = rng.choice(idx, size=n_events, replace=True)
        boot[b, :, 0] = summarize(L[s], axis=0)
        boot[b, :, 1] = summarize(W[s], axis=0)

    k = _chi2_2_quantile(ci_level)
    ell = []
    for j in range(n_nodes):
        pts = boot[:, j, :]  # (n_boot, 2)
        cov = np.cov(pts.T)  # covariance of the bootstrap estimate cloud
        width, height, angle = _cov_to_ellipse(cov, k)
        ell.append((summary_lwp[j], summary_iwv[j], width, height, angle))

    return summary_lwp, summary_iwv, ell


# =====================================================================
# Parametric trajectory + per-node ellipse (NO bootstrap)
# =====================================================================
def _parametric_trajectory(L, W, ci_level, ellipse_type="confidence", min_events=4):
    """
    Analytic alternative to _bootstrap_trajectory using only the original
    data points (no resampling).

    ellipse_type:
      'confidence' : confidence ellipse for the MEAN trajectory position,
                     using the small-sample F (Hotelling T^2) scaling:
                         c2 = [2(n-1) / (n(n-2))] * F_{2, n-2}(ci_level)
                     -> shrinks with n; converges to the bootstrap /chi2_2
                     standard-error ellipse as n grows.
      'dispersion' : ellipse describing the SPREAD of individual events
                     (1-(1-ci) coverage assuming bivariate normal), using
                         c2 = chi2_2(ci_level)
                     -> does NOT shrink with n.

    Nodes with fewer than `min_events` finite paired samples get a None
    ellipse entry (the mean point is still plotted, just no ellipse).
    """
    n_nodes = L.shape[1]
    mean_lwp = np.nanmean(L, axis=0)
    mean_iwv = np.nanmean(W, axis=0)

    ell = []
    for j in range(n_nodes):
        x, y = L[:, j], W[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        n = x.size
        if n < min_events:
            ell.append(None)
            continue

        cov = np.cov(np.vstack([x, y]))  # sample covariance S (ddof=1)

        if ellipse_type == "dispersion":
            c2 = _chi2_2_quantile(ci_level)
        elif ellipse_type == "confidence":
            if _HAVE_F and n > 2:
                F = _f_dist.ppf(ci_level, 2, n - 2)
                c2 = (2.0 * (n - 1) / (n * (n - 2))) * F
            else:
                # large-n fallback: chi2_2 / n standard-error ellipse
                c2 = _chi2_2_quantile(ci_level) / n
        else:
            raise ValueError(f"Unknown ellipse_type: {ellipse_type!r}")

        width, height, angle = _cov_to_ellipse(cov, c2)
        ell.append((mean_lwp[j], mean_iwv[j], width, height, angle))

    return mean_lwp, mean_iwv, ell


# =====================================================================
# Main
# =====================================================================
def plot_lwp_iwv_trajectory(
    warm_cloud_datasets,
    nodes=(-180, -150, -120, -90, -60, -30, -15),
    half_width=10,
    ellipse_nodes=(-120, -60, -15),
    ci_level=0.95,
    ellipse_method="bootstrap",
    ellipse_type="confidence",
    n_boot=2000,
    lwp_var="tclw",
    iwv_var="tcwv",
    lwp_flag="tclw_dpg_flag",
    iwv_flag="tcwv_dpg_flag",
    anomaly_reference="long_tail",
    anomaly_tail_minutes=-180,
    conv_threshold=0.5,
    hail_threshold=0.5,
    phases=("liquid", "ice", "mixed"),
    phase_labels=None,
    phase_colors=None,
    balanced=True,
    figsize=(15, 7),
    xlim=None,
    ylim=None,
    point_size=45,
    arrow=True,
    sharexy=True,
    show_zero_lines=True,
    suptitle=None,
    savefig=None,
    random_state=None,
    return_data=False,
    statistic="mean",
):
    """
    Phase-space (ΔLWP, ΔIWV) mean or median trajectory vs time before rain, two
    panels (convective | non-convective), one coloured path per phase,
    with bootstrap confidence ellipses at `ellipse_nodes`.

    Parameters
    ----------
    warm_cloud_datasets : dict
        {"liquid": ds, "ice": ds, "mixed": ds}; each ds needs
        'minutes_before_rain', 'prob_q95_q95', 'prob_hail'.
    nodes : sequence of int
        Lead times (min before rain, negative) to sample. Keep EVEN
        spacing so arc length reads as rate. Default 30-min steps from
        -180 to -15 (terminal node before the wet-radome LWP bias).
    half_width : int
        Half-width (min) of the averaging bin at each node. Keep
        <= half the node spacing to avoid overlapping bins. Default 10.
    ellipse_nodes : sequence of int
        Subset of `nodes` at which to draw confidence ellipses. Fewer is
        cleaner; default 3 nodes.
    ci_level : float
        Confidence level for the ellipses (default 0.95).
    n_boot : int
        Bootstrap resamples over events (default 2000).
    balanced : bool
        If True (default), only events valid at *every* node contribute,
        so the trajectory reflects evolution, not membership churn.
    statistic : {'mean', 'median'}
        Summary across events at each node. Median uses bootstrap confidence
        ellipses. With anomaly_reference='none', input values are already
        baseline-subtracted event anomalies.
    arrow : bool
        Draw an arrowhead on the final segment (towards rain onset).
    sharexy : bool
        Share x/y limits across both panels for fair visual comparison.
    Other parameters mirror the scatter / grid functions.

    Returns
    -------
    fig, axes                  (default)
    fig, axes, data            (if return_data=True)
    """
    if phase_labels is None:
        phase_labels = _default_phase_labels()
    if phase_colors is None:
        phase_colors = {"liquid": "#1f77b4", "ice": "#d62728", "mixed": "#2ca02c"}
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")
    if statistic == "median" and ellipse_method != "bootstrap":
        raise ValueError("Median trajectories need bootstrap confidence ellipses")
    nodes = list(nodes)
    rng = np.random.default_rng(random_state)

    regimes = ["convective", "non-convective"]

    # ---- Build per (regime, phase) trajectories ------------------------
    data = {}  # (regime, phase) -> dict
    for phase in phases:
        ds = warm_cloud_datasets.get(phase)
        if ds is None or ds.sizes.get("time", 0) == 0:
            for regime in regimes:
                data[(regime, phase)] = None
            continue
        try:
            split = _warm_regime_datasets(
                ds, conv_threshold, hail_threshold, require_coords=True
            )
        except ValueError as exc:
            raise ValueError(
                f"Phase '{phase}' missing 'prob_q95_q95'/'prob_hail' coords."
            ) from exc

        for regime, rds in split.items():
            L, W, n_total = _node_matrices(
                rds,
                nodes,
                half_width,
                lwp_var,
                iwv_var,
                lwp_flag,
                iwv_flag,
                anomaly_reference,
                anomaly_tail_minutes,
                balanced=balanced,
            )
            if L.shape[0] < 3:  # need >=3 events for either method
                data[(regime, phase)] = dict(
                    n_kept=L.shape[0], n_total=n_total, insufficient=True
                )
                continue
            if ellipse_method == "bootstrap":
                m_lwp, m_iwv, ell = _bootstrap_trajectory(
                    L, W, n_boot, ci_level, rng, statistic=statistic
                )
            elif ellipse_method == "parametric":
                m_lwp, m_iwv, ell = _parametric_trajectory(
                    L, W, ci_level, ellipse_type=ellipse_type
                )
            else:
                raise ValueError(
                    f"ellipse_method must be 'bootstrap' or 'parametric', "
                    f"got {ellipse_method!r}"
                )
            data[(regime, phase)] = dict(
                **{f"{statistic}_lwp": m_lwp, f"{statistic}_iwv": m_iwv},
                ellipses=ell,
                n_kept=L.shape[0],
                n_total=n_total,
                insufficient=False,
            )

    # ---- Plot ----------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=sharexy, sharey=sharexy, constrained_layout=True
    )

    ell_set = set(ellipse_nodes)
    node_is_ell = [n in ell_set for n in nodes]

    for ax, regime in zip(axes, regimes):
        for phase in phases:
            d = data.get((regime, phase))
            if d is None or d.get("insufficient", True):
                continue
            color = phase_colors.get(phase, "gray")
            label = f"{phase_labels.get(phase, phase)} " f"(n={d['n_kept']})"
            x, y = d[f"{statistic}_lwp"], d[f"{statistic}_iwv"]

            # connecting line
            ax.plot(x, y, "-", color=color, lw=1.6, alpha=0.9, zorder=3)
            # nodes
            ax.scatter(
                x,
                y,
                s=point_size,
                c=color,
                edgecolors="k",
                linewidths=0.5,
                zorder=4,
                label=label,
            )
            # start marker (open square) and arrowhead at onset
            ax.scatter(
                x[0],
                y[0],
                s=point_size * 1.6,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                marker="s",
                zorder=5,
            )
            if arrow and len(x) >= 2:
                ax.annotate(
                    "",
                    xy=(x[-1], y[-1]),
                    xytext=(x[-2], y[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
                    zorder=4,
                )

            # ellipses at selected nodes
            for j, ent in enumerate(d["ellipses"]):
                if not node_is_ell[j] or ent is None:
                    continue
                cx, cy, w, h, ang = ent
                ax.add_patch(
                    Ellipse(
                        (cx, cy),
                        width=w,
                        height=h,
                        angle=ang,
                        facecolor=color,
                        alpha=0.06,
                        edgecolor=color,
                        lw=1.0,
                        zorder=2,
                    )
                )

        if show_zero_lines:
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
            ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax.set_title(regime.capitalize(), fontweight="bold")
        ax.set_xlabel("ΔLWP (g$~$m$^{-2}$)")
        ax.grid(True, alpha=0.25)
        if ellipse_method == "parametric" and ellipse_type == "dispersion":
            ell_desc = f"{ci_level:.0%} spread ellipses"
        elif ellipse_method == "parametric":
            ell_desc = f"{ci_level:.0%} mean CI (F)"
        else:
            ell_desc = f"{ci_level:.0%} mean CI (boot)"
        ax.legend(loc="best", framealpha=0.85, fontsize=9)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel("ΔIWV (kg$~$m$^{-2}$)")

    if suptitle is None:
        suptitle = (
            f"LWP–IWV anomaly trajectory before rain "
            f"(□ = {nodes[0]} min → ▶ {nodes[-1]} min)"
        )
    fig.suptitle(suptitle, fontweight="bold")

    style_fonts(fig)                 # common font family and sizes (figure_style.py)
    if savefig:
        fig.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Saved to {savefig}")

    # ---- Report --------------------------------------------------------
    print(f"\nBalanced event counts (valid at all {len(nodes)} nodes):")
    for regime in regimes:
        for phase in phases:
            d = data.get((regime, phase))
            if d is None:
                continue
            tag = " (INSUFFICIENT, skipped)" if d.get("insufficient") else ""
            print(
                f"  {regime:>15} / {phase_labels.get(phase, phase):>3}: "
                f"{d['n_kept']}/{d['n_total']}{tag}"
            )

    if return_data:
        return fig, axes, data
    return fig, axes
