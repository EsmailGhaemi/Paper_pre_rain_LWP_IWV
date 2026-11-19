"""
Sensitivity of the convective classification to weights and threshold
======================================================================

The four criterion scores (prob_reflectivity, prob_zdr_column, prob_kdp,
prob_doppler) are stored per grid column and time step for every event. Here they
are recombined with other weights, the event statistic (P95 over the grid, then
P95 over the event) is recomputed, and the number of convective events is counted
(Fig. A1 of the paper).

    WEIGHT_SETS, THRESHOLDS     tested weight combinations and thresholds
    event_probability           event statistic for one weight combination
    run_weight_sensitivity      number of convective events per weight set
    run_threshold_sensitivity   number of convective events per threshold
    stability_metrics           coefficient of variation of the event counts
    classification_agreement    share of events classified the same by all weight sets
    plot_sensitivity            Fig. A1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figure_style import style_fonts

#                 reflectivity  ZDR column   KDP        spectrum width
WEIGHT_SETS = [
    {"name": "Baseline",           "w_Z": 0.35, "w_ZDR": 0.30, "w_KDP": 0.25, "w_SW": 0.10},
    {"name": "Z_dominant",         "w_Z": 0.70, "w_ZDR": 0.10, "w_KDP": 0.10, "w_SW": 0.10},
    {"name": "ZDR_dominant",       "w_Z": 0.10, "w_ZDR": 0.60, "w_KDP": 0.20, "w_SW": 0.10},
    {"name": "KDP_dominant",       "w_Z": 0.10, "w_ZDR": 0.20, "w_KDP": 0.60, "w_SW": 0.10},
    {"name": "Reflectivity_focus", "w_Z": 0.50, "w_ZDR": 0.20, "w_KDP": 0.20, "w_SW": 0.10},
    {"name": "Polarimetric_focus", "w_Z": 0.20, "w_ZDR": 0.35, "w_KDP": 0.35, "w_SW": 0.10},
    {"name": "No_SW",              "w_Z": 0.40, "w_ZDR": 0.35, "w_KDP": 0.25, "w_SW": 0.00},
]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]   # convective if event probability >= threshold


def event_probability(components, events, weights):
    """Event statistic of the convective probability for one weight combination.

    Parameters
    ----------
    components : xr.Dataset
        prob_reflectivity, prob_zdr_column, prob_kdp, prob_doppler with dimensions
        (time, y_utm, x_utm), all events on one time axis.
    events : pd.DataFrame
        Columns ``start`` and ``end``.
    weights : dict
        Keys w_Z, w_ZDR, w_KDP, w_SW.

    Returns
    -------
    np.ndarray
        P95 over the grid, then P95 over the event time steps; NaN if the event has
        no radar time step.
    """
    conv_prob = (weights["w_Z"] * components["prob_reflectivity"]
                 + weights["w_ZDR"] * components["prob_zdr_column"]
                 + weights["w_KDP"] * components["prob_kdp"]
                 + weights["w_SW"] * components["prob_doppler"])
    q95_space = conv_prob.quantile(0.95, dim=["x_utm", "y_utm"]).compute()

    probs = []
    for _, ev in events.iterrows():
        in_event = (q95_space.time >= ev["start"]) & (q95_space.time <= ev["end"])
        probs.append(float(q95_space.where(in_event, drop=True).quantile(0.95).item())
                     if in_event.any() else np.nan)
    return np.array(probs)


def run_weight_sensitivity(components, events, threshold=0.5):
    """Number of convective events for every weight set in :data:`WEIGHT_SETS`.

    Returns
    -------
    results : pd.DataFrame
        One row per weight set (weights, n_convective, n_nonconvective, pct_convective).
    categories : dict
        Boolean convective flag per event for each weight set.
    """
    rows, categories = [], {}
    n_events = len(events)
    for w in WEIGHT_SETS:
        is_conv = event_probability(components, events, w) >= threshold
        n_conv = int(np.nansum(is_conv))
        rows.append({"name": w["name"], "w_Z": w["w_Z"], "w_ZDR": w["w_ZDR"], "w_KDP": w["w_KDP"],
                     "w_SW": w["w_SW"], "n_convective": n_conv, "n_nonconvective": n_events - n_conv,
                     "pct_convective": n_conv / n_events * 100})
        categories[w["name"]] = is_conv
    return pd.DataFrame(rows), categories


def run_threshold_sensitivity(components, events, weights=WEIGHT_SETS[0]):
    """Number of convective events for every threshold in :data:`THRESHOLDS` (baseline weights)."""
    probs = event_probability(components, events, weights)
    return pd.DataFrame([{"threshold": t, "n_convective": int(np.nansum(probs >= t)),
                          "pct_convective": np.nansum(probs >= t) / len(events) * 100}
                         for t in THRESHOLDS])


def stability_metrics(results):
    """Mean, standard deviation and coefficient of variation (CV) of the event counts."""
    rows = []
    for m in ["n_convective", "pct_convective"]:
        vals = results[m]
        cv = vals.std() / vals.mean() * 100 if vals.mean() != 0 else 0
        rows.append({"metric": m, "mean": vals.mean(), "std": vals.std(), "CV_pct": cv,
                     "min": vals.min(), "max": vals.max(),
                     "stability": "High" if cv < 10 else ("Moderate" if cv < 25 else "Low")})
    return pd.DataFrame(rows)


def classification_agreement(categories):
    """Share of events [%] that all weight sets put in the same class."""
    names = list(categories)
    n = len(categories[names[0]])
    agree = sum(1 for i in range(n) if len({categories[name][i] for name in names}) == 1)
    return agree / n * 100


def plot_sensitivity(results, thresholds):
    """Fig. A1: convective events per weight set (left) and per threshold (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")

    x = range(len(results))
    axes[0].bar(x, results["n_convective"], color="darkred", edgecolor="black", linewidth=0.4, alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results["name"], rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("N Convective Events")
    axes[0].set_title("Weight Sensitivity")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(thresholds["threshold"], thresholds["n_convective"], marker="o", color="darkblue",
                 lw=1.25, ms=4.5, markerfacecolor="white", markeredgewidth=1.0)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("N Convective")
    axes[1].set_title("Threshold Sensitivity")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].spines[["top", "right"]].set_visible(False)

    style_fonts(fig, tight=True)     # common font family and sizes (figure_style.py)
    return fig, axes
