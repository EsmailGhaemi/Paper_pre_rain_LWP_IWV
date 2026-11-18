"""
Echo-top statistics, isotherm heights and pre-event segments
============================================================

    find_freezing_level                    height of an isotherm from the microwave-radiometer
                                           temperature profile (used with -5 degC)
    spatial_echo_top_stats                 spatial median / P90 / P95 of the radar echo top
    event_echo_top_stats                   echo-top statistics around each rain onset
    pre_event_segments_with_prior_intensity  6-h pre-onset time series of each event
    extract_pre_events_by_cloud_type       pre-onset time series split by rain type
    exclude_times_fast                     cloud-only time steps (outside rain events)
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr


def spatial_echo_top_stats(
    ds: xr.Dataset,
    events: pd.DataFrame,
    x_slice: Optional[Union[slice, Tuple[float, float]]] = None,
    y_slice: Optional[Union[slice, Tuple[float, float]]] = None,
    *,
    echo_var: str = "echo_top_height",
    time_dim: str = "time",
    x_dim: str = "X",
    y_dim: str = "Y",
) -> xr.Dataset:
    """
    Compute spatial median, p90, p95 of echo_top_height over (X, Y)
    at every time step that has at least one valid pixel.
    Then label each time step with the start/end it falls within.

    Parameters
    ----------
    ds : xarray Dataset with echo_top_height variable
    events : DataFrame with columns "start", "end", "duration"      
    x_slice, y_slice : optional slices to select a spatial subset
    echo_var : name of the echo top variable in ds
    time_dim, x_dim, y_dim : names of the dimensions in ds

    Returns
    -------
    Xarray dataset with variables:
        sp_median_m, sp_p90_m, sp_p95_m : spatial statistics
        start, end : event start/end labels for each time step
    """
    required_cols = {"start", "end", "duration"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if echo_var not in ds:
        raise ValueError(f"'{echo_var}' not in dataset.")
    for d in (time_dim, x_dim, y_dim):
        if d not in ds.dims:
            raise ValueError(f"Dimension '{d}' not in dataset.")

    events = events.copy()
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])

    def _to_slice(v):
        if v is None:
            return None
        if isinstance(v, slice):
            return v
        return slice(v[0], v[1])

    ds = ds.sortby(time_dim)
    if (xs := _to_slice(x_slice)) is not None:
        ds = ds.sel({x_dim: xs})
    if (ys := _to_slice(y_slice)) is not None:
        ds = ds.sel({y_dim: ys})

    da = ds[echo_var]

    # Drop all-NaN time steps
    valid_mask = da.notnull().any(dim=(x_dim, y_dim))
    da = da.sel({time_dim: valid_mask})

    # Spatial statistics (vectorised)
    q = da.quantile([0.5, 0.9, 0.95], dim=(x_dim, y_dim), skipna=True)
    times = pd.DatetimeIndex(da[time_dim].values)

    # Assign event labels
    ev_start_arr = np.full(len(times), np.datetime64("NaT"), dtype="datetime64[ns]")
    ev_end_arr = np.full(len(times), np.datetime64("NaT"), dtype="datetime64[ns]")

    for _, row in events.iterrows():
        mask = (times >= row["start"]) & (times <= row["end"])
        ev_start_arr[mask] = row["start"]
        ev_end_arr[mask] = row["end"]

    out = xr.Dataset(
        {
            "sp_median_m": (time_dim, q.sel(quantile=0.5).values),
            "sp_p90_m": (time_dim, q.sel(quantile=0.9).values),
            "sp_p95_m": (time_dim, q.sel(quantile=0.95).values),
            "start": (time_dim, ev_start_arr),
            "end": (time_dim, ev_end_arr),
        },
        coords={time_dim: da[time_dim].values},
        attrs={
            "description": "Spatial echo-top statistics at each valid time step.",
            "echo_variable": echo_var,
            "units": "metres",
        },
    )
    return out


def event_echo_top_stats(
    stats_ds: xr.Dataset,
    events: pd.DataFrame,
    freezing_ds: xr.Dataset,
    *,
    pre_event_hours: float = 1.0,
    after_start_offset: float = 1.0,
    fl_var: str = "freezing_level_height",
    fl_qc_var: str = "QC",
    fl_time_dim: str = "time",
    fl_window_start_hours: float = 1.0,
    fl_window_end_minutes: float = 30.0,
    fl_fallback_max_hours: float = 2.0,
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Aggregate per-time-step spatial echo-top statistics into per-event summaries,
    and attach a freezing-level height derived from a radiometer/sounding dataset.

    Echo-top aggregation:
        Pre-event window  [start - pre_event_hours, start] -> ev_median_m, ev_p90_m, ev_p95_m
        During-event check [start, start + after_start_offset] -> check_dur_ev_*

    Freezing-level logic (over [start - fl_window_start_hours, start - fl_window_end_minutes]):
        1. Select time steps where QC < 4  ->  average their heights.
        2. If all QC >= 4  ->  find nearest QC < 4 step (within fl_fallback_max_hours).
    Parameters
    ----------
    stats_ds : Dataset with spatial echo-top statistics (from spatial_echo_top_stats)
    events : DataFrame of the events with columns "start", "end", "duration"
    freezing_ds : Dataset with freezing-level height and QC variable
    pre_event_hours : hours before event start to compute pre-event stats
    after_start_offset : hours after event start to check during-event stats
    fl_var : name of the freezing-level height variable in freezing_ds
    fl_qc_var : name of the QC variable in freezing_ds
    fl_time_dim : name of the time dimension in freezing_ds
    fl_window_start_hours : hours before event start to begin freezing-level window
    fl_window_end_minutes : minutes after event finish to end freezing-level window
    fl_fallback_max_hours : maximum hours to look for nearest good QC if none in window
    time_dim : name of the time dimension in stats_ds

    Returns
    -------
    xr.Dataset
        Dataset with per-event statistics and freezing-level height.
    """
    events = events.copy()
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])

    pre_offset = pd.Timedelta(hours=pre_event_hours)
    after_offset = pd.Timedelta(hours=after_start_offset)
    fl_win_start = pd.Timedelta(hours=fl_window_start_hours)
    fl_win_end = pd.Timedelta(minutes=fl_window_end_minutes)
    fl_max_gap = pd.Timedelta(hours=fl_fallback_max_hours)

    echo_times = pd.DatetimeIndex(stats_ds[time_dim].values)
    fl_times = pd.DatetimeIndex(freezing_ds[fl_time_dim].values)
    fl_heights = freezing_ds[fl_var].values
    fl_qc = freezing_ds[fl_qc_var].values

    records = []
    for _, row in events.iterrows():
        ev_start = row["start"]
        ev_end = row["end"]

        # --- Echo-top: pre-event window [start - pre_offset, start] ---
        et_mask_pre = (echo_times >= ev_start - pre_offset) & (echo_times <= ev_start)
        n_pre = et_mask_pre.sum()

        # --- Echo-top: during-event check [start, start + after_offset] ---
        et_mask_dur = (echo_times >= ev_start - after_offset) & (
            echo_times <= ev_start + after_offset
        )
        n_dur = et_mask_dur.sum()

        # Pre-event stats (guarded against empty)
        if n_pre > 0:
            sub = stats_ds.sel({time_dim: et_mask_pre})
            ev_med = np.nanmedian(sub["sp_median_m"].values)
            ev_p90 = np.nanpercentile(sub["sp_p90_m"].values, 90)
            ev_p95 = np.nanpercentile(sub["sp_p95_m"].values, 95)
            ev_median_sp90 = np.nanmedian(sub["sp_p90_m"].values)
        else:
            ev_med = ev_p90 = ev_p95 = ev_median_sp90 = np.nan

        # During-event check stats (guarded against empty)
        if n_dur > 0:
            sub_check = stats_ds.sel({time_dim: et_mask_dur})
            check_in_ev_med = np.nanmedian(sub_check["sp_median_m"].values)
            check_in_ev_p90 = np.nanpercentile(sub_check["sp_p90_m"].values, 90)
            check_in_ev_med_p90 = np.nanmedian(sub_check["sp_p90_m"].values)
        else:
            check_in_ev_med = check_in_ev_p90 = check_in_ev_med_p90 = np.nan

        # --- Freezing-level height ---
        fl_t0 = ev_start - fl_win_start
        fl_t1 = ev_start - fl_win_end
        fl_mask = (fl_times >= fl_t0) & (fl_times <= fl_t1)

        freezing_val = np.nan
        if fl_mask.any():
            good_qc = fl_mask & (fl_qc < 4)
            if good_qc.any():
                freezing_val = np.nanmean(fl_heights[good_qc])
            else:
                freezing_val = _nearest_good_fl(
                    fl_times, fl_heights, fl_qc, fl_t0, fl_t1, fl_max_gap
                )
        else:
            freezing_val = _nearest_good_fl(
                fl_times, fl_heights, fl_qc, fl_t0, fl_t1, fl_max_gap
            )

        records.append(
            dict(
                start=ev_start,
                end=ev_end,
                ev_median_m=ev_med,
                ev_p90_m=ev_p90,
                ev_p95_m=ev_p95,
                ev_median_s_p90=ev_median_sp90,
                check_dur_ev_med=check_in_ev_med,
                check_dur_ev_90=check_in_ev_p90,
                check_dur_ev_med_p90=check_in_ev_med_p90,
                n_timesteps=int(n_pre),
                n_in_ev_ts=int(n_dur),
                freezing_level_m=freezing_val,
            )
        )

    df = pd.DataFrame(records)
    n_events = len(df)

    out = xr.Dataset(
        {
            "ev_median_m": ("event", df["ev_median_m"].values),
            "ev_p90_m": ("event", df["ev_p90_m"].values),
            "ev_p95_m": ("event", df["ev_p95_m"].values),
            "ev_median_s_p90": ("event", df["ev_median_s_p90"].values),
            "check_dur_ev_med": ("event", df["check_dur_ev_med"].values),
            "check_dur_ev_90": ("event", df["check_dur_ev_90"].values),
            "check_dur_ev_med_p90": ("event", df["check_dur_ev_med_p90"].values),
            "n_timesteps": ("event", df["n_timesteps"].values.astype(int)),
            "n_dur_ev": ("event", df["n_in_ev_ts"].values.astype(int)),
            "start": ("event", df["start"].values),
            "end": ("event", df["end"].values),
            "freezing_level_m": ("event", df["freezing_level_m"].values),
        },
        coords={"event": np.arange(n_events)},
        attrs={
            "description": (
                "Per-event echo-top statistics with freezing level. "
                f"Pre-event: [start - {pre_event_hours}h, start]. "
                f"During-event check: [start, start + {after_start_offset}h]. "
                f"Freezing level: mean of QC<4 steps in "
                f"[-{fl_window_start_hours}h to -{fl_window_end_minutes}min]."
            ),
            "units": "metres",
        },
    )
    return out


def _nearest_good_fl(fl_times, fl_heights, fl_qc, fl_t0, fl_t1, fl_max_gap):
    """Fallback: find the nearest good-QC freezing-level measurement.

    Parameters
    ----------
    fl_times : array of datetime64 
    fl_heights : array of freezing-level heights
    fl_qc : array of QC values
    fl_t0, fl_t1 : datetime64 start/end of the window to search
    fl_max_gap : maximum timedelta to accept nearest good QC

    Returns
    -------
    freezing-level height (float) or np.nan if none found."""
    all_good_idx = np.where(fl_qc < 4)[0]
    if len(all_good_idx) == 0:
        return np.nan
    window_centre = fl_t0 + (fl_t1 - fl_t0) / 2
    diffs = np.abs(fl_times[all_good_idx] - window_centre)
    nearest_pos = all_good_idx[np.argmin(diffs)]
    if diffs.min() <= fl_max_gap:
        return float(fl_heights[nearest_pos])
    return np.nan


def find_freezing_level(T, QC, h_coord="h_msl", threshold=273.15):
    """Find height where T crosses the given threshold via sign-change interpolation.

    Note: When called with threshold=268.15, this finds the -5 deg C isotherm,
    not the 0 deg C freezing level.

    Parameters
    ----------
    T : xarray DataArray of temperature (K) with height dimension h_coord
    QC : xarray DataArray of quality control flags (same shape as T)
    h_coord : name of the height dimension in T and QC
    threshold : temperature threshold (K) to find the crossing height

    Returns
    -------
        h_zero : height (m) where T crosses threshold (NaN if no crossing)
        qc_below : QC value just below the crossing height (NaN if no crossing)
        qc_above : QC value just above the crossing height (NaN if no crossing)
    """

    def _interp_zero(t_profile, qc_profile, h_values):
        t_shifted = t_profile - threshold
        sign_change = np.diff(np.sign(t_shifted))
        idx = np.where(sign_change != 0)[0]

        if len(idx) == 0:
            return np.nan, np.nan, np.nan

        i = idx[0]
        t0, t1 = t_shifted[i], t_shifted[i + 1]
        h0, h1 = h_values[i], h_values[i + 1]

        # Guard against division by zero (both levels at exactly the threshold)
        denom = t1 - t0
        if abs(denom) < 1e-12:
            h_zero = (h0 + h1) / 2.0
        else:
            h_zero = h0 + (0 - t0) * (h1 - h0) / denom

        qc_below = qc_profile[i]
        qc_above = qc_profile[i + 1]
        return h_zero, qc_below, qc_above

    h_zero, qc_below, qc_above = xr.apply_ufunc(
        _interp_zero,
        T,
        QC,
        T[h_coord],
        input_core_dims=[[h_coord], [h_coord], [h_coord]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return h_zero, qc_below, qc_above


def pre_event_segments_with_prior_intensity(
    events_df,
    source_data,
    flag_var="tclw_dpg_flag",
    lookback_hours=3.0,
    buffer_minutes=0.0,
    fill_no_prior=-1,
):
    """Extract pre-event segments for each event and flag prior precipitation.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have columns 'start' and 'end'. Optionally 'prob_q95_q95', 'prob_hail'.
    source_data : xr.Dataset
        Observation dataset with a 'time' dimension.
    flag_var : str
        Name of the QC flag variable in source_data. If a timestep has flag==1
        (interpolated), the prior_event_intensity is set to fill_no_prior.
    lookback_hours : float
        Hours before event start to extract.
    buffer_minutes : float
        Minutes before start to exclude (buffer between window and event).
    fill_no_prior : int
        Value assigned to prior_event_intensity when interpolated data is detected.

    Returns
    -------
    segments : list[xr.Dataset]
        Pre-event segments with coordinates: event_id, minutes_before_rain,
        prior_event_intensity, and optionally prob_q95_q95, prob_hail.
    events_no_data : list[pd.Series]
        Events for which no observation data was available.
    """
    segments, events_no_data = [], []
    df = events_df.copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values("start").reset_index(drop=True)

    for i in range(len(df)):
        s = df.at[i, "start"]
        window_start = s - pd.Timedelta(hours=lookback_hours)
        window_end = s - pd.Timedelta(minutes=buffer_minutes)
        seg = source_data.sel(time=slice(window_start, window_end))

        if "time" not in seg.dims or seg.sizes.get("time", 0) == 0:
            events_no_data.append(df.iloc[i])
            continue

        # Check if any timestep in the lookback window has flag == 1 (interpolated)
        if flag_var in seg.data_vars:
            has_prior_flag = np.any(seg[flag_var].values == 1)
        else:
            has_prior_flag = False

        prior_int = fill_no_prior if has_prior_flag else 0
        seg = seg.assign_coords(event_id=i)

        if "prob_q95_q95" in df.columns:
            seg = seg.assign_coords(prob_q95_q95=df.at[i, "prob_q95_q95"])
            seg = seg.assign_coords(prob_hail=df.at[i, "prob_hail"])

        # Minutes before rain (negative values)
        delta_minutes = (pd.to_datetime(seg.time.values) - s) / pd.Timedelta(minutes=1)
        mins_before = np.floor(delta_minutes).astype(int)
        seg = seg.assign_coords(minutes_before_rain=("time", mins_before))
        seg = seg.assign_coords(prior_event_intensity=prior_int)

        segments.append(seg)

    return segments, events_no_data


def extract_pre_events_by_cloud_type(
    events_df,
    cloud_datasets,
    obs_data,
    flag_var="tclw_dpg_flag",
    lookback_hours=6,
    buffer_minutes=0,
):
    """
    Extract pre-event segments for multiple cloud types in one call.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events to filter (must have 'start' column).
    cloud_datasets : dict[str, xr.Dataset]
        Mapping of cloud type name -> dataset with 'start' coordinate.
        E.g. {"liquid": liquid_clouds, "ice": ice_clouds, "PG": PG}
    obs_data : xr.Dataset
        Observation data for pre-event extraction.
    flag_var : str
        Flag variable name in obs_data.

    Returns
    -------
    dict[str, xr.Dataset]
        Mapping of cloud type name -> concatenated pre-event dataset.
    """
    results = {}
    for name, cloud_ds in cloud_datasets.items():
        cloud_starts = cloud_ds["start"].values

        mask = events_df["start"].isin(cloud_starts)
        filtered = events_df[mask]

        if len(filtered) == 0:
            print(f"  Warning: no events matched for cloud type '{name}'")
            continue

        segments, _ = pre_event_segments_with_prior_intensity(
            filtered,
            obs_data,
            flag_var=flag_var,
            lookback_hours=lookback_hours,
            buffer_minutes=buffer_minutes,
        )
        if segments:
            results[name] = xr.concat(segments, dim="time")
        else:
            print(f"  Warning: no pre-event data for cloud type '{name}'")

    return results


def exclude_times_fast(X_big, X_sub=None, intervals_df=None, inclusive="both"):
    """Remove times matching X_sub or falling within event intervals.
    
    Parameters
    ----------
    X_big : array-like
        Array of times to filter.
    X_sub : array-like, optional
        Times to exclude from X_big.
    intervals_df : pd.DataFrame, optional
        DataFrame with 'start' and 'end' columns defining intervals to exclude.
    inclusive : str, optional
        How to treat the interval boundaries. One of 'both', 'left', 'right', or 'neither'.
        
    Returns
    -------
    X_stage : np.ndarray    
    """
    X_big = np.asarray(X_big)

    keep_mask = np.ones(X_big.shape[0], dtype=bool)
    if X_sub is not None:
        X_sub = np.asarray(X_sub)
        keep_mask &= ~np.isin(X_big, X_sub)

    X_stage = X_big[keep_mask]
    if intervals_df is None or len(intervals_df) == 0:
        return X_stage, np.zeros(0, dtype=bool)

    df = intervals_df.copy()
    if not {"start", "end"}.issubset(df.columns):
        raise ValueError("intervals_df must contain 'start' and 'end' columns.")

    order = np.argsort(X_stage)
    X_sorted = X_stage[order]

    starts = df["start"].to_numpy()
    ends = df["end"].to_numpy()
    idx = np.argsort(starts)
    starts = starts[idx]
    ends = ends[idx]

    remove_sorted = np.zeros(X_sorted.shape[0], dtype=bool)

    for s, e in zip(starts, ends):
        if inclusive == "both":
            left = np.searchsorted(X_sorted, s, side="left")
            right = np.searchsorted(X_sorted, e, side="right")
        elif inclusive == "left":
            left = np.searchsorted(X_sorted, s, side="left")
            right = np.searchsorted(X_sorted, e, side="left")
        elif inclusive == "right":
            left = np.searchsorted(X_sorted, s, side="right")
            right = np.searchsorted(X_sorted, e, side="right")
        elif inclusive == "neither":
            left = np.searchsorted(X_sorted, s, side="right")
            right = np.searchsorted(X_sorted, e, side="left")
        else:
            raise ValueError("inclusive must be one of: both, left, right, neither")

        if right > left:
            remove_sorted[left:right] = True

    keep_sorted = ~remove_sorted
    keep_stage = np.empty_like(keep_sorted)
    keep_stage[order] = keep_sorted

    return X_stage[keep_stage], remove_sorted
