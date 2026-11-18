"""
Rain-event detection from rain-gauge time series
=================================================

Detects rain time steps at a WegenerNet station, groups them into rain events
and merges events that are separated by short dry gaps.

    select_precip_timesteps
    detect_events_vectorized
    merge_close_events_vectorized
"""

import xarray as xr
import pandas as pd
import numpy as np


def select_precip_timesteps (ds, threshold=0.1,roll_time=5):
    
    """
    Extract data with precipitation.
    
    Parameters 
    - ds : xarray dataset
        Dataset with precip and temp (as precipitation and temperature)
    - threshold: float
        Minimum value to consider the step as precipitation step
    - roll_time : integer
        Number of timesteps to look forward and backward to check the minimum threshold
    
    Returns:
    - ds_filtered : xarray Dataset
        Filtered dataset with precipitation 
    """
    mask = ((ds.precip*12 > threshold) & ((ds.precip*12).rolling(time=roll_time, center=True).min() > threshold))
    times = ds.precip.time.where((mask).compute(), drop=True).time.values
    temp_g = ds.temp.where((mask).compute(), drop=True)
    precip_amount = ds.precip.where((mask).compute(), drop=True)
    filtered = pd.DataFrame([{"time": t,
                              "precip_val_g": precip_amount.sel(time=t).values,
                              "temp":temp_g.sel(time=t, method='nearest').values} for t in times])
    return filtered


    
def detect_events_vectorized(data, gap_threshold_minutes=30):
    """
    Separate precipitation events based on a time interval
    """
    
    if hasattr(data, 'time'):
        times = pd.to_datetime(data.time.values)
    else:
        times = pd.to_datetime(data.values)
    
    times = pd.Series(times).sort_values().reset_index(drop=True)
    
    # Calculate time differences
    time_diffs = times.diff()
    gap_threshold = pd.Timedelta(minutes=gap_threshold_minutes)
    
    # Find event boundaries
    event_starts = times[time_diffs > gap_threshold].tolist()
    event_starts.insert(0, times.iloc[0])  # Add first event start
    
    event_ends = times[time_diffs.shift(-1) > gap_threshold].tolist()
    event_ends.append(times.iloc[-1])  # Add last event end
    
    # Create events dataframe
    events_df = pd.DataFrame({
        'start': event_starts,
        'end': event_ends
    })
    
    events_df['duration'] = events_df['end'] - events_df['start']
    
    return events_df


def merge_close_events_vectorized(events_df, merge_threshold_hours=1.5):
    """
    Merge events if the time interval is 1.5 hours
    """
    
    if len(events_df) <= 1:
        return events_df.copy()
    
    # Sort events by start time
    events_sorted = events_df.sort_values('start').reset_index(drop=True)
    
    merge_threshold = pd.Timedelta(hours=merge_threshold_hours)
    
    # Calculate gaps between consecutive events
    gaps = events_sorted['start'].shift(-1) - events_sorted['end']
    
    # Find events that should NOT be merged (gap > threshold or last event)
    should_end_group = (gaps > merge_threshold) | gaps.isna()
    
    # Create group identifiers
    group_ids = should_end_group.shift(1, fill_value=False).cumsum()
    
    # Group and aggregate
    merged = events_sorted.groupby(group_ids).agg({
        'start': 'min',
        'end': 'max',
        'prob_q95_q95': 'max',
        'prob_median_max': 'max',
        'prob_hail': 'max'
    }).reset_index(drop=True)
    
    # Calculate duration
    merged['duration'] = merged['end'] - merged['start']
    
    return merged
