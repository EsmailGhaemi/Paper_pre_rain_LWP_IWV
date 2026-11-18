"""
Rain-event statistics
=====================

Adds per-event statistics (precipitation variability, mean 2-m temperature,
duration, ...) to an event table.

    calculate_event_statistics
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def calculate_event_statistics(
    events_df: pd.DataFrame,
    source_data: Union[pd.DataFrame, 'xarray.Dataset'],
    variable_name: str,
    rain_rate_data: Optional[Union[pd.DataFrame, 'xarray.Dataset']] = None,
    rain_rate_var: str = 'precip_val_g'
) -> pd.DataFrame:
    """
    Calculate statistics for each event and add them to the events DataFrame.

    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame with 'start' and 'end' columns representing event periods
    source_data : pd.DataFrame or xarray.Dataset
        Dataset containing the variable to analyze
    variable_name : str
        Name of the variable to calculate statistics for
    rain_rate_data : pd.DataFrame or xarray.Dataset, optional
        Separate dataset for rain rate (defaults to source_data if None)
    rain_rate_var : str, optional
        Name of rain rate variable (default: 'precip_val_g')

    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added columns: mean_value, max_value, std_value,
        mean_intensity_mm_hr, sump, dur_h
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = events_df.copy()

    # Initialize lists for statistics
    stats = {
        'mean_values': [],
        'max_values': [],
        'std_values': [],
        'mean_intensities': [],
        'sum_p': [],
        'duration_h': [],
        'mean_temp': [],
        'min_temp' :[]
    }

    # Use source_data for rain rate if not provided
    rain_rate_data = source_data if rain_rate_data is None else rain_rate_data

    for _, event in events_df.iterrows():
        event_start = event['start']
        event_end = event['end']

        try:
            # Extract data for the event period
            event_data = source_data.sel(time=slice(event_start, event_end))

            # Calculate statistics for the variable
            var_data = event_data[variable_name] if hasattr(event_data, variable_name) else event_data
            stats['mean_values'].append(float(var_data.mean().values))
            stats['max_values'].append(float(var_data.max().values))
            stats['std_values'].append(float(var_data.std().values))
            stats['mean_temp'].append(float(event_data.temp.mean(
                
            ).values))
            stats['min_temp'].append(float(event_data.temp.min().values))
            # Calculate rain intensity (total precipitation / duration in hours)
            rain_data = rain_rate_data.sel(time=slice(event_start, event_end))
            total_precip = float(rain_data[rain_rate_var].sum().values)

            # Calculate duration in hours
            duration_hours = (event_end - event_start) / pd.Timedelta(hours=1)
            if duration_hours == 0 and total_precip > 0.01:
                duration_hours = 5 / 60  # Assume 5-minute duration for zero-duration events

            # Calculate intensity in mm/hr
            intensity = total_precip / duration_hours if duration_hours > 0 else np.nan

            stats['sum_p'].append(total_precip)
            stats['duration_h'].append(duration_hours)
            stats['mean_intensities'].append(intensity)

        except Exception as e:
            # Handle errors by appending NaN values
            for key in stats:
                stats[key].append(np.nan)
            print(f"Error processing event {event_start} to {event_end}: {str(e)}")

    # Add statistics to DataFrame
    result_df['sump'] = stats['sum_p']
    result_df['dur_h'] = stats['duration_h']
    result_df['mean_value'] = stats['mean_values']
    result_df['max_value'] = stats['max_values']
    result_df['std_value'] = stats['std_values']
    result_df['mean_intensity'] = stats['mean_intensities']
    result_df['mean_temp'] = stats['mean_temp']
    result_df['min_temp'] = stats['min_temp']
    return result_df
