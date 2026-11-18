"""
Isotherm height from the microwave-radiometer temperature profile
=================================================================

    calculate_minus5_isotherm_height   -5 degC isotherm height, its vertical-grid
                                       uncertainty band and a quality flag
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_T0_SMOOTHING_WINDOW = 5


def calculate_minus5_isotherm_height(
    tprof_ds,
    *,
    temperature_var="T_profile",
    flag_var="T_profile_dpg_flag",
    use_dpg_flag=True,
    isotherm_temperature_K=268.15,
    height_dim="h_msl",
    smoothing_window=None,
):
    """
    Calculate the -5 degC isotherm height on the native temperature-profile grid.

    The returned height is not vertically interpolated. It is the grid level with
    the coldest temperature that is still warmer than -5 degC. QC does not remove
    points from the line; it is stored as qc_pass so plots can use a dashed style
    where QC failed.
    """
    if smoothing_window is None:
        smoothing_window = DEFAULT_T0_SMOOTHING_WINDOW

    temperature = tprof_ds[temperature_var]
    warm_temperature = temperature.where(temperature > isotherm_temperature_K)
    native_height = warm_temperature.idxmin(
        height_dim, skipna=True, fill_value=np.nan
    ).rename("minus5_isotherm_height_m").compute()

    if use_dpg_flag and flag_var in tprof_ds:
        selected_height_flag = (
            tprof_ds[flag_var]
            .where(tprof_ds[height_dim] == native_height)
            .max(height_dim, skipna=True)
            .rename("selected_height_qc_flag")
            .compute()
        )
        qc_pass = (selected_height_flag < 1).rename("qc_pass")
    else:
        selected_height_flag = xr.full_like(native_height, np.nan).rename(
            "selected_height_qc_flag"
        )
        qc_pass = xr.full_like(native_height, True, dtype=bool).rename("qc_pass")

    height_grid = np.asarray(tprof_ds[height_dim].values, dtype=float)
    height_midpoints = (height_grid[:-1] + height_grid[1:]) / 2
    cell_lower_edges = np.r_[
        height_grid[0] - (height_grid[1] - height_grid[0]) / 2,
        height_midpoints,
    ]
    cell_upper_edges = np.r_[
        height_midpoints,
        height_grid[-1] + (height_grid[-1] - height_grid[-2]) / 2,
    ]
    cell_width = cell_upper_edges - cell_lower_edges
    lower_half_width = height_grid - cell_lower_edges
    upper_half_width = cell_upper_edges - height_grid

    local_resolution = xr.apply_ufunc(
        np.interp,
        native_height,
        kwargs={"xp": height_grid, "fp": cell_width},
    ).rename("vertical_resolution_m")
    lower_uncertainty = xr.apply_ufunc(
        np.interp,
        native_height,
        kwargs={"xp": height_grid, "fp": lower_half_width},
    )
    upper_uncertainty = xr.apply_ufunc(
        np.interp,
        native_height,
        kwargs={"xp": height_grid, "fp": upper_half_width},
    )

    if smoothing_window and "time" in native_height.dims:
        smoothing_window = min(int(smoothing_window), native_height.sizes["time"])
        plot_height = native_height.rolling(
            time=smoothing_window, center=True, min_periods=1
        ).mean().where(native_height.notnull())
    else:
        plot_height = native_height
    plot_height = plot_height.rename("minus5_isotherm_height_smoothed_m")

    band_lower = (plot_height - lower_uncertainty).rename("band_lower_m")
    band_upper = (plot_height + upper_uncertainty).rename("band_upper_m")

    return xr.Dataset(
        {
            "height_m": native_height,
            "plot_height_m": plot_height,
            "vertical_resolution_m": local_resolution,
            "band_lower_m": band_lower,
            "band_upper_m": band_upper,
            "selected_height_qc_flag": selected_height_flag,
            "qc_pass": qc_pass,
        }
    )


__all__ = [
    "calculate_minus5_isotherm_height",
    "split_definitions",
    "summarize_temperature_profile_vertical_resolution",
    "transform_dataset_coordinates",
]
