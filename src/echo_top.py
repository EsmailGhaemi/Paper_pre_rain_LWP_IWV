"""
Radar echo-top height around the radiometer
===========================================

Echo-top height (ETH) = highest grid level with Z_h above a threshold (0 dBZ) in
each column of the Cartesian X-band grid. It is the cloud-top proxy used to
classify the rain type (warm / cold / mixed rain, Fig. 2 of the paper) by
comparing the median and P90 of the ETH with the -5 degC isotherm.

    echo_top_height          ETH of one gridded radar dataset
    echo_top_height_files    ETH of many event files -> one dataset
"""

from pathlib import Path

import numpy as np
import xarray as xr

RADIOMETER_XY = (564565.44, 5200868.51)   # radiometer location, UTM 33N [m]


def echo_top_height(ds, zh_min=0.0, center=RADIOMETER_XY, half_width=3000.0):
    """Echo-top height of every column in a box around the radiometer.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded radar data with ``Z_h_corr`` [dBZ] and dimensions
        (time, h_msl_m, y_utm, x_utm).
    zh_min : float
        Reflectivity threshold [dBZ]; gates with Z_h > zh_min are echo.
    center : tuple of float
        Box centre (x, y) in UTM 33N [m].
    half_width : float
        Half width of the box [m].

    Returns
    -------
    xr.DataArray
        ``echo_top_height`` [m a.s.l.] (time, y_utm, x_utm); time steps without
        any echo in the box are removed.
    """
    ds = ds.sel(x_utm=slice(center[0] - half_width, center[0] + half_width),
                y_utm=slice(center[1] - half_width, center[1] + half_width))
    zh = ds["Z_h_corr"].where(ds["Z_h_corr"] > zh_min)
    height = ds["h_msl_m"] * xr.ones_like(zh)
    eth = height.where(zh.notnull()).max(dim="h_msl_m").load()
    has_echo = eth.notnull().any(dim=["x_utm", "y_utm"])
    return eth.sel(time=has_echo).rename("echo_top_height")


def echo_top_height_files(files, **kwargs):
    """Echo-top height of several gridded event files, merged on one time axis.

    Overlapping event windows give duplicate time steps; the first is kept.

    Parameters
    ----------
    files : list of str or Path
        Gridded radar files (output of the gridding in notebook 02).
    **kwargs
        Passed to :func:`echo_top_height`.

    Returns
    -------
    xr.Dataset
        ``echo_top_height`` (time, y_utm, x_utm), sorted in time.
    """
    parts = []
    for f in sorted(Path(p) for p in files):
        with xr.open_dataset(f) as ds:
            eth = echo_top_height(ds, **kwargs)
        if eth.time.size > 0:
            parts.append(eth)
    eth = xr.concat(parts, dim="time").sortby("time")
    eth = eth.isel(time=~eth.indexes["time"].duplicated(keep="first"))
    eth.attrs = {"long_name": "radar echo-top height (highest gate with Z_h > 0 dBZ)",
                 "units": "m a.s.l."}
    return eth.to_dataset()
