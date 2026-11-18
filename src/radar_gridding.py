"""
Polar X-band sweeps to a 3-D Cartesian grid (CAPPI)
====================================================

Interpolates WegenerNet 3D X-band radar sweeps (L1b v1, polar coordinates:
elevation x azimuth x range) to constant-altitude levels on a regular UTM 33N
grid around the microwave radiometer. The interpolation is a wradlib CAPPI with
inverse-distance weighting of the nearest radar gates.

The gridder (gate and grid geometry, neighbour search) is built once and then
applied to every time step and variable.

    build_wrl_gridder       geometry + wradlib CAPPI interpolator
    wradlib_area_over_time  apply the gridder to a time window of one L1b file

References
----------
Heistermann, M., Jacobi, S., and Pfaff, T. (2013): Technical Note: An open source
library for processing weather radar data (wradlib), Hydrol. Earth Syst. Sci., 17,
863-871, https://doi.org/10.5194/hess-17-863-2013
"""

import numpy as np
import xarray as xr
import wradlib as wrl
from osgeo import osr

osr.UseExceptions()

# Radar site: longitude, latitude (deg) and antenna altitude (m a.s.l.)
RADAR_LONLAT_ALT = (15.9321, 46.8453, 630.0)
# The same site in projected coordinates (x, y, z in m) for each EPSG code used
RADAR_SITE_UTM = {
    32633: (571067.5198691811, 5188394.819884371, 630.0),   # UTM zone 33N
}


def build_wrl_gridder(
    l1b_nc,
    x_min, x_max, y_min, y_max, dx, dy,
    h_levels,
    *, epsg=32633, rep_time_index=0,
    nnearest=4, p=2.0,
    pol_dims=("elevation", "azimuth", "range"),
    site_lonlat=RADAR_LONLAT_ALT, site_utm=None,
):
    """Build a reusable wradlib CAPPI interpolator for one target grid.

    Parameters
    ----------
    l1b_nc : str or Path
        L1b sweep file; only its polar geometry (elevations, azimuths, ranges)
        is read.
    x_min, x_max, y_min, y_max : float
        Target grid edges in the projected CRS (m).
    dx, dy : float
        Target grid spacing (m).
    h_levels : sequence of float
        Target altitudes (m a.s.l.).
    epsg : int
        Projected CRS of the target grid (default UTM 33N).
    rep_time_index : int
        Time step whose polar geometry is used (the scan strategy is fixed).
    nnearest, p : int, float
        Number of neighbouring gates and power of the inverse-distance weights.
    pol_dims : tuple of str
        Names of the polar dimensions in the file, in (elevation, azimuth, range)
        order.
    site_lonlat : tuple
        Radar longitude, latitude (deg) and altitude (m a.s.l.).
    site_utm : tuple, optional
        Radar site in the projected CRS; if None it is taken from
        ``RADAR_SITE_UTM`` or projected with pyproj.

    Returns
    -------
    dict
        ``gridder`` (wradlib CAPPI object), polar and grid shapes, the target
        axes ``x``, ``y``, ``z`` and the polar dimension order.
    """
    ds = xr.open_dataset(l1b_nc, decode_cf=True).isel(time=0)

    rlon, rlat, rh = (float(c) for c in site_lonlat)
    site_ll = (rlon, rlat, rh)

    # polar geometry: elevation (deg), azimuth (deg), range (m)
    sel = (lambda da: da.isel(time=rep_time_index)) if "time" in ds.dims else (lambda da: da)
    elevs = sel(ds[pol_dims[0]]).values
    azims = sel(ds[pol_dims[1]]).values
    ranges = sel(ds[pol_dims[2]]).values
    polshape = (len(elevs), len(azims), len(ranges))

    # gate centres in the projected CRS, flattened in (elevation, azimuth, range) order;
    # the beam height includes the 4/3 effective Earth radius model of wradlib
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(int(epsg))
    polxyz = wrl.vpr.volcoords_from_polar(site_ll, elevs, azims, ranges, crs=crs)

    # radar site in projected metres (centre of the blind-spot mask of the CAPPI)
    if site_utm is not None:
        site_xyz = tuple(float(c) for c in site_utm)
    elif site_lonlat == RADAR_LONLAT_ALT and int(epsg) in RADAR_SITE_UTM:
        site_xyz = RADAR_SITE_UTM[int(epsg)]
    else:
        from pyproj import Transformer
        x0, y0 = Transformer.from_crs(4326, int(epsg), always_xy=True).transform(rlon, rlat)
        site_xyz = (float(x0), float(y0), rh)

    # target grid points (z = altitude a.s.l.)
    x = np.arange(x_min, x_max + 0.5 * dx, dx)
    y = np.arange(y_min, y_max + 0.5 * dy, dy)
    z = np.asarray(h_levels, float)
    gridxyz = wrl.util.gridaspoints(z, y, x)
    gridshape = (len(z), len(y), len(x))

    # CAPPI: points outside the lowest/highest elevation or beyond the maximum range are masked
    gridder = wrl.vpr.CAPPI(
        polxyz, gridxyz,
        maxrange=ranges.max(), minelev=elevs.min(), maxelev=elevs.max(),
        site=site_xyz,
        ipclass=wrl.ipol.Idw, nnearest=nnearest, p=p,
    )

    ds.close()
    return {"gridder": gridder, "polshape": polshape, "x": x, "y": y, "z": z,
            "gridshape": gridshape, "site_xyz": site_xyz, "pol_dims": tuple(pol_dims)}


def wradlib_area_over_time(
    l1b_nc, grid, *,
    begin_time=None, end_time=None,
    vars_to_get=("Z_h_corr", "zdr_corr", "kdp", "rhohv", "vradh", "wradh", "phidp"),
    qc_vars=("ste_prob", "rad_qcs_flag", "qcs_flag", "quality_flag"),
    chunks_time=None,
):
    """Interpolate the sweeps of one L1b file to the Cartesian grid.

    Gates flagged by any of the quality variables are set to NaN before the
    interpolation: ``ste_prob`` > 0.5 (second-trip echo probability) or any
    other flag > 0.

    Parameters
    ----------
    l1b_nc : str or Path
        L1b sweep file (one day).
    grid : dict
        Output of :func:`build_wrl_gridder`.
    begin_time, end_time : str or datetime-like, optional
        Time window (inclusive). All time steps if both are None.
    vars_to_get : tuple of str
        Radar variables to grid (those missing in the file are skipped).
    qc_vars : tuple of str
        Quality variables used to mask gates.
    chunks_time : int, optional
        Dask chunk size along time when opening the file.

    Returns
    -------
    xr.Dataset
        Variables with dimensions (time, h_msl_m, y_utm, x_utm).
    """
    open_kwargs = dict(decode_cf=True)
    if chunks_time:
        open_kwargs["chunks"] = {"time": chunks_time}
    ds_all = xr.open_dataset(l1b_nc, **open_kwargs)
    ds = ds_all if "time" not in ds_all.dims else (
        ds_all if (begin_time is None and end_time is None)
        else ds_all.sel(time=slice(begin_time, end_time))
    )
    has_time = "time" in ds.dims
    times = np.array(ds["time"].values) if has_time else np.array([np.datetime64("NaT")])

    Ne, Na, Nr = grid["polshape"]
    Nz, Ny, Nx = grid["gridshape"]
    pol_dims = grid["pol_dims"]
    gridder = grid["gridder"]
    n_src, nt = Ne * Na * Nr, times.size

    present_qc = [q for q in qc_vars if q in ds]
    present_vars = [v for v in vars_to_get if v in ds]
    if not present_vars:
        return xr.Dataset(coords={"time": times, "h_msl_m": grid["z"],
                                  "y_utm": grid["y"], "x_utm": grid["x"]})

    # Read one variable at a time over the whole window, in (time, elevation, azimuth,
    # range) order, flattened to (nt, n_src): every stored chunk is decompressed once.
    canon = (("time",) + tuple(pol_dims)) if has_time else tuple(pol_dims)

    def _load(name):
        return ds[name].transpose(*canon).values.reshape(nt, n_src)

    # gate mask from the quality variables
    qc = np.zeros((nt, n_src), bool)
    for q in present_qc:
        arr = _load(q)
        qc |= (arr > 0.5) if q == "ste_prob" else (arr > 0)

    out = {}
    for v in present_vars:
        values = _load(v)
        arrs = np.empty((nt, Nz, Ny, Nx))
        for ti in range(nt):
            pol = np.where(qc[ti], np.nan, values[ti])
            gridded = gridder(pol)
            arrs[ti] = np.ma.masked_invalid(gridded).filled(np.nan).reshape(Nz, Ny, Nx)
        out[v] = arrs
        del values

    ds_all.close()
    data_vars = {v: (("time", "h_msl_m", "y_utm", "x_utm"), out[v]) for v in present_vars}
    coords = {"time": times, "h_msl_m": grid["z"], "y_utm": grid["y"], "x_utm": grid["x"]}
    return xr.Dataset(data_vars, coords=coords)
