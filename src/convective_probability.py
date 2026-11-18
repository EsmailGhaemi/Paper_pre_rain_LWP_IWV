"""
Convective and hail probability from X-band polarimetric radar
==============================================================

Fuzzy (0-1) scores of convective signatures on the 3-D Cartesian radar grid
(time, h_msl_m, y_utm, x_utm), combined into a convective probability per
column. The hail probability is computed separately and is NOT part of the
convective probability (convection can occur without hail).

Processing chain (Fig. 3 of the paper)

1. quality control of the gates (:func:`preprocess_radar_data`)
2. four convective criteria, each a column score in [0, 1]
   - reflectivity structure        (Steiner et al., 1995)
   - Z_DR column above the melting layer (Kumjian et al., 2014; Snyder et al., 2015)
   - K_DP (liquid water content)   (Bringi and Chandrasekar, 2001)
   - Doppler spectrum width and radial-velocity shear (Park et al., 2021)
3. weighted sum -> convective probability; separate hail probability
   (Aydin et al., 1986; Heinselman and Ryzhkov, 2006)

The melting layer (ML) is placed between the freezing level (0 degC isotherm from
the microwave-radiometer temperature profile) and 500 m above it.

Variable names follow the WegenerNet 3D L1 files: Z_h_corr [dBZ], zdr_corr [dB],
kdp [deg km-1], rhohv [-], vradh [m s-1], wradh [m s-1].

References
----------
Aydin, K., Seliga, T. A., and Balaji, V. (1986): Remote sensing of hail with a
    dual linear polarization radar, J. Clim. Appl. Meteorol., 25, 1475-1484.
Bringi, V. N. and Chandrasekar, V. (2001): Polarimetric Doppler Weather Radar,
    Cambridge University Press.
Heinselman, P. L. and Ryzhkov, A. V. (2006): Validation of polarimetric hail
    detection, Weather Forecast., 21, 839-850.
Kumjian, M. R. et al. (2014): The anatomy and physics of ZDR columns, J. Appl.
    Meteorol. Clim., 53, 1820-1843, https://doi.org/10.1175/JAMC-D-13-0354.1
Park, J. et al. (2021), Remote Sens., 13, 726, https://doi.org/10.3390/rs13040726
Snyder, J. C. et al. (2015): A ZDR column detection algorithm to examine
    convective storm updrafts, Weather Forecast., 30, 1819-1844,
    https://doi.org/10.1175/WAF-D-15-0068.1
Steiner, M., Houze, R. A., and Yuter, S. E. (1995): Climatological
    characterization of three-dimensional storm structure from operational radar
    and rain gauge data, J. Appl. Meteorol., 34, 1978-2007.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

# Default weights of the four convective criteria (sum = 1)
DEFAULT_WEIGHTS = {
    "reflectivity": 0.35,   # most validated criterion (Steiner et al., 1995)
    "zdr_column": 0.30,     # updraft signature (Snyder et al., 2015)
    "kdp": 0.25,            # liquid water content
    "doppler": 0.10,        # turbulence; fewest validation studies
}
DEFAULT_FREEZING_LEVEL_M = 3000.0   # used when no radiometer profile is available [m a.s.l.]


# =============================================================================
# 1. Quality control
# =============================================================================
def preprocess_radar_data(
    ds: xr.Dataset,
    zh_min_threshold: float = 5.0,
    zh_noise_floor: float = -20.0,
    rhohv_min: float = 0.70,
    rhohv_precipitation_min: float = 0.80,
    zdr_valid_range: Tuple[float, float] = (-2.0, 6.0),
    kdp_valid_range: Tuple[float, float] = (-0.5, 10.0),
    vrad_max_magnitude: float = 50.0,
) -> xr.Dataset:
    """Remove noise, clutter and non-meteorological echoes.

    A gate is kept if Z_h > ``zh_noise_floor``, rho_hv >= ``rhohv_min``, Z_DR and
    K_DP are inside their valid ranges and |v_r| <= ``vrad_max_magnitude``.
    Gates with |v_r| < 0.5 m s-1 and rho_hv < 0.90 (stationary, low correlation:
    ground clutter) are removed.

    Parameters
    ----------
    ds : xr.Dataset
        Radar variables Z_h_corr, zdr_corr, kdp, rhohv and optionally vradh, wradh.
    zh_min_threshold : float
        Minimum Z_h of precipitation [dBZ]; only used for ``precip_quality_mask``.
    zh_noise_floor : float
        Z_h below this value is noise [dBZ].
    rhohv_min : float
        rho_hv below this value is clutter or noise.
    rhohv_precipitation_min : float
        Minimum rho_hv of precipitation; only used for ``precip_quality_mask``.
    zdr_valid_range, kdp_valid_range : tuple of float
        Physically valid Z_DR [dB] and K_DP [deg km-1] ranges.
    vrad_max_magnitude : float
        Maximum valid |radial velocity| [m s-1].

    Returns
    -------
    xr.Dataset
        Masked radar variables (NaN where the gate fails the checks), plus the
        flags ``quality_mask`` and ``precip_quality_mask`` (1 = valid).
    """
    ds_filtered = ds.copy(deep=True)

    # basic quality masks
    zh_valid = ds["Z_h_corr"] > zh_noise_floor
    rhohv_valid = ds["rhohv"] >= rhohv_min

    # precipitation echoes (stored as a flag only, not applied to the data)
    precipitation_mask = (ds["rhohv"] >= rhohv_precipitation_min) & (ds["Z_h_corr"] >= zh_min_threshold)

    # physically valid ranges
    zdr_valid = (ds["zdr_corr"] >= zdr_valid_range[0]) & (ds["zdr_corr"] <= zdr_valid_range[1])
    kdp_valid = (ds["kdp"] >= kdp_valid_range[0]) & (ds["kdp"] <= kdp_valid_range[1])

    # Doppler checks with |v_r| (the sign only gives the direction)
    if "vradh" in ds:
        vrad_valid = np.abs(ds["vradh"]) <= vrad_max_magnitude
        stationary = np.abs(ds["vradh"]) < 0.5                  # [m s-1]
        clutter_mask = stationary & (ds["rhohv"] < 0.90)
    else:
        vrad_valid = xr.ones_like(ds["Z_h_corr"], dtype=bool)
        clutter_mask = xr.zeros_like(ds["Z_h_corr"], dtype=bool)

    quality_mask = zh_valid & rhohv_valid & zdr_valid & kdp_valid & vrad_valid & ~clutter_mask
    precip_quality_mask = quality_mask & precipitation_mask

    for var in ("Z_h_corr", "zdr_corr", "kdp", "rhohv", "vradh", "wradh"):
        if var in ds:
            ds_filtered[var] = ds[var].where(quality_mask, np.nan)

    ds_filtered["quality_mask"] = quality_mask.astype(np.int8)
    ds_filtered["precip_quality_mask"] = precip_quality_mask.astype(np.int8)
    return ds_filtered


# =============================================================================
# 2. Convective criteria (each returns a column score in [0, 1])
# =============================================================================
def _score_reflectivity(zh, heights, freezing_height, zh_convective_core_threshold,
                        zh_background_diff_threshold, zh_convective_threshold):
    """Tiered reflectivity score; the first matching condition sets the gate score.

    1.0  convective core, Z_h > 45 dBZ
    0.8  peak: Z_h exceeds the 5 x 5 gate background mean by > 8 dB
    0.6  Z_h > 40 dBZ
    0.4  Z_h > 30 dBZ somewhere more than 1 km above the freezing level (elevated echo top)
    else (Z_h - 20) / 25, limited to [0, 0.3]
    The column score is the maximum over height (Steiner et al., 1995).
    """
    convective_cores = (zh > zh_convective_core_threshold).astype(float)

    zh_background = zh.rolling(x_utm=5, y_utm=5, center=True, min_periods=1).mean()
    zh_diff = zh - zh_background

    zh_above_freezing = zh.where(heights > (freezing_height + 1000))
    echo_top_criterion = (zh_above_freezing > 30.0).any(dim="h_msl_m").astype(float)

    score = xr.where(
        convective_cores > 0, 1.0,
        xr.where(
            zh_diff > zh_background_diff_threshold, 0.8,
            xr.where(
                zh > zh_convective_threshold, 0.6,
                xr.where(echo_top_criterion > 0, 0.4, ((zh - 20) / 25).clip(0, 0.3)),
            ),
        ),
    )
    return score.clip(0, 1).fillna(0).max(dim="h_msl_m", skipna=True)


def _score_zdr_column(zdr, heights, ml_bottom, ml_top, zdr_column_threshold):
    """Z_DR column (updraft) score.

    Column strength from the maximum Z_DR above the melting-layer top:
    >= 1 dB -> 0.5 ... 1, 0.5-1 dB -> 0 ... 0.3 (Snyder et al., 2015).
    Bonus 0.1 / 0.2 if two / three or more levels above the ML have Z_DR >= 1 dB.
    Size sorting below the ML: mean Z_DR > 1.5 dB -> 0 ... 0.3.
    Score = 0.6 * strength + 0.25 * sorting + bonus, limited to [0, 1].
    """
    zdr_above_ml = zdr.where(heights > ml_top)
    zdr_below_ml = zdr.where(heights < ml_bottom)

    max_zdr_above = zdr_above_ml.max(dim="h_msl_m", skipna=True)
    column_strength = xr.where(
        max_zdr_above >= zdr_column_threshold,
        ((max_zdr_above - zdr_column_threshold) / 2.0 + 0.5).clip(0.5, 1),
        xr.where(max_zdr_above >= 0.5, ((max_zdr_above - 0.5) / 0.5 * 0.3), 0),
    )

    # vertical extent: number of levels above the ML with Z_DR >= threshold
    n_layers = (zdr >= zdr_column_threshold).where(heights > ml_top).sum(dim="h_msl_m", skipna=True)
    extent_bonus = xr.where(n_layers >= 3, 0.2, xr.where(n_layers >= 2, 0.1, 0))

    # size sorting of large drops below the ML
    mean_zdr_below = zdr_below_ml.mean(dim="h_msl_m", skipna=True)
    sorting = xr.where(mean_zdr_below > 1.5, ((mean_zdr_below - 1.5) / 2.0).clip(0, 0.3), 0)

    return (0.6 * column_strength + 0.25 * sorting + extent_bonus).clip(0, 1).fillna(0)


def _score_kdp(kdp, heights, ml_bottom, kdp_threshold, kdp_convective_threshold):
    """K_DP score: column maximum > 1.5 deg km-1 -> 1; 0.3-1.5 deg km-1 -> 0 ... 0.8.

    Plus 0.2 (limited to 1) if the mean K_DP below the ML exceeds 2 x 0.3 deg km-1
    (Bringi and Chandrasekar, 2001).
    """
    max_kdp = kdp.max(dim="h_msl_m", skipna=True)
    mean_kdp_below = kdp.where(heights < ml_bottom).mean(dim="h_msl_m", skipna=True)

    score = xr.where(
        max_kdp > kdp_convective_threshold, 1.0,
        xr.where(
            max_kdp > kdp_threshold,
            ((max_kdp - kdp_threshold) / (kdp_convective_threshold - kdp_threshold)).clip(0, 0.8),
            0,
        ),
    ).clip(0, 1).fillna(0)
    return xr.where(mean_kdp_below > kdp_threshold * 2, (score + 0.2).clip(0, 1), score)


def _score_doppler(ds, quality_mask, zh, use_doppler, spectrum_width_convective_threshold,
                   spectrum_width_moderate_threshold, vrad_shear_threshold, like):
    """Turbulence score from the spectrum width and the horizontal shear of |v_r|.

    Spectrum width: column max |w_r| (gates with Z_h > 10 dBZ) > 4 m s-1 -> 1,
    2.5-4 m s-1 -> 0 ... 1 (Park et al., 2021).
    Shear: column max of the 3 x 3 gate standard deviation of |v_r| > 10 m s-1 -> 1,
    otherwise linear, limited to 0.5.
    Both available: 0.7 * width + 0.3 * shear. Shear only: 0.8 above 10 m s-1.
    """
    def _shear_std():
        vrad_abs = np.abs(ds["vradh"]).where(quality_mask)
        vrad_std = vrad_abs.rolling(x_utm=3, y_utm=3, center=True, min_periods=2).std()
        return vrad_std.max(dim="h_msl_m", skipna=True)

    if use_doppler and "wradh" in ds:
        wrad_abs = np.abs(ds["wradh"]).where(quality_mask & (zh > 10))
        max_width = wrad_abs.max(dim="h_msl_m", skipna=True)
        score_width = xr.where(
            max_width > spectrum_width_convective_threshold, 1.0,
            xr.where(
                max_width > spectrum_width_moderate_threshold,
                (max_width - spectrum_width_moderate_threshold)
                / (spectrum_width_convective_threshold - spectrum_width_moderate_threshold),
                0,
            ),
        ).clip(0, 1).fillna(0)
        if "vradh" not in ds:
            return score_width
        max_std = _shear_std()
        score_shear = xr.where(max_std > vrad_shear_threshold, 1.0,
                               (max_std / vrad_shear_threshold).clip(0, 0.5)).fillna(0)
        return (0.7 * score_width + 0.3 * score_shear).clip(0, 1)

    if use_doppler and "vradh" in ds:
        max_std = _shear_std()
        return xr.where(max_std > vrad_shear_threshold, 0.8,
                        (max_std / vrad_shear_threshold).clip(0, 0.5)).fillna(0)

    return xr.zeros_like(like)


def _score_hail(zh, zdr, rhohv, hail_zh_min, hail_zh_threshold, hail_zdr_min, hail_zdr_max,
                hail_rhohv_threshold):
    """Hail signature: geometric mean of three gate scores, maximum over height.

    Z_h: required > 45 dBZ, score 1 at 50 dBZ; Z_DR in [-0.5, 1.5] dB:
    1 - |Z_DR| / 1.5; rho_hv in [0.80, 0.92): 1 - (rho_hv - 0.80) / 0.12
    (Aydin et al., 1986; Heinselman and Ryzhkov, 2006).
    """
    s_zh = xr.where(
        zh > hail_zh_threshold, 1.0,
        xr.where(zh > hail_zh_min, (zh - hail_zh_min) / (hail_zh_threshold - hail_zh_min), 0),
    ).clip(0, 1)
    s_zdr = xr.where((zdr >= hail_zdr_min) & (zdr <= hail_zdr_max),
                     1.0 - (np.abs(zdr) / hail_zdr_max), 0).clip(0, 1)
    s_rhohv = xr.where((rhohv >= 0.80) & (rhohv < hail_rhohv_threshold),
                       1.0 - ((rhohv - 0.80) / (hail_rhohv_threshold - 0.80)), 0).clip(0, 1)
    combined = xr.where(zh > hail_zh_min, (s_zh * s_zdr * s_rhohv) ** (1 / 3), 0)
    return combined.max(dim="h_msl_m", skipna=True).fillna(0)


def _normalised_weights(weights, ds, use_doppler):
    """Copy of the weights; the Doppler weight is redistributed if Doppler data are missing."""
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if not use_doppler or ("wradh" not in ds and "vradh" not in ds):
        weights["doppler"] = 0.0
        remaining = sum(v for k, v in weights.items() if k != "doppler")
        if remaining > 0:
            for k in weights:
                if k != "doppler":
                    weights[k] = weights[k] / remaining
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


# =============================================================================
# 3. Convective and hail probability of one time step
# =============================================================================
def calculate_convective_probability(
    ds: xr.Dataset,
    freezing_height: float,
    melting_layer_thickness: float = 500.0,
    weights: Optional[Dict[str, float]] = None,
    zh_convective_threshold: float = 40.0,
    zh_convective_core_threshold: float = 45.0,
    zh_background_diff_threshold: float = 8.0,
    zdr_column_threshold: float = 1.0,
    kdp_threshold: float = 0.3,
    kdp_convective_threshold: float = 1.5,
    rhohv_quality_threshold: float = 0.85,
    hail_zh_threshold: float = 50.0,
    hail_zh_min: float = 45.0,
    hail_zdr_max: float = 1.5,
    hail_zdr_min: float = -0.5,
    hail_rhohv_threshold: float = 0.92,
    use_doppler: bool = True,
    spectrum_width_convective_threshold: float = 4.0,
    spectrum_width_moderate_threshold: float = 2.5,
    vrad_shear_threshold: float = 10.0,
) -> xr.Dataset:
    """Convective and hail probability of every radar column at one time step.

    Parameters
    ----------
    ds : xr.Dataset
        Radar variables with dimensions (h_msl_m, y_utm, x_utm).
    freezing_height : float
        Height of the 0 degC isotherm [m a.s.l.].
    melting_layer_thickness : float
        The melting layer extends from the freezing level to this distance above
        it [m].
    weights : dict, optional
        Weights of 'reflectivity', 'zdr_column', 'kdp', 'doppler'; normalised to
        sum 1. Default :data:`DEFAULT_WEIGHTS`.
    rhohv_quality_threshold : float
        Gates with rho_hv below this value are not scored (hail uses all gates).
    other parameters
        Thresholds of the criteria, see the ``_score_*`` functions.

    Returns
    -------
    xr.Dataset
        ``convective_probability`` and ``hail_probability`` (y_utm, x_utm) in
        [0, 1], and the four criterion scores ``prob_reflectivity``,
        ``prob_zdr_column``, ``prob_kdp``, ``prob_doppler``.
    """
    weights = _normalised_weights(weights, ds, use_doppler)

    heights = ds["h_msl_m"]
    ml_bottom = freezing_height
    ml_top = freezing_height + melting_layer_thickness

    # gates used for the convective criteria
    quality_mask = ds["rhohv"] >= rhohv_quality_threshold
    zh = ds["Z_h_corr"].where(quality_mask)
    zdr = ds["zdr_corr"].where(quality_mask)
    kdp = ds["kdp"].where(quality_mask)

    p_ref = _score_reflectivity(zh, heights, freezing_height, zh_convective_core_threshold,
                                zh_background_diff_threshold, zh_convective_threshold)
    p_zdr = _score_zdr_column(zdr, heights, ml_bottom, ml_top, zdr_column_threshold)
    p_kdp = _score_kdp(kdp, heights, ml_bottom, kdp_threshold, kdp_convective_threshold)
    p_dop = _score_doppler(ds, quality_mask, zh, use_doppler, spectrum_width_convective_threshold,
                           spectrum_width_moderate_threshold, vrad_shear_threshold, like=p_ref)
    p_hail = _score_hail(zh, zdr, ds["rhohv"], hail_zh_min, hail_zh_threshold, hail_zdr_min,
                         hail_zdr_max, hail_rhohv_threshold)

    convective_probability = (weights["reflectivity"] * p_ref + weights["zdr_column"] * p_zdr
                              + weights["kdp"] * p_kdp + weights["doppler"] * p_dop).clip(0, 1)

    out = xr.Dataset({
        "convective_probability": convective_probability,
        "hail_probability": p_hail,
        "prob_reflectivity": p_ref,
        "prob_zdr_column": p_zdr,
        "prob_kdp": p_kdp,
        "prob_doppler": p_dop,
    })
    out["convective_probability"].attrs = {
        "long_name": "convective precipitation probability (hail not included)",
        "units": "1",
        "freezing_height_m": freezing_height,
        "weights": str(weights),
    }
    out["hail_probability"].attrs = {"long_name": "hail probability (separate from convection)", "units": "1"}
    return out


def calculate_convective_probability_timeseries(
    ds: xr.Dataset,
    freezing_heights: xr.DataArray,
    preprocess: bool = True,
    **kwargs,
) -> xr.Dataset:
    """Apply :func:`calculate_convective_probability` to every time step.

    Parameters
    ----------
    ds : xr.Dataset
        Radar variables with dimensions (time, h_msl_m, y_utm, x_utm).
    freezing_heights : xr.DataArray
        Freezing level [m a.s.l.] per time step (nearest time is used) or a scalar.
        NaN -> ``DEFAULT_FREEZING_LEVEL_M``.
    preprocess : bool
        Apply :func:`preprocess_radar_data` first.
    **kwargs
        Passed to :func:`calculate_convective_probability`.

    Returns
    -------
    xr.Dataset
        Output of :func:`calculate_convective_probability` with a time dimension.
    """
    results = []
    for t in ds.time.values:
        ds_t = ds.sel(time=t)
        if "time" in freezing_heights.dims:
            fz_height = float(freezing_heights.sel(time=t, method="nearest").values)
        else:
            fz_height = float(freezing_heights.values)
        if np.isnan(fz_height):
            fz_height = DEFAULT_FREEZING_LEVEL_M
        if preprocess:
            ds_t = preprocess_radar_data(ds_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN columns
            result_t = calculate_convective_probability(ds_t, freezing_height=fz_height, **kwargs)
        results.append(result_t.expand_dims(time=[t]))
    return xr.concat(results, dim="time")


# =============================================================================
# 4. Freezing level from the microwave radiometer and per-event statistics
# =============================================================================
def freezing_level_at_radar_times(t_profile, t_flag, radar_times, find_freezing_level,
                                  method="0C", window=None, fallback_window=np.timedelta64(3, "h")):
    """Freezing level [m a.s.l.] from the radiometer profile, on the radar time steps.

    Parameters
    ----------
    t_profile, t_flag : xr.DataArray
        Temperature profile [K] and its quality flag, dimensions (time, h_msl).
    radar_times : array of datetime64
        Time steps of the radar data.
    find_freezing_level : callable
        ``analysis.find_freezing_level`` (lowest crossing, linear interpolation).
    method : {'0C', 'lowest_warm_level'}
        '0C': lowest height where the profile crosses 273.15 K (used here).
        'lowest_warm_level': lowest height with T >= 273.15 K. This was used to make
        ``summer_events_with_conv_prob_20260630.nc``; it gives the lowest profile
        level (320 m) whenever the surface is above 0 degC, i.e. NOT the 0 degC level.
        Kept only to compare with that file.
    window : tuple of datetime64, optional
        Time window of the profiles (default: first and last radar time step). The
        paper uses rain onset +/- 30 min.
    fallback_window : timedelta64
        If the profile is missing during ``window``, the valid profile nearest to the
        window centre within +/- ``fallback_window`` is used; otherwise
        ``DEFAULT_FREEZING_LEVEL_M``.

    Returns
    -------
    xr.DataArray
        Freezing level with dimension time (radar time steps).
    """
    radar_times = np.asarray(radar_times)
    t0, t1 = window if window is not None else (radar_times.min(), radar_times.max())
    t0, t1 = np.datetime64(t0), np.datetime64(t1)
    centre = t0 + (t1 - t0) / 2
    window = t_profile.sel(time=slice(t0, t1))
    if window.time.size == 0 or not bool(window.notnull().any()):
        # nearest profile with data within +/- fallback_window of the window centre
        near = t_profile.sel(time=slice(centre - fallback_window, centre + fallback_window))
        valid_times = near.time[near.notnull().any("h_msl").values]
        if valid_times.size == 0:
            return xr.DataArray(np.full(radar_times.size, DEFAULT_FREEZING_LEVEL_M),
                                coords={"time": radar_times}, dims="time")
        nearest = valid_times.values[np.abs(valid_times.values - centre).argmin()]
        window = t_profile.sel(time=[nearest])

    if method == "0C":
        flag = t_flag.sel(time=window.time)
        fl, _, _ = find_freezing_level(window, flag, threshold=273.15)
    elif method == "lowest_warm_level":
        warm = window.where(window >= 273.15)
        fl = warm.coords["h_msl"].where(warm.notnull()).min(dim="h_msl")
    else:
        raise ValueError("method must be '0C' or 'lowest_warm_level'")

    fl = fl.fillna(DEFAULT_FREEZING_LEVEL_M)
    if fl.get_index("time").has_duplicates:
        fl = fl.groupby("time").mean()
    return fl.interp(time=radar_times, method="nearest", kwargs={"fill_value": "extrapolate"})


def event_statistics(result: xr.Dataset) -> Dict[str, float]:
    """Event values of the convective and hail probability (as in the paper).

    prob_q95_q95 : 95th percentile over the grid, then 95th percentile over time
                   (event is convective if >= 0.5)
    prob_hail    : the same statistic of the hail probability (hail if >= 0.5)
    prob_q90_q90, prob_median_max : alternative statistics
    """
    cp = result["convective_probability"]
    return {
        "prob_q95_q95": cp.quantile(0.95, dim=["x_utm", "y_utm"]).quantile(0.95).item(),
        "prob_q90_q90": cp.quantile(0.90, dim=["x_utm", "y_utm"]).quantile(0.90).item(),
        "prob_median_max": cp.median(dim=["x_utm", "y_utm"]).max().item(),
        "prob_hail": result["hail_probability"].quantile(0.95, dim=["x_utm", "y_utm"]).quantile(0.95).item(),
    }
