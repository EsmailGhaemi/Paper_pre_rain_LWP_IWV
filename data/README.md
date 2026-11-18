# Data

The notebooks expect the folder layout below. Only the small files in `processed/` and `static/` are part of the repository; `raw/` and `derived/` must be filled locally (download the data, or link existing folders with `ln -s`).

```
data/
├── processed/   shipped: small derived files used by the notebooks
├── static/      shipped: figure parts that are not made by code (photo, map, flowcharts)
├── raw/         NOT shipped: WegenerNet data (download, see below)
└── derived/     NOT shipped: large files made by notebook 02
```

## `processed/` (in the repository)

| File | Content | Made by |
| --- | --- | --- |
| `all_events.nc` | 644 rain events at the superstation (station 156), 2021–2025: `start`, `end`, `duration` | notebook 01, Section 3 (`select_precip_timesteps` + `detect_events_vectorized`, 120-min gap); identical to the events detected there |
| `summer_events_with_conv_prob_20260630.nc` | the same events with the convective and hail probability (`prob_q95_q95`, `prob_hail`, `prob_q90_q90`, `prob_median_max`); NaN where no radar data were available | notebook 02, Section 4 (run of June 2026, earlier freezing-level definition, see notebook 02) |
| `filtered_events_with_conv_index.csv` | 390 warm-season events with a convective probability and event statistics (rain sum, mean intensity, mean 2-m temperature, ...) | notebook 01, Section 3 |
| `echo_top_height_filtered_202606.nc` | radar echo-top height (highest gate with Z_h > 0 dBZ) on the 200 m grid, radiometer ± 3 km, every 2.5 min of all gridded event windows | notebook 02, Section 5 |
| `era5_tclw_tcwv_radiometer.nc` | ERA5 total column cloud liquid water (`tclw`, kg m⁻²) and water vapour (`tcwv`, kg m⁻²), hourly, at the grid point of the radiometer (47.01 °N, 15.87 °E), subset of the WegenerNet 3D auxiliary ERA5 data cube | subset of the WEGN-3DO AUX ERA5 time series |

## `static/` (in the repository)

| File | Used in |
| --- | --- |
| `topography_map.png` | f01 (a): topography, radar coverage and stations |
| `instruments_photo.png` | f01 (b): instruments at the Raabtal superstation |
| `flowchart_rain_type.pdf` | f02 (LaTeX/TikZ) |
| `flowchart_convective_probability.pdf` | f03 (LaTeX/TikZ) |

## `raw/` (download, not in the repository)

| Folder | Content | Source |
| --- | --- | --- |
| `raw/WegenerNet_Transfer/L2a/v2.0-rc1/TSR/` | microwave radiometer (HATPRO) L2a v2.0-rc1: `TotalColumnLiquidWater`, `TotalColumnWaterVapor`, `TroposphericProfiles`, daily files 2021–2025 | WegenerNet 3D L2a v2.0-rc1, https://doi.org/10.5281/zenodo.21350494 |
| `raw/WegenerNet_Transfer/L2a/v2.0-rc1/WSN_GNSS/` | GNSS IWV (station 181) | same |
| `raw/WegenerNet_Transfer/L2a/v2.0-rc1/CSX/`, `.../CSR/` | infrared radiometer zenith cloud masks, narrow (9.6–11.5 µm) and wide (8–14 µm) band | same |
| `raw/WegenerNet_Transfer/HRR/SYN/` | infrared radiometer cloud base height, 1-s HRR files | same |
| `raw/radar_L1b/` | X-band radar L1b v1.0 polar sweeps, `WN3D_L1b_v1_PrecipRadar_Sweeps_YYYY-MM-DD.nc` (only needed to re-grid, notebook 02) | WegenerNet 3D L1b v1.0, https://doi.org/10.25364/WEGC/WPS3D-L1B-10 |
| `raw/radar_L2/` | X-band radar L2 v1.0 precipitation, `WN3D_L2_BD_v1_Precipitation_YYYY-MM-DD.nc` (event figures) | WegenerNet 3D L2 v1.0, https://doi.org/10.25364/WEGC/WPS3D-L2-10 |
| `raw/stations/` | climate station L2 v8.0 files `WN_L2_V8_BD_St<id>_2021-01-01d00h00m_2025-09-23d00h00m_UTC.csv`, stations 43, 44, 151, 156 | WegenerNet L2 v8.0, https://doi.org/10.25364/WEGC/WPS8.0:2025.1 |
| `raw/gauge_grid/` | gauge-grid precipitation `WN_L2_BD_v8_UTM_TF1_UTC_YYYY-MM-DD.nc` for the event days (f01, f04) | WegenerNet L2 v8.0 gridded data, WegenerNet Data Portal |

WegenerNet data are available at the WegenerNet Data Portal, https://wegenernet.uni-graz.at/ (registration required), under the terms of use given there.

## `derived/` (made by notebook 02, not in the repository)

| Folder | Content | Size |
| --- | --- | --- |
| `derived/radar_L1_cartesian/` | X-band variables on the 3-D grid (200 m, radiometer ± 3 km, 1–10 km a.s.l.), one file per event window `processedL1_YYYY-MM-DD_HH-MM.nc` (onset − 2 h … onset + 1.5 h) | ~40 GB |
| `derived/convective_components/` | criterion scores of the convective probability per grid column and time step, one file per event window (input of Fig. A1) | ~0.5 GB |
| `derived/convective_components_202606/` | the same from the run used for the submitted manuscript (no quality control before scoring, 600 m melting layer, earlier freezing-level definition) | ~0.5 GB |

## Data citations

- Kvas, A., Fuchsberger, J., Kirchengast, G., and Scheidl, D. (2024): WegenerNet 3D Open-Air Laboratory Level 1b Data, Version 1.0, Wegener Center for Climate and Global Change, University of Graz, https://doi.org/10.25364/WEGC/WPS3D-L1B-10
- Kvas, A., Fuchsberger, J., Kirchengast, G., and Scheidl, D. (2024): WegenerNet 3D Open-Air Laboratory Level 2 Data, Version 1.0, Wegener Center for Climate and Global Change, University of Graz, https://doi.org/10.25364/WEGC/WPS3D-L2-10
- Fuchsberger, J., Kirchengast, G., and Bichler, C. (2025): WegenerNet climate station network Level 2 data, Wegener Center for Climate and Global Change, University of Graz, https://doi.org/10.25364/WEGC/WPS8.0:2025.1
- WegenerNet 3D Level 2a GNSS and radiometer data, Version 2.0-rc1, https://doi.org/10.5281/zenodo.21350494
- Hersbach, H. et al. (2020): The ERA5 global reanalysis, Q. J. Roy. Meteor. Soc., 146, 1999–2049, https://doi.org/10.1002/qj.3803
