# LWP and IWV before rain onset (WegenerNet 3D Open-Air Laboratory)

Code for the paper

> Ghaemi, E., Kvas, A., Foelsche, U., and Kollias, P.: Regime-dependent evolution of water vapor, cloud liquid, and cloud base before rain onset in the southeastern Alpine forelands.

About four years (April 2021 – September 2025) of collocated observations of the WegenerNet 3D Open-Air Laboratory in southeast Austria are used: a microwave radiometer (liquid water path LWP, integrated water vapour IWV, temperature profiles), an infrared radiometer (cloud base height, cloud masks), a GNSS station (IWV), an X-band dual-polarisation radar and rain gauges. Rain events are split into convective and non-convective events with a multi-criteria radar scheme and into warm, cold and mixed rain by comparing the radar echo top with the −5 °C isotherm. The notebooks reproduce every figure of the paper.

## Repository layout

```
├── notebooks/
│   ├── 01_paper_figures.ipynb                 # all paper figures, f01 ... f13, fA02 ... fA10
│   └── 02_radar_convective_probability.ipynb  # radar gridding, convective/hail probability, echo top, fA01
├── src/                                       # functions used by the notebooks
├── data/                                      # inputs (see data/README.md)
├── figures/                                   # output: f01.png ... fA10.png (PNG, 300 dpi)
├── environment.yml
├── CITATION.cff
└── LICENSE
```

## Notebooks

**`02_radar_convective_probability.ipynb`** — X-band radar processing (Fig. 3 of the paper). Runs for two example events by default; set `RUN_ALL_EVENTS = True` to process all events.

1. polar L1b sweeps → 3-D Cartesian grid around the radiometer (wradlib CAPPI, inverse-distance weighting)
2. 0 °C and −5 °C isotherm heights from the microwave-radiometer temperature profile
3. quality control, convective probability (reflectivity, Z_DR column, K_DP, Doppler) and separate hail probability; event value = P95 over the grid, then P95 over onset ± 30 min
4. radar echo-top height (Z_h > 0 dBZ)
5. sensitivity of the classification to weights and threshold (Fig. A1)
6. example event: Z_h and Z_DR above the radiometer with the −5 °C isotherm

**`01_paper_figures.ipynb`** — loads all observations, detects and classifies the rain events, extracts the six hours before each rain onset and draws the figures in figure order:

| Figure | Content |
| --- | --- |
| f01 | study area, instruments, mean precipitation of all events |
| f02, f03 | flowcharts: rain type, convective probability |
| f04 | precipitation over the storm duration and gauge-grid maps |
| f05 | monthly LWP and IWV |
| f06 | LWP and IWV anomalies before rain onset |
| f07 | cloud base height before rain onset |
| f08 | cloud fraction before rain onset |
| f09 | joint evolution of ΔLWP and ΔIWV |
| f10–f13 | example events 8 June 2023 and 12 June 2024 |
| fA01 | sensitivity of the convective classification (notebook 02) |
| fA02 | ERA5 vs radiometer and GNSS |
| fA03 | absolute LWP and IWV before rain onset |
| fA04 | hour of day of the pre-onset time steps |
| fA05–fA10 | example events 19 August 2024, 16 May 2024, 16 March 2025 |

## How to run

```bash
conda env create -f environment.yml
conda activate lwp_iwv_before_rain
python -m ipykernel install --user --name lwp_iwv_before_rain
```

1. Put the WegenerNet data into `data/raw/` (or link existing folders), see `data/README.md`.
2. Notebook 02 makes the gridded radar files (`data/derived/radar_L1_cartesian/`) and the convective probability of the events. The results used for the manuscript are shipped in `data/processed/`, so notebook 01 runs without notebook 02 if the gridded radar files of the five example events are available.
3. Run notebook 01 from the `notebooks/` folder.

Run times on the test machine (a server with 252 GB of memory): notebook 01 about 25 min, notebook 02 (two examples) about 5 min. Notebook 01 reads four years of 2.5-min radiometer data and 1-s cloud-base data into memory, so use a machine with plenty of memory.

## Notes

* **Freezing level of the convective probability.** The melting layer is defined from the 0 °C level of the radiometer profile. The convective probabilities in `data/processed/summer_events_with_conv_prob_20260630.nc` were computed with a definition: lowest profile height with T ≥ 0 °C, i.e. the lowest profile level when the surface is above 0 °C; 3000 m when no profile was available. Notebook 02 compares both for the examples.

* **Fig. A1** in the manuscript uses criterion scores computed without quality control and with a 600 m melting layer (`data/derived/convective_components_202606/`).
* **Radar quality flag.** The gridding masks gates with the quality variables `ste_prob`, `rad_qcs_flag`, `qcs_flag` or `quality_flag`; the L1b files carry `radar_qcs_flag`, so no gate is removed before gridding (as for the gridded files used in the paper).
* `plot_lwp_boxen` / `plot_iwv_boxen` (`src/notebook_helpers.py`) call `seaborn.set_theme()`, which changes the matplotlib style of later figures; notebook 01 restores it after f05.
* Fonts: all figures use DejaVu Sans with the sizes in `src/figure_style.py` (tick labels 13 pt, axis labels 14 pt, panel titles 15 pt bold, legends and annotations 12 pt). `style_fonts(fig)` applies them to every figure just before it is saved, so all panels of a composite figure have the same letter size.
* GDAL/PROJ: notebook 02 sets `PROJ_DATA` to the PROJ database of the active Python environment before importing wradlib.

## Licence and citation

Code: MIT licence (`LICENSE`). Please cite the paper and this code (`CITATION.cff`). Data: see `data/README.md` for sources, licences and citations.
