# Octopamine instructs head direction plasticity

Analysis code to reproduce the figures and statistics of

> Plitt, Turner-Evans, Co, Layden, Eddison, Ray, Jayaraman & Fisher. *Octopamine instructs head direction
> plasticity.* bioRxiv (2025). https://doi.org/10.64898/2025.12.11.693783

The repository contains an installable Python package, `el_hd_plasticity`, with the shared analysis code. It also
has one folder per experiment under `figures/`, each holding a script that regenerates that experiment's figures
and a notebook that runs the same analyses step by step.

## Installation

Python 3.11 or later is required. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yfisher-lab/plitt_turnerevans_oa_instructs_hd_plasticity.git
cd plitt_turnerevans_oa_instructs_hd_plasticity
uv sync --extra dev          # creates .venv from uv.lock
```

or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The dependencies include [SessionTools](https://github.com/yfisher-lab/SessionTools), the lab's two-photon
preprocessing package, which is installed from GitHub. The other notable dependencies are `navis` and
`neuprint-python` (connectomics), `statsmodels` and `pingouin` (statistics), and `imageio[ffmpeg]` (movies).

A `Dockerfile` and `docker-compose.yml` are also provided. The compose file mounts the data drive that the
default paths point to (see [Data](#data)).

## Data

The two-photon analyses read preprocessed SessionTools sessions laid out as

```
<basedir>/[<condition or genotype>/]<date>/<fly>/<session>/preprocess.pkl
```

with a `sessions.csv` in `<basedir>` that has one row per session and at least the columns `date`, `fly` and
`session` (see `el_hd_plasticity.two_photon.loading`). These preprocessed sessions are not part of the repository.
The default `BASEDIR` and `DEFAULT_FIGDIR` at the top of each analysis script point to the authors' data drive
(`/media/mplitt/SSD_storage/...`). Pass `--basedir` and `--figdir` to use other locations.

The RNA-seq count tables, differential expression tables and gene lists are in `data/rna_seq/`.

The connectomics analyses query [neuPrint](https://neuprint.janelia.org) (the hemibrain dataset by default). Set
the `NEUPRINT_APPLICATION_CREDENTIALS` environment variable to your neuPrint token.

## Usage

Each experiment folder has an analysis script and a matching notebook. Running the script regenerates every
figure and statistics table for that experiment:

```bash
python figures/oamb_ko/oamb_ko_analysis.py --basedir /path/to/sessions --figdir /path/to/output
python figures/oamb_ko/oamb_ko_analysis.py --skip-session-figures    # skip the slow per-fly heatmaps
```

The notebook imports the same script as a module and calls its `plot_*` functions one at a time. Each `plot_*`
function saves its figure(s) and statistics tables to the figure folder and returns them together with the
numbers behind them.

| Folder | Experiment | Script(s) | Notebook(s) |
|---|---|---|---|
| `epg_imaging_across_dark` | EPG imaging across closed loop, dark and closed loop again | `epg_across_dark_analysis.py` | `EPG_across_dark.ipynb`, `modeling_figs.ipynb` |
| `el_epg_coimaging` | EL and EPG co-imaging (R18B05-LexA > jGCaMP8s, R60D05-Gal4 > jRGECO1a), and the Turner-Evans dual-color dataset | `el_epg_coimaging_analysis.py`, `dte_coimaging_analysis.py` | `R18B05LexA_jGCaMP8s_R60D05Gal4_jRGECO1a.ipynb`, `DTE_data.ipynb` |
| `oa_da_imaging` | GRAB-DA and GRAB-OA imaging in the ER | `grabda_graboa.py` | `GRABDA_GRABOA.ipynb` |
| `tbh_rnai` | TBH (and VGluT) RNAi in the EL neurons | `r18b05gal4_r60d05lexa_analysis.py`, `el_split_analysis.py` | `R18B05Gal4_R60D05LexA.ipynb`, `EL_split.ipynb` |
| `oamb_ko` | Oamb knockout | `oamb_ko_analysis.py` | `global_ko.ipynb` |
| `octalpha2r_ko` | OctAlpha2R knockout and RNAi | `octalpha2r_ko_analysis.py`, `octalpha2r_rnai_analysis.py` | `global_ko.ipynb`, `octalpha2r_rnai.ipynb` |
| `opto_remapping` | Optogenetic remapping of the EPG bump by EL stimulation, with HcKCR1 controls and a dark-period variant | `r18b05lexa_r60d05gal4_analysis.py`, `r18b05lexa_r60d05gal4_hckcr1_analysis.py`, `r60d05_el_hckcr1_analysis.py`, `el_jgcamp7f_cschrimson_analysis.py`, `el_stim_epg_wdark_analysis.py` | one notebook per script |
| `epg_silencing` | EPG silencing with shibire-ts | `epg_silencing_analysis.py` | `epg_shiTS.ipynb` |
| `ex_vivo_gflamp` | *Ex vivo* GFlamp imaging of ER and EPG neurons during EL stimulation | `er_epg_gflamp_analysis.py` | `ER_EPG_Gflamp_R18B05LexA_CsChrimsontdT.ipynb` |
| `connectomics` | Hemibrain connectivity and synapse-distance analyses | (see below) | several notebooks |
| `rna_seq/el_exr2` | RNA-seq of EL and ExR2 neurons | `RNA-Seq Preprocessing *.Rmd` (R) | `RNA-Seq plotting.ipynb` |

The notebooks at the top of `figures/` and in `figures/scratch/` are for debugging and exploration. They are not
needed to reproduce the paper.

### Connectomics

Four scripts compute synapse distances in the hemibrain and pickle the results that the `figures/connectomics`
notebooks plot. They are slow and need neuPrint access:

```bash
python -m el_hd_plasticity.connectomics.run_synapse_distance --help          # EPG vs EL/ExR inputs on ER neurons
python -m el_hd_plasticity.connectomics.run_epg_instance_distance --help     # ER outputs onto different EPG columns
python -m el_hd_plasticity.connectomics.run_epg_postsynapse_distance --help  # ER inputs along EPG neurons
python -m el_hd_plasticity.connectomics.run_mb_synapse_distance --help       # mushroom body comparison
```

### Supplemental movies

`el_hd_plasticity.two_photon.session_movie` renders movies of real closed-loop sessions. Each frame shows the fly
in the virtual arena, EPG (or ER sensor) activity on the ellipsoid body, the session heatmap building up, and the
bump-cue offset histogram accumulating. `remapping_movie` does the same for the optogenetic remapping sessions,
and also renders an animated schematic of the pairing protocol. The analysis scripts wrap these renderers in
`make_*_movie` functions, for example `make_across_dark_movie` in `epg_across_dark_analysis.py` and
`make_remapping_session_movie` in the opto_remapping scripts. The movies are written to `<figdir>/movies/`. The
fly and rig schematics they use are in `figures/movie_assets/`, cut from the Figure 1 artwork by
`make_movie_assets.py`.

## Package layout

```
src/el_hd_plasticity/
├── two_photon/        imaging analyses
│   ├── loading.py           find and load sessions (SessionManager)
│   ├── session.py           ring timeseries: dF/F, heading, PVA, bump-cue offset
│   ├── stats.py             per-session / per-fly offset, PV magnitude and PVA velocity statistics
│   ├── bump_shape.py        phase nulling, Fourier interpolation, bump width (FWHM)
│   ├── plotting.py          shared heatmap and offset-histogram figures
│   ├── group_comparison.py  genotype comparisons in closed loop and dark (RNAi, knockouts)
│   ├── remapping.py         optogenetic remapping analyses
│   ├── dual_imaging.py      EL + EPG co-imaging
│   ├── grab_sensors.py      GRAB-DA / GRAB-OA
│   ├── ex_vivo.py           ex vivo GFlamp responses
│   ├── epg_across_dark.py, tbh_rnai.py, oa_receptor_ko.py, epg_silencing.py, shibirets.py
│   ├── session_movie.py, remapping_movie.py   supplemental movies
│   └── utilities.py
├── connectomics/      neuPrint access, EB coordinate system, connectivity, synapse distances
└── rna_seq/           RNA-seq plotting helpers
```

## Tests

```bash
pytest                 # tests/ — uses synthetic data, no sessions or neuPrint access needed
ruff check src tests figures
```

## Contact

Mark Plitt, mark.plitt@berkeley.edu
