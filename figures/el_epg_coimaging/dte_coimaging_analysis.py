"""EL and EPG co-imaging, Turner-Evans dataset (dual-color trials, closed loop only): analyses and figures.

Run ``python dte_coimaging_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``DTE_data.ipynb`` does) and call the pieces one at a time. It follows ``el_epg_coimaging_analysis.py`` (which
holds the shared plotting helpers and the channel convention) but works on the pickled trial dictionaries
instead of a session table: ``data[fly_path][trial_name]`` is a trial dict, and ``channel_lookup[fly_path]``
maps 'EPG' and 'EL' to the keys of its fluorescence arrays. All trials are closed loop.
"""

import argparse
import pathlib
import pickle

import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_epg_coimaging_analysis import (
    CHANNEL_COLORS,
    CHANNELS,
    CL_COLOR,
    DH_BINS,
    _despine,
    _jitter,
    _mean_sem,
    _wilcoxon_vs,
    pva_diff_vs_speed_plot,
    save_tables,
)

DATA_PATH = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/DTE_EL_EPG_dual_color/PreProcDat')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/DTE_EPG_EL_coimaging')

TURN_DH_RANGE = (100 / 180 * np.pi, 2 * np.pi)       # turning speeds (rad/s) counted as fast turns: 100-360 deg/s
ONLY_CLOSED_LOOP = ((1, CL_COLOR, 'closed loop'),)


# ----------------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """Load the pickled trial dictionaries and the per-fly channel lookup.

    Returns:
        Tuple of (data, channel_lookup). In flies of the G60D05RSS27 line the EPG is imaged in the green channel
        (``DF_G``) and the EL in the red channel; in all others the opposite.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    channel_lookup = {fly: ({'EPG': 'DF_G', 'EL': 'DF_R'} if 'G60D05RSS27' in fly else {'EPG': 'DF_R', 'EL': 'DF_G'})
                      for fly in data}
    return data, channel_lookup


def fly_name(fly_path):
    """Short fly name of a (Windows) fly path."""
    return fly_path.split('\\')[-1]


def iter_timeseries(data, channel_lookup, **ts_kwargs):
    """Yield ``(fly_path, trial_name, ts)`` for every trial."""
    for fly, trials in data.items():
        for trial_name, trial_dict in trials.items():
            yield fly, trial_name, nmp.session.GetTS_DTE(trial_dict, channel_lookup[fly], **ts_kwargs)


# ----------------------------------------------------------------------------
# Offset statistics
# ----------------------------------------------------------------------------
def plot_abs_offset_diff(figdir, data, channel_lookup):
    """Absolute offset difference between EPG and EL per fly (closed loop), against chance (pi/2).

    Returns:
        Tuple of (figure, per-fly table, Wilcoxon test against chance).
    """
    offset_stats = nmp.dual_imaging.offset_stats_DTE(data, channel_lookup)
    unique = nmp.dual_imaging.offset_stats_unique(offset_stats, cl_only=True)
    unique = unique[unique['cl'] == 1]

    df = unique.copy()
    df['x'] = df['fly_id'].map(_jitter(df['fly_id'], width=.02))
    fig, ax = plt.subplots(figsize=[1, 4])
    ax.scatter(df['x'], df['abs_offset_diff'], color='black', s=100, alpha=.4)
    ax.plot([-.05, .05], df['abs_offset_diff'].mean() * np.ones(2), color='black', linewidth=8)
    ax.axhline(np.pi / 2, color='black', linestyle='--', linewidth=1, alpha=.5, zorder=-1)
    ax.set_xlim([-.15, .15])
    ax.set_ylim([0, np.pi])
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                  labels=['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.set_xticks([0], labels=[r'$| \Delta$ offset $|$'])
    ax.set_ylabel(r'$| \Delta$ offset $|$')
    _despine(ax, bottom=True)
    fig.savefig(pathlib.Path(figdir) / 'PVA_diff_sns.pdf', bbox_inches='tight')

    test = _wilcoxon_vs(unique['abs_offset_diff'], np.pi / 2)
    test['dof'] = len(unique) - 1
    return fig, unique, test


# ----------------------------------------------------------------------------
# Activity during fast turns
# ----------------------------------------------------------------------------
def plot_turn_profiles(figdir, data, channel_lookup, dh_range=TURN_DH_RANGE):
    """Turn-triggered EPG and EL bump profiles over all trials."""
    timeseries = (ts for _, _, ts in iter_timeseries(data, channel_lookup, dh_sigma=.3))
    ccw, cw = nmp.dual_imaging.turn_triggered_profiles(timeseries, *dh_range)
    fig, pva = nmp.dual_imaging.plot_turn_profiles(ccw, cw, colors=[CHANNEL_COLORS[c] for c in CHANNELS])
    fig.savefig(pathlib.Path(figdir) / 'EPG_centered_turns.pdf', format='pdf', bbox_inches='tight')
    return fig, {name: {c: float(v) for c, v in zip(CHANNELS, angles, strict=True)} for name, angles in pva.items()}


# ----------------------------------------------------------------------------
# PV magnitude and PVA difference versus rotation speed
# ----------------------------------------------------------------------------
def compute_binned_stats(data, channel_lookup, dh_bins=DH_BINS):
    """Per-fly PV magnitude of both channels and signed PVA difference binned by heading velocity.

    Returns:
        Dict with 'rho1_dig', 'rho2_dig', 'pva_diff' -> DataFrame (fly_id, cl, mean array per bin) averaged over
        the trials of each fly.
    """
    stats_df = nmp.dual_imaging.rho_stats_DTE(data, channel_lookup, dh_bins, abs=False)
    return {col: stats_df.groupby(['fly_id', 'cl'])[col].apply(lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()
            for col in ('rho1_dig', 'rho2_dig', 'pva_diff')}


def plot_rho_vs_speed(figdir, grouped, dh_bins=DH_BINS):
    """PV magnitude of EPG and EL against rotation speed (closed loop)."""
    centers = dh_bins + np.diff(dh_bins)[0] / 2
    fig, ax = plt.subplots(figsize=[5, 5])
    for ch, name in enumerate(CHANNELS):
        col = f'rho{ch + 1}_dig'
        mu, sem = _mean_sem(np.stack(grouped[col].loc[grouped[col]['cl'] == 1, col]))
        ax.plot(centers, mu, color=CHANNEL_COLORS[name], label=name)
        ax.fill_between(centers, mu - sem, mu + sem, color=CHANNEL_COLORS[name], alpha=.4)
    ax.set_title('Closed loop')
    ax.set_ylabel('PV Magnitude')
    ax.set_xlabel('rot. speed (rad/s)')
    _despine(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / 'PV_magnitude_summary.pdf', format='pdf', bbox_inches='tight')
    return fig


def plot_pva_diff_vs_speed(figdir, grouped, dh_bins=DH_BINS):
    """Signed PVA difference (EL - EPG) against rotation speed (closed loop), with the Spearman correlation."""
    fig, stats = pva_diff_vs_speed_plot(grouped['pva_diff'], dh_bins, conditions=ONLY_CLOSED_LOOP,
                                        ylim=(-np.pi / 16, 3 * np.pi / 32))
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / 'PVA_diff_v_speed.pdf', format='pdf', bbox_inches='tight')
    return fig, stats


# ----------------------------------------------------------------------------
# Single-trial figures
# ----------------------------------------------------------------------------
def plot_trial_heatmaps(figdir, data, channel_lookup, close=True):
    """EPG / EL heatmaps with offset histograms of every trial, in ``<figdir>/<fly>/<fly>_trial<i>_heatmaps.pdf``.

    Returns:
        Number of trials plotted.
    """
    n = 0
    for fly, trials in data.items():
        name = fly_name(fly)
        folder = pathlib.Path(figdir) / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, (trial_name, trial_dict) in enumerate(trials.items()):
            ts = nmp.session.GetTS_DTE(trial_dict, channel_lookup[fly])
            fig, _ = nmp.dual_imaging.plot_sess_heatmaps_w_hist(ts, name, trial_name, vmin=0, vmax=1)
            fig.savefig(folder / f'{name}_trial{i}_heatmaps.pdf', bbox_inches='tight')
            n += 1
            if close:
                plt.close('all')
    return n


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def main(data_path=DATA_PATH, figdir=DEFAULT_FIGDIR, trial_figures=True):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    figdir = pathlib.Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    data, channel_lookup = load_data(data_path)
    print(f'{len(data)} flies, {sum(len(trials) for trials in data.values())} trials')
    tables = {}

    _, unique, test = plot_abs_offset_diff(figdir, data, channel_lookup)
    tables['abs_offset_diff_vs_chance'] = pd.DataFrame([test])
    tables['abs_offset_diff_per_fly'] = unique

    _, pva = plot_turn_profiles(figdir, data, channel_lookup)
    tables['turn_profile_pva'] = pd.DataFrame(pva).rename_axis('channel').reset_index()

    grouped = compute_binned_stats(data, channel_lookup)
    plot_rho_vs_speed(figdir, grouped)
    _, spearman = plot_pva_diff_vs_speed(figdir, grouped)
    tables['pva_diff_vs_speed_spearman'] = pd.DataFrame(spearman).T.rename_axis('condition').reset_index()

    if trial_figures:
        print(f'{plot_trial_heatmaps(figdir, data, channel_lookup)} trial figures')

    save_tables(tables, figdir)
    for name, table in tables.items():
        print(f'\n=== {name} ===\n{table}')
    return tables


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-path', type=pathlib.Path, default=DATA_PATH, help='pickled trial dictionaries')
    parser.add_argument('--figdir', type=pathlib.Path, default=DEFAULT_FIGDIR, help='where figures and tables are saved')
    parser.add_argument('--skip-trial-figures', action='store_true', help='skip the per-trial heatmaps (slow)')
    args = parser.parse_args()
    main(args.data_path, args.figdir, trial_figures=not args.skip_trial_figures)
