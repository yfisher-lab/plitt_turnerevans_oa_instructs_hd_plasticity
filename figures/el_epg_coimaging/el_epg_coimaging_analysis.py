"""EL and EPG co-imaging (R18B05-LexA > jGCaMP8s, R60D05-Gal4 > jRGECO1a): analyses and figures.

Run ``python el_epg_coimaging_analysis.py`` to regenerate every figure and statistics table, or import the
module (as ``R18B05LexA_jGCaMP8s_R60D05Gal4_jRGECO1a.ipynb`` does) and call the pieces one at a time. Each
``plot_*`` function saves its figure(s) in the figure folder and returns the figure(s) together with the
numbers behind them. The Turner-Evans dataset (``dte_coimaging_analysis.py``) reuses the helpers here.

``closed_loop`` in the session table counts the closed-loop blocks preceding a session: 0 = dark,
1 or more = closed loop. Statistics use ``cl`` = 1 for closed loop and 0 for dark.

Channel 0 of the imaging data is taken to be the EPG and channel 1 the EL (``CHANNELS``), as in
:mod:`el_hd_plasticity.two_photon.dual_imaging`. Change ``CHANNELS`` if the order in your data is reversed.
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R18B05LexA_jGCaMP8s_R60D05Gal4_jRGECO1a')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/R18B05LexA_jGCaMP8s_R60D05Gal4_jRGECO1a')

CHANNELS = ('EPG', 'EL')                         # name of imaging channel 0 and 1
CHANNEL_COLORS = {'EPG': 'black', 'EL': 'green'}
DARK_COLOR = plt.cm.Purples(.8)
CL_COLOR = 'black'

DH_BINS = np.linspace(-np.pi, np.pi, num=11)     # signed heading velocity bins (rad/s); last bin is open-ended
PV_BINS = np.linspace(0, .8, num=20)             # bins of the channel-averaged PV magnitude; last bin is open-ended
TURN_DH_RANGE = (np.pi, 2 * np.pi)               # turning speeds (rad/s) counted as fast turns: 180-360 deg/s
TURN_CLOSED_LOOP = 0                             # turn-triggered profiles use dark sessions
ANGLE_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
ANGLE_TICK_LABELS = ['0', r'$\pi$/4', r'$\pi$/2', r'$3\pi$/4', r'$\pi$']


# ----------------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------------
def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Session manager of the usable co-imaging sessions."""
    return nmp.loading.SessionManager(basedir=basedir, figfolder=figdir)


def _despine(ax, bottom=False):
    for side in ('top', 'right', *(('bottom',) if bottom else ())):
        ax.spines[side].set_visible(False)


def _jitter(fly_ids, width=.05, seed=0):
    """Dict fly_id -> horizontal jitter so points of different flies don't overlap."""
    fly_ids = pd.unique(fly_ids)
    return dict(zip(fly_ids, np.random.default_rng(seed).uniform(-width, width, len(fly_ids)), strict=True))


def _mean_sem(x):
    """Mean and SEM over flies (axis 0), ignoring NaNs (bins without data)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=0), sp.stats.sem(x, axis=0, nan_policy='omit')


def _wilcoxon_vs(x, reference):
    """Wilcoxon signed-rank test of `x` against a constant (or paired values) `reference`."""
    W, p = sp.stats.wilcoxon(np.asarray(x, dtype=float) - reference)
    return {'n': len(x), 'W': float(W), 'p': float(p)}


def save_tables(tables, figdir):
    """Write every DataFrame of the dict `tables` to ``<figdir>/stats/<name>.csv``; return `tables`."""
    folder = pathlib.Path(figdir) / 'stats'
    folder.mkdir(exist_ok=True, parents=True)
    for name, table in tables.items():
        table.to_csv(folder / f'{name}.csv', index=False)
    return tables


def merge_closed_loop(stats_df):
    """Count all closed-loop blocks as ``cl = 1`` (copy)."""
    stats_df = stats_df.copy()
    stats_df.loc[stats_df['cl'] > 1, 'cl'] = 1
    return stats_df


def paired_dark_vs_cl_plot(df, ycol, ylabel, ylim=(0, np.pi), yticks=ANGLE_TICKS, yticklabels=ANGLE_TICK_LABELS):
    """Dark (x=0) vs closed loop (x=1) value per fly, joined by lines; the style of the EPG-across-dark plots.

    Args:
        df: Table with fly_id, ``cl`` (0 / 1) and `ycol`.
    """
    df = df.copy()
    df['x'] = df['cl'] + df['fly_id'].map(_jitter(df['fly_id']))
    fig, ax = plt.subplots(figsize=[2, 4])
    sns.lineplot(data=df, x='x', y=ycol, units='fly_id', estimator=None, color='black', zorder=-10,
                 ax=ax, alpha=.3)
    sns.scatterplot(data=df, x='x', y=ycol, hue='cl', palette={0: DARK_COLOR, 1: CL_COLOR},
                    s=100, legend=False, ax=ax)
    ax.set_xticks([0, 1], labels=('dark', 'closed loop'))
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_xlim([-.15, 1.15])
    ax.set_ylim(ylim)
    ax.set_yticks(yticks, labels=yticklabels)
    _despine(ax, bottom=True)
    return fig, ax


def binned_vs_speed_plot(grouped, col, bins, ylabel, yticks=None, yticklabels=None, ylim=None,
                         conditions=((1, CL_COLOR, 'closed loop'), (0, DARK_COLOR, 'dark'))):
    """Mean +- SEM over flies of a binned statistic against rotation speed, one line per condition."""
    centers = bins + np.diff(bins)[0] / 2            # the last bin is open-ended
    fig, ax = plt.subplots(figsize=[5, 5])
    for cl, color, label in conditions:
        dat = np.stack(grouped.loc[grouped['cl'] == cl, col])
        mu, sem = _mean_sem(dat)
        ax.plot(centers, mu, color=color, label=label)
        ax.fill_between(centers, mu - sem, mu + sem, color=color, alpha=.4)
    ax.set_xticks(np.linspace(-np.pi, np.pi, 9),
                  labels=[r'-$\pi$', r'-3$\pi$/4', r'-$\pi$/2', r'-$\pi$/4', '0', r'$\pi$/4', r'$\pi$/2',
                          r'3$\pi$/4', r'$\pi$'])
    if yticks is not None:
        ax.set_yticks(yticks, labels=yticklabels)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('rot. speed (rad/s)')
    ax.set_ylabel(ylabel)
    _despine(ax)
    ax.legend()
    return fig, ax


def spearman_vs_speed(dat, bins):
    """Spearman correlation of a binned statistic (flies x bins) with the bin centers, NaNs omitted."""
    centers = bins + np.diff(bins)[0] / 2
    r, p = sp.stats.spearmanr(dat.ravel(), (centers[np.newaxis, :] * np.ones_like(dat)).ravel(), nan_policy='omit')
    return {'r': float(r), 'p': float(p)}


PVA_DIFF_YTICKS = [-np.pi / 16, -np.pi / 32, 0, np.pi / 32, np.pi / 16]
PVA_DIFF_YTICK_LABELS = [r'-$\pi$/16', r'-$\pi$/32', '0', r'$\pi$/32', r'$\pi$/16']


def pva_diff_vs_speed_plot(grouped_pvd, bins, conditions=((1, CL_COLOR, 'closed loop'), (0, DARK_COLOR, 'dark')),
                           ylim=(-np.pi / 16, np.pi / 16)):
    """Signed PVA difference (EL - EPG) against rotation speed, with the Spearman correlation per condition.

    Returns:
        Tuple of (figure, dict condition label -> Spearman result).
    """
    fig, ax = binned_vs_speed_plot(grouped_pvd, 'pva_diff', bins, r'$\Delta$ PVA (EL - EPG)', PVA_DIFF_YTICKS,
                                   PVA_DIFF_YTICK_LABELS, ylim, conditions)
    ax.axhline(0, color='k', linestyle='--', alpha=.5, zorder=-1)
    stats = {label: spearman_vs_speed(np.stack(grouped_pvd.loc[grouped_pvd['cl'] == cl, 'pva_diff']), bins)
             for cl, _, label in conditions}
    return fig, stats


# ----------------------------------------------------------------------------
# Offset statistics: PVA difference and offset difference between the channels
# ----------------------------------------------------------------------------
def compute_offset_stats(sess_manager):
    """Per-fly two-channel offset statistics, one closed-loop and one dark row per fly (flies with both)."""
    return nmp.dual_imaging.offset_stats_unique(nmp.dual_imaging.offset_stats(sess_manager))


def plot_pva_diff(sess_manager, offset_unique):
    """Absolute PVA difference between EPG and EL per fly, dark vs closed loop.

    The dashed lines mark chance (pi/2, black) and one glomerulus (pi/4, red). Each condition is tested
    against chance and against the mean bump width (FWHM) of each channel in closed loop; the chance tests
    are Holm-corrected across closed loop and dark.

    Returns:
        Tuple of (figure, DataFrame of tests).
    """
    fig, ax = paired_dark_vs_cl_plot(offset_unique, 'pva_diff', r'$| \Delta PVA |$')
    ax.axhline(np.pi / 2, color='k', linestyle='--', label='chance', zorder=-1)
    ax.axhline(np.pi / 4, color='r', linestyle='--', label='1 glomerulus', zorder=-1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.savefig(sess_manager.fig_folder / 'PVA_diff.pdf', bbox_inches='tight')

    cl = offset_unique['cl'] == 1
    widths = {name: offset_unique.loc[cl, f'fwhm_ch{i + 1}'].mean() for i, name in enumerate(CHANNELS)}
    rows = []
    for condition, mask in (('closed loop', cl), ('dark', ~cl)):
        pva_diff = offset_unique.loc[mask, 'pva_diff']
        rows.append({'condition': condition, 'reference': 'chance (pi/2)', **_wilcoxon_vs(pva_diff, np.pi / 2)})
        for name, width in widths.items():
            rows.append({'condition': condition, 'reference': f'{name} FWHM (closed loop mean)',
                         **_wilcoxon_vs(pva_diff, width)})
    tests = pd.DataFrame(rows)
    chance = tests['reference'] == 'chance (pi/2)'
    tests['p_holm_chance'] = np.nan
    tests.loc[chance, 'p_holm_chance'] = pg.multicomp(tests.loc[chance, 'p'].to_numpy(), method='holm')[1]
    return fig, tests


def plot_abs_offset_diff(sess_manager, offset_unique):
    """Absolute offset difference between EPG and EL per fly, dark vs closed loop, against chance (pi/2)."""
    fig, ax = paired_dark_vs_cl_plot(offset_unique, 'abs_offset_diff', r'$| \Delta$ offset $|$')
    ax.axhline(np.pi / 2, color='k', linestyle='--', alpha=.5, zorder=-1)
    fig.savefig(sess_manager.fig_folder / 'PVA_diff_sns.pdf', bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Activity during fast turns
# ----------------------------------------------------------------------------
def plot_turn_profiles(sess_manager, closed_loop=TURN_CLOSED_LOOP, dh_range=TURN_DH_RANGE):
    """Turn-triggered EPG and EL bump profiles of the sessions with ``closed_loop == closed_loop``."""
    rows = sess_manager.sess_df.loc[sess_manager.sess_df['closed_loop'] == closed_loop]
    timeseries = (nmp.session.GetTS(sess_manager.load_row(row), channels=(0, 1), h_sigma=0.1, dh_sigma=0.3)
                  for _, row in rows.iterrows())
    ccw, cw = nmp.dual_imaging.turn_triggered_profiles(timeseries, *dh_range)
    fig, pva = nmp.dual_imaging.plot_turn_profiles(ccw, cw, colors=[CHANNEL_COLORS[c] for c in CHANNELS])
    fig.savefig(sess_manager.fig_folder / 'EPG_centered_turns.pdf', format='pdf', bbox_inches='tight')
    return fig, {name: {c: float(v) for c, v in zip(CHANNELS, angles, strict=True)} for name, angles in pva.items()}


# ----------------------------------------------------------------------------
# PV magnitude and PVA difference versus rotation speed
# ----------------------------------------------------------------------------
def compute_binned_stats(sess_manager, dh_bins=DH_BINS):
    """Per-fly PV magnitude of both channels and signed PVA difference binned by heading velocity.

    Returns:
        Dict with 'rho1', 'rho2', 'pva_diff' -> DataFrame (fly_id, cl, mean array per bin) averaged over
        the sessions of each fly and condition. rho1 / rho2 are the PV magnitudes of channel 0 / 1.
    """
    stats_df = merge_closed_loop(nmp.dual_imaging.rho_stats(sess_manager, dh_bins, abs=False))
    return {col: stats_df.groupby(['fly_id', 'cl'])[col].apply(lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()
            for col in ('rho1_dig', 'rho2_dig', 'pva_diff')}


def plot_rho_vs_speed(sess_manager, grouped, dh_bins=DH_BINS):
    """PV magnitude of EPG and EL against rotation speed: closed loop | dark."""
    centers = dh_bins + np.diff(dh_bins)[0] / 2
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=[10, 5])
    for ax, cl in zip(axs, (1, 0), strict=True):
        for ch, name in enumerate(CHANNELS):
            col = f'rho{ch + 1}_dig'
            dat = np.stack(grouped[col].loc[grouped[col]['cl'] == cl, col])
            mu, sem = _mean_sem(dat)
            ax.plot(centers, mu, color=CHANNEL_COLORS[name], label=name)
            ax.fill_between(centers, mu - sem, mu + sem, color=CHANNEL_COLORS[name], alpha=.4)
        ax.set_xlabel('rot. speed (rad/s)')
        _despine(ax)
    axs[0].set_title('Closed loop')
    axs[1].set_title('Dark')
    axs[0].set_ylabel('PV Magnitude')
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'PV_magnitude_summary.pdf', format='pdf', bbox_inches='tight')
    return fig


def plot_pva_diff_vs_speed(sess_manager, grouped, dh_bins=DH_BINS):
    """Signed PVA difference (EL - EPG) against rotation speed in closed loop and dark, with Spearman tests."""
    fig, stats = pva_diff_vs_speed_plot(grouped['pva_diff'], dh_bins)
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'PVA_diff_v_speed.pdf', format='pdf', bbox_inches='tight')
    return fig, stats


def plot_pva_diff_vs_pv_magnitude(sess_manager, pv_bins=PV_BINS):
    """Absolute PVA difference against the channel-averaged PV magnitude, closed loop vs dark."""
    stats_df = merge_closed_loop(nmp.dual_imaging.pvdiff_rho_stats(sess_manager, pv_bins))
    grouped = stats_df.groupby(['fly_id', 'cl'])['pva_diff'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()

    centers = pv_bins + np.diff(pv_bins)[0] / 2
    fig, ax = plt.subplots(figsize=[5, 5])
    for cl, color, label in ((1, CL_COLOR, 'closed loop'), (0, DARK_COLOR, 'dark')):
        mu, sem = _mean_sem(np.stack(grouped.loc[grouped['cl'] == cl, 'pva_diff']))
        ax.plot(centers, mu, color=color, label=label)
        ax.fill_between(centers, mu - sem, mu + sem, color=color, alpha=.4)
    ax.set_yticks(ANGLE_TICKS, labels=ANGLE_TICK_LABELS)
    ax.set_ylabel(r'$| \Delta$ PVA $|$')
    ax.set_xlabel(r'$\| PV \|_2$')
    _despine(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'deltaPVA_PV_magnitude_summary.pdf', format='pdf', bbox_inches='tight')
    return fig, grouped


# ----------------------------------------------------------------------------
# Single-session figures
# ----------------------------------------------------------------------------
def plot_session_figures(sess_manager, close=True):
    """EPG / EL heatmaps with offset histograms, and the PVA difference histogram, of every session.

    Figures go to ``<fig_folder>/sessions/<fly_id>_<session>/``.

    Returns:
        Number of sessions plotted.
    """
    session_folder = sess_manager.fig_folder / 'sessions'
    n = 0
    for _, row in sess_manager.iterrows():
        folder = session_folder / f"{row['fly_id']}_{row['session']}"
        folder.mkdir(exist_ok=True, parents=True)
        ts = nmp.session.GetTS(sess_manager.load_row(row), channels=(0, 1))

        fig, _ = nmp.dual_imaging.plot_sess_heatmaps_w_hist(ts, row['fly_id'], row['session'], vmin=-.5, vmax=3)
        fig.savefig(folder / 'heatmaps.pdf', format='pdf', bbox_inches='tight')
        fig, _ = nmp.dual_imaging.plot_pva_diff_histograms(ts, row['fly_id'], row['session'])
        fig.savefig(folder / 'pva_diff.pdf', format='pdf', bbox_inches='tight')
        n += 1
        if close:
            plt.close('all')
    return n


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def main(basedir=BASEDIR, figdir=DEFAULT_FIGDIR, session_figures=True):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    sess_manager = load_session_manager(basedir, figdir)
    print(f'{sess_manager.sess_df["fly_id"].nunique()} flies, {len(sess_manager.sess_df)} sessions')
    tables = {}

    offset_unique = compute_offset_stats(sess_manager)
    print(f'{offset_unique["fly_id"].nunique()} flies with closed loop and dark sessions')
    tables['pva_diff_tests'] = plot_pva_diff(sess_manager, offset_unique)[1]
    plot_abs_offset_diff(sess_manager, offset_unique)

    _, pva = plot_turn_profiles(sess_manager)
    tables['turn_profile_pva'] = pd.DataFrame(pva).rename_axis('channel').reset_index()

    grouped = compute_binned_stats(sess_manager)
    plot_rho_vs_speed(sess_manager, grouped)
    _, spearman = plot_pva_diff_vs_speed(sess_manager, grouped)
    tables['pva_diff_vs_speed_spearman'] = pd.DataFrame(spearman).T.rename_axis('condition').reset_index()
    plot_pva_diff_vs_pv_magnitude(sess_manager)

    if session_figures:
        print(f'{plot_session_figures(sess_manager)} session figure sets')

    save_tables(tables, sess_manager.fig_folder)
    for name, table in tables.items():
        print(f'\n=== {name} ===\n{table}')
    return tables


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--basedir', type=pathlib.Path, default=BASEDIR, help='folder with sessions.csv')
    parser.add_argument('--figdir', type=pathlib.Path, default=DEFAULT_FIGDIR, help='where figures and tables are saved')
    parser.add_argument('--skip-session-figures', action='store_true', help='skip the per-session figures (slow)')
    args = parser.parse_args()
    main(args.basedir, args.figdir, session_figures=not args.skip_session_figures)
