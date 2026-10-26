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
import scipy as sp
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
RHO_MIN = .05                                    # minimum PV magnitude for a time point to count as a bump
TURN_ALIGNMENT = 'phase'                         # 'phase': sub-ROI bump phase shift; 'max': whole-ROI peak shift
ANGLE_TICKS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
ANGLE_TICK_LABELS = ['0', r'$\pi$/4', r'$\pi$/2', r'$3\pi$/4', r'$\pi$']
SCATTER_ALPHA=.4


# ----------------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------------
def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Session manager of the usable co-imaging sessions."""
    return nmp.loading.SessionManager(basedir=basedir, figfolder=figdir)


def merge_closed_loop(stats_df):
    """Count all closed-loop blocks as ``cl = 1`` (copy)."""
    stats_df = stats_df.copy()
    stats_df.loc[stats_df['cl'] > 1, 'cl'] = 1
    return stats_df


def paired_dark_vs_cl_plot(df, ycol, ylabel, ylim=(0, np.pi), yticks=ANGLE_TICKS, yticklabels=ANGLE_TICK_LABELS, scatter_alpha=SCATTER_ALPHA):
    """Dark (x=0) vs closed loop (x=1) value per fly (points only, not joined); the style of the EPG-across-dark plots.

    Args:
        df: Table with fly_id, ``cl`` (0 / 1) and `ycol`.
    """
    fig, ax = plt.subplots(figsize=[2, 4])
    nmp.plotting.paired_plot(ax, df, 'cl', [(0, 'dark', DARK_COLOR), (1, 'closed loop', CL_COLOR)], ycol, connect=False,
                             scatter_alpha=SCATTER_ALPHA)
    
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks, labels=yticklabels)
    return fig, ax


def binned_vs_speed_plot(grouped, col, bins, ylabel, yticks=None, yticklabels=None, ylim=None,
                         conditions=((1, CL_COLOR, 'closed loop'), (0, DARK_COLOR, 'dark'))):
    """Mean +- SEM over flies of a binned statistic against rotation speed, one line per condition."""
    centers = bins + np.diff(bins)[0] / 2            # the last bin is open-ended
    fig, ax = plt.subplots(figsize=[5, 5])
    for cl, color, label in conditions:
        dat = np.stack(grouped.loc[grouped['cl'] == cl, col])
        mu, sem = nmp.utilities.mean_sem(dat)
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
    nmp.plotting.despine(ax)
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


def paired_angle_plot(ax, df, xcol, levels, ycol, ylim, connect=True):
    """:func:`nmp.plotting.paired_plot` of an angle (radians), with y-ticks at the multiples of pi/4 inside `ylim`."""
    nmp.plotting.paired_plot(ax, df, xcol, levels, ycol, connect)
    ax.set_ylim(ylim)
    ticks = [(tick, label) for tick, label in zip(ANGLE_TICKS, ANGLE_TICK_LABELS, strict=True)
             if ylim[0] - 1e-9 <= tick <= ylim[1] + 1e-9]
    ax.set_yticks([tick for tick, _ in ticks], labels=[label for _, label in ticks])


def plot_bump_profiles_by_channel(grouped, cl_levels, normalize_rows=True, raw_label='z-scored dF/F'):
    """Mean +- SEM bump profile over flies of EPG and EL, one column per condition.

    Args:
        grouped: Output of :func:`nmp.dual_imaging.group_bump_stats` (``profile``, ``channel``, ``cl``).
        cl_levels: List of ``(cl value, title)`` giving the columns.
        normalize_rows: Add a second row with each fly's profile min-max normalized.
        raw_label: y-label of the un-normalized profiles.
    """
    def minmax(x):
        return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))

    rows = [(lambda x: x, raw_label)] + ([(minmax, 'normalized dF/F')] if normalize_rows else [])
    fig, axs = plt.subplots(len(rows), len(cl_levels), figsize=[5 * len(cl_levels), 5 * len(rows)], squeeze=False,
                            sharex=True)
    for r, (transform, ylabel) in enumerate(rows):
        for c, (cl, title) in enumerate(cl_levels):
            ax = axs[r, c]
            for channel, name in enumerate(CHANNELS):
                sel = grouped.loc[(grouped['cl'] == cl) & (grouped['channel'] == channel)]
                mu, sem = nmp.utilities.mean_sem(transform(np.stack(sel['profile'])))
                x = np.arange(mu.size) - mu.size // 2
                ax.plot(x, mu, color=CHANNEL_COLORS[name], linewidth=3, label=f'{name} (n={len(sel)})')
                ax.fill_between(x, mu - sem, mu + sem, color=CHANNEL_COLORS[name], alpha=.3)
            ax.set_ylabel(ylabel if c == 0 else '')
            ax.set_xlabel('ROI relative to PVA' if r == len(rows) - 1 else '')
            if r == 0:
                ax.set_title(title)
            nmp.plotting.despine(ax)
    axs[0, 0].legend()
    fig.tight_layout()
    return fig


def bump_width_channel_tests(grouped, conditions):
    """Wilcoxon signed-rank tests of the bump width of EL vs EPG within each condition (Holm-corrected).

    Args:
        grouped: Output of :func:`nmp.dual_imaging.group_bump_stats`.
        conditions: List of ``(cl value, name)``.
    """
    rows = []
    for cl, name in conditions:
        wide = grouped.loc[grouped['cl'] == cl].pivot(index='fly_id', columns='channel', values='fwhm').dropna()
        rows.append({'condition': name, 'comparison': 'EL vs EPG', **nmp.stats.wilcoxon_vs(wide[1], wide[0].to_numpy()),
                     'median_diff': float(np.median(wide[1] - wide[0]))})
    return nmp.stats.add_holm(pd.DataFrame(rows))


# ----------------------------------------------------------------------------
# Bump width: EL vs EPG, closed loop vs dark
# ----------------------------------------------------------------------------
def compute_bump_stats(sess_manager, rho_min=RHO_MIN):
    """Mean phase-nulled bump profile and width (FWHM) per fly, condition and channel (all closed-loop blocks merged)."""
    bump_stats = merge_closed_loop(nmp.dual_imaging.bump_stats(sess_manager, rho_min=rho_min))
    return nmp.dual_imaging.group_bump_stats(bump_stats).dropna(subset='fwhm')


def plot_bump_profiles(sess_manager, grouped):
    """Mean bump profile of EPG and EL: closed loop | dark (raw and min-max normalized)."""
    fig = plot_bump_profiles_by_channel(grouped, [(1, 'Closed loop'), (0, 'Dark')])
    fig.savefig(sess_manager.fig_folder / 'bump_profile_EL_v_EPG.pdf', bbox_inches='tight')
    return fig


BUMP_WIDTH_YLIM = (np.pi / 2, np.pi)     # widths of the sessions lie within this range


def plot_bump_width_by_channel(sess_manager, grouped, connect=True):
    """Bump width (FWHM) of EPG vs EL per fly, closed loop | dark, with Wilcoxon tests (Holm-corrected).

    Returns:
        Tuple of (figure, DataFrame of tests).
    """
    conditions = [(1, 'closed loop'), (0, 'dark')]
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=[4, 4])
    levels = [(c, name, CHANNEL_COLORS[name]) for c, name in enumerate(CHANNELS)]
    for ax, (cl, name) in zip(axs, conditions, strict=True):
        sel = grouped.loc[grouped['cl'] == cl]
        sel = sel[sel.groupby('fly_id')['channel'].transform('nunique') == len(CHANNELS)]
        paired_angle_plot(ax, sel, 'channel', levels, 'fwhm', BUMP_WIDTH_YLIM, connect)
        ax.set_title(f'{name} (n={sel["fly_id"].nunique()})', fontsize=9)
    axs[0].set_ylabel('bump FWHM (rad)')
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'bump_width_EL_v_EPG.pdf', bbox_inches='tight')
    return fig, bump_width_channel_tests(grouped, conditions)


def plot_bump_width_by_condition(sess_manager, grouped, connect=True):
    """Bump width (FWHM) in the dark vs closed loop per fly, EPG | EL, with Wilcoxon tests (Holm-corrected).

    Returns:
        Tuple of (figure, DataFrame of tests).
    """
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=[4, 4])
    levels = [(0, 'dark', DARK_COLOR), (1, 'closed loop', CL_COLOR)]
    rows = []
    for ax, (channel, name) in zip(axs, enumerate(CHANNELS), strict=True):
        sel = grouped.loc[grouped['channel'] == channel]
        sel = sel[sel.groupby('fly_id')['cl'].transform('nunique') == 2]
        paired_angle_plot(ax, sel, 'cl', levels, 'fwhm', BUMP_WIDTH_YLIM, connect)
        ax.set_title(f'{name} (n={sel["fly_id"].nunique()})', fontsize=9)
        wide = sel.pivot(index='fly_id', columns='cl', values='fwhm')
        rows.append({'channel': name, 'comparison': 'closed loop vs dark', **nmp.stats.wilcoxon_vs(wide[1], wide[0].to_numpy()),
                     'median_diff': float(np.median(wide[1] - wide[0]))})
    axs[0].set_ylabel('bump FWHM (rad)')
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'bump_width_dark_v_closed_loop.pdf', bbox_inches='tight')
    return fig, nmp.stats.add_holm(pd.DataFrame(rows))


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

    ax.plot([-.05, .05], np.median(offset_unique.loc[offset_unique['cl']==0,'pva_diff'])*np.ones([2,]), color=plt.cm.Purples(.8), linewidth=8)
    ax.plot([1-.05, 1+.05], np.median(offset_unique.loc[offset_unique['cl']==1, 'pva_diff'])*np.ones([2,]), color='black', linewidth=8)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.savefig(sess_manager.fig_folder / 'PVA_diff.pdf', bbox_inches='tight')

    cl = offset_unique['cl'] == 1
    widths = {name: offset_unique.loc[cl, f'fwhm_ch{i + 1}'].mean() for i, name in enumerate(CHANNELS)}
    rows = []
    for condition, mask in (('closed loop', cl), ('dark', ~cl)):
        pva_diff = offset_unique.loc[mask, 'pva_diff']
        rows.append({'condition': condition, 'reference': 'chance (pi/2)', **nmp.stats.wilcoxon_vs(pva_diff, np.pi / 2)})
        for name, width in widths.items():
            rows.append({'condition': condition, 'reference': f'{name} FWHM (closed loop mean)',
                         **nmp.stats.wilcoxon_vs(pva_diff, width)})
    tests = pd.DataFrame(rows)
    chance = tests['reference'] == 'chance (pi/2)'
    tests = nmp.stats.add_holm(tests, chance).rename(columns={'p_holm': 'p_holm_chance'})
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
def plot_turn_profiles(sess_manager, closed_loop=TURN_CLOSED_LOOP, dh_range=TURN_DH_RANGE, alignment=TURN_ALIGNMENT):
    """Turn-triggered EPG and EL bump profiles of the sessions with ``closed_loop == closed_loop``.

    `alignment` is 'phase' (sub-ROI shift by the EPG bump phase) or 'max' (whole-ROI shift to the EPG peak).
    """
    rows = sess_manager.sess_df.loc[sess_manager.sess_df['closed_loop'] == closed_loop]
    timeseries = (nmp.session.GetTS(sess_manager.load_row(row), channels=(0, 1), h_sigma=0.1, dh_sigma=0.3)
                  for _, row in rows.iterrows())
    ccw, cw = nmp.dual_imaging.turn_triggered_profiles(timeseries, *dh_range, alignment=alignment)
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
            mu, sem = nmp.utilities.mean_sem(dat)
            ax.plot(centers, mu, color=CHANNEL_COLORS[name], label=name)
            ax.fill_between(centers, mu - sem, mu + sem, color=CHANNEL_COLORS[name], alpha=.4)
        ax.set_xlabel('rot. speed (rad/s)')
        nmp.plotting.despine(ax)
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
        mu, sem = nmp.utilities.mean_sem(np.stack(grouped.loc[grouped['cl'] == cl, 'pva_diff']))
        ax.plot(centers, mu, color=color, label=label)
        ax.fill_between(centers, mu - sem, mu + sem, color=color, alpha=.4)
    ax.set_yticks(ANGLE_TICKS, labels=ANGLE_TICK_LABELS)
    ax.set_ylabel(r'$| \Delta$ PVA $|$')
    ax.set_xlabel(r'$\| PV \|_2$')
    nmp.plotting.despine(ax)
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

    grouped_bump = compute_bump_stats(sess_manager)
    plot_bump_profiles(sess_manager, grouped_bump)
    tables['bump_width_EL_vs_EPG'] = plot_bump_width_by_channel(sess_manager, grouped_bump)[1]
    tables['bump_width_closed_loop_vs_dark'] = plot_bump_width_by_condition(sess_manager, grouped_bump)[1]

    _, pva = plot_turn_profiles(sess_manager)
    tables['turn_profile_pva'] = pd.DataFrame(pva).rename_axis('channel').reset_index()

    grouped = compute_binned_stats(sess_manager)
    plot_rho_vs_speed(sess_manager, grouped)
    _, spearman = plot_pva_diff_vs_speed(sess_manager, grouped)
    tables['pva_diff_vs_speed_spearman'] = pd.DataFrame(spearman).T.rename_axis('condition').reset_index()
    plot_pva_diff_vs_pv_magnitude(sess_manager)

    if session_figures:
        print(f'{plot_session_figures(sess_manager)} session figure sets')

    nmp.utilities.save_tables(tables, sess_manager.fig_folder)
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
