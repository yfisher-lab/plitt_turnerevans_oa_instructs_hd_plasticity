"""EPG imaging across closed loop, dark and closed loop again: analyses and figures.

Run ``python epg_across_dark_analysis.py`` to regenerate every figure and the statistics summary, or
import the module (as ``EPG_across_dark.ipynb`` does) and call the pieces one at a time. Each
``plot_*`` function takes the :class:`~el_hd_plasticity.two_photon.loading.SessionManager`, saves
its figure(s) in ``sess_manager.fig_folder`` and returns the figure(s) together with the
numbers behind them.

``closed_loop`` in the session table counts the closed-loop blocks preceding a session: 0 = dark,
1 = first closed-loop block, 2 = second closed-loop block. Unless noted, analyses leave out the
first closed-loop session and treat the later ones as "closed loop" (see :func:`merge_closed_loop`).
"""

import argparse
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group

DEFAULT_BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/EPG_jGCaMP7f_OctB1RRNAi')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/EPG_jGCaMP7f_OctB1RRNAi')
CONDITION = 'control'

META_COLS = ('fly_id', 'condition', 'closed_loop')
DH_BINS = np.linspace(-2, 2, num=20)        # signed heading velocity bins for dPVA (rad/s)
RHO_BINS = np.linspace(0, 2, num=10)        # absolute heading velocity bins for PVA magnitude (rad/s)
RHO_MIN = .05                               # minimum PVA magnitude for a time point to count as a bump
LAGS = nmp.stats.DEFAULT_LAGS               # lags (s) of the PVA velocity / heading velocity correlation
NEURAL_SHIFT = -.2                          # the shift session.SessionTimeseries.from_session applies by default
DARK_COLOR = plt.cm.Purples(.8)
CL_COLOR = 'black'
TWINDOW = (0, 180.1)                        # seconds of the session heatmaps
CL_BLOCK = 2                                # the closed-loop block compared with darkness in the bump analyses
# one group, so the bump width figures of group_comparison (used by the knockout analyses) apply here too
GROUPS = [Group(CONDITION, CONDITION, CL_COLOR)]


# ----------------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------------
def load_session_manager(basedir=DEFAULT_BASEDIR, figdir=DEFAULT_FIGDIR, condition=CONDITION):
    """Session manager of the usable sessions of one condition."""
    sess_manager = nmp.loading.SessionManager(
        basedir=basedir,
        figfolder=figdir,
        folder_cols=('condition',),
        basedir_col='basedir',
        id_cols=['condition'])
    df = sess_manager.sess_df
    sess_manager.filter_rows((df['condition'] == condition) & (df['usable'] == 1))
    return sess_manager


def merge_closed_loop(stats_df):
    """Drop the first closed-loop session and relabel later closed-loop sessions ``closed_loop = 1``."""
    stats_df = stats_df.loc[stats_df['closed_loop'] != 1].copy()
    stats_df.loc[stats_df['closed_loop'] > 1, 'closed_loop'] = 1
    return stats_df


def _paired_dark_vs_cl_plot(df, ycol, ylabel):
    """Dark (x=0) vs closed loop (x=1) value per fly, joined by lines. `df` needs fly_id, closed_loop, `ycol`."""
    fig, ax = plt.subplots(figsize=[2, 4])
    nmp.plotting.paired_plot(ax, df, 'closed_loop', [(0, 'dark', DARK_COLOR), (1, 'closed loop', CL_COLOR)],
                             ycol)
    ax.set_ylabel(ylabel)
    return fig, ax


def _binned_dark_vs_cl_plot(grouped, col, bins, xlabel, ylabel, linewidth=None, alpha=.3):
    """Mean +- SEM over flies of a binned statistic, closed loop vs dark."""
    fig, ax = plt.subplots(figsize=[5, 5])
    for closed_loop, color, label in ((1, CL_COLOR, 'closed loop'), (0, DARK_COLOR, 'dark')):
        dat = np.stack(grouped.loc[grouped['closed_loop'] == closed_loop, col])
        mu, sem = nmp.utilities.mean_sem(dat)
        ax.plot(bins, mu, color=color, linewidth=linewidth, label=label)
        ax.fill_between(bins, mu - sem, mu + sem, color=color, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    nmp.plotting.despine(ax)
    fig.legend()
    return fig, ax


# ----------------------------------------------------------------------------
# Offset statistics: PVA magnitude and bump-cue offset
# ----------------------------------------------------------------------------
def compute_offset_stats(sess_manager):
    """Per-session circular statistics of the bump-cue offset (see :func:`stats.offset_stats`)."""
    return nmp.stats.offset_stats(sess_manager, meta_cols=META_COLS)


def plot_offset_summary(sess_manager, offset_stats):
    """PVA magnitude (offset vector length) per fly, dark vs closed loop.

    Closed loop here is the mean over every closed-loop session (``closed_loop >= 1``), including the
    first block.
    """
    unique = nmp.stats.offset_stats_unique(offset_stats, group_col='condition', dark_col=None)
    fig, ax = _paired_dark_vs_cl_plot(unique, 'offset_mag', 'PVA mag.')
    ax.set_ylim([0, 1.])
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    fig.savefig(sess_manager.fig_folder / 'PVA_mag_dark_v_closed_loop.pdf', bbox_inches='tight')

    wide = unique.pivot(index='fly_id', columns='closed_loop', values='offset_mag').dropna()
    return fig, nmp.stats.wilcoxon_vs(wide[0], wide[1])


def _offset_wide(offset_stats):
    """Mean offset of closed-loop blocks 1 and 2 per fly (flies with both)."""
    return offset_stats.pivot(index='fly_id', columns='closed_loop', values='offset_mean')[[1, 2]].dropna()


def plot_offset_stability(sess_manager, offset_stats):
    """Bump-cue offset in the first vs second closed-loop block, and its absolute change per fly.

    Returns:
        Tuple of ((fig_offsets, fig_change), stats) where stats holds the median absolute change.
    """
    wide = _offset_wide(offset_stats)

    # wrap closed loop 2 onto the branch closest to closed loop 1, for plotting only
    cl_1 = wide[1].to_numpy()
    cl_2_expand = wide[2].to_numpy()[:, np.newaxis] + np.array([2 * np.pi, 0, -2 * np.pi])[np.newaxis, :]
    nearest = np.argmin(np.abs(cl_2_expand - cl_1[:, np.newaxis]), axis=-1, keepdims=True)
    plot_wide = wide.copy()
    plot_wide[2] = np.take_along_axis(cl_2_expand, nearest, axis=-1)[:, 0]

    df = plot_wide.reset_index().melt(id_vars='fly_id', var_name='closed_loop', value_name='offset_mean')
    df['x'] = df['closed_loop'] - 1 + df['fly_id'].map(nmp.utilities.fly_jitter(df['fly_id']))

    fig_offsets, ax = plt.subplots(figsize=[2, 4])
    sns.lineplot(data=df, x='x', y='offset_mean', units='fly_id', estimator=None, color='black',
                 zorder=-10, ax=ax, alpha=.3)
    sns.scatterplot(data=df, x='x', y='offset_mean', hue='closed_loop', palette={1: 'black', 2: 'black'},
                    s=100, legend=False, ax=ax)
    ax.set_xticks([0, 1], labels=('closed loop 1', 'closed loop 2'))
    ax.set_ylabel('bump-cue offset')
    ax.set_xlabel('')
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  labels=[r"-$\pi$", r"-$\pi$/2", 0, r"$\pi$/2", r"$\pi$"])
    ax.set_xlim([-.15, 1.15])
    nmp.plotting.despine(ax, bottom=True)
    fig_offsets.savefig(sess_manager.fig_folder / 'offset_cl1_v_cl2.pdf', bbox_inches='tight')

    diff = pd.Series(np.abs(np.angle(np.exp(1j * (wide[1] - wide[2])))), index=wide.index)   # per fly
    diff_df = pd.DataFrame({'diff': diff, 'x': diff.index.map(nmp.utilities.fly_jitter(diff.index, width=.02))}, index=diff.index)

    fig_change, ax = plt.subplots(figsize=[1, 4])
    sns.scatterplot(data=diff_df, x='x', y='diff', color='black', s=100, legend=False, alpha=.4, ax=ax)
    ax.plot([-.05, .05], np.median(diff) * np.ones([2]), color='black', linewidth=8)
    ax.plot([-.1, .1], [np.pi / 2, np.pi / 2], 'k--', alpha=.5, zorder=-1, label='chance')
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                  labels=[0, r"$\pi$/4", r"$\pi$/2", r"$3\pi$/4", r"$\pi$"])
    ax.set_xticks([.0], labels=[r'$| \Delta \theta |$'])
    ax.set_ylabel(r'$| \Delta $  offset $|$')
    nmp.plotting.despine(ax, bottom=True)
    fig_change.savefig(sess_manager.fig_folder / 'dOffset_cl1_v_cl2.pdf', bbox_inches='tight')

    return (fig_offsets, fig_change), {'n': len(diff), 'median_abs_change': float(np.median(diff)), 'wilcoxon': nmp.stats.wilcoxon_vs(diff, np.pi/2)}


# ----------------------------------------------------------------------------
# Attractor dynamics: dPVA and PVA magnitude versus heading velocity
# ----------------------------------------------------------------------------
def plot_dphi_vs_dheading(sess_manager, dh_bins=DH_BINS):
    """PVA angular velocity as a function of heading angular velocity, closed loop vs dark."""
    dphi_stats = merge_closed_loop(nmp.stats.dphi_stats(sess_manager, dh_bins, meta_cols=META_COLS))
    grouped = dphi_stats.groupby(['fly_id', 'closed_loop'])['dphi_dig'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()
    fig, _ = _binned_dark_vs_cl_plot(grouped, 'dphi_dig', dh_bins, 'dHeading (rad/s)', 'dPVA (rad/s)',
                                     linewidth=5)
    fig.savefig(sess_manager.fig_folder / 'dphi_vs_dheading.pdf', bbox_inches='tight')
    return fig, grouped


def plot_rho_vs_dheading(sess_manager, dh_bins=RHO_BINS):
    """PVA magnitude as a function of absolute heading angular velocity, closed loop vs dark."""
    rho_stats = merge_closed_loop(nmp.stats.rho_stats(sess_manager, dh_bins, meta_cols=META_COLS))
    grouped = rho_stats.groupby(['fly_id', 'closed_loop'])['rho_dig'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()
    fig, _ = _binned_dark_vs_cl_plot(grouped, 'rho_dig', dh_bins, r'$\|$dHeading$\|$ (rad/s)',
                                     'PV magnitude', alpha=.4)
    fig.savefig(sess_manager.fig_folder / 'PVMag_vs_dheading.pdf', bbox_inches='tight')
    return fig, grouped


# ----------------------------------------------------------------------------
# Lag of the neural phase behind behavior
# ----------------------------------------------------------------------------
def plot_phase_lag(sess_manager, lags=LAGS):
    """How far the PVA trails the heading: the correlation of their velocities as a function of lag.

    Two figures: the lag curve closed loop versus dark, and the same pooled over every session. The lag of the
    peak of the pooled curve is the shift ``session.SessionTimeseries.from_session`` applies as ``neural_shift``
    (drawn as a dashed line for comparison).

    Returns:
        Tuple of ((condition figure, pooled figure), dict with the peak lags in seconds).
    """
    lag_stats = nmp.stats.lag_stats(sess_manager, meta_cols=META_COLS, lags=lags)
    
    lag_stats['closed_loop'] = (lag_stats['closed_loop'] > 1).astype(int)
    grouped = lag_stats.groupby(['fly_id', 'closed_loop'])['xcorr'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()

    fig, ax = _binned_dark_vs_cl_plot(grouped, 'xcorr', lags, 'time lag (s)', 'dPVA/dt vs dHeading/dt corr')
    ax.axvline(NEURAL_SHIFT, color='k', linestyle='--', alpha=.3)
    ax.set_xticks(lags[::2])
    fig.savefig(sess_manager.fig_folder / 'xcorr_by_condition.pdf', bbox_inches='tight')

    per_fly = np.stack(grouped.groupby('fly_id')['xcorr'].apply(lambda x: np.nanmean(np.stack(x), axis=0)))
    mu, sem = nmp.utilities.mean_sem(per_fly)
    fig_all, ax = plt.subplots(figsize=[5, 5])
    ax.plot(lags, mu, color=CL_COLOR)
    ax.fill_between(lags, mu - sem, mu + sem, color=CL_COLOR, alpha=.3)
    ax.axvline(NEURAL_SHIFT, color='k', linestyle='--', alpha=.3)
    ax.set_xticks(lags[::2])
    ax.set_xlabel('time lag (s)')
    ax.set_ylabel('dPVA/dt vs dHeading/dt corr')
    ax.set_ylim([0,.13])
    nmp.plotting.despine(ax)
    fig_all.savefig(sess_manager.fig_folder / 'xcorr_all.pdf', bbox_inches='tight')

    fly_peaks = nmp.stats.peak_lag(per_fly, lags)
    summary = {'n_flies': len(per_fly), 'n_sessions': len(lag_stats),
               'peak_lag_all': float(nmp.stats.peak_lag(mu, lags)[0]),
               'peak_lag_per_fly_median': float(np.median(fly_peaks)),
               'peak_lag_per_fly_range': [float(fly_peaks.min()), float(fly_peaks.max())],
               'applied_neural_shift': NEURAL_SHIFT}
    for closed_loop, label in ((1, 'closed_loop'), (0, 'dark')):
        curve = np.stack(grouped.loc[grouped['closed_loop'] == closed_loop, 'xcorr']).mean(axis=0)
        summary[f'peak_lag_{label}'] = float(nmp.stats.peak_lag(curve, lags)[0])
    return (fig, fig_all), summary


# ----------------------------------------------------------------------------
# Bump shape
# ----------------------------------------------------------------------------
def compute_bump_stats(sess_manager, rho_min=RHO_MIN):
    """Mean phase-nulled bump profile and width (FWHM) per fly for dark and closed loop.

    Closed loop is the second block (:func:`merge_closed_loop` drops the first), so the two conditions
    come from sessions next to each other in the experiment. Only flies with both are kept.
    """
    bump_stats = merge_closed_loop(nmp.stats.bump_stats(sess_manager, meta_cols=META_COLS, rho_min=rho_min))
    grouped = nmp.stats.group_bump_stats(bump_stats).dropna(subset='fwhm')
    return grouped[grouped.groupby('fly_id')['closed_loop'].transform('nunique') == 2]


def plot_bump_profile(sess_manager, bump_grouped):
    """Mean +- SEM bump profile over flies (raw and min-max normalized), dark vs closed loop.

    Above the profiles, a bar per condition spans that condition's mean FWHM, centered on the bump
    peak, so the widths that :func:`plot_bump_width` tests can be read off the profiles themselves.
    """
    def minmax(x):
        return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))

    dark = np.stack(bump_grouped.loc[bump_grouped['closed_loop'] == 0, 'profile'])
    cl = np.stack(bump_grouped.loc[bump_grouped['closed_loop'] == 1, 'profile'])
    conditions = ((dark, DARK_COLOR, 'dark', 0), (cl, CL_COLOR, 'closed loop', 1))
    n_rois = dark.shape[1]
    # the x axis counts ROIs, so the widths (radians) are converted at 2*pi per ring
    mean_fwhm = {value: bump_grouped.loc[bump_grouped['closed_loop'] == value, 'fwhm'].mean()
                 for *_, value in conditions}

    fig, axs = plt.subplots(1, 2, figsize=[10, 5])
    for ax, transform, ylabel in zip(axs, (lambda x: x, minmax), ('z-scored dF/F', 'normalized dF/F'), strict=True):
        top = -np.inf
        for dat, color, label, _ in conditions:
            mu, sem = nmp.utilities.mean_sem(transform(dat))
            x = np.arange(mu.size) - mu.size // 2
            ax.plot(x, mu, color=color, linewidth=5, label=label)
            ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=.3)
            top = max(top, (mu + sem).max())
        span = top - ax.get_ylim()[0]
        for i, (_, color, _, value) in enumerate(conditions):
            half = mean_fwhm[value] * n_rois / (2 * np.pi) / 2
            ax.plot([-half, half], np.full(2, top + (.08 + .09 * i) * span), color=color, linewidth=5,
                    solid_capstyle='butt')
        ax.set_xlabel('ROI relative to PVA')
        ax.set_ylabel(ylabel)
        nmp.plotting.despine(ax)
    axs[0].legend(loc='center left')
    fig.savefig(sess_manager.fig_folder / 'bump_profile_dark_v_closed_loop.pdf', bbox_inches='tight')
    return fig, {'n': len(dark), 'mean_fwhm_dark': float(mean_fwhm[0]), 'mean_fwhm_closed_loop': float(mean_fwhm[1])}


def compute_fly_bump_profiles(sess_manager, rho_min=RHO_MIN, cl_block=CL_BLOCK):
    """Phase-nulled bump profile of every time point with a bump, pooled per fly and condition.

    Unlike :func:`compute_bump_stats`, which averages each session over time first, this keeps the
    individual time points, so a single fly's profile comes with a spread.

    Args:
        rho_min: Minimum PV magnitude for a time point to count as a bump.
        cl_block: ``closed_loop`` value of the closed-loop sessions used (the second block by
            default, the one the dark session is compared with throughout).

    Returns:
        Dict (fly_id, closed_loop) -> array of shape (n_timepoints, n_rois), the peak in the middle.
    """
    profiles = {}
    for _, row in sess_manager.iterrows():
        if row['closed_loop'] not in (0, cl_block):
            continue
        ts = nmp.session.GetTS(sess_manager.load_row(row))
        key = (row['fly_id'], int(row['closed_loop']))
        profiles.setdefault(key, []).append(nmp.bump_shape.phase_null_profiles(ts.dff, rho_min=rho_min))
    return {key: np.concatenate(value) for key, value in profiles.items()}


def plot_fly_bump_profiles(sess_manager, fly_profiles, ncols=4, cl_block=CL_BLOCK):
    """Mean +- SEM bump profile of each fly, dark vs closed loop, one panel per fly.

    The band is the SEM over that fly's time points. Each profile is min-max normalized as in
    :func:`plot_bump_profile` (the SEM scaled by the same factor), so the panels compare widths
    rather than amplitudes. Flies without both conditions are left out.

    Returns:
        Tuple of (figure, dict with the number of flies drawn).
    """
    flies = sorted(fly for fly, closed_loop in fly_profiles
                   if closed_loop == 0 and (fly, cl_block) in fly_profiles)
    nrows = int(np.ceil(len(flies) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=[2.4 * ncols, 2.1 * nrows],
                            sharex=True, sharey=True, squeeze=False)
    for ax, fly in zip(axs.flat, flies, strict=False):
        for closed_loop, color, label in ((cl_block, CL_COLOR, 'closed loop'), (0, DARK_COLOR, 'dark')):
            mu, sem = nmp.utilities.mean_sem(fly_profiles[(fly, closed_loop)])
            scale = mu.max() - mu.min()          # the SEM is scaled with the mean, so the band keeps its meaning
            mu, sem = (mu - mu.min()) / scale, sem / scale
            x = np.arange(mu.size) - mu.size // 2
            ax.plot(x, mu, color=color, linewidth=2, label=label)
            ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=.3)
        ax.set_title(fly, fontsize=6)
        nmp.plotting.despine(ax)
    for ax in axs.flat[len(flies):]:
        ax.set_visible(False)
    for col in range(ncols):                     # label the lowest panel of each column, not the hidden ones
        drawn = [ax for ax in axs[:, col] if ax.get_visible()]
        if drawn:
            drawn[-1].set_xlabel('ROI relative to PVA')
            drawn[-1].tick_params(labelbottom=True)
    for ax in axs[:, 0]:
        ax.set_ylabel('normalized dF/F')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'bump_profile_per_fly.pdf', bbox_inches='tight')
    return fig, {'n_flies': len(flies)}


def bump_widths(bump_grouped, condition=CONDITION):
    """Per-fly dark and closed-loop bump widths, in the wide format of ``group_comparison.compute_widths``."""
    wide = bump_grouped.pivot(index='fly_id', columns='closed_loop', values='fwhm').dropna()
    return pd.DataFrame({'fly_id': wide.index.to_numpy(), 'condition': condition,
                         'dark_width': wide[0].to_numpy(), 'cl_width': wide[1].to_numpy()})


def plot_bump_width(sess_manager, bump_grouped):
    """Bump width (FWHM) per fly, dark vs closed loop, with a Wilcoxon signed-rank test.

    Drawn by the same functions as the knockout comparisons (:mod:`group_comparison`), with this
    experiment's single group: widths paired by fly, and their change on its own axis.

    Returns:
        Tuple of ((paired figure, change figure), the Wilcoxon test).
    """
    widths = bump_widths(bump_grouped)
    figdir = sess_manager.fig_folder
    fig = nmp.group_comparison.plot_widths(figdir, widths, 'condition', GROUPS, ylims=(np.pi / 2, np.pi + .1))
    fig_change = nmp.group_comparison.plot_width_change(figdir, widths, 'condition', GROUPS, ylims=(-np.pi/4, np.pi/4), height=2)
    return (fig, fig_change), nmp.stats.wilcoxon_vs(widths['cl_width'].to_numpy(), widths['dark_width'].to_numpy())


# ----------------------------------------------------------------------------
# Single-fly session figures
# ----------------------------------------------------------------------------
EPOCH_CLOSED_LOOP = {'closed_loop 1': 1, 'dark': 0, 'closed_loop 2': 2}      # epoch -> closed_loop value


def plot_session_figures(sess_manager, close=True):
    """Heatmaps with offset histograms, offset histograms and polar offset vectors for every fly.

    Flies need a closed loop 1, dark and closed loop 2 session; others are skipped. Figures go to
    ``<fig_folder>/sessions/<fly_id>/``.

    Returns:
        List of the fly ids that were plotted.
    """
    session_folder = sess_manager.fig_folder / 'sessions'
    sess_df = sess_manager.sess_df
    plotted = []

    for fly in sess_df['fly_id'].unique():
        fly_df = sess_df.loc[sess_df['fly_id'] == fly]
        epochs = {key: fly_df.loc[fly_df['closed_loop'] == cl] for key, cl in EPOCH_CLOSED_LOOP.items()}
        
        if any(len(df) == 0 for df in epochs.values()):
            print(f'Skipping {fly}: missing closed loop 1, dark or closed loop 2')
            continue

        ts_dict = {'fly': fly,           
                   **{key: nmp.session.GetTS(sess_manager.load_row(df.iloc[0])) for key, df in epochs.items()}}
        fly_folder = session_folder / fly
        fly_folder.mkdir(parents=True, exist_ok=True)

        fig, _ = nmp.epg_across_dark.plot_sess_heatmaps_w_hist(ts_dict, vmax=2.5, twindow=TWINDOW)
        fig.savefig(fly_folder / 'heatmaps.pdf', bbox_inches='tight')

        (fig_hist, _), (fig_polar, _) = nmp.epg_across_dark.plot_sess_histograms(ts_dict)
        fig_hist.savefig(fly_folder / 'offset_hist.pdf', bbox_inches='tight')
        fig_polar.savefig(fly_folder / 'offset_polar.pdf', bbox_inches='tight')

        plotted.append(fly)
        if close:
            plt.close('all')

    return plotted


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def main(basedir=DEFAULT_BASEDIR, figdir=DEFAULT_FIGDIR, condition=CONDITION):
    """Generate all figures and write the statistics summary (``stats_summary.json``) to `figdir`."""
    plt.rcParams['pdf.fonttype'] = 42
    sess_manager = load_session_manager(basedir, figdir, condition)
    print(f'{sess_manager.sess_df["fly_id"].nunique()} flies, {len(sess_manager.sess_df)} sessions')

    summary = {}
    offset_stats = compute_offset_stats(sess_manager)
    summary['pva_mag_dark_vs_closed_loop'] = plot_offset_summary(sess_manager, offset_stats)[1]
    summary['offset_cl1_vs_cl2'] = plot_offset_stability(sess_manager, offset_stats)[1]
    plot_dphi_vs_dheading(sess_manager)
    plot_rho_vs_dheading(sess_manager)
    summary['phase_lag'] = plot_phase_lag(sess_manager)[1]

    bump_grouped = compute_bump_stats(sess_manager)
    summary['bump_profile'] = plot_bump_profile(sess_manager, bump_grouped)[1]
    summary['bump_profile_per_fly'] = plot_fly_bump_profiles(sess_manager, compute_fly_bump_profiles(sess_manager))[1]
    summary['bump_width_dark_vs_closed_loop'] = plot_bump_width(sess_manager, bump_grouped)[1]
    summary['session_figures_flies'] = plot_session_figures(sess_manager)

    for name, result in summary.items():
        print(f'{name}: {result}')
    with open(sess_manager.fig_folder / 'stats_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--basedir', type=pathlib.Path, default=DEFAULT_BASEDIR, help='folder with sessions.csv')
    parser.add_argument('--figdir', type=pathlib.Path, default=DEFAULT_FIGDIR, help='where figures are saved')
    parser.add_argument('--condition', default=CONDITION, help='value of the condition column to analyse')
    args = parser.parse_args()
    main(args.basedir, args.figdir, args.condition)
