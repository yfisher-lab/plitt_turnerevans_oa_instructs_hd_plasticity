"""GRAB-DA and GRAB-OA imaging (R20A02/R56H10 split-Gal4 EPG-region sensor expression): analyses and figures.

Run ``python grabda_graboa.py`` to regenerate every figure and statistics table, or import the module (as
``GRABDA_GRABOA.ipynb`` does) and call the pieces one at a time. Every ``plot_*`` function takes the
:class:`~el_hd_plasticity.two_photon.loading.SessionManager` objects it needs, saves its figure(s) in
``<figdir>`` and returns the figure(s), usually together with the numbers behind them.

``closed_loop`` in the session table counts the closed-loop blocks preceding a session: 0 = dark, 1 or
more = closed loop (later blocks are merged into closed loop for the binned analyses).

Sensor groups are keyed ``'DA'`` and ``'OA'`` throughout; ``GROUPS`` fixes their order and colors.
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
from statsmodels.formula.api import ols

import el_hd_plasticity.two_photon as nmp

DA_BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R20A02_R56H10_split_GRABDA2m')
OA_BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R20A02_R56H10_split_GRABOA1')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/GRAB')

# group -> (heatmap colormap, line color)
GROUPS = {'DA': ('Blues', 'blue'), 'OA': ('Greens', 'green')}
DARK_COLOR = plt.cm.Purples(.8)
CL_COLOR = 'black'

DH_BINS = np.linspace(0, 3.14, num=20)          # bins of absolute rotation speed (rad/s); last bin is open-ended
EXAMPLE_ROWS = {'DA': 14, 'OA': 3}              # rows of the session tables shown as example sessions
TWINDOW = (0, 180)                              # seconds of the session heatmaps


# ----------------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------------
def load_session_managers(da_basedir=DA_BASEDIR, oa_basedir=OA_BASEDIR, figdir=DEFAULT_FIGDIR):
    """Session managers of the GRAB-DA and GRAB-OA datasets, as a dict keyed by group.

    Figures of each sensor go to ``<figdir>/GRABDA2m`` and ``<figdir>/GRABOA1`` (session figures); the
    summary figures go to `figdir` itself.
    """
    figdir = pathlib.Path(figdir)
    figdir.mkdir(exist_ok=True, parents=True)
    return {
        'DA': nmp.loading.SessionManager(basedir=da_basedir, figfolder=figdir / 'GRABDA2m',
                                         id_cols=['grab'], extra_cols={'grab': 'DA'}),
        'OA': nmp.loading.SessionManager(basedir=oa_basedir, figfolder=figdir / 'GRABOA1',
                                         id_cols=['grab'], extra_cols={'grab': 'OA'}),
    }


def _despine(ax, bottom=False):
    for side in ('top', 'right', *(('bottom',) if bottom else ())):
        ax.spines[side].set_visible(False)


def _jitter(fly_ids, width=.05, seed=0):
    """Dict fly_id -> horizontal jitter so paired points of different flies don't overlap."""
    fly_ids = pd.unique(fly_ids)
    return dict(zip(fly_ids, np.random.default_rng(seed).uniform(-width, width, len(fly_ids)), strict=True))


def _mean_sem(x):
    """Mean and SEM over flies (axis 0), ignoring NaNs (bins without data)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=0), sp.stats.sem(x, axis=0, nan_policy='omit')


def _tidy(table):
    """Pingouin tables carry list-valued columns that CSV handles poorly; stringify them."""
    return table.map(lambda v: str(v) if isinstance(v, (list, tuple, np.ndarray)) else v)


def save_tables(tables, figdir):
    """Write every DataFrame of the nested dict `tables` to ``<figdir>/stats/<name>.csv``; return `tables`."""
    folder = pathlib.Path(figdir) / 'stats'
    folder.mkdir(exist_ok=True, parents=True)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            _tidy(table).to_csv(folder / f'{name}.csv', index=False)
        elif isinstance(table, str):
            (folder / f'{name}.txt').write_text(table)
    return tables


# ----------------------------------------------------------------------------
# Offset statistics: offset variance and vector length, dark vs closed loop
# ----------------------------------------------------------------------------
def compute_offset_stats(managers):
    """Per-fly offset statistics of every group: one row per fly, condition and group.

    Returns:
        DataFrame with fly_id, closed_loop (1 = closed loop, 0 = dark), dark, offset_mean, offset_mag,
        offset_var, offset_var_logit and ``group``.
    """
    unique = []
    for group, sess_manager in managers.items():
        stats_df = nmp.grab_sensors.offset_stats(sess_manager)
        unique.append(nmp.grab_sensors.offset_stats_unique(stats_df).assign(group=group))
    return pd.concat(unique, ignore_index=True)


def plot_offset_summary(figdir, offset_unique, col, ylabel, ylim, filename):
    """`col` per fly, dark vs closed loop, one panel per group.

    Each panel is styled like the paired plots of ``epg_across_dark_analysis`` (2 x 4 in, no bottom spine).
    """
    fig, axs = plt.subplots(1, len(GROUPS), sharey=True, figsize=[2 * len(GROUPS), 4])
    for ax, group in zip(axs, GROUPS, strict=True):
        df = offset_unique.loc[offset_unique['group'] == group].copy()
        df['x'] = df['closed_loop'] + df['fly_id'].map(_jitter(df['fly_id']))
        sns.lineplot(data=df, x='x', y=col, units='fly_id', estimator=None, color='black', zorder=-10,
                     ax=ax, alpha=.3)
        sns.scatterplot(data=df, x='x', y=col, hue='closed_loop', palette={0: DARK_COLOR, 1: CL_COLOR},
                        s=100, legend=False, ax=ax)
        ax.set_title(f'{group} (n={df["fly_id"].nunique()})')
        ax.set_xticks([0, 1], labels=['dark', 'closed loop'])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim([-.15, 1.15])
        ax.set_ylim(ylim)
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        _despine(ax, bottom=True)
    axs[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, format='pdf', bbox_inches='tight')
    return fig


def offset_var_tests(figdir, offset_unique):
    """Does the logit offset variance differ between dark and closed loop, and between DA and OA?

    Runs a mixed ANOVA (group between flies, dark/closed loop within flies), the pairwise follow-ups
    with Holm correction (both orders of the interaction) and DA vs OA t tests in closed loop and in the
    dark. Also saves a QQ plot of the logit variance.

    Returns:
        Dict of result tables.
    """
    kw = {'dv': 'offset_var_logit', 'between': 'group', 'subject': 'fly_id', 'within': 'dark',
          'data': offset_unique}

    fig, ax = plt.subplots(figsize=[4, 4])
    pg.qqplot(offset_unique['offset_var_logit'], ax=ax)
    fig.savefig(pathlib.Path(figdir) / 'offset_var_logit_qqplot.pdf', bbox_inches='tight')

    ttests = {}
    for name, dark in (('closed_loop', 0), ('dark', 1)):
        cond = offset_unique.loc[offset_unique['dark'] == dark]
        ttest = sp.stats.ttest_ind(cond.loc[cond['group'] == 'DA', 'offset_var_logit'],
                                   cond.loc[cond['group'] == 'OA', 'offset_var_logit'])
        ttests[name] = {'condition': name, 't': ttest.statistic, 'p': ttest.pvalue, 'df': ttest.df}
    return {
        'offset_var_mixed_anova': pg.mixed_anova(**kw),
        'offset_var_pairwise_group_first': pg.pairwise_tests(**kw, padjust='holm', parametric=True,
                                                             interaction=True, within_first=False),
        'offset_var_pairwise_dark_first': pg.pairwise_tests(**kw, padjust='holm', parametric=True,
                                                            interaction=True, within_first=True),
        'offset_var_ttest_DA_vs_OA': pd.DataFrame(list(ttests.values())),
    }


# ----------------------------------------------------------------------------
# PVA magnitude and outer-ring fluorescence versus rotation speed
# ----------------------------------------------------------------------------
def compute_binned_stats(managers, dh_bins=DH_BINS):
    """PVA magnitude and outer-ring fluorescence binned by absolute rotation speed.

    Sessions with ``closed_loop > 1`` count as closed loop (1); sessions are averaged per fly and
    condition.

    Returns:
        Tuple of (grouped, reg_stats): `grouped` is a dict group -> per-fly binned arrays (``rho_dig``,
        ``F_dig``, see :func:`stats.group_rho_stats`); `reg_stats` is the long-format table of all groups
        (fly_id, closed_loop, rho, log_rho, dh, F, log_F, group) for regression.
    """
    grouped, reg = {}, []
    for group, sess_manager in managers.items():
        rho_stats = nmp.grab_sensors.rho_stats(sess_manager, dh_bins)
        rho_stats.loc[rho_stats['closed_loop'] > 1, 'closed_loop'] = 1
        grouped[group] = nmp.stats.group_rho_stats(rho_stats)
        reg.append(nmp.grab_sensors.reformat_rho_stats_for_reg(grouped[group], dh_bins).assign(group=group))
    return grouped, pd.concat(reg, ignore_index=True)


def plot_binned_summary(figdir, grouped, col, ylabel, filename, dh_bins=DH_BINS):
    """Mean +- SEM over flies of a binned statistic (``rho_dig`` or ``F_dig``): closed loop | dark, DA vs OA."""
    centers = dh_bins + np.diff(dh_bins)[0] / 2          # the last bin is open-ended (>= last edge)
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=[6, 4])
    for ax, closed_loop in zip(axs, (1, 0), strict=True):
        for group, (_, color) in GROUPS.items():
            df = grouped[group]
            dat = np.stack(df.loc[df['closed_loop'] == closed_loop, col])
            mu, sem = _mean_sem(dat)
            ax.plot(centers, mu, color=color, label=group)
            ax.fill_between(centers, mu - sem, mu + sem, color=color, alpha=.4)
        ax.set_xlabel('rot. speed (rad/s)')
        _despine(ax)
    axs[0].set_title('Closed loop')
    axs[1].set_title('Dark')
    axs[0].set_ylabel(ylabel)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, format='pdf', bbox_inches='tight')
    return fig


def binned_stats_tests(figdir, reg_stats, dv):
    """Do DA and OA differ in how `dv` (``log_rho`` or ``log_F``) depends on rotation speed?

    Mixed ANOVA (group between flies, speed bin within flies) in closed loop and in the dark, an OLS model
    ``dv ~ dh * group`` in closed loop, and per-bin QQ plots of `dv`.

    Returns:
        Dict of result tables / text.
    """
    cl = reg_stats.loc[reg_stats['closed_loop'] == 1]
    dark = reg_stats.loc[reg_stats['closed_loop'] == 0]

    dhs = np.sort(reg_stats['dh'].unique())
    ncols = 5
    fig, axs = plt.subplots(int(np.ceil(len(dhs) / ncols)), ncols, figsize=[3 * ncols, 3 * np.ceil(len(dhs) / ncols)])
    for ax, dh in zip(axs.flat, dhs, strict=False):
        pg.qqplot(reg_stats.loc[reg_stats['dh'] == dh, dv], ax=ax)
        ax.set_title(f'dh={dh:.2f}', fontsize=9)
    for ax in axs.flat[len(dhs):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / f'{dv}_qqplots.pdf', bbox_inches='tight')
    plt.close(fig)

    def anova(data):
        return pg.mixed_anova(dv=dv, between='group', within='dh', subject='fly_id', data=data)

    return {
        f'{dv}_mixed_anova_closed_loop': anova(cl),
        f'{dv}_mixed_anova_dark': anova(dark),
        f'{dv}_ols_closed_loop': str(ols(f'{dv} ~ dh*C(group)', cl).fit().summary()),
    }


# ----------------------------------------------------------------------------
# Example sessions
# ----------------------------------------------------------------------------
def plot_example_session(sess_manager, row_index, cmap, figdir, plot_times=np.arange(0, 361, 60)):
    """Heading-aligned heatmap and first- vs second-half mean profile of one session.

    The mean profile of the first half of the session is rolled to put its peak at the center ROI; the
    second half is rolled by the same amount, showing whether the peak location is stable.

    Returns:
        Tuple of (fig_heatmap, fig_mean).
    """
    row = sess_manager.sess_df.iloc[row_index]
    fly_id = row['fly_id']
    sess = nmp.session.GetTS(sess_manager.load_row(row))
    center = nmp.utilities.HEADING_CENTER_ROI
    n_rois = sess.dff.shape[0]
    color = plt.get_cmap(cmap)

    fig_heatmap, axs = plt.subplots(2, 1, figsize=[12, 4], sharex=True)
    x = np.arange(sess.dff.shape[1])
    time = sess.time - sess.time.iloc[0]
    axs[0].imshow(sess.dff, cmap=cmap, aspect='auto', vmin=-.5, vmax=3, interpolation='none')
    axs[1].imshow(sess.dff_h_aligned, cmap=cmap, aspect='auto', vmin=-.5, vmax=3, interpolation='none')
    axs[0].scatter(x, nmp.utilities.heading_to_roi_coords(sess.heading, n_rois), color='orange', s=5)
    axs[1].scatter(x, center * np.ones_like(x), color='orange', s=5)
    for ax in axs:
        nmp.plotting.format_roi_axis(ax, 'y')
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axs[0].set_title(fly_id)
    nmp.plotting.format_time_axis(axs[1], time, plot_times)
    fig_heatmap.tight_layout()
    fig_heatmap.savefig(pathlib.Path(figdir) / f'{fly_id}_aligned_heatmap.pdf', bbox_inches='tight')

    fig_mean, axs = plt.subplots(1, 2, figsize=[6, 3], sharey=True)
    halfway = sess.dff_h_aligned.shape[1] // 2
    first = sess.dff_h_aligned[:, :halfway].mean(axis=-1)
    second = sess.dff_h_aligned[:, halfway:].mean(axis=-1)
    shift = np.argmax(first)
    reference = np.roll(first, -shift - center)
    axs[0].plot(first, color='k')
    axs[0].plot(reference, color=color(.4), linestyle='--', alpha=.6)
    axs[1].plot(np.roll(second, -shift - center), color=color(.8))
    axs[1].plot(reference, color=color(.4), linestyle='--', alpha=.6)
    for ax in axs:
        ax.set_ylim([-.5, 1])
        ax.set_xticks([0, (n_rois - 1) / 2, n_rois - 1], labels=[r'-$\pi$', '0', r'$\pi$'])
        ax.set_ylabel('z-scored F')
        ax.set_xlabel('EB position (centered)')
        _despine(ax)
    axs[0].set_title('First half')
    axs[1].set_title('Second half')
    fig_mean.tight_layout()
    fig_mean.savefig(pathlib.Path(figdir) / f'{fly_id}_aligned_mean.pdf', bbox_inches='tight')
    return fig_heatmap, fig_mean


def plot_example_heatmap_w_hist(sess_manager, row_index, figdir, twindow=TWINDOW):
    """Heatmap with offset histogram of one session (magenta colormap, as in the figure)."""
    row = sess_manager.sess_df.iloc[row_index]
    ts_dict = {'fly': row['fly_id'], 'closed loop': nmp.session.GetTS(sess_manager.load_row(row))}
    magenta = sns.light_palette(plt.cm.cool(.7), as_cmap=True)
    fig, _ = nmp.grab_sensors.plot_sess_heatmaps_w_hist(ts_dict, twindow=twindow, vmin=-.5, vmax=3, cmap=magenta)
    fig.savefig(pathlib.Path(figdir) / f'{row["fly_id"]}_graboa_heatmap_w_hist.pdf', format='pdf', bbox_inches='tight')
    return fig


def plot_session_figures(sess_manager, cmap, twindow=TWINDOW, close=True):
    """Heatmaps, offset histograms and polar plots of every session.

    Figures go to ``<fig_folder>/sessions/<fly_id>/<session>/``.

    Returns:
        Number of sessions plotted.
    """
    session_folder = sess_manager.fig_folder / 'sessions'
    n = 0
    for _, row in sess_manager.iterrows():
        ts_dict = {'fly': row['fly_id'], row['session']: nmp.session.GetTS(sess_manager.load_row(row))}
        fig_heatmaps = nmp.grab_sensors.plot_sess_heatmaps(ts_dict, cmap=cmap, twindow=twindow, vmin=-.5, vmax=3)
        (fig_hist, _), (fig_polar, _) = nmp.grab_sensors.plot_sess_histograms(ts_dict, cmap=cmap)

        folder = session_folder / row['fly_id'] / row['session']
        folder.mkdir(exist_ok=True, parents=True)
        fig_heatmaps.savefig(folder / 'heatmaps.pdf', format='pdf', bbox_inches='tight')
        fig_hist.savefig(folder / 'offset_hist.pdf', format='pdf', bbox_inches='tight')
        fig_polar.savefig(folder / 'offset_polar.pdf', format='pdf', bbox_inches='tight')
        n += 1
        if close:
            plt.close('all')
    return n


# ----------------------------------------------------------------------------
# Heading-aligned responses across flies
# ----------------------------------------------------------------------------
def compute_split_halves(managers):
    """Peak-aligned first- and second-half profiles of every closed-loop session, per group."""
    return {group: nmp.grab_sensors.align_peaks_split_halves(sess_manager)
            for group, sess_manager in managers.items()}


def plot_split_half_alignment(figdir, halves):
    """Heatmaps (one row per session) and mean +- SEM of the heading-aligned profile, first vs second half.

    Returns:
        Tuple of (fig_heatmaps, fig_average).
    """
    fig_heatmaps, axs = plt.subplots(len(GROUPS), 2, figsize=[6, 6], constrained_layout=True)
    for row, (group, (cmap, _)) in enumerate(GROUPS.items()):
        for col, (half, name) in enumerate(zip(halves[group], ('first half', 'second half'), strict=True)):
            h = axs[row, col].imshow(half, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none')
            fig_heatmaps.colorbar(h, ax=axs[row, col])
            axs[row, col].set_title(f'{group} {name}')
            nmp.plotting.format_roi_axis(axs[row, col], 'x')
            axs[row, col].set_ylabel('Sessions')
    fig_heatmaps.savefig(pathlib.Path(figdir) / 'heading_aligned_heatmaps.pdf', format='pdf', bbox_inches='tight')

    fig_average, axs = plt.subplots(1, 2, figsize=[4, 2.5], sharey=True, sharex=True)
    for group, (_, color) in GROUPS.items():
        for ax, half in zip(axs, halves[group], strict=True):
            mu, sem = half.mean(axis=0), sp.stats.sem(half, axis=0, nan_policy='omit')
            ax.plot(mu, color=color, label=group)
            ax.fill_between(np.arange(mu.shape[0]), mu - sem, mu + sem, color=color, alpha=.4)
    for ax, name in zip(axs, ('first half', 'second half'), strict=True):
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('aligned ROIs')
        ax.set_xticks(nmp.plotting.ROI_TICKS, labels=nmp.plotting.ROI_TICK_LABELS)
        _despine(ax)
    axs[0].set_ylabel('z scored F')
    axs[0].legend()
    fig_average.tight_layout()
    fig_average.savefig(pathlib.Path(figdir) / 'heading_aligned_average.pdf', format='pdf', bbox_inches='tight')
    return fig_heatmaps, fig_average


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def main(da_basedir=DA_BASEDIR, oa_basedir=OA_BASEDIR, figdir=DEFAULT_FIGDIR, session_figures=True):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    managers = load_session_managers(da_basedir, oa_basedir, figdir)
    tables = {}

    offset_unique = compute_offset_stats(managers)
    plot_offset_summary(figdir, offset_unique, 'offset_var', 'Offset variance', [0, 1.],
                        'offset_variance_summary.pdf')
    plot_offset_summary(figdir, offset_unique, 'offset_mag', 'Offset vec length', [0, 1.],
                        'offset_mag_summary.pdf')
    tables.update(offset_var_tests(figdir, offset_unique))

    grouped, reg_stats = compute_binned_stats(managers)
    plot_binned_summary(figdir, grouped, 'rho_dig', 'PV Magnitude', 'PV_magnitude_summary.pdf')
    plot_binned_summary(figdir, grouped, 'F_dig', 'Whole EB z-scored F', 'F_summary.pdf')
    tables.update(binned_stats_tests(figdir, reg_stats, 'log_rho'))
    tables.update(binned_stats_tests(figdir, reg_stats, 'log_F'))

    for group, sess_manager in managers.items():
        cmap = GROUPS[group][0]
        plot_example_session(sess_manager, EXAMPLE_ROWS[group], cmap, figdir)
    plot_example_heatmap_w_hist(managers['OA'], EXAMPLE_ROWS['OA'], figdir)

    plot_split_half_alignment(figdir, compute_split_halves(managers))

    if session_figures:
        for group, sess_manager in managers.items():
            n = plot_session_figures(sess_manager, GROUPS[group][0])
            print(f'{group}: {n} session figure sets')

    save_tables(tables, figdir)
    for name, table in tables.items():
        print(f'\n=== {name} ===\n{table}')
    return tables


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--da-basedir', type=pathlib.Path, default=DA_BASEDIR, help='GRAB-DA folder with sessions.csv')
    parser.add_argument('--oa-basedir', type=pathlib.Path, default=OA_BASEDIR, help='GRAB-OA folder with sessions.csv')
    parser.add_argument('--figdir', type=pathlib.Path, default=DEFAULT_FIGDIR, help='where figures and tables are saved')
    parser.add_argument('--skip-session-figures', action='store_true', help='skip the per-session figures (slow)')
    args = parser.parse_args()
    main(args.da_basedir, args.oa_basedir, args.figdir, session_figures=not args.skip_session_figures)
