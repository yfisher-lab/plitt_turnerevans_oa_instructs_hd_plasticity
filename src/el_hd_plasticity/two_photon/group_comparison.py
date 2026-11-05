"""Comparison of groups of flies (genotypes, RNAi lines, treatments) in closed loop and in the dark.

The tbh RNAi, octopamine receptor knockout and similar experiments all ask the same questions of a session
table with a ``group_col`` column ('rnai' or 'condition') naming the group of each fly: does the bump-cue
offset (vector length / variance), the PV magnitude as a function of rotation speed, the PVA velocity as a
function of rotation speed and the bump width differ between groups, and between closed loop and dark? The
functions here do these analyses for any such table; the figure scripts only say which groups to compare.

Session tables need ``fly_id``, ``closed_loop`` (number of preceding closed-loop blocks) and ``dark``
(>= 1 for dark sessions) columns besides `group_col`.
"""

import itertools
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols

from . import bump_shape, plotting, session, stats, tbh_rnai, utilities

DARK_COLOR = plt.cm.Purples(.8)
CL_COLOR = 'black'


@dataclass(frozen=True)
class Group:
    """How one group of flies is identified and drawn.

    Attributes:
        value: Value of the group column.
        label: Name shown in titles and legends.
        color: Line / fill color.
        linestyle: Line style of the group's curves.
        hatch: Hatch pattern of the group's error bands (None for a plain fill).
    """
    value: str
    label: str
    color: Any = 'black'
    linestyle: str = '-'
    hatch: str | None = None


def _pdf(figdir, filename):
    return pathlib.Path(figdir) / filename


def complete_flies(grouped):
    """Rows of the flies that have both a closed-loop and a dark row in a table grouped by ``fly_id`` and ``dark``."""
    return grouped[grouped.groupby('fly_id')['dark'].transform('nunique') == 2]


# ----------------------------------------------------------------------------
# Offset statistics
# ----------------------------------------------------------------------------
def compute_offset_stats(session_manager, group_col, min_cl=1):
    """Per-fly offset statistics: one closed-loop and one dark row per fly (flies with both).

    Args:
        session_manager: Session manager object.
        group_col: Session-table column naming the group of each fly.
        min_cl: Sessions with ``closed_loop >= min_cl`` count as closed loop.

    Returns:
        DataFrame with fly_id, closed_loop (1 = closed loop, 0 = dark), `group_col`, dark, offset_mean, offset_mag,
        offset_var and offset_var_logit.
    """
    stats_df = stats.offset_stats(session_manager, meta_cols=('fly_id', 'closed_loop', 'dark', group_col))
    return stats.add_offset_var(stats.offset_stats_unique(stats_df, group_col=group_col, min_cl=min_cl))


def paired_group_panels(figdir, df, group_col, groups, ycol, ylabel, ylim, filename, yticks=None, yticklabels=None,
                        connect=True, levels=None, xcol='closed_loop'):
    """One panel per group with the dark (x=0) and closed-loop (x=1) value of `ycol` of every fly.

    Returns:
        The figure (saved as `filename` in `figdir`).
    """
    levels = levels or [(0, 'dark', DARK_COLOR), (1, 'closed loop', CL_COLOR)]
    fig, axs = plt.subplots(1, len(groups), sharey=True, figsize=[(2.25) * len(groups), 4], squeeze=False)
    for ax, group in zip(axs[0], groups, strict=True):
        sel = df.loc[df[group_col] == group.value]
        plotting.paired_plot(ax, sel, xcol, levels, ycol, connect)
        ax.set_title(f'{group.label}\nn={sel["fly_id"].nunique()}', fontsize=9)
        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks, labels=yticklabels)
    axs[0, 0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(_pdf(figdir, filename), format='pdf', bbox_inches='tight')
    return fig


def plot_offset_mag(figdir, offset_unique, group_col, groups, filename='offset_mag_summary.pdf', ylims = None):
    """Offset vector length per fly, dark vs closed loop, one panel per group."""
    if ylims is None:
        ylims = [0, 1]

    return paired_group_panels(figdir, offset_unique, group_col, groups, 'offset_mag', 'offset vector length',
                               ylims, filename, yticks=[0, .2, .4, .6, .8, 1.])


def plot_offset_polar(figdir, offset_unique, group_col, groups, filename='offset_hist_summary.pdf'):
    """Mean offset vector of each fly in closed loop, in polar coordinates, colored by group."""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    closed_loop = offset_unique.loc[offset_unique['dark'] == 0]
    for group in groups:
        sel = closed_loop.loc[closed_loop[group_col] == group.value]
        for offset, mag in zip(sel['offset_mean'], sel['offset_mag'], strict=True):
            ax.plot([offset, offset], [0, mag], linewidth=3, color=group.color, alpha=.4)
        ax.plot([], [], linewidth=3, color=group.color, label=group.label)
    ax.set_rticks([.2, .4, .6])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), frameon=False)
    fig.savefig(_pdf(figdir, filename), format='pdf', bbox_inches='tight')
    return fig


def offset_var_tests(figdir, offset_unique, group_col):
    """Do the groups differ in the logit offset variance, and does it change between closed loop and dark?

    Mixed ANOVA (group between flies, closed loop / dark within flies) and the parametric pairwise follow-ups of
    the interaction in both orders (Holm-corrected, as pingouin reports them), plus a QQ plot of the logit
    variance saved in `figdir`.

    Returns:
        Dict of result tables.
    """
    kw = {'dv': 'offset_mag_logit', 'between': group_col, 'subject': 'fly_id', 'within': 'dark', 'data': offset_unique}
    fig, ax = plt.subplots(figsize=[4, 4])
    pg.qqplot(offset_unique['offset_mag_logit'], ax=ax)
    fig.savefig(_pdf(figdir, 'offset_mag_logit_qqplot.pdf'), bbox_inches='tight')
    plt.close(fig)
    return {
        'offset_var_mixed_anova': pg.mixed_anova(**kw),
        'offset_var_pairwise_dark_effect': pg.pairwise_tests(**kw, padjust='holm', parametric=True,
                                                             interaction=True, within_first=False),
        'offset_var_pairwise_group_effect': pg.pairwise_tests(**kw, padjust='holm', parametric=True,
                                                              interaction=True, within_first=True),
    }


# ----------------------------------------------------------------------------
# PV magnitude and PVA velocity versus rotation speed
# ----------------------------------------------------------------------------
BINNED = {'rho': ('rho_dig', stats.rho_stats), 'dphi': ('dphi_dig', stats.dphi_stats)}


def compute_binned_stats(session_manager, group_col, kind, bins, exclude_first_cl=True):
    """PV magnitude ('rho') or PVA velocity ('dphi') binned by rotation speed, averaged per fly and condition.

    Args:
        session_manager: Session manager object.
        group_col: Session-table column naming the group of each fly.
        kind: 'rho' (bins of absolute rotation speed) or 'dphi' (bins of signed rotation speed).
        bins: Bin edges (rad/s); the last bin is open-ended.
        exclude_first_cl: Leave out the sessions of the first closed-loop block (``closed_loop == 1``) and treat
            later ones as closed loop. Dark sessions are those with ``dark >= 1``.

    Returns:
        DataFrame with fly_id, dark (0 / 1), `group_col` and the binned statistic (``rho_dig`` or ``dphi_dig``,
        an array with one value per bin).
    """
    value, fn = BINNED[kind]
    df = fn(session_manager, bins, meta_cols=('fly_id', 'closed_loop', group_col, 'dark'))
    if exclude_first_cl:
        df = df.loc[df['closed_loop'] != 1].copy()
    df.loc[df['dark'] >= 1, 'dark'] = 1
    return df.groupby(['fly_id', 'dark', group_col])[value].apply(
        lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()


def group_arrays(grouped, group_col, value, groups):
    """Per-fly binned arrays of each group and condition, for flies with both: ``{group value: {dark: array}}``."""
    grouped = complete_flies(grouped)
    return {group.value: {dark: np.stack(grouped.loc[(grouped[group_col] == group.value) & (grouped['dark'] == dark),
                                                     value])
                          for dark in (0, 1)}
            for group in groups}


def plot_binned_vs_speed(figdir, grouped, group_col, groups, kind, bins, filename, ylabel, identity_line=False,
                         xlim=None, ylim=None):
    """Mean +- SEM over flies of a binned statistic against rotation speed: closed loop | dark, one line per group."""
    value = BINNED[kind][0]
    centers = bins + np.diff(bins)[0] / 2             # the last bin is open-ended
    arrays = group_arrays(grouped, group_col, value, groups)
    fig, axs = plt.subplots(1, 2, figsize=[10, 5], sharey=True)
    for ax, dark, title in zip(axs, (0, 1), ('Closed loop', 'Darkness'), strict=True):
        for group in groups:
            dat = arrays[group.value][dark]
            mu, sem = utilities.mean_sem(dat)
            ax.plot(centers, mu, color=group.color, linewidth=5, linestyle=group.linestyle,
                    label=f'{group.label} (n={len(dat)})')
            ax.fill_between(centers, mu - sem, mu + sem, color=group.color, alpha=.3, hatch=group.hatch)
        if identity_line:
            ax.plot(centers[[0, -1]], centers[[0, -1]], 'k--', alpha=.3)
        ax.set_title(title)
        ax.set_xlabel('Rot. Vel. (rad/s)')
        if xlim is not None:
            ax.set_xlim(xlim)
        plotting.despine(ax)
    if ylim is not None:
        axs[0].set_ylim(ylim)
    axs[0].set_ylabel(ylabel)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(_pdf(figdir, filename), format='pdf', bbox_inches='tight')
    return fig


def mean_over_bins_tests(grouped, group_col, groups, kind='rho'):
    """Do the groups differ in the PV magnitude averaged over speed bins, in closed loop and in the dark?

    Per condition: a Kruskal-Wallis test across groups (when there are more than two) and Mann-Whitney tests of
    every pair of groups, Holm-corrected within the condition.

    Returns:
        DataFrame with one row per test.
    """
    value = BINNED[kind][0]
    arrays = group_arrays(grouped, group_col, value, groups)
    rows = []
    for dark, condition in ((0, 'closed loop'), (1, 'dark')):
        means = {g.value: np.nanmean(arrays[g.value][dark], axis=-1) for g in groups}
        if len(groups) > 2:
            h, p = sp.stats.kruskal(*means.values())
            rows.append({'condition': condition, 'test': 'Kruskal-Wallis', 'comparison': 'all groups',
                         'statistic': float(h), 'p': float(p)})
        for a, b in itertools.combinations(groups, 2):
            u, p = sp.stats.mannwhitneyu(means[a.value], means[b.value])
            rows.append({'condition': condition, 'test': 'Mann-Whitney U', 'comparison': f'{a.label} vs {b.label}',
                         'statistic': float(u), 'p': float(p)})
    tests = pd.DataFrame(rows)
    tests['p_holm'] = np.nan
    pairwise = tests[tests['test'] == 'Mann-Whitney U']
    for _, sub in pairwise.groupby('condition'):
        tests.loc[sub.index, 'p_holm'] = pg.multicomp(sub['p'].to_numpy(), method='holm')[1]
    return tests


def rho_regression_tests(figdir, grouped, group_col, bins):
    """Do the groups differ in how the PV magnitude depends on rotation speed (and on closed loop vs dark)?

    Reshapes the binned PV magnitude to one row per fly and bin and runs, on closed-loop data, a mixed ANOVA of
    log PV magnitude (group between flies, speed bin within flies) with pairwise follow-ups, the same for the raw
    PV magnitude, and a closed-loop vs dark comparison of the raw PV magnitude across all data. Saves QQ plots.

    Returns:
        Dict of result tables.
    """
    reg = stats.reformat_binned_stats_for_reg(grouped, bins, {'rho': 'rho_dig'}, meta_cols=('fly_id', group_col, 'dark'))
    reg['log_rho'] = np.log(reg['rho'] + 1e-6)
    cl = reg.loc[reg['dark'] == 0]

    fig, ax = plt.subplots(figsize=[4, 4])
    pg.qqplot(reg['log_rho'], ax=ax)
    fig.savefig(_pdf(figdir, 'log_rho_qqplot.pdf'), bbox_inches='tight')
    plt.close(fig)

    def pairwise(dv, within, data):
        return pg.pairwise_tests(dv=dv, between=group_col, subject='fly_id', within=within, data=data,
                                 padjust='holm', parametric=True, interaction=True, within_first=False)

    return {
        'log_rho_mixed_anova_closed_loop': pg.mixed_anova(dv='log_rho', between=group_col, within='dh',
                                                          subject='fly_id', data=cl),
        'log_rho_pairwise_closed_loop': pairwise('log_rho', 'dh', cl),
        'rho_pairwise_closed_loop': pairwise('rho', 'dh', cl),
        'rho_pairwise_dark_vs_closed_loop': pairwise('rho', 'dark', reg),
    }


def dphi_regression_tests(grouped, group_col, groups, reference, bins):
    """Does the slope of PVA velocity against rotation speed differ between groups?

    Fits ``dphi ~ dh * group`` (OLS, `reference` group as baseline) separately in closed loop and in the dark,
    and Wald-tests all slope differences (``dh:group``) and all intercept differences (``group``) together. The
    slope differences of the individual groups are also Holm-corrected.

    Args:
        reference: Value of the group column used as the baseline of the model.

    Returns:
        Dict with a 'dphi_wald_tests' table, a 'dphi_coefficients' table, and one 'dphi_ols_<condition>' summary text
        per condition.
    """
    reg = stats.reformat_binned_stats_for_reg(grouped, bins, {'dphi': 'dphi_dig'}, meta_cols=('fly_id', group_col, 'dark'))
    term = f'C({group_col}, Treatment("{reference}"))'
    others = [g.value for g in groups if g.value != reference]
    wald, coefficients, out = [], [], {}
    for dark, condition in ((0, 'closed_loop'), (1, 'dark')):
        result = ols(f'dphi ~ dh*{term}', reg.loc[reg['dark'] == dark]).fit()
        out[f'dphi_ols_{condition}'] = str(result.summary())
        for name, terms in (('slope differences (dh:group)', [f'dh:{term}[T.{v}]' for v in others]),
                            ('intercept differences (group)', [f'{term}[T.{v}]' for v in others])):
            test = result.wald_test(', '.join(f'({t} = 0)' for t in terms), scalar=True)
            wald.append({'condition': condition, 'terms': name, 'F': float(test.statistic), 'p': float(test.pvalue)})
        slopes = pd.DataFrame({'condition': condition, 'term': [f'dh:{term}[T.{v}]' for v in others]})
        slopes['coef'] = [result.params[t] for t in slopes['term']]
        slopes['p'] = [result.pvalues[t] for t in slopes['term']]
        coefficients.append(stats.add_holm(slopes))
    out['dphi_wald_tests'] = pd.DataFrame(wald)
    out['dphi_coefficients'] = pd.concat(coefficients, ignore_index=True)
    return out


# ----------------------------------------------------------------------------
# Bump width
# ----------------------------------------------------------------------------
def compute_widths(session_manager, group_col, closed_loop_blocks=(1,), rho_min=.05):
    """Bump width (FWHM, radians) per fly in closed loop and in the dark.

    Each session's time-averaged, phase-nulled bump profile (time points with PV magnitude above `rho_min`) gives a
    width (:func:`bump_shape.fwhm`); widths are averaged over the sessions of each fly and condition.

    Args:
        closed_loop_blocks: ``closed_loop`` values of the sessions counted as closed loop (e.g. ``(2,)`` for the second
            block only); those sessions must also have ``dark == 0``. Dark sessions are those with ``dark == 1``.

    Returns:
        DataFrame with fly_id, `group_col`, ``cl_width`` and ``dark_width`` for the flies with both.
    """
    bump = stats.bump_stats(session_manager, meta_cols=('fly_id', group_col, 'closed_loop', 'dark'), rho_min=rho_min)
    bump['width'] = bump_shape.fwhm(np.stack(bump['profile']), factor=16, baseline='min')
    cl = bump.loc[bump['closed_loop'].isin(closed_loop_blocks) & (bump['dark'] == 0)]
    dark = bump.loc[bump['dark'] == 1]
    widths = pd.DataFrame({'cl_width': cl.groupby('fly_id')['width'].mean(),
                           'dark_width': dark.groupby('fly_id')['width'].mean()}).dropna()
    group = bump.groupby('fly_id')[group_col].first()
    return widths.join(group).reset_index()[['fly_id', group_col, 'cl_width', 'dark_width']]


def plot_widths(figdir, widths, group_col, groups, ylims = None, filename='bump_width_dark_v_closed_loop.pdf', connect=True):
    """Bump width per fly, dark vs closed loop, one panel per group."""
    if ylims is None:
        ylims = (np.pi/2, np.pi)
    long = widths.melt(id_vars=['fly_id', group_col], value_vars=['dark_width', 'cl_width'], var_name='which_width',
                       value_name='width')
    long['closed_loop'] = (long['which_width'] == 'cl_width').astype(int)
    return paired_group_panels(figdir, long, group_col, groups, 'width', 'bump FWHM (rad)',
                               ylims, filename, yticks=[np.pi / 2, 3 * np.pi / 4, np.pi],
                               yticklabels=[r'$\pi$/2', r'$3\pi$/4', r'$\pi$'], connect=connect)


def width_change(widths):
    """Per-fly change in bump width between darkness and closed loop (closed loop minus dark), in radians."""
    return widths['cl_width'] - widths['dark_width']


def plot_width_change(figdir, widths, group_col, groups, ylims=None, filename='bump_width_change.pdf', height=4):
    """Change in bump width between darkness and closed loop, the groups side by side.

    One column of flies per group (jittered points and the group median as a bar), so the groups are
    compared directly rather than a panel apart as in :func:`plot_widths`. Drawn in the style of the
    other per-fly summaries (:func:`plotting.paired_plot` and the offset change of the across-dark
    analysis): one column's width per group, seaborn points and no bottom spine.

    Args:
        widths: Per-fly widths from :func:`compute_widths`.
        ylims: Limits of the change axis, in radians; -pi/2 to pi/4 if None. Flies outside them are
            drawn off the axis.

    Returns:
        The figure (saved as `filename` in `figdir`).
    """
    change = widths.assign(width_change=width_change(widths))
    change['x'] = (change[group_col].map({g.value: i for i, g in enumerate(groups)})
                   + change['fly_id'].map(utilities.fly_jitter(change['fly_id'], width=.02)))
    fig, ax = plt.subplots(figsize=[1. * len(groups), height])
    for i, group in enumerate(groups):
        sel = change.loc[change[group_col] == group.value]
        sns.scatterplot(data=sel, x='x', y='width_change', color=group.color, s=100, alpha=.4,
                        legend=False, ax=ax)
        ax.plot([i - .05, i + .05], sel['width_change'].median() * np.ones(2), color=group.color, linewidth=8)
    ax.plot([-.1, len(groups) - .9], [0, 0], 'k--', alpha=.5, zorder=-1)   # no change between the two conditions
    ax.set_xticks(range(len(groups)),
                  labels=[f'{g.label}\nn={(change[group_col] == g.value).sum()}' for g in groups], fontsize=8)
    ax.set_xlabel('')
    ax.set_xlim(-.15, len(groups) - 1 + .15)
    ax.set_ylabel('bump FWHM change (rad)\nclosed loop - dark')
    ax.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4],
                  labels=[r'-$\pi$/2', r'-$\pi$/4', '0', r'$\pi$/4'])
    ax.set_ylim((-np.pi / 2, np.pi / 4) if ylims is None else ylims)
    plotting.despine(ax, bottom=True)
    # no tight_layout: one column per group is narrower than the two-line y label, which bbox_inches handles
    fig.savefig(_pdf(figdir, filename), format='pdf', bbox_inches='tight')
    return fig


def width_tests(widths, group_col, groups):
    """Wilcoxon tests of closed loop vs dark within each group, and Mann-Whitney tests between groups of the
    closed-loop width and of the change between the two conditions (:func:`width_change`).

    Each family is Holm-corrected separately.

    Returns:
        DataFrame with one row per test.
    """
    rows = [{'family': 'closed loop vs dark', 'comparison': g.label,
             **stats.wilcoxon_vs(widths.loc[widths[group_col] == g.value, 'cl_width'],
                                 widths.loc[widths[group_col] == g.value, 'dark_width'].to_numpy())}
            for g in groups]
    change = widths.assign(width_change=width_change(widths))
    for column, family in (('cl_width', 'closed loop, between groups'),
                           ('width_change', 'change (closed loop - dark), between groups')):
        for a, b in itertools.combinations(groups, 2):
            u, p = sp.stats.mannwhitneyu(change.loc[change[group_col] == a.value, column],
                                         change.loc[change[group_col] == b.value, column])
            rows.append({'family': family, 'comparison': f'{a.label} vs {b.label}',
                         'W': float(u), 'p': float(p)})
    tests = pd.DataFrame(rows)
    tests['p_holm'] = np.nan
    for _, sub in tests.groupby('family'):
        tests.loc[sub.index, 'p_holm'] = pg.multicomp(sub['p'].to_numpy(), method='holm')[1]
    return tests


# ----------------------------------------------------------------------------
# Single-fly session figures
# ----------------------------------------------------------------------------
def plot_session_figures(figdir, session_manager, min_closed_loop=2, twindow=(0, 180.1), vmin=-.5, vmax=3, close=True):
    """Heatmaps of one closed-loop and one dark session of every fly, with offset histograms.

    The closed-loop session is the first of the fly's sessions with ``closed_loop >= min_closed_loop``; the dark one
    the first with ``dark == 1``. Flies without both are skipped. Figures go to
    ``<figdir>/sessions/<fly_id>/heatmaps.pdf``.

    Returns:
        List of the fly ids plotted.
    """
    sess_df = session_manager.sess_df
    plotted = []
    for fly in sess_df['fly_id'].unique():
        fly_df = sess_df.loc[sess_df['fly_id'] == fly]
        cl = fly_df.loc[fly_df['closed_loop'] >= min_closed_loop]
        dark = fly_df.loc[fly_df['dark'] == 1]
        if len(cl) == 0 or len(dark) == 0:
            continue
        ts_dict = {'fly': fly,
                   'closed_loop': session.GetTS(session_manager.load_row(cl.iloc[0])),
                   'dark': session.GetTS(session_manager.load_row(dark.iloc[0]))}
        fig, _ = tbh_rnai.plot_sess_heatmaps_w_hist(ts_dict, twindow=twindow, vmin=vmin, vmax=vmax)
        folder = pathlib.Path(figdir) / 'sessions' / fly
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / 'heatmaps.pdf', bbox_inches='tight')
        plotted.append(fly)
        if close:
            plt.close('all')
    return plotted
