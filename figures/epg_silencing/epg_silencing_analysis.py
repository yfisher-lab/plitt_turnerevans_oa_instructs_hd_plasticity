"""EPG silencing with shibire-ts (R18B05-LexA > jGCaMP7b, EPG-Gal4 > shi-ts): analyses and figures.

Run ``python epg_silencing_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``epg_shiTS.ipynb`` does) and call the pieces one at a time. Each ``plot_*`` function saves its figure in the
figure folder and returns it together with the numbers behind it.

Each fly is imaged at the permissive temperature (``scan_type`` = 'baseline', 25 C) and after heating
(``scan_type`` = 'post_heat', 33 C). ``condition`` is 'exp' (EPG > shi-ts) or 'gal4_only' (control).
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R18B05LexA_jGCaMP7b_EPGGal4_shiTS')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/R18B05LexA_jGCaMP7b_EPGGal4_shiTS')

DH_BINS = np.linspace(0, 3.14, num=20)         # bins of absolute rotation speed (rad/s); the last bin is open-ended
RHO_MEAN_FROM_BIN = 3                          # per-fly mean PV magnitude uses the bins from this one on
SCAN_TYPES = ('baseline', 'post_heat')
SCAN_LABELS = {'baseline': '25°C', 'post_heat': '33°C'}
BASELINE_COLOR = plt.cm.Greys(.5)
HEAT_COLOR = 'black'
CONDITIONS = {'gal4_only': 'Gal4-only control', 'exp': r'$\mathregular{shi^{TS}}$'}      # panel order and titles
TWINDOW = (0, 180.1)                           # seconds of the session heatmaps


# ----------------------------------------------------------------------------
# Loading and helpers
# ----------------------------------------------------------------------------
def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Session manager of the usable sessions; fly ids include the condition."""
    sess_manager = nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, folder_cols=('condition',),
                                              id_cols=['condition'])
    return sess_manager


# ----------------------------------------------------------------------------
# PV magnitude versus rotation speed
# ----------------------------------------------------------------------------
def compute_binned_stats(sess_manager, dh_bins=DH_BINS):
    """PV magnitude binned by absolute rotation speed, averaged over the sessions of each fly and scan type.

    Returns:
        DataFrame with fly_id, scan_type, condition, ``rho_dig`` (array, one value per bin) and ``rho_mean``
        (mean of the bins from ``RHO_MEAN_FROM_BIN`` on).
    """
    stats_df = nmp.epg_silencing.rho_stats(sess_manager, dh_bins)
    grouped = stats_df.groupby(['fly_id', 'scan_type', 'condition'])['rho_dig'].apply(
        lambda x: np.nanmean(np.stack(x), axis=0)).reset_index()
    grouped['rho_mean'] = grouped['rho_dig'].apply(lambda x: np.nanmean(x[RHO_MEAN_FROM_BIN:]))
    return grouped


def complete_flies(grouped):
    """Rows of the flies that have both a baseline and a post-heat session."""
    return grouped[grouped.groupby('fly_id')['scan_type'].transform('nunique') == len(SCAN_TYPES)]


def plot_rho_vs_speed(sess_manager, grouped, dh_bins=DH_BINS):
    """Mean +- SEM PV magnitude against rotation speed at 25 and 33 C: Gal4-only control | EPG > shi-ts."""
    centers = dh_bins + np.diff(dh_bins)[0] / 2         # the last bin is open-ended
    grouped = complete_flies(grouped)
    fig, axs = plt.subplots(1, 2, figsize=[10, 5], sharey=True)
    for ax, (condition, title) in zip(axs, CONDITIONS.items(), strict=True):
        sel = grouped.loc[grouped['condition'] == condition]
        for scan_type, color, linestyle, hatch, label in (
                ('baseline', 'black', '-', None, '25C'), ('post_heat', plt.cm.Greys(.6), '--', '//', '33C')):
            mu, sem = nmp.utilities.mean_sem(np.stack(sel.loc[sel['scan_type'] == scan_type, 'rho_dig']))
            ax.plot(centers, mu, color=color, linewidth=5, linestyle=linestyle, label=label)
            ax.fill_between(centers, mu - sem, mu + sem, color=color, alpha=.3, hatch=hatch)
        ax.set_title(f'{title} (n={sel["fly_id"].nunique()})')
        ax.set_xlabel('Rot. Vel. (rad/s)')
        nmp.plotting.despine(ax)
    axs[0].set_ylabel('PV magnitude')
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'PVMag_v_RotVel.pdf', format='pdf', bbox_inches='tight')
    return fig


# ----------------------------------------------------------------------------
# Mean PV magnitude at 25 vs 33 C
# ----------------------------------------------------------------------------
def plot_rho_mean(sess_manager, grouped):
    """Mean PV magnitude per fly at 25 vs 33 C: Gal4-only control | EPG > shi-ts."""
    grouped = complete_flies(grouped)
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=[4, 4])
    levels = [(scan_type, SCAN_LABELS[scan_type], color)
              for scan_type, color in zip(SCAN_TYPES, (BASELINE_COLOR, HEAT_COLOR), strict=True)]
    for ax, (condition, title) in zip(axs, CONDITIONS.items(), strict=True):
        df = grouped.loc[grouped['condition'] == condition].copy()
        nmp.plotting.paired_plot(ax, df, 'scan_type', levels, 'rho_mean')
        ax.set_title(f'{title} (n={df["fly_id"].nunique()})', fontsize=9)
        ax.set_ylim([0, 1])
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
    axs[0].set_ylabel('PV magnitude')
    fig.tight_layout()
    fig.savefig(sess_manager.fig_folder / 'offset_mag_summary.pdf', format='pdf', bbox_inches='tight')
    return fig


def rho_mean_tests(grouped):
    """Does heating reduce the mean PV magnitude more in EPG > shi-ts than in controls?

    Runs a mixed ANOVA (condition between flies, scan type within flies), then the non-parametric pairwise
    follow-ups of the interaction in both orders: the effect of heating within each condition (Wilcoxon) and
    the effect of condition at each temperature (Mann-Whitney). The four interaction contrasts are
    Bonferroni-corrected together.

    Returns:
        Dict of result tables: 'rho_mean_mixed_anova', 'rho_mean_pairwise_heat_effect',
        'rho_mean_pairwise_condition_effect' (Holm-corrected within each table, as pingouin reports them) and
        'rho_mean_interaction_contrasts' (the four contrasts with the Bonferroni-corrected p).
    """
    kw = {'dv': 'rho_mean', 'between': 'condition', 'subject': 'fly_id', 'within': 'scan_type', 'data': grouped}
    heat_effect = pg.pairwise_tests(**kw, padjust='holm', parametric=False, interaction=True, within_first=False)
    condition_effect = pg.pairwise_tests(**kw, padjust='holm', parametric=False, interaction=True, within_first=True)

    contrasts = pd.concat([
        heat_effect.iloc[-2:].assign(family='heat effect within condition'),
        condition_effect.iloc[-2:].assign(family='condition effect within temperature')],
        ignore_index=True)
    reject, p_bonferroni = pg.multicomp(contrasts['p_unc'].to_numpy(), alpha=0.05, method='bonferroni')
    contrasts['p_bonferroni'], contrasts['reject'] = p_bonferroni, reject

    return {
        'rho_mean_mixed_anova': pg.mixed_anova(**kw),
        'rho_mean_pairwise_heat_effect': heat_effect,
        'rho_mean_pairwise_condition_effect': condition_effect,
        'rho_mean_interaction_contrasts': contrasts,
    }


# ----------------------------------------------------------------------------
# Single-fly session figures
# ----------------------------------------------------------------------------
def plot_session_figures(sess_manager, twindow=TWINDOW, close=True):
    """Heatmaps of the baseline and post-heat session of every fly, with offset histograms (no heading overlay).

    Flies need both sessions. Figures go to ``<fig_folder>/sessions/<fly_id>/heatmaps.pdf``.

    Returns:
        List of the fly ids plotted.
    """
    session_folder = sess_manager.fig_folder / 'sessions'
    sess_df = sess_manager.sess_df
    plotted = []
    for fly in sess_df['fly_id'].unique():
        fly_df = sess_df.loc[sess_df['fly_id'] == fly]
        rows = {scan_type: fly_df.loc[fly_df['scan_type'] == scan_type] for scan_type in SCAN_TYPES}
        if any(len(df) == 0 for df in rows.values()):
            print(f'Skipping {fly}: missing baseline or post_heat session')
            continue

        ts_dict = {'fly': fly,
                   **{scan_type: nmp.session.GetTS(sess_manager.load_row(df.iloc[0])) for scan_type, df in rows.items()}}
        fig, _ = nmp.shibirets.plot_sess_heatmaps_w_hist(ts_dict, twindow=twindow)

        folder = session_folder / fly
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / 'heatmaps.pdf', bbox_inches='tight')
        plotted.append(fly)
        if close:
            plt.close('all')
    return plotted


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def main(basedir=BASEDIR, figdir=DEFAULT_FIGDIR, session_figures=True):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    sess_manager = load_session_manager(basedir, figdir)
    print(f'{sess_manager.sess_df["fly_id"].nunique()} flies, {len(sess_manager.sess_df)} sessions')

    grouped = compute_binned_stats(sess_manager)
    plot_rho_vs_speed(sess_manager, grouped)
    plot_rho_mean(sess_manager, grouped)
    tables = rho_mean_tests(grouped)

    if session_figures:
        print(f'{len(plot_session_figures(sess_manager))} flies with session figures')

    nmp.utilities.save_tables(tables, sess_manager.fig_folder)
    for name, table in tables.items():
        print(f'\n=== {name} ===\n{table}')
    return tables


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--basedir', type=pathlib.Path, default=BASEDIR, help='folder with sessions.csv')
    parser.add_argument('--figdir', type=pathlib.Path, default=DEFAULT_FIGDIR, help='where figures and tables are saved')
    parser.add_argument('--skip-session-figures', action='store_true', help='skip the per-fly session heatmaps (slow)')
    args = parser.parse_args()
    main(args.basedir, args.figdir, session_figures=not args.skip_session_figures)
