"""Optogenetic remapping with a dark period before one of the two cues (R18B05-LexA > CsChrimson, R60D05-Gal4 > jGCaMP7f).

Run ``python el_stim_epg_wdark_analysis.py`` to regenerate every figure and statistics table, or import the module
(as ``EL_stim_EPG_image_wdark.ipynb`` does) and call the pieces one at a time. The analyses are the shared ones of
:mod:`el_hd_plasticity.two_photon.remapping`; this file says which sessions to use.

Every fly of the single condition ('replication') saw both the 0 degree and the 180 degree cue, and the ``post_dark``
column of the session table says which of the two was preceded by darkness. The comparison is therefore within fly --
the cue that followed darkness against the one that did not -- rather than between groups as in the other remapping
experiments.
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group
from el_hd_plasticity.two_photon.remapping import (
    STANDARD_SCAN_TYPES,
    plot_aligned_opto,
    plot_opto_heatmaps,
    plot_polar_summary,
    plot_vec_len_comparison,
)

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R18B05LexA_CsChrimsontdT_R60D05Gal4_jGCaMP7f/remapping_w_dark')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/R18B05LexA_CsChrimsontdT_R60D05Gal4_jGCaMP7f_wdark')

GROUPS = [Group('replication', 'EL stim', 'green')]   # the one condition of this session table
OPTO_ZSCORE = True                                    # as in the other CsChrimson remapping analyses

# The two cues of each fly, told apart by whether darkness preceded them.
DARK_LABELS = {0: 'no dark', 1: 'after dark'}
DARK_GROUPS = [Group('no dark', 'no dark', 'black'), Group('after dark', 'after dark', plt.cm.Purples(.8))]
DARK_LEVELS = [(i, group.label, group.color) for i, group in enumerate(DARK_GROUPS)]   # x positions of the two

# Flies split by which of the two cues followed the dark period.
ORDER_GROUPS = [Group('dark before 0 deg', 'dark before the 0 deg cue', plt.cm.PuRd(.8)),
                Group('dark before 180 deg', 'dark before the 180 deg cue', plt.cm.GnBu(.8))]


def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Usable sessions, with the condition in the fly id."""
    return nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, folder_cols=('condition',),
                                      id_cols=['condition'])


def opto_analysis(sess_manager):
    """Optogenetic responses: heatmaps by stimulus location and the response aligned to the stimulated position.

    Returns:
        Tuple of (responses aligned to the stimulated position, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    opto = nmp.remapping.compute_opto_responses(sess_manager, GROUPS, zscore=OPTO_ZSCORE)
    plot_opto_heatmaps(figdir, opto, GROUPS)
    aligned = nmp.remapping.align_opto_responses(opto)
    plot_aligned_opto(figdir, aligned, GROUPS)
    return aligned, {}


def error_by_dark(sum_stats):
    """Remapping error of each fly after the cue that followed darkness and after the one that did not.

    The error of a cue is the circular distance of the offset it left behind from where perfect remapping would
    put it (0 for the 0 degree cue, pi for the 180 degree cue), as in :func:`remapping.avg_diff`; which of the two
    is the dark one is read from the ``post_dark`` of the scan.

    Args:
        sum_stats: Output of :func:`remapping.build_sum_stats` with ``extra_metrics=('post_dark',)``.

    Returns:
        DataFrame with fly_id and the error in radians after each of the two cues, one row per fly and column
        per condition ('no dark', 'after dark').
    """
    error = {}
    for scan, target in (('post_0deg', 0.), ('post_180deg', np.pi)):
        diff = np.abs(target - sum_stats[f'{scan} mean offset'])
        error[scan] = np.minimum(diff, 2 * np.pi - diff)
    dark_first = sum_stats['post_0deg post_dark'] == 1          # darkness came before the 0 degree cue
    return pd.DataFrame({'fly_id': sum_stats['fly_id'].to_numpy(),
                         'no dark': np.where(dark_first, error['post_180deg'], error['post_0deg']),
                         'after dark': np.where(dark_first, error['post_0deg'], error['post_180deg'])})


def plot_error_by_dark(figdir, errors, filename='remapping_error_by_dark.pdf'):
    """The two remapping errors of every fly, the cue that followed darkness against the one that did not."""
    long = errors.melt(id_vars='fly_id', value_vars=[g.value for g in DARK_GROUPS],
                       var_name='dark', value_name='error')
    long['post_dark'] = (long['dark'] == 'after dark').astype(int)     # not 'x': paired_plot uses that name itself
    fig, ax = plt.subplots(figsize=[3, 4])
    nmp.plotting.paired_plot(ax, long, 'post_dark', DARK_LEVELS, 'error')
    ax.axhline(np.pi / 2, color='k', linestyle='--', alpha=.8, linewidth=1)      # chance
    ax.set_ylim(-.1, np.pi)
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                  labels=['0', r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$'])
    ax.set_ylabel('remapping error (rad)')
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def error_by_dark_tests(errors):
    """Wilcoxon tests of the two errors against each other and of each against chance (pi/2), Holm-corrected."""
    rows = [{'comparison': 'after dark vs no dark',
             **nmp.stats.wilcoxon_vs(errors['after dark'], errors['no dark'].to_numpy())}]
    rows += [{'comparison': f'{label} vs chance', **nmp.stats.wilcoxon_vs(errors[label], np.pi / 2)}
             for label in DARK_LABELS.values()]
    return nmp.stats.add_holm(pd.DataFrame(rows))


def offset_analysis(sess_manager):
    """Offset (bump-cue) remapping: vector lengths, offsets of every fly and the remapping error with and without dark.

    Returns:
        Tuple of (per-fly summary table, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    summary = nmp.remapping.session_offset_summary(
        sess_manager, meta_cols=('fly_id', 'condition', 'scan_type', 'post_dark'))
    # the vector lengths of every scan, the two cues of a fly told apart by whether darkness preceded them
    plot_vec_len_comparison(figdir, summary.assign(condition=summary['post_dark'].map(DARK_LABELS)), DARK_GROUPS,
                            filename='vec_len_by_dark.pdf')

    sum_stats = nmp.remapping.build_sum_stats(summary, summary['fly_id'].unique(), scan_types=STANDARD_SCAN_TYPES,
                                              extra_metrics=('post_dark',))
    sum_stats['condition'] = np.where(sum_stats['post_0deg post_dark'] == 1,
                                      ORDER_GROUPS[0].value, ORDER_GROUPS[1].value)
    plot_polar_summary(figdir, {g.value: sum_stats.loc[sum_stats['condition'] == g.value] for g in ORDER_GROUPS},
                       ORDER_GROUPS, filename='offset_polar_by_dark.pdf')

    errors = error_by_dark(sum_stats)
    plot_error_by_dark(figdir, errors)
    tables = {'remapping_error_by_dark_tests': error_by_dark_tests(errors),
              'n_flies': sum_stats['condition'].value_counts().rename('n_flies').rename_axis('group').reset_index()}
    return sum_stats, tables


def plot_session_figures(sess_manager):
    """Heatmaps with offset histograms and polar plots of the baseline and two post-cue scans of every fly."""
    return nmp.remapping.plot_session_figures(sess_manager.fig_folder, sess_manager, GROUPS)


def main(basedir=BASEDIR, figdir=DEFAULT_FIGDIR, session_figures=True):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    sess_manager = load_session_manager(basedir, figdir)
    print(f'{sess_manager.sess_df["fly_id"].nunique()} flies, {len(sess_manager.sess_df)} sessions')
    tables = {}
    for analysis in (opto_analysis, offset_analysis):
        tables.update(analysis(sess_manager)[1])
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
