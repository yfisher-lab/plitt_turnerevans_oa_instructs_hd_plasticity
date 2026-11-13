"""Optogenetic remapping by EL stimulation (EL > CsChrimson and jGCaMP7f): analyses and figures.

Run ``python el_jgcamp7f_cschrimson_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``EL_jGCaMP7f_CsChrimsontdT.ipynb`` does) and call the pieces one at a time. The analyses are the shared ones of
:mod:`el_hd_plasticity.two_photon.remapping`; this file says which sessions and groups to compare.
Group: 'remap' (EL stimulated, paired with a visual cue shifted by 0 / 180 degrees); there is no control group.
"""

import argparse
import pathlib

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group
from el_hd_plasticity.two_photon.remapping import (
    plot_aligned_opto,
    plot_min_dist_summary,
    plot_opto_heatmaps,
    plot_polar_summary,
    plot_remap_error,
    plot_vec_len_comparison,
)

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/EL_jGCaMP7f_CsChrimsontdT')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/EL_jGCaMP7f_CsChrimsontdT')

GROUPS = [Group('remap', 'remap', 'black')]
REFERENCE = 'remap'  # group the other groups are compared with
ZERO_GROUPS = ()  # groups whose first post-cue offset is unwrapped toward baseline
OPTO_ZSCORE = False


def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Usable sessions of the groups of interest, with the condition in the fly id."""
    sess_manager = nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, id_cols=['condition'])
    return sess_manager


def opto_analysis(sess_manager):
    """Optogenetic responses: heatmaps by stimulus location.

    Returns:
        Tuple of (responses aligned to the stimulated position, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    opto = nmp.remapping.compute_opto_responses(sess_manager, GROUPS, zscore=OPTO_ZSCORE)
    plot_opto_heatmaps(figdir, opto, GROUPS)
    aligned = nmp.remapping.align_opto_responses(opto)
    plot_aligned_opto(figdir, aligned, GROUPS)
    tables = {}
    return aligned, tables


def offset_analysis(sess_manager):
    """Offset (bump-cue) remapping: vector lengths, offsets of every fly and the remapping error.

    Returns:
        Tuple of (per-session offset summary, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    summary = nmp.remapping.session_offset_summary(sess_manager)
    sum_stats = nmp.remapping.sum_stats_by_group(summary, GROUPS)
    plot_vec_len_comparison(figdir, summary, GROUPS)
    plot_polar_summary(figdir, sum_stats, GROUPS)
    plot_min_dist_summary(figdir, sum_stats, GROUPS, zero_groups=ZERO_GROUPS)
    _, error_tests = plot_remap_error(figdir, sum_stats, GROUPS, REFERENCE)
    tables = {'remapping_error_tests': error_tests, 'mean_offsets': nmp.remapping.mean_offsets(sum_stats),
              'n_flies': pd.DataFrame({'group': list(sum_stats), 'n_flies': [len(df) for df in sum_stats.values()]})}
    return summary, tables


def plot_session_figures(sess_manager):
    """Heatmaps with offset histograms and polar plots of the baseline and two post-cue scans of every fly."""
    return nmp.remapping.plot_session_figures(sess_manager.fig_folder, sess_manager, GROUPS)


def plot_pairing_scans(sess_manager, close=True):
    """Heatmap of the remap_0deg and remap_180deg (pairing) scans of every fly, with the VR angle and the stimulated
    wedge of each pairing (:func:`el_hd_plasticity.two_photon.remapping.plot_pairing_scans`).

    The table of how many of the programmed pairings each scan recorded is saved as ``<fig_folder>/stats/pairing_scans.csv``.

    Returns:
        DataFrame with one row per pairing scan.
    """
    summary = nmp.remapping.plot_pairing_scans(sess_manager.fig_folder, sess_manager, GROUPS, close=close)
    nmp.utilities.save_tables({'pairing_scans': summary}, sess_manager.fig_folder)
    return summary


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
