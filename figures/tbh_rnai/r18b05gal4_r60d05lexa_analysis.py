"""TBH RNAi in the EL (R18B05-Gal4 > tbh RNAi, R60D05-LexA > jGCaMP7b): analyses and figures.

Run ``python r18b05gal4_r60d05lexa_analysis.py`` to regenerate every figure and statistics table, or import the
module (as ``R18B05Gal4_R60D05LexA.ipynb`` does) and call the pieces one at a time. The analyses are the
shared ones of :mod:`el_hd_plasticity.two_photon.group_comparison`; this file says which sessions and groups
to compare. Groups: 'control' (UAS-RNAi without Gal4), 'gal4_control' (Gal4 without the RNAi line of interest) and
'rnai_67968' (TBH RNAi). Sessions of the first closed-loop block are left out of the speed analyses, and the
tbh RNAi convention applies: more than one closed-loop block of experience counts as closed loop.
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R60D05LexA_jGCaMP7b_R18B05Gal4_TBHRNAi')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/R60D05LexA_jGCaMP7b_R18B05Gal4_TBHRNAi')

GROUP_COL = 'rnai'
GROUPS = [Group('control', 'UAS control', 'black'),
          Group('gal4_control', 'Gal4 control', plt.cm.Greys(.6), '--', '//'),
          Group('rnai_67968', 'TBH RNAi', 'green')]
REFERENCE = 'rnai_67968'                          # baseline group of the PVA velocity regression
MIN_CL = nmp.tbh_rnai.MIN_CL                      # closed-loop blocks of experience for a session to count as closed loop
RHO_BINS = np.linspace(0, np.pi / 2, num=20)      # bins of absolute rotation speed (rad/s)
DPHI_BINS = np.linspace(-2, 2, num=20)            # bins of signed rotation speed (rad/s)
WIDTH_CL_BLOCKS = (2,)                            # closed-loop blocks used for the bump width


def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Usable sessions of the groups of interest (neither closed-loop nor dark sessions and RNAi line 76062 removed)."""
    sess_manager = nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, folder_cols=(GROUP_COL,))
    df = sess_manager.sess_df
    df.columns = df.columns.str.strip()                      # the table has a column named 'rnai_line '
    df[GROUP_COL] = np.where(df['gal4_line'] != 'control', df['rnai_line'], 'gal4_control')
    df['fly_id'] = df['date'] + '_' + df['fly'] + '_' + df['gal4_line'] + '_' + df['rnai_line']
    sess_manager.filter_rows(~((df['closed_loop'] == 0) & (df['dark'] == 0)) & (df['rnai_line'] != 'rnai_76062'))
    return sess_manager


def offset_analysis(sess_manager):
    """Offset vector length and variance, dark vs closed loop.

    Returns:
        Tuple of (per-fly offset table, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    unique = nmp.group_comparison.compute_offset_stats(sess_manager, GROUP_COL, MIN_CL)
    nmp.group_comparison.plot_offset_mag(figdir, unique, GROUP_COL, GROUPS)
    nmp.group_comparison.plot_offset_polar(figdir, unique, GROUP_COL, GROUPS)
    tables = nmp.group_comparison.offset_var_tests(figdir, unique, GROUP_COL)
    tables['offset_n_flies'] = unique.groupby(GROUP_COL)['fly_id'].nunique().rename('n_flies').reset_index()
    return unique, tables


def rho_analysis(sess_manager):
    """PV magnitude vs rotation speed: curves, mean-over-speed tests and regression tests."""
    figdir = sess_manager.fig_folder
    grouped = nmp.group_comparison.compute_binned_stats(sess_manager, GROUP_COL, 'rho', RHO_BINS)
    nmp.group_comparison.plot_binned_vs_speed(figdir, grouped, GROUP_COL, GROUPS, 'rho', RHO_BINS,
                                              'PVMag_v_RotVel.pdf', 'PV magnitude')
    tables = {'rho_mean_over_speed_tests': nmp.group_comparison.mean_over_bins_tests(grouped, GROUP_COL, GROUPS)}
    tables.update(nmp.group_comparison.rho_regression_tests(figdir, grouped, GROUP_COL, RHO_BINS))
    return grouped, tables


def dphi_analysis(sess_manager):
    """PVA velocity vs rotation speed: curves and regression tests of the slope."""
    figdir = sess_manager.fig_folder
    grouped = nmp.group_comparison.compute_binned_stats(sess_manager, GROUP_COL, 'dphi', DPHI_BINS)
    nmp.group_comparison.plot_binned_vs_speed(figdir, grouped, GROUP_COL, GROUPS, 'dphi', DPHI_BINS,
                                              'dPVA_v_RotVel.pdf', 'PVA Vel. (rad/s)', identity_line=True)
    return grouped, nmp.group_comparison.dphi_regression_tests(grouped, GROUP_COL, GROUPS, REFERENCE, DPHI_BINS)


def width_analysis(sess_manager):
    """Bump width, dark vs closed loop, per group."""
    widths = nmp.group_comparison.compute_widths(sess_manager, GROUP_COL, WIDTH_CL_BLOCKS)
    nmp.group_comparison.plot_widths(sess_manager.fig_folder, widths, GROUP_COL, GROUPS)
    return widths, {'bump_width_tests': nmp.group_comparison.width_tests(widths, GROUP_COL, GROUPS)}


def plot_session_figures(sess_manager, **kwargs):
    """Closed-loop and dark heatmaps with offset histograms of every fly."""
    return nmp.group_comparison.plot_session_figures(sess_manager.fig_folder, sess_manager, min_closed_loop=MIN_CL, **kwargs)


def main(basedir=BASEDIR, figdir=DEFAULT_FIGDIR, session_figures=True):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    sess_manager = load_session_manager(basedir, figdir)
    print(f'{sess_manager.sess_df["fly_id"].nunique()} flies, {len(sess_manager.sess_df)} sessions')
    tables = {}
    for analysis in (offset_analysis, rho_analysis, dphi_analysis, width_analysis):
        tables.update(analysis(sess_manager)[1])
    if session_figures:
        print(f'{len(plot_session_figures(sess_manager))} flies with session figures')
    nmp.utilities.save_tables({k: v for k, v in tables.items() if isinstance(v, (pd.DataFrame, str))},
                              sess_manager.fig_folder)
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
