"""Oamb knockout (heterozygous vs knockout, R27F02-LexA > jGCaMP8s): analyses and figures.

Run ``python oamb_ko_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``global_ko.ipynb`` does) and call the pieces one at a time. The analyses are the shared ones of
:mod:`el_hd_plasticity.two_photon.group_comparison`; this file says which sessions and groups to compare.
Groups: 'het' (heterozygous control) and 'ko'. All closed-loop blocks count as closed loop; sessions are not
excluded from the speed analyses.
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/Ti{RFP}Ti_Oamb_R27F02LexA_jGCaMP8s')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/Oamb_KO')

GROUP_COL = 'condition'
GROUPS = [Group('het', 'heterozygous', 'black'),
          Group('ko', 'KO', plt.cm.Greys(.6), '--', '//')]
REFERENCE = 'het'                                 # baseline group of the PVA velocity regression
MIN_CL = nmp.oa_receptor_ko.MIN_CL                # closed-loop blocks of experience for a session to count as closed loop
RHO_BINS = np.linspace(0, 3.14, num=20)           # bins of absolute rotation speed (rad/s)
DPHI_BINS = np.linspace(-1, 1, num=20)            # bins of signed rotation speed (rad/s)
WIDTH_CL_BLOCKS = (1,)                            # closed-loop blocks used for the bump width
SESSION_TWINDOW = (0, 180.1)                      # seconds of the session heatmaps


def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Usable sessions, with the condition in the fly id."""
    return nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, folder_cols=(GROUP_COL,), id_cols=[GROUP_COL])


def offset_analysis(sess_manager):
    """Offset vector length and variance, dark vs closed loop.

    Returns:
        Tuple of (per-fly offset table, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    unique = nmp.group_comparison.compute_offset_stats(sess_manager, GROUP_COL, MIN_CL)
    nmp.group_comparison.plot_offset_mag(figdir, unique, GROUP_COL, GROUPS, ylims=[-.05,1.05])
    tables = nmp.group_comparison.offset_var_tests(figdir, unique, GROUP_COL)
    tables['offset_n_flies'] = unique.groupby(GROUP_COL)['fly_id'].nunique().rename('n_flies').reset_index()
    return unique, tables


def rho_analysis(sess_manager):
    """PV magnitude vs rotation speed: curves and tests of the mean over speed."""
    figdir = sess_manager.fig_folder
    grouped = nmp.group_comparison.compute_binned_stats(sess_manager, GROUP_COL, 'rho', RHO_BINS, exclude_first_cl=False)
    nmp.group_comparison.plot_binned_vs_speed(figdir, grouped, GROUP_COL, GROUPS, 'rho', RHO_BINS,
                                              'PVMag_v_RotVel.pdf', 'PV magnitude')
    tables = {'rho_mean_over_speed_tests': nmp.group_comparison.mean_over_bins_tests(grouped, GROUP_COL, GROUPS)}
    return grouped, tables


def dphi_analysis(sess_manager):
    """PVA velocity vs rotation speed: curves and regression tests of the slope."""
    figdir = sess_manager.fig_folder
    grouped = nmp.group_comparison.compute_binned_stats(sess_manager, GROUP_COL, 'dphi', DPHI_BINS, exclude_first_cl=False)
    nmp.group_comparison.plot_binned_vs_speed(figdir, grouped, GROUP_COL, GROUPS, 'dphi', DPHI_BINS,
                                              'dPVA_v_RotVel.pdf', 'PVA Vel. (rad/s)', xlim=(-1, 1), ylim=(-1, 1))
    return grouped, nmp.group_comparison.dphi_regression_tests(grouped, GROUP_COL, GROUPS, REFERENCE, DPHI_BINS)


def width_analysis(sess_manager):
    """Bump width, dark vs closed loop, per group: one panel per group, and the change with the groups side by side."""
    widths = nmp.group_comparison.compute_widths(sess_manager, GROUP_COL, WIDTH_CL_BLOCKS)
    nmp.group_comparison.plot_widths(sess_manager.fig_folder, widths, GROUP_COL, GROUPS, ylims = (np.pi/2, np.pi+.1))
    nmp.group_comparison.plot_width_change(sess_manager.fig_folder, widths, GROUP_COL, GROUPS)
    return widths, {'bump_width_tests': nmp.group_comparison.width_tests(widths, GROUP_COL, GROUPS)}


def plot_session_figures(sess_manager):
    """Closed-loop and dark heatmaps with offset histograms of every fly."""
    return nmp.group_comparison.plot_session_figures(sess_manager.fig_folder, sess_manager, min_closed_loop=MIN_CL, twindow=SESSION_TWINDOW, vmax=1.5)


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
