"""OctAlpha2R RNAi (Dh31-Gal4 > OctAlpha2R RNAi, R27F02-LexA > jGCaMP7b): analyses and figures.

Run ``python octalpha2r_rnai_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``octalpha2r_rnai.ipynb`` does) and call the pieces one at a time. The analyses are the shared ones of
:mod:`el_hd_plasticity.two_photon.group_comparison`; this file says which sessions and groups to compare.
Groups: 'gal4_only', 'rnai_only' and 'rnai'. Only the offset analyses were done for this dataset.
"""

import argparse
import pathlib

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group

BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/Dh31Gal4_OctAlpha2R_RNAi_R27F02LexA_jGCaMP7b')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/Dh31Gal4_OctAlpha2R_RNAi_R27F02LexA_jGCaMP7b')

GROUP_COL = 'condition'
GROUPS = [Group('gal4_only', 'Gal4 only', 'black'),
          Group('rnai_only', 'RNAi only', plt.cm.Greys(.6), '--', '//'),
          Group('rnai', 'OctAlpha2R RNAi', 'green')]
MIN_CL = nmp.oa_receptor_ko.MIN_CL                # closed-loop blocks of experience for a session to count as closed loop


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
    nmp.group_comparison.plot_offset_mag(figdir, unique, GROUP_COL, GROUPS)
    tables = nmp.group_comparison.offset_var_tests(figdir, unique, GROUP_COL)
    tables['offset_n_flies'] = unique.groupby(GROUP_COL)['fly_id'].nunique().rename('n_flies').reset_index()
    return unique, tables


def main(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Generate all figures and write the statistics tables to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    sess_manager = load_session_manager(basedir, figdir)
    print(f'{sess_manager.sess_df["fly_id"].nunique()} flies, {len(sess_manager.sess_df)} sessions')
    tables = {}
    for analysis in (offset_analysis,):
        tables.update(analysis(sess_manager)[1])
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
    args = parser.parse_args()
    main(args.basedir, args.figdir)
