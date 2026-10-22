"""Ex vivo GFlamp responses of ER and EPG neurons to EL stimulation (R18B05-LexA > CsChrimson): analyses and figures.

Run ``python er_epg_gflamp_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``ER_EPG_Gflamp_R18B05LexA_CsChrimsontdT.ipynb`` does) and call the pieces one at a time. The EL neurons are
stimulated optogenetically while GFlamp in the ER or EPG neurons (``cell_type``) is imaged; the responses are summarized
by :mod:`el_hd_plasticity.two_photon.ex_vivo`. Groups: 'exp' (EL expresses CsChrimson) and 'nocschr_control'.
"""

import argparse
import pathlib

import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as nmp
from el_hd_plasticity.two_photon.group_comparison import Group

BASEDIRS = {
    'ER': pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R56H10_R20A02_split_GFlamp2_R18B05LexA_CsChrimsontdT'),
    'EPG': pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R60D05Gal4_GFlamp2_R18B05LexA_CsChrimsontdT'),
}
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/ex_vivo_gflamp')

# groups compared within each cell type
GROUPS = {
    'ER': (Group('exp', 'ER GFlamp, EL CsChrimson', 'magenta'),
           Group('nocschr_control', 'ER GFlamp, EL no CsChrimson control', plt.cm.Greys(.6), '--', '//')),
    'EPG': (Group('exp', 'EPG GFlamp, EL CsChrimson', 'black'),
            Group('nocschr_control', 'EPG GFlamp, EL no CsChrimson control', plt.cm.Greys(.6), '--', '//')),
}
MAX_DFF_YLIM = (0, .5)


def load_session_managers(basedirs=BASEDIRS, figdir=DEFAULT_FIGDIR):
    """Session managers of the preparations of each cell type, with the cell type and condition in the fly id."""
    return {cell_type: nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, folder_cols=('condition',),
                                                  id_cols=('condition', 'cell_type'),
                                                  extra_cols={'cell_type': cell_type})
            for cell_type, basedir in basedirs.items()}


def compute_responses(session_managers):
    """Response summaries of every session: see :func:`el_hd_plasticity.two_photon.ex_vivo.session_responses`."""
    return nmp.ex_vivo.session_responses(session_managers)


def plot_mean_dff(figdir, responses, plot_time, groups=GROUPS):
    """Mean +- SEM response to the EL stimulus of each group, one panel per cell type."""
    fig, axs = plt.subplots(1, len(groups), figsize=[4 * len(groups), 4], sharey=True, sharex=True, squeeze=False)
    for ax, (cell_type, cell_groups) in zip(axs[0], groups.items(), strict=True):
        nmp.plotting.despine(ax)
        for group in cell_groups:
            sel = responses.loc[(responses['cell_type'] == cell_type) & (responses['condition'] == group.value)]
            dff = np.vstack(sel['mean_dff'])
            mu, sem = dff.mean(axis=0), sp.stats.sem(dff, axis=0)
            ax.plot(plot_time, mu, color=group.color, linestyle=group.linestyle, linewidth=2.5,
                    label=f'{group.label} (n={len(sel)})')
            ax.fill_between(plot_time, mu - sem, mu + sem, color=group.color, alpha=.3, hatch=group.hatch)
        ax.set_xlabel('time (s)')
    axs[0, 0].set_ylabel('dF/F')
    fig.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / 'mean_dff.pdf', bbox_inches='tight')
    return fig


def plot_max_dff(figdir, responses, groups=GROUPS, ylim=MAX_DFF_YLIM):
    """Peak response of every session (points) and the mean (bar), by cell type and group."""
    fig, ax = plt.subplots(figsize=[4, 4])
    for cell_type, cell_groups in groups.items():
        kw = {'data': responses.loc[responses['cell_type'] == cell_type], 'x': 'cell_type', 'y': 'max_dff',
              'hue': 'condition', 'hue_order': [g.value for g in cell_groups],
              'palette': {g.value: g.color for g in cell_groups}}
        sns.stripplot(**kw, ax=ax, alpha=.5, dodge=True, s=8, legend=False)
        sns.pointplot(**kw, ax=ax, estimator=np.mean, errorbar=None, marker='_', markersize=20,
                      markeredgewidth=6, dodge=.4, linestyle='none', legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('peak dF/F')
    ax.set_ylim(ylim)
    nmp.plotting.despine(ax)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / 'max_dff.pdf', bbox_inches='tight')
    return fig


def max_dff_tests(responses):
    """Mann-Whitney tests of the peak response: experimental vs control within each cell type, and ER vs EPG.

    p-values are Holm-corrected across the three comparisons.
    """
    def peaks(cell_type, condition):
        return responses.loc[(responses['cell_type'] == cell_type) & (responses['condition'] == condition), 'max_dff']

    comparisons = {'ER: experimental vs control': (peaks('ER', 'exp'), peaks('ER', 'nocschr_control')),
                   'EPG: experimental vs control': (peaks('EPG', 'exp'), peaks('EPG', 'nocschr_control')),
                   'ER vs EPG (experimental)': (peaks('ER', 'exp'), peaks('EPG', 'exp'))}
    rows = []
    for name, (x, y) in comparisons.items():
        u, p = sp.stats.mannwhitneyu(x, y)
        rows.append({'comparison': name, 'n': f'{len(x)}, {len(y)}', 'U': float(u), 'p': float(p)})
    return nmp.stats.add_holm(pd.DataFrame(rows))


def main(basedirs=BASEDIRS, figdir=DEFAULT_FIGDIR):
    """Generate all figures and write the statistics table to ``<figdir>/stats``."""
    plt.rcParams['pdf.fonttype'] = 42
    figdir = pathlib.Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    responses, plot_time = compute_responses(load_session_managers(basedirs, figdir))
    print(responses.groupby(['cell_type', 'condition'])['fly_id'].agg(['size', 'nunique']).rename(
        columns={'size': 'sessions', 'nunique': 'flies'}))
    plot_mean_dff(figdir, responses, plot_time)
    plot_max_dff(figdir, responses)
    tables = {'max_dff_tests': max_dff_tests(responses)}
    nmp.utilities.save_tables(tables, figdir)
    print(tables['max_dff_tests'])
    return tables


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--figdir', type=pathlib.Path, default=DEFAULT_FIGDIR, help='where figures and tables are saved')
    args = parser.parse_args()
    main(figdir=args.figdir)
