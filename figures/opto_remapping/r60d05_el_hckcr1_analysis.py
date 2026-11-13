"""Optogenetic remapping with HcKCR1 in the EL (R60D05-LexA > jGCaMP7b, EL > CsChrimson, HcKCR1): analyses and figures.

Run ``python r60d05_el_hckcr1_analysis.py`` to regenerate every figure and statistics table, or import the module (as
``R60D05_remap_EL_hckcr1.ipynb`` does) and call the pieces one at a time. The analyses are the shared ones of
:mod:`el_hd_plasticity.two_photon.remapping`; this file says which sessions and groups to compare.
Groups: 'uas_control' and 'hckcr1'. The responses to the optogenetic stimulus are z-scored per fly and ROI.
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
    plot_aligned_opto,
    plot_min_dist_summary,
    plot_opto_by_location,
    plot_opto_heatmaps,
    plot_polar_summary,
    plot_remap_error,
    plot_vec_len_comparison,
)
 
BASEDIR = pathlib.Path('/media/mplitt/SSD_storage/2P_scratch/R60D05LexA_jGCaMP7b_CsChrimsontdT_EL_hcKCR1')
DEFAULT_FIGDIR = pathlib.Path('/media/mplitt/SSD_storage/fig_scratch/R60D05LexA_jGCaMP7b_CsChrimsontdT_EL_hcKCR1')

GROUPS = [Group('uas_control', 'UAS control', 'black'),
          Group('hckcr1', 'HcKCR1', 'green')]
REFERENCE = 'uas_control'  # group the other groups are compared with
ZERO_GROUPS = ('uas_control', 'hckcr1')  # groups whose first post-cue offset is unwrapped toward baseline
OPTO_ZSCORE = True
SESSION_VMAX = 1.5                              # color limit of the session heatmaps


def load_session_manager(basedir=BASEDIR, figdir=DEFAULT_FIGDIR):
    """Usable sessions of the groups of interest, with the condition in the fly id."""
    sess_manager = nmp.loading.SessionManager(basedir=basedir, figfolder=figdir, folder_cols=('condition',),
                                              id_cols=['condition'])
    df = sess_manager.sess_df
    sess_manager.filter_rows(df.index != df.index[-1])      # the last session of the table belongs to an unfinished fly
    return sess_manager


def opto_analysis(sess_manager):
    """Optogenetic responses: heatmaps by stimulus location, by stimulus location.

    Returns:
        Tuple of (responses aligned to the stimulated position, dict of statistics tables).
    """
    figdir = sess_manager.fig_folder
    opto = nmp.remapping.compute_opto_responses(sess_manager, GROUPS, zscore=OPTO_ZSCORE)
    plot_opto_heatmaps(figdir, opto, GROUPS)
    plot_opto_by_location(figdir, opto, GROUPS)
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
    return nmp.remapping.plot_session_figures(sess_manager.fig_folder, sess_manager, GROUPS, vmax=SESSION_VMAX)


def plot_pairing_scans(sess_manager, close=True):
    """Heatmap of the remap_0deg and remap_180deg (pairing) scans of every fly, with the VR angle and the stimulated
    wedge of each pairing (:func:`el_hd_plasticity.two_photon.remapping.plot_pairing_scans`).

    The table of how many of the programmed pairings each scan recorded is saved as ``<fig_folder>/stats/pairing_scans.csv``.

    Returns:
        DataFrame with one row per pairing scan.
    """
    summary = nmp.remapping.plot_pairing_scans(sess_manager.fig_folder, sess_manager, GROUPS, vmax=SESSION_VMAX, close=close)
    nmp.utilities.save_tables({'pairing_scans': summary}, sess_manager.fig_folder)
    return summary


# fly and VR-rig schematics cut from the Figure 1 artwork by figures/movie_assets/make_movie_assets.py
MOVIE_ASSETS = pathlib.Path(__file__).resolve().parent.parent / 'movie_assets'
PAIRING_SCANS = {0.: 'remap_0deg', np.pi: 'remap_180deg'}     # opto-paired offset -> scan type of its pairing scan


def make_remapping_schematic_movie(sess_manager, paired_offset=np.pi, outfile=None, snapshot_times=None, **kwargs):
    """Animated schematic of the remapping protocol (see :mod:`el_hd_plasticity.two_photon.remapping_movie`).

    The activity is simulated; the pairing schedule (angles, cycles, durations, step of the cue) is read from this
    experiment's first pairing scan with the given opto-paired offset, so the movie runs the protocol as recorded.

    Args:
        paired_offset: The opto-paired offset to show: pi (the 180 degree pairing, as in the figure panel) or 0.
        outfile: MP4 path; ``<fig_folder>/movies/remapping_schematic_<scan>.mp4`` by default.
        snapshot_times: If given, save PNG frames at these experiment times (s) next to `outfile` instead of the
            movie, to check the layout quickly.
        **kwargs: Passed to :class:`remapping_movie.RemappingSchematicMovie` (and on to the simulation).

    Returns:
        Path of the movie, or the list of snapshot paths.
    """
    scan = PAIRING_SCANS[min(PAIRING_SCANS, key=lambda offset: abs(np.angle(np.exp(1j * (offset - paired_offset)))))]
    row = sess_manager.sess_df.loc[sess_manager.sess_df['scan_type'] == scan].iloc[0]
    protocol = nmp.remapping_movie.PairingProtocol.from_session(sess_manager.load_row(row), paired_offset)
    movie = nmp.remapping_movie.RemappingSchematicMovie(protocol, fly_image=MOVIE_ASSETS / 'fly_topdown.png',
                                                        rig_image=MOVIE_ASSETS / 'vr_rig.png', **kwargs)
    if outfile is None:
        outfile = sess_manager.fig_folder / 'movies' / f'remapping_schematic_{scan}.mp4'
    outfile = pathlib.Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        if snapshot_times is not None:
            paths = [outfile.with_name(f'{outfile.stem}_t{t:05.1f}.png') for t in snapshot_times]
            for t, path in zip(snapshot_times, paths, strict=True):
                movie.snapshot(movie.frame_at(t), path)
            return paths, protocol
        return movie.save(outfile), protocol
    finally:
        plt.close(movie.fig)


REMAP_MOVIE_FLY = '27_11_2024_fly2_condition_uas_control'     # the control example of the remapping figure
REMAP_MOVIE_GAP = 15.                       # s of experiment between consecutive sessions in the movie
REMAP_MOVIE_SPEED = 5.                      # experiment seconds per movie second


def make_remapping_session_movie(sess_manager, fly_id=REMAP_MOVIE_FLY, gap_s=REMAP_MOVIE_GAP, speed=REMAP_MOVIE_SPEED,
                                 outfile=None, snapshot_times=None, **kwargs):
    """Movie of all of a fly's imaging sessions, appended in the order acquired (baseline, the pairing scans and the
    post-pairing scans), `gap_s` seconds of experiment apart (see :func:`remapping_movie.remap_sequence_movie`).

    The heatmap and ellipsoid body change colormap at the first stimulation of each pairing scan; each colormap's
    offset histogram is that of the following closed-loop scan. Every stimulation is drawn as a laser cone at the
    stimulated wedge; the arena goes dark while the screen is off.

    Returns:
        Tuple of (path of the movie, or the list of snapshot paths; the :class:`remapping_movie.SessionSequence`).
    """
    if outfile is None:
        outfile = sess_manager.fig_folder / 'movies' / f'{fly_id}_all_sessions.mp4'
    return nmp.remapping_movie.remap_sequence_movie(
        sess_manager, fly_id, outfile, vmax=SESSION_VMAX, gap_s=gap_s, speed=speed, snapshot_times=snapshot_times,
        fly_image=MOVIE_ASSETS / 'fly_topdown.png', rig_image=MOVIE_ASSETS / 'vr_rig.png', **kwargs)


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
