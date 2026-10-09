"""Analysis and figures for optogenetic remapping experiments.

In these experiments the EPG bump offset is measured at ``baseline`` and again after the
optogenetic stimulus was paired with a visual cue shifted by 0 or 180 degrees (``post_0deg``,
``post_180deg``; or ``post_novis1``/``post_novis2`` when no visual cue was shown). The session
table has one row per scan with its ``scan_type``; per-scan offset summaries ('mean offset',
'mean vec. len', 'offset var.') are averaged per fly with :func:`build_sum_stats` and compared
with :func:`avg_diff` (remapping error) and the figure helpers below.

Figures take ``ts_dict``: a dict with a ``'fly'`` name and :class:`session.SessionTimeseries` objects for
``'baseline'``, ``'post_0deg'`` and ``'post_180deg'``.
"""

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

from . import plotting
from .utilities import (
    DEFAULT_OFFSET_BINS,
    DEFAULT_PLOT_TIMES,
    SCAN_POSITIONS,
    SHORT_PLOT_TIMES,
    nearest_equivalent_angle,
)

STANDARD_SCAN_TYPES = ('baseline', 'post_0deg', 'post_180deg')
NOVIS_SCAN_TYPES = ('baseline', 'post_novis1', 'post_novis2')
ALL_SCAN_TYPES = ('baseline', 'post_0deg', 'post_180deg', 'post_novis1', 'post_novis2')

# Summary statistic name used in sum_stats columns -> per-scan column of the session table
SUMMARY_METRICS = {'mean offset': 'mean offset', 'vec. len': 'mean vec. len', 'offset var.': 'offset var.'}

# Heatmap/histogram colormap of each scan type
SCAN_CMAPS = {'baseline': 'Greys', 'post_0deg': 'PuRd', 'post_180deg': 'GnBu'}


# ----------------------------------------------------------------------------
# Optogenetic responses
# ----------------------------------------------------------------------------
def get_opto_resp(pp, zscore=False):
    """Extract per-ROI optogenetic responses aligned to stimulus offsets.

    Scans the voltage recording for opto trigger onsets, masks the immediate
    frames around triggers, computes dF/F (z-scored) and returns the mean
    response for a short window after each opto offset organized as
    (n_rois, n_points, n_reps).

    NOTE: the masked frames are set to NaN in ``pp.timeseries['rois']`` in place.

    Args:
        pp: Preprocessed session object with attributes `voltage_recording_aligned`,
            `metadata`, and `timeseries` used below.
        zscore: If True, z-score the resulting opto responses across all values.

    Returns:
        opto_resp: ndarray shaped (n_rois, n_points, n_reps) containing mean responses.
    """
    opto_trig = pp.voltage_recording_aligned[' Opto Trigger']

    # find rising edges of opto trigger (>1000 indicates trigger)
    opto_trig_bool = np.concatenate([[0], np.diff((opto_trig > 1000).astype(int)) > 0])
    opto_inds = np.flatnonzero(opto_trig_bool)

    # frame timing and offset calculation
    volume_dt = np.diff(np.array(pp.metadata['frame_times']), axis=0)[0, 0]
    n_frames = int(pp.metadata['mark_points']['duration'] / 1000 / volume_dt)
    opto_off_inds = opto_inds + n_frames - 1

    # mask data around opto onsets (avoid contamination) and initial frames
    for ind in opto_inds:
        pp.timeseries['rois'][:, :, ind:ind + n_frames] = np.nan
    pp.timeseries['rois'][:, :, :int(2 / volume_dt)] = np.nan

    # compute z-scored dF/F for ROIs (last returned array assumed to be per-frame)
    dff = pp.calculate_zscored_F('rois', exp_detrend=False, zscore=True, background_ts='background')[-1, :, :]

    # expected grid of points x reps (hard-coded in upstream pipeline)
    n_rois = dff.shape[0]
    n_points = 8
    n_reps = 5

    assert n_points * n_reps == len(opto_off_inds), "wrong number of opto inds"

    # reshape indices to (n_reps, n_points) to make indexing explicit
    inds_matrix = opto_off_inds.reshape((n_reps, n_points))
    opto_resp = np.zeros((n_rois, n_points, n_reps))

    for rep in range(n_reps):
        for point in range(n_points):
            ind = inds_matrix[rep, point]
            # average a short window after offset (3 frames)
            opto_resp[:, point, rep] = np.nanmean(dff[:, ind:ind + 3], axis=-1)

    if zscore:
        # zscore across all values per ROI
        for r in range(n_rois):
            opto_resp[r, :, :] = sp.stats.zscore(opto_resp[r, :, :], axis=None)
    return opto_resp


def stack_opto_responses(df, load_row, condition, folder, scan_type):
    """Stack z-scored opto responses of all sessions of a condition and scan type.

    Args:
        df: Session table with condition, scan_type and folder columns.
        load_row: Callable loading a preprocessed session from a table row.
        condition: Value of the ``condition`` column to select.
        folder: Value of the ``folder`` column to select.
        scan_type: Value of the ``scan_type`` column to select.

    Returns:
        Array of shape (n_sessions, n_rois, n_points, n_reps); see :func:`get_opto_resp`.
    """
    sel = df.loc[(df['condition'] == condition) & (df['scan_type'] == scan_type) & (df['folder'] == folder)]
    return np.stack([get_opto_resp(load_row(row), zscore=True) for _, row in sel.iterrows()])


def get_mu_sem(opto_resp_al, condition, scan_type):
    """Mean and SEM (across sessions) of the aligned opto response for a condition and scan type.

    Args:
        opto_resp_al: Nested dict ``opto_resp_al[condition][scan_type]`` of arrays shaped
            (n_sessions, n_rois, n_points, n_reps).
        condition: Key of the first level.
        scan_type: Key of the second level.

    Returns:
        Tuple (mu, sem), each of length n_rois.
    """
    resp = opto_resp_al[condition][scan_type].mean(axis=(2, 3))
    mu = resp.mean(axis=0)
    sem = sp.stats.sem(resp, axis=0)
    return mu, sem


# ----------------------------------------------------------------------------
# Per-fly offset summaries
# ----------------------------------------------------------------------------
def load_sess_df(basedir, folder):
    """Load ``<basedir>/sessions.csv`` and add the columns used by the remapping analyses.

    Args:
        basedir: pathlib.Path of the experiment folder containing sessions.csv.
        folder: Label of this experiment folder, used to tell pooled experiments apart.

    Returns:
        Session DataFrame with a unique ``fly_id`` (date, fly, condition, folder), empty
        'mean offset' / 'offset var.' / 'mean vec. len' columns to be filled per scan, and
        ``folder`` and ``basedir`` columns.
    """
    sess_df = pd.read_csv(basedir / 'sessions.csv')

    sess_df['fly_id'] = [row['date'] + '_' + row['fly'] + '_' + row['condition'] + '_' + folder
                         for _, row in sess_df.iterrows()]
    sess_df['mean offset'] = np.nan
    sess_df['offset var.'] = np.nan
    sess_df['mean vec. len'] = np.nan
    sess_df['folder'] = folder
    sess_df['basedir'] = str(basedir)
    return sess_df


def build_sum_stats(df, fly_list, conditions=None, scan_types=None, extra_metrics=()):
    """Average the per-scan offset summaries of each fly, one column per scan type and statistic.

    Args:
        df: Session table with fly_id, scan_type and the per-scan summary columns of
            :data:`SUMMARY_METRICS` (plus any `extra_metrics`).
        fly_list: fly_id of each fly to include.
        conditions: Optional condition label for each fly (same length as `fly_list`); adds a
            ``condition`` column.
        scan_types: Scan types to summarize. If None, the standard set (baseline, post_0deg,
            post_180deg) is used, or the no-visual-cue set (baseline, post_novis1, post_novis2)
            if the first fly_id contains 'novis'. Scan types a fly has no sessions for are NaN.
        extra_metrics: Additional per-scan columns of `df` to average (e.g. ``('post_dark',)``).

    Returns:
        DataFrame with fly_id, [condition] and columns '<scan_type> mean offset',
        '<scan_type> vec. len', '<scan_type> offset var.' (and one per extra metric).
    """
    fly_list = list(fly_list)
    if scan_types is None:
        scan_types = NOVIS_SCAN_TYPES if 'novis' in fly_list[0] else STANDARD_SCAN_TYPES
    metrics = {**SUMMARY_METRICS, **{m: m for m in extra_metrics}}

    columns = ['fly_id'] + (['condition'] if conditions is not None else [])
    columns += [f'{s} {name}' for s in scan_types for name in metrics]

    rows = []
    for i, fly in enumerate(fly_list):
        row = {'fly_id': fly}
        if conditions is not None:
            row['condition'] = conditions[i]
        fly_df = df.loc[df['fly_id'] == fly]
        for s in scan_types:
            scan_df = fly_df.loc[fly_df['scan_type'] == s]
            for name, col in metrics.items():
                row[f'{s} {name}'] = scan_df[col].mean()
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def avg_diff(sum_stats, novis=None):
    """Average remapping error: distance of the post-0deg/180deg offsets from 0 and pi.

    For each fly, the circular distance of the offset after the 0 degree cue from 0, and of the
    offset after the 180 degree cue from pi, averaged. 0 is perfect remapping; pi/2 is chance.

    Args:
        sum_stats: Output of :func:`build_sum_stats` (or a subset of its rows).
        novis: If True use the no-visual-cue scans (post_novis1/post_novis2) instead of
            post_0deg/post_180deg. If None, decided by whether `sum_stats` has no-visual-cue
            columns; pass explicitly when the table contains both sets (e.g. NaN filled).

    Returns:
        Series of average errors in radians, indexed like `sum_stats`.
    """
    if novis is None:
        novis = 'post_novis1 mean offset' in sum_stats.columns
    scan0, scan180 = ('post_novis1', 'post_novis2') if novis else ('post_0deg', 'post_180deg')

    diff0 = np.abs(sum_stats[f'{scan0} mean offset'])
    diff0 = np.minimum(diff0, 2 * np.pi - diff0)

    diff180 = np.abs(np.pi - sum_stats[f'{scan180} mean offset'])
    diff180 = np.minimum(diff180, 2 * np.pi - diff180)

    return (diff0 + diff180) / 2


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_min_dist(df, h_ax, x=SCAN_POSITIONS, color='black', zero=False, novis=None):
    """Plot each fly's offset at baseline and after the two cues, as lines across scan types.

    Post-cue offsets are shifted by multiples of 2pi to lie closest to the baseline offset, so a
    fly that rotates by a small amount does not appear to jump across the wrap-around.

    Args:
        df: Output of :func:`build_sum_stats`.
        h_ax: Axes to draw on.
        x: x positions of the three scan types.
        color: Line/marker color.
        zero: Also unwrap the first post-cue offset toward baseline (default: plot as is).
        novis: Plot post_novis1/post_novis2 instead of post_0deg/post_180deg. If None, decided by
            whether `df` has no-visual-cue columns.
    """
    if novis is None:
        novis = 'post_novis1 mean offset' in df.columns
    scan0, scan180 = ('post_novis1', 'post_novis2') if novis else ('post_0deg', 'post_180deg')

    for _, row in df.iterrows():
        b = row['baseline mean offset']

        p0 = row[f'{scan0} mean offset']
        if zero:
            p0 = nearest_equivalent_angle(p0, b)
        p180 = nearest_equivalent_angle(row[f'{scan180} mean offset'], b)

        h_ax.scatter(x, [b, p0, p180], color=color)
        h_ax.plot(x, [b, p0, p180], color=color, alpha=.2)


def plot_row_sum_stats(sum_stats, ax, row, condition, novis=False):
    """Draw each fly's mean offset vector (and the across-fly mean vector) on a row of polar axes.

    Baseline is drawn on ``ax[row, 0]`` and the two post-cue scans on ``ax[row, 1]``. For the
    standard scan types the circular mean offset after each cue is also printed.

    Args:
        sum_stats: Output of :func:`build_sum_stats` with a ``condition`` column.
        ax: 2D array of polar axes.
        row: Row of `ax` to draw on.
        condition: Value of the ``condition`` column to plot (also used as title).
        novis: Draw post_novis1/post_novis2 instead of post_0deg/post_180deg.
    """
    stats = sum_stats.loc[sum_stats['condition'] == condition].copy()
    if novis:
        post = {'post_novis1': plt.cm.PuOr(.2), 'post_novis2': plt.cm.PuOr(.8)}
    else:
        post = {'post_0deg': plt.cm.PuRd(.8), 'post_180deg': plt.cm.GnBu(.8)}
    remap_label = {'post_0deg': 'remap 0', 'post_180deg': 'remap 180'}

    def mean_vector(scan):
        """Across-fly mean of the offset vectors (vec. len, mean offset) of one scan type."""
        offset = stats[f'{scan} mean offset']
        v = stats[f'{scan} vec. len'] * (np.cos(offset) + 1j * np.sin(offset))
        return v.mean()

    for _, r in stats.iterrows():
        ax[row, 0].plot(r['baseline mean offset'] * np.ones([2, ]), [0, r['baseline vec. len']],
                        color=plt.cm.Greys(.8), alpha=.4, linewidth=2)
        for scan, color in post.items():
            ax[row, 1].plot(r[f'{scan} mean offset'] * np.ones([2, ]), [0, r[f'{scan} vec. len']],
                            color=color, alpha=.4, linewidth=2)

    baseline_v = mean_vector('baseline')
    ax[row, 0].plot(np.angle(baseline_v) * np.ones((2,)), np.absolute(baseline_v) * np.array([0, 1]),
                    color='black', linewidth=4)

    for scan, color in post.items():
        v = mean_vector(scan)
        ax[row, 1].plot(np.angle(v) * np.ones((2,)), np.absolute(v) * np.array([0, 1]),
                        color=color, linewidth=4, label=scan)
        if not novis:
            print(f"{condition} {remap_label[scan]} {np.angle(v)}")

    ax[row, 0].set_title(condition)
    ax[row, 1].set_title(condition)

    for col in range(2):
        ax[row, col].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
        ax[row, col].set_yticks([0, .2, .4, .6, .8])
        ax[row, col].set_ylim(0, .8)

    ax[row, 1].legend()


def _scan_traces(ts_dict):
    """One :class:`plotting.Trace` per scan type (baseline, post_0deg, post_180deg)."""
    return [plotting.Trace.from_ts(key, ts_dict[key], cmap) for key, cmap in SCAN_CMAPS.items()]


def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=SHORT_PLOT_TIMES):
    """Plot per-session heatmaps aligned to time and heading.

    Args:
        ts_dict: Dict with keys 'baseline', 'post_0deg', 'post_180deg' each with
                 attributes dff, time, heading, dff_h_aligned.
        vmin, vmax: Color limits for heatmap.
        plot_times: Time points (in seconds) to mark on x-axis.

    Returns:
        fig, ax: Matplotlib figure and axes.
    """
    return plotting.plot_heatmap_grid(_scan_traces(ts_dict), ts_dict['fly'], [15, 6], vmin, vmax,
                                      plot_times, clip_times=False)


def plot_sess_histograms(ts_dict, bins=DEFAULT_OFFSET_BINS):
    """Offset histogram and polar mean-vector plot for baseline, post_0deg and post_180deg.

    Args:
        ts_dict: Dict with 'fly' and the timeseries objects of :func:`plot_sess_heatmaps`.
        bins: Bins for the histogram.

    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    traces = _scan_traces(ts_dict)
    traces[0].color = 'black'  # baseline; post-cue scans use their colormap
    return plotting.plot_offset_histograms(traces, bins, hist_title=ts_dict['fly'],
                                           polar_title=ts_dict['fly'], polar_ylim=[0, .8])


def plot_sess_heatmaps_w_hist(ts_dict, bins=DEFAULT_OFFSET_BINS, vmin=-.5, vmax=3,
                              plot_times=DEFAULT_PLOT_TIMES, twindow=None, sess_order=(0, 1, 2)):
    """Heatmaps of baseline, post_0deg and post_180deg, each with its offset distribution.

    Args:
        ts_dict: Dict with 'fly' and the timeseries objects of :func:`plot_sess_heatmaps`.
        bins: Bins for the offset histogram.
        vmin, vmax: Color limits for heatmap.
        plot_times: Time points (in seconds) to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        sess_order: Row of the figure used for (baseline, post_0deg, post_180deg).

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    return plotting.plot_heatmaps_with_offset_hist(
        _scan_traces(ts_dict), ts_dict['fly'], figsize=[8, 6], vmin=vmin, vmax=vmax,
        plot_times=plot_times, twindow=twindow, bins=bins, slots=sess_order,
        share_axes=True, hist_xlim_zero=True)
