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

import pathlib

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt

from . import plotting, session, stats, utilities
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
def opto_onsets(pp):
    """Frame indices of the optogenetic stimulations of a scan: rising edges of the opto trigger (> 1000)."""
    trigger = pp.voltage_recording_aligned[' Opto Trigger'].to_numpy()
    return np.flatnonzero(np.concatenate([[0], np.diff((trigger > 1000).astype(int)) > 0]))


def masked_pairing_activity(pp):
    """Z-scored dF/F of a pairing scan with the frames of every stimulation, and the first 2 s, masked.

    The frames recorded during each stimulation (its mark-points ``duration``) are set to NaN before dF/F is
    computed, without exponential detrending, so only the imaging between stimulations remains.

    NOTE: the masked frames are set to NaN in ``pp.timeseries['rois']`` in place.

    Returns:
        Tuple of (stimulation onset frames, frames per stimulation, dF/F of the last channel (n_rois, n_frames)).
    """
    onsets = opto_onsets(pp)
    volume_dt = np.diff(np.array(pp.metadata['frame_times']), axis=0)[0, 0]
    n_frames = int(pp.metadata['mark_points']['duration'] / 1000 / volume_dt)
    for ind in onsets:
        pp.timeseries['rois'][:, :, ind:ind + n_frames] = np.nan
    pp.timeseries['rois'][:, :, :int(2 / volume_dt)] = np.nan
    dff = pp.calculate_zscored_F('rois', exp_detrend=False, zscore=True, background_ts='background')[-1, :, :]
    return onsets, n_frames, dff


def programmed_stimulations(pp):
    """Number of stimulations the scan's mark-points protocol was set up to deliver.

    Stimulation points that occur only once in the point list are placeholders (an unrepeated, zero-power first point)
    and are not counted.
    """
    points = [int(point['Index']) for point in np.atleast_1d(pp.metadata['mark_points']['points_list'])]
    counts = {index: points.count(index) for index in set(points)}
    return sum(count for count in counts.values() if count > 1)


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
    opto_inds, n_frames, dff = masked_pairing_activity(pp)
    opto_off_inds = opto_inds + n_frames - 1

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


def opto_responses_by_fly(session_manager, condition, scan_type, zscore=False):
    """Per-fly optogenetic responses of one condition and scan type.

    Args:
        session_manager: Session manager object whose session table has ``condition`` and ``scan_type`` columns.
        condition: Value of the ``condition`` column to select.
        scan_type: Value of the ``scan_type`` column to select (e.g. 'remap_0deg').
        zscore: Passed to :func:`get_opto_resp`.

    Returns:
        Dict fly_id -> array of shape (n_rois, n_points, n_reps), averaged over the fly's sessions.
    """
    df = session_manager.sess_df
    sel = df.loc[(df['condition'] == condition) & (df['scan_type'] == scan_type)]
    responses = {}
    for fly, fly_df in sel.groupby('fly_id', sort=False):
        responses[fly] = np.mean([get_opto_resp(session_manager.load_row(row), zscore=zscore)
                                  for _, row in fly_df.iterrows()], axis=0)
    return responses


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
#  ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Experiment-level analyses (shared by the remapping figure scripts)
# ----------------------------------------------------------------------------
# Groups are :class:`group_comparison.Group` objects whose ``value`` is the ``condition`` of the session table; groups
# whose value contains 'novis' were not shown a visual cue (post_novis1 / post_novis2 scans, remap_novis opto scans).
OPTO_SCANS = ('remap_0deg', 'remap_180deg')
NOVIS_OPTO_SCAN = 'remap_novis'
OFFSET_SCAN_TYPES = ('baseline', 'post_0deg', 'post_180deg', 'post_novis1', 'post_novis2')
OPTO_PANELS = (('remap_0deg', 'PuRd', 'remap 0'), ('remap_180deg', 'GnBu', 'remap pi'), ('combined', 'Greys', 'combined'))
N_STIM_POINTS = 8


def is_novis(group):
    """True for groups that were not shown a visual cue."""
    return 'novis' in group.value


class _Rows:
    """The rows of a session manager selected by a boolean mask (enough of its interface for the statistics)."""

    def __init__(self, session_manager, mask):
        self._df = session_manager.sess_df.loc[mask]
        self.load_row = session_manager.load_row

    def iterrows(self):
        return self._df.iterrows()


def session_offset_summary(session_manager, scan_types=OFFSET_SCAN_TYPES,
                           meta_cols=('fly_id', 'condition', 'scan_type')):
    """Per-session offset summary of the scans without optogenetic stimulation.

    Args:
        scan_types: Scan types summarized.
        meta_cols: Session-table columns copied to the result. Add one to carry it into
            :func:`build_sum_stats` as an extra metric (e.g. ``'post_dark'``).

    Returns:
        DataFrame with the `meta_cols` and the columns 'mean offset' (circular mean of the offset),
        'mean vec. len' and 'offset var.' used by :func:`build_sum_stats`.
    """
    rows = _Rows(session_manager, session_manager.sess_df['scan_type'].isin(scan_types))
    summary = stats.add_offset_var(stats.offset_stats(rows, meta_cols=meta_cols))
    return summary.drop(columns='offset_var_logit').rename(columns={
        'offset_mean': 'mean offset', 'offset_mag': 'mean vec. len', 'offset_var': 'offset var.'})


def sum_stats_by_group(summary, groups):
    """Per-fly summaries (:func:`build_sum_stats`) of each group: ``{group value: DataFrame}``."""
    out = {}
    for group in groups:
        flies = summary.loc[summary['condition'] == group.value, 'fly_id'].unique()
        scan_types = NOVIS_SCAN_TYPES if is_novis(group) else STANDARD_SCAN_TYPES
        out[group.value] = build_sum_stats(summary, flies, conditions=[group.value] * len(flies), scan_types=scan_types)
    return out


def compute_opto_responses(session_manager, groups, zscore=False):
    """Optogenetic responses of every group, averaged per fly: ``{group value: {scan: array}}``.

    Arrays have shape (n_flies, n_rois, n_points, n_reps). Groups shown a visual cue have 'remap_0deg', 'remap_180deg'
    and 'combined' (the mean of the two, for flies with both); no-visual-cue groups have 'remap_novis' and
    'combined' (the same array).
    """
    out = {}
    for group in groups:
        scans = (NOVIS_OPTO_SCAN,) if is_novis(group) else OPTO_SCANS
        per_fly = {scan: opto_responses_by_fly(session_manager, group.value, scan, zscore) for scan in scans}
        resp = {scan: np.stack(list(flies.values())) for scan, flies in per_fly.items()}
        if len(scans) == 2:
            both = [fly for fly in per_fly[scans[0]] if fly in per_fly[scans[1]]]
            resp['combined'] = np.stack([(per_fly[scans[0]][fly] + per_fly[scans[1]][fly]) / 2 for fly in both])
        else:
            resp['combined'] = resp[scans[0]]
        out[group.value] = resp
    return out


def align_opto_responses(opto):
    """Roll the responses to each stimulus location so that the stimulated position is at the middle ROI."""
    aligned = {}
    for group, scans in opto.items():
        aligned[group] = {}
        for scan, arr in scans.items():
            arr = np.copy(arr)
            for i in range(N_STIM_POINTS):
                arr[:, :, i, :] = np.roll(arr[:, :, i, :], 2 * i - 8, axis=1)
            aligned[group][scan] = arr
    return aligned


def _style_resp_axis(ax):
    plotting.despine(ax)
    ax.set_ylabel('Z-scored F')
    ax.set_xlabel('ROIs')
    ax.set_xticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])


def plot_opto_heatmaps(figdir, opto, groups, filename='opto_resp_heatmaps.pdf'):
    """Mean optogenetic response (ROIs x stimulus location) of each group: remap 0 | remap pi | combined."""
    fig, axs = plt.subplots(len(groups), 3, figsize=[12, 4.5 * len(groups)], sharex=True, sharey=True, squeeze=False)
    for row, group in enumerate(groups):
        for col, (scan, cmap, label) in enumerate(OPTO_PANELS):
            ax = axs[row, col]
            if is_novis(group) and scan != 'combined':
                ax.axis('off')
                continue
            h = ax.imshow(np.nanmean(opto[group.value][scan], axis=(0, 3)), aspect=.5, vmin=-.5, vmax=.5, cmap=cmap)
            fig.colorbar(h, ax=ax)
            ax.set_title(f'{group.label}: {label}' if scan != 'combined' else f'{group.label}: combined', fontsize=9)
            ax.set_ylabel('ROIs')
            ax.set_xlabel('Stim. Location')
            ax.set_xticks([0, 2, 4, 6], labels=[r'0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
            ax.set_yticks([-.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'2$\pi$'])
            if not is_novis(group) and scan != 'combined':           # position of the stimulus
                y = np.arange(0, 2 * np.pi, np.pi / 4)[::-1] * 8 / np.pi + .5
                ax.scatter(np.arange(N_STIM_POINTS), y if scan == 'remap_0deg' else np.roll(y, 4), color='orange', s=40)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def plot_opto_by_location(figdir, opto, groups, filename='opto_resp_by_location.pdf'):
    """Mean +- SEM optogenetic response of each group (ROIs on the x axis) at each of the 8 stimulus locations."""
    fig, axs = plt.subplots(2, N_STIM_POINTS, figsize=[16, 4.5], sharey=True)
    x = np.arange(16)
    for row, scan in enumerate(OPTO_SCANS):
        for group in groups:
            arr = opto[group.value][scan].mean(axis=3)               # flies x rois x points
            mu, sem = arr.mean(axis=0), sp.stats.sem(arr, axis=0)
            for i in range(N_STIM_POINTS):
                axs[row, i].plot(x, mu[:, i], color=group.color, label=group.label)
                axs[row, i].fill_between(x, mu[:, i] - sem[:, i], mu[:, i] + sem[:, i], color=group.color, alpha=.5)
    labels = ['0', r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$', r'5$\pi$/4', r'3$\pi$/2', r'7$\pi$/4']
    for ax in axs.flat:
        plotting.despine(ax)
        ax.set_xticks([-.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'2$\pi$'])
        ax.set_xlabel('ROIs')
    for i, label in enumerate(labels):
        axs[0, i].set_title(label, fontsize=9)
    axs[0, 0].set_ylabel('remap 0')
    axs[1, 0].set_ylabel('remap pi')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def plot_aligned_opto(figdir, aligned, groups, filename='aligned_opto_resp.pdf'):
    """Mean +- SEM response aligned to the stimulated position (ROIs centered), per group: remap 0 | remap pi | combined."""
    fig, axs = plt.subplots(1, 3, figsize=[12, 4])
    x = np.linspace(-np.pi, np.pi, num=16)
    for ax, (scan, _, label) in zip(axs, OPTO_PANELS, strict=True):
        for group in groups:
            if is_novis(group) and scan != 'combined':
                continue
            resp = aligned[group.value][scan].mean(axis=(2, 3))
            mu, sem = resp.mean(axis=0), sp.stats.sem(resp, axis=0)
            ax.plot(x, mu, color=group.color, label=f'{group.label} (n={len(resp)})')
            ax.fill_between(x, mu - sem, mu + sem, color=group.color, alpha=.2)
        ax.set_title(label)
        _style_resp_axis(ax)
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def aligned_window_tests(aligned, groups, reference, window=slice(7, 10)):
    """Mann-Whitney tests of the combined aligned response around the stimulated position, `reference` vs each other group.

    The response of each fly is averaged over the ROIs in `window` (by default the three around the stimulated
    position). p-values are Holm-corrected across the comparisons.
    """
    resp = {g.value: aligned[g.value]['combined'].mean(axis=(2, 3))[:, window].mean(axis=-1) for g in groups}
    ref_label = next(g.label for g in groups if g.value == reference)
    rows = []
    for group in groups:
        if group.value == reference:
            continue
        u, p = sp.stats.mannwhitneyu(resp[reference], resp[group.value])
        rows.append({'comparison': f'{ref_label} vs {group.label}', 'U': float(u), 'p': float(p)})
    return stats.add_holm(pd.DataFrame(rows))


def plot_vec_len_comparison(figdir, summary, groups, filename='remapping_vec_len_comparison.pdf'):
    """Offset vector length of every session by scan type and group (points, and the mean as a bar)."""
    order = [s for s in OFFSET_SCAN_TYPES if s in set(summary['scan_type'])]
    data = summary.loc[summary['condition'].isin([g.value for g in groups]) & summary['scan_type'].isin(order)]
    kw = {'data': data, 'x': 'scan_type', 'y': 'mean vec. len', 'hue': 'condition', 'order': order,
          'hue_order': [g.value for g in groups], 'palette': {g.value: g.color for g in groups}}
    fig, ax = plt.subplots(figsize=[6, 4])
    sns.stripplot(**kw, ax=ax, alpha=.5, dodge=True, s=8, legend=False)
    sns.pointplot(**kw, ax=ax, estimator=np.mean, errorbar=None, marker='_', markersize=20, markeredgewidth=6,
                  dodge=.4 if len(groups) > 1 else False, linestyle='none', legend=False)
    for group in groups:
        ax.plot([], [], color=group.color, linewidth=4, label=group.label)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    ax.set_xlabel('')
    ax.set_ylabel('offset vector length')
    ax.set_ylim([0, 1])
    plotting.despine(ax)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def mean_offsets(sum_stats):
    """Across-fly mean offset vector (angle and length) of each scan type of each group, as a table."""
    rows = []
    for group, df in sum_stats.items():
        scans = NOVIS_SCAN_TYPES if 'post_novis1 mean offset' in df.columns else STANDARD_SCAN_TYPES
        for scan in scans:
            v = (df[f'{scan} vec. len'] * np.exp(1j * df[f'{scan} mean offset'])).mean()
            rows.append({'group': group, 'scan_type': scan, 'n_flies': int(df[f'{scan} mean offset'].notna().sum()),
                         'mean_offset': float(np.angle(v)), 'mean_vec_len': float(np.abs(v))})
    return pd.DataFrame(rows)


def plot_polar_summary(figdir, sum_stats, groups, filename='remapping_all_flies_polar.pdf'):
    """Offset vector of every fly and their mean, one row per group: baseline | after the cues."""
    fig, axs = plt.subplots(len(groups), 2, subplot_kw={'projection': 'polar'}, figsize=[8, 4.5 * len(groups)],
                            squeeze=False)
    for row, group in enumerate(groups):
        plot_row_sum_stats(sum_stats[group.value], axs, row, group.value, novis=is_novis(group))
        for col in range(2):
            axs[row, col].set_title(group.label)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def plot_min_dist_summary(figdir, sum_stats, groups, zero_groups=(), filename='remapping_all_flies.pdf'):
    """Mean offset of every fly at baseline and after the two cues, one panel per group.

    `zero_groups` lists the values of the groups whose first post-cue offset is also unwrapped toward baseline
    (see :func:`plot_min_dist`).
    """
    fig, axs = plt.subplots(1, len(groups), figsize=[4 * len(groups), 4], sharex=True, sharey=True, squeeze=False)
    for ax, group in zip(axs[0], groups, strict=True):
        plot_min_dist(sum_stats[group.value], ax, color=group.color, zero=group.value in zero_groups)
        ax.set_title(group.label)
        ax.set_xticks([0, 1, 2], ['baseline', 'post 0deg', 'post 180deg'] if not is_novis(group)
                      else ['baseline', 'post novis 1', 'post novis 2'])
        ax.set_yticks([-np.pi, 0, np.pi], [r'-$\pi$', '0', r'$\pi$'])
        ax.set_ylabel('Mean Offset')
        plotting.despine(ax)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')
    return fig


def plot_remap_error(figdir, sum_stats, groups, reference, filename='remapping_error.pdf'):
    """Average remapping error of every fly (see :func:`avg_diff`), per group, with tests.

    Each group is tested against chance (pi/2, Wilcoxon) and `reference` against every other group
    (Mann-Whitney); both families are Holm-corrected.

    Returns:
        Tuple of (figure, DataFrame of tests).
    """
    errors = {g.value: avg_diff(sum_stats[g.value]).dropna() for g in groups}
    fig, ax = plt.subplots(figsize=[1.2 * len(groups) + .6, 3.5])
    for i, group in enumerate(groups):
        err = errors[group.value]
        ax.scatter(i + np.linspace(0, .1, num=len(err)), err, color=group.color, alpha=.3, s=80)
        ax.plot([i - .05, i + .15], err.median() * np.ones(2), color=group.color, linewidth=8)
    ax.axhline(np.pi / 2, color='k', linestyle='--', alpha=.5, zorder=-1)
    ax.set_xticks(np.arange(len(groups)) + .05, labels=[g.label for g in groups], fontsize=8)
    ax.set_xlim([-.5, len(groups)-.5])
    ax.set_ylabel('Avg remapping error')
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                  labels=[0, r'$\pi$/4', r'$\pi$/2', r'$3\pi$/4', r'$\pi$'])
    plotting.despine(ax)
    fig.tight_layout()
    fig.savefig(pathlib.Path(figdir) / filename, bbox_inches='tight')

    rows = [{'family': 'error vs chance', 'comparison': g.label, 'n': len(errors[g.value]),
             **{k: v for k, v in stats.wilcoxon_vs(errors[g.value], np.pi / 2).items() if k != 'n'}} for g in groups]
    ref_label = next(g.label for g in groups if g.value == reference)
    tests = []
    for group in groups:
        for group2 in groups[1:]:
            # if group.value != group2.value: #reference:
            if (group, group2) in tests or (group2, group) in tests or group.value == group2.value:
                continue
            tests.append((group, group2))

            u, p = sp.stats.mannwhitneyu(errors[group.value], errors[group2.value])
            rows.append({'family': 'error between groups', 'comparison': f'{group.label} vs {group2.label}',
                        'W': float(u), 'p': float(p)})

    if len(groups)>2:
        h, p = sp.stats.kruskal(*[errors[g.value] for g in groups])
        rows.append({'family': 'kruskal', 'comparison': 'all groups kruskal wallis', 'H': float(h), 'p': float(p)})

    tests = pd.DataFrame(rows)
    tests['p_holm'] = np.nan
    for _, sub in tests.groupby('family'):
        tests.loc[sub.index, 'p_holm'] = stats.add_holm(sub.copy())['p_holm']
    return fig, tests


PAIRING_CMAPS = {'remap_0deg': 'PuRd', 'remap_180deg': 'GnBu'}


def stimulated_roi(k):
    """ROI stimulated at the `k`-th pairing (0-based), as :func:`align_opto_responses` places it."""
    return (16 - 2 * (k % N_STIM_POINTS)) % 16


def plot_pairing_heatmaps(fly, scans, vmin=-.5, vmax=3):
    """Heatmap of each pairing scan of a fly, with the VR angle and the stimulated wedge of every pairing.

    Args:
        fly: Title of the figure.
        scans: List of (scan type, session name, preprocessed session), in the order to draw them.

    Returns:
        Tuple of (figure, list of dicts summarizing each scan: scan_type, session, pairings recorded and programmed,
        complete, duration_s).
    """
    fig, axs = plt.subplots(len(scans), 1, figsize=[13, 3.2 * len(scans)], squeeze=False)
    summary = []
    for ax, (scan_type, session_name, pp) in zip(axs[:, 0], scans, strict=True):
        programmed = programmed_stimulations(pp)
        onsets, n_frames, dff = masked_pairing_activity(pp)
        voltage = pp.voltage_recording_aligned
        time = (voltage['Time(ms)'].to_numpy() - voltage['Time(ms)'].iloc[0]) / 1000
        heading = np.angle(np.exp(1j * (-voltage[' Heading'].to_numpy() - np.pi)))     # as session.GetTS does
        dt = np.diff(time).mean()

        cmap = plt.get_cmap(PAIRING_CMAPS.get(scan_type, 'Greys')).copy()
        cmap.set_bad('white')                                       # the masked stimulation frames
        h = ax.imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none',
                      extent=(time[0] - dt / 2, time[-1] + dt / 2, dff.shape[0] - .5, -.5))
        fig.colorbar(h, ax=ax, pad=.01, label='z-scored F')
        ax.scatter(time, utilities.heading_to_roi_coords(heading, dff.shape[0]), color='orange', s=2, zorder=2)
        stim_mid = time[np.minimum(onsets + n_frames // 2, len(time) - 1)]
        ax.scatter(stim_mid, [stimulated_roi(k) for k in range(len(onsets))], marker='o', s=40, facecolor='none',
                   edgecolor='red', linewidth=1.5, zorder=3)
        ax.set_ylim(dff.shape[0] - .5, -.5)
        plotting.format_roi_axis(ax, 'y')
        ax.set_xlabel('Time (s)')
        complete = len(onsets) >= programmed
        ax.set_title(f'{scan_type} ({session_name}): {len(onsets)} of {programmed} pairings recorded, '
                     f'{time[-1]:.0f} s' + ('' if complete else '  -- INCOMPLETE'), fontsize=11,
                     color='black' if complete else 'red')
        summary.append({'scan_type': scan_type, 'session': session_name, 'pairings_recorded': len(onsets),
                        'pairings_programmed': programmed, 'complete': complete, 'duration_s': float(time[-1])})
    axs[-1, 0].plot([], [], 'o', color='orange', markersize=3, label='VR angle')
    axs[-1, 0].plot([], [], 'o', markerfacecolor='none', markeredgecolor='red', label='stimulated wedge')
    axs[-1, 0].legend(loc='upper right', bbox_to_anchor=(1., -.25), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(fly)
    fig.tight_layout()
    return fig, summary


def plot_pairing_scans(figdir, session_manager, groups, scan_types=OPTO_SCANS, vmin=-.5, vmax=3, close=True):
    """Pairing-scan heatmaps of every fly (:func:`plot_pairing_heatmaps`), to see which scans are complete.

    Scans are drawn in the order in which they were acquired (the last three digits of the session name). Figures go to
    ``<figdir>/sessions/<fly_id>/pairing_heatmaps.pdf``.

    Returns:
        DataFrame with one row per pairing scan: fly_id, condition, scan_type, session, pairings recorded and
        programmed, complete and duration_s.
    """
    sess_df = session_manager.sess_df
    sess_df = sess_df.loc[sess_df['condition'].isin([g.value for g in groups]) & sess_df['scan_type'].isin(scan_types)]
    rows = []
    for fly in sess_df['fly_id'].unique():
        fly_df = sess_df.loc[sess_df['fly_id'] == fly]
        fly_df = fly_df.iloc[np.argsort([int(name[-3:]) for name in fly_df['session']])]
        scans = [(row['scan_type'], row['session'], session_manager.load_row(row)) for _, row in fly_df.iterrows()]
        fig, summary = plot_pairing_heatmaps(fly, scans, vmin=vmin, vmax=vmax)
        folder = pathlib.Path(figdir) / 'sessions' / fly
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / 'pairing_heatmaps.pdf', bbox_inches='tight')
        condition = fly_df['condition'].iloc[0]
        rows += [{'fly_id': fly, 'condition': condition, **row} for row in summary]
        if close:
            plt.close(fig)
    return pd.DataFrame(rows)


def plot_session_figures(figdir, session_manager, groups, vmin=-.5, vmax=3, close=True):
    """Heatmaps with offset histograms, offset histograms and polar plots of the baseline and two post-cue scans of each fly.

    Flies need all three scans. The rows of the heatmap figure follow the order in which the scans were acquired
    (the last three digits of the session name). Figures go to ``<figdir>/sessions/<fly_id>/``.

    Returns:
        List of the fly ids plotted.
    """
    sess_df = session_manager.sess_df
    plotted = []
    for fly in sess_df.loc[sess_df['condition'].isin([g.value for g in groups]), 'fly_id'].unique():
        fly_df = sess_df.loc[sess_df['fly_id'] == fly]
        scans = {s: fly_df.loc[fly_df['scan_type'] == s] for s in STANDARD_SCAN_TYPES}
        if any(len(df) == 0 for df in scans.values()):
            continue
        ts_dict = {'fly': fly, **{s: session.GetTS(session_manager.load_row(df.iloc[0])) for s, df in scans.items()}}
        sess_order = np.argsort([int(scans[s].iloc[0]['session'][-3:]) for s in STANDARD_SCAN_TYPES])

        folder = pathlib.Path(figdir) / 'sessions' / fly
        folder.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_sess_heatmaps_w_hist(ts_dict, vmin=vmin, vmax=vmax, sess_order=sess_order)
        fig.savefig(folder / 'heatmaps.pdf', bbox_inches='tight')
        (fig_hist, _), (fig_polar, _) = plot_sess_histograms(ts_dict)
        fig_hist.savefig(folder / 'offset_hist.pdf', bbox_inches='tight')
        fig_polar.savefig(folder / 'offset_polar.pdf', bbox_inches='tight')
        plotted.append(fly)
        if close:
            plt.close('all')
    return plotted
