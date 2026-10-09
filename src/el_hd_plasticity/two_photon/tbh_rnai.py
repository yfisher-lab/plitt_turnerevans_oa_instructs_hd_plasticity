"""Analysis and visualization of tbh RNAi fly two-photon imaging data.

Computes offset, rho and PVA-velocity statistics from imaging sessions of tbh RNAi (and control)
flies, and draws the closed-loop vs dark figures. The statistics are thin wrappers over
:mod:`stats` that fix the session-table column holding the genotype (``rnai``); the closed-loop
vs dark figures (``plot_sess_*``) are shared with :mod:`oa_receptor_ko`.
"""

from matplotlib import pyplot as plt

from . import plotting, stats
from .utilities import DEFAULT_OFFSET_BINS, DEFAULT_PLOT_TIMES

# Session-table columns copied into the per-session statistics.
OFFSET_META_COLS = ('fly_id', 'closed_loop', 'dark', 'rnai')
BINNED_META_COLS = ('fly_id', 'closed_loop', 'rnai', 'dark')
# Per-session columns carried into regression tables.
REG_META_COLS = ('fly_id', 'rnai', 'dark')

# Sessions with at least this many closed-loop blocks count as closed loop when aggregating per
# fly, i.e. more than one block (>= 10 min) of prior closed-loop experience.
MIN_CL = 2


def offset_stats(sess_df, load_row, **kwargs):
    """Compute offset statistics (variance, mean, magnitude) for each session.

    Computes circular statistics on EPG offset (difference between heading and PVA)
    for each imaging session across different closed-loop and dark conditions.

    Args:
        sess_df: DataFrame with session metadata including fly_id, closed_loop, dark, rnai columns.
        load_row: Callable that loads preprocessed data for a session row.
        **kwargs: Passed to :meth:`session.SessionTimeseries.from_session`.

    Returns:
        DataFrame with columns: fly_id, closed_loop, dark, rnai, offset_var, offset_mean, offset_mag.
    """
    return stats.offset_stats(sess_df, load_row, meta_cols=OFFSET_META_COLS, **kwargs)


def offset_stats_unique(stats_df):
    """Aggregate offset statistics per fly by averaging across sessions.

    Reduces multiple sessions per condition to single values by averaging, creating
    one row per condition per fly. Requires flies to have both closed-loop (more than one
    closed-loop block of experience) and dark data. Applies logit transform for mixed
    effects modeling.

    Args:
        stats_df: Output from offset_stats().

    Returns:
        DataFrame with one row per condition per fly containing aggregated statistics.
    """
    return stats.offset_stats_unique(stats_df, group_col='rnai', min_cl=MIN_CL)


def offset_stats_plot(stats_df_unique):
    """Reformat aggregated stats for side-by-side plotting (wide format).

    Args:
        stats_df_unique: Output from offset_stats_unique().

    Returns:
        DataFrame with one row per fly, columns for dark and cl values of each metric.
    """
    return stats.offset_stats_plot(stats_df_unique, group_col='rnai')


def rho_stats(sess_df, load_row, dh_bins):
    """Compute pva magnitude binned by heading velocity.

    Computes mean rho (population vector average magnitude) for each bin of
    absolute heading velocity.

    Args:
        sess_df: Session DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.

    Returns:
        DataFrame with rho_dig: array of mean rho values for each dh bin.
    """
    return stats.rho_stats(sess_df, load_row, dh_bins, meta_cols=BINNED_META_COLS)


def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned rho stats to long format for regression modeling.

    Args:
        grouped_stats: Output from rho_stats().
        dh_bins: Bin edges used in rho_stats().

    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded).
    """
    return stats.reformat_binned_stats_for_reg(grouped_stats, dh_bins, {'rho': 'rho_dig'},
                                               meta_cols=REG_META_COLS)


def dphi_stats(sess_df, load_row, dh_bins):
    """Compute PVA angular velocity binned by heading velocity.

    Args:
        sess_df: Session DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.

    Returns:
        DataFrame with dphi_dig: array of mean PVA velocity for each dh bin.
    """
    return stats.dphi_stats(sess_df, load_row, dh_bins, meta_cols=BINNED_META_COLS)


def reformat_dphi_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned PVA velocity stats to long format for regression modeling.

    Args:
        grouped_stats: Output from dphi_stats().
        dh_bins: Bin edges used in dphi_stats().

    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded).
    """
    return stats.reformat_binned_stats_for_reg(grouped_stats, dh_bins, {'dphi': 'dphi_dig'},
                                               meta_cols=REG_META_COLS)


# ----------------------------------------------------------------------------
# Closed-loop vs dark figures (ts_dict has 'fly', 'closed_loop' and 'dark' timeseries)
# ----------------------------------------------------------------------------
def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES):
    """Plot DFF heatmaps for closed-loop and dark conditions side-by-side.

    Shows raw and heading-aligned neural activity for comparison between conditions.

    Args:
        ts_dict: Dictionary with 'fly' key and 'closed_loop' and 'dark' timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.

    Returns:
        Tuple of (figure, axes).
    """
    traces = [plotting.Trace.from_ts('closed_loop', ts_dict['closed_loop'], 'Greys'),
              plotting.Trace.from_ts('dark', ts_dict['dark'], 'Purples')]
    return plotting.plot_heatmap_grid(traces, ts_dict['fly'], [15, 4], vmin, vmax, plot_times,
                                      clip_times=False)


def plot_sess_histograms(ts_dict, bins=DEFAULT_OFFSET_BINS):
    """Plot offset distributions and mean vectors for closed-loop and dark conditions.

    Creates histogram and polar plots showing PVA offset tuning and population
    vector magnitude for each condition.

    Args:
        ts_dict: Dictionary with 'fly' key and 'closed_loop' and 'dark' timeseries objects.
        bins: Bins for histogram computation.

    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    traces = [plotting.Trace.from_ts('closed_loop', ts_dict['closed_loop'], color='black'),
              plotting.Trace.from_ts('dark', ts_dict['dark'], color=plt.cm.Purples(.8))]
    return plotting.plot_offset_histograms(traces, bins, hist_title=ts_dict['fly'],
                                           polar_title=ts_dict['fly'], polar_ylim=[0, .8])


def plot_sess_heatmaps_w_hist(ts_dict, vmin=-.5, vmax=3, plot_times=DEFAULT_PLOT_TIMES,
                              twindow=None, bins=DEFAULT_OFFSET_BINS):
    """Plot heatmaps with integrated offset histograms for both conditions.

    Creates a 2-row layout showing DFF heatmaps alongside offset histograms,
    with vector length statistics for each condition.

    Args:
        ts_dict: Dictionary with 'fly' key and 'closed_loop' and 'dark' timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        bins: Bins for histogram computation.

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    traces = [plotting.Trace.from_ts('closed_loop', ts_dict['closed_loop'], 'Greys'),
              plotting.Trace.from_ts('dark', ts_dict['dark'], 'Purples')]
    return plotting.plot_heatmaps_with_offset_hist(
        traces, ts_dict['fly'], figsize=[8, 4], vmin=vmin, vmax=vmax, plot_times=plot_times,
        twindow=twindow, bins=bins, hist_xticks=[0, .1, .2], heatmap_titles=False)
