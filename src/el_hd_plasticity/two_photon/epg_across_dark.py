"""Figures for EPG imaging across closed loop, dark, and closed loop again.

Each figure takes ``ts_dict``: a dict with a ``'fly'`` name and :class:`session.GetTS` objects
for the three epochs ``'closed_loop 1'``, ``'dark'`` and ``'closed_loop 2'``.
"""

from matplotlib import pyplot as plt

from . import plotting
from .utilities import DEFAULT_OFFSET_BINS, DEFAULT_PLOT_TIMES

# (epoch, heatmap colormap) in plotting order
EPOCHS = (('closed_loop 1', 'Greys'), ('dark', 'Purples'), ('closed_loop 2', 'Greys'))


def _heatmap_traces(ts_dict):
    """One :class:`plotting.Trace` per epoch, colored by :data:`EPOCHS`."""
    return [plotting.Trace.from_ts(key, ts_dict[key], cmap) for key, cmap in EPOCHS]


def plot_transpose_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES, twindow=None):
    """Plot transposed (time vs ROI) heatmaps across three experimental conditions.

    Creates a 2x3 subplot grid showing raw and heading-aligned DFF data for three conditions:
    closed loop 1, dark, and closed loop 2.

    Args:
        ts_dict: Dictionary with 'fly' key and condition keys ('closed_loop 1', 'dark', 'closed_loop 2')
                 containing timeseries objects with dff, dff_h_aligned, time, heading attributes.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on y-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.

    Returns:
        Tuple of (figure, axes).
    """
    return plotting.plot_heatmap_grid(_heatmap_traces(ts_dict), ts_dict['fly'], [6, 15], vmin, vmax,
                                      plot_times, twindow, transpose=True)


def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES, twindow=None):
    """Plot heatmaps of DFF activity across three experimental conditions.

    Creates a 3x2 subplot grid showing raw and heading-aligned DFF data for three conditions:
    closed loop 1, dark, and closed loop 2.

    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.

    Returns:
        Tuple of (figure, axes).
    """
    return plotting.plot_heatmap_grid(_heatmap_traces(ts_dict), ts_dict['fly'], [15, 6], vmin, vmax,
                                      plot_times, twindow)


def plot_sess_histograms(ts_dict, bins=DEFAULT_OFFSET_BINS):
    """Plot offset histograms and polar plot across three conditions.

    Creates histogram and polar plots showing EPG offset distribution and mean vector
    for each condition (the second closed-loop epoch is hatched/dashed).

    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects
                 with offset and offset_c attributes.
        bins: Bins for histogram computation.

    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    traces = [plotting.Trace.from_ts('closed_loop 1', ts_dict['closed_loop 1'], color='black'),
              plotting.Trace.from_ts('dark', ts_dict['dark'], color=plt.cm.Purples(.8)),
              plotting.Trace.from_ts('closed_loop 2', ts_dict['closed_loop 2'], color='black', hatch='/')]
    return plotting.plot_offset_histograms(traces, bins, hist_title=ts_dict['fly'],
                                           polar_title=ts_dict['fly'], outline=False)


def plot_sess_heatmaps_w_hist(ts_dict, bins=DEFAULT_OFFSET_BINS, vmin=-.5, vmax=.5,
                              plot_times=DEFAULT_PLOT_TIMES, twindow=None):
    """Plot heatmaps with integrated offset histograms for all three conditions.

    Creates a 3-row layout showing DFF heatmaps alongside offset histograms for each
    experimental condition.

    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        bins: Bins for histogram computation.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    return plotting.plot_heatmaps_with_offset_hist(
        _heatmap_traces(ts_dict), ts_dict['fly'], figsize=[8, 2], vmin=vmin, vmax=vmax,
        plot_times=plot_times, twindow=twindow, bins=bins, share_axes=True, hist_xlim_zero=True)
