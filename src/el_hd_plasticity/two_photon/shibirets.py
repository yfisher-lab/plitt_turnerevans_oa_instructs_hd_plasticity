"""Figures for shibire-ts (heat-induced silencing) experiments."""


from . import plotting
from .utilities import DEFAULT_OFFSET_BINS, DEFAULT_PLOT_TIMES


def plot_sess_heatmaps_w_hist(ts_dict, vmin=-.5, vmax=3, plot_times=DEFAULT_PLOT_TIMES,
                              twindow=None, bins=DEFAULT_OFFSET_BINS, show_behavior=True, figsize=None):
    """Plot heatmaps with integrated offset histograms for both conditions.

    Creates a 2-row layout showing DFF heatmaps alongside offset histograms,
    with vector length statistics for each condition. Under each heatmap the fly's heading and
    rotation speed are drawn in their own panels, as in the GRAB sensor session figures.

    Args:
        ts_dict: Dictionary with 'fly' key and 'baseline' and 'post_heat' timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        bins: Bins for histogram computation.
        show_behavior: Draw the heading and rotation-speed panels under each heatmap.
        figsize: Figure size; tall enough for the behavior panels by default.

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    if figsize is None:
        figsize = [8, 7] if show_behavior else [8, 4]
    traces = [plotting.Trace.from_ts(key, ts_dict[key], 'Greens') for key in ('baseline', 'post_heat')]
    return plotting.plot_heatmaps_with_offset_hist(
        traces, ts_dict['fly'], figsize=figsize, vmin=vmin, vmax=vmax, plot_times=plot_times,
        twindow=twindow, bins=bins, hist_xticks=[0, .1, .2], heatmap_titles=False, show_heading=False,
        show_behavior=show_behavior)
