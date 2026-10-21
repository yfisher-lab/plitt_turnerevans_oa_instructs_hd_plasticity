"""Shared figure building blocks for EPG/EL ring imaging data.

The experiment modules (``epg_across_dark``, ``tbh_rnai``, ``remapping``, ...) draw the same
three kinds of figures for different sets of conditions:

* :func:`plot_heatmap_grid` - raw and heading-aligned dF/F heatmaps, one row per condition.
* :func:`plot_heatmaps_with_offset_hist` - dF/F heatmaps with the offset distribution beside each.
* :func:`plot_offset_histograms` - offset histogram and polar mean-vector plots.

Each takes a list of :class:`Trace` (one per condition/channel) so that the experiment modules
only have to say which traces to draw and in which colors.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import AutoMinorLocator

from . import utilities
from .utilities import DEFAULT_OFFSET_BINS, DEFAULT_PLOT_TIMES, N_EB_ROIS

# Heatmap tick positions: ROI 0, middle ROI and last ROI (imshow pixel edges) <-> 0, pi, 2pi
ROI_TICKS = [-0.5, N_EB_ROIS / 2 - 0.5, N_EB_ROIS - 0.5]
ROI_TICK_LABELS = [r'0', r'$\pi$', r'$2\pi$']
# Position of the heading trace after heading alignment (center of the ROI axis)
ALIGNED_HEADING_POS = (N_EB_ROIS - 1) / 2


@dataclass
class Trace:
    """Data for one condition (or imaging channel) of a figure.

    Attributes:
        label: Name shown as title/legend entry.
        dff: (n_rois, n_timepoints) activity.
        dff_aligned: (n_rois, n_timepoints) heading-aligned activity (heatmap grids only).
        heading: Heading in radians, length n_timepoints.
        offset: Heading offset in radians.
        offset_c: Complex offset (unit vectors).
        time: Time of each sample in seconds (pandas Series when ticks are clipped to the data).
        cmap: Colormap name or object for heatmaps.
        hatch: Hatch pattern; draws the offset histogram as an open, hatched outline instead of
            a filled one (used to mark a repeated condition).
        color: Color for histogram/polar plots; defaults to ``cmap(.8)``.
    """
    label: str
    dff: Any
    heading: Any
    offset: Any
    offset_c: Any
    time: Any
    cmap: Any = 'Greys'
    dff_aligned: Any = None
    hatch: str | None = None
    color: Any = None

    @classmethod
    def from_ts(cls, label, ts, cmap='Greys', channel=None, hatch=None, color=None):
        """Build a Trace from a :class:`~session.SessionTimeseries` object.

        Args:
            label: Name of the condition.
            ts: Timeseries object with dff, dff_h_aligned, heading, offset, offset_c and time.
            cmap: Colormap name or object.
            channel: For multi-channel timeseries, the channel to draw.
            hatch: See :class:`Trace`.
            color: See :class:`Trace`.
        """
        sel = (lambda a: a) if channel is None else (lambda a: a[channel])
        return cls(label=label, dff=sel(ts.dff), dff_aligned=sel(ts.dff_h_aligned), heading=ts.heading,
                   offset=sel(ts.offset), offset_c=sel(ts.offset_c), time=ts.time,
                   cmap=cmap, hatch=hatch, color=color)

    @property
    def draw_color(self):
        """Color for histogram/polar plots: `color` if given, else the colormap at 0.8."""
        return self.color if self.color is not None else plt.get_cmap(self.cmap)(.8)


# ----------------------------------------------------------------------------
# Axis formatting
# ----------------------------------------------------------------------------
def format_roi_axis(ax, axis='y', label='ROIs'):
    """Label an axis as ROI index with ticks at 0, pi and 2pi."""
    if axis == 'y':
        ax.set_ylabel(label)
        ax.set_yticks(ROI_TICKS, labels=ROI_TICK_LABELS)
    else:
        ax.set_xlabel(label)
        ax.set_xticks(ROI_TICKS, labels=ROI_TICK_LABELS)


def format_time_axis(ax, time, plot_times, axis='x', clip=True, label='Time (s)'):
    """Label a sample-index axis in seconds.

    Args:
        ax: Axes to format.
        time: Time of each plotted sample (pandas Series).
        plot_times: Times (s) at which to place ticks.
        axis: 'x' or 'y'.
        clip: If True, drop `plot_times` beyond the end of the data.
        label: Axis label.

    Returns:
        The tick times that were used.
    """
    if clip:
        plot_times = plot_times[plot_times < time.iloc[-1]]
    ticks = utilities.get_time_tick_indices(time, plot_times)
    if axis == 'x':
        ax.set_xticks(ticks, labels=plot_times)
        ax.set_xlabel(label)
    else:
        ax.set_yticks(ticks, labels=plot_times)
        ax.set_ylabel(label)
    return plot_times


def _format_offset_hist_panel(ax, xticks=None):
    """Style a vertical offset histogram: pi ticks, orange line at zero offset, right-hand axis."""
    ax.set_yticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', r'0', r'$\pi$'])
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim([np.pi, -np.pi])
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, axis='y', linestyle='-', alpha=0.8, linewidth=2.5, which='major')
    ygridlines = ax.get_ygridlines()
    ygridlines[1].set_color('orange')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1.5, which='minor')
    ax.set_ylabel('Offset')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_xlabel('Prop.')


# ----------------------------------------------------------------------------
# Heatmaps
# ----------------------------------------------------------------------------
def plot_heatmap_grid(traces, title, figsize, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES,
                      twindow=None, clip_times=True, transpose=False):
    """Raw and heading-aligned dF/F heatmaps with the heading trace overlaid.

    One row per trace (columns if `transpose`), raw activity on the left/top and heading-aligned
    activity on the right/bottom. Axes are shared; heading is drawn in orange.

    Args:
        traces: List of :class:`Trace` (needs dff, dff_aligned, heading, time, cmap, label).
        title: Figure title.
        figsize: Figure size.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Times (s) to mark on the time axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        clip_times: Drop `plot_times` beyond the end of the data.
        transpose: If True, time runs down the y-axis and ROIs along x.

    Returns:
        Tuple of (figure, axes).
    """
    n = len(traces)
    fig, ax = plt.subplots(*((2, n) if transpose else (n, 2)), figsize=figsize, sharey=True, sharex=True)

    def orient(a):
        """Transpose heatmap data when time runs down the y-axis."""
        return a.T if transpose else a

    def scatter(a, x, y, **kw):
        """Orange trace of `y` (ROI coordinate) against `x` (sample index); axes swap if transposed."""
        a.scatter(*((y, x) if transpose else (x, y)), color='orange', s=5, **kw)

    for i, tr in enumerate(traces):
        raw_ax, aligned_ax = (ax[0, i], ax[1, i]) if transpose else (ax[i, 0], ax[i, 1])
        mask = utilities.time_window_mask(tr.time, twindow)
        dff, time = tr.dff[:, mask], tr.time[mask]

        x = np.arange(dff.shape[1])
        heading_ = utilities.heading_to_roi_coords(tr.heading[mask])

        h = raw_ax.imshow(orient(dff), aspect='auto', cmap=tr.cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=raw_ax)
        scatter(raw_ax, x, heading_)

        h = aligned_ax.imshow(orient(tr.dff_aligned[:, mask]), aspect='auto', cmap=tr.cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=aligned_ax)
        scatter(aligned_ax, x, ALIGNED_HEADING_POS * np.ones_like(heading_))

        format_roi_axis(raw_ax, 'x' if transpose else 'y')
        format_time_axis(raw_ax, time, plot_times, 'y' if transpose else 'x', clip=clip_times)
        raw_ax.set_title(tr.label)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


def plot_heatmaps_with_offset_hist(traces, title, figsize, vmin=-.5, vmax=3, plot_times=DEFAULT_PLOT_TIMES,
                                   twindow=None, bins=DEFAULT_OFFSET_BINS, slots=None, share_axes=False,
                                   hist_xticks=None, heatmap_titles=True, show_heading=True,
                                   hist_title='vec_length', hist_xlim_zero=False):
    """dF/F heatmaps (time on x) with the offset distribution beside each and a colorbar.

    Args:
        traces: List of :class:`Trace` (needs dff, heading, offset, offset_c, time, cmap, label).
        title: Figure title.
        figsize: Figure size.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Times (s) to mark on the time axis.
        twindow: Optional tuple (start_time, end_time) to limit the heatmap window (the offset
            histogram always uses all data).
        bins: Bins for the offset histogram.
        slots: Row index of each trace; defaults to the order of `traces`.
        share_axes: Share the heatmap y-axis and histogram x-axis across rows.
        hist_xticks: Optional x-ticks (proportion) for the histograms.
        heatmap_titles: Title each heatmap with the trace label.
        show_heading: Overlay the heading trace on the heatmap.
        hist_title: Statistic shown above each histogram: 'vec_length' (mean offset vector length)
            or 'variance' (circular variance of the offset).
        hist_xlim_zero: Start the first histogram's x-axis at 0.

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    n = len(traces)
    slots = range(n) if slots is None else slots
    fig = plt.figure(figsize=figsize)
    gs = GS(n, 4, figure=fig, width_ratios=[6, 1, .8, .2], wspace=.25, hspace=.8)
    heatmap_axs = [fig.add_subplot(gs[0, 0])]
    heatmap_axs.extend([fig.add_subplot(gs[i, 0], sharey=heatmap_axs[0] if share_axes else None)
                        for i in range(1, n)])
    hist_axs = [fig.add_subplot(gs[0, 1])]
    hist_axs.extend([fig.add_subplot(gs[i, 1], sharex=hist_axs[0] if share_axes else None)
                     for i in range(1, n)])
    cbar_ax = [fig.add_subplot(gs[i, 3]) for i in range(n)]

    for tr, row in zip(traces, slots, strict=True):
        cmap = plt.get_cmap(tr.cmap)
        mask = utilities.time_window_mask(tr.time, twindow)
        dff, time = tr.dff[:, mask], tr.time[mask]

        x = np.arange(dff.shape[1])
        h = heatmap_axs[row].imshow(dff, aspect='auto', cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        fig.colorbar(h, cax=cbar_ax[row])
        if show_heading:
            heatmap_axs[row].scatter(x, utilities.heading_to_roi_coords(tr.heading[mask]), color='orange', s=5)

        format_roi_axis(heatmap_axs[row], 'y')
        heatmap_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        format_time_axis(heatmap_axs[row], time, plot_times, 'x')
        if heatmap_titles:
            heatmap_axs[row].set_title(tr.label)

        hist, centers = utilities.normalized_histogram(tr.offset, bins)
        if tr.hatch is None:
            hist_axs[row].fill_betweenx(centers, 0, hist, color=cmap(.8), alpha=.5)
        else:
            hist_axs[row].fill_betweenx(centers, 0, hist, alpha=1, hatch=tr.hatch,
                                        color='none', edgecolor=cmap(.8))
        _format_offset_hist_panel(hist_axs[row], xticks=hist_xticks)

        if hist_title == 'variance':
            hist_axs[row].set_title(f"variance={sp.stats.circvar(tr.offset, low=-np.pi, high=np.pi):.2f}",
                                    fontsize=8)
        else:
            hist_axs[row].set_title(f"vec length={np.abs(tr.offset_c.mean()):.2f}", fontsize=8)

    if hist_xlim_zero:
        hist_axs[0].set_xlim(left=0)

    fig.suptitle(title)
    fig.subplots_adjust(top=.9)      # keep the figure title clear of the first row's titles
    return fig, (heatmap_axs, hist_axs, cbar_ax)


# ----------------------------------------------------------------------------
# Offset histograms
# ----------------------------------------------------------------------------
def plot_offset_histograms(traces, bins=DEFAULT_OFFSET_BINS, hist_title=None, polar_title=None,
                           suptitle=None, polar_ylim=None, legend=True, outline=True):
    """Offset histogram and polar plot of the mean offset vector, one entry per trace.

    Args:
        traces: List of :class:`Trace` (needs label, offset, offset_c, color or cmap, hatch).
        bins: Bins for the histogram.
        hist_title: Title of the histogram axes.
        polar_title: Title of the polar axes.
        suptitle: Figure title applied to both figures.
        polar_ylim: Optional (min, max) radial limits for the polar plot.
        legend: Add a legend (trace labels) to the polar plot.
        outline: Draw a thin outline (in the fill color) around filled histograms. Hatched
            histograms are always outlined.

    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})

    for tr in traces:
        color, hatch = tr.draw_color, tr.hatch
        hist, centers = utilities.normalized_histogram(tr.offset, bins)
        offset_c_mu = tr.offset_c.mean()

        if hatch:
            ax_hist.fill_between(centers, hist, color='none', alpha=1, hatch=hatch, edgecolor=color)
        elif outline:
            ax_hist.fill_between(centers, hist, color=color, alpha=.4)
        else:
            # an explicit edgecolor=None overrides the edge that color= would set
            ax_hist.fill_between(centers, hist, color=color, alpha=.4, edgecolor=None)
        ax_polar.plot(np.angle(offset_c_mu) * np.ones([2]), [0, np.abs(offset_c_mu)],
                      color=color, linewidth=2, label=tr.label,
                      linestyle='--' if hatch else '-', alpha=.4 if hatch else 1)

    if suptitle is not None:
        fig_hist.suptitle(suptitle)
        fig_polar.suptitle(suptitle)

    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax_hist.set_xlabel('Offset')
    ax_hist.set_yticks([0, .05, .1, .15, .2, .25])
    ax_hist.set_ylim([0, .25])
    ax_hist.set_ylabel('Proportion')
    if hist_title is not None:
        ax_hist.set_title(hist_title)
    fig_hist.tight_layout()

    ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax_polar.set_yticks([.2, .4, .6, .8])     # no 0 label: it would sit under the vectors' common origin
    ax_polar.set_rlabel_position(67.5)         # between the angular labels, away from the vectors
    if polar_ylim is not None:
        ax_polar.set_ylim(polar_ylim)
    if polar_title is not None:
        ax_polar.set_title(polar_title)
    if legend:      # outside the axes so it never covers a vector
        ax_polar.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), frameon=False)
    fig_polar.tight_layout()

    return (fig_hist, ax_hist), (fig_polar, ax_polar)
