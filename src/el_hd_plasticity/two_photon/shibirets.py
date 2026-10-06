
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from . import session, utilities



def plot_sess_heatmaps_w_hist(ts_dict, vmin=-.5, vmax=3, plot_times=np.arange(0, 360, 60),
                              twindow=None, bins=np.linspace(-np.pi, np.pi, num=17)):
    """Plot heatmaps with integrated offset histograms for both conditions.
    
    Creates a 2-row layout showing DFF heatmaps alongside offset histograms,
    with vector length statistics for each condition.
    
    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        bins: Bins for histogram computation.
        
    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    fig = plt.figure(figsize=[8, 4])
    gs = GS(2, 4, figure=fig, width_ratios=[6, 1, .8, .2], wspace=.05, hspace=.8)
    heatmap_axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    hist_axs = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    cbar_ax = [fig.add_subplot(gs[i, 3]) for i in range(2)]

    def plot_row(key, row, cmap):
        ts = ts_dict[key]
        dff = ts.dff
        heading = ts.heading
        offset = ts.offset
        time = ts.time

        # Apply time window mask if specified
        if twindow is not None:
            mask = (time >= twindow[0]) & (time <= twindow[1])
        else:
            mask = np.ones_like(time, dtype=bool)

        dff = dff[:, mask]
        time = time[mask]
        heading = heading[mask]

        x = np.arange(dff.shape[1])
        heading_ = (heading + np.pi) / (2 * np.pi) * 15
        
        # DFF heatmap
        h = heatmap_axs[row].imshow(dff, aspect='auto', cmap=cmap, interpolation='none', 
                                   vmin=vmin, vmax=vmax)
        fig.colorbar(h, cax=cbar_ax[row])
        # heatmap_axs[row].scatter(x, heading_, color='orange', s=5)

        heatmap_axs[row].set_ylabel('ROIs')
        heatmap_axs[row].set_yticks([-0.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        heatmap_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        
        _plot_times = plot_times[plot_times < time.iloc[-1]]
        heatmap_axs[row].set_xticks(utilities.get_time_tick_indices(time, _plot_times), labels=_plot_times)
        heatmap_axs[row].set_xlabel('Time (s)')

        # Offset histogram
        centers = (bins[:-1] + bins[1:]) / 2
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist / hist.sum()
        hist_axs[row].fill_betweenx(centers, 0, hist, color=cmap(.8), alpha=.5)
        
        hist_axs[row].set_yticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', r'0', r'$\pi$'])
        hist_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        hist_axs[row].set_ylim([np.pi, -np.pi])
        hist_axs[row].set_xticks([0, .1, .2])
        hist_axs[row].grid(True, axis='y', linestyle='-', alpha=0.8, linewidth=2.5, which='major')
        ygridlines = hist_axs[row].get_ygridlines()
        ygridlines[1].set_color('orange')
        hist_axs[row].grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1.5, which='minor')
        hist_axs[row].set_ylabel('Offset')
        hist_axs[row].yaxis.tick_right()
        hist_axs[row].yaxis.set_label_position('right')
        hist_axs[row].set_xlabel('Prop.')
        
        # Vector length as title
        offset_len = np.abs(ts.offset_c.mean())
        hist_axs[row].set_title(f"vec length={offset_len:.2f}")

    plot_row('baseline', 0, plt.cm.get_cmap('Greens'))
    plot_row('post_heat', 1, plt.cm.get_cmap('Greens'))
    
    fig.suptitle(ts_dict['fly'])
    return fig, (heatmap_axs, hist_axs, cbar_ax)
