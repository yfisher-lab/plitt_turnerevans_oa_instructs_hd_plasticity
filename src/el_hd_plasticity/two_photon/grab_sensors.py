"""Session management and analysis for sensor grab imaging data.

This module provides classes and functions for managing and analyzing two-photon
imaging sessions, including statistical computations on offset, 
and fluorescence measurements across different behavioral conditions.
"""

import pathlib
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import SessionTools.two_photon as st2p
from . import session, utilities

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.gridspec import GridSpec as GS



class SessMaker():
    """Manage sensor grab imaging sessions and provide data access.
    
    Loads session metadata, filters usable sessions, and provides methods to
    access preprocessed imaging data and session properties.
    
    Attributes:
        basedir: Path to base directory containing session data.
        sess_df: DataFrame with session metadata.
        fig_folder: Path where figures will be saved.
    """

    def __init__(self, basedir=None, figfolder=None, mkdir=True, id_suffix=None):
        """Initialize session manager.
        
        Args:
            basedir: Path to base directory containing sessions.csv.
            figfolder: Path where output figures will be saved.
            mkdir: If True, create figfolder if it doesn't exist.
            id_suffix: Optional suffix to append to fly_id.
        """
        self.basedir = pathlib.Path(basedir)
        self.sess_df = pd.read_csv(self.basedir / 'sessions.csv')
        self._id_suffix = id_suffix
        self.fig_folder = pathlib.Path(figfolder)
        if mkdir:
            self.fig_folder.mkdir(parents=True, exist_ok=True)

        self.add_fly_id()
        self.filter_usable()
        

    def add_fly_id(self):
        """Create unique fly identifiers from date, fly, and optional suffix."""
        fly_id = []
        if self._id_suffix is None:
            for _, row in self.sess_df.iterrows():
                fly_id.append(row['date'] + '_' + row['fly'])
        else:
            for _, row in self.sess_df.iterrows():
                fly_id.append(row['date'] + '_' + row['fly'] + '_' + self._id_suffix)
            self.sess_df['grab'] = self._id_suffix
        
        self.sess_df['fly_id'] = fly_id

    def filter_usable(self):
        """Filter to only usable sessions (usable==1)."""
        self.sess_df = self.sess_df.loc[self.sess_df['usable'] == 1]

    def load_row(self, row):
        """Load preprocessed imaging data for a session row.
        
        Args:
            row: Session row from sess_df.
            
        Returns:
            EBImagingSession object with preprocessed data.
        """
        outdir = pathlib.PurePath(self.basedir / row['date'] / row['fly'] / row['session'] / 'preprocess.pkl')
        return st2p.preprocessing.EBImagingSession.from_file(outdir)
    
    @property
    def fly_ids(self):
        """Return array of unique fly identifiers."""
        return self.sess_df['fly_id'].unique()



def offset_stats(sess_cls):
    """Compute offset statistics for each session.
    
    Computes circular statistics of offset (difference between heading and PVA)
    for each imaging session.
    
    Args:
        sess_cls: SessMaker instance providing session metadata and data access.
        
    Returns:
        DataFrame with columns: fly_id, cl, offset_var, offset_mean, offset_mag.
    """
    rows = []

    for _, row in sess_cls.sess_df.iterrows():
        ts = session.GetTS(sess_cls.load_row(row))
        offset_c = ts.offset_c.mean()
        
        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'offset_var': sp.stats.circvar(ts.offset),
            'offset_mean': np.angle(offset_c),
            'offset_mag': np.abs(offset_c)
        })

    return pd.DataFrame(rows)
    
    
def offset_stats_unique(stats_df):
    """Aggregate offset statistics per fly by averaging across sessions.
    
    Reduces multiple sessions per condition to single values by averaging, creating
    one row per condition per fly. Requires flies to have both closed-loop and dark data.
    Applies logit transform for mixed effects modeling.
    
    Args:
        stats_df: Output from offset_stats().
        
    Returns:
        DataFrame with one row per condition per fly containing aggregated statistics.
    """
    rows = []

    for fly in stats_df['fly_id'].unique():
        cl_mask = (stats_df['fly_id'] == fly) & (stats_df['cl'] >= 1)
        dark_mask = (stats_df['fly_id'] == fly) & (stats_df['cl'] == 0)
        
        if cl_mask.sum() > 0 and dark_mask.sum() > 0:
            # Closed-loop aggregation
            cl_var = stats_df['offset_var'].loc[cl_mask].mean()
            cl_mean = sp.stats.circmean(stats_df['offset_mean'].loc[cl_mask])
            cl_mag = stats_df['offset_mag'].loc[cl_mask].mean()
            
            rows.append({
                'fly_id': fly,
                'cl': 1,
                'dark': 0,
                'offset_var': cl_var,
                'offset_var_logit': sp.special.logit(cl_var),
                'offset_mean': cl_mean,
                'offset_mag': cl_mag
            })
        
            # Dark aggregation
            dark_var = stats_df['offset_var'].loc[dark_mask].mean()
            dark_mean = sp.stats.circmean(stats_df['offset_mean'].loc[dark_mask])
            dark_mag = stats_df['offset_mag'].loc[dark_mask].mean()
            
            rows.append({
                'fly_id': fly,
                'cl': 0,
                'dark': 1,
                'offset_var': dark_var,
                'offset_var_logit': sp.special.logit(dark_var),
                'offset_mean': dark_mean,
                'offset_mag': dark_mag
            })
            
    return pd.DataFrame(rows)

def offset_stats_plot(stats_df_unique):
    """Reformat aggregated stats for side-by-side plotting (wide format).
    
    Pivots the long-format aggregated statistics into wide format with separate
    columns for dark and closed-loop conditions for each metric.
    
    Args:
        stats_df_unique: Output from offset_stats_unique().
        
    Returns:
        DataFrame with one row per fly, columns for dark and cl values of each metric.
    """
    rows = []
    
    for fly in stats_df_unique['fly_id'].unique():
        fly_mask = stats_df_unique['fly_id'] == fly
        fly_data = stats_df_unique.loc[fly_mask]
        
        dark_data = fly_data.loc[fly_data['dark'] == 1].iloc[0]
        cl_data = fly_data.loc[fly_data['dark'] == 0].iloc[0]
        
        rows.append({
            'fly_id': fly,
            'offset_var_dark': dark_data['offset_var'],
            'offset_var_closed_loop': cl_data['offset_var'],
            'offset_mag_dark': dark_data['offset_mag'],
            'offset_mag_closed_loop': cl_data['offset_mag']
        })
    
    return pd.DataFrame(rows)


def rho_stats(sess_cls, dh_bins, dataframe=False, load_row=None):
    """Compute motor neuron and fluorescence activity binned by heading velocity.
    
    Computes mean rho (motor neuron population vector magnitude) and outer ring
    fluorescence for each bin of absolute heading velocity.
    
    Args:
        sess_cls: SessMaker instance or DataFrame with session data.
        dh_bins: Bin edges for heading velocity.
        dataframe: If True, sess_cls is a DataFrame; if False, it's a SessMaker.
        load_row: Callable to load data (required if dataframe=True).
        
    Returns:
        DataFrame with rho_dig and F_dig: arrays of mean values for each bin.
    """
    rows = []
    
    if not dataframe:
        sess_df = sess_cls.sess_df
        load_row = sess_cls.load_row
    else:
        sess_df = sess_cls

    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row))
        
        # Compute heading velocity and bin by absolute value
        dh = np.diff(np.unwrap(ts.heading_sm)) / ts.dt
        dh = np.concatenate([[0], dh])
        dh_dig = np.digitize(np.abs(dh), dh_bins) - 1

        # Average rho and outer ring fluorescence for each bin
        rho_dig = np.array([ts.rho[dh_dig == i].mean() for i in range(len(dh_bins))])
        F_dig = np.array([ts.outer_ring[dh_dig == i].mean() for i in range(len(dh_bins))])
        
        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'rho_dig': rho_dig,
            'F_dig': F_dig
        })
        
    return pd.DataFrame(rows)
        
        
def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned rho stats to long format for regression modeling.
    
    Unpivots rho and fluorescence data from one row per session to one row per bin,
    suitable for linear mixed effects modeling. Includes log-transformed values.
    
    Args:
        grouped_stats: Output from rho_stats().
        dh_bins: Bin edges used in rho_stats().
        
    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded).
    """
    rows = []
    
    for _, row in grouped_stats.iterrows():
        for i, dh in enumerate(dh_bins):
            rho = row['rho_dig'][i]
            F = row['F_dig'][i]
            if not np.isnan(rho):  # Skip NaN values
                rows.append({
                    'fly_id': row['fly_id'],
                    'cl': row['cl'],
                    'rho': rho,
                    'log_rho': np.log(rho),
                    'dh': dh,
                    'F': F,
                    'log_F': np.log(F)
                })

    return pd.DataFrame(rows)


def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=np.arange(0, 360, 60),
                       cmap='Greys', twindow=None):
    """Plot DFF heatmap with heading and rotation speed time series.
    
    Creates a 3-row layout with neural activity heatmap, heading angle trace,
    and rotation speed trace.
    
    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        cmap: Colormap name for heatmap.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        
    Returns:
        Figure object.
    """
    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.5)
    
    ax_heatmap = fig.add_subplot(gs[0])
    ax_heading = fig.add_subplot(gs[1], sharex=ax_heatmap)
    ax_dh = fig.add_subplot(gs[2], sharex=ax_heatmap)

    def plot_rows(key, cmap):
        ts = ts_dict[key]
        dff = ts.dff
        time = ts.time

        # Apply time window mask if specified
        if twindow is not None:
            mask = (time >= twindow[0]) & (time <= twindow[1])
        else:
            mask = np.ones_like(time, dtype=bool)
        
        dff = dff[:, mask]
        time = time[mask]
        
        x = np.arange(dff.shape[1])
        
        # DFF heatmap
        h = ax_heatmap.imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                              interpolation='none')
        fig.colorbar(h, ax=ax_heatmap)
        
        # Heading trace
        ax_heading.scatter(x, ts.heading[mask] + np.pi, color='orange', s=2)
        ax_heading.set_ylim([2*np.pi, 0])
        fig.colorbar(h, ax=ax_heading)

        # Rotation speed trace
        ax_dh.plot(x, np.abs(ts.dh[mask]), color='black')
        fig.colorbar(h, ax=ax_dh)
        
        # Axes labels and formatting
        ax_heatmap.set_ylabel('ROIs')
        ax_heatmap.set_yticks([-0.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        _plot_times = plot_times[plot_times < time.iloc[-1]]
        ax_heatmap.set_xticks(utilities.get_time_tick_indices(time, _plot_times), labels=_plot_times)
        ax_heatmap.set_xlabel('Time (s)')

        ax_heading.set_yticks([0, np.pi, 2*np.pi], labels=[r'0', r'$\pi$', r'$2\pi$'])
        ax_heading.set_ylabel('Heading')

        ax_dh.set_ylim([0, 10])
        ax_dh.set_yticks([0, 10])
        ax_dh.set_ylabel('rot speed')
        
        ax_heatmap.set_title(key)
        
    # Plot all condition keys (skip 'fly' key)
    for key in ts_dict.keys():
        if key == 'fly':
            fig.suptitle(ts_dict[key])
        else:
            plot_rows(key, cmap)
    
    return fig

def plot_transpose_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=np.arange(0, 360, 60),
                            cmap='Greys', twindow=None):
    """Plot transposed DFF heatmap with heading and rotation speed.
    
    Creates a 3-column layout with neural activity heatmap (transposed for ROIs on x-axis),
    heading angle trace, and rotation speed trace.
    
    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on y-axis.
        cmap: Colormap name for heatmap.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        
    Returns:
        Figure object.
    """
    fig = plt.figure(figsize=[8, 15])
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 2, 2])
    
    ax_heatmap = fig.add_subplot(gs[0])
    ax_heading = fig.add_subplot(gs[1], sharey=ax_heatmap)
    ax_dh = fig.add_subplot(gs[2], sharey=ax_heatmap)

    def plot_cols(key, cmap):
        ts = ts_dict[key]
        dff = ts.dff
        time = ts.time

        # Apply time window mask if specified
        if twindow is not None:
            mask = (time >= twindow[0]) & (time <= twindow[1])
        else:
            mask = np.ones_like(time, dtype=bool)
        
        dff = dff[:, mask]
        time = time[mask]
        
        x = np.arange(dff.shape[1])
        
        # Transposed DFF heatmap (ROIs on x-axis, time on y-axis)
        h = ax_heatmap.imshow(dff.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax_heatmap)
        
        # Heading trace
        ax_heading.scatter(ts.heading[mask] + np.pi, x, color='orange', s=5)
        ax_heading.set_xlim([2*np.pi, 0])
        fig.colorbar(h, ax=ax_heading)

        # Rotation speed trace
        ax_dh.plot(np.abs(ts.dh[mask]), x, color='black')
        fig.colorbar(h, ax=ax_dh)
        
        # Axes labels and formatting
        ax_heatmap.set_xlabel('ROIs')
        ax_heatmap.set_xticks([-0.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        _plot_times = plot_times[plot_times < time.iloc[-1]]
        ax_heatmap.set_yticks(utilities.get_time_tick_indices(time, _plot_times), labels=_plot_times)
        ax_heatmap.set_ylabel('Time (s)')

        ax_heading.set_xticks([0, np.pi, 2*np.pi], labels=[r'0', r'$\pi$', r'$2\pi$'])
        ax_heading.set_xlabel('Heading')
        ax_heading.set_xlim([0, 2*np.pi])

        ax_dh.set_xlim([0, 10])
        ax_dh.set_xticks([0, 10])
        ax_dh.set_xlabel('rot speed')
        
        ax_heatmap.set_title(key)
        
    # Plot all condition keys (skip 'fly' key)
    for key in ts_dict.keys():
        if key == 'fly':
            fig.suptitle(ts_dict[key])
        else:
            plot_cols(key, cmap)
    
    fig.tight_layout()
    
    return fig



def plot_sess_histograms(ts_dict, bins=np.linspace(-np.pi, np.pi, num=17), 
                         cmap='Greys'):
    """Plot offset distributions and mean vectors.
    
    Creates histogram and polar plots showing offset tuning and population
    vector magnitude for each condition.
    
    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        bins: Bins for histogram computation.
        cmap: Colormap name for visualization.
        
    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
    centers = (bins[1:] + bins[:-1]) / 2

    def plot_hist(key, cmap, hatch=None):
        ts = ts_dict[key]
        hist, _ = np.histogram(ts.offset, bins=bins)
        hist = hist / hist.sum()
        
        offset_c_mu = ts.offset_c.mean()
        
        _cmap = plt.get_cmap(cmap)
        color = _cmap(.8)
        if hatch is not None:
            ax_hist.fill_between(centers, hist, color='none', alpha=.4, hatch=hatch, edgecolor=color)
            ax_polar.plot(np.angle(offset_c_mu) * np.ones([2]), [0, np.abs(offset_c_mu)], 
                         color=color, linewidth=2, label=key, linestyle='--', alpha=.4)
        else:
            ax_hist.fill_between(centers, hist, color=color, alpha=.4)
            ax_polar.plot(np.angle(offset_c_mu) * np.ones([2]), [0, np.abs(offset_c_mu)], 
                         color=color, linewidth=2, label=key)
        ax_hist.set_title(key)        
        ax_polar.set_title(key)
        
    # Plot all condition keys (skip 'fly' key)
    for key in ts_dict.keys():
        if key == 'fly':
            fig_hist.suptitle(ts_dict[key])
            fig_polar.suptitle(ts_dict[key])
        else:
            plot_hist(key, cmap)  
    
    # Histogram formatting
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax_hist.set_xlabel('Offset')
    ax_hist.set_yticks([0, .05, .1, .15, .2, .25])
    ax_hist.set_ylim([0, .25])
    ax_hist.set_ylabel('Proportion')
    fig_hist.tight_layout()    
    
    # Polar formatting
    ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax_polar.set_yticks([0, .2, .4, .6, .8])
    ax_polar.set_title(ts_dict['fly'])
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)


def _apply_time_window(dff, heading, time, twindow=None):
    """Apply a time window mask to data arrays.
    
    Args:
        dff: DFF data array.
        heading: Heading data array.
        time: Time array.
        twindow: Tuple of (start_time, end_time) or None for all data.
        
    Returns:
        Tuple of (dff_masked, heading_masked, time_masked, mask).
    """
    if twindow is not None:
        mask = (time >= twindow[0]) & (time <= twindow[1])
    else:
        mask = np.ones_like(time, dtype=bool)
    
    return dff[:, mask], heading[mask], time[mask], mask

def plot_sess_heatmaps_w_hist(ts_dict, bins=np.linspace(-np.pi, np.pi, num=17), vmin=-.5, vmax=.5,
                              plot_times=np.arange(0, 360, 60), twindow=None, cmap = plt.cm.Greens):
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
    fig = plt.figure(figsize=[8, 2])
    gs = GS(1, 4, figure=fig, width_ratios=[6, 1, .8, .2], wspace=.05, hspace=.8)
    heatmap_axs = [fig.add_subplot(gs[0]),]
    # heatmap_axs.extend([fig.add_subplot(gs[i, 0], sharey=heatmap_axs[0]) for i in range(1, 3)])
    hist_axs = [fig.add_subplot(gs[1]),]
    # hist_axs.extend([fig.add_subplot(gs[i, 1], sharex=hist_axs[0]) for i in range(1, 3)])
    # cbar_ax = [fig.add_subplot(gs[i, 3]) for i in range(3)]
    cbar_ax = [fig.add_subplot(gs[3]),]
    
    def plot_row(key, row, cmap, hatch=None):
        ts = ts_dict[key]
        dff, heading_masked, time, _ = _apply_time_window(ts.dff, ts.heading, ts.time, twindow)
        
        x = np.arange(dff.shape[1])
        heading_ = (heading_masked + np.pi) / (2 * np.pi) * 15
        h = heatmap_axs[row].imshow(dff, aspect='auto', cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        fig.colorbar(h, cax=cbar_ax[row])
        heatmap_axs[row].scatter(x, heading_, color='orange', s=5)

        heatmap_axs[row].set_ylabel('ROIs')
        heatmap_axs[row].set_yticks([-0.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        heatmap_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        
        _plot_times = plot_times[plot_times < time.iloc[-1]]
        heatmap_axs[row].set_xticks(utilities.get_time_tick_indices(time, _plot_times), labels=_plot_times)
        heatmap_axs[row].set_xlabel('Time (s)')
        heatmap_axs[row].set_title(key)

        centers = (bins[:-1] + bins[1:]) / 2
        hist, _ = np.histogram(ts.offset, bins=bins)
        hist = hist / hist.sum()
        
        if hatch is None:
            hist_axs[row].fill_betweenx(centers, 0, hist, color=cmap(.8), alpha=.5)
        else:
            hist_axs[row].fill_betweenx(centers, 0, hist, alpha=1, hatch=hatch, 
                                       color='none', edgecolor=cmap(.8))

        hist_axs[row].set_yticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', r'0', r'$\pi$'])
        hist_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        hist_axs[row].set_ylim([np.pi, -np.pi])
        hist_axs[row].grid(True, axis='y', linestyle='-', alpha=0.8, linewidth=2.5, which='major')
        ygridlines = hist_axs[row].get_ygridlines()
        ygridlines[1].set_color('orange')
        hist_axs[row].grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1.5, which='minor')
        hist_axs[row].set_ylabel('Offset')
        hist_axs[row].yaxis.tick_right()
        hist_axs[row].yaxis.set_label_position('right')
        hist_axs[row].set_xlabel('Prop.')

        offset_len = np.abs(ts.offset_c.mean())
        hist_axs[row].set_title(f"vec length={offset_len:.2f}")

    
    plot_row('closed loop', 0, cmap)
    hist_axs[0].set_xlim(left=0)
    
    fig.suptitle(ts_dict['fly'])
    return fig, (heatmap_axs, hist_axs, cbar_ax)