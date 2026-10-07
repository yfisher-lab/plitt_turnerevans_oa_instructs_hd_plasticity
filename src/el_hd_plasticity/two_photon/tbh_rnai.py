"""Analysis and visualization of tbh RNAi fly two-photon imaging data.

This module provides functions for computing behavioral statistics (offset, rho, phi)
from two-photon imaging sessions of tbh RNAi flies and visualization functions.
Compares responses between closed loop and dark conditions across different RNAi lines.
"""

import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from . import session, utilities




def offset_stats(sess_df, load_row, **kwargs):
    """Compute offset statistics (variance, mean, magnitude) for each session.
    
    Computes circular statistics on EPG offset (difference between heading and PVA)
    for each imaging session across different closed-loop and dark conditions.
    
    Args:
        sess_df: DataFrame with session metadata including fly_id, closed_loop, dark, rnai columns.
        load_row: Callable that loads preprocessed data for a session row.
        
    Returns:
        DataFrame with columns: fly_id, cl, rnai, dark, offset_var, offset_mean, offset_mag.
    """
    rows = []
    
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), **kwargs)
        offset_c = ts.offset_c.mean()
        
        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'dark': row['dark'],
            'rnai': row['rnai'],
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
        # cl_mask: cl>1 means at least 10 min prior closed-loop experience
        cl_mask = (stats_df['fly_id'] == fly) & (stats_df['cl'] > 1)
        dark_mask = (stats_df['fly_id'] == fly) & (stats_df['dark'] >= 1)
        
        # Only include flies with both conditions
        if cl_mask.sum() > 0 and dark_mask.sum() > 0:
            rnai_val = stats_df['rnai'].loc[cl_mask].iloc[0]
            
            # Closed-loop aggregation
            cl_var = stats_df['offset_var'].loc[cl_mask].mean()
            cl_mean = sp.stats.circmean(stats_df['offset_mean'].loc[cl_mask])
            cl_mag = stats_df['offset_mag'].loc[cl_mask].mean()
            
            rows.append({
                'fly_id': fly,
                'cl': 1,
                'rnai': rnai_val,
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
                'rnai': rnai_val,
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
            'rnai': dark_data['rnai'],
            'offset_var_dark': dark_data['offset_var'],
            'offset_var_closed_loop': cl_data['offset_var'],
            'offset_mag_dark': dark_data['offset_mag'],
            'offset_mag_closed_loop': cl_data['offset_mag']
        })
    
    return pd.DataFrame(rows)


def rho_stats(sess_df, load_row, dh_bins):
    """Compute pva magnitude binned by heading velocity.
    
    Computes mean rho (population vector average magnitude) for each bin of
    absolute heading velocity.
    
    Args:
        sess_df: Session  DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.
        
    Returns:
        DataFrame with rho_dig: array of mean rho values for each dh bin.
    """
    rows = []
    
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row))
        
        # Compute heading velocity and bin by absolute value
        dh = np.diff(np.unwrap(ts.heading_sm)) / ts.dt
        dh = np.concatenate([[0], dh])
        dh_dig = np.digitize(np.abs(dh), dh_bins) - 1
        
        # Average rho for each bin
        rho_dig = np.array([ts.rho[dh_dig == i].mean() for i in range(len(dh_bins))])
        
        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'rnai': row['rnai'],
            'dark': row['dark'],
            'rho_dig': rho_dig
        })
    
    return pd.DataFrame(rows)
        
        
def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned rho stats to long format for regression modeling.
    
    Unpivots rho data from one row per session to one row per bin, suitable
    for linear mixed effects modeling.
    
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
            if not np.isnan(rho):  # Skip NaN values
                rows.append({
                    'fly_id': row['fly_id'],
                    'rnai': row['rnai'],
                    'dark': row['dark'],
                    'rho': rho,
                    'dh': dh
                })
    
    return pd.DataFrame(rows)


def dphi_stats(sess_df, load_row, dh_bins):
    """Compute PVA angular velocity binned by heading velocity.
    
    Measures how fast the population vector angle (PVA) rotates as a function of
    turning speed.
    
    Args:
        sess_df: Session DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.
        
    Returns:
        DataFrame with dphi_dig: array of mean PVA velocity for each dh bin.
    """
    rows = []
    
    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), t_sigma=.1, h_sigma=.1)
        
        # Compute velocities
        dh = np.diff(np.unwrap(ts.heading_sm)) / ts.dt
        d_phi = np.diff(np.unwrap(ts.phi)) / ts.dt
        
        dh_dig = np.digitize(dh, dh_bins) - 1
        
        # Average PVA velocity for each bin
        dphi_dig = np.array([d_phi[dh_dig == i].mean() for i in range(len(dh_bins))])
        
        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'rnai': row['rnai'],
            'dark': row['dark'],
            'dh_dig': dh_dig,
            'dphi_dig': dphi_dig
        })
    
    return pd.DataFrame(rows)
        
        
def reformat_dphi_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned PVA velocity stats to long format for regression modeling.
    
    Unpivots dphi data from one row per session to one row per bin.
    
    Args:
        grouped_stats: Output from dphi_stats().
        dh_bins: Bin edges used in dphi_stats().
        
    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded).
    """
    rows = []
    
    for _, row in grouped_stats.iterrows():
        for i, dh in enumerate(dh_bins):
            dphi = row['dphi_dig'][i]
            if not np.isnan(dphi):  # Skip NaN values
                rows.append({
                    'fly_id': row['fly_id'],
                    'rnai': row['rnai'],
                    'dark': row['dark'],
                    'dphi': dphi,
                    'dh': dh
                })
    
    return pd.DataFrame(rows)


def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=np.arange(0, 360, 60)):
    """Plot DFF heatmaps for closed-loop and dark conditions side-by-side.
    
    Shows raw and heading-aligned neural activity for comparison between conditions.
    
    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        
    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(2, 2, figsize=[15, 4], sharey=True, sharex=True)
    
    def plot_row(key, row, cmap):
        ts = ts_dict[key]
        dff = ts.dff
        time = ts.time
        x = np.arange(dff.shape[1])
        heading_ = (ts.heading + np.pi) / (2 * np.pi) * 15
        
        # DFF heatmap
        h = ax[row, 0].imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row, 0])
        ax[row, 0].scatter(x, heading_, color='orange', s=5)
        
        # Heading-aligned DFF
        h = ax[row, 1].imshow(ts.dff_h_aligned, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row, 1])
        ax[row, 1].scatter(x, 7.5 * np.ones_like(heading_), color='orange', s=5)
        
        ax[row, 0].set_ylabel('ROIs')
        ax[row, 0].set_yticks([-0.5, 7.5, 15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        ax[row, 0].set_xticks(utilities.get_time_tick_indices(time, plot_times), labels=plot_times)
        ax[row, 0].set_xlabel('Time (s)')
        ax[row, 0].set_title(key)
    
    plot_row('closed_loop', 0, 'Greys')
    plot_row('dark', 1, 'Purples')
    fig.suptitle(ts_dict['fly'])
    fig.tight_layout()
    
    return fig, ax

def plot_sess_histograms(ts_dict, bins=np.linspace(-np.pi, np.pi, num=17)):
    """Plot offset distributions and mean vectors for closed-loop and dark conditions.
    
    Creates histogram and polar plots showing PVA offset tuning and population
    vector magnitude for each condition.
    
    Args:
        ts_dict: Dictionary with 'fly' key and condition keys containing timeseries objects.
        bins: Bins for histogram computation.
        
    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
    centers = (bins[1:] + bins[:-1]) / 2
    
    def plot_hist(key, color):
        ts = ts_dict[key]
        hist, _ = np.histogram(ts.offset, bins=bins)
        hist = hist / hist.sum()
        
        ax_hist.fill_between(centers, hist, color=color, alpha=.4)
        
        offset_c_mu = ts.offset_c.mean()
        ax_polar.plot(np.angle(offset_c_mu) * np.ones([2]), [0, np.abs(offset_c_mu)], 
                     color=color, linewidth=2, label=key)
    
    plot_hist('closed_loop', 'black')
    plot_hist('dark', plt.cm.Purples(.8))
    
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax_hist.set_xlabel('Offset')
    ax_hist.set_yticks([0, .05, .1, .15, .2, .25])
    ax_hist.set_ylim([0, .25])
    ax_hist.set_ylabel('Proportion')
    ax_hist.set_title(ts_dict['fly'])
    fig_hist.tight_layout()
    
    ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax_polar.set_yticks([0, .2, .4, .6, .8])
    ax_polar.set_ylim([0, .8])
    ax_polar.set_title(ts_dict['fly'])
    ax_polar.legend()
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)


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
        heatmap_axs[row].scatter(x, heading_, color='orange', s=5)

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

    plot_row('closed_loop', 0, plt.cm.get_cmap('Greys'))
    plot_row('dark', 1, plt.cm.get_cmap('Purples'))
    
    fig.suptitle(ts_dict['fly'])
    return fig, (heatmap_axs, hist_axs, cbar_ax)
