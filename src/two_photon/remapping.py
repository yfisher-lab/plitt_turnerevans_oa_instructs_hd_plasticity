import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from .utilities import get_time_tick_indices


def get_opto_resp(pp, zscore=False):
    """Extract per-ROI optogenetic responses aligned to stimulus offsets.

    Scans the voltage recording for opto trigger onsets, masks the immediate
    frames around triggers, computes dF/F (z-scored) and returns the mean
    response for a short window after each opto offset organized as
    (n_rois, n_points, n_reps).

    Args:
        pp: Preprocessed session object with attributes `voltage_recording_aligned`,
            `metadata`, and `timeseries` used below.
        zscore: If True, z-score the resulting opto responses across all values.

    Returns:
        opto_resp: ndarray shaped (n_rois, n_points, n_reps) containing mean responses.
    """
    time = pp.voltage_recording_aligned['Time(ms)']
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
def plot_min_dist(df, h_ax, x=np.arange(3), color='black', zero=False):
    if not 'post_novis1 mean offset' in df.columns:
    
        for _, row in df.iterrows():
            b = row['baseline mean offset']
            
            if zero:
                _p0 = row['post_0deg mean offset']
                p0 = np.array([_p0, _p0+2*np.pi, _p0-2*np.pi])
                p0 = p0[np.argmin(np.abs(b-p0))]
            else:
                p0 = row['post_0deg mean offset']
        
            
            _p180 = row['post_180deg mean offset']
            p180 = np.array([_p180, _p180+2*np.pi, _p180-2*np.pi])
            p180 = p180[np.argmin(np.abs(b-p180))]
            
            h_ax.scatter(x, [b,p0,p180], color=color)
            # h_ax.scatter(x, [b,p0,p180], color=plt.cm.hsv((b+np.pi)/(2*np.pi)), alpha=1)
            h_ax.plot(x, [b,p0,p180], color=color, alpha=.2)
            
    else:
        for _, row in df.iterrows():
            b = row['baseline mean offset']
            
            if zero:
                _p0 = row['post_novis1 mean offset']
                p0 = np.array([_p0, _p0+2*np.pi, _p0-2*np.pi])
                p0 = p0[np.argmin(np.abs(b-p0))]
            else:
                p0 = row['post_novis1 mean offset']
        
            
            _p180 = row['post_novis2 mean offset']
            p180 = np.array([_p180, _p180+2*np.pi, _p180-2*np.pi])
            p180 = p180[np.argmin(np.abs(b-p180))]
            
            h_ax.scatter(x, [b,p0,p180], color=color)
            # h_ax.scatter(x, [b,p0,p180], color=plt.cm.hsv((b+np.pi)/(2*np.pi)), alpha=1)
            h_ax.plot(x, [b,p0,p180], color=color, alpha=.2)

def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times = np.arange(0, 180, 60)):
    """Plot per-session heatmaps aligned to time and heading.
    
    Args:
        ts_dict: Dict with keys 'baseline', 'post_0deg', 'post_180deg' each with
                 attributes dff, time, heading, dff_h_aligned.
        vmin, vmax: Color limits for heatmap.
        plot_times: Time points (in seconds) to mark on x-axis.
    
    Returns:
        fig, ax: Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(3, 2, figsize=[15,6], sharey=True, sharex=True)
        
    
    def plot_row(key, row, cmap):
        dff = ts_dict[key].dff
        time = ts_dict[key].time
        x = np.arange(dff.shape[1])
        heading_ = (ts_dict[key].heading+np.pi)/(2*np.pi)*15
        h = ax[row,0].imshow(dff, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row,0])
        
        ax[row,0].scatter(x, heading_, color='orange', s=5)
        
        h = ax[row,1].imshow(ts_dict[key].dff_h_aligned, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(h, ax=ax[row,1])
        ax[row,1].scatter(x, 7.5*np.ones_like(heading_), color='orange', s=5)
        
        ax[row,0].set_ylabel('ROIs')
        ax[row,0].set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        
        ax[row,0].set_xticks(get_time_tick_indices(time, plot_times), labels=plot_times)
        ax[row,0].set_xlabel('Time (s)')
        
        ax[row,0].set_title(key)
        
        
        
    
    plot_row('baseline', 0, 'Greys')
    plot_row('post_0deg', 1, 'PuRd')
    plot_row('post_180deg', 2, 'GnBu')
    fig.suptitle(ts_dict['fly'])
    
    fig.tight_layout()
    
    return fig, ax
    
    
 
def plot_sess_histograms(ts_dict, bins = np.linspace(-np.pi, np.pi, num=17)):
    
    fig_hist, ax_hist = plt.subplots()
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection':'polar'})
    centers = (bins[1:]+bins[:-1])/2
    def plot_hist(key, color):
        offset = ts_dict[key].offset
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist/hist.sum()
        
        ax_hist.fill_between(centers, hist, color=color, alpha=.4)
        
        offset_c_mu = ts_dict[key].offset_c.mean()
        ax_polar.plot(np.angle(offset_c_mu)*np.ones([2,]), [0, np.abs(offset_c_mu)], color=color, linewidth=2, label=key)
        
        
    plot_hist('baseline', 'black')
    plot_hist('post_0deg', plt.cm.PuRd(.8))
    plot_hist('post_180deg', plt.cm.GnBu(.8))
    
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xticks([-np.pi,0,np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax_hist.set_xlabel('Offset')
    ax_hist.set_yticks([0,.05, .1,.15, .2, .25])
    ax_hist.set_ylim([0,.25])
    ax_hist.set_ylabel('Proportion')
    ax_hist.set_title(ts_dict['fly'])        
    fig_hist.tight_layout()    
    
    ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2'])
    ax_polar.set_yticks([0,.2, .4, .6, .8])
    ax_polar.set_ylim([0,.8])
    ax_polar.set_title(ts_dict['fly'])
    ax_polar.legend()
    fig_polar.tight_layout()
    
    return (fig_hist, ax_hist), (fig_polar, ax_polar)


def plot_sess_heatmaps_w_hist(ts_dict, bins = np.linspace(-np.pi, np.pi, num=17),vmin=-.5, vmax=3, 
                              plot_times = np.arange(0, 360, 60), twindow = None, sess_order = (0,1,2)):
    
    fig = plt.figure(figsize=[8,6])
    gs = GS(3,4, figure=fig, width_ratios=[6,1,.8, .2],wspace=.05,hspace=.8)
    heatmap_axs = [fig.add_subplot(gs[0,0])]
    heatmap_axs.extend([fig.add_subplot(gs[i,0],sharey=heatmap_axs[0]) for i in range(1,3)])
    hist_axs = [fig.add_subplot(gs[0,1])]
    hist_axs.extend([fig.add_subplot(gs[i,1],sharex=hist_axs[0]) for i in range(1,3)])
    cbar_ax = [fig.add_subplot(gs[i,3]) for i in range(3)]
    
    def plot_row(key, row, cmap):
        dff = ts_dict[key].dff
        heading = ts_dict[key].heading
        offset = ts_dict[key].offset
        
        time = ts_dict[key].time

        if twindow is not None:
            mask = (time>=twindow[0]) * (time<=twindow[1])
        else:
            mask = np.ones_like(time)>0
        dff= dff[:,mask]
        time = time[mask]
        heading = heading[mask]

        x = np.arange(dff.shape[1])
        heading_ = (heading + np.pi) / (2 * np.pi) * 15
        h = heatmap_axs[row].imshow(dff, aspect='auto', cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        fig.colorbar(h, cax=cbar_ax[row])
        heatmap_axs[row].scatter(x, heading_, color='orange', s=5)

        heatmap_axs[row].set_ylabel('ROIs')
        heatmap_axs[row].set_yticks([-0.5,7.5,15.5], labels=[r'0', r'$\pi$', r'$2\pi$'])
        heatmap_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        
        _plot_times = plot_times[plot_times<time.iloc[-1]]
        heatmap_axs[row].set_xticks(get_time_tick_indices(time, _plot_times), labels=_plot_times)
        heatmap_axs[row].set_xlabel('Time (s)')
        
        heatmap_axs[row].set_title(key)

        
      
        centers = (bins[:-1] + bins[1:]) / 2
        hist, _ = np.histogram(offset, bins=bins)
        hist = hist / hist.sum()  # normalize
       
        hist_axs[row].fill_betweenx(centers, 0, hist, color=cmap(.8), alpha=.5)
        
        
        hist_axs[row].set_yticks([-np.pi, 0,  np.pi], 
                                 labels=[r'-$\pi$',  r'0', r'$\pi$'])
        hist_axs[row].yaxis.set_minor_locator(AutoMinorLocator())
        hist_axs[row].set_ylim([np.pi, -np.pi])
        # hist_axs[row].set_xlim(left=0)
        hist_axs[row].grid(True, axis='y', linestyle='-', alpha=0.8,linewidth=2.5, which='major')
        ygridlines = hist_axs[row].get_ygridlines()
        ygridlines[1].set_color('orange')
        hist_axs[row].grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=1.5, which='minor')
        hist_axs[row].set_ylabel('Offset')
        hist_axs[row].yaxis.tick_right()
        hist_axs[row].yaxis.set_label_position('right')
        hist_axs[row].set_xlabel('Prop.')
        offset_len = np.abs(ts_dict[key].offset_c.mean())
        hist_axs[row].set_title(f"vec length={offset_len:.2f}" )
      

        


    plot_row('baseline', sess_order[0], plt.cm.Greys)
    plot_row('post_0deg', sess_order[1], plt.cm.PuRd)
    plot_row('post_180deg', sess_order[2], plt.cm.GnBu)
    
    hist_axs[0].set_xlim(left=0)
    fig.suptitle(ts_dict['fly'])
    return fig, (heatmap_axs, hist_axs, cbar_ax)  # Return the figure and axes for further customization if needed