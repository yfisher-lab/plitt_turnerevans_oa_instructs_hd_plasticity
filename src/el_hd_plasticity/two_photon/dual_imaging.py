"""Analysis and figures for dual-channel (EPG + EL) co-imaging sessions.

Sessions have two imaging channels (channel 0 = EPG, channel 1 = EL), so timeseries objects
carry per-channel ``dff``, ``phi``, ``rho`` and ``offset``. Statistics compare the two channels
(offset difference, PVA difference, widths) in closed loop and in the dark.

Two data sources are supported: preprocessed sessions listed in a session table (functions
taking ``sess_df`` and ``load_row``) and Dan Turner-Evans' trial dictionaries (``*_DTE``
functions).
"""

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as GS

from . import plotting, session, utilities
from .utilities import DEFAULT_OFFSET_BINS, DEFAULT_PLOT_TIMES, binned_mean, wrap_angle

CH_COLORS = ('Greys', 'Greens')  # default colormaps for (EPG, EL)


# ----------------------------------------------------------------------------
# Offset statistics
# ----------------------------------------------------------------------------
def _offset_stats_row(ts):
    """Per-session two-channel offset statistics for a :class:`session.SessionTimeseries`-like object."""
    mean_offsets = np.angle(ts.offset_c.mean(axis=-1))
    offset_var = sp.stats.circvar(ts.offset, axis=-1)
    offset_diff = wrap_angle(ts.offset[0, :] - ts.offset[1, :])
    pva_diff = np.abs(wrap_angle(np.diff(ts.phi, axis=0))).mean()

    return {
        'offset_ch1': mean_offsets[0],
        'offset_ch2': mean_offsets[1],
        'offset_var_ch1': offset_var[0],
        'offset_var_ch2': offset_var[1],
        'offset_diff': sp.stats.circmean(offset_diff),
        'abs_offset_diff': sp.stats.circmean(np.abs(offset_diff)),
        'pva_diff': pva_diff,
        'fwhm_ch1': ts.fwhm[0],
        'fwhm_ch2': ts.fwhm[1]
    }


def offset_stats_DTE(data, channel_lookup):
    """Compute offset statistics for each trial of the DTE dataset.

    Args:
        data: Dict mapping fly path -> dict of trial name -> trial dict.
        channel_lookup: Dict mapping fly path -> channel mapping for :meth:`session.TrialTimeseries.from_trial_dict`.

    Returns:
        DataFrame with fly_id, cl (always 1) and the two-channel offset statistics.
    """
    rows = []

    for fly, multitrial_dict in data.items():
        flyname = fly.split("\\")[-1]
        for trial_dict in multitrial_dict.values():
            ts = session.GetTS_DTE(trial_dict, channel_lookup[fly], dh_sigma=.3)
            rows.append({'fly_id': flyname, 'cl': 1, **_offset_stats_row(ts)})

    return pd.DataFrame(rows)


def offset_stats(sess_df, load_row):
    """Compute two-channel offset statistics for each session.

    Args:
        sess_df: Session DataFrame with fly_id and closed_loop columns.
        load_row: Callable to load preprocessed data.

    Returns:
        DataFrame with fly_id, cl and the two-channel offset statistics.
    """
    rows = []

    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=[0, 1])
        rows.append({'fly_id': row['fly_id'], 'cl': row['closed_loop'], **_offset_stats_row(ts)})

    return pd.DataFrame(rows)


def _aggregate_sessions(stats_df, mask, fly, cl, offset_diff_agg):
    """Average two-channel offset statistics over the sessions selected by `mask`."""
    sub = stats_df.loc[mask]
    return {
        'fly_id': fly,
        'cl': cl,
        'offset_ch1': sp.stats.circmean(sub['offset_ch1']),
        'offset_ch2': sp.stats.circmean(sub['offset_ch2']),
        'offset_var_ch1': sub['offset_var_ch1'].mean(),
        'offset_var_ch2': sub['offset_var_ch2'].mean(),
        'offset_diff': offset_diff_agg(sub['offset_diff']),
        'abs_offset_diff': sub['abs_offset_diff'].mean(),
        'pva_diff': sub['pva_diff'].mean(),
        'fwhm_ch1': sub['fwhm_ch1'].mean(),
        'fwhm_ch2': sub['fwhm_ch2'].mean()
    }


def offset_stats_unique(stats_df, cl_only=False):
    """Aggregate offset statistics per fly, comparing closed loop and dark conditions.

    Args:
        stats_df: Output from :func:`offset_stats` or :func:`offset_stats_DTE`.
        cl_only: If True, keep flies that have closed-loop sessions even without dark sessions
            (no dark row is produced for them); otherwise require both.

    Returns:
        DataFrame with one closed-loop (cl=1) and, if available, one dark (cl=0) row per fly.
    """
    # NOTE: the closed-loop row wraps the mean offset difference to (-pi, pi] whereas the dark row
    # uses scipy's circmean convention [0, 2pi). The conventions are kept as they were when the
    # published numbers were generated.
    def cl_diff_agg(offset_diff):
        """Circular mean of the offset difference, wrapped to (-pi, pi]."""
        return np.angle(np.exp(1j * offset_diff).mean())

    rows = []

    for fly in stats_df['fly_id'].unique():
        fly_mask = stats_df['fly_id'] == fly
        cl_mask = fly_mask & (stats_df['cl'] >= 1)
        dark_mask = fly_mask & (stats_df['cl'] == 0)

        has_cl = cl_mask.sum() > 0
        has_dark = dark_mask.sum() > 0

        if (has_cl if cl_only else (has_cl and has_dark)):
            rows.append(_aggregate_sessions(stats_df, cl_mask, fly, 1, cl_diff_agg))
            if has_dark:
                rows.append(_aggregate_sessions(stats_df, dark_mask, fly, 0, sp.stats.circmean))

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Statistics binned by heading velocity / rho
# ----------------------------------------------------------------------------
def _rho_stats_binned(ts, dh_bins, abs):
    """Two-channel rho and PVA difference binned by heading velocity (see :func:`rho_stats`)."""
    n_bins = len(dh_bins)
    dh_dig = np.digitize(np.abs(ts.dh) if abs else ts.dh, dh_bins) - 1

    pvd = wrap_angle(np.diff(ts.phi, axis=0))
    pvd = (np.abs(pvd) if abs else pvd).ravel()

    return {
        'rho1_dig': binned_mean(ts.rho[0], dh_dig, n_bins),
        'rho2_dig': binned_mean(ts.rho[1], dh_dig, n_bins),
        'pva_diff': binned_mean(pvd, dh_dig, n_bins)
    }


def rho_stats(sess_df, load_row, dh_bins, abs=True):
    """Compute rho (PVA magnitude) of both channels and the PVA difference binned by heading velocity.

    Args:
        sess_df: Session DataFrame with fly_id and closed_loop columns.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.
        abs: If True, bin by absolute heading velocity and use the absolute PVA difference.

    Returns:
        DataFrame with fly_id, cl, rho1_dig, rho2_dig and pva_diff (arrays, one value per bin).
    """
    rows = []

    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=[0, 1], dh_sigma=.3)
        rows.append({'fly_id': row['fly_id'], 'cl': row['closed_loop'],
                     **_rho_stats_binned(ts, dh_bins, abs)})

    return pd.DataFrame(rows)


def rho_stats_DTE(data, channel_lookup, dh_bins, abs=True):
    """Compute rho statistics from DTE data binned by heading velocity (see :func:`rho_stats`)."""
    rows = []

    for fly, multitrial_dict in data.items():
        for trial_dict in multitrial_dict.values():
            ts = session.GetTS_DTE(trial_dict, channel_lookup[fly], dh_sigma=.3)
            rows.append({'fly_id': fly, 'cl': 1, **_rho_stats_binned(ts, dh_bins, abs)})

    return pd.DataFrame(rows)


def pvdiff_rho_stats(sess_df, load_row, pv_bins):
    """Compute the PVA difference between channels binned by mean rho (PVA magnitude).

    Args:
        sess_df: Session DataFrame with fly_id and closed_loop columns.
        load_row: Callable to load preprocessed data.
        pv_bins: Bin edges for the channel-averaged rho.

    Returns:
        DataFrame with fly_id, cl and pva_diff (array, one value per bin).
    """
    rows = []

    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=[0, 1])

        pva_diff = np.abs(wrap_angle(np.diff(ts.phi, axis=0))).ravel()
        rho_dig = np.digitize(ts.rho.mean(axis=0), bins=pv_bins)

        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'pva_diff': binned_mean(pva_diff, rho_dig, pv_bins.shape[0])
        })

    return pd.DataFrame(rows)


def cross_corr_stats(sess_df, load_row, delays=None, times=None):
    """Compute the cross-correlation of the two channels' PVA as a function of delay.

    Args:
        sess_df: Session DataFrame with fly_id and closed_loop columns.
        load_row: Callable to load preprocessed data.
        delays: Integer sample delays applied to channel 2 (default -50..50).
        times: Times (s) at which to evaluate the interpolated correlation (default
            ``delays * 0.01``).

    Returns:
        DataFrame with fly, cl, r (correlation at `times`) and argmax (index of the peak).
    """
    delays = np.arange(-50, 51) if delays is None else delays
    times = delays * 0.01 if times is None else times
    rows = []

    for _, row in sess_df.iterrows():
        ts = session.GetTS(load_row(row), channels=(0, 1))

        r = np.array([np.abs(np.correlate(np.exp(1j * ts.phi[0, :]),
                                          np.exp(1j * np.roll(ts.phi[1, :], d)))
                             / ts.phi.shape[-1])[0]
                      for d in delays])

        f = sp.interpolate.interp1d(delays * ts.dt, r, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        r_interp = f(times)

        rows.append({
            'fly': row['fly_id'],
            'cl': row['closed_loop'],
            'r': r_interp,
            'argmax': np.argmax(r_interp)
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _channel_traces(ts, cmaps):
    """Per-channel :class:`plotting.Trace` list for (EPG, EL)."""
    return [plotting.Trace.from_ts(label, ts, cmap=cmap, channel=ch)
            for ch, (label, cmap) in enumerate(zip(('EPGs', 'ELs'), cmaps, strict=True))]


def plot_sess_heatmaps(ts, fly_id, sess_name, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES,
                       ch1_heatmap='Greys', ch2_heatmap='Greens'):
    """Heatmaps of each channel (raw and heading-aligned) plus the PVA of both channels and their difference.

    Args:
        ts: Two-channel :class:`session.SessionTimeseries`.
        fly_id: Fly identifier for the title.
        sess_name: Session name for the title.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Times (s) to mark on the time axis.
        ch1_heatmap: Colormap for channel 1.
        ch2_heatmap: Colormap for channel 2.

    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(3, 2, figsize=[15, 6], sharey=True, sharex=True)

    x = np.arange(ts.dff.shape[-1])
    heading_ = utilities.heading_to_roi_coords(ts.heading)
    aligned_pos = plotting.ALIGNED_HEADING_POS * np.ones_like(heading_)

    for row, (cmap, ch) in enumerate(((ch1_heatmap, 0), (ch2_heatmap, 1))):
        h = ax[row, 0].imshow(ts.dff[ch, :, :], aspect='auto', interpolation='none',
                              cmap=cmap, vmin=vmin, vmax=vmax)
        ax[row, 0].scatter(x, heading_, s=5, color='orange')
        fig.colorbar(h, ax=ax[row, 0])

        h = ax[row, 1].imshow(ts.dff_h_aligned[ch, :, :], aspect='auto', interpolation='none',
                              cmap=cmap, vmin=vmin, vmax=vmax)
        ax[row, 1].scatter(x, aligned_pos, s=5, color='orange')
        fig.colorbar(h, ax=ax[row, 1])

    ax[0, 0].set_title('Ch 1')
    ax[1, 0].set_title('Ch 2')

    phi_ = utilities.heading_to_roi_coords(ts.phi)
    ax[2, 0].scatter(x, phi_[0, :], color=plt.get_cmap(ch1_heatmap)(.8), s=5, alpha=.4)
    ax[2, 0].scatter(x, phi_[1, :], color=plt.get_cmap(ch2_heatmap)(.8), s=5, alpha=.4)
    fig.colorbar(h, ax=ax[2, 0])

    phi_diff = utilities.heading_to_roi_coords(wrap_angle(np.diff(ts.phi, axis=0)))
    ax[2, 1].scatter(x, phi_diff, c='blue', s=5)
    fig.colorbar(h, ax=ax[2, 1])

    plot_times = plot_times[plot_times < ts.time.iloc[-1]]
    for a in ax.flatten():
        plotting.format_roi_axis(a, 'y')
        plotting.format_time_axis(a, ts.time, plot_times, 'x', clip=False)

    fig.suptitle(f'{fly_id} - {sess_name}')
    fig.tight_layout()

    return fig, ax


def plot_pva_diff_histograms(ts, fly_id, sess_name,
                             bins=DEFAULT_OFFSET_BINS,
                             color='black'):
    """Histogram of the PVA difference between the two channels.

    Args:
        ts: Two-channel :class:`session.SessionTimeseries`.
        fly_id: Fly identifier for the title.
        sess_name: Session name for the title.
        bins: Bins for the histogram.
        color: Fill color.

    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots()

    pva_diff = wrap_angle(np.diff(ts.phi, axis=0))
    hist, _ = utilities.normalized_histogram(pva_diff, bins)

    ax.fill_between(bins[:-1], hist, color=color, alpha=.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
    ax.set_xlabel(r'$\Delta$ PVA')
    ax.set_ylabel('Proportion')

    fig.suptitle(f'{fly_id} - {sess_name}')
    fig.tight_layout()
    return fig, ax


def plot_sess_histograms(ts, fly_id, sess_name, bins=DEFAULT_OFFSET_BINS,
                         cmaps=CH_COLORS):
    """Offset histograms and polar mean-vector plots for both channels.

    Args:
        ts: Two-channel :class:`session.SessionTimeseries`.
        fly_id: Fly identifier for the title.
        sess_name: Session name for the title.
        bins: Bins for the histogram.
        cmaps: Colormaps for (EPG, EL); the color drawn is each colormap at 0.8.

    Returns:
        Tuple of ((fig_hist, ax_hist), (fig_polar, ax_polar)).
    """
    return plotting.plot_offset_histograms(_channel_traces(ts, cmaps), bins=bins,
                                           suptitle=f'{fly_id} - {sess_name}', legend=False)


def plot_transpose_heatmaps(ts, fly_id, sess_name, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES,
                            ch1_heatmap='Greys', ch2_heatmap='Greens'):
    """Transposed (time down, ROIs across) heatmaps of both channels, their PVAs and PVA difference.

    Args:
        ts: Two-channel :class:`session.SessionTimeseries`.
        fly_id: Fly identifier for the title.
        sess_name: Session name for the title.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Times (s) to mark on the time axis.
        ch1_heatmap: Colormap for channel 1.
        ch2_heatmap: Colormap for channel 2.

    Returns:
        Tuple of (figure, axes).
    """
    fig = plt.figure(figsize=[9, 15])
    gs = GS(1, 4, width_ratios=[2, 2, 1, 1])
    ax = [fig.add_subplot(gs[i]) for i in range(4)]

    x = np.arange(ts.dff.shape[-1])
    heading_ = utilities.heading_to_roi_coords(ts.heading)

    for a, cmap, ch in ((ax[0], ch1_heatmap, 0), (ax[1], ch2_heatmap, 1)):
        h = a.imshow(ts.dff[ch, :, :].T, aspect='auto', interpolation='none',
                     cmap=cmap, vmin=vmin, vmax=vmax)
        a.scatter(heading_, x, s=5, color='orange')
        fig.colorbar(h, ax=a)

    ax[0].set_title('Ch 1')
    ax[1].set_title('Ch 2')

    phi_ = utilities.heading_to_roi_coords(ts.phi)
    ax[2].scatter(phi_[0, :], x, color=plt.get_cmap(ch1_heatmap)(.8), s=5, alpha=.4)
    ax[2].scatter(phi_[1, :], x, color=plt.get_cmap(ch2_heatmap)(.8), s=5, alpha=.4)
    fig.colorbar(h, ax=ax[2])

    phi_diff = utilities.heading_to_roi_coords(wrap_angle(np.diff(ts.phi, axis=0)))
    ax[3].scatter(phi_diff, x, c='blue', s=5)
    fig.colorbar(h, ax=ax[3])

    plot_times = plot_times[plot_times < ts.time.iloc[-1]]
    for a in ax:
        plotting.format_roi_axis(a, 'x')
        plotting.format_time_axis(a, ts.time, plot_times, 'y', clip=False)
        a.set_ylim([ts.dff.shape[-1] - 0.5, -.5])

    fig.suptitle(f'{fly_id} - {sess_name}')
    fig.tight_layout()

    return fig, ax


def plot_sess_heatmaps_w_hist(ts, fly_id, sess_name, vmin=-.5, vmax=3, plot_times=DEFAULT_PLOT_TIMES,
                              ch1_heatmap='Greys', ch2_heatmap='Greens', twindow=None,
                              bins=DEFAULT_OFFSET_BINS):
    """EPG and EL heatmaps, each with its offset distribution (titled with the offset variance).

    Args:
        ts: Two-channel :class:`session.SessionTimeseries`.
        fly_id: Fly identifier for the title.
        sess_name: Session name for the title.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Times (s) to mark on the time axis.
        ch1_heatmap: Colormap for channel 1 (EPGs).
        ch2_heatmap: Colormap for channel 2 (ELs).
        twindow: Optional tuple (start_time, end_time) to limit the heatmap window.
        bins: Bins for the offset histogram.

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    return plotting.plot_heatmaps_with_offset_hist(
        _channel_traces(ts, (ch1_heatmap, ch2_heatmap)), f'{fly_id} - {sess_name}', figsize=[8, 4],
        vmin=vmin, vmax=vmax, plot_times=plot_times, twindow=twindow, bins=bins, hist_title='variance')
