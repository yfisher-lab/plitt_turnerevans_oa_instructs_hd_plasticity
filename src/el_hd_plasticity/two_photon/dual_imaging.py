"""Analysis and figures for dual-channel (EPG + EL) co-imaging sessions.

Sessions have two imaging channels (channel 0 = EPG, channel 1 = EL), so timeseries objects
carry per-channel ``dff``, ``phi``, ``rho`` and ``offset``. Statistics compare the two channels
(offset difference, PVA difference, widths) in closed loop and in the dark.

Two data sources are supported: preprocessed sessions managed by a
:class:`~loading.SessionManager` (functions taking ``session_manager``) and Dan Turner-Evans' trial
dictionaries (``*_DTE`` functions).
"""

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as GS
from matplotlib.ticker import AutoMinorLocator

from . import bump_shape, plotting, session, stats, utilities
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


def offset_stats(session_manager):
    """Compute two-channel offset statistics for each session.

    Args:
        session_manager: Session manager object (session table with fly_id and closed_loop columns).

    Returns:
        DataFrame with fly_id, cl and the two-channel offset statistics.
    """
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), channels=[0, 1])
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


def rho_stats(session_manager, dh_bins, abs=True):
    """Compute rho (PVA magnitude) of both channels and the PVA difference binned by heading velocity.

    Args:
        session_manager: Session manager object.
        dh_bins: Bin edges for heading velocity.
        abs: If True, bin by absolute heading velocity and use the absolute PVA difference.

    Returns:
        DataFrame with fly_id, cl, rho1_dig, rho2_dig and pva_diff (arrays, one value per bin).
    """
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), channels=[0, 1], dh_sigma=.3)
        rows.append({'fly_id': row['fly_id'], 'cl': row['closed_loop'],
                     **_rho_stats_binned(ts, dh_bins, abs)})

    return pd.DataFrame(rows)


def rho_stats_DTE(data, channel_lookup, dh_bins, abs=True):
    """Compute rho statistics from DTE data binned by heading velocity (see :func:`rho_stats`)."""
    rows = []

    for fly, multitrial_dict in data.items():
        flyname = fly.split("\\")[-1]
        for trial_dict in multitrial_dict.values():
            ts = session.GetTS_DTE(trial_dict, channel_lookup[fly], dh_sigma=.3)
            rows.append({'fly_id': flyname, 'cl': 1, **_rho_stats_binned(ts, dh_bins, abs)})

    return pd.DataFrame(rows)


def pvdiff_rho_stats(session_manager, pv_bins):
    """Compute the PVA difference between channels binned by mean rho (PVA magnitude).

    Args:
        session_manager: Session manager object.
        pv_bins: Bin edges for the channel-averaged rho; the last bin is open-ended.

    Returns:
        DataFrame with fly_id, cl and pva_diff (array, one value per bin).
    """
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), channels=[0, 1])

        pva_diff = np.abs(wrap_angle(np.diff(ts.phi, axis=0))).ravel()
        rho_dig = np.digitize(ts.rho.mean(axis=0), bins=pv_bins) - 1     # bin i = [pv_bins[i], pv_bins[i + 1])

        rows.append({
            'fly_id': row['fly_id'],
            'cl': row['closed_loop'],
            'pva_diff': binned_mean(pva_diff, rho_dig, pv_bins.shape[0])
        })

    return pd.DataFrame(rows)


def cross_corr_stats(session_manager, delays=None, times=None):
    """Compute the cross-correlation of the two channels' PVA as a function of delay.

    Args:
        session_manager: Session manager object.
        delays: Integer sample delays applied to channel 2 (default -50..50).
        times: Times (s) at which to evaluate the interpolated correlation (default
            ``delays * 0.01``).

    Returns:
        DataFrame with fly, cl, r (correlation at `times`) and argmax (index of the peak).
    """
    delays = np.arange(-50, 51) if delays is None else delays
    times = delays * 0.01 if times is None else times
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), channels=(0, 1))

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
# Bump shape
# ----------------------------------------------------------------------------
def _bump_profiles(ts, rho_min):
    """Time-averaged phase-nulled bump profile of each channel, shape (n_channels, n_rois)."""
    return [bump_shape.mean_phase_null_profile(channel, rho_min=rho_min) for channel in ts.dff]


def _bump_rows(fly_id, cl, ts, rho_min):
    return [{'fly_id': fly_id, 'cl': cl, 'channel': channel, 'profile': profile}
            for channel, profile in enumerate(_bump_profiles(ts, rho_min))]


def bump_stats(session_manager, rho_min=.05):
    """Time-averaged, phase-nulled bump profile of each channel of each session.

    Args:
        session_manager: Session manager object.
        rho_min: Minimum first-harmonic magnitude for a time point to be included (see
            :func:`bump_shape.mean_phase_null_profile`); applied to each channel separately.

    Returns:
        DataFrame with one row per session and channel: fly_id, cl, channel (0 / 1) and ``profile``, an
        array of shape (n_rois,) with the bump peak in the middle.
    """
    rows = []
    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), channels=[0, 1])
        rows += _bump_rows(row['fly_id'], row['closed_loop'], ts, rho_min)
    return pd.DataFrame(rows)


def bump_stats_DTE(data, channel_lookup, rho_min=.05):
    """:func:`bump_stats` for the DTE dataset (one row per trial and channel; ``cl`` is always 1)."""
    rows = []
    for fly, multitrial_dict in data.items():
        for trial_dict in multitrial_dict.values():
            ts = session.GetTS_DTE(trial_dict, channel_lookup[fly])
            rows += _bump_rows(fly.split("\\")[-1], 1, ts, rho_min)
    return pd.DataFrame(rows)


def group_bump_stats(bump_stats_df, **kwargs):
    """Average the bump profiles over the sessions of each fly, condition and channel and measure their width.

    Args:
        bump_stats_df: Output of :func:`bump_stats` or :func:`bump_stats_DTE`.
        **kwargs: Passed to :func:`stats.group_bump_stats` (``factor``, ``baseline``).

    Returns:
        DataFrame with one row per fly, cl and channel: the mean ``profile`` and its ``fwhm`` (radians).
    """
    return stats.group_bump_stats(bump_stats_df, group_cols=('fly_id', 'cl', 'channel'), **kwargs)


# ----------------------------------------------------------------------------
# Activity during fast turns
# ----------------------------------------------------------------------------
def turn_triggered_profiles(timeseries, dh_min, dh_max, sortby_index=0, alignment='phase'):
    """Mean bump profile of each channel during fast counter-clockwise and clockwise turns.

    For every timeseries the time points with ``dh_min < dh < dh_max`` (counter-clockwise) or
    ``-dh_max < dh < -dh_min`` (clockwise) are shifted so the bump of channel `sortby_index` sits at the
    center ROI (both channels by the same amount), averaged over time, and min-max normalized per channel.
    Timeseries without such time points are skipped for that direction.

    Args:
        timeseries: Iterable of two-channel timeseries (``dff`` of shape (channels, ROIs, time) and ``dh``,
            see :func:`session.GetTS` and :func:`session.GetTS_DTE` with ``dh_sigma=.3``).
        dh_min, dh_max: Range of turning speeds (rad/s).
        sortby_index: Channel whose bump defines the shift applied to both channels.
        alignment: 'phase' (default) shifts to sub-ROI precision by nulling the first-harmonic phase, as in
            the bump shape analyses (:func:`bump_shape.shift_by_phase`); 'max' rolls by whole ROIs so the
            peak ROI is at the center (:func:`bump_shape.shift_by_max`).

    Returns:
        Tuple (ccw, cw) of arrays of shape (n_timeseries, n_channels, n_rois).
    """
    shifts = {'phase': bump_shape.shift_by_phase, 'max': bump_shape.shift_by_max}
    if alignment not in shifts:
        raise ValueError(f"alignment must be 'phase' or 'max', not {alignment!r}")
    ccw, cw = [], []
    for ts in timeseries:
        for mask, out in ((np.logical_and(ts.dh > dh_min, ts.dh < dh_max), ccw),
                          (np.logical_and(ts.dh < -dh_min, ts.dh > -dh_max), cw)):
            if mask.sum() > 0:
                out.append(shifts[alignment](ts.dff[:, :, mask], sortby_index).mean(axis=-1))

    def normalize(arr):
        arr = np.array(arr)
        lo, hi = arr.min(axis=-1, keepdims=True), arr.max(axis=-1, keepdims=True)
        return (arr - lo) / (hi - lo)

    return normalize(ccw), normalize(cw)


def plot_turn_profiles(ccw, cw, colors=('black', 'green')):
    """Mean +- SEM turn-triggered bump profiles of both channels (EPG, EL), clockwise | counter-clockwise.

    The line next to each profile marks its population-vector angle.

    Args:
        ccw, cw: Output of :func:`turn_triggered_profiles`.
        colors: Colors of (EPG, EL).

    Returns:
        Tuple of (figure, pva) where pva maps 'clockwise' / 'counter-clockwise' to the PVA (rad) of
        each channel's mean profile.
    """
    fig, axs = plt.subplots(1, 2, figsize=[5, 2.5], sharey=True)
    n_rois = cw.shape[-1]
    phases = np.linspace(-np.pi, np.pi, num=n_rois + 1)
    phases = (phases[:-1] + phases[1:]) / 2
    phases -= phases[utilities.HEADING_CENTER_ROI]
    pva = {}

    for ax, arr, name in ((axs[0], cw, 'clockwise'), (axs[1], ccw, 'counter-clockwise')):
        mu, sem = arr.mean(axis=0), sp.stats.sem(arr, axis=0)
        pva[name] = np.angle((mu * np.exp(1j * phases[np.newaxis, :])).mean(axis=-1))
        for ch, (label, color) in enumerate(zip(('EPG', 'EL'), colors, strict=True)):
            ax.plot(mu[ch], phases, color=color, label=label)
            ax.fill_betweenx(phases, mu[ch] - sem[ch], mu[ch] + sem[ch], color=color, alpha=.3)
            ax.plot([0, .3], [pva[name][ch]] * 2, color=color)
        ax.set_title(f'{name} turns (n={len(arr)})', fontsize=9)
        ax.set_xlabel('Norm. F')
        ax.set_yticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[0].set_ylabel('ROIs centered')
    axs[0].legend()
    fig.tight_layout()
    return fig, pva


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
        vmin=vmin, vmax=vmax, plot_times=plot_times, twindow=twindow, bins=bins, hist_title='vec length')
