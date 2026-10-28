"""Analysis and figures for GRAB sensor (e.g. GRAB-DA, GRAB-OA) imaging data.

Sessions are managed with :class:`~loading.SessionManager`. Besides the
EPG offset statistics shared with the other experiments (see :mod:`stats`), these sessions are
analysed for outer-ring sensor fluorescence versus rotation speed.

Figures take ``ts_dict``: a dict with a ``'fly'`` name and one :class:`session.SessionTimeseries` per
condition key (e.g. ``'closed loop'``).
"""

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from . import bump_shape, plotting, session, stats, utilities
from .utilities import DEFAULT_OFFSET_BINS, DEFAULT_PLOT_TIMES


def offset_stats(session_manager, **ts_kwargs):
    """Compute offset statistics for each session.

    Computes circular statistics of offset (difference between heading and PVA)
    for each imaging session.

    Args:
        session_manager: Session manager object.
        **ts_kwargs: Passed to :meth:`session.SessionTimeseries.from_session`.

    Returns:
        DataFrame with columns: fly_id, closed_loop, offset_mean, offset_mag.
    """
    return stats.offset_stats(session_manager, **ts_kwargs)


def offset_stats_unique(stats_df):
    """Aggregate offset statistics per fly by averaging across sessions.

    Reduces multiple sessions per condition to single values by averaging, creating
    one row per condition per fly (closed loop: ``closed_loop >= 1``, dark: ``closed_loop == 0``). Requires
    flies to have both closed-loop and dark data. Adds the circular variance of the offset
    (``offset_var = 1 - offset_mag``, see :func:`stats.add_offset_var`) and its logit for mixed effects modeling.

    Args:
        stats_df: Output from offset_stats().

    Returns:
        DataFrame with one row per condition per fly: fly_id, closed_loop (1 = closed loop, 0 =
        dark), dark, offset_mean, offset_mag, offset_var, offset_var_logit.
    """
    return stats.add_offset_var(stats.offset_stats_unique(stats_df, dark_col=None))


def offset_stats_plot(stats_df_unique):
    """Reformat aggregated stats for side-by-side plotting (wide format).

    Args:
        stats_df_unique: Output from offset_stats_unique().

    Returns:
        DataFrame with one row per fly and ``offset_mag_dark`` / ``offset_mag_closed_loop``.
    """
    return stats.offset_stats_plot(stats_df_unique)


def rho_stats(session_manager, dh_bins):
    """Compute motor neuron and fluorescence activity binned by heading velocity.

    Computes mean rho (population vector magnitude) and outer ring
    fluorescence for each bin of absolute heading velocity.

    Args:
        session_manager: Session manager object.
        dh_bins: Bin edges for heading velocity.

    Returns:
        DataFrame with rho_dig and F_dig: arrays of mean values for each bin.
    """
    return stats.rho_stats(session_manager, dh_bins, outer_ring=True)


def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned rho stats to long format for regression modeling.

    Unpivots rho and fluorescence data from one row per session to one row per bin,
    suitable for linear mixed effects modeling. Includes log-transformed values.

    Args:
        grouped_stats: Output from rho_stats().
        dh_bins: Bin edges used in rho_stats().

    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded) and columns
        fly_id, closed_loop, rho, log_rho, dh, F, log_F.
    """
    df = stats.reformat_binned_stats_for_reg(grouped_stats, dh_bins, {'rho': 'rho_dig', 'F': 'F_dig'},
                                             meta_cols=('fly_id', 'closed_loop'))
    df['log_rho'] = np.log(df['rho'])
    df['log_F'] = np.log(df['F'])
    return df[['fly_id', 'closed_loop', 'rho', 'log_rho', 'dh', 'F', 'log_F']]


def align_peaks_split_halves(session_manager, rot_speed_mask=True, min_rot_speed=.5, alignment='phase'):
    """Heading-aligned mean activity profile of the first and second half of each closed-loop session.

    For each closed-loop session the heading-aligned activity is averaged over the first and
    second half of the (optionally rotation-speed-masked) time points. Both halves are shifted by
    the same amount, chosen so the first half's bump sits at the center ROI, which shows whether
    the bump location is stable within a session.

    Args:
        session_manager: Session manager object.
        rot_speed_mask: If True, only use time points with rotation speed above `min_rot_speed`.
        min_rot_speed: Rotation speed threshold (rad/s).
        alignment: 'phase' (default) shifts to sub-ROI precision by nulling the first-harmonic phase, as in
            the dual imaging analyses (:func:`bump_shape.shift_by_phase`); 'max' rolls by whole ROIs so the
            peak ROI is at the center (:func:`bump_shape.shift_by_max`).

    Returns:
        Tuple of two (n_sessions, n_rois) arrays: aligned first-half and second-half profiles.
    """
    shifts = {'phase': bump_shape.shift_by_phase, 'max': bump_shape.shift_by_max}
    if alignment not in shifts:
        raise ValueError(f"alignment must be 'phase' or 'max', not {alignment!r}")

    first_half_aligned, second_half_aligned = [], []
    for _, row in session_manager.iterrows():
        if row['closed_loop'] > 0:
            ts = session.GetTS(session_manager.load_row(row))

            if rot_speed_mask:
                dh_mask = np.abs(ts.dh) > min_rot_speed
                dff_h_aligned = ts.dff_h_aligned[:, dh_mask]
            else:
                dff_h_aligned = ts.dff_h_aligned

            split = dff_h_aligned.shape[1] // 2
            # the two half means as the two "channels" of one time point, so both are shifted by the
            # first half's bump position and any drift between the halves is preserved
            halves = np.stack([dff_h_aligned[:, :split].mean(axis=1),
                               dff_h_aligned[:, split:].mean(axis=1)])[:, :, np.newaxis]
            aligned = shifts[alignment](halves, 0)[:, :, 0]
            first_half_aligned.append(aligned[0])
            second_half_aligned.append(aligned[1])

    return np.stack(first_half_aligned), np.stack(second_half_aligned)


def plot_sess_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES,
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
    fig = plt.figure(figsize=[7, 5.5])
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.35, top=.88)

    ax_heatmap = fig.add_subplot(gs[0])
    ax_heading = fig.add_subplot(gs[1], sharex=ax_heatmap)
    ax_dh = fig.add_subplot(gs[2], sharex=ax_heatmap)

    def plot_rows(key):
        """Draw the heatmap, heading and rotation-speed traces of one condition."""
        ts = ts_dict[key]
        mask = utilities.time_window_mask(ts.time, twindow)
        dff, time = ts.dff[:, mask], ts.time[mask]
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
        plotting.format_roi_axis(ax_heatmap, 'y')
        plotting.format_time_axis(ax_heatmap, time, plot_times, 'x')

        ax_heading.set_yticks([0, np.pi, 2*np.pi], labels=[r'0', r'$\pi$', r'$2\pi$'])
        ax_heading.set_ylabel('Heading')

        ax_dh.set_ylim([0, 10])
        ax_dh.set_yticks([0, 10])
        ax_dh.set_ylabel('rot speed')

        ax_heatmap.set_title(key)

        # one shared time axis, labeled under the bottom row
        for ax in (ax_heatmap, ax_heading):
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        ax_dh.set_xlabel('Time (s)')

    # Plot all condition keys (skip 'fly' key)
    for key in ts_dict:
        if key == 'fly':
            fig.suptitle(ts_dict[key], y=.97)
        else:
            plot_rows(key)

    return fig


def plot_transpose_heatmaps(ts_dict, vmin=-.5, vmax=.5, plot_times=DEFAULT_PLOT_TIMES,
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

    def plot_cols(key):
        """Draw the transposed heatmap, heading and rotation-speed traces of one condition."""
        ts = ts_dict[key]
        mask = utilities.time_window_mask(ts.time, twindow)
        dff, time = ts.dff[:, mask], ts.time[mask]
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
        plotting.format_roi_axis(ax_heatmap, 'x')
        plotting.format_time_axis(ax_heatmap, time, plot_times, 'y')

        ax_heading.set_xticks([0, np.pi, 2*np.pi], labels=[r'0', r'$\pi$', r'$2\pi$'])
        ax_heading.set_xlabel('Heading')
        ax_heading.set_xlim([0, 2*np.pi])

        ax_dh.set_xlim([0, 10])
        ax_dh.set_xticks([0, 10])
        ax_dh.set_xlabel('rot speed')

        ax_heatmap.set_title(key)

    # Plot all condition keys (skip 'fly' key)
    for key in ts_dict:
        if key == 'fly':
            fig.suptitle(ts_dict[key])
        else:
            plot_cols(key)

    fig.tight_layout()

    return fig


def plot_sess_histograms(ts_dict, bins=DEFAULT_OFFSET_BINS,
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
    keys = [key for key in ts_dict if key != 'fly']
    traces = [plotting.Trace.from_ts(key, ts_dict[key], cmap) for key in keys]
    return plotting.plot_offset_histograms(traces, bins, hist_title=keys[-1], polar_title=None,
                                           suptitle=ts_dict['fly'], legend=False)


def plot_sess_heatmaps_w_hist(ts_dict, bins=DEFAULT_OFFSET_BINS, vmin=-.5, vmax=.5,
                              plot_times=DEFAULT_PLOT_TIMES, twindow=None, cmap='Greens'):
    """Plot the closed-loop heatmap with its integrated offset histogram.

    Args:
        ts_dict: Dictionary with 'fly' key and a 'closed loop' timeseries object.
        bins: Bins for histogram computation.
        vmin: Minimum value for heatmap colorscale.
        vmax: Maximum value for heatmap colorscale.
        plot_times: Time points to mark on x-axis.
        twindow: Optional tuple (start_time, end_time) to limit data window.
        cmap: Colormap name or object.

    Returns:
        Tuple of (figure, (heatmap_axs, hist_axs, cbar_ax)).
    """
    traces = [plotting.Trace.from_ts('closed loop', ts_dict['closed loop'], cmap)]
    return plotting.plot_heatmaps_with_offset_hist(
        traces, ts_dict['fly'], figsize=[8, 3.2], vmin=vmin, vmax=vmax, plot_times=plot_times,
        twindow=twindow, bins=bins, hist_xlim_zero=True)
