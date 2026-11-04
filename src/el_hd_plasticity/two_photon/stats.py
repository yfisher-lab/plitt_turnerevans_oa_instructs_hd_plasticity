"""Per-session and per-fly EPG statistics shared by the experiment modules.

The experiment modules (``tbh_rnai``, ``oa_receptor_ko``, ``grab_sensors``, ...) differ only in
which session-table columns identify a session (``rnai``, ``condition``, ...), so the
computations live here and are parameterized by those column names:

* ``offset_stats`` / ``offset_stats_unique`` / ``offset_stats_plot``: circular statistics of the
  heading offset per session, averaged per fly, and reshaped for plotting.
* ``rho_stats`` / ``dphi_stats``: population-vector magnitude and angular velocity binned by
  heading velocity, and ``reformat_binned_stats_for_reg`` to unpivot them for regression.
* ``bump_stats`` / ``group_bump_stats``: time-averaged, phase-nulled bump profile per session, and
  its mean profile and width (FWHM) per fly and condition.

Session tables are DataFrames with one row per imaging session; ``load_row`` is a function
mapping a row to a preprocessed session (see :mod:`loading`).
"""

import numpy as np
import pandas as pd
import pingouin as pg
import scipy as sp

from . import bump_shape, session
from .utilities import binned_mean

# Session-table columns copied into per-session statistics by default.
BASIC_META_COLS = ('fly_id', 'closed_loop')


def _meta(row, meta_cols):
    """Pick the identifying columns `meta_cols` of a session row (kept under their own names)."""
    return {col: row[col] for col in meta_cols}


# ----------------------------------------------------------------------------
# Offset statistics
# ----------------------------------------------------------------------------
def offset_stats(session_manager, meta_cols=None, **ts_kwargs):
    """Compute offset statistics (mean, magnitude) for each session and channel.

    Computes circular statistics on the EPG offset (difference between heading and PVA)
    for each imaging session. Sessions with several channels (``ts.offset_c`` of shape
    (channels, time), e.g. ``channels=[0, 1]``) give one row per session per channel.

    Args:
        session_manager: Session manager object.
        meta_cols: Session-table columns copied to the result under the same names.
            Defaults to ``('fly_id', 'closed_loop')``.
        **ts_kwargs: Passed to :meth:`session.SessionTimeseries.from_session`.

    Returns:
        DataFrame with the `meta_cols`, a ``channel`` column (only for multi-channel sessions;
        the index along the first dimension of ``ts.offset_c``), and offset_mean, offset_mag.
    """
    meta_cols = BASIC_META_COLS if meta_cols is None else meta_cols
    rows = []

    for _, row in session_manager.iterrows():
        pp = session_manager.load_row(row)
        ts = session.SessionTimeseries.from_session(pp, **ts_kwargs)

        multichannel = ts.offset_c.ndim > 1
        offset_c = ts.offset_c if multichannel else ts.offset_c[np.newaxis]     # (channels, time)

        mean_c = offset_c.mean(axis=-1)

        for channel in range(offset_c.shape[0]):
            rows.append({
                **_meta(row, meta_cols),
                **({'channel': channel} if multichannel else {}),
                'offset_mean': np.angle(mean_c[channel]),
                'offset_mag': np.abs(mean_c[channel])
            })

    return pd.DataFrame(rows)


def _channel_groups(stats_df, fly_mask):
    """Channels present for one fly (a single ``None`` when the table has no channel column)."""
    return stats_df.loc[fly_mask, 'channel'].unique() if 'channel' in stats_df.columns else [None]


def offset_stats_unique(stats_df, group_col=None, min_cl=1, dark_col='dark'):
    """Aggregate offset statistics per fly by averaging across sessions.

    Reduces multiple sessions per condition to single values by averaging, creating one
    closed-loop row and one dark row per fly (and per channel, if `stats_df` has a ``channel``
    column). Only flies with both closed-loop and dark sessions are kept. See :func:`add_offset_var` for the
    offset variance and its logit, used for mixed effects modeling.

    Args:
        stats_df: Output from :func:`offset_stats`; needs fly_id, closed_loop, offset_mean,
            offset_mag, and `dark_col` / `group_col` if given.
        group_col: Optional column identifying the experimental group of each fly (e.g.
            'rnai', 'condition'); copied to the result.
        min_cl: Sessions with ``closed_loop >= min_cl`` count as closed loop.
            ``closed_loop`` is the number of closed-loop blocks preceding a session, so e.g.
            ``min_cl=2`` requires at least 10 min of prior closed-loop experience.
        dark_col: Column that is >= 1 for dark sessions. If None, dark sessions are those
            with ``closed_loop == 0``.

    Returns:
        DataFrame with one row per condition per fly (per channel) and columns fly_id,
        [channel], closed_loop (1 = closed-loop row, 0 = dark row), [group_col], dark,
        offset_mean, offset_mag.
    """
    rows = []

    for fly in stats_df['fly_id'].unique():
        fly_mask = stats_df['fly_id'] == fly
        group = {} if group_col is None else {group_col: stats_df[group_col].loc[fly_mask].iloc[0]}

        for channel in _channel_groups(stats_df, fly_mask):
            fly_ch = fly_mask if channel is None else fly_mask & (stats_df['channel'] == channel)
            cl_mask = fly_ch & (stats_df['closed_loop'] >= min_cl)
            if dark_col is None:
                dark_mask = fly_ch & (stats_df['closed_loop'] == 0)
            else:
                dark_mask = fly_ch & (stats_df[dark_col] >= 1)

            # Only include flies with both conditions
            if cl_mask.sum() < 1 or dark_mask.sum() < 1:
                continue

            for cl, dark, mask in ((1, 0, cl_mask), (0, 1, dark_mask)):
                rows.append({
                    'fly_id': fly,
                    **({} if channel is None else {'channel': channel}),
                    'closed_loop': cl,
                    **group,
                    'dark': dark,
                    'offset_mean': sp.stats.circmean(stats_df['offset_mean'].loc[mask]),
                    'offset_mag': stats_df['offset_mag'].loc[mask].mean()
                })

    return pd.DataFrame(rows)


def add_offset_var(stats_df):
    """Add the circular variance of the offset and its logit to a table with ``offset_mag``.

    ``offset_var = 1 - offset_mag`` (the circular variance of the offset), and ``offset_var_logit`` is its logit,
    used for mixed effects modeling. Works on per-session or per-fly tables; returns a copy.
    """
    stats_df = stats_df.copy()
    stats_df['offset_var'] = 1 - stats_df['offset_mag']
    stats_df['offset_var_logit'] = sp.special.logit(stats_df['offset_var'])
    return stats_df


def offset_stats_plot(stats_df_unique, group_col=None):
    """Reformat aggregated stats for side-by-side plotting (wide format).

    Pivots the long-format aggregated statistics into wide format with separate
    columns for dark and closed-loop conditions for each metric.

    Args:
        stats_df_unique: Output from :func:`offset_stats_unique`.
        group_col: Optional group column to carry over (see :func:`offset_stats_unique`).

    Returns:
        DataFrame with one row per fly (per channel, if present), columns for dark and
        closed-loop values of each metric.
    """
    rows = []

    for fly in stats_df_unique['fly_id'].unique():
        fly_mask = stats_df_unique['fly_id'] == fly
        for channel in _channel_groups(stats_df_unique, fly_mask):
            fly_data = stats_df_unique.loc[fly_mask if channel is None
                                           else fly_mask & (stats_df_unique['channel'] == channel)]

            dark_data = fly_data.loc[fly_data['dark'] == 1].iloc[0]
            cl_data = fly_data.loc[fly_data['dark'] == 0].iloc[0]

            group = {} if group_col is None else {group_col: dark_data[group_col]}
            rows.append({
                'fly_id': fly,
                **({} if channel is None else {'channel': channel}),
                **group,
                'offset_mag_dark': dark_data['offset_mag'],
                'offset_mag_closed_loop': cl_data['offset_mag']
            })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Statistics binned by heading velocity
# ----------------------------------------------------------------------------
def rho_stats(session_manager, dh_bins, meta_cols=None, outer_ring=False, **ts_kwargs):
    """Compute PVA magnitude (rho) binned by absolute heading velocity.

    Args:
        session_manager: Session manager object.
        dh_bins: Bin edges for absolute heading velocity.
        meta_cols: Session-table columns copied to the result under the same names.
            Defaults to ``('fly_id', 'closed_loop')``.
        outer_ring: If True, also average the outer ring fluorescence in each bin (``F_dig``).
        **ts_kwargs: Passed to :meth:`session.SessionTimeseries.from_session`.

    Returns:
        DataFrame with the `meta_cols` plus ``rho_dig`` (and ``F_dig``): arrays holding the mean
        value in each heading-velocity bin (NaN for empty bins).
    """
    meta_cols = BASIC_META_COLS if meta_cols is None else meta_cols
    n_bins = len(dh_bins)
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), **ts_kwargs)
        dh_dig = np.digitize(np.abs(ts.dh), dh_bins) - 1

        out = {**_meta(row, meta_cols), 'rho_dig': binned_mean(ts.rho, dh_dig, n_bins)}
        if outer_ring:
            out['F_dig'] = binned_mean(ts.outer_ring, dh_dig, n_bins)
        rows.append(out)

    return pd.DataFrame(rows)


def group_rho_stats(rho_stats, group_cols=('fly_id', 'closed_loop')):
    """Average binned rho / outer-ring fluorescence over the sessions of each fly and condition.

    Args:
        rho_stats: Output of :func:`rho_stats` with ``outer_ring=True``.
        group_cols: Columns of `rho_stats` defining a fly and condition.

    Returns:
        DataFrame with one row per group holding the NaN-aware mean ``F_dig`` and ``rho_dig``
        arrays.
    """
    return rho_stats.groupby(list(group_cols)).apply(
        lambda x: pd.Series({
            'F_dig': np.nanmean(np.stack(x['F_dig']), axis=0),
            'rho_dig': np.nanmean(np.stack(x['rho_dig']), axis=0)
        })).reset_index()


def dphi_stats(session_manager, dh_bins, meta_cols=None, **ts_kwargs):
    """Compute PVA angular velocity binned by (signed) heading velocity.

    Measures how fast the population vector angle (PVA) rotates as a function of
    turning speed.

    Args:
        session_manager: Session manager object.
        dh_bins: Bin edges for heading velocity.
        meta_cols: Session-table columns copied to the result under the same names.
            Defaults to ``('fly_id', 'closed_loop')``.
        **ts_kwargs: Passed to :meth:`session.SessionTimeseries.from_session` (default ``t_sigma=.1, h_sigma=.1``).

    Returns:
        DataFrame with the `meta_cols` plus ``dh_dig`` (bin index of each time point) and
        ``dphi_dig``: array of mean PVA velocity for each dh bin.
    """
    meta_cols = BASIC_META_COLS if meta_cols is None else meta_cols
    ts_kwargs = {'t_sigma': .1, 'h_sigma': .1, **ts_kwargs}
    n_bins = len(dh_bins)
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), **ts_kwargs)

        # Compute velocities
        dh = np.diff(np.unwrap(ts.heading_sm)) / ts.dt
        d_phi = np.diff(np.unwrap(ts.phi)) / ts.dt

        dh_dig = np.digitize(dh, dh_bins) - 1

        rows.append({
            **_meta(row, meta_cols),
            'dh_dig': dh_dig,
            'dphi_dig': binned_mean(d_phi, dh_dig, n_bins)
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Lag between the neural phase and behavior
# ----------------------------------------------------------------------------
DEFAULT_LAGS = np.arange(-10, 11) / 10      # seconds; the lag axis shared by sessions of different frame rates


def lag_xcorr(ts, lags=DEFAULT_LAGS):
    """Correlation of PVA angular velocity with heading angular velocity, the neural trace shifted in time.

    The neural velocity is rolled by whole frames and the correlations are interpolated onto `lags`, so that
    sessions sampled at different rates can be averaged on one lag axis. The lag of the largest correlation is
    how far the phase trails the heading, and is negative when the neural signal lags the behavior -- it is the
    quantity ``session.SessionTimeseries.from_session`` then corrects for with ``neural_shift``.

    Args:
        ts: Timeseries of one session, built with ``neural_shift=0`` (otherwise the lag that is already
            corrected for is subtracted from the one measured here).
        lags: Lags in seconds at which the correlation is reported.

    Returns:
        Array of correlation coefficients, one per lag.
    """
    dphi = np.diff(np.unwrap(ts.phi)) / ts.dt
    dh = np.diff(np.unwrap(ts.heading_sm)) / ts.dt
    frames = int(np.ceil(np.abs(lags).max() / ts.dt)) + 1              # enough whole frames to cover `lags`
    shifts = np.arange(-frames, frames + 1)
    corr = [np.corrcoef(np.roll(dphi, int(shift)), dh)[0, 1] for shift in shifts]
    return np.interp(lags, shifts * ts.dt, corr)


def lag_stats(session_manager, meta_cols=None, lags=DEFAULT_LAGS, **ts_kwargs):
    """Correlation between PVA velocity and heading velocity at each lag, one row per session.

    Args:
        session_manager: Session manager object.
        meta_cols: Session-table columns copied to the result under the same names.
            Defaults to ``('fly_id', 'closed_loop')``.
        lags: Lags in seconds (see :func:`lag_xcorr`).
        **ts_kwargs: Passed to :meth:`session.SessionTimeseries.from_session`. ``neural_shift`` defaults to 0
            here rather than to the correction that constructor usually applies, because the point of this
            statistic is to measure that correction.

    Returns:
        DataFrame with the `meta_cols` plus ``xcorr``: array with one correlation per lag.
    """
    meta_cols = BASIC_META_COLS if meta_cols is None else meta_cols
    ts_kwargs = {'neural_shift': 0, **ts_kwargs}
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), **ts_kwargs)
        rows.append({**_meta(row, meta_cols), 'xcorr': lag_xcorr(ts, lags)})

    return pd.DataFrame(rows)


def peak_lag(xcorr, lags=DEFAULT_LAGS):
    """Lag of the largest correlation of each row of `xcorr` (an array of lag curves, or a single one)."""
    xcorr = np.atleast_2d(xcorr)
    return np.asarray(lags)[np.nanargmax(xcorr, axis=-1)]


# ----------------------------------------------------------------------------
# Bump shape statistics
# ----------------------------------------------------------------------------
def bump_stats(session_manager, meta_cols=None, rho_min=.05, **ts_kwargs):
    """Compute the time-averaged, phase-nulled bump profile of each session.

    Args:
        session_manager: Session manager object.
        meta_cols: Session-table columns copied to the result under the same names.
            Defaults to ``('fly_id', 'closed_loop')``.
        rho_min: Minimum first-harmonic magnitude for a time point to be included (see
            :func:`bump_shape.mean_phase_null_profile`).
        **ts_kwargs: Passed to :meth:`session.SessionTimeseries.from_session`.

    Returns:
        DataFrame with the `meta_cols` plus ``profile``: array of shape (n_rois,) with the bump
        peak in the middle.
    """
    meta_cols = BASIC_META_COLS if meta_cols is None else meta_cols
    rows = []

    for _, row in session_manager.iterrows():
        ts = session.GetTS(session_manager.load_row(row), **ts_kwargs)
        rows.append({**_meta(row, meta_cols),
                     'profile': bump_shape.mean_phase_null_profile(ts.dff, rho_min=rho_min)})

    return pd.DataFrame(rows)


def group_bump_stats(bump_stats, group_cols=('fly_id', 'closed_loop'), factor=16, baseline='min'):
    """Average bump profiles over the sessions of each fly and condition and measure their width.

    Args:
        bump_stats: Output of :func:`bump_stats`.
        group_cols: Columns of `bump_stats` defining a fly and condition.
        factor, baseline: Passed to :func:`bump_shape.fwhm`.

    Returns:
        DataFrame with one row per group holding the mean ``profile`` and its ``fwhm`` (radians;
        NaN if the profile never drops below half maximum).
    """
    grouped = (bump_stats.groupby(list(group_cols))['profile']
               .apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index())
    grouped['fwhm'] = bump_shape.fwhm(np.stack(grouped['profile']), factor=factor, baseline=baseline)
    return grouped


def reformat_binned_stats_for_reg(grouped_stats, dh_bins, value_cols, meta_cols=('fly_id',)):
    """Reformat binned stats to long format for regression modeling.

    Unpivots data from one row per session (arrays with one value per heading-velocity bin) to
    one row per bin per session, suitable for linear mixed effects modeling.

    Args:
        grouped_stats: Output from :func:`rho_stats` or :func:`dphi_stats`.
        dh_bins: Bin edges used to compute `grouped_stats`; the bin edge is reported as ``dh``.
        value_cols: Mapping of output column -> binned column of `grouped_stats`, e.g.
            ``{'rho': 'rho_dig'}``. Bins where the first value is NaN are dropped.
        meta_cols: Per-session columns copied to every output row.

    Returns:
        Long-format DataFrame with columns `meta_cols`, the `value_cols` names, and ``dh``.
    """
    first_value = next(iter(value_cols))
    columns = [*meta_cols, *value_cols, 'dh']
    rows = []

    for _, row in grouped_stats.iterrows():
        for i, dh in enumerate(dh_bins):
            if np.isnan(row[value_cols[first_value]][i]):  # Skip NaN values
                continue
            rows.append({
                **{col: row[col] for col in meta_cols},
                **{out_name: row[src_col][i] for out_name, src_col in value_cols.items()},
                'dh': dh
            })

    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------
# Small tests shared by the figure scripts
# ----------------------------------------------------------------------------
def wilcoxon_vs(x, reference):
    """Wilcoxon signed-rank test of `x` against a constant, or against paired values `reference`.

    Returns:
        Dict with the number of values ``n``, the statistic ``W`` and ``p``.
    """
    W, p = sp.stats.wilcoxon(np.asarray(x, dtype=float) - np.asarray(reference, dtype=float))
    return {'n': len(x), 'W': float(W), 'p': float(p)}


def add_holm(tests, mask=None):
    """Add Holm-corrected p-values of the rows selected by `mask` (default: all) as ``p_holm`` to a table with ``p``."""
    mask = np.ones(len(tests), dtype=bool) if mask is None else mask
    tests['p_holm'] = np.nan
    tests.loc[mask, 'p_holm'] = pg.multicomp(tests.loc[mask, 'p'].to_numpy(), method='holm')[1]
    return tests
