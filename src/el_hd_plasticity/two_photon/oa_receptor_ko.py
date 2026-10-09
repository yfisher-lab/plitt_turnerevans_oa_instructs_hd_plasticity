"""Analysis and visualization of octopamine receptor knockout/RNAi imaging data.

Same analyses as :mod:`tbh_rnai`, but sessions are grouped by the ``condition`` column
(genotype/treatment) instead of ``rnai``. The closed-loop vs dark figures are identical to
those in :mod:`tbh_rnai` and are re-exported from there.
"""

from . import stats
from .tbh_rnai import (  # noqa: F401
    plot_sess_heatmaps,
    plot_sess_heatmaps_w_hist,
    plot_sess_histograms,
)

# Output column -> session-table column for per-session statistics.
OFFSET_META_COLS = {'fly_id': 'fly_id', 'cl': 'closed_loop', 'dark': 'dark', 'condition': 'condition'}
BINNED_META_COLS = {'fly_id': 'fly_id', 'cl': 'closed_loop', 'condition': 'condition', 'dark': 'dark'}
# Per-session columns carried into regression tables.
REG_META_COLS = ('fly_id', 'condition', 'dark')

# Sessions with at least this many closed-loop blocks count as closed loop when aggregating per fly.
MIN_CL = 1


def offset_stats(sess_df, load_row, **kwargs):
    """Compute offset statistics (variance, mean, magnitude) for each session.

    Computes circular statistics on EPG offset (difference between heading and PVA)
    for each imaging session across different closed-loop and dark conditions.

    Args:
        sess_df: DataFrame with session metadata including fly_id, closed_loop, dark, condition columns.
        load_row: Callable that loads preprocessed data for a session row.
        **kwargs: Passed to :class:`session.GetTS`.

    Returns:
        DataFrame with columns: fly_id, cl, dark, condition, offset_var, offset_mean, offset_mag.
    """
    return stats.offset_stats(sess_df, load_row, meta_cols=OFFSET_META_COLS, **kwargs)


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
    return stats.offset_stats_unique(stats_df, group_col='condition', min_cl=MIN_CL)


def offset_stats_plot(stats_df_unique):
    """Reformat aggregated stats for side-by-side plotting (wide format).

    Args:
        stats_df_unique: Output from offset_stats_unique().

    Returns:
        DataFrame with one row per fly, columns for dark and cl values of each metric.
    """
    return stats.offset_stats_plot(stats_df_unique, group_col='condition')


def rho_stats(sess_df, load_row, dh_bins):
    """Compute pva magnitude binned by heading velocity.

    Args:
        sess_df: Session DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.

    Returns:
        DataFrame with rho_dig: array of mean rho values for each dh bin.
    """
    return stats.rho_stats(sess_df, load_row, dh_bins, meta_cols=BINNED_META_COLS)


def reformat_rho_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned rho stats to long format for regression modeling.

    Args:
        grouped_stats: Output from rho_stats().
        dh_bins: Bin edges used in rho_stats().

    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded).
    """
    return stats.reformat_binned_stats_for_reg(grouped_stats, dh_bins, {'rho': 'rho_dig'},
                                               meta_cols=REG_META_COLS)


def dphi_stats(sess_df, load_row, dh_bins):
    """Compute PVA angular velocity binned by heading velocity.

    Args:
        sess_df: Session DataFrame.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.

    Returns:
        DataFrame with dphi_dig: array of mean PVA velocity for each dh bin.
    """
    return stats.dphi_stats(sess_df, load_row, dh_bins, meta_cols=BINNED_META_COLS)


def reformat_dphi_stats_for_reg(grouped_stats, dh_bins):
    """Reformat binned PVA velocity stats to long format for regression modeling.

    Args:
        grouped_stats: Output from dphi_stats().
        dh_bins: Bin edges used in dphi_stats().

    Returns:
        Long-format DataFrame with one row per bin per session (NaN values excluded).
    """
    return stats.reformat_binned_stats_for_reg(grouped_stats, dh_bins, {'dphi': 'dphi_dig'},
                                               meta_cols=REG_META_COLS)
