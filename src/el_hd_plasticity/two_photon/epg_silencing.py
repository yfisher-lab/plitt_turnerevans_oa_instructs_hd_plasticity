"""Statistics for EPG silencing (shibire-ts) experiments."""

from . import stats

# Output column -> session-table column
META_COLS = {'fly_id': 'fly_id', 'scan_type': 'scan_type', 'condition': 'condition'}


def rho_stats(sess_df, load_row, dh_bins):
    """Compute PVA magnitude and outer ring fluorescence binned by heading velocity.

    Computes mean rho (population vector average magnitude) and outer ring fluorescence for
    each bin of absolute heading velocity.

    Args:
        sess_df: Session DataFrame with fly_id, scan_type and condition columns.
        load_row: Callable to load preprocessed data.
        dh_bins: Bin edges for heading velocity.

    Returns:
        DataFrame with fly_id, scan_type, condition, and rho_dig / F_dig: arrays of mean values
        for each dh bin.
    """
    return stats.rho_stats(sess_df, load_row, dh_bins, meta_cols=META_COLS, outer_ring=True)
