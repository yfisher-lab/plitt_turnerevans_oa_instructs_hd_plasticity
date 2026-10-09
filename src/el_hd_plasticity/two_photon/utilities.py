"""Small helpers shared across the two-photon analysis modules.

Everything here is generic (time axes, angle wrapping, binning, histograms) and has
no dependency on a particular experiment. Experiment-specific analysis lives in the
sibling modules; figure helpers live in :mod:`plotting`.
"""

import numpy as np

# Number of ROIs (wedges) the ellipsoid body is divided into in this dataset, and the
# ROI index heading is rolled to when aligning activity to heading.
N_EB_ROIS = 16
HEADING_CENTER_ROI = N_EB_ROIS // 2


def _readonly(arr):
    """Freeze an array so it is safe to share as a default argument."""
    arr.flags.writeable = False
    return arr


# Shared defaults for figure functions (read-only: they are shared between calls)
DEFAULT_OFFSET_BINS = _readonly(np.linspace(-np.pi, np.pi, num=17))   # 16 bins of offset
DEFAULT_PLOT_TIMES = _readonly(np.arange(0, 360, 60))                 # time-axis ticks (s)
SHORT_PLOT_TIMES = _readonly(np.arange(0, 180, 60))                   # for shorter scans
SCAN_POSITIONS = _readonly(np.arange(3))                              # x positions of 3 scan types


# ----------------------------------------------------------------------------
# Time axis helpers
# ----------------------------------------------------------------------------
def get_time_tick_indices(time, plot_times):
    """Find the closest indices in time array for given plot times.

    Args:
        time: Time array (typically pandas Series).
        plot_times: Array of time values to find indices for.

    Returns:
        List of indices in time array closest to each plot_time.
    """
    return [np.argmin(np.abs(time - t)) for t in plot_times]


def time_window_mask(time, twindow=None):
    """Boolean mask selecting samples inside a time window.

    Args:
        time: Time array (numpy array or pandas Series).
        twindow: Tuple of (start_time, end_time), inclusive, or None to keep everything.

    Returns:
        Boolean mask with the same length as `time`.
    """
    if twindow is None:
        return np.ones_like(time, dtype=bool)
    return (time >= twindow[0]) & (time <= twindow[1])


# ----------------------------------------------------------------------------
# Angles
# ----------------------------------------------------------------------------
def wrap_angle(angle):
    """Wrap angles (radians) into (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(angle)))


def heading_to_roi_coords(heading, n_rois=N_EB_ROIS):
    """Convert heading in (-pi, pi] to the ROI-axis coordinate used on heatmaps.

    Heatmap y-ticks run from ROI 0 (heading -pi) to ROI ``n_rois - 1`` (heading +pi), so
    the heading trace overlaid on a heatmap is ``(heading + pi) / 2pi * (n_rois - 1)``.
    """
    return (heading + np.pi) / (2 * np.pi) * (n_rois - 1)


def nearest_equivalent_angle(angle, reference):
    """Shift `angle` by a multiple of 2pi so that it lies as close as possible to `reference`.

    Only shifts of 0 and +/- one turn are considered, which is sufficient for angles
    already wrapped to (-pi, pi].
    """
    candidates = np.array([angle, angle + 2 * np.pi, angle - 2 * np.pi])
    return candidates[np.argmin(np.abs(reference - candidates))]


# ----------------------------------------------------------------------------
# Binning and histograms
# ----------------------------------------------------------------------------
def binned_mean(values, bin_idx, n_bins):
    """Mean of `values` within each integer bin index 0..n_bins-1.

    Args:
        values: 1D array.
        bin_idx: Integer bin label for each element of `values` (e.g. from ``np.digitize``).
        n_bins: Number of bins to report.

    Returns:
        Array of length `n_bins`; bins with no samples are NaN.
    """
    values, bin_idx = np.asarray(values), np.asarray(bin_idx)
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = bin_idx == i
        if in_bin.any():
            out[i] = values[in_bin].mean()
    return out


def normalized_histogram(x, bins):
    """Histogram of `x` normalized to sum to 1.

    Returns:
        Tuple of (proportion per bin, bin centers).
    """
    hist, _ = np.histogram(x, bins=bins)
    centers = (bins[1:] + bins[:-1]) / 2
    return hist / hist.sum(), centers


def print_nested_keys(data, indent=0):
    """Recursively print the keys of a nested dictionary, indented by depth."""
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + str(key))
            print_nested_keys(value, indent + 1)
