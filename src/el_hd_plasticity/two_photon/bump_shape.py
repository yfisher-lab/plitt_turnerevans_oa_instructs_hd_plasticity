"""Analysis of activity profiles around a ring of ROIs (e.g. the ellipsoid body).

Functions here operate on arrays whose ROI axis is circular: Fourier-based
interpolation and phase nulling, width measurements, and rolling activity so a
chosen reference (peak, population-vector angle, heading) sits at a fixed ROI.
"""

import numpy as np

from .utilities import HEADING_CENTER_ROI, N_EB_ROIS


def mean_profile_fwhm(dff):
    """Width of the time-averaged, peak-aligned activity profile.

    Each time point is rolled so its peak ROI is at index 0, the aligned profiles are
    averaged over time, min-max normalized, and the number of ROIs at or above half
    maximum is converted to radians. This is the (coarse, ROI-resolution) width stored
    as ``fwhm`` on :class:`~el_hd_plasticity.two_photon.session.GetTS`.

    Args:
        dff: Array of shape (n_rois, n_timepoints).

    Returns:
        Width in radians (multiples of ``2*pi/n_rois``).
    """
    n_rois = dff.shape[0]
    max_inds = np.argmax(dff, axis=0)
    dff_aligned = np.zeros_like(dff)
    for i in range(dff.shape[1]):
        dff_aligned[:, i] = np.roll(dff[:, i], -max_inds[i])
    mu = dff_aligned.mean(axis=1)
    mu = (mu - np.amin(mu)) / (np.amax(mu) - np.amin(mu))
    return (mu >= .5).sum() * 2 * np.pi / n_rois


def fourier_upsample(F, factor=16):
    """Circularly interpolate each row of F (T, N) to N*factor points."""
    T, N = F.shape
    M = N * factor
    X = np.fft.fft(F, axis=1)
    Xp = np.zeros((T, M), dtype=complex)
    h = N // 2
    Xp[:, :h + 1] = X[:, :h + 1]
    Xp[:, -(N - h - 1):] = X[:, h + 1:]
    if N % 2 == 0:                      # split the Nyquist bin between +/- frequencies
        Xp[:, h] *= 0.5
        Xp[:, M - h] = Xp[:, h]
    return np.real(np.fft.ifft(Xp, axis=1)) * factor


def fwhm(F, factor=16, baseline="min"):
    """Nonparametric FWHM in radians for each circular profile in F (T, N).

    Profiles are Fourier-upsampled by `factor`, rolled so the peak is at index 0, and the
    half-max crossings on either side of the peak are located by linear interpolation.

    baseline="min":  half-way between trough and peak (matches the paper's fit-based width)
    baseline="zero": half of the peak value
    Returns NaN for frames where the profile never drops below half max.
    """
    P = fourier_upsample(np.atleast_2d(F), factor)
    T, M = P.shape
    out = np.full(T, np.nan)
    for t in range(T):
        p = np.roll(P[t], -np.argmax(P[t]))        # put the peak at index 0
        lo = p.min() if baseline == "min" else 0.0
        if p[0] <= lo:
            continue
        half = lo + (p[0] - lo) / 2
        below = p < half
        if not below.any():
            continue
        right = np.argmax(below)          # first sample below half, going right
        left = np.argmax(below[::-1])     # first sample below half, going left
        xr = (right - 1) + (p[right - 1] - half) / (p[right - 1] - p[right])
        a, b = p[(-left + 1) % M], p[(-left) % M]   # a above half, b below half
        xl = (left - 1) + (a - half) / (a - b)
        out[t] = (xr + xl) * 2 * np.pi / M
    return out


def phase_null(F):
    """Remove the phase of the first Fourier harmonic from each ring profile.

    Args:
        F: (T, N) array of dF/F, N ROIs around a ring (EB, or one PB hemisphere).

    Returns:
        Tuple of
            F_centered: (T, N) profiles with the first-harmonic phase nulled and rolled so
                the peak is in the middle,
            phase: (T,) first-harmonic phase in radians,
            c1_mag: (T,) first-harmonic magnitude,
            rho: (T,) first-harmonic magnitude normalized by the mean (c0),
            harmonic_ratio: (T,) second/first harmonic magnitude ratio.
    """
    _, N = F.shape
    X = np.fft.fft(F, axis=1)
    c0 = X[:, 0]                  # (T, N)
    c1 = X[:, 1]
    c2 = X[:, 2]

    rho = np.abs(c1) / np.abs(c0)   # normalized first-harmonic magnitude
    harmonic_ratio = np.abs(c2) / np.abs(c1)  # second/first harmonic ratio
    phase = np.angle(c1)                       # sign: fft uses exp(-i*2*pi*k*n/N)
    k = np.fft.fftfreq(N, d=1/N)                # integer harmonics, incl. negatives
    X_shifted = X * np.exp(-1j * np.outer(phase, k))   # null the phase of every harmonic
    F_centered = np.real(np.fft.ifft(X_shifted, axis=1))
    F_centered = np.roll(F_centered, N//2, axis=1)  # roll so that the peak is in the middle
    return F_centered, phase, np.abs(c1), rho, harmonic_ratio


def shift_by_max(arr, sortby_index):
    """Roll every time point so the peak ROI of one channel sits at the center ROI.

    NOTE: modifies `arr` in place (and returns it).

    Args:
        arr: (n_channels, n_rois, n_timepoints) array.
        sortby_index: Channel whose peak defines the shift applied to all channels.

    Returns:
        `arr`, rolled along the ROI axis.
    """
    for t in range(arr.shape[-1]):
        max_index = np.argmax(arr[sortby_index, :, t])
        arr[:, :, t] = np.roll(arr[:, :, t], -max_index + HEADING_CENTER_ROI, axis=1)
    return arr


def shift_by_pva(arr, pva, sortby_index, n_rois=N_EB_ROIS):
    """Roll every time point so one channel's population-vector angle sits at the center ROI.

    NOTE: modifies `arr` in place (and returns it).

    Args:
        arr: (n_channels, n_rois, n_timepoints) array.
        pva: (n_channels, n_timepoints) population vector angle in (-pi, pi].
        sortby_index: Channel whose PVA defines the shift applied to all channels.
        n_rois: Number of ROIs the angle is discretized into.

    Returns:
        `arr`, rolled along the ROI axis.
    """
    for t in range(arr.shape[-1]):
        shift = int(np.round((pva[sortby_index, t] + np.pi) / 2 / np.pi * n_rois))
        arr[:, :, t] = np.roll(arr[:, :, t], -shift + HEADING_CENTER_ROI, axis=1)
    return arr
