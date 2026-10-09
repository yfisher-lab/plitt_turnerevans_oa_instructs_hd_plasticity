"""Session timeseries extraction and analysis for two-photon imaging data.

Provides immutable timeseries objects for neural activity around a ring of ROIs, with circular
statistics (heading, offset, population vector) and activity aligned to heading.

Two data sources are supported, sharing the same statistics (:class:`RingTimeseries`):

* :class:`SessionTimeseries` - preprocessed SessionTools imaging sessions (``EBImagingSession``):
  ``SessionTimeseries.from_session(pp, ...)``.
* :class:`TrialTimeseries` - behavioral/neural trial dictionaries from Dan Turner-Evans' dataset:
  ``TrialTimeseries.from_trial_dict(trial_dict, channel_mapping, ...)``.

``GetTS`` and ``GetTS_DTE`` are the historical names of the two constructors and remain as aliases
of ``SessionTimeseries.from_session`` and ``TrialTimeseries.from_trial_dict``.

The computation lives in the module-level functions below; the classes only hold the results.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import scipy as sp
import SessionTools.two_photon as st2p

from . import ring_analysis
from .utilities import HEADING_CENTER_ROI, heading_to_roi_index


# ----------------------------------------------------------------------------
# Computation (pure functions of arrays)
# ----------------------------------------------------------------------------
def _smooth_heading(heading, dt, h_sigma):
    """Heading smoothed as a unit vector (no wrap-around artifacts).

    Args:
        heading: Heading in radians.
        dt: Sample period (s).
        h_sigma: Gaussian width in seconds; no smoothing if 0.
    """
    x_h, y_h = st2p.utilities.pol2cart(np.ones_like(heading), heading)
    if h_sigma > 0:
        x_h = sp.ndimage.gaussian_filter1d(x_h, h_sigma / dt)
        y_h = sp.ndimage.gaussian_filter1d(y_h, h_sigma / dt)
    _, heading_sm = st2p.utilities.cart2pol(x_h, y_h)
    return heading_sm


def _heading_velocity(heading_sm, dt, dh_sigma):
    """Derivative of the smoothed heading (rad/s; 0 at the first sample), optionally Gaussian smoothed.

    Args:
        heading_sm: Smoothed heading in radians.
        dt: Sample period (s).
        dh_sigma: Gaussian width in seconds; no smoothing if 0.
    """
    dh = np.concatenate([[0], np.diff(np.unwrap(heading_sm)) / dt])
    if dh_sigma > 0:
        dh = sp.ndimage.gaussian_filter1d(dh, dh_sigma / dt)
    return dh


def _population_vector(dff, heading):
    """Population vector and heading offset of ring activity.

    ROIs are treated as unit vectors at the centers of ``n_rois`` equal angular bins spanning
    -pi to pi, weighted by activity.

    Args:
        dff: (n_rois, T) or (n_channels, n_rois, T) activity.
        heading: Heading in radians, length T.

    Returns:
        Tuple (n_rois, pv_c, rho, phi, offset): the population vector as a complex number, its
        magnitude and angle, and the angle relative to heading wrapped to (-pi, pi]. Each is
        (T,) for (n_rois, T) input and (n_channels, T) otherwise.
    """
    n_rois = dff.shape[-2]
    roi_angle = np.linspace(-np.pi, np.pi, num=n_rois + 1)[:, np.newaxis]
    roi_angle = roi_angle[:-1] + np.diff(roi_angle, axis=0) / 2  # center of each ROI bin

    channels = dff[np.newaxis] if dff.ndim == 2 else dff
    n_chan, n_time = channels.shape[0], channels.shape[-1]
    pv_c = np.zeros((n_chan, n_time), dtype=complex)
    rho = np.zeros((n_chan, n_time))
    phi = np.zeros((n_chan, n_time))
    offset = np.zeros((n_chan, n_time))

    for chan in range(n_chan):
        x_f, y_f = st2p.utilities.pol2cart(channels[chan], roi_angle)
        x_mu, y_mu = x_f.mean(axis=0), y_f.mean(axis=0)
        pv_c[chan] = x_mu + 1j * y_mu
        rho[chan], phi[chan] = np.abs(pv_c[chan]), np.angle(pv_c[chan])
        offset[chan] = np.angle(np.exp(1j * (phi[chan] - heading)))

    if dff.ndim == 2:
        pv_c, rho, phi, offset = pv_c[0], rho[0], phi[0], offset[0]
    return n_rois, pv_c, rho, phi, offset


def _align_to_heading(dff, heading):
    """Shift each time point's activity profile, to sub-ROI precision, so heading is at a fixed ROI.

    At every time point the profile is shifted along the ROI axis so the current heading falls on ROI
    ``HEADING_CENTER_ROI``, enabling averaging of activity at each heading. The heading is not
    binned: it is converted to a fractional ROI index (:func:`utilities.heading_to_roi_index`; one
    ROI = ``2pi / n_rois`` of heading, ROI centers at ``-pi + (j + 1/2) * 2pi / n_rois``) and the
    profile is shifted by that amount with the Fourier shift theorem (a phase ramp across spatial
    frequencies), which interpolates band-limited profiles exactly. A shift by a whole number of
    ROIs is the same as ``np.roll``. For an even number of ROIs the Nyquist component keeps only
    its real part.

    Args:
        dff: (n_rois, T) or (n_channels, n_rois, T) activity.
        heading: Full-resolution heading in radians, length T.

    Returns:
        Real array with the shape of `dff`.
    """
    n = dff.shape[-2]
    shift = heading_to_roi_index(heading, n) - HEADING_CENTER_ROI    # aligned[..., j, t] = dff[..., j + shift[t], t]

    freqs = np.arange(n // 2 + 1)
    phase = np.exp(2j * np.pi * freqs[:, np.newaxis] * shift[np.newaxis, :] / n)    # (freqs, time)
    spectrum = np.fft.rfft(dff, axis=-2)                                            # (..., freqs, time)
    return np.fft.irfft(spectrum * phase, n=n, axis=-2)


def _profile_widths(dff):
    """FWHM of the mean peak-aligned profile (see :func:`ring_analysis.mean_profile_fwhm`): a float for
    (n_rois, T) input, an array with one value per channel for (n_channels, n_rois, T)."""
    if dff.ndim > 2:
        return np.array([ring_analysis.mean_profile_fwhm(channel) for channel in dff])
    return ring_analysis.mean_profile_fwhm(dff)


def _smooth_activity(dff, dt, t_sigma, circ_sigma, neural_shift):
    """Gaussian smoothing in time and across the ring, then a time shift aligning neural to behavior.

    Args:
        dff: (..., n_rois, T) activity.
        dt: Sample period (s).
        t_sigma: Gaussian width in seconds along time; no smoothing if 0.
        circ_sigma: Gaussian width in ROI bins across the (circular) ROI axis; no smoothing if 0.
        neural_shift: Time offset in seconds (typically negative), applied as a circular shift.
    """
    if t_sigma > 0:
        dff = sp.ndimage.gaussian_filter1d(dff, t_sigma / dt, axis=-1)
    if circ_sigma > 0:
        dff = sp.ndimage.gaussian_filter1d(dff, circ_sigma, axis=-2, mode='wrap')
    return np.roll(dff, int(neural_shift / dt), axis=-1)


# ----------------------------------------------------------------------------
# Immutable result objects
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False, repr=False)
class RingTimeseries:
    """Neural activity around a ring of ROIs with circular statistics; immutable.

    Fields cannot be reassigned, and the NumPy arrays among them are frozen
    (``ts.dff[0] = 0`` raises ``ValueError``); copy first to edit (``ts.dff.copy()``). ``time`` (a
    pandas Series) and the dict fields of subclasses cannot be frozen, so they are protected from
    reassignment only. Build instances with the ``from_*`` classmethods of the subclasses.

    Attributes:
        dff: Neural timeseries; shape (n_rois, n_timepoints) or (n_channels, n_rois, n_timepoints).
        heading: Animal heading in radians.
        heading_sm: Smoothed heading.
        dh: Heading velocity (rad/s; 0 at the first sample).
        offset: Heading offset (phi - heading) wrapped to (-pi, pi]; same leading shape as rho.
        phi: Population vector angle.
        rho: Population vector magnitude.
        pv_c: Population vector as a complex number (rho * exp(1j * phi)).
        dff_h_aligned: Neural activity shifted (to sub-ROI precision) so heading is at a fixed ROI.
        time: Time of each sample (seconds, pandas Series).
        n_rois: Number of ROIs around the ring.
        dt: Mean sample period (seconds).
        fwhm: Width of the mean peak-aligned profile (radians); an array with one value per
            channel for multi-channel data.
    """

    dff: np.ndarray
    heading: np.ndarray
    heading_sm: np.ndarray
    dh: np.ndarray
    offset: np.ndarray
    phi: np.ndarray
    rho: np.ndarray
    pv_c: np.ndarray
    dff_h_aligned: np.ndarray
    time: Any
    n_rois: int
    dt: float
    fwhm: Any

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def __repr__(self):
        n_chan = f", n_channels={self.dff.shape[0]}" if self.dff.ndim == 3 else ""
        return f"{type(self).__name__}(n_rois={self.n_rois}{n_chan}, n_samples={self.dff.shape[-1]}, dt={self.dt:.4g})"

    @property
    def offset_c(self):
        """Complex representation of heading offset as unit vectors.

        Returns:
            ndarray of dtype complex with magnitude 1 and angle equal to self.offset.
        """
        return np.exp(1j * self.offset)

    @classmethod
    def _from_arrays(cls, *, dff, heading, time, dt, h_sigma, dh_sigma, **extra):
        """Compute the shared statistics from smoothed activity and heading and build the object.

        Args:
            dff: Smoothed activity, (n_rois, T) or (n_channels, n_rois, T).
            heading: Heading in radians, length T.
            time: Time of each sample.
            dt: Sample period (s).
            h_sigma: Gaussian width in seconds for smoothing heading.
            dh_sigma: Gaussian width in seconds for smoothing heading velocity.
            **extra: Additional fields of the subclass.
        """
        heading_sm = _smooth_heading(heading, dt, h_sigma)
        n_rois, pv_c, rho, phi, offset = _population_vector(dff, heading)
        return cls(dff=dff, heading=heading, heading_sm=heading_sm,
                   dh=_heading_velocity(heading_sm, dt, dh_sigma),
                   offset=offset, phi=phi, rho=rho, pv_c=pv_c,
                   dff_h_aligned=_align_to_heading(dff, heading),
                   time=time, n_rois=n_rois, dt=dt, fwhm=_profile_widths(dff), **extra)


@dataclass(frozen=True, eq=False, repr=False)
class TrialTimeseries(RingTimeseries):
    """Timeseries from Dan Turner-Evans' behavioral/neural data: EPG and EL channels.

    Attributes (in addition to those of :class:`RingTimeseries`):
        trial_dict: Dict with behavioral data and neural timeseries by channel.
        channel_mapping: Dict mapping channel names ('EPG', 'EL') to trial_dict keys.
        dff: Shape (2, n_rois, n_timepoints), channel 0 = EPG, 1 = EL; the other per-channel
            fields (offset, phi, rho, pv_c, dff_h_aligned, fwhm) follow the same order.
    """

    trial_dict: Any
    channel_mapping: Any

    @classmethod
    def from_trial_dict(cls, trial_dict, channel_mapping, circ_sigma=.5, t_sigma=.1, h_sigma=.1,
                        neural_shift=-.2, dh_sigma=0, closed=True):
        """Smooth the EPG and EL activity of a trial and compute the circular statistics.

        Args:
            trial_dict: Dict with 'behavDat' (DataFrame) and neural timeseries arrays.
            channel_mapping: Dict mapping 'EPG' and 'EL' to keys of `trial_dict`.
            circ_sigma: Gaussian smoothing width (in ROI bins) across neural population.
            t_sigma: Gaussian smoothing width (in seconds) along time axis.
            h_sigma: Gaussian smoothing width (in seconds) for heading.
            neural_shift: Time offset (in seconds) to align neural to behavior (typically negative).
            dh_sigma: Gaussian smoothing width for heading derivative.
            closed: If True, only use closed-loop behavior; else use all data.
        """
        behav_data = trial_dict['behavDat']
        mask = behav_data['closed'] == 1 if closed else np.ones(len(behav_data), dtype=bool)

        time = behav_data['Elapsed time'].loc[mask]
        dt = np.diff(time).mean()
        heading = np.asarray(np.angle(np.exp(1j * (-np.pi / 180 * behav_data['Rotational offset'].loc[mask]))))

        dff = np.stack((trial_dict[channel_mapping['EPG']][mask, :].T,
                        trial_dict[channel_mapping['EL']][mask, :].T), axis=0)

        return cls._from_arrays(dff=_smooth_activity(dff, dt, t_sigma, circ_sigma, neural_shift),
                                heading=heading, time=time, dt=dt, h_sigma=h_sigma, dh_sigma=dh_sigma,
                                trial_dict=trial_dict, channel_mapping=channel_mapping)


@dataclass(frozen=True, eq=False, repr=False)
class SessionTimeseries(RingTimeseries):
    """Timeseries from a preprocessed imaging session: (z-scored) dF/F of the ROIs plus the outer ring.

    Attributes (in addition to those of :class:`RingTimeseries`):
        pp: Preprocessed session object with voltage recordings and calcium imaging data.
        dff: Z-scored, smoothed dF/F; shape (n_rois, n_timepoints) or
            (n_channels, n_rois, n_timepoints) depending on the ``channels`` index.
        dff_noz: Same as ``dff`` but not z-scored and not smoothed.
        outer_ring: Z-scored, time-smoothed activity from the EB outer ring region.
        outer_ring_noz: Same as ``outer_ring`` but not z-scored.
    """

    pp: Any
    dff_noz: np.ndarray
    outer_ring: np.ndarray
    outer_ring_noz: np.ndarray

    @classmethod
    def from_session(cls, pp, channels=-1, exp_detrend=True, background_ts='background',
                     circ_sigma=.5, t_sigma=.1, h_sigma=.1, neural_shift=-.2, dh_sigma=0):
        """Extract and smooth dF/F from a preprocessed session and compute the circular statistics.

        Args:
            pp: Preprocessed session object.
            channels: Index into the channel axis of the ROI timeseries; -1 selects the last
                channel (default). Pass a list such as [0, 1] to keep several channels.
            exp_detrend: If True, exponential detrending is applied to dF/F.
            background_ts: Background timeseries to use for dF computation.
            circ_sigma: Gaussian smoothing width (in ROI bins) across neural population.
            t_sigma: Gaussian smoothing width (in seconds) along time axis.
            h_sigma: Gaussian smoothing width (in seconds) for heading.
            neural_shift: Time offset (in seconds) to align neural to behavior (typically negative).
            dh_sigma: Gaussian smoothing width for heading derivative.
        """
        voltage = pp.voltage_recording_aligned
        time = voltage['Time(ms)'] / 1000
        time = time - time[0]
        dt = np.diff(time).mean()
        heading = np.angle(np.exp(1j * (-1 * voltage[' Heading'].to_numpy() - np.pi)))

        def zscored_F(roi_name, zscore):
            return pp.calculate_zscored_F(roi_name, exp_detrend=exp_detrend, zscore=zscore,
                                          background_ts=background_ts)

        def outer_ring_ts(zscore):
            ts = np.squeeze(zscored_F('outer_ring', zscore))
            return sp.ndimage.gaussian_filter1d(ts, t_sigma / dt, axis=-1) if t_sigma > 0 else ts

        dff = zscored_F('rois', True)[channels, :, :]
        return cls._from_arrays(dff=_smooth_activity(dff, dt, t_sigma, circ_sigma, neural_shift),
                                heading=heading, time=time, dt=dt, h_sigma=h_sigma, dh_sigma=dh_sigma,
                                pp=pp, dff_noz=zscored_F('rois', False)[channels, :, :],
                                outer_ring=outer_ring_ts(True), outer_ring_noz=outer_ring_ts(False))


# Historical names of the constructors (``GetTS(pp, ...)`` and ``GetTS_DTE(trial_dict, channel_mapping, ...)``)
GetTS = SessionTimeseries.from_session
GetTS_DTE = TrialTimeseries.from_trial_dict
