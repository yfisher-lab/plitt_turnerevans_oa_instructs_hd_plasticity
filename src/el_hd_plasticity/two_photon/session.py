"""Session timeseries extraction and analysis for two-photon imaging data.

Provides classes for extracting neural timeseries, computing circular statistics
(heading, offset, population vector), and aligning activity to behavior.

Two data sources are supported, sharing the same circular-statistics machinery
(:class:`_RingTimeseries`):

* :class:`GetTS` - preprocessed SessionTools imaging sessions (``EBImagingSession``).
* :class:`GetTS_DTE` - behavioral/neural trial dictionaries from Dan Turner-Evans' dataset.
"""

import numpy as np
import scipy as sp
import SessionTools.two_photon as st2p

from . import ring_analysis
from .utilities import HEADING_CENTER_ROI


class _RingTimeseries:
    """Shared circular statistics for neural activity arranged around a ring of ROIs.

    Subclasses are responsible for loading data and setting ``time``, ``dt``, ``heading``
    and ``dff`` (see ``get_ts``); this class provides the heading smoothing, population
    vector / offset computation, heading alignment and width measurement common to both.

    Attributes:
        dff: Neural timeseries; shape (n_rois, n_timepoints) or (n_channels, n_rois, n_timepoints).
        heading: Animal heading in radians.
        heading_sm: Smoothed heading.
        dh: Heading velocity (rad/s; 0 at the first sample).
        offset: Heading offset (phi - heading) wrapped to (-pi, pi]; same leading shape as rho.
        phi: Population vector angle.
        rho: Population vector magnitude.
        pv_c: Population vector as a complex number (rho * exp(1j * phi)).
        heading_bins: Edges of the heading bins used by ``heading_aligned``.
        dff_h_aligned: Neural activity rolled so heading is at a fixed ROI.
        time: Time of each sample (seconds).
        n_rois: Number of ROIs around the ring.
        dt: Mean sample period (seconds).
        fwhm: Width of the mean peak-aligned profile (radians); a list for multi-channel data.
    """

    def __init__(self):
        """Declare the attributes filled in by the subclass' ``get_ts`` and the methods below."""
        self.dff = None
        self.heading = None
        self.heading_sm = None
        self.dh = None
        self.offset = None
        self.phi = None
        self.rho = None
        self.pv_c = None
        self.heading_bins = None
        self.dff_h_aligned = None
        self.time = None
        self.n_rois = None
        self.dt = None
        self.fwhm = None

    @property
    def offset_c(self):
        """Complex representation of heading offset as unit vectors.

        Returns:
            ndarray of dtype complex with magnitude 1 and angle equal to self.offset.
        """
        return np.exp(1j * self.offset)

    def _smooth_heading(self, h_sigma):
        """Set ``heading_sm``: heading smoothed as a unit vector (no wrap-around artifacts).

        Args:
            h_sigma: Gaussian width in seconds; no smoothing if 0.
        """
        x_h, y_h = st2p.utilities.pol2cart(np.ones_like(self.heading), self.heading)
        if h_sigma > 0:
            x_h = sp.ndimage.gaussian_filter1d(x_h, h_sigma / self.dt)
            y_h = sp.ndimage.gaussian_filter1d(y_h, h_sigma / self.dt)
        _, self.heading_sm = st2p.utilities.cart2pol(x_h, y_h)

    def _population_vector(self):
        """Set ``pv_c``, ``rho``, ``phi`` and ``offset`` from ``dff`` and ``heading``.

        ROIs are treated as evenly spaced unit vectors from -pi to pi, weighted by activity.
        Works for (n_rois, T) data (1D outputs) and (n_channels, n_rois, T) data (2D outputs).
        """
        self.n_rois = self.dff.shape[-2]
        roi_angle = np.linspace(-np.pi, np.pi, num=self.n_rois)[:, np.newaxis]

        dff = self.dff[np.newaxis] if self.dff.ndim == 2 else self.dff
        n_chan, n_time = dff.shape[0], dff.shape[-1]
        pv_c = np.zeros((n_chan, n_time), dtype=complex)
        rho = np.zeros((n_chan, n_time))
        phi = np.zeros((n_chan, n_time))
        offset = np.zeros((n_chan, n_time))

        for chan in range(n_chan):
            x_f, y_f = st2p.utilities.pol2cart(dff[chan], roi_angle)
            x_mu, y_mu = x_f.mean(axis=0), y_f.mean(axis=0)
            pv_c[chan] = x_mu + 1j * y_mu
            rho[chan], phi[chan] = st2p.utilities.cart2pol(x_mu, y_mu)
            _, offset[chan] = st2p.utilities.cart2pol(
                *st2p.utilities.pol2cart(np.ones(self.heading.shape), phi[chan] - self.heading))

        if self.dff.ndim == 2:
            pv_c, rho, phi, offset = pv_c[0], rho[0], phi[0], offset[0]
        self.pv_c, self.rho, self.phi, self.offset = pv_c, rho, phi, offset

    def _heading_velocity(self, dh_sigma):
        """Set ``dh``: derivative of the smoothed heading, optionally Gaussian smoothed.

        Args:
            dh_sigma: Gaussian width in seconds; no smoothing if 0.
        """
        dh = np.diff(np.unwrap(self.heading_sm)) / self.dt
        self.dh = np.concatenate([[0], dh])
        if dh_sigma > 0:
            self.dh = sp.ndimage.gaussian_filter1d(self.dh, dh_sigma / self.dt)

    def heading_aligned(self):
        """Align neural activity to heading by rolling each timepoint to a canonical position.

        Creates ``dff_h_aligned`` where each time point is rolled along the ROI axis so the
        current heading bin lands on ROI ``HEADING_CENTER_ROI``, enabling averaging of
        activity at each heading.
        """
        self.heading_bins = np.linspace(-np.pi, np.pi, num=self.n_rois + 1)
        heading_dig = np.digitize(self.heading, self.heading_bins) - 1

        # np.roll(x, s)[i] == x[(i - s) % n] with s = HEADING_CENTER_ROI - heading_dig
        roi_idx = (np.arange(self.n_rois)[:, np.newaxis] + heading_dig[np.newaxis, :]
                   - HEADING_CENTER_ROI) % self.n_rois
        if self.dff.ndim == 2:
            self.dff_h_aligned = np.take_along_axis(self.dff, roi_idx, axis=0)
        else:
            self.dff_h_aligned = np.take_along_axis(self.dff, roi_idx[np.newaxis], axis=1)

    def calc_fwhm(self):
        """Compute full-width at half-max (FWHM) of population tuning (see
        :func:`ring_analysis.mean_profile_fwhm`); one value per channel if multi-channel."""
        if self.dff.ndim > 2:  # if there are multiple channels
            self.fwhm = [ring_analysis.mean_profile_fwhm(self.dff[i]) for i in range(self.dff.shape[0])]
        else:
            self.fwhm = ring_analysis.mean_profile_fwhm(self.dff)


class GetTS_DTE(_RingTimeseries):
    """Extract and compute timeseries statistics from Dan Turner-Evans behavioral/neural data.

    Smooths dF/F for the EPG and EL channels, calculates circular statistics (heading offset,
    population vector), and aligns neural activity to heading for each trial epoch.

    Attributes:
        trial_dict: Dict with behavioral data and neural timeseries by channel.
        channel_mapping: Dict mapping channel names ('EPG', 'EL') to trial_dict keys.
        dff: Neural timeseries; shape (2, n_rois, n_timepoints), channel 0 = EPG, 1 = EL.
        offset, phi, rho, pv_c, dff_h_aligned, fwhm: Per channel; see :class:`_RingTimeseries`.
    """

    def __init__(self, trial_dict, channel_mapping, **kwargs):
        """
        Args:
            trial_dict: Dict with 'behavDat' (DataFrame) and neural timeseries arrays.
            channel_mapping: Dict mapping 'EPG' and 'EL' to keys of `trial_dict`.
            **kwargs: Passed to ``get_ts``.
        """
        super().__init__()
        self.trial_dict = trial_dict
        self.channel_mapping = channel_mapping

        self.get_ts(**kwargs)
        self.heading_aligned()
        self.calc_fwhm()

    def get_ts(self, circ_sigma=.5, t_sigma=.1, h_sigma=.1, neural_shift=-.2, dh_sigma=0, closed=True):
        """Extract and compute timeseries and circular statistics.

        Args:
            circ_sigma: Gaussian smoothing width (in ROI bins) across neural population.
            t_sigma: Gaussian smoothing width (in seconds) along time axis.
            h_sigma: Gaussian smoothing width (in seconds) for heading.
            neural_shift: Time offset (in seconds) to align neural to behavior (typically negative).
            dh_sigma: Gaussian smoothing width for heading derivative.
            closed: If True, only use closed-loop behavior; else use all data.
        """
        behav_data = self.trial_dict['behavDat']
        mask = behav_data['closed'] == 1 if closed else np.ones(len(behav_data), dtype=bool)

        self.time = behav_data['Elapsed time'].loc[mask]

        self.dt = np.diff(self.time).mean()
        neural_shift_inds = int(neural_shift / self.dt)

        self.heading = np.angle(np.exp(1j * (-np.pi / 180 * behav_data['Rotational offset'].loc[mask])))
        self._smooth_heading(h_sigma)

        self.dff = np.stack((self.trial_dict[self.channel_mapping['EPG']][mask, :].T,
                             self.trial_dict[self.channel_mapping['EL']][mask, :].T), axis=0)
        if t_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, t_sigma / self.dt, axis=-1)
        if circ_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, circ_sigma, axis=-2, mode='wrap')

        self.dff = np.roll(self.dff, neural_shift_inds, axis=-1)

        self._population_vector()
        self._heading_velocity(dh_sigma)


class GetTS(_RingTimeseries):
    """Extract and compute timeseries statistics from preprocessed imaging session.

    Computes (z-scored) dF/F for neural ROIs, calculates circular statistics (heading
    offset, population vector), and aligns neural activity to heading.

    Attributes:
        pp: Preprocessed session object with voltage recordings and calcium imaging data.
        dff: Z-scored, smoothed dF/F; shape (n_rois, n_timepoints) or
            (n_channels, n_rois, n_timepoints) depending on the ``channels`` index.
        dff_noz: Same as ``dff`` but not z-scored and not smoothed.
        outer_ring: Z-scored, time-smoothed activity from the EB outer ring region.
        outer_ring_noz: Same as ``outer_ring`` but not z-scored.
        heading, heading_sm, dh, offset, phi, rho, pv_c, dff_h_aligned, fwhm:
            See :class:`_RingTimeseries`.
    """

    def __init__(self, pp, **kwargs):
        """Initialize session and compute timeseries and statistics.

        Args:
            pp: Preprocessed session object.
            **kwargs: Passed to get_ts() method.
        """
        super().__init__()
        self.pp = pp
        self.dff_noz = None
        self.outer_ring = None
        self.outer_ring_noz = None

        self.get_ts(**kwargs)
        self.heading_aligned()
        self.calc_fwhm()

    def get_ts(self, channels=-1, exp_detrend=True, background_ts='background',
               circ_sigma=.5, t_sigma=.1, h_sigma=.1, neural_shift=-.2, dh_sigma=0):
        """Extract and compute timeseries and circular statistics from preprocessed session.

        Args:
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
        voltage = self.pp.voltage_recording_aligned
        self.time = voltage['Time(ms)'] / 1000
        self.time = self.time - self.time[0]

        self.dt = np.diff(self.time).mean()
        neural_shift_inds = int(neural_shift / self.dt)

        self.heading = np.angle(np.exp(1j * (-1 * voltage[' Heading'].to_numpy() - np.pi)))
        self._smooth_heading(h_sigma)

        self.dff = self.pp.calculate_zscored_F('rois', exp_detrend=exp_detrend, zscore=True,
                                               background_ts=background_ts)[channels, :, :]
        self.dff_noz = self.pp.calculate_zscored_F('rois', exp_detrend=exp_detrend, zscore=False,
                                                   background_ts=background_ts)[channels, :, :]

        if t_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, t_sigma / self.dt, axis=-1)
        if circ_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, circ_sigma, axis=-2, mode='wrap')

        self.dff = np.roll(self.dff, neural_shift_inds, axis=-1)

        self._population_vector()

        self.outer_ring = self._outer_ring_ts('outer_ring', True, exp_detrend, background_ts, t_sigma)
        self.outer_ring_noz = self._outer_ring_ts('outer_ring', False, exp_detrend, background_ts, t_sigma)

        self._heading_velocity(dh_sigma)

    def _outer_ring_ts(self, roi_name, zscore, exp_detrend, background_ts, t_sigma):
        """dF/F of a single (non-ring) ROI group, squeezed and smoothed in time (if `t_sigma` > 0)."""
        ts = np.squeeze(self.pp.calculate_zscored_F(roi_name, exp_detrend=exp_detrend, zscore=zscore,
                                                    background_ts=background_ts))
        return sp.ndimage.gaussian_filter1d(ts, t_sigma / self.dt) if t_sigma > 0 else ts
