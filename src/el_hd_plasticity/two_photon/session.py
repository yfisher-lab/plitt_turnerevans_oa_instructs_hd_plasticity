"""Session timeseries extraction and analysis for two-photon imaging data.

Provides classes for extracting neural timeseries, computing circular statistics
(heading, offset, population vector), and aligning activity to behavior.
"""

import numpy as np
import scipy as sp

import SessionTools.two_photon as st2p

class GetTS_DTE():
    """Extract and compute timeseries statistics from Dan Turner-Evans behavioral/neural data.
    
    Computes dF/F for multi-channel data, calculates circular statistics (heading offset,
    population vector), and aligns neural activity to heading for each trial epoch.
    
    Attributes:
        trial_dict: Dict with behavioral data and neural timeseries by channel.
        channel_mapping: Dict mapping channel names (e.g., 'EPG', 'EL') to trial_dict keys.
        dff: Neural timeseries; shape (n_channels, n_rois, n_timepoints).
        heading: Animal heading in radians.
        heading_sm: Smoothed heading.
        offset: Heading offset (phi - heading) per channel; shape (n_channels, n_timepoints).
        phi: Population vector angle per channel.
        rho: Population vector magnitude per channel.
        dff_h_aligned: Neural activity aligned to heading bins.
        offset_c: Complex representation of offset (exp(1j*offset)).
        fwhm: Full-width at half-max of population tuning per channel.
    """

    def __init__(self, trial_dict, channel_mapping, **kwargs):
        self.trial_dict = trial_dict
        self.channel_mapping = channel_mapping
        self.dff = None
        self.heading = None
        self.heading_sm = None
        self.offset = None
        self.phi = None
        self.rho = None
        self.heading_bins = None
        self.dff_h_aligned = None
        self.time = None
        self.n_rois = None
        self.dt = None
        self.fwhm = None
        self.pv_c = None

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
        mask = behav_data['closed']==1

        self.time = behav_data['Elapsed time'].loc[mask]

        self.dt = np.diff(self.time).mean()
        neural_shift_inds = int(neural_shift/self.dt)

        self.heading = np.angle(np.exp(1j*(-np.pi/180*behav_data['Rotational offset'].loc[mask])))
        x_h, y_h = st2p.utilities.pol2cart(np.ones_like(self.heading), self.heading)
        if h_sigma > 0:
            x_h, y_h = sp.ndimage.gaussian_filter1d(x_h, h_sigma/self.dt), sp.ndimage.gaussian_filter1d(y_h, h_sigma/self.dt)
        _, self.heading_sm = st2p.utilities.cart2pol(x_h, y_h)

        self.dff = np.stack((self.trial_dict[self.channel_mapping['EPG']][mask,:].T, self.trial_dict[self.channel_mapping['EL']][mask,:].T), axis=0)
        if t_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, t_sigma/self.dt, axis=-1)
        if circ_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, circ_sigma, axis=-2, mode='wrap')

        self.dff = np.roll(self.dff, neural_shift_inds, axis=-1)
        
        self.n_rois = self.dff.shape[-2]  
        self.rho = np.zeros((self.dff.shape[0], self.dff.shape[-1]))
        self.phi = np.zeros((self.dff.shape[0], self.dff.shape[-1]))
        self.offset = np.zeros((self.dff.shape[0], self.dff.shape[-1]))
        self.pv_c = np.zeros((self.dff.shape[0], self.dff.shape[-1]), dtype=complex)

        for chan in range(self.dff.shape[0]):
            x_f,y_f = st2p.utilities.pol2cart(self.dff[chan, :, :] ,np.linspace(-np.pi,np.pi,num=self.n_rois)[:,np.newaxis])
            self.pv_c[chan,:] = x_f.mean(axis=0) + 1j*y_f.mean(axis=0)
            self.rho[chan, :], self.phi[chan,:] = st2p.utilities.cart2pol(x_f.mean(axis=0), y_f.mean(axis=0))
        
            _,self.offset[chan,:] = st2p.utilities.cart2pol(*st2p.utilities.pol2cart(np.ones(self.heading.shape),self.phi[chan,:]-self.heading))
            

        dh = np.diff(np.unwrap(self.heading_sm))/self.dt
        self.dh = np.concatenate([[0], dh])
        if dh_sigma > 0:
            self.dh = sp.ndimage.gaussian_filter1d(self.dh, dh_sigma/self.dt)

    @property
    def offset_c(self):
        """Complex representation of heading offset as unit vectors.
        
        Returns:
            ndarray of dtype complex with magnitude 1 and angle equal to self.offset.
        """
        return np.exp(1j*self.offset)
    
    def heading_aligned(self):
        """Align neural activity to heading bins by rolling each timepoint 
        
        Creates dff_h_aligned where each heading is rotated to a fixed reference, enabling
        averaging of activity at each heading.
        """
        self.heading_bins = np.linspace(-np.pi, np.pi, num=self.n_rois+1)
        heading_dig = np.digitize(self.heading, self.heading_bins)-1

        self.dff_h_aligned = np.zeros_like(self.dff)
        if self.dff.ndim == 2:
            for ind in range(self.heading.shape[0]):
                self.dff_h_aligned[:,ind] = np.roll(self.dff[:,ind], -heading_dig[ind]+8)
        else:
            for chan in range(self.dff.shape[0]):
                for ind in range(self.heading.shape[0]):
                    self.dff_h_aligned[chan,:,ind] = np.roll(self.dff[chan,:,ind], -heading_dig[ind]+8)

    def calc_fwhm(self):
        """Compute full-width at half-max (FWHM) of population heading tuning.
        
        For each channel, aligns activity to peak response and computes the angular width
        where normalized response >= 0.5.
        """
        def _fwhm(dff):
            max_inds = np.argmax(dff, axis=0)
            dff_aligned = np.zeros_like(dff)
            for i in range(dff.shape[1]):
                dff_aligned[:,i] = np.roll(dff[:,i], -max_inds[i])
            mu = dff_aligned.mean(axis=1)
            mu = (mu-np.amin(mu))/(np.amax(mu)-np.amin(mu))
            return ((mu>=.5).sum()*2*np.pi/self.n_rois)

        if self.dff.ndim > 2: # if there are multiple channels
            self.fwhm = [_fwhm(self.dff[i,:,:]) for i in range(self.dff.shape[0])]
        else:
            self.fwhm = _fwhm(self.dff)


class GetTS():
    """Extract and compute timeseries statistics from preprocessed imaging session.
    
    Computes dF/F for neural ROIs, calculates circular statistics (heading offset,
    population vector),  and aligns neural activity to heading.
    
    Attributes:
        pp: Preprocessed session object with voltage recordings and calcium imaging data.
        dff: Z-scored dF/F timeseries; shape (n_rois, n_timepoints) or (n_channels, n_rois, n_timepoints).
        heading: Animal heading in radians.
        heading_sm: Smoothed heading.
        offset: Heading offset (phi - heading).
        phi: Population vector angle.
        rho: Population vector magnitude.
        dff_h_aligned: Neural activity aligned to heading bins.
        offset_c: Complex representation of offset (exp(1j*offset)).
        outer_ring: Activity from EB outer ring region.
        fwhm: Full-width at half-max of population tuning.
        dh: Time derivative of heading.
    """
    
    def __init__(self, pp, **kwargs):
        """Initialize session and compute timeseries and statistics.
        
        Args:
            pp: Preprocessed session object.
            **kwargs: Passed to get_ts() method.
        """
        self.pp = pp
        self.dff = None
        self.heading = None
        self.heading_sm = None
        self.offset = None
        self.phi = None
        self.rho = None
        self.heading_bins = None
        self.dff_h_aligned = None
        self.time = None
        self.n_rois = None
        self.dt = None
        self.dh = None
        self.fwhm = None
        self.pv_c = None

        self.outer_ring = None

        self.get_ts(**kwargs)
        self.heading_aligned()
        self.calc_fwhm()

        
        
    def get_ts(self, channels=-1, exp_detrend=True, background_ts='background',
               circ_sigma=.5, t_sigma=.1, h_sigma=.1, neural_shift=-.2, dh_sigma=0):
        """Extract and compute timeseries and circular statistics from preprocessed session.
        
        Args:
            channels: ROI channels to extract; -1 uses all ROIs (default).
            exp_detrend: If True, exponential detrending is applied to dF/F.
            zscore: If True, z-score dF/F across time.
            background_ts: Background timeseries to use for dF computation.
            circ_sigma: Gaussian smoothing width (in ROI bins) across neural population.
            t_sigma: Gaussian smoothing width (in seconds) along time axis.
            h_sigma: Gaussian smoothing width (in seconds) for heading.
            neural_shift: Time offset (in seconds) to align neural to behavior (typically negative).
            dh_sigma: Gaussian smoothing width for heading derivative.
        """
        self.time = self.pp.voltage_recording_aligned['Time(ms)']/1000
        self.time = self.time-self.time[0]
        
        
        self.dt = np.diff(self.time).mean()
        neural_shift_inds = int(neural_shift/self.dt)
        
        self.heading = np.angle(np.exp(1j*(-1*self.pp.voltage_recording_aligned[' Heading'].to_numpy()-np.pi)))
        x_h, y_h = st2p.utilities.pol2cart(np.ones_like(self.heading), self.heading)
        if h_sigma > 0:
            x_h, y_h = sp.ndimage.gaussian_filter1d(x_h, h_sigma/self.dt), sp.ndimage.gaussian_filter1d(y_h, h_sigma/self.dt)
        _, self.heading_sm = st2p.utilities.cart2pol(x_h, y_h)
        
        self.dff = self.pp.calculate_zscored_F('rois', exp_detrend=exp_detrend, zscore=True, 
                                               background_ts=background_ts)[channels,:,:]

        self.dff_noz = self.pp.calculate_zscored_F('rois', exp_detrend=exp_detrend, zscore=False, 
                                               background_ts=background_ts)[channels,:,:]
        
        if t_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, t_sigma/self.dt, axis=-1)
        if circ_sigma > 0:
            self.dff = sp.ndimage.gaussian_filter1d(self.dff, circ_sigma, axis=-2, mode='wrap')
        # self.dff = sp.ndimage.gaussian_filter1d(sp.ndimage.gaussian_filter1d(self.dff, t_sigma/self.dt, axis=-1),
        #                                         circ_sigma,axis=-2, mode='wrap')
        
        # self.dff -= np.amin(self.dff,axis=-1, keepdims=True) 
        # self.dff += 1E-3

        self.dff = np.roll(self.dff, neural_shift_inds, axis=-1)
        
        self.n_rois = self.dff.shape[-2]  
        if self.dff.ndim == 2:
            x_f,y_f = st2p.utilities.pol2cart(self.dff ,np.linspace(-np.pi,np.pi,num=self.n_rois)[:,np.newaxis])
            self.pv_c = x_f.mean(axis=0) + 1j*y_f.mean(axis=0)
            self.rho, self.phi = st2p.utilities.cart2pol(x_f.mean(axis=0), y_f.mean(axis=0))
            
            _,self.offset = st2p.utilities.cart2pol(*st2p.utilities.pol2cart(np.ones(self.heading.shape),self.phi-self.heading))

        else: # if there are multiple channels
            self.rho = np.zeros((self.dff.shape[0], self.dff.shape[-1]))
            self.phi = np.zeros((self.dff.shape[0], self.dff.shape[-1]))
            self.offset = np.zeros((self.dff.shape[0], self.dff.shape[-1]))
            self.pv_c = np.zeros((self.dff.shape[0], self.dff.shape[-1]), dtype=complex)

            for chan in range(self.dff.shape[0]):
                x_f,y_f = st2p.utilities.pol2cart(self.dff[chan, :, :] ,np.linspace(-np.pi,np.pi,num=self.n_rois)[:,np.newaxis])
                self.pv_c[chan,:] = x_f.mean(axis=0) + 1j*y_f.mean(axis=0)
                self.rho[chan, :], self.phi[chan,:] = st2p.utilities.cart2pol(x_f.mean(axis=0), y_f.mean(axis=0))
            
                _,self.offset[chan,:] = st2p.utilities.cart2pol(*st2p.utilities.pol2cart(np.ones(self.heading.shape),self.phi[chan,:]-self.heading))
            

        self.outer_ring = np.squeeze(self.pp.calculate_zscored_F('outer_ring', exp_detrend=exp_detrend, zscore=True,
                                                     background_ts=background_ts))
        self.outer_ring = sp.ndimage.gaussian_filter1d(self.outer_ring, t_sigma/self.dt)

        self.outer_ring_noz = np.squeeze(self.pp.calculate_zscored_F('outer_ring', exp_detrend=exp_detrend, zscore=False,
                                                             background_ts=background_ts))
        self.outer_ring_noz = sp.ndimage.gaussian_filter1d(self.outer_ring_noz, t_sigma/self.dt)

        dh = np.diff(np.unwrap(self.heading_sm))/self.dt
        self.dh = np.concatenate([[0], dh])
        if dh_sigma > 0:
            self.dh = sp.ndimage.gaussian_filter1d(self.dh, dh_sigma/self.dt)
        
        
    def heading_aligned(self):
        """Align neural activity to heading bins by rolling each timepoint to a canonical position.
        
        Creates dff_h_aligned where each heading is rotated to a fixed reference, enabling
        averaging of activity at each heading.
        """
        self.heading_bins = np.linspace(-np.pi, np.pi, num=self.n_rois+1)
        heading_dig = np.digitize(self.heading, self.heading_bins)-1

        self.dff_h_aligned = np.zeros_like(self.dff)
        if self.dff.ndim == 2:
            for ind in range(self.heading.shape[0]):
                self.dff_h_aligned[:,ind] = np.roll(self.dff[:,ind], -heading_dig[ind]+8)
        else:
            for chan in range(self.dff.shape[0]):
                for ind in range(self.heading.shape[0]):
                    self.dff_h_aligned[chan,:,ind] = np.roll(self.dff[chan,:,ind], -heading_dig[ind]+8)
    
    @property
    def offset_c(self):
        """Complex representation of heading offset as unit vectors.
        
        Returns:
            ndarray of dtype complex with magnitude 1 and angle equal to self.offset.
        """
        return np.exp(1j*self.offset)
    
    def calc_fwhm(self):
        """Compute full-width at half-max (FWHM) of population heading tuning.
        
        For each channel (if multi-channel), aligns activity to peak response and computes
        the angular width where normalized response >= 0.5.
        """
        def _fwhm(dff):
            max_inds = np.argmax(dff, axis=0)
            dff_aligned = np.zeros_like(dff)
            for i in range(dff.shape[1]):
                dff_aligned[:,i] = np.roll(dff[:,i], -max_inds[i])
            mu = dff_aligned.mean(axis=1)
            mu = (mu-np.amin(mu))/(np.amax(mu)-np.amin(mu))
            return ((mu>=.5).sum()*2*np.pi/self.n_rois)

        if self.dff.ndim > 2: # if there are multiple channels
            self.fwhm = [_fwhm(self.dff[i,:,:]) for i in range(self.dff.shape[0])]
        else:
            self.fwhm = _fwhm(self.dff)