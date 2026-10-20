"""Shared fixtures: synthetic imaging sessions that stand in for preprocessed data.

The real data are not part of the repository, so tests build small sessions with a known
structure (an activity bump that follows heading) and exercise the analysis code on those.
"""

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as tp


class FakePP:
    """Minimal stand-in for ``EBImagingSession``: what :class:`session.GetTS` reads."""

    def __init__(self, seed=0, n=3000, dt=0.05, n_rois=16, n_ch=2):
        rng = np.random.default_rng(seed)
        t = np.arange(n) * dt
        heading_raw = np.cumsum(rng.normal(0, 0.05, n)) + 0.5 * np.sin(t / 20)
        self.n = n
        self.voltage_recording_aligned = pd.DataFrame({'Time(ms)': t * 1000 + 1234.0, ' Heading': heading_raw})
        roi_ang = np.linspace(-np.pi, np.pi, n_rois)[:, None]
        center = -heading_raw - np.pi + rng.normal(0, .4, n)
        self._rois = np.stack([np.exp(2 * np.cos(roi_ang - center[None, :] - 0.3 * c)) + rng.normal(0, .3, (n_rois, n))
                               for c in range(n_ch)])
        self._outer = rng.normal(size=(1, 1, n)) + 3

    def calculate_zscored_F(self, name, exp_detrend=True, zscore=True, background_ts='background'):
        out = (self._rois if name == 'rois' else self._outer).copy()
        if zscore:
            out = (out - out.mean(axis=-1, keepdims=True)) / out.std(axis=-1, keepdims=True)
        return out


def make_trial_dict(seed=0, n=3000, dt=0.05, n_rois=16):
    """Synthetic Turner-Evans style trial: behavior table plus EPG and EL ROI timeseries."""
    rng = np.random.default_rng(seed)
    rot = np.cumsum(rng.normal(0, 3, n)) % 360 - 180
    closed = np.ones(n, dtype=int)
    closed[:300] = 0
    behav = pd.DataFrame({'Elapsed time': np.arange(n) * dt, 'Rotational offset': rot, 'closed': closed})
    ang = np.linspace(-np.pi, np.pi, n_rois)[None, :]
    center = (-np.pi / 180 * rot)[:, None]
    return {'behavDat': behav,
            'epg_roi': np.exp(2 * np.cos(ang - center)) + rng.normal(0, .3, (n, n_rois)),
            'el_roi': np.exp(2 * np.cos(ang - center - .4)) + rng.normal(0, .3, (n, n_rois))}


CHANNELS = {'EPG': 'epg_roi', 'EL': 'el_roi'}


@pytest.fixture
def pp():
    return FakePP(seed=1)


@pytest.fixture
def ts(pp):
    return tp.session.GetTS(pp)


@pytest.fixture
def ts2(pp):
    return tp.session.GetTS(pp, channels=[0, 1])


@pytest.fixture
def sess_df():
    """6 flies x 4 sessions: two closed-loop blocks (cl=1, 2) and two dark sessions."""
    rows = []
    for f in range(6):
        for s, (cl, dark) in enumerate([(1, 0), (2, 0), (0, 1), (0, 1)]):
            rows.append({'fly_id': f'fly{f}', 'closed_loop': cl, 'dark': dark, 'rnai': f % 2,
                         'condition': 'ab'[f % 2], 'scan_type': 'baseline', 'seed': 10 * f + s})
    return pd.DataFrame(rows)


@pytest.fixture
def load_row():
    return lambda row: FakePP(seed=int(row['seed']), n=1500)


class FakeManager:
    """Minimal stand-in for ``SessionManager``: a session table plus a row loader."""

    def __init__(self, sess_df, load_row):
        self.sess_df = sess_df
        self.load_row = load_row

    def iterrows(self):
        return self.sess_df.iterrows()


@pytest.fixture
def manager(sess_df, load_row):
    return FakeManager(sess_df, load_row)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')
