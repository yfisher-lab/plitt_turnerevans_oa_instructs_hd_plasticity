import numpy as np
import pandas as pd
import pytest
from conftest import FakeManager

from el_hd_plasticity.two_photon import ex_vivo

MARK_POINTS = {'initial_delay': 5000, 'duration': 500, 'inter_point_delay': 9500, 'repetitions': 3}


class FakeExVivoSession:
    """Outer-ring fluorescence with an exponential response after each of three pulses."""

    def __init__(self, amplitude=.4, dt=.1, n=1500):
        t = np.arange(n) * dt
        self.metadata = {'frame_times': np.stack([t, t + dt], axis=1), 'mark_points': MARK_POINTS}
        f = np.full(n, 10.)
        for stim in ex_vivo.OptoResponses.stim_times_from_metadata(MARK_POINTS):
            after = t > stim + .2
            f[after] += 10 * amplitude * np.exp(-(t[after] - stim) / 1.5)
        self.timeseries = {'outer_ring': f[np.newaxis, np.newaxis, :]}


def test_stim_times_follow_the_mark_points_protocol():
    assert np.allclose(ex_vivo.OptoResponses.stim_times_from_metadata(MARK_POINTS), [5, 15, 25])


def test_responses_are_baseline_normalized_peaks_on_a_common_clock():
    resp = ex_vivo.OptoResponses.from_session('fly', 'exp', 's-000', FakeExVivoSession(amplitude=.4))
    assert resp.dff.shape == (3, len(resp.plot_time)) and resp.max_dff.shape == resp.aoc.shape == (3,)
    assert np.isnan(resp.dff[:, (resp.plot_time >= -.2) & (resp.plot_time <= .22)]).all()   # stimulation artifact
    assert np.allclose(np.nanmean(resp.dff[:, resp.plot_time < -.2]), 0, atol=1e-9)           # baseline is zero
    assert resp.max_dff == pytest.approx(.4 * np.exp(-.3 / 1.5), abs=.03)                     # peak just after the artifact
    after = resp.plot_time > ex_vivo.ARTIFACT[1]
    assert np.allclose(resp.aoc, np.nansum(resp.dff[:, after], axis=1) * ex_vivo.STEP)               # dF/F x s
    assert np.allclose(resp.aoc, .4 * 1.5 * np.exp(-.22 / 1.5), rtol=.1)                          # integral of the exponential


def test_responses_are_immutable():
    resp = ex_vivo.OptoResponses.from_session('fly', 'exp', 's-000', FakeExVivoSession())
    with pytest.raises(ValueError):
        resp.dff[0, 0] = 1
    with pytest.raises(AttributeError):
        resp.fly_id = 'other'
    assert 'fly_id=fly' in repr(resp)


def test_session_responses_table_over_cell_types():
    def manager(amplitude):
        df = pd.DataFrame({'fly_id': ['a', 'b'], 'condition': ['exp', 'nocschr_control'], 'session': ['s0', 's1']})
        return FakeManager(df, lambda row: FakeExVivoSession(amplitude if row['condition'] == 'exp' else .02))

    table, plot_time = ex_vivo.session_responses({'ER': manager(.5), 'EPG': manager(.2)})
    assert list(table.columns) == ['cell_type', 'fly_id', 'condition', 'session', 'mean_dff', 'aoc', 'max_dff']
    assert len(table) == 4 and len(table['mean_dff'].iloc[0]) == len(plot_time)
    peaks = table.set_index(['cell_type', 'condition'])['max_dff']
    assert peaks['ER', 'exp'] > peaks['EPG', 'exp'] > peaks['EPG', 'nocschr_control']
