import numpy as np
import pytest
from conftest import CHANNELS, make_trial_dict

import el_hd_plasticity.two_photon as tp

N_ROIS = 16


def reference_heading_alignment(dff, heading, bins):
    """Slow, obviously-correct version of ``heading_aligned``: np.roll each time point."""
    dig = np.digitize(heading, bins) - 1
    out = np.zeros_like(dff)
    for t in range(dff.shape[-1]):
        out[..., t] = np.roll(dff[..., t], -dig[t] + 8, axis=-1)     # ROI axis is last once time is indexed out
    return out


def test_single_channel_shapes(ts, pp):
    n = pp.n
    assert ts.dff.shape == (N_ROIS, n)
    for name in ('heading', 'heading_sm', 'dh', 'rho', 'phi', 'offset', 'pv_c', 'outer_ring'):
        assert getattr(ts, name).shape == (n,), name
    assert ts.dff_h_aligned.shape == ts.dff.shape
    assert np.isscalar(ts.fwhm) or np.ndim(ts.fwhm) == 0


def test_multi_channel_shapes_and_fwhm(ts2, pp):
    assert ts2.dff.shape == (2, N_ROIS, pp.n)
    assert ts2.rho.shape == ts2.phi.shape == ts2.offset.shape == (2, pp.n)
    assert len(ts2.fwhm) == 2


def test_channel_selection_is_consistent(ts, ts2):
    # last channel of the two-channel session == the default single-channel session
    np.testing.assert_array_equal(ts.dff, ts2.dff[-1])
    np.testing.assert_array_equal(ts.offset, ts2.offset[-1])


def test_offset_is_wrapped_heading_difference(ts):
    np.testing.assert_allclose(np.exp(1j * ts.offset), np.exp(1j * (ts.phi - ts.heading)), atol=1e-12)
    assert np.all(np.abs(ts.offset) <= np.pi + 1e-12)
    np.testing.assert_allclose(ts.offset_c, np.exp(1j * ts.offset))


def test_population_vector_matches_definition(ts):
    ang = np.linspace(-np.pi, np.pi, ts.n_rois)[:, None]
    pv = (ts.dff * np.exp(1j * ang)).mean(axis=0)
    np.testing.assert_allclose(ts.pv_c, pv, atol=1e-12)
    np.testing.assert_allclose(ts.rho, np.abs(pv), atol=1e-12)


@pytest.mark.parametrize('channels', [-1, [0, 1]])
def test_heading_alignment_matches_roll_reference(pp, channels):
    ts = tp.session.GetTS(pp, channels=channels)
    ref = reference_heading_alignment(ts.dff, ts.heading, ts.heading_bins)
    np.testing.assert_array_equal(ts.dff_h_aligned, ref)


def test_heading_velocity_starts_at_zero_and_is_smoothed_on_request(pp):
    a = tp.session.GetTS(pp)
    b = tp.session.GetTS(pp, dh_sigma=.3)
    assert a.dh[0] == 0
    assert np.std(b.dh) < np.std(a.dh)


def test_neural_shift_moves_activity_in_time(pp):
    a = tp.session.GetTS(pp, neural_shift=0, t_sigma=0, circ_sigma=0)
    b = tp.session.GetTS(pp, neural_shift=-.5, t_sigma=0, circ_sigma=0)
    shift = int(-.5 / a.dt)
    np.testing.assert_array_equal(b.dff, np.roll(a.dff, shift, axis=-1))


def test_dte_session_uses_closed_loop_samples_only():
    td = make_trial_dict(3)
    ts = tp.session.GetTS_DTE(td, CHANNELS)
    n_closed = int((td['behavDat']['closed'] == 1).sum())
    assert ts.dff.shape == (2, N_ROIS, n_closed)
    assert ts.offset.shape == (2, n_closed) and len(ts.fwhm) == 2


def test_dte_closed_false_keeps_all_samples():
    td = make_trial_dict(3)
    ts = tp.session.GetTS_DTE(td, CHANNELS, closed=False)
    assert ts.dff.shape[-1] == len(td['behavDat'])


def test_no_temporal_smoothing_leaves_outer_ring_finite(pp):
    ts = tp.session.GetTS(pp, t_sigma=0)
    assert np.isfinite(ts.outer_ring).all() and np.isfinite(ts.outer_ring_noz).all()
