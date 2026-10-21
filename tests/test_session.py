import dataclasses

import numpy as np
import pytest
from conftest import CHANNELS, make_trial_dict

import el_hd_plasticity.two_photon as tp

N_ROIS = 16


def ring_profile(x, n=N_ROIS):
    """Band-limited (harmonics 1-3) periodic bump on a ring of n ROIs, evaluated at ROI coordinates x."""
    return sum(w * np.cos(2 * np.pi * k * x / n) for k, w in ((1, 1.0), (2, .5), (3, .25)))


def aligned_by(dff, heading):
    """Heading alignment of a plain array (the pure function behind ``dff_h_aligned``)."""
    return tp.session._align_to_heading(dff, np.asarray(heading))


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
    ang = (-np.pi + (np.arange(ts.n_rois) + .5) * 2 * np.pi / ts.n_rois)[:, None]    # ROI j at its bin center
    pv = (ts.dff * np.exp(1j * ang)).mean(axis=0)
    np.testing.assert_allclose(ts.pv_c, pv, atol=1e-12)
    np.testing.assert_allclose(ts.rho, np.abs(pv), atol=1e-12)


def test_alignment_by_whole_rois_is_a_roll():
    rng = np.random.default_rng(0)
    k = np.arange(N_ROIS)                                         # heading exactly at the center of ROI k
    heading = -np.pi + 2 * np.pi * (k + .5) / N_ROIS
    dff = rng.normal(size=(N_ROIS, N_ROIS))
    aligned = aligned_by(dff, heading)
    for t, kt in enumerate(k):
        np.testing.assert_allclose(aligned[:, t], np.roll(dff[:, t], -(kt - 8)), atol=1e-9)


def test_alignment_is_sub_roi_a_bump_following_heading_lands_exactly_at_the_center():
    rng = np.random.default_rng(1)
    heading = rng.uniform(-np.pi, np.pi, 50)
    peak = (heading + np.pi) / (2 * np.pi) * N_ROIS - .5           # bump peak at the heading, in fractional ROI indices
    dff = ring_profile(np.arange(N_ROIS)[:, None] - peak[None, :])
    aligned = aligned_by(dff, heading)
    expected = ring_profile(np.arange(N_ROIS)[:, None] - 8.0) * np.ones((1, 50))
    np.testing.assert_allclose(aligned, expected, atol=1e-10)      # peak at ROI 8 for every heading


def test_alignment_preserves_total_activity_and_is_real():
    rng = np.random.default_rng(2)
    dff = rng.normal(size=(2, N_ROIS, 30))
    heading = rng.uniform(-np.pi, np.pi, 30)
    aligned = aligned_by(dff, heading)
    assert aligned.shape == dff.shape and np.isrealobj(aligned)
    np.testing.assert_allclose(aligned.sum(axis=-2), dff.sum(axis=-2), atol=1e-10)


def test_alignment_treats_channels_independently():
    rng = np.random.default_rng(3)
    dff = rng.normal(size=(2, N_ROIS, 30))
    heading = rng.uniform(-np.pi, np.pi, 30)
    both = aligned_by(dff, heading)
    for ch in (0, 1):
        np.testing.assert_allclose(both[ch], aligned_by(dff[ch], heading), atol=1e-12)


@pytest.mark.parametrize('channels', [-1, [0, 1]])
def test_session_alignment_keeps_shape_and_total_activity(pp, channels):
    ts = tp.session.GetTS(pp, channels=channels)
    assert ts.dff_h_aligned.shape == ts.dff.shape
    np.testing.assert_allclose(ts.dff_h_aligned.sum(axis=-2), ts.dff.sum(axis=-2), atol=1e-9)


def test_session_alignment_centers_the_population_vector_on_heading(ts):
    # offset = phi - heading; after alignment the bump sits where the offset puts it relative to ROI 8
    ang = np.linspace(-np.pi, np.pi, ts.n_rois)[:, None]
    pv = (ts.dff_h_aligned * np.exp(1j * ang)).mean(axis=0)
    assert np.abs(np.angle(np.mean(pv))) < 1.0                     # clustered, not spread over the ring


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


# --- read-only attributes ------------------------------------------------------------------------
BASE_ATTRS = ['dff', 'heading', 'heading_sm', 'dh', 'offset', 'phi', 'rho', 'pv_c', 'dff_h_aligned',
              'time', 'n_rois', 'dt', 'fwhm']
ARRAY_ATTRS = ['dff', 'heading', 'heading_sm', 'dh', 'offset', 'phi', 'rho', 'pv_c', 'dff_h_aligned']


@pytest.mark.parametrize('name', BASE_ATTRS + ['offset_c', 'pp', 'dff_noz', 'outer_ring', 'outer_ring_noz'])
def test_getts_attributes_cannot_be_reassigned_or_deleted(ts, name):
    with pytest.raises(AttributeError):
        setattr(ts, name, 0)
    with pytest.raises(AttributeError):
        delattr(ts, name)


@pytest.mark.parametrize('name', ARRAY_ATTRS + ['dff_noz', 'outer_ring', 'outer_ring_noz'])
def test_getts_arrays_cannot_be_modified_in_place(ts, name):
    arr = getattr(ts, name)
    assert not arr.flags.writeable
    with pytest.raises(ValueError):
        arr[..., 0] = 0


def test_multichannel_arrays_are_frozen_too(ts2):
    for name in ARRAY_ATTRS + ['fwhm']:
        assert not getattr(ts2, name).flags.writeable
    assert ts2.fwhm.shape == (2,)


def test_dte_attributes_are_read_only_and_frozen():
    ts = tp.session.GetTS_DTE(make_trial_dict(3), CHANNELS)
    for name in BASE_ATTRS + ['trial_dict', 'channel_mapping']:
        with pytest.raises(AttributeError):
            setattr(ts, name, 0)
    for name in ARRAY_ATTRS:
        assert not getattr(ts, name).flags.writeable
    assert isinstance(ts.heading, np.ndarray)


def test_offset_c_is_computed_from_offset(ts):
    np.testing.assert_allclose(ts.offset_c, np.exp(1j * ts.offset))


def test_copies_of_read_only_arrays_are_editable(ts):
    dff = ts.dff.copy()
    dff[0] = 0
    assert ts.dff[0].any()                          # the session is untouched


# --- construction: from_session / from_trial_dict, the GetTS aliases, direct construction ------------
def test_from_session_is_what_getts_calls(pp):
    a = tp.session.SessionTimeseries.from_session(pp, channels=[0, 1], dh_sigma=.3)
    b = tp.session.GetTS(pp, channels=[0, 1], dh_sigma=.3)
    assert type(a) is type(b) is tp.session.SessionTimeseries
    for name in ARRAY_ATTRS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_from_trial_dict_is_what_getts_dte_calls():
    td = make_trial_dict(3)
    a = tp.session.TrialTimeseries.from_trial_dict(td, CHANNELS)
    b = tp.session.GetTS_DTE(td, CHANNELS)
    assert type(a) is type(b) is tp.session.TrialTimeseries
    np.testing.assert_array_equal(a.dff, b.dff)


def test_session_and_trial_timeseries_share_the_ring_base_class(ts):
    assert isinstance(ts, tp.session.RingTimeseries)
    assert isinstance(tp.session.GetTS_DTE(make_trial_dict(1), CHANNELS), tp.session.RingTimeseries)


def test_extra_fields_of_each_source(ts, pp):
    assert ts.pp is pp
    assert ts.outer_ring.shape == ts.outer_ring_noz.shape == ts.heading.shape
    assert ts.dff_noz.shape == ts.dff.shape


def test_directly_constructed_objects_freeze_their_arrays(ts):
    fields = {f.name: getattr(ts, f.name) for f in dataclasses.fields(tp.session.SessionTimeseries)}
    fields['dff'] = np.array(ts.dff)                 # a fresh, writeable array
    assert fields['dff'].flags.writeable
    rebuilt = tp.session.SessionTimeseries(**fields)
    assert not rebuilt.dff.flags.writeable and not fields['dff'].flags.writeable


def test_repr_is_short(ts, ts2):
    assert repr(ts).startswith('SessionTimeseries(n_rois=16, n_samples=3000')
    assert 'n_channels=2' in repr(ts2) and len(repr(ts2)) < 120


def test_objects_compare_by_identity_not_by_array_contents(pp):
    a, b = tp.session.GetTS(pp), tp.session.GetTS(pp)
    assert a == a and a != b                          # no elementwise array comparison
    assert len({a, b}) == 2                           # hashable


def test_neural_shift_applies_to_dff_noz_and_outer_ring(pp):
    base = tp.session.GetTS(pp, neural_shift=0)
    shifted = tp.session.GetTS(pp, neural_shift=-.2)
    k = int(-.2 / base.dt)
    for name in ('dff', 'dff_noz', 'outer_ring', 'outer_ring_noz'):
        assert np.allclose(getattr(shifted, name), np.roll(getattr(base, name), k, axis=-1)), name
