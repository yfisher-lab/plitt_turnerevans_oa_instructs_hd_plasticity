import numpy as np
import pytest
from conftest import CHANNELS, FakeManager, make_trial_dict

import el_hd_plasticity.two_photon as tp

di = tp.dual_imaging
BINS = np.array([0, .1, .2, .5, 1, 2, 4])


def test_offset_stats_columns_and_values(manager):
    df = di.offset_stats(manager)
    assert len(df) == len(manager.sess_df)
    assert {'offset_ch1', 'offset_ch2', 'offset_diff', 'abs_offset_diff', 'pva_diff', 'fwhm_ch1', 'fwhm_ch2'} <= set(df.columns)
    assert df['abs_offset_diff'].between(0, 2 * np.pi).all()


def test_offset_stats_unique_requires_both_conditions_unless_cl_only(manager):
    df = di.offset_stats(manager)
    n_flies = manager.sess_df['fly_id'].nunique()
    both = di.offset_stats_unique(df)
    assert len(both) == 2 * n_flies
    closed_only = df[df['cl'] >= 1]
    assert di.offset_stats_unique(closed_only).empty
    kept = di.offset_stats_unique(closed_only, cl_only=True)
    assert len(kept) == n_flies and (kept['cl'] == 1).all()


@pytest.mark.parametrize('use_abs', [True, False])
def test_rho_stats_structure(manager, use_abs):
    df = di.rho_stats(manager, BINS, abs=use_abs)
    for col in ('rho1_dig', 'rho2_dig', 'pva_diff'):
        assert all(len(r) == len(BINS) for r in df[col])


def test_pvdiff_rho_stats_and_cross_corr(sess_df, load_row):
    subset = FakeManager(sess_df.iloc[:4], load_row)
    pv = di.pvdiff_rho_stats(subset, np.linspace(0, 1, 6))
    assert all(len(r) == 6 for r in pv['pva_diff'])
    cc = di.cross_corr_stats(FakeManager(sess_df.iloc[:2], load_row))
    assert all(len(r) == 101 for r in cc['r'])
    cc_short = di.cross_corr_stats(FakeManager(sess_df.iloc[:2], load_row), delays=np.arange(-5, 6))
    assert all(len(r) == 11 for r in cc_short['r'])      # explicit arguments are honoured


def test_pvdiff_rho_stats_bins_are_left_edges(sess_df, load_row):
    """Bin i holds the time points whose mean rho is in [pv_bins[i], pv_bins[i + 1])."""
    pv_bins = np.linspace(0, 1, 6)
    row = sess_df.iloc[0]
    out = di.pvdiff_rho_stats(FakeManager(sess_df.iloc[:1], load_row), pv_bins)['pva_diff'].iloc[0]

    ts = tp.session.GetTS(load_row(row), channels=(0, 1))
    pva_diff = np.abs(tp.utilities.wrap_angle(np.diff(ts.phi, axis=0))).ravel()
    rho = ts.rho.mean(axis=0)
    for i in range(len(pv_bins) - 1):
        in_bin = (rho >= pv_bins[i]) & (rho < pv_bins[i + 1])
        expected = pva_diff[in_bin].mean() if in_bin.any() else np.nan
        assert np.allclose(out[i], expected, equal_nan=True)


def test_turn_triggered_profiles_shape_and_normalization(sess_df, load_row):
    timeseries = [tp.session.GetTS(load_row(row), channels=(0, 1), dh_sigma=.3) for _, row in sess_df.iloc[:3].iterrows()]
    ccw, cw = di.turn_triggered_profiles(timeseries, 0, 100)
    assert ccw.shape == (3, 2, 16) and cw.shape == (3, 2, 16)
    assert np.allclose(ccw.min(axis=-1), 0) and np.allclose(ccw.max(axis=-1), 1)
    ccw_max, cw_max = di.turn_triggered_profiles(timeseries, 0, 100, alignment='max')
    assert ccw_max.shape == ccw.shape
    with pytest.raises(ValueError):
        di.turn_triggered_profiles(timeseries, 0, 100, alignment='nope')
    fig, pva = di.plot_turn_profiles(ccw, cw)
    assert set(pva) == {'clockwise', 'counter-clockwise'} and pva['clockwise'].shape == (2,)


def test_dte_variants():
    data = {'C:\\x\\flyA': {'t1': make_trial_dict(1), 't2': make_trial_dict(2)}}
    lookup = {k: CHANNELS for k in data}
    stats = di.offset_stats_DTE(data, lookup)
    assert list(stats['fly_id'].unique()) == ['flyA'] and len(stats) == 2
    rho = di.rho_stats_DTE(data, lookup, BINS, abs=False)
    assert len(rho) == 2 and rho['fly_id'].iloc[0] == 'flyA'
