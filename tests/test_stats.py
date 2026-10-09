import numpy as np
import pandas as pd
import pytest

import el_hd_plasticity.two_photon as tp

BINS = np.array([0, .1, .2, .5, 1, 2, 4])
SIGNED_BINS = np.array([-4, -2, -1, -.5, 0, .5, 1, 2, 4])


@pytest.fixture
def offsets(sess_df, load_row):
    return tp.stats.offset_stats(sess_df, load_row)


def test_offset_stats_has_one_row_per_session(offsets, sess_df):
    assert len(offsets) == len(sess_df)
    assert list(offsets.columns) == ['fly_id', 'cl', 'offset_var', 'offset_mean', 'offset_mag']
    assert offsets['offset_mag'].between(0, 1).all()
    assert offsets['offset_mean'].abs().le(np.pi).all()


def test_offset_stats_meta_cols_are_renamed_and_ordered(sess_df, load_row):
    df = tp.stats.offset_stats(sess_df, load_row, meta_cols={'fly': 'fly_id', 'group': 'rnai'})
    assert list(df.columns)[:2] == ['fly', 'group']


def test_offset_stats_unique_aggregates_closed_loop_and_dark(offsets, sess_df):
    both = offsets.assign(dark=sess_df['dark'].values)
    uniq = tp.stats.offset_stats_unique(both)
    assert len(uniq) == 2 * sess_df['fly_id'].nunique()
    assert set(uniq['cl']) == {0, 1} and set(uniq['dark']) == {0, 1}
    cl = uniq[(uniq['fly_id'] == 'fly0') & (uniq['cl'] == 1)].iloc[0]
    expected = both[(both['fly_id'] == 'fly0') & (both['cl'] >= 1)]['offset_var'].mean()
    assert np.isclose(cl['offset_var'], expected)
    assert np.isclose(cl['offset_var_logit'], np.log(expected / (1 - expected)))


def test_offset_stats_unique_min_cl_and_missing_condition(offsets, sess_df):
    both = offsets.assign(dark=sess_df['dark'].values)
    strict = tp.stats.offset_stats_unique(both, min_cl=2)
    cl = strict[(strict['fly_id'] == 'fly0') & (strict['cl'] == 1)].iloc[0]
    assert np.isclose(cl['offset_var'], both[(both['fly_id'] == 'fly0') & (both['cl'] == 2)]['offset_var'].iloc[0])
    no_dark = both[both['dark'] == 0]
    assert tp.stats.offset_stats_unique(no_dark).empty


def test_offset_stats_unique_without_dark_column_uses_cl_zero(offsets):
    uniq = tp.stats.offset_stats_unique(offsets, dark_col=None)
    assert len(uniq) == 2 * offsets['fly_id'].nunique()


def test_offset_stats_plot_is_wide_with_group_column(offsets, sess_df):
    both = offsets.assign(dark=sess_df['dark'].values, rnai=sess_df['rnai'].values)
    wide = tp.stats.offset_stats_plot(tp.stats.offset_stats_unique(both, group_col='rnai'), group_col='rnai')
    assert len(wide) == sess_df['fly_id'].nunique()
    assert {'offset_var_dark', 'offset_var_closed_loop', 'offset_mag_dark', 'offset_mag_closed_loop', 'rnai'} <= set(wide.columns)


def test_rho_stats_bins_and_outer_ring(sess_df, load_row):
    df = tp.stats.rho_stats(sess_df, load_row, BINS, outer_ring=True)
    assert {'rho_dig', 'F_dig'} <= set(df.columns)
    assert all(len(r) == len(BINS) for r in df['rho_dig'])
    assert 'F_dig' not in tp.stats.rho_stats(sess_df, load_row, BINS).columns


def test_dphi_stats_bins(sess_df, load_row):
    df = tp.stats.dphi_stats(sess_df, load_row, SIGNED_BINS)
    assert all(len(r) == len(SIGNED_BINS) for r in df['dphi_dig'])


def test_reformat_drops_nan_bins_and_reports_bin_edge():
    grouped = pd.DataFrame({'fly_id': ['a', 'b'], 'cl': [1, 0],
                            'rho_dig': [np.array([1., np.nan, 3.]), np.array([np.nan, 5., 6.])]})
    long = tp.stats.reformat_binned_stats_for_reg(grouped, np.array([0., 1., 2.]), {'rho': 'rho_dig'},
                                                  meta_cols=('fly_id', 'cl'))
    assert list(long.columns) == ['fly_id', 'cl', 'rho', 'dh']
    assert long[['fly_id', 'rho', 'dh']].values.tolist() == [['a', 1., 0.], ['a', 3., 2.], ['b', 5., 1.], ['b', 6., 2.]]


def test_reformat_of_empty_input_keeps_columns():
    grouped = pd.DataFrame({'fly_id': ['a'], 'rho_dig': [np.array([np.nan, np.nan])]})
    assert list(tp.stats.reformat_binned_stats_for_reg(grouped, np.array([0., 1.]), {'rho': 'rho_dig'}).columns) \
        == ['fly_id', 'rho', 'dh']


def test_group_rho_stats_averages_sessions_per_fly_and_condition():
    rs = pd.DataFrame({'fly_id': ['a', 'a', 'b'], 'cl': [1, 1, 1],
                       'rho_dig': [np.array([1., 2.]), np.array([3., np.nan]), np.array([5., 6.])],
                       'F_dig': [np.array([1., 1.]), np.array([3., 3.]), np.array([0., 0.])]})
    g = tp.stats.group_rho_stats(rs).set_index('fly_id')
    np.testing.assert_allclose(g.loc['a', 'rho_dig'], [2., 2.])      # nan-aware mean
    np.testing.assert_allclose(g.loc['a', 'F_dig'], [2., 2.])


@pytest.mark.parametrize('module, group', [(tp.tbh_rnai, 'rnai'), (tp.oa_receptor_ko, 'condition')])
def test_experiment_wrappers_carry_group_column(module, group, sess_df, load_row):
    offsets = module.offset_stats(sess_df, load_row)
    assert group in offsets.columns and 'dark' in offsets.columns
    assert group in module.offset_stats_unique(offsets).columns
    rho = module.rho_stats(sess_df, load_row, BINS)
    reg = module.reformat_rho_stats_for_reg(rho, BINS)
    assert list(reg.columns) == ['fly_id', group, 'dark', 'rho', 'dh']
    dphi = module.dphi_stats(sess_df, load_row, SIGNED_BINS)
    assert list(module.reformat_dphi_stats_for_reg(dphi, SIGNED_BINS).columns) == ['fly_id', group, 'dark', 'dphi', 'dh']


def test_tbh_requires_more_than_one_closed_loop_block_but_oa_does_not(sess_df, load_row):
    assert tp.tbh_rnai.MIN_CL == 2 and tp.oa_receptor_ko.MIN_CL == 1


def test_epg_silencing_rho_stats_columns(sess_df, load_row):
    df = tp.epg_silencing.rho_stats(sess_df, load_row, BINS)
    assert list(df.columns) == ['fly_id', 'scan_type', 'condition', 'rho_dig', 'F_dig']
