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
    assert list(offsets.columns) == ['fly_id', 'closed_loop', 'offset_var', 'offset_mean', 'offset_mag']
    assert offsets['offset_mag'].between(0, 1).all()
    assert offsets['offset_mean'].abs().le(np.pi).all()


def test_offset_stats_meta_cols_are_copied_in_order_under_their_own_names(sess_df, load_row):
    df = tp.stats.offset_stats(sess_df, load_row, meta_cols=('rnai', 'fly_id'))
    assert list(df.columns)[:2] == ['rnai', 'fly_id']


def test_offset_stats_unique_aggregates_closed_loop_and_dark(offsets, sess_df):
    both = offsets.assign(dark=sess_df['dark'].values)
    uniq = tp.stats.offset_stats_unique(both)
    assert len(uniq) == 2 * sess_df['fly_id'].nunique()
    assert set(uniq['closed_loop']) == {0, 1} and set(uniq['dark']) == {0, 1}
    cl = uniq[(uniq['fly_id'] == 'fly0') & (uniq['closed_loop'] == 1)].iloc[0]
    expected = both[(both['fly_id'] == 'fly0') & (both['closed_loop'] >= 1)]['offset_var'].mean()
    assert np.isclose(cl['offset_var'], expected)
    assert np.isclose(cl['offset_var_logit'], np.log(expected / (1 - expected)))


def test_offset_stats_unique_min_cl_and_missing_condition(offsets, sess_df):
    both = offsets.assign(dark=sess_df['dark'].values)
    strict = tp.stats.offset_stats_unique(both, min_cl=2)
    cl = strict[(strict['fly_id'] == 'fly0') & (strict['closed_loop'] == 1)].iloc[0]
    assert np.isclose(cl['offset_var'], both[(both['fly_id'] == 'fly0') & (both['closed_loop'] == 2)]['offset_var'].iloc[0])
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
    grouped = pd.DataFrame({'fly_id': ['a', 'b'], 'closed_loop': [1, 0],
                            'rho_dig': [np.array([1., np.nan, 3.]), np.array([np.nan, 5., 6.])]})
    long = tp.stats.reformat_binned_stats_for_reg(grouped, np.array([0., 1., 2.]), {'rho': 'rho_dig'},
                                                  meta_cols=('fly_id', 'closed_loop'))
    assert list(long.columns) == ['fly_id', 'closed_loop', 'rho', 'dh']
    assert long[['fly_id', 'rho', 'dh']].values.tolist() == [['a', 1., 0.], ['a', 3., 2.], ['b', 5., 1.], ['b', 6., 2.]]


def test_reformat_of_empty_input_keeps_columns():
    grouped = pd.DataFrame({'fly_id': ['a'], 'rho_dig': [np.array([np.nan, np.nan])]})
    assert list(tp.stats.reformat_binned_stats_for_reg(grouped, np.array([0., 1.]), {'rho': 'rho_dig'}).columns) \
        == ['fly_id', 'rho', 'dh']


def test_group_rho_stats_averages_sessions_per_fly_and_condition():
    rs = pd.DataFrame({'fly_id': ['a', 'a', 'b'], 'closed_loop': [1, 1, 1],
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


# --- multi-channel offset statistics (channels are the first dimension of ts.offset_c) -------------
@pytest.fixture
def two_channel(sess_df, load_row):
    return tp.stats.offset_stats(sess_df, load_row, channels=[0, 1])


def test_offset_stats_multichannel_gives_one_row_per_session_per_channel(two_channel, sess_df):
    assert len(two_channel) == 2 * len(sess_df)
    assert list(two_channel.columns) == ['fly_id', 'closed_loop', 'channel', 'offset_var', 'offset_mean', 'offset_mag']
    assert set(two_channel['channel']) == {0, 1}
    # sessions stay in order, channels alternate within a session
    assert two_channel['channel'].tolist()[:4] == [0, 1, 0, 1]


def test_offset_stats_multichannel_matches_single_channel_runs(two_channel, sess_df, load_row):
    for ch in (0, 1):
        single = tp.stats.offset_stats(sess_df, load_row, channels=ch)
        assert 'channel' not in single.columns
        got = two_channel[two_channel['channel'] == ch].reset_index(drop=True)
        for col in ('offset_var', 'offset_mean', 'offset_mag'):
            np.testing.assert_allclose(got[col], single[col], atol=1e-12)


def test_offset_stats_single_channel_has_no_channel_column(offsets):
    assert 'channel' not in offsets.columns


def test_offset_stats_unique_and_plot_work_per_channel(two_channel, sess_df):
    both = two_channel.assign(dark=np.repeat(sess_df['dark'].values, 2))
    uniq = tp.stats.offset_stats_unique(both)
    assert len(uniq) == 2 * 2 * sess_df['fly_id'].nunique()          # fly x channel x (closed loop, dark)
    assert set(uniq['channel']) == {0, 1}
    ch1 = uniq[(uniq['fly_id'] == 'fly0') & (uniq['channel'] == 1) & (uniq['closed_loop'] == 1)].iloc[0]
    expected = both[(both['fly_id'] == 'fly0') & (both['channel'] == 1) & (both['closed_loop'] >= 1)]['offset_var'].mean()
    assert np.isclose(ch1['offset_var'], expected)
    wide = tp.stats.offset_stats_plot(uniq)
    assert len(wide) == 2 * sess_df['fly_id'].nunique() and set(wide['channel']) == {0, 1}
