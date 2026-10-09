import numpy as np
import pytest
from conftest import CHANNELS, make_trial_dict

import el_hd_plasticity.two_photon as tp

di = tp.dual_imaging
BINS = np.array([0, .1, .2, .5, 1, 2, 4])


def test_offset_stats_columns_and_values(sess_df, load_row):
    df = di.offset_stats(sess_df, load_row)
    assert len(df) == len(sess_df)
    assert {'offset_ch1', 'offset_ch2', 'offset_diff', 'abs_offset_diff', 'pva_diff', 'fwhm_ch1', 'fwhm_ch2'} <= set(df.columns)
    assert df['abs_offset_diff'].between(0, 2 * np.pi).all()


def test_offset_stats_unique_requires_both_conditions_unless_cl_only(sess_df, load_row):
    df = di.offset_stats(sess_df, load_row)
    both = di.offset_stats_unique(df)
    assert len(both) == 2 * sess_df['fly_id'].nunique()
    closed_only = df[df['cl'] >= 1]
    assert di.offset_stats_unique(closed_only).empty
    kept = di.offset_stats_unique(closed_only, cl_only=True)
    assert len(kept) == sess_df['fly_id'].nunique() and (kept['cl'] == 1).all()


@pytest.mark.parametrize('use_abs', [True, False])
def test_rho_stats_structure(sess_df, load_row, use_abs):
    df = di.rho_stats(sess_df, load_row, BINS, abs=use_abs)
    for col in ('rho1_dig', 'rho2_dig', 'pva_diff'):
        assert all(len(r) == len(BINS) for r in df[col])


def test_pvdiff_rho_stats_and_cross_corr(sess_df, load_row):
    pv = di.pvdiff_rho_stats(sess_df.iloc[:4], load_row, np.linspace(0, 1, 6))
    assert all(len(r) == 6 for r in pv['pva_diff'])
    cc = di.cross_corr_stats(sess_df.iloc[:2], load_row)
    assert all(len(r) == 101 for r in cc['r'])
    cc_short = di.cross_corr_stats(sess_df.iloc[:2], load_row, delays=np.arange(-5, 6))
    assert all(len(r) == 11 for r in cc_short['r'])      # explicit arguments are honoured


def test_dte_variants():
    data = {'C:\\x\\flyA': {'t1': make_trial_dict(1), 't2': make_trial_dict(2)}}
    lookup = {k: CHANNELS for k in data}
    stats = di.offset_stats_DTE(data, lookup)
    assert list(stats['fly_id'].unique()) == ['flyA'] and len(stats) == 2
    rho = di.rho_stats_DTE(data, lookup, BINS, abs=False)
    assert len(rho) == 2 and rho['fly_id'].iloc[0] == 'C:\\x\\flyA'
