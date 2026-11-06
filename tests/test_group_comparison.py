import numpy as np
import pytest
import scipy as sp

import el_hd_plasticity.two_photon as tp
from el_hd_plasticity.two_photon.group_comparison import Group

gc = tp.group_comparison
GROUPS = [Group('a', 'group a'), Group('b', 'group b', 'green', '--', '//')]
BINS = np.linspace(0, 3, num=6)
SIGNED_BINS = np.linspace(-2, 2, num=7)


@pytest.fixture
def offset_unique(manager):
    return gc.compute_offset_stats(manager, 'condition')


def test_offset_stats_has_closed_loop_and_dark_row_per_fly_with_variance(offset_unique, sess_df):
    assert len(offset_unique) == 2 * sess_df['fly_id'].nunique()
    assert {'condition', 'offset_mag', 'offset_var', 'offset_var_logit'} <= set(offset_unique.columns)
    assert np.allclose(offset_unique['offset_var'], 1 - offset_unique['offset_mag'])


def test_offset_figures_and_tests(offset_unique, tmp_path):
    gc.plot_offset_mag(tmp_path, offset_unique, 'condition', GROUPS)
    gc.plot_offset_polar(tmp_path, offset_unique, 'condition', GROUPS)
    tables = gc.offset_var_tests(tmp_path, offset_unique, 'condition')
    assert set(tables) == {'offset_var_mixed_anova', 'offset_var_pairwise_dark_effect', 'offset_var_pairwise_group_effect'}
    assert {'offset_mag_summary.pdf', 'offset_hist_summary.pdf', 'offset_var_logit_qqplot.pdf'} <= {p.name for p in tmp_path.iterdir()}
    assert list(tables['offset_var_mixed_anova']['Source']) == ['condition', 'dark', 'Interaction']


def test_binned_stats_merge_dark_sessions_and_optionally_drop_first_closed_loop(manager):
    kept = gc.compute_binned_stats(manager, 'condition', 'rho', BINS, exclude_first_cl=False)
    dropped = gc.compute_binned_stats(manager, 'condition', 'rho', BINS, exclude_first_cl=True)
    for grouped in (kept, dropped):
        assert set(grouped['dark']) == {0, 1}
        assert all(len(r) == len(BINS) for r in grouped['rho_dig'])
        assert len(grouped) == 2 * manager.sess_df['fly_id'].nunique()        # one closed-loop and one dark row per fly
    # dark rows are unaffected; closed-loop rows average only the second block when the first one is dropped
    key = ['fly_id', 'dark']
    a, b = (g.sort_values(key).reset_index(drop=True) for g in (kept, dropped))
    assert all(np.allclose(x, y, equal_nan=True) for x, y in zip(a.loc[a['dark'] == 1, 'rho_dig'], b.loc[b['dark'] == 1, 'rho_dig'], strict=True))
    assert not all(np.allclose(x, y, equal_nan=True) for x, y in zip(a.loc[a['dark'] == 0, 'rho_dig'], b.loc[b['dark'] == 0, 'rho_dig'], strict=True))


def test_binned_plot_and_mean_tests(manager, tmp_path):
    grouped = gc.compute_binned_stats(manager, 'condition', 'rho', BINS)
    gc.plot_binned_vs_speed(tmp_path, grouped, 'condition', GROUPS, 'rho', BINS, 'rho.pdf', 'PV magnitude')
    assert (tmp_path / 'rho.pdf').exists()
    tests = gc.mean_over_bins_tests(grouped, 'condition', GROUPS)
    assert list(tests['condition']) == ['closed loop', 'dark'] and set(tests['test']) == {'Mann-Whitney U'}
    arrays = gc.group_arrays(grouped, 'condition', 'rho_dig', GROUPS)
    expected = sp.stats.mannwhitneyu(np.nanmean(arrays['a'][0], axis=-1), np.nanmean(arrays['b'][0], axis=-1)).pvalue
    assert np.isclose(tests['p'].iloc[0], expected)


def test_dphi_regression_tests_cover_both_conditions(manager):
    grouped = gc.compute_binned_stats(manager, 'condition', 'dphi', SIGNED_BINS)
    out = gc.dphi_regression_tests(grouped, 'condition', GROUPS, 'a', SIGNED_BINS)
    wald = out['dphi_wald_tests']
    # one slope test and one intercept test, each spanning both conditions rather than one per condition
    assert len(wald) == len(gc.DPHI_TERM_FAMILIES) == 2
    assert list(wald['df']) == [2, 2]              # one non-reference group x (closed-loop term, dark term)
    assert 'condition' not in wald.columns
    group_term = 'C(condition, Treatment("a"))[T.b]'       # the one non-reference group
    assert list(out['dphi_coefficients']['term']) == [f'dh:{group_term}', f'dh:{group_term}:C(dark)[T.1]']
    assert 'p_holm' in out['dphi_coefficients']
    # one model over both conditions, with the condition in it
    assert 'C(dark)' in out['dphi_mixedlm'] and 'Group Var' in out['dphi_mixedlm']


def test_widths_and_tests(manager, tmp_path):
    widths = gc.compute_widths(manager, 'condition', closed_loop_blocks=(1, 2))
    assert {'cl_width', 'dark_width'} <= set(widths.columns) and len(widths) == manager.sess_df['fly_id'].nunique()
    gc.plot_widths(tmp_path, widths, 'condition', GROUPS)
    gc.plot_width_change(tmp_path, widths, 'condition', GROUPS)
    assert {'bump_width_dark_v_closed_loop.pdf', 'bump_width_change.pdf'} <= {p.name for p in tmp_path.iterdir()}
    np.testing.assert_allclose(gc.width_change(widths), widths['cl_width'] - widths['dark_width'])
    tests = gc.width_tests(widths, 'condition', GROUPS)
    assert list(tests['family']) == ['closed loop vs dark'] * 2 + ['closed loop, between groups',
                                                                  'change (closed loop - dark), between groups']
    assert tests['p_holm'].notna().all()


def test_session_figures_need_both_a_closed_loop_and_a_dark_session(manager, tmp_path):
    plotted = gc.plot_session_figures(tmp_path, manager, min_closed_loop=1)
    assert len(plotted) == manager.sess_df['fly_id'].nunique()
    assert all((tmp_path / 'sessions' / fly / 'heatmaps.pdf').exists() for fly in plotted)
