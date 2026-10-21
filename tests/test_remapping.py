import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from el_hd_plasticity.two_photon import remapping as rm


def scan_table():
    """Two standard flies and one no-visual-cue fly, two baseline sessions for the first."""
    rows = []

    def add(fly, scan, offset, vec=.5, var=.2, dark=0.):
        rows.append({'fly_id': fly, 'scan_type': scan, 'mean offset': offset, 'mean vec. len': vec,
                     'offset var.': var, 'post_dark': dark})
    add('f1', 'baseline', 0.2)
    add('f1', 'baseline', 0.4)
    add('f1', 'post_0deg', 0.1)
    add('f1', 'post_180deg', 3.0)
    add('f2', 'baseline', -1.0)
    add('f2', 'post_0deg', -0.5)
    add('f2', 'post_180deg', -3.0, dark=1.)
    add('n1_novis', 'baseline', 0.0)
    add('n1_novis', 'post_novis1', 1.0)
    add('n1_novis', 'post_novis2', 2.0)
    return pd.DataFrame(rows)


def test_build_sum_stats_averages_sessions_per_scan_type():
    ss = rm.build_sum_stats(scan_table(), ['f1', 'f2'])
    assert list(ss.columns)[:4] == ['fly_id', 'baseline mean offset', 'baseline vec. len', 'baseline offset var.']
    assert np.isclose(ss.loc[0, 'baseline mean offset'], 0.3)          # mean of 0.2 and 0.4
    assert np.isclose(ss.loc[1, 'post_180deg mean offset'], -3.0)


def test_build_sum_stats_autodetects_novis_flies():
    ss = rm.build_sum_stats(scan_table(), ['n1_novis'])
    assert 'post_novis1 mean offset' in ss.columns and 'post_0deg mean offset' not in ss.columns


def test_build_sum_stats_all_scan_types_fills_nan_and_adds_conditions():
    ss = rm.build_sum_stats(scan_table(), ['f1', 'n1_novis'], conditions=['a', 'b'], scan_types=rm.ALL_SCAN_TYPES)
    assert list(ss['condition']) == ['a', 'b']
    assert np.isnan(ss.loc[0, 'post_novis1 mean offset']) and np.isnan(ss.loc[1, 'post_0deg mean offset'])


def test_build_sum_stats_extra_metrics():
    ss = rm.build_sum_stats(scan_table(), ['f2'], extra_metrics=('post_dark',))
    assert ss.loc[0, 'post_180deg post_dark'] == 1.0


def test_avg_diff_zero_for_perfect_remapping_and_pi_over_two_for_chance():
    perfect = pd.DataFrame({'post_0deg mean offset': [0.0], 'post_180deg mean offset': [np.pi]})
    assert rm.avg_diff(perfect).iloc[0] == pytest.approx(0)
    # offsets that do not move after either cue are off by 0 and by pi: average pi/2
    chance = pd.DataFrame({'post_0deg mean offset': [0.0], 'post_180deg mean offset': [0.0]})
    assert rm.avg_diff(chance).iloc[0] == pytest.approx(np.pi / 2)


def test_avg_diff_error_is_circular():
    ss = pd.DataFrame({'post_0deg mean offset': [-np.pi + .1], 'post_180deg mean offset': [-np.pi + .1]})
    # -pi+0.1 is 0.1 away from pi on the circle, and (pi - 0.1) away from 0
    assert rm.avg_diff(ss).iloc[0] == pytest.approx((np.pi - .1 + .1) / 2)


def test_avg_diff_explicit_novis_flag_overrides_column_detection():
    ss = rm.build_sum_stats(scan_table(), ['f1', 'n1_novis'], scan_types=rm.ALL_SCAN_TYPES)
    std, novis = rm.avg_diff(ss, novis=False), rm.avg_diff(ss, novis=True)
    assert np.isfinite(std.iloc[0]) and np.isnan(std.iloc[1])
    assert np.isnan(novis.iloc[0]) and np.isfinite(novis.iloc[1])


def test_plot_min_dist_unwraps_post_cue_offset_toward_baseline():
    ss = pd.DataFrame({'baseline mean offset': [3.0], 'post_0deg mean offset': [0.0],
                       'post_180deg mean offset': [-3.0]})
    _, ax = plt.subplots()
    rm.plot_min_dist(ss, ax, zero=True)
    ys = ax.lines[0].get_ydata()
    assert ys[0] == 3.0
    assert ys[1] == pytest.approx(0.0)                 # 0 is already nearest to 3.0 among {0, 2pi, -2pi}
    assert ys[2] == pytest.approx(-3.0 + 2 * np.pi)    # -3 is brought next to the baseline at +3


def test_get_mu_sem_shapes():
    d = {'c': {'s': np.random.default_rng(0).normal(size=(5, 16, 8, 5))}}
    mu, sem = rm.get_mu_sem(d, 'c', 's')
    assert mu.shape == sem.shape == (16,)


def test_plot_row_sum_stats_draws_each_fly_and_the_mean(capsys):
    ss = rm.build_sum_stats(scan_table(), ['f1', 'f2'], conditions=['x', 'x'])
    _, ax = plt.subplots(2, 2, subplot_kw={'projection': 'polar'})
    rm.plot_row_sum_stats(ss, ax, 1, 'x')
    assert len(ax[1, 0].lines) == 2 + 1              # two flies + mean vector
    assert len(ax[1, 1].lines) == 2 * 2 + 2          # two flies x two scans + two mean vectors
    assert 'remap 0' in capsys.readouterr().out


def test_get_opto_resp_shape_and_masking():
    class PP:
        pass
    n, n_rois, vol_dt = 400, 4, 0.5
    trig = np.zeros(n)
    onsets = np.arange(10, 10 + 40 * 8, 8)[:40] * 1.0
    onsets = (np.arange(40) * 9 + 10)
    trig[onsets] = 5000
    pp = PP()
    pp.voltage_recording_aligned = pd.DataFrame({' Opto Trigger': trig})
    pp.metadata = {'frame_times': [[0.0], [vol_dt]], 'mark_points': {'duration': 1000}}
    pp.timeseries = {'rois': np.ones((1, n_rois, n))}
    pp.calculate_zscored_F = lambda *a, **k: pp.timeseries['rois'].copy()
    resp = rm.get_opto_resp(pp)
    assert resp.shape == (n_rois, 8, 5)
    assert np.isnan(pp.timeseries['rois'][:, :, :int(2 / vol_dt)]).all()      # documented in-place masking


# --- experiment-level analyses --------------------------------------------------------------------
from conftest import FakeManager  # noqa: E402

from el_hd_plasticity.two_photon.group_comparison import Group  # noqa: E402

GROUPS = [Group('remap', 'remap', 'green'), Group('ctrl', 'control', 'black'), Group('novis_control', 'no vis', 'grey')]


@pytest.fixture
def remap_manager(sess_df, load_row):
    """Three flies per group, each with baseline / post_0deg / post_180deg (or novis) sessions."""
    rows = []
    for c, cond in enumerate(('remap', 'ctrl', 'novis_control')):
        for f in range(3):
            scans = ('baseline', 'post_novis1', 'post_novis2') if cond == 'novis_control' else rm.STANDARD_SCAN_TYPES
            for s, scan in enumerate(scans):
                rows.append({'fly_id': f'{cond}_fly{f}', 'condition': cond, 'scan_type': scan,
                             'session': f'{scan}-00{s}', 'seed': 100 * c + 10 * f + s})
    import pandas as pd
    return FakeManager(pd.DataFrame(rows), load_row)


def test_session_offset_summary_names_columns_for_build_sum_stats(remap_manager):
    summary = rm.session_offset_summary(remap_manager)
    assert list(summary.columns) == ['fly_id', 'condition', 'scan_type', 'mean offset', 'mean vec. len', 'offset var.']
    assert np.allclose(summary['offset var.'], 1 - summary['mean vec. len'])
    sum_stats = rm.sum_stats_by_group(summary, GROUPS)
    assert list(sum_stats) == ['remap', 'ctrl', 'novis_control'] and all(len(df) == 3 for df in sum_stats.values())
    assert 'post_novis1 mean offset' in sum_stats['novis_control'].columns
    assert 'post_0deg mean offset' in sum_stats['remap'].columns


def test_opto_responses_are_averaged_per_fly_combined_and_aligned(remap_manager, monkeypatch):
    import pandas as pd

    manager_df = remap_manager.sess_df.copy()
    manager_df['scan_type'] = manager_df['scan_type'].replace(
        {'post_0deg': 'remap_0deg', 'post_180deg': 'remap_180deg', 'post_novis1': 'remap_novis'})
    # a second remap_0deg session for the first remap fly: the two are averaged
    dup = manager_df[(manager_df['condition'] == 'remap') & (manager_df['scan_type'] == 'remap_0deg')].iloc[:1].copy()
    dup['session'], dup['seed'] = 'remap_0deg-009', dup['seed'] + 1000
    # the 'loaded session' is the row's seed, and the mocked opto response is a constant array equal to it
    manager = FakeManager(pd.concat([manager_df, dup], ignore_index=True), lambda row: row['seed'])
    monkeypatch.setattr(rm, 'get_opto_resp', lambda seed, zscore=False: np.full((16, 8, 5), float(seed)))

    opto = rm.compute_opto_responses(manager, GROUPS)
    seeds0 = manager_df[(manager_df['condition'] == 'remap') & (manager_df['scan_type'] == 'remap_0deg')]['seed'].to_numpy(float)
    seeds180 = manager_df[(manager_df['condition'] == 'remap') & (manager_df['scan_type'] == 'remap_180deg')]['seed'].to_numpy(float)
    assert opto['remap']['remap_0deg'].shape == (3, 16, 8, 5)
    assert opto['remap']['remap_0deg'][0, 0, 0, 0] == pytest.approx((seeds0[0] + seeds0[0] + 1000) / 2)   # averaged per fly
    assert opto['remap']['remap_0deg'][1, 0, 0, 0] == seeds0[1]
    assert opto['remap']['combined'][1, 0, 0, 0] == pytest.approx((seeds0[1] + seeds180[1]) / 2)
    assert opto['novis_control']['combined'] is opto['novis_control']['remap_novis']

    rng = np.random.default_rng(0)
    opto['remap']['remap_0deg'] = rng.normal(size=(3, 16, 8, 5))
    aligned = rm.align_opto_responses(opto)
    i = 3
    assert np.allclose(aligned['remap']['remap_0deg'][:, :, i], np.roll(opto['remap']['remap_0deg'][:, :, i], 2 * i - 8, axis=1))


def test_aligned_window_tests_and_figures(tmp_path):
    rng = np.random.default_rng(1)
    peak = np.zeros((6, 16, 8, 5))
    peak[:, 7:10] = 1
    aligned = {'remap': {s: peak + rng.normal(0, .1, peak.shape) for s in ('remap_0deg', 'remap_180deg', 'combined')},
               'ctrl': {s: rng.normal(0, .1, peak.shape) for s in ('remap_0deg', 'remap_180deg', 'combined')},
               'novis_control': {'remap_novis': peak, 'combined': peak}}
    tests = rm.aligned_window_tests(aligned, GROUPS, 'remap')
    assert list(tests['comparison']) == ['remap vs control', 'remap vs no vis'] and tests['p_holm'].notna().all()
    rm.plot_aligned_opto(tmp_path, aligned, GROUPS)
    rm.plot_opto_heatmaps(tmp_path, aligned, GROUPS)
    rm.plot_opto_by_location(tmp_path, aligned, GROUPS[:2])
    assert {'aligned_opto_resp.pdf', 'opto_resp_heatmaps.pdf', 'opto_resp_by_location.pdf'} <= {p.name for p in tmp_path.iterdir()}


def test_offset_summary_figures_and_error_tests(remap_manager, tmp_path):
    summary = rm.session_offset_summary(remap_manager)
    sum_stats = rm.sum_stats_by_group(summary, GROUPS)
    rm.plot_vec_len_comparison(tmp_path, summary, GROUPS)
    rm.plot_polar_summary(tmp_path, sum_stats, GROUPS)
    rm.plot_min_dist_summary(tmp_path, sum_stats, GROUPS, zero_groups=('ctrl',))
    _, tests = rm.plot_remap_error(tmp_path, sum_stats, GROUPS, 'remap')
    assert list(tests['family']) == ['error vs chance'] * 3 + ['error between groups'] * 2
    assert tests['p_holm'].notna().all()
    table = rm.mean_offsets(sum_stats)
    assert set(table['group']) == {'remap', 'ctrl', 'novis_control'} and len(table) == 9
    assert len(rm.plot_session_figures(tmp_path, remap_manager, GROUPS[:2])) == 6
