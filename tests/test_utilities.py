import numpy as np
import pandas as pd

from el_hd_plasticity.two_photon import utilities as u


def test_get_time_tick_indices_picks_nearest_sample():
    time = pd.Series(np.arange(0, 10, 0.5))
    assert u.get_time_tick_indices(time, [0, 1, 4.9]) == [0, 2, 10]


def test_time_window_mask_is_inclusive_and_defaults_to_all():
    t = pd.Series(np.arange(10.0))
    assert u.time_window_mask(t, (2, 4)).sum() == 3
    assert u.time_window_mask(t).all()


def test_wrap_angle_range_and_equivalence():
    a = np.array([-7.0, -3.5, 0, 3.5, 7.0])
    w = u.wrap_angle(a)
    assert np.all(np.abs(w) <= np.pi + 1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a))
    np.testing.assert_allclose(np.sin(w), np.sin(a))


def test_heading_to_roi_coords_endpoints_and_center():
    np.testing.assert_allclose(u.heading_to_roi_coords(np.array([-np.pi, 0, np.pi])), [0, 7.5, 15])


def test_heading_to_roi_index_is_centered_on_roi_centers():
    n = 16
    centers = -np.pi + (np.arange(n) + .5) * 2 * np.pi / n
    np.testing.assert_allclose(u.heading_to_roi_index(centers, n), np.arange(n), atol=1e-12)
    assert u.heading_to_roi_index(0.0, n) == 7.5


def test_nearest_equivalent_angle_unwraps_toward_reference():
    assert np.isclose(u.nearest_equivalent_angle(3.0, -3.0), 3.0 - 2 * np.pi)
    assert u.nearest_equivalent_angle(0.5, 0.0) == 0.5


def test_binned_mean_empty_bins_are_nan():
    out = u.binned_mean([0, 1, 1, 2, 5.0], [0, 1, 1, 3, 3], 5)
    assert out[0] == 0 and out[1] == 1 and out[3] == 3.5
    assert np.isnan(out[2]) and np.isnan(out[4])


def test_normalized_histogram_sums_to_one():
    hist, centers = u.normalized_histogram(np.array([0, 0, 1, 1.5]), np.array([0, 1, 2.0]))
    np.testing.assert_allclose(hist, [.5, .5])
    np.testing.assert_allclose(centers, [.5, 1.5])


def test_print_nested_keys(capsys):
    u.print_nested_keys({'a': {'b': {'c': 1}}, 'd': 2})
    assert capsys.readouterr().out.splitlines() == ['a', '  b', '    c', 'd']


def test_mean_sem_ignores_nan_bins_and_jitter_is_per_fly():
    x = np.array([[1., np.nan], [3., np.nan], [5., 2.]])
    mu, sem = u.mean_sem(x)
    assert mu[0] == 3 and mu[1] == 2 and np.isclose(sem[0], np.std([1, 3, 5], ddof=1) / np.sqrt(3))
    jitter = u.fly_jitter(['a', 'b', 'a'], width=.1)
    assert set(jitter) == {'a', 'b'} and all(abs(v) <= .1 for v in jitter.values())
    assert jitter == u.fly_jitter(['a', 'b'], width=.1)


def test_save_tables_writes_csv_and_text_and_stringifies_list_cells(tmp_path):
    import pandas as pd
    tables = {'t': pd.DataFrame({'a': [1, 2], 'cells': [[1, 2], [3]]}), 'summary': 'text'}
    assert u.save_tables(tables, tmp_path) is tables
    assert (tmp_path / 'stats' / 'summary.txt').read_text() == 'text'
    assert pd.read_csv(tmp_path / 'stats' / 't.csv')['cells'].tolist() == ['[1, 2]', '[3]']
