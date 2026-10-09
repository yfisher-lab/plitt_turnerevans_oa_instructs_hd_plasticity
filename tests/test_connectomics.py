"""Connectomics helpers, tested offline: neuPrint fetches are mocked and the EB mesh is replaced by a
simple stand-in coordinate system."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('neuprint')
pytest.importorskip('navis')

import neuprint

from el_hd_plasticity.connectomics import connectomics as cn
from el_hd_plasticity.connectomics import synapse_distance as sd


class FakeEB:
    """Same interface as :class:`EBCoordinateSystem` with closed-form transforms."""
    def change_of_basis(self, x):
        return (x - 5.) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]]).T

    def get_circ_coordinates(self, x):
        return cn.EBCoordinateSystem.cart3d_to_polar2d(x)

    def scale_radius(self, phase, radius):
        return (radius - 1) / 3

    def get_xsec_coords(self, phase, radius, z):
        return cn.EBCoordinateSystem.cart3d_to_polar2d(np.stack([np.asarray(radius) - 2, np.asarray(z) - .5], axis=1))

    def scale_xsec_rad(self, phase, xsec_phase, xsec_rad):
        return xsec_rad / 2


def test_cart3d_to_polar2d():
    phase, radius = cn.EBCoordinateSystem.cart3d_to_polar2d(np.array([[1., 1., 9.], [0., -2., 0.]]))
    np.testing.assert_allclose(phase, [np.pi / 4, -np.pi / 2])
    np.testing.assert_allclose(radius, [np.sqrt(2), 2])


def test_add_eb_coords_adds_columns_in_place_for_the_requested_side():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f'{a}_{s}': rng.uniform(0, 20, 30) for a in 'xyz' for s in ('pre', 'post')})
    out = cn.add_eb_coords(df, FakeEB(), type='pre')
    assert out is df
    for col in ('x_eb', 'y_eb', 'z_eb', 'phase', 'radius', 'radius_scaled', 'xsec_phase', 'xsec_radius',
                'xsec_radius_scaled', 'xsec_x', 'xsec_y', 'xsec_x_scaled', 'xsec_y_scaled'):
        assert col in df.columns
    # y and z are flipped relative to the plain change of basis
    basis = FakeEB().change_of_basis(df[['x_pre', 'y_pre', 'z_pre']].values)
    np.testing.assert_allclose(df['x_eb'], basis[:, 0])
    np.testing.assert_allclose(df['y_eb'], -basis[:, 1])
    np.testing.assert_allclose(df['xsec_x'], df['xsec_radius'] * np.cos(df['xsec_phase']))


def test_peeled_convex_hull_removes_outliers():
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.normal(size=(300, 2)), [[50., 50.], [-50., 50.], [0., -50.]]])
    full = cn.peeled_convex_hull(pts, n=0)
    peeled = cn.peeled_convex_hull(pts, n=3)
    assert np.abs(full).max() == 50 and np.abs(peeled).max() < 10


# --- synapse distances ---------------------------------------------------------------------------
def line_skeleton(n=40, step=100.0):
    xyz = np.stack([np.arange(n) * step, np.zeros(n), np.zeros(n)], axis=1)
    return pd.DataFrame({'rowId': np.arange(1, n + 1), 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2],
                         'radius': 5.0, 'link': np.concatenate([[-1], np.arange(1, n)]),
                         'distance': np.concatenate([[0], np.full(n - 1, step)])})


@pytest.fixture
def straight_neuron(monkeypatch):
    monkeypatch.setattr(neuprint, 'fetch_skeleton', lambda *a, **k: line_skeleton())


def synapses(pre_x, post_x, pre='epg_pre', post='el_post'):
    rows = [{'x': x, 'y': 0., 'z': 0., 'type': pre} for x in pre_x] + \
           [{'x': x, 'y': 0., 'z': 0., 'type': post} for x in post_x]
    return pd.DataFrame(rows)


def test_calc_syn_distance_on_a_straight_neurite(straight_neuron):
    # skeleton runs along x in 100-voxel steps; upsampling keeps nodes at least that dense
    res = sd.calc_syn_distance((1, synapses([500.], [800., 2000.])), 'epg_pre', ['el_post'])
    assert res['euclidean_distance']['el_post'][0] == pytest.approx(300.0, abs=1)
    assert res['cable_distance']['el_post'][0] == pytest.approx(300.0, abs=1)   # straight line: cable == euclid


def test_calc_syn_distance_is_nan_beyond_cutoffs_and_per_post_type(straight_neuron):
    far = 2 * sd.MAX_EUCLIDEAN_NM / sd.VOXEL_NM                                  # beyond the 10 um search radius
    res = sd.calc_syn_distance((1, synapses([0.], [far], post='a_post')), 'epg_pre', ['a_post', 'b_post'])
    assert np.isnan(res['cable_distance']['a_post'][0])
    assert np.isnan(res['cable_distance']['b_post'][0])                          # no such synapses at all
    assert set(res) == {'cable_distance', 'euclidean_distance'}


def test_neuron_synapse_args_selects_each_neurons_pre_and_post_sites():
    syns = pd.DataFrame({'type': ['epg_pre', 'epg_pre', 'el_post', 'el_post', 'exr2_post'],
                         'bodyId_pre': [1, 2, 9, 9, 8], 'bodyId_post': [5, 6, 1, 2, 2]})
    (b1, s1), (b2, s2) = sd.neuron_synapse_args(syns, [1, 2], 'epg_pre')
    assert (b1, b2) == (1, 2)
    assert s1.index.tolist() == [0, 2]            # its own pre site + the post site on it
    assert s2.index.tolist() == [1, 3, 4]


def test_label_synapses_sets_type_and_location():
    df = pd.DataFrame({'x_pre': [1.], 'y_pre': [2.], 'z_pre': [3.], 'x_post': [4.], 'y_post': [5.], 'z_post': [6.]})
    sd.label_synapses(df, 'lbl', 'post')
    assert df.loc[0, ['type', 'x', 'y', 'z']].tolist() == ['lbl', 4., 5., 6.]
