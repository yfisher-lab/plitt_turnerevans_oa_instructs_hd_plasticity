"""Connectomics helpers, tested offline: neuPrint fetches are mocked and the EB mesh is replaced by a
simple stand-in coordinate system."""
import pickle

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


# --- EB frame, synapse phase ---------------------------------------------------------------------
def real_eb(origin=(5., 5., 5.)):
    """An :class:`EBCoordinateSystem` with a hand-set basis (no neuPrint fetch)."""
    eb = object.__new__(cn.EBCoordinateSystem)
    eb.orig_origin = np.array(origin)
    eb.eb_basis = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]])
    return eb


def test_to_eb_frame_is_change_of_basis_with_y_and_z_flipped():
    eb = real_eb()
    x = np.random.default_rng(0).uniform(0, 10, (6, 3))
    expected = eb.change_of_basis(x) * np.array([1, -1, -1])
    np.testing.assert_allclose(eb.to_eb_frame(x), expected)
    df = pd.DataFrame(x, columns=['x', 'y', 'z'])
    out = cn.add_eb_coords(df, FakeEB(), type=None)                       # plain x, y, z columns
    np.testing.assert_allclose(out[['x_eb', 'y_eb', 'z_eb']].values, FakeEB().change_of_basis(x) * np.array([1, -1, -1]))


def test_synapse_phase_is_the_angle_the_synapses_span_and_phase_width_their_spread():
    rng = np.random.default_rng(2)
    angles = 2.0 + rng.uniform(-.3, .3, 400)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1) * rng.uniform(1, 2, (400, 1))
    hull, phase = cn.synapse_phase(xy, n=3)
    assert phase == pytest.approx(2.0, abs=.05) and hull.shape[1] == 2
    uniform = np.linspace(-.5, .5, 1001) + 3.0                                # spread across the wrap-around at pi
    assert cn.phase_width(np.angle(np.exp(1j * uniform))) == pytest.approx(.9, abs=.01)


# --- connectivity --------------------------------------------------------------------------------
from el_hd_plasticity.connectomics import connectivity as cv  # noqa: E402


@pytest.fixture
def small_connectome():
    neuron_df = pd.DataFrame({'bodyId': [1, 2, 3, 4, 5], 'type': ['EL', 'EL', 'EPG', 'ER4d', 'ER2_a']})
    conn_df = pd.DataFrame({'bodyId_pre': [1, 2, 3, 3, 4], 'bodyId_post': [4, 4, 1, 5, 3],
                            'type_pre': ['EL', 'EL', 'EPG', 'EPG', 'ER4d'],
                            'type_post': ['ER4d', 'ER4d', 'EL', 'ER2_a', 'EPG'], 'weight': [4, 6, 8, 2, 10]})
    counts = pd.DataFrame({'n_presynapses': [100., 200., 50., 40., 10., 10., 10.],
                           'n_postsynapses': [10., 20., 30., 100., 80., 5., 5.]},
                          index=['EL', 'EPG', 'ER4d', 'ER2_a', 'ER2_b', 'ER2_c', 'ER2_d'])
    return neuron_df, conn_df, counts


def test_relative_weights_divide_by_synapse_counts_and_neurons_of_the_type(small_connectome):
    neuron_df, conn_df, _ = small_connectome
    pre = pd.Series({1: 40., 2: 60., 3: 20., 4: 50.})
    post = pd.Series({1: 100., 3: 50., 4: 10., 5: 20.})
    out = cv.add_relative_weights(conn_df, neuron_df, pre, post)
    first = out.iloc[0]                                  # EL 1 -> ER4d 4, weight 4
    assert first['relative_weight'] == pytest.approx(4 / 10 / 1)              # / post synapses / n ER4d
    assert first['pcnt_output'] == pytest.approx(4 / 40 / 2)                  # / pre synapses / n EL
    assert 'relative_weight' not in conn_df.columns                           # the input is not modified


def test_type_connectivity_merges_er_types_and_keeps_only_the_partner_edges(small_connectome):
    neuron_df, conn_df, counts = small_connectome
    table, n_neurons = cv.type_connectivity(conn_df, neuron_df, counts, merge=('ER4d', 'ER2_a'), merged_name='ER')
    edges = {(r.type_pre, r.type_post): r for r in table.itertuples()}
    assert set(edges) == {('EL', 'ER'), ('EPG', 'ER'), ('EPG', 'EL'), ('ER', 'EPG')}
    assert edges['EL', 'ER'].weight == 10 and edges['EPG', 'ER'].weight == 2
    assert edges['EL', 'ER'].relative_weight == pytest.approx(10 / (counts.loc['ER4d', 'n_postsynapses'] + counts.loc['ER2_a', 'n_postsynapses']))
    assert edges['EPG', 'EL'].pcnt_output == pytest.approx(8 / 200) and edges['EPG', 'EL'].nrn_weight == pytest.approx(8 / 2)
    assert n_neurons['ER'] == 2 and n_neurons['EL'] == 2
    graph = cv.connectivity_graph(table, n_neurons)
    assert graph.nodes['ER']['n_neurons'] == 2 and graph['EL']['ER']['weight'] == 10


def test_count_partners_counts_connections_above_the_weight_threshold(small_connectome):
    _, conn_df, _ = small_connectome
    assert cv.count_partners(conn_df, [4, 1, 9], min_weight=4).tolist() == [1, 1, 0]


# --- synapse distance results --------------------------------------------------------------------
def test_load_distance_table_converts_voxels_to_microns_and_labels_sources(tmp_path):
    for neuron_type, scale in (('ER4d', 1.), ('ER2_a', 2.)):
        results = [{'cable_distance': {'el_post': np.array([125., np.nan]) * scale, 'exr2_post': np.array([250.]) * scale},
                    'euclidean_distance': {}}]
        with open(tmp_path / f'{neuron_type}_syn_distances.pkl', 'wb') as f:
            pickle.dump(results, f)
    table = sd.load_distance_table(tmp_path, ('ER4d', 'ER2_a'), ('el_post', 'exr2_post'))
    assert list(table.columns) == ['er', 'source', 'distances']
    er4d_el = table[(table['er'] == 'ER4d') & (table['source'] == 'el')]['distances'].to_numpy()
    assert er4d_el[0] == pytest.approx(1.0) and np.isnan(er4d_el[1])           # 125 voxels x 8 nm = 1 micron
    assert table[(table['er'] == 'ER2_a') & (table['source'] == 'exr2')]['distances'].iloc[0] == pytest.approx(4.0)


def test_partner_count_table_groups_neurons_by_family():
    ers = pd.DataFrame({'bodyId': [1, 2, 3], 'type': ['ER4d', 'ER2_a', 'ER2_b']})
    conn = pd.DataFrame({'bodyId_post': [1, 1, 2, 3, 3], 'weight': [5, 6, 2, 4, 9]})
    table = sd.partner_count_table({'EL': conn}, ers, groups=(('ER4d',), ('ER2_a', 'ER2_b')), min_weight=3)
    assert table['group'].tolist() == ["('ER4d',)", "('ER2_a', 'ER2_b')", "('ER2_a', 'ER2_b')"]
    assert table['EL'].tolist() == [2, 0, 2]
    assert sd.ER_TYPES[0] == 'ER4d' and set(sd.ER_TYPES) == {t for g in sd.ER_GROUPS for t in g}


# --- figures -------------------------------------------------------------------------------------
def test_connectomics_figures_render(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from el_hd_plasticity.connectomics import plotting as cp
    distances = pd.DataFrame({'er': 'ER4d', 'source': ['el'] * 30 + ['exr2'] * 30,
                              'distances': np.random.default_rng(0).uniform(0, 12, 60)})
    cp.plot_distance_histogram(distances, ('el', 'exr2'), ('green', 'blue'))
    cp.plot_family_distance_histogram(distances, ('ER4d',))
    cp.plot_cumulative_distances(distances, (('ER4d',),))
    table = pd.DataFrame({'group': ["('ER4d',)", "('ER2_a',)"], 'EL': [3, 5]})
    cp.plot_partner_counts(table, 'EL', 'n', (('ER4d',), ('ER2_a',)))
    graph = cv.connectivity_graph(pd.DataFrame({'type_pre': ['A'], 'type_post': ['B'], 'weight': [3000.], 'relative_weight': [.1],
                                                'pcnt_output': [.2], 'nrn_weight': [5.]}), {'A': 2, 'B': 3})
    cp.plot_connectivity_graph(graph, {'A': 'g', 'B': 'k'}, {'A': [0, 0], 'B': [1, 1]})
    fake = type('E', (), {'eb_vertices_xsec': {'radius': np.random.default_rng(1).uniform(1, 2, 50),
                                                 'phase': np.random.default_rng(2).uniform(-3, 3, 50)}})()
    syn = pd.DataFrame({'xsec_x': np.random.default_rng(3).normal(size=80), 'xsec_y': np.random.default_rng(4).normal(size=80)})
    cp.plot_xsec_density([(syn, 'Greys', .6)], fake)
    fig, ax = plt.subplots()
    cp.clean_axes(ax)
    plt.close('all')


# --- datasets and comparison across datasets -----------------------------------------------------
from el_hd_plasticity.connectomics import datasets as ds  # noqa: E402


def make_table_dataset(name='toy', scale=1):
    """EL 1-2, EPG 3-4, ER4d 5, ER2_a 6, PEN_a 7; synapses in the EB (plus some elsewhere that must be ignored)."""
    neurons = pd.DataFrame({'bodyId': [1, 2, 3, 4, 5, 6, 7],
                            'type': ['EL', 'EL', 'EPG', 'EPG', 'ER4d', 'ER2_a', 'PEN_a(PEN1)']})
    pairs = {(3, 1): 5, (4, 1): 4, (4, 2): 3, (1, 5): 6, (2, 5): 4, (1, 6): 4, (3, 5): 2, (7, 5): 3, (5, 3): 4, (6, 4): 3}
    rows = [(pre, post, 'EB') for (pre, post), n in pairs.items() for _ in range(n * scale)]
    rows += [(3, 1, 'PB')] * 20 + [(1, 5, 'LAL')] * 20
    return ds.TableDataset(name, neurons, pd.DataFrame(rows, columns=['pre_id', 'post_id', 'neuropil']))


def test_match_types_uses_full_regular_expression_matches():
    types = pd.Series(['ER2_a', 'ER2_b', 'ER4d', 'EPG', 'ER2'])
    assert ds.match_types(types, ['ER2.*']).tolist() == [True, True, False, False, True]
    assert ds.match_types(types, 'EPG').tolist() == [False, False, False, True, False]


def test_table_dataset_restricts_to_the_region_and_counts_synapses():
    d = make_table_dataset()
    assert d.neurons('ER2.*')['bodyId'].tolist() == [6]
    assert d.neurons(['EL', 'EPG'], role='pre')['bodyId'].tolist() == [1, 2, 3, 4]
    assert d.neurons(['EL', 'EPG'], role='post')['bodyId'].tolist() == [1, 2, 3, 4]
    conn = d.connections([3, 4], [1, 2])
    assert conn.set_index(['bodyId_pre', 'bodyId_post'])['weight'].to_dict() == {(3, 1): 5, (4, 1): 4, (4, 2): 3}  # PB ignored
    assert d.connections([3, 4], [1, 2], min_weight=4)['weight'].tolist() == [5, 4]
    assert d.synapse_counts([1, 2], 'pre').to_dict() == {1: 10, 2: 4}
    assert d.synapse_counts([5], 'post').to_dict() == {5: 15}
    counts = d.type_synapse_counts(d.neurons(['EL', 'EPG']))
    assert counts.loc['EL', 'n_presynapses'] == 14 and counts.loc['EPG', 'n_postsynapses'] == 7


def test_dataset_type_connectivity_and_er_inputs_on_a_table_dataset():
    d = make_table_dataset()
    table, n_neurons = cv.dataset_type_connectivity(d, types=('EL', 'EPG', 'ER4d', 'ER2.*'), min_weight=3)
    edges = {(r.type_pre, r.type_post): r for r in table.itertuples()}
    assert set(edges) == {('EPG', 'EL'), ('EL', 'ER2/4d'), ('ER2/4d', 'EPG')}      # EPG -> ER has only 2 synapses
    assert edges['EL', 'ER2/4d'].weight == 14 and n_neurons['ER2/4d'] == 2 and n_neurons['EL'] == 2
    assert edges['EPG', 'EL'].relative_weight == pytest.approx(12 / 12)         # / the 12 EB postsynapses of the ELs
    assert edges['EL', 'ER2/4d'].pcnt_output == pytest.approx(14 / 14)         # / the 14 EB presynapses of the ELs
    er = cv.er_input_connections(d, input_types=('EL', 'EPG', 'PEN_a.*'), er_types=('ER4d', 'ER2.*'), min_weight=1)
    assert set(er['type_pre']) == {'EL', 'EPG', 'PEN_a(PEN1)'} and set(er['type_post']) == {'ER4d', 'ER2_a'}
    assert er.loc[er['type_pre'] == 'PEN_a(PEN1)', 'weight'].iloc[0] == 3
    matrix = cv.er_input_matrix(er, 'weight', ('EL', 'EPG', 'PEG', 'PEN_a(PEN1)'))
    assert matrix.loc['PEG'].eq(0).all() and matrix.loc['EL', 'ER4d'] == 10


def test_compare_datasets_stacks_tables_and_marginals(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from el_hd_plasticity.connectomics import plotting as cp
    datasets = {'a': make_table_dataset('a'), 'b': make_table_dataset('b', scale=2)}
    table = cv.compare_datasets(datasets, cv.dataset_type_connectivity, types=('EL', 'EPG', 'ER4d', 'ER2.*'), min_weight=3)
    assert set(table['dataset']) == {'a', 'b'}
    weights = table.set_index(['dataset', 'type_pre', 'type_post'])['weight']
    assert weights['b', 'EL', 'ER2/4d'] == 2 * weights['a', 'EL', 'ER2/4d']
    cp.plot_connectivity_comparison(table, 'relative_weight')
    conns = {name: cv.er_input_connections(d, input_types=('EL', 'EPG', 'PEN_a.*'), er_types=('ER4d', 'ER2.*'), min_weight=1)
             for name, d in datasets.items()}
    marginals = cv.er_input_marginals(conns, 'weight', ('EL', 'EPG', 'PEN_a(PEN1)'))
    assert list(marginals.columns) == ['a', 'b'] and (marginals['b'] == 2 * marginals['a']).all()
    cp.plot_input_comparison(marginals)
    plt.close('all')


def test_table_dataset_accepts_an_edge_list_with_synapse_numbers():
    neurons = pd.DataFrame({'bodyId': [1, 2, 3], 'type': ['EL', 'EPG', 'ER4d']})
    edges = pd.DataFrame({'pre': [2, 2, 1, 1], 'post': [1, 1, 3, 3], 'roi': ['EB', 'PB', 'EB', 'EB'], 'n': [5, 9, 2, 3]})
    d = ds.TableDataset('edges', neurons, edges, pre_col='pre', post_col='post', roi_col='roi', weight_col='n')
    assert d.connections([1, 2], [1, 3]).set_index(['bodyId_pre', 'bodyId_post'])['weight'].to_dict() == {(1, 3): 5, (2, 1): 5}
    assert d.synapse_counts([1, 2], 'pre').to_dict() == {1: 5, 2: 5}
    assert d.synapse_counts([1, 3], 'post').to_dict() == {1: 5, 3: 5}


# --- Codex ---------------------------------------------------------------------------------------
import gzip  # noqa: E402
import io  # noqa: E402
import urllib.error  # noqa: E402


class StubCodex(ds.CodexClient):
    """A Codex client that serves in-memory tables (gzipped) instead of downloading them."""

    def __init__(self, tables, cache_dir, token='user@host:secret'):
        super().__init__(token=token, cache_dir=cache_dir)
        self.tables, self.opened = tables, []

    def _open(self, data_product, dataset):
        self.opened.append((data_product, dataset))
        content = self.tables[data_product]
        if isinstance(content, Exception):
            raise content
        return io.BytesIO(content if content[:2] == b'\x1f\x8b' or content[:1] == b'<' else gzip.compress(content))


def test_codex_client_needs_a_token_caches_downloads_and_rejects_web_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(ds.params, 'CODEX_TOKEN', None)
    with pytest.raises(RuntimeError, match='CODEX_API_TOKEN'):
        ds.CodexClient(cache_dir=tmp_path)
    client = StubCodex({'good': b'a,b\n1,2\n', 'bad': b'<html>please log in</html>'}, tmp_path)
    assert client.read('fafb', 'good')['b'].tolist() == [2]
    client.read('fafb', 'good')
    assert client.opened == [('good', 'fafb')]                                       # the second read comes from the cache
    with pytest.raises(RuntimeError, match='gzipped'):
        client.read('fafb', 'bad')
    assert not client.path('fafb', 'bad').exists() and not list(tmp_path.glob('*.part'))
    assert 'download?dataset=fafb' in ds.CodexClient.products_url('fafb')


def test_codex_url_carries_the_raw_token_and_http_errors_do_not_leak_it(tmp_path, monkeypatch):
    seen = []

    def fake_urlopen(url):
        seen.append(url)
        raise urllib.error.HTTPError(url, 403, 'Forbidden', None, None)

    monkeypatch.setattr(ds.urllib.request, 'urlopen', fake_urlopen)
    client = ds.CodexClient(token='user@host:secret', cache_dir=tmp_path)
    with pytest.raises(RuntimeError) as error:
        client.download('banc', 'neurons')
    assert seen == ['https://codex.flywire.ai/api/download_resource?api_token=user@host:secret&data_product=neurons&dataset=banc']
    assert '403' in str(error.value) and 'secret' not in str(error.value)


def test_load_codex_keeps_the_region_restores_id_prefixes_and_maps_types(tmp_path):
    prefix = 720575940 * 10 ** 9
    connections = pd.DataFrame({'pre_root_id_720575940': [1, 2, 2, 5], 'post_root_id_720575940': [2, 1, 1, 1],
                                'neuropil': ['EB', 'EB', 'PB', 'EB'], 'syn_count': [4, 6, 50, 7]})
    cell_types = pd.DataFrame({'root_id_720575940': [1, 2, 3], 'primary_type': ['EL_x', 'EPG_x', None]})
    client = StubCodex({'connections_princeton': connections.to_csv(index=False).encode(),
                        'consolidated_cell_types': cell_types.to_csv(index=False).encode()}, tmp_path)
    d = ds.load_codex('FAFB', 'fafb', client=client, type_map=lambda t: t.str.removesuffix('_x'), chunksize=2)
    assert d.name == 'FAFB' and d.neurons('.*')['type'].tolist() == ['EL', 'EPG']
    assert d.neurons('.*')['bodyId'].tolist() == [prefix + 1, prefix + 2]
    conn = d.connections([prefix + 1, prefix + 2, prefix + 5], [prefix + 1, prefix + 2])
    assert conn.set_index(['bodyId_pre', 'bodyId_post'])['weight'].to_dict() == {
        (prefix + 1, prefix + 2): 4, (prefix + 2, prefix + 1): 6, (prefix + 5, prefix + 1): 7}        # the PB row is dropped
    assert client.opened[0] == ('connections_princeton', 'fafb')


def test_load_codex_uses_the_cell_type_table_of_each_dataset(tmp_path):
    connections = pd.DataFrame({'pre_root_id': [1, 2], 'post_root_id': [2, 1], 'neuropil': ['EB', 'EB'], 'syn_count': [5, 6]})
    neurons = pd.DataFrame({'Root ID': [1, 2], 'Primary Cell Type': ['EPG', 'EL']})
    client = StubCodex({'connections_princeton': connections.to_csv(index=False).encode(),
                        'neurons': neurons.to_csv(index=False).encode()}, tmp_path)
    d = ds.load_codex('BANC', 'banc', client=client)
    assert d.neurons('EL')['bodyId'].tolist() == [2]
    assert d.connections([1], [2])['weight'].tolist() == [5]
    assert [r[0] for r in client.opened] == ['connections_princeton', 'neurons']


def test_ring_outline_follows_the_flipped_eb_frame_and_eb_density_renders():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from el_hd_plasticity.connectomics import plotting as cp
    eb = real_eb()
    eb._radius_min_lookup = lambda phase: 1 + .1 * np.sin(phase)     # radius depends on the (unflipped) phase
    eb._radius_max_lookup = lambda phase: 2 + 0 * phase
    inner, outer = eb.ring_outline(n=9)
    assert inner.shape == outer.shape == (9, 2)
    np.testing.assert_allclose(np.hypot(*outer.T), 2)
    # the point at EB-frame phase p has the radius the lookup gives at -p (y is flipped)
    phase = np.linspace(-np.pi, np.pi, 9)
    np.testing.assert_allclose(np.hypot(*inner.T), 1 + .1 * np.sin(-phase))
    np.testing.assert_allclose(np.arctan2(inner[:, 1], inner[:, 0]), np.angle(np.exp(1j * phase)), atol=1e-12)
    rng = np.random.default_rng(0)
    syn = pd.DataFrame({'x_eb': rng.normal(size=80), 'y_eb': rng.normal(size=80)})
    fig, axs = cp.plot_eb_density([(syn, 'Greys', .6), (syn, 'Purples', .6)], eb, fill=True, share_axes=False)
    assert axs.shape == (1, 2) and axs[0, 0].get_aspect() == 1.0
    plt.close('all')


# --- distances between synapses onto different EPG instances ----------------------------------------
def labeled(x_by_label):
    rows = [{'x': x, 'y': 0., 'z': 0., 'type': label} for label, xs in x_by_label.items() for x in xs]
    return pd.DataFrame(rows)


def test_calc_instance_distance_is_the_cable_distance_to_the_nearest_synapse_of_another_instance(straight_neuron):
    res = sd.calc_instance_distance((1, labeled({'A': [500., 600.], 'B': [900.]})))
    assert list(res['instance']) == ['A', 'A', 'B']
    np.testing.assert_allclose(res['cable_distance'], [400., 300., 300.], atol=1)        # A@500 -> B@900, A@600 -> B@900, B -> A@600


def test_calc_instance_distance_is_nan_beyond_the_cutoff_and_without_another_instance(straight_neuron):
    far = 2 * sd.MAX_CABLE_NM / sd.VOXEL_NM
    res = sd.calc_instance_distance((1, labeled({'A': [0.], 'B': [far]})))
    assert np.isnan(res['cable_distance']).all()                                          # 30 microns apart, cutoff is 15
    only = sd.calc_instance_distance((1, labeled({'A': [0., 300.]})))
    assert np.isnan(only['cable_distance']).all()                                         # a single instance has no partner


def test_run_instance_distances_and_loader(straight_neuron, tmp_path):
    syns = labeled({'A': [500.], 'B': [900.]}).assign(bodyId_pre=[1, 1])
    syns = pd.concat([syns, labeled({'A': [100.], 'B': [200.]}).assign(bodyId_pre=[2, 2])], ignore_index=True)
    results = sd.run_instance_distances(syns, [1, 2], tmp_path / 'ER4d_epg_instance_distances.pkl', processes=1)
    assert len(results) == 2
    table = sd.load_instance_distance_table(tmp_path, ['ER4d'])
    assert list(table.columns) == ['er', 'source', 'distances'] and set(table['source']) == {'epg_instance'}
    np.testing.assert_allclose(sorted(table['distances']), sorted([400, 400, 100, 100]) * np.array(sd.VOXEL_NM) / 1000, atol=.02)


def test_adjacent_instances_are_the_main_instance_and_its_nearest_neighbors():
    import pandas as pd
    syns = pd.DataFrame({'type': ['A'] * 3 + ['B'] * 2 + ['C'] * 2 + ['D'] * 2,
                         'x_eb': [0] * 3 + [10] * 2 + [30] * 2 + [100] * 2, 'y_eb': 0, 'z_eb': 0})
    assert sd.adjacent_instances(syns) == ['A', 'B', 'C']
