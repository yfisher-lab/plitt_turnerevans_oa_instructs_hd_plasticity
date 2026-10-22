"""Connectivity between hemibrain cell types: weights normalized by synapse counts and type-level graphs.

The functions that fetch from neuPrint (``fetch_*``) are thin; the normalization and grouping are pure functions of the
tables they return, so they can be used on any adjacency table with ``bodyId_pre`` / ``bodyId_post``, ``weight`` and the
neuron types.
"""

import navis.interfaces.neuprint as neu
import networkx as nx
import numpy as np
import pandas as pd

from .datasets import match_types

EB = ['EB']
MIN_ROI_WEIGHT = 3
# Types merged into one node of the type-level graph.
MERGED_ER = ('ER4d', 'ER2.*')                 # regular expressions: the hemibrain splits ER2 into ER2_a-d, FAFB has a single ER2
MERGED_ER_NAME = 'ER2/4d'


# ----------------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------------
def fetch_connections(pre_query, post_query, rois=EB, min_roi_weight=MIN_ROI_WEIGHT):
    """Adjacencies between two sets of neurons within `rois`, with the neuron types merged in.

    Args:
        pre_query, post_query: ``NeuronCriteria`` or body IDs of the presynaptic / postsynaptic neurons.
        rois: ROIs the connections are counted in.
        min_roi_weight: Minimum connection weight within the ROIs.

    Returns:
        Tuple of (neuron table, connection table with ``type_pre`` and ``type_post``).
    """
    neuron_df, conn_df = neu.fetch_adjacencies(pre_query, post_query, rois=rois, properties=['type', 'instance', 'bodyID'],
                                               min_roi_weight=min_roi_weight)
    return neuron_df, neu.merge_neuron_properties(neuron_df, conn_df, ['type'])


def fetch_synapse_counts(body_ids, roi='EB', side='pre'):
    """Number of primary ``side`` ('pre' or 'post') synapses in `roi` of each neuron: Series indexed by body ID."""
    syn = neu.fetch_synapses(list(body_ids), neu.SynapseCriteria(rois=[roi], type=side, primary_only=True))
    return syn['bodyId'].value_counts()


def fetch_type_synapse_counts(types, roi='EB'):
    """Number of pre- and postsynapses in `roi` of all neurons of each type (DataFrame indexed by type)."""
    counts = pd.DataFrame(index=list(types), columns=['n_presynapses', 'n_postsynapses'], dtype=float)
    for neuron_type in types:
        for side, col in (('pre', 'n_presynapses'), ('post', 'n_postsynapses')):
            counts.loc[neuron_type, col] = neu.fetch_synapses(
                neu.NeuronCriteria(type=[neuron_type]), neu.SynapseCriteria(type=side, rois=[roi])).shape[0]
    return counts


def strongest_partner(post_id, pre_ids, roi='EB'):
    """Body ID of the neuron in `pre_ids` with the strongest connection (in `roi`) onto neuron `post_id`."""
    _, conn_df = neu.queries.fetch_adjacencies(pre_ids, post_id, rois=roi)
    return conn_df.sort_values('weight', ascending=False)['bodyId_pre'].iloc[0]


# ----------------------------------------------------------------------------
# Normalizing weights
# ----------------------------------------------------------------------------
def add_relative_weights(conn_df, neuron_df, presynapse_counts, postsynapse_counts):
    """Add weights of single connections normalized by the synapse counts of the neurons involved.

    ``relative_weight`` is the weight divided by the number of postsynapses of the postsynaptic neuron and by the number of
    neurons of its type; ``pcnt_output`` the weight divided by the number of presynapses of the presynaptic neuron and by
    the number of neurons of its type.

    Args:
        conn_df: Connections with bodyId_pre, bodyId_post, type_pre, type_post, weight.
        neuron_df: The neurons involved (with ``type``), used to count the neurons of each type.
        presynapse_counts, postsynapse_counts: Synapses per neuron (Series indexed by body ID, see
            :func:`fetch_synapse_counts`).

    Returns:
        `conn_df` with the added ``relative_weight`` and ``pcnt_output`` columns (copy).
    """
    n_per_type = neuron_df['type'].value_counts()
    conn_df = conn_df.copy()
    conn_df['relative_weight'] = (conn_df['weight'] / conn_df['bodyId_post'].map(postsynapse_counts)
                                  / conn_df['type_post'].map(n_per_type))
    conn_df['pcnt_output'] = (conn_df['weight'] / conn_df['bodyId_pre'].map(presynapse_counts)
                              / conn_df['type_pre'].map(n_per_type))
    return conn_df


def type_connectivity(conn_df, neuron_df, type_counts, merge=MERGED_ER, merged_name=MERGED_ER_NAME,
                      partners=('EL', 'EPG')):
    """Connectivity between cell types, with several types merged into one node.

    Weights are summed over the connections of each pair of types and normalized: ``relative_weight`` by the
    postsynapses of the postsynaptic type, ``pcnt_output`` by the presynapses of the presynaptic type and ``nrn_weight`` by
    the number of neurons of the postsynaptic type. The types in `merge` are then also combined into the node `merged_name`
    (connections with each of the `partners`, in both directions), and only connections among the `partners` and the
    merged node are kept (no self-connections, no merged constituents).

    Args:
        conn_df: Connections with type_pre, type_post, weight.
        neuron_df: The neurons involved (with ``type``).
        type_counts: Pre- and postsynapses per type (see :func:`fetch_type_synapse_counts`).
        merge: Types (regular expressions) to combine.
        merged_name: Name of the combined node.
        partners: Types that connect with the merged node.

    Returns:
        Tuple of (connection table with type_pre, type_post, weight, relative_weight, pcnt_output, nrn_weight, and the
        number of neurons of each type as a dict including the merged node).
    """
    n_neurons = neuron_df['type'].value_counts()
    merge = list(n_neurons.index[match_types(pd.Series(n_neurons.index), merge)])      # the types present that are merged
    counts = type_counts.copy()
    counts.loc[merged_name] = counts.reindex(merge).sum()
    n_neurons_all = {**n_neurons.to_dict(), merged_name: int(n_neurons.reindex(merge).sum())}

    def normalized(df):
        df = df.copy()
        df['relative_weight'] = df['weight'] / df['type_post'].map(counts['n_postsynapses'])
        df['pcnt_output'] = df['weight'] / df['type_pre'].map(counts['n_presynapses'])
        df['nrn_weight'] = df['weight'] / df['type_post'].map(n_neurons_all)
        return df

    grouped = conn_df.groupby(['type_pre', 'type_post'], as_index=False)['weight'].sum()
    to_merged = grouped[grouped['type_pre'].isin(partners) & grouped['type_post'].isin(merge)]
    from_merged = grouped[grouped['type_pre'].isin(merge) & grouped['type_post'].isin(partners)]
    merged = pd.concat([
        to_merged.assign(type_post=merged_name).groupby(['type_pre', 'type_post'], as_index=False)['weight'].sum(),
        from_merged.assign(type_pre=merged_name).groupby(['type_pre', 'type_post'], as_index=False)['weight'].sum()],
        ignore_index=True)
    keep = [*partners, merged_name]
    table = normalized(pd.concat([grouped, merged], ignore_index=True))
    table = table[table['type_pre'].isin(keep) & table['type_post'].isin(keep) & (table['type_pre'] != table['type_post'])]
    return table.reset_index(drop=True), n_neurons_all


def connectivity_graph(type_table, n_neurons, edge_attrs=('weight', 'relative_weight', 'pcnt_output', 'nrn_weight')):
    """Directed graph of the connections between types; nodes carry their number of neurons (``n_neurons``)."""
    graph = nx.from_pandas_edgelist(type_table, source='type_pre', target='type_post', edge_attr=list(edge_attrs),
                                    create_using=nx.DiGraph)
    nx.set_node_attributes(graph, {node: n_neurons[node] for node in graph.nodes}, 'n_neurons')
    return graph


# ----------------------------------------------------------------------------
# Partner counts
# ----------------------------------------------------------------------------
def count_partners(conn_df, post_ids, min_weight):
    """Number of presynaptic partners with a connection weight above `min_weight`, for each neuron in `post_ids`."""
    strong = conn_df.loc[conn_df['weight'] > min_weight]
    counts = strong['bodyId_post'].value_counts()
    return np.array([counts.get(body_id, 0) for body_id in post_ids])


# ----------------------------------------------------------------------------
# Connectivity of a dataset (see :mod:`datasets`) and comparison across datasets
# ----------------------------------------------------------------------------
INPUT_TYPES = ('EL', 'EPG', 'PEG', 'PEN_a.*', 'PEN_b.*')                 # presynaptic types of the ER input matrices
ER_TYPES = ('ER1.*', 'ER2.*', 'ER3.*', 'ER4.*')
GRAPH_TYPES = ('EL', 'EPG', 'ER4d', 'ER2.*')                             # types of the type-level graph


def er_input_connections(dataset, input_types=INPUT_TYPES, er_types=ER_TYPES, min_weight=MIN_ROI_WEIGHT):
    """Connections onto ER neurons in the region, with weights normalized by synapse counts (:func:`add_relative_weights`).

    Args:
        dataset: A :class:`~el_hd_plasticity.connectomics.datasets.ConnectomeDataset`.
        input_types, er_types: Presynaptic types and ER types (regular expressions).
        min_weight: Minimum number of synapses of a connection.

    Returns:
        Connection table with bodyId_pre, bodyId_post, type_pre, type_post, weight, relative_weight and pcnt_output.
    """
    pre = dataset.neurons(input_types, role='pre')
    post = dataset.neurons(er_types, role='post')
    conn = dataset.connections(pre['bodyId'], post['bodyId'], min_weight)
    neurons = pd.concat([pre, post]).drop_duplicates('bodyId')
    types = neurons.set_index('bodyId')['type']
    conn = conn.assign(type_pre=conn['bodyId_pre'].map(types), type_post=conn['bodyId_post'].map(types))
    return add_relative_weights(conn, neurons, dataset.synapse_counts(conn['bodyId_pre'].unique(), 'pre'),
                                dataset.synapse_counts(conn['bodyId_post'].unique(), 'post'))


def dataset_type_connectivity(dataset, types=GRAPH_TYPES, min_weight=MIN_ROI_WEIGHT, **kwargs):
    """Type-level connectivity (:func:`type_connectivity`) of a dataset.

    Returns:
        Tuple of (connection table, number of neurons of each type).
    """
    neurons = dataset.neurons(types)
    conn = dataset.connections(neurons['bodyId'], neurons['bodyId'], min_weight)
    type_of = neurons.set_index('bodyId')['type']
    conn = conn.assign(type_pre=conn['bodyId_pre'].map(type_of), type_post=conn['bodyId_post'].map(type_of))
    return type_connectivity(conn, neurons, dataset.type_synapse_counts(neurons), **kwargs)


def er_input_matrix(conn_df, weight_col='relative_weight', input_types=None):
    """Weight of the connections from each presynaptic type (rows) onto each ER type (columns).

    The weights of the connections of all neurons of a pair of types are combined by
    :func:`navis.interfaces.neuprint.connection_table_to_matrix`. Types without connections are zero rows;
    `input_types` gives the rows to keep, in order.
    """
    matrix = neu.connection_table_to_matrix(conn_df, weight_col=weight_col, sort_by='type', group_cols='type')
    if input_types is not None:
        matrix = matrix.reindex(list(input_types)).fillna(0)
    return matrix


def compare_datasets(datasets, func, **kwargs):
    """Apply an analysis to several datasets and stack the resulting tables, with a ``dataset`` column.

    Args:
        datasets: Dict or list of :class:`~el_hd_plasticity.connectomics.datasets.ConnectomeDataset`.
        func: Function ``func(dataset, **kwargs)`` returning a DataFrame, or a tuple whose first element is one (e.g.
            :func:`dataset_type_connectivity`).
        **kwargs: Passed to `func`.
    """
    datasets = datasets.values() if isinstance(datasets, dict) else datasets
    tables = []
    for dataset in datasets:
        result = func(dataset, **kwargs)
        table = result[0] if isinstance(result, tuple) else result
        tables.append(table.assign(dataset=dataset.name))
    return pd.concat(tables, ignore_index=True)


def er_input_marginals(conn_dfs, weight_col='relative_weight', input_types=('EL', 'EPG', 'PEG', 'PEN_a(PEN1)', 'PEN_b(PEN2)')):
    """Mean over the ER types of the weight of each presynaptic type onto them, per dataset (rows: types, columns: datasets).

    Args:
        conn_dfs: Dict dataset name -> output of :func:`er_input_connections`.
        input_types: Presynaptic types in the order of the rows (the names of the datasets' types).
    """
    return pd.DataFrame({name: er_input_matrix(conn_df, weight_col, input_types).mean(axis=1)
                         for name, conn_df in conn_dfs.items()})
