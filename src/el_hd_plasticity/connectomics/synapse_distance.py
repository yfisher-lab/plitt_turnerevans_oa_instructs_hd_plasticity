"""Cable and Euclidean distances between synapses along a neuron's skeleton.

For each neuron, every synapse of a "pre" structure (e.g. EPG output synapses onto a ring neuron)
is compared with the nearest synapses of one or more "post" structures (e.g. EL input synapses
onto the same ring neuron): the Euclidean distance to the closest post synapse, and the shortest
path length along the neuron's (upsampled) skeleton to any post synapse within
:data:`MAX_EUCLIDEAN_NM` -- limited to :data:`MAX_CABLE_NM`.

Synapse tables are built by the ``run_*`` scripts with a ``type`` column naming the structure
(e.g. 'epg_pre', 'el_post') and ``x/y/z`` columns giving the location on the neuron of interest.
"""

import pathlib
import pickle
from functools import partial
from multiprocessing import Pool

import networkx as nx
import neuprint as neu_orig
import numpy as np
import pandas as pd

# neuPrint coordinates are in voxels of 8 nm
VOXEL_NM = 8
MAX_EUCLIDEAN_NM = 10 * 1000   # only post synapses closer than this are candidates for the path search
MAX_CABLE_NM = 15 * 1000       # paths longer than this are not searched

# Ring (ER) neuron types of the hemibrain, and their grouping into the families shown in the figures.
ER_TYPES = ('ER4d', 'ER4m',
            'ER2_a', 'ER2_b', 'ER2_c', 'ER2_d',
            'ER3w_b', 'ER3a_c', 'ER3p_a', 'ER3w_a', 'ER3a_b', 'ER3p_b', 'ER3d_c', 'ER3d_a',
            'ER3d_d', 'ER3d_b', 'ER3a_a', 'ER3a_d', 'ER3m',
            'ER1_b', 'ER1_a')
ER_GROUPS = (('ER4d',), ('ER4m',),
             ('ER2_a', 'ER2_b', 'ER2_c', 'ER2_d'),
             ('ER3w_a', 'ER3w_b'), ('ER3a_a', 'ER3a_b', 'ER3a_c', 'ER3a_d'),
             ('ER3p_a', 'ER3p_b'),
             ('ER3d_a', 'ER3d_b', 'ER3d_c', 'ER3d_d'),
             ('ER3m',),
             ('ER1_b', 'ER1_a'))

HEMISPHERES = ('L', 'R')       # the hemisphere of a cell type instance, e.g. EPG(PB08)_L3

# The two EPG instances innervating the same EB wedge, paired as in Hulse et al. 2021 (eLife 10:e66039).
EPG_COLUMN_PAIRS = (('L1', 'R1'), ('L2', 'R8'), ('L3', 'R7'), ('L4', 'R6'),
                    ('L5', 'R5'), ('L6', 'R4'), ('L7', 'R3'), ('L8', 'R2'))
EPG_COLUMNS = {side: f'{left}/{right}' for left, right in EPG_COLUMN_PAIRS for side in (left, right)}


def instance_hemisphere(instances):
    """Hemisphere ('L'/'R') of cell type instances like ``EPG(PB08)_L3``; NaN where there is no such suffix."""
    return pd.Series(instances).str.extract(r'_([LR])\d*$', expand=False).to_numpy()


def instance_column(instances):
    """EB column of EPG instances like ``EPG(PB08)_L3``, pairing the hemispheres as in :data:`EPG_COLUMN_PAIRS`.

    The two instances of a column (e.g. ``EPG(PB08)_L3`` and ``EPG(PB08)_R7``) innervate the same wedge of the EB and
    share the label ``'L3/R7'``. Instances without an ``_L``/``_R`` suffix give NaN.
    """
    suffix = pd.Series(instances).str.extract(r'_([LR]\d+)$', expand=False)
    return suffix.map(EPG_COLUMNS).to_numpy()


def calc_syn_distance(args, pre_type, post_types, max_euclidean_nm=MAX_EUCLIDEAN_NM, max_cable_nm=MAX_CABLE_NM):
    """Distance from each `pre_type` synapse to the nearest synapse of each post structure.

    Args:
        args: Tuple (bodyId, synapses) with the neuron's body ID and a DataFrame of its synapses
            (columns x, y, z, type, ...) as built for :func:`neuprint.attach_synapses_to_skeleton`.
        pre_type: Value of ``type`` marking the reference synapses.
        post_types: Values of ``type`` of the structures to measure distance to.
        max_euclidean_nm: Only post synapses closer than this are candidates for the path search; None for all of them.
        max_cable_nm: Longest path to search (nm); None for no limit.

    Returns:
        Dict with 'cable_distance' and 'euclidean_distance', each a dict mapping every post type
        to an array with one distance (in voxels) per `pre_type` synapse; NaN where no post
        synapse is within range or there is no path.
    """
    body_id, cell_syns = args[0], args[1]

    # get skeleton and attach synapses
    skel = neu_orig.fetch_skeleton(body_id, heal=True, with_distances=True)
    skel = neu_orig.upsample_skeleton(skel, 1)
    skel = neu_orig.attach_synapses_to_skeleton(skel, cell_syns)

    # convert skeleton to networkx graph and add the structure of each node
    struct_dict = dict(zip(skel['rowId'], skel['structure'], strict=True))
    skel_nx = neu_orig.skeleton_df_to_nx(skel, with_distances=True, directed=False)
    nx.set_node_attributes(skel_nx, struct_dict, 'structure')

    # node names of the reference synapses
    pre_nodes = skel.loc[skel['structure'] == pre_type, 'rowId'].tolist()

    euclidean_cutoff = np.inf if max_euclidean_nm is None else max_euclidean_nm / VOXEL_NM
    cable_cutoff = None if max_cable_nm is None else max_cable_nm / VOXEL_NM

    distances, euc_distances = {}, {}
    for key in post_types:
        post_syns = skel.loc[skel['structure'] == key, :]

        distances[key] = np.zeros((len(pre_nodes),))
        euc_distances[key] = np.zeros((len(pre_nodes),))
        for i, pre_node in enumerate(pre_nodes):
            try:
                # filter post nodes to ones within 10 microns euclidean distance
                pre_syn = skel.loc[skel['rowId'] == pre_node, :]
                euc_dist = np.linalg.norm(pre_syn.loc[:, ('x', 'y', 'z')]._values - post_syns.loc[:, ('x', 'y', 'z')]._values, axis=1)
                euc_distances[key][i] = np.nanmin(euc_dist)
                post_syns_close = post_syns.loc[euc_dist < euclidean_cutoff, :]

                post_nodes = post_syns_close['rowId'].tolist()
                dist, _ = nx.multi_source_dijkstra(skel_nx, post_nodes, target=pre_node,
                                                   weight='distance', cutoff=cable_cutoff)
                distances[key][i] = np.nanmin(dist)
            except Exception:  # noqa: BLE001
                # Deliberately broad: one neuron without a post synapse in range or without a path
                # within the cutoff (nanmin of an empty array, NetworkXNoPath, ...) becomes NaN rather
                # than aborting a long parallel run over hundreds of neurons.
                distances[key][i] = np.nan
                euc_distances[key][i] = np.nan

    return {'cable_distance': distances, 'euclidean_distance': euc_distances}


def calc_instance_distance(args):
    """Distance from each synapse to the nearest synapse of a different group, along the neuron's cable.

    Each synapse carries a group label (e.g. the cell type instance of its postsynaptic partner). For every synapse the
    shortest path along the neuron's (upsampled) skeleton to any synapse with a *different* label is measured, limited
    to :data:`MAX_CABLE_NM` unless a different limit is given.

    Args:
        args: Tuple (bodyId, synapses[, max_cable_nm]) with the neuron's body ID, a DataFrame of its synapses with
            columns x, y, z and ``type`` (the group label), one row per synapse, and optionally the longest path (nm)
            to search (default :data:`MAX_CABLE_NM`; None for no limit).

    Returns:
        Dict with ``instance`` (the group label of each synapse, in the order of `synapses`) and ``cable_distance`` (the
        distance in voxels to the nearest synapse of another group; NaN if there is none within the cutoff).
    """
    body_id, syns = args[0], args[1].reset_index(drop=True)
    max_cable_nm = args[2] if len(args) > 2 else MAX_CABLE_NM
    if syns.empty:          # e.g. a neuron without a synapse onto the hemisphere being analysed
        return {'instance': np.array([], dtype=object), 'cable_distance': np.array([])}
    skel = neu_orig.fetch_skeleton(body_id, heal=True, with_distances=True)
    skel = neu_orig.upsample_skeleton(skel, 1)
    skel = neu_orig.attach_synapses_to_skeleton(skel, syns)
    skel_nx = neu_orig.skeleton_df_to_nx(skel, with_distances=True, directed=False)

    # the synapses were appended to the skeleton as nodes, in order
    syn_nodes = skel.loc[skel['structure'] != 'neurite', ['rowId', 'structure']]
    labels = syn_nodes['structure'].astype(str).to_numpy()
    distance = np.full(len(syn_nodes), np.nan)
    cutoff = None if max_cable_nm is None else max_cable_nm / VOXEL_NM
    for label in np.unique(labels):
        others = syn_nodes.loc[labels != label, 'rowId'].tolist()
        if not others:
            continue
        reach = nx.multi_source_dijkstra_path_length(skel_nx, others, cutoff=cutoff, weight='distance')
        mine = labels == label
        distance[mine] = [reach.get(node, np.nan) for node in syn_nodes.loc[mine, 'rowId']]
    return {'instance': labels, 'cable_distance': distance}


def adjacent_instances(syns, n_neighbors=2):
    """The most-synapsed instance of a neuron and the instances whose synapses lie closest to it.

    Args:
        syns: One neuron's synapses with ``type`` (instance) and ``x_eb``, ``y_eb``, ``z_eb`` columns.
        n_neighbors: Number of neighboring instances to return.

    Returns:
        List of instance labels: the instance with most synapses, then the `n_neighbors` instances whose synapse centroids
        are nearest to its centroid.
    """
    main = syns['type'].value_counts().index[0]
    centroids = syns.groupby('type')[['x_eb', 'y_eb', 'z_eb']].mean()
    separation = np.linalg.norm(centroids - centroids.loc[main], axis=1)
    neighbors = centroids.index[np.argsort(separation)]
    return [main] + [instance for instance in neighbors if instance != main][:n_neighbors]


def neuron_synapse_args(syns, bodyIds, pre_type):
    """Split a pooled synapse table into one (bodyId, synapses) argument per neuron.

    Args:
        syns: DataFrame with ``type`` ('<name>_pre' / '<name>_post') and the body IDs of the
            presynaptic (``bodyId_pre``) and postsynaptic (``bodyId_post``) neurons.
        bodyIds: Body IDs of the neurons of interest.
        pre_type: ``type`` of the reference synapses, which are located on the *presynaptic*
            neuron; all other (post) types are located on the *postsynaptic* neuron.

    Returns:
        List of (bodyId, synapse DataFrame) tuples for :func:`calc_syn_distance`.
    """
    args = []
    for body_id in bodyIds:
        mask = (syns['type'] == pre_type) & (syns['bodyId_pre'] == body_id)
        mask = mask | (syns['type'].str.contains('post') & (syns['bodyId_post'] == body_id))
        args.append((body_id, syns.loc[mask, :]))
    return args


def set_synapse_location(syns, side):
    """Copy x/y/z of the pre or post side of synapse connections into columns x, y, z, in place."""
    syns['x'], syns['y'], syns['z'] = syns[f'x_{side}'], syns[f'y_{side}'], syns[f'z_{side}']
    return syns


def label_synapses(syns, label, side):
    """Tag a synapse-connection table with a structure ``type`` and the location x/y/z of `side`."""
    syns['type'] = label
    return set_synapse_location(syns, side)


def run_distances(syns, bodyIds, pre_type, post_types, outfile, processes=8,
                  max_euclidean_nm=MAX_EUCLIDEAN_NM, max_cable_nm=MAX_CABLE_NM):
    """Compute synapse distances for many neurons in parallel and pickle the results.

    Args:
        syns: Pooled synapse table (see :func:`neuron_synapse_args`).
        bodyIds: Body IDs of the neurons to analyse.
        pre_type: ``type`` of the reference synapses.
        post_types: ``type`` values of the structures to measure distance to.
        outfile: Path of the pickle to write (list of :func:`calc_syn_distance` results, in
            the order of `bodyIds`).
        processes: Number of worker processes.
        max_euclidean_nm, max_cable_nm: Search limits of :func:`calc_syn_distance`; None for no limit.

    Returns:
        The list of results.
    """
    args = neuron_synapse_args(syns, bodyIds, pre_type)
    calc = partial(calc_syn_distance, pre_type=pre_type, post_types=post_types,
                   max_euclidean_nm=max_euclidean_nm, max_cable_nm=max_cable_nm)
    with Pool(processes=processes) as pool:
        results = pool.map(calc, args)

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)
    return results


def run_instance_distances(syns, bodyIds, outfile, processes=8, max_cable_nm=MAX_CABLE_NM, neuron_col='bodyId_pre'):
    """Compute :func:`calc_instance_distance` for many neurons in parallel and pickle the results.

    Args:
        syns: Synapse table with x, y, z, ``type`` (group label) and `neuron_col`.
        bodyIds: Body IDs of the neurons to analyse.
        outfile: Path of the pickle to write (list of results in the order of `bodyIds`).
        processes: Number of worker processes.
        max_cable_nm: Longest path to search (nm); None for no limit.
        neuron_col: Column naming the neuron each synapse is located on -- ``bodyId_pre`` when the distances are
            measured along the presynaptic neuron, ``bodyId_post`` when they are measured along the postsynaptic one.
    """
    args = [(body_id, syns.loc[syns[neuron_col] == body_id, ['x', 'y', 'z', 'type']], max_cable_nm)
            for body_id in bodyIds]
    with Pool(processes=processes) as pool:
        results = pool.map(calc_instance_distance, args)
    with open(outfile, 'wb') as f:
        pickle.dump(results, f)
    return results


# ----------------------------------------------------------------------------
# Loading results
# ----------------------------------------------------------------------------
def load_results(path):
    """Results written by :func:`run_distances` (list of :func:`calc_syn_distance` outputs, one per neuron)."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def distances_um(results, structure):
    """Cable distances (microns) of all neurons' reference synapses to `structure` (e.g. 'el_post'), NaNs included."""
    return np.concatenate([r['cable_distance'][structure] for r in results]) * VOXEL_NM / 1000


def load_distance_table(folder, types, structures, names=None, filename='{type}_syn_distances.pkl'):
    """Cable distances of several neuron types to several structures, as a long table.

    Args:
        folder: Folder with the pickles written by the ``run_*`` scripts.
        types: Neuron types (formatted into `filename`).
        structures: ``type`` values of the post structures, e.g. ``('el_post', 'exr2_post')``.
        names: Label of each structure in the table (default: the structure without its '_post' suffix).
        filename: File name pattern of the pickle of each type.

    Returns:
        DataFrame with columns ``er`` (the neuron type), ``source`` (the structure) and ``distances`` (microns, NaN where
        no post synapse was in range).
    """
    names = names or [name.removesuffix('_post') for name in structures]
    frames = []
    for neuron_type in types:
        results = load_results(pathlib.Path(folder) / filename.format(type=neuron_type))
        for structure, name in zip(structures, names, strict=True):
            distances = distances_um(results, structure)
            frames.append(pd.DataFrame({'er': neuron_type, 'source': name, 'distances': distances}))
    return pd.concat(frames, ignore_index=True)


def load_instance_distance_table(folder, types, name='epg_column', filename='{type}_epg_column_distances.pkl'):
    """Distances to the nearest synapse of another group (see :func:`calc_instance_distance`) as a long table.

    Args:
        folder: Folder with the pickles written by :mod:`run_epg_instance_distance`.
        types: Neuron types (formatted into `filename`).
        name: Label of these distances in the ``source`` column.
        filename: File name pattern of the pickle of each type.

    Returns:
        DataFrame with ``er`` (the neuron type), ``source`` (`name`), ``group`` (the EPG group the synapse belongs to)
        and ``distances`` (microns; NaN where no synapse of another group was within range), in the format of
        :func:`load_distance_table`.
    """
    frames = []
    for neuron_type in types:
        results = load_results(pathlib.Path(folder) / filename.format(type=neuron_type))
        distances = np.concatenate([r['cable_distance'] for r in results]) * VOXEL_NM / 1000
        groups = np.concatenate([r['instance'] for r in results])
        frames.append(pd.DataFrame({'er': neuron_type, 'source': name, 'group': groups, 'distances': distances}))
    return pd.concat(frames, ignore_index=True)


def partner_count_table(conn_dfs, ers, groups=ER_GROUPS, min_weight=3):
    """Number of strong presynaptic partners of every neuron, by family of neuron types.

    Args:
        conn_dfs: Dict name -> connection table (bodyId_post, weight) of the partner population onto the neurons `ers`.
        ers: Neuron table with ``bodyId`` and ``type``.
        groups: Tuples of neuron types forming a family.
        min_weight: Partners count when their connection weight is above this.

    Returns:
        DataFrame with one row per neuron: ``group`` (the family, as a string), ``bodyId`` and one column per entry of
        `conn_dfs` with the partner count.
    """
    rows = []
    for group in groups:
        members = ers.loc[ers['type'].isin(group), 'bodyId']
        table = pd.DataFrame({'group': str(group), 'bodyId': members.to_numpy()})
        for name, conn_df in conn_dfs.items():
            strong = conn_df.loc[conn_df['weight'] > min_weight, 'bodyId_post'].value_counts()
            table[name] = [int(strong.get(body_id, 0)) for body_id in members]
        rows.append(table)
    return pd.concat(rows, ignore_index=True)
