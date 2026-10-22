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


def calc_syn_distance(args, pre_type, post_types):
    """Distance from each `pre_type` synapse to the nearest synapse of each post structure.

    Args:
        args: Tuple (bodyId, synapses) with the neuron's body ID and a DataFrame of its synapses
            (columns x, y, z, type, ...) as built for :func:`neuprint.attach_synapses_to_skeleton`.
        pre_type: Value of ``type`` marking the reference synapses.
        post_types: Values of ``type`` of the structures to measure distance to.

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

    euclidean_cutoff = MAX_EUCLIDEAN_NM / VOXEL_NM
    cable_cutoff = MAX_CABLE_NM / VOXEL_NM

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


def run_distances(syns, bodyIds, pre_type, post_types, outfile, processes=8):
    """Compute synapse distances for many neurons in parallel and pickle the results.

    Args:
        syns: Pooled synapse table (see :func:`neuron_synapse_args`).
        bodyIds: Body IDs of the neurons to analyse.
        pre_type: ``type`` of the reference synapses.
        post_types: ``type`` values of the structures to measure distance to.
        outfile: Path of the pickle to write (list of :func:`calc_syn_distance` results, in
            the order of `bodyIds`).
        processes: Number of worker processes.

    Returns:
        The list of results.
    """
    args = neuron_synapse_args(syns, bodyIds, pre_type)
    with Pool(processes=processes) as pool:
        results = pool.map(partial(calc_syn_distance, pre_type=pre_type, post_types=post_types), args)

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
