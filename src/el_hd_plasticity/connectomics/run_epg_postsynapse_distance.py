"""Compute cable distances between the input synapses an EPG neuron receives from different ER neurons.

The mirror image of :mod:`run_epg_instance_distance`: the distances are measured along the *postsynaptic* EPG neuron
rather than along the presynaptic ER neuron. For every ER->EPG synapse (located on the EPG neuron), the shortest distance
along the EPG neuron's skeleton to another ER->EPG synapse of a different group is computed, within one ER type.

Two groupings are available (``--groupings``):
    source      the ER neuron the synapse comes from -- the nearest input from any other ER neuron of the type.
    hemisphere  the hemisphere of that ER neuron (e.g. ``ER4d(ring)_L`` vs ``ER4d(ring)_R``) -- with two groups, the
                nearest input from an ER neuron of the *other* hemisphere.

They are pickled per ER type and grouping as ``<outdir>/<ER type>_epg_post_<grouping>_distances.pkl`` (see
:func:`synapse_distance.calc_instance_distance` for the format).

With ``--el`` the distance from each ER->EPG synapse to the nearest EL->EPG synapse on the same EPG neuron is computed
instead -- the postsynaptic counterpart of the EL distances of :mod:`run_synapse_distance`. Like the groupings above it
searches without a length limit, so every synapse with an EL input on its EPG gets a distance. It is pickled as
``<outdir>/<ER type>_epg_post_el_distances.pkl`` (the format of :func:`synapse_distance.calc_syn_distance`).
Requires neuPrint access.

Usage:
    python -m el_hd_plasticity.connectomics.run_epg_postsynapse_distance [--outdir DIR] [--processes N]
        [--types ER4d ...] [--groupings source hemisphere] [--el]
"""

import argparse
import pathlib

import navis.interfaces.neuprint as neu
import neuprint as neu_orig
import pandas as pd

from . import synapse_distance
from .connectomics import npt_client
from .synapse_distance import instance_hemisphere

DEFAULT_OUTDIR = '/media/mplitt/SSD_storage/fig_scratch/EL_connectomics/syn_distances'
GROUPINGS = ('source', 'hemisphere')
ER_POST_TYPE, EL_POST_TYPE = 'er_post', 'el_post'   # both located on the EPG neuron


def get_epg_er_synapses(ertype='ER4d'):
    """Fetch the EB synapses from ER neurons of one type onto EPG neurons, located on the EPG side.

    Returns:
        Tuple of (synapse table with bodyId_post (the EPG the synapse is located on), x, y, z of the postsynaptic site,
        ``source`` = the ER neuron's body ID and ``hemisphere`` = the hemisphere of that ER neuron, EPG neuron table).
    """
    ers, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type=ertype))
    epgs, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='EPG'))
    syns = neu_orig.fetch_synapse_connections(ers['bodyId'], epgs['bodyId'], neu_orig.SynapseCriteria(rois='EB'))
    syns = synapse_distance.set_synapse_location(syns, 'post')
    syns['source'] = syns['bodyId_pre']
    syns['hemisphere'] = instance_hemisphere(syns['bodyId_pre'].map(ers.set_index('bodyId')['instance']))
    return syns, epgs


def get_epg_el_synapses(ertype='ER4d'):
    """Fetch the EB synapses onto EPG neurons from ER neurons of one type and from EL, both on the EPG side.

    Returns:
        Tuple of (synapse table with ``type`` 'er_post' / 'el_post' and x, y, z of the postsynaptic site on the EPG,
        EPG neuron table).
    """
    ers, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type=ertype))
    epgs, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='EPG'))
    els, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='EL'))
    criteria = neu_orig.SynapseCriteria(rois='EB')

    syns = [synapse_distance.label_synapses(neu_orig.fetch_synapse_connections(source, epgs['bodyId'], criteria),
                                            label, 'post')
            for source, label in ((ers['bodyId'], ER_POST_TYPE), (els['bodyId'], EL_POST_TYPE))]
    return pd.concat(syns, ignore_index=True), epgs


def run_er_type_el(er_type, outdir, processes=8):
    """Compute and pickle, for one ER type, the distance from each ER->EPG synapse to the nearest EL->EPG synapse."""
    syns, epgs = get_epg_el_synapses(er_type)
    return synapse_distance.run_distances(syns, epgs['bodyId'].tolist(), ER_POST_TYPE, [EL_POST_TYPE],
                                          pathlib.Path(outdir) / f'{er_type}_epg_post_el_distances.pkl',
                                          processes=processes, max_euclidean_nm=None, max_cable_nm=None)


def run_er_type(er_type, outdir, processes=8, max_cable_nm=None, groupings=GROUPINGS):
    """Compute and pickle the postsynaptic distances for one ER type, once per grouping.

    Returns:
        Dict grouping -> the results of :func:`synapse_distance.run_instance_distances` for that grouping.
    """
    syns, epgs = get_epg_er_synapses(er_type)
    results = {}
    for grouping in groupings:
        outfile = pathlib.Path(outdir) / f'{er_type}_epg_post_{grouping}_distances.pkl'
        results[grouping] = synapse_distance.run_instance_distances(
            syns.assign(type=syns[grouping]), epgs['bodyId'].tolist(), outfile,
            processes=processes, max_cable_nm=max_cable_nm, neuron_col='bodyId_post')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='folder for the output pickles')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--types', nargs='+', default=['ER4d'], help='ER types to analyse')
    parser.add_argument('--groupings', nargs='+', default=list(GROUPINGS), choices=GROUPINGS,
                        help='what counts as a different source of a synapse')
    parser.add_argument('--el', action='store_true',
                        help='instead measure the distance from each ER->EPG synapse to the nearest EL->EPG synapse')
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = npt_client()      # keep a reference: neuPrint only holds the default client weakly
    for er_type in args.types:
        print(er_type)
        if args.el:
            run_er_type_el(er_type, outdir, processes=args.processes)
        else:
            run_er_type(er_type, outdir, processes=args.processes, groupings=args.groupings)
