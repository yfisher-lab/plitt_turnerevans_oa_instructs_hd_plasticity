"""Compute cable distances between the output synapses of ER neurons onto EPG neurons of different cell type instances.

For every synapse of an ER neuron onto an EPG neuron (located on the ER neuron), the shortest distance along the ER neuron's
skeleton to a synapse onto an EPG of a *different instance* (e.g. ``EPG(PB08)_L3`` vs ``EPG(PB08)_R3``) is computed.

Results are pickled per ER type as ``<outdir>/<ER type>_epg_instance_distances.pkl`` (see
:func:`synapse_distance.calc_instance_distance` for the format). Requires neuPrint access.

Usage:
    python -m el_hd_plasticity.connectomics.run_epg_instance_distance [--outdir DIR] [--processes N] [--types ER4d ER2_a ...]
"""

import argparse
import pathlib

import navis.interfaces.neuprint as neu
import neuprint as neu_orig

from . import synapse_distance
from .connectomics import npt_client
from .synapse_distance import ER_TYPES

DEFAULT_OUTDIR = '/media/mplitt/SSD_storage/fig_scratch/EL_connectomics/syn_distances'


def get_er_epg_synapses(ertype='ER4d'):
    """Fetch the EB synapses from ER neurons of one type onto EPG neurons, labeled with the EPG instance.

    Returns:
        Tuple of (synapse table with bodyId_pre, x, y, z and ``type`` = instance of the EPG, ER neuron table). The coordinates
        are those of the presynaptic site on the ER neuron.
    """
    ers, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type=ertype))
    epgs, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='EPG'))
    syns = neu_orig.fetch_synapse_connections(ers['bodyId'], epgs['bodyId'], neu_orig.SynapseCriteria(rois='EB'))
    syns = synapse_distance.label_synapses(syns, 'instance', 'pre')
    syns['type'] = syns['bodyId_post'].map(epgs.set_index('bodyId')['instance'])
    return syns, ers


def run_er_type(er_type, outdir, processes=8, max_cable_nm=synapse_distance.MAX_CABLE_NM):
    """Compute and pickle the instance distances for every neuron of one ER type."""
    syns, nrns = get_er_epg_synapses(er_type)
    return synapse_distance.run_instance_distances(syns, nrns['bodyId'].tolist(),
                                                   pathlib.Path(outdir) / f'{er_type}_epg_instance_distances.pkl',
                                                   processes=processes, max_cable_nm=max_cable_nm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='folder for the output pickles')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--unlimited-types', nargs='*', default=[t for t in ER_TYPES if t != 'ER4d'],
                        help='types searched without a cable-distance limit, so every synapse with another instance '
                             'on its neuron gets a partner (the others use the 15 um limit of the EL analyses)')
    parser.add_argument('--types', nargs='+', default=list(ER_TYPES), help='ER types to analyse')
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = npt_client()      # keep a reference: neuPrint only holds the default client weakly
    for er_type in args.types:
        print(er_type)
        run_er_type(er_type, outdir, processes=args.processes,
                    max_cable_nm=None if er_type in args.unlimited_types else synapse_distance.MAX_CABLE_NM)
