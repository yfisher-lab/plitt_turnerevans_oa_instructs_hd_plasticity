"""Compute cable/Euclidean distances between EPG output synapses and EL/ExR2/ExR3 input synapses
on ring (ER) neurons of the hemibrain, for each ER type.

Results are pickled per ER type as ``<outdir>/<ER type>_syn_distances.pkl`` (see
:func:`synapse_distance.calc_syn_distance` for the format). Requires neuPrint access.

Usage:
    python -m el_hd_plasticity.connectomics.run_synapse_distance [--outdir DIR] [--processes N]
"""

import argparse
import pathlib

import navis.interfaces.neuprint as neu
import neuprint as neu_orig
import pandas as pd

from . import synapse_distance
from .connectomics import npt_client
from .synapse_distance import ER_TYPES

DEFAULT_OUTDIR = '/media/mplitt/SSD_storage/fig_scratch/EL_connectomics/syn_distances'

PRE_TYPE = 'epg_pre'
POST_TYPES = ['el_post', 'exr2_post', 'exr3_post']


def _fetch_neurons(neuron_type):
    """Table of all hemibrain neurons of one type."""
    neurons, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type=neuron_type))
    return neurons


def get_er_syn(ertype='ER4d'):
    """Fetch EB synapses between ER neurons of one type and their EPG, EL, ExR2 and ExR3 partners.

    Args:
        ertype: Hemibrain neuron type of the ring neurons.

    Returns:
        Tuple of (synapse table with ``type`` and x/y/z, ER neuron table). EPG->ER synapses are
        located at the presynaptic site on the EPG ('epg_pre'); EL/ExR2/ExR3->ER synapses at the
        postsynaptic site on the ER neuron ('el_post', 'exr2_post', 'exr3_post').
    """
    ers = _fetch_neurons(ertype)
    criteria = neu_orig.SynapseCriteria(rois='EB')

    er_epg_synapses = neu_orig.fetch_synapse_connections(ers['bodyId'], _fetch_neurons('EPG')['bodyId'], criteria)
    syns = [synapse_distance.label_synapses(er_epg_synapses, PRE_TYPE, 'pre')]
    for partner, label in (('EL', 'el_post'), ('ExR2', 'exr2_post'), ('ExR3', 'exr3_post')):
        partner_syns = neu_orig.fetch_synapse_connections(_fetch_neurons(partner)['bodyId'], ers['bodyId'], criteria)
        syns.append(synapse_distance.label_synapses(partner_syns, label, 'post'))

    return pd.concat(syns, ignore_index=True), ers


def run_er_type(er_type, outdir, processes=8):
    """Compute and pickle synapse distances for every neuron of one ER type."""
    syns, nrns = get_er_syn(er_type)
    return synapse_distance.run_distances(syns, nrns['bodyId'].tolist(), PRE_TYPE, POST_TYPES,
                                          outdir / f'{er_type}_syn_distances.pkl', processes=processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='folder for the output pickles')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = npt_client()      # keep a reference: neuPrint only holds the default client weakly
    for er_type in ER_TYPES:
        print(er_type)
        run_er_type(er_type, outdir, processes=args.processes)
