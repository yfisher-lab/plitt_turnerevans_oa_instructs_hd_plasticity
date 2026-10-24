"""Compute cable/Euclidean distances between KC output synapses onto MBONs and DAN input synapses
onto the same Kenyon cells (KCg) in the gamma lobe of the hemibrain.

Results are pickled as ``<outdir>/kc_syn_distances.pkl`` (see
:func:`synapse_distance.calc_syn_distance` for the format). Requires neuPrint access.

Usage:
    python -m el_hd_plasticity.connectomics.run_mb_synapse_distance [--outdir DIR] [--processes N]
"""

import argparse
import pathlib

import navis.interfaces.neuprint as neu
import neuprint as neu_orig
import pandas as pd

from . import synapse_distance
from .connectomics import npt_client

DEFAULT_OUTDIR = '/media/mplitt/SSD_storage/fig_scratch/EL_connectomics/syn_distances'

GAMMA_LOBE = ['gL(L)', 'gL(R)']
PRE_TYPE = 'mbon_pre'
POST_TYPES = ['dan_post']


def get_kc_syn():
    """Fetch gamma-lobe synapses from KCg cells onto MBONs and from DANs onto KCg cells.

    Returns:
        Tuple of (synapse table with ``type`` and x/y/z, KC neuron table). KC->MBON synapses are
        located at the presynaptic site on the KC ('mbon_pre'); DAN->KC synapses at the
        postsynaptic site on the KC ('dan_post').
    """
    kcs, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='KCg.*'))
    mbons, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type='MBON.*', inputRois=GAMMA_LOBE, roi_req='any'))
    dans, _ = neu.queries.fetch_neurons(neu.NeuronCriteria(type=['PPL.*', 'PAM.*'], outputRois=GAMMA_LOBE,
                                                           roi_req='any'))

    criteria = neu_orig.SynapseCriteria(rois=GAMMA_LOBE)
    kc_mbon_synapses = neu_orig.fetch_synapse_connections(kcs['bodyId'], mbons['bodyId'], criteria)
    dan_kc_synapses = neu_orig.fetch_synapse_connections(dans['bodyId'], kcs['bodyId'], criteria)

    kc_syns = pd.concat([synapse_distance.label_synapses(kc_mbon_synapses, PRE_TYPE, 'pre'),
                         synapse_distance.label_synapses(dan_kc_synapses, 'dan_post', 'post')],
                        ignore_index=True)
    return kc_syns, kcs


def run_kc_type(outdir, processes=8):
    """Compute and pickle synapse distances for every KCg neuron."""
    syns, nrns = get_kc_syn()
    return synapse_distance.run_distances(syns, nrns['bodyId'].tolist(), PRE_TYPE, POST_TYPES,
                                          outdir / 'kc_syn_distances.pkl', processes=processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default=DEFAULT_OUTDIR, help='folder for the output pickle')
    parser.add_argument('--processes', type=int, default=8)
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = npt_client()      # keep a reference: neuPrint only holds the default client weakly
    run_kc_type(outdir, processes=args.processes)
