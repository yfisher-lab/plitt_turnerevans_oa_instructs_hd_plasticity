"""Connectome datasets behind one interface, so the connectivity analyses run unchanged on any of them.

A :class:`ConnectomeDataset` answers the few questions the analyses of :mod:`connectivity` ask: which neurons have
which cell types, how strongly are two sets of neurons connected within a brain region (by default the EB), and how
many synapses do neurons have there. Neurons are identified by an integer ``bodyId`` (the body ID of neuPrint
datasets, the root ID of flywire datasets) and cell types follow the hemibrain naming (``EPG``, ``EL``, ``ER4d``,
``ER2_a``, ...); datasets that name types differently pass a ``type_map``.

Backends:

* :class:`NeuprintDataset`: the hemibrain and the male CNS (MCNS), served by neuPrint.
* :class:`TableDataset`: any dataset held in tables, i.e. a table of neurons (id, type) and a table of connections
  (pre id, post id, brain region, number of synapses). The flywire datasets (FAFB, BANC) are used through this:
  :class:`CodexClient` downloads their tables from Codex (https://codex.flywire.ai), which offers static downloads
  rather than a query API, and :func:`load_codex` turns them into a dataset.
"""

import pathlib
import shutil
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import neuprint as npt
import pandas as pd

from . import params

REGION = 'EB'


def match_types(types, patterns):
    """Boolean mask of the entries of `types` (Series) that fully match one of the regular expressions `patterns`."""
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    regex = '|'.join(f'(?:{p})' for p in patterns)
    return types.astype(str).str.fullmatch(regex)


class ConnectomeDataset(ABC):
    """What the connectivity analyses need from a connectome.

    Attributes:
        name: Label of the dataset in tables and figures.
        roi: Brain region the connectivity and synapse counts are restricted to.
    """

    name: str
    roi: str = REGION

    @abstractmethod
    def neurons(self, types, role=None):
        """Neurons of the given types (regular expressions, e.g. ``'ER2.*'``) as a DataFrame with ``bodyId`` and ``type``.

        Args:
            types: Type name(s) or regular expression(s).
            role: 'pre' keeps only neurons with output synapses in the region, 'post' only neurons with input synapses
                in it; None keeps all neurons of the types.
        """

    @abstractmethod
    def connections(self, pre_ids, post_ids, min_weight=1):
        """Connections within the region from `pre_ids` onto `post_ids`.

        Returns:
            DataFrame with ``bodyId_pre``, ``bodyId_post`` and ``weight`` (number of synapses in the region), for pairs
            with at least `min_weight` synapses.
        """

    @abstractmethod
    def synapse_counts(self, ids, side='pre'):
        """Number of ``side`` ('pre' or 'post') synapses in the region of each neuron: Series indexed by body ID."""

    def type_synapse_counts(self, neurons, types=None):
        """Pre- and postsynapses in the region of all neurons of each type (DataFrame indexed by type).

        Args:
            neurons: Neuron table with ``bodyId`` and ``type``.
            types: Types to report (default: all in `neurons`).
        """
        types = list(neurons['type'].unique()) if types is None else list(types)
        counts = pd.DataFrame(index=types, columns=['n_presynapses', 'n_postsynapses'], dtype=float)
        for col, side in (('n_presynapses', 'pre'), ('n_postsynapses', 'post')):
            per_neuron = self.synapse_counts(neurons['bodyId'], side)
            totals = neurons.assign(n=neurons['bodyId'].map(per_neuron).fillna(0)).groupby('type')['n'].sum()
            counts[col] = totals.reindex(types).fillna(0)
        return counts

    def __repr__(self):
        return f'{type(self).__name__}({self.name!r}, roi={self.roi!r})'


class NeuprintDataset(ConnectomeDataset):
    """A dataset served by neuPrint (hemibrain, male CNS).

    Args:
        name: Label of the dataset.
        dataset: neuPrint dataset name, e.g. ``'hemibrain:v1.2.1'`` or ``'male-cns:v0.9'``.
        server: neuPrint server.
        token: neuPrint token (default: :data:`params.NPT_TOKEN`).
        roi: Region of the connectivity.
        type_map: Optional function from the dataset's type names (Series) to hemibrain-style names.
    """

    def __init__(self, name, dataset, server='neuprint.janelia.org', token=None, roi=REGION, type_map=None):
        self.name, self.dataset, self.roi, self.type_map = name, dataset, roi, type_map
        self.client = npt.Client(server, dataset=dataset, token=token or params.NPT_TOKEN)

    def neurons(self, types, role=None):
        roi_kwargs = {} if role is None else {{'pre': 'outputRois', 'post': 'inputRois'}[role]: [self.roi], 'roi_req': 'any'}
        criteria = npt.NeuronCriteria(type=[types] if isinstance(types, str) else list(types), client=self.client,
                                      **roi_kwargs)
        neurons, _ = npt.fetch_neurons(criteria, client=self.client)
        neurons = neurons[['bodyId', 'type']].copy()
        if self.type_map is not None:
            neurons['type'] = self.type_map(neurons['type'])
        return neurons

    def connections(self, pre_ids, post_ids, min_weight=1):
        _, conn = npt.fetch_adjacencies(list(pre_ids), list(post_ids), rois=[self.roi], min_roi_weight=min_weight,
                                        client=self.client)
        return conn.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    def synapse_counts(self, ids, side='pre'):
        criteria = npt.SynapseCriteria(rois=[self.roi], type=side, primary_only=True, client=self.client)
        syn = npt.fetch_synapses(list(ids), criteria, client=self.client)
        return syn['bodyId'].value_counts()


class TableDataset(ConnectomeDataset):
    """A dataset held in tables.

    The connections can be given per synapse (one row each) or as an edge list (a number of synapses per pair of neurons
    and region). Synapse counts of a neuron are the sums over the rows it appears in, so they include only the
    connections present in the table.

    Args:
        name: Label of the dataset.
        neurons: DataFrame with ``bodyId`` and ``type`` (hemibrain-style type names) of the neurons of interest.
        connections: DataFrame with the ids of the presynaptic and postsynaptic neurons, the brain region, and optionally
            the number of synapses.
        roi: Region of the connectivity.
        pre_col, post_col, roi_col: Columns of `connections` holding those.
        weight_col: Column with the number of synapses of each row; if None every row is one synapse.
    """

    def __init__(self, name, neurons, connections, roi=REGION, pre_col='pre_id', post_col='post_id', roi_col='neuropil',
                 weight_col=None):
        self.name, self.roi = name, roi
        self._neurons = neurons[['bodyId', 'type']].reset_index(drop=True)
        connections = connections.loc[connections[roi_col] == roi]
        table = pd.DataFrame({'bodyId_pre': connections[pre_col].to_numpy(), 'bodyId_post': connections[post_col].to_numpy(),
                              'weight': 1 if weight_col is None else connections[weight_col].to_numpy()})
        self._edges = table.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    def neurons(self, types, role=None):
        neurons = self._neurons[match_types(self._neurons['type'], types)]
        if role is not None:
            neurons = neurons[neurons['bodyId'].isin(self._edges[f'bodyId_{role}'].unique())]
        return neurons.reset_index(drop=True)

    def connections(self, pre_ids, post_ids, min_weight=1):
        edges = self._edges[self._edges['bodyId_pre'].isin(list(pre_ids)) & self._edges['bodyId_post'].isin(list(post_ids))]
        return edges[edges['weight'] >= min_weight].reset_index(drop=True)

    def synapse_counts(self, ids, side='pre'):
        counts = self._edges.groupby(f'bodyId_{side}')['weight'].sum()
        return counts[counts.index.isin(list(ids))]


# ----------------------------------------------------------------------------
# FAFB and BANC from Codex
# ----------------------------------------------------------------------------
CODEX_URL = 'https://codex.flywire.ai'
CODEX_ID_PREFIX = 720575940 * 10 ** 9      # some Codex files drop the common prefix of the FlyWire root ids
CHUNK_BYTES = 1 << 20


class CodexClient:
    """Downloads the static data products of FlyWire Codex (https://codex.flywire.ai/download).

    Codex has no live query API: it serves gzipped CSV files per dataset ('fafb', 'banc'). The token goes into the URL
    exactly as copied from the Account page of Codex (it contains ':' and '@', which must not be percent-encoded), and the
    URL is built by hand for that reason. Downloads are cached on disk.

    Args:
        token: Codex API token; default: ``CODEX_API_TOKEN`` environment variable, see :data:`params.CODEX_TOKEN`.
        cache_dir: Where downloaded files are kept (default: ``~/.cache/el_hd_plasticity/codex``).
    """

    def __init__(self, token=None, cache_dir=None):
        self.token = token or params.CODEX_TOKEN
        if not self.token:
            raise RuntimeError(f'No Codex API token: pass token= or set the CODEX_API_TOKEN environment variable '
                               f'(copy it from {CODEX_URL}/account).')
        self.cache_dir = pathlib.Path(cache_dir or pathlib.Path.home() / '.cache' / 'el_hd_plasticity' / 'codex')

    @staticmethod
    def products_url(dataset):
        """Web page listing the data products of `dataset` (the names to pass as ``data_product``)."""
        return f'{CODEX_URL}/download?dataset={dataset}'

    def _open(self, data_product, dataset):
        """Open the download of a data product (a file-like object of the gzipped CSV)."""
        url = f'{CODEX_URL}/api/download_resource?api_token={self.token}&data_product={data_product}&dataset={dataset}'
        try:
            return urllib.request.urlopen(url)
        except urllib.error.HTTPError as error:       # the URL contains the token: keep it out of the message
            raise RuntimeError(f'Codex download of {data_product!r} ({dataset}) failed with HTTP {error.code}; check the API '
                               f'token and the data products at {self.products_url(dataset)}') from None

    def path(self, dataset, data_product):
        """Path of the cached file of a data product."""
        return self.cache_dir / f'{dataset}_{data_product}.csv.gz'

    def download(self, dataset, data_product, force=False):
        """Download a data product (unless cached); returns the path of the gzipped CSV."""
        path = self.path(dataset, data_product)
        if force or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.part')
            with self._open(data_product, dataset) as response:
                first = response.read(CHUNK_BYTES)
                if first[:2] != b'\x1f\x8b':
                    raise RuntimeError(f'Codex did not return a gzipped table for {data_product!r} ({dataset}); '
                                       f'see the data products at {self.products_url(dataset)}')
                with open(partial, 'wb') as f:
                    f.write(first)
                    shutil.copyfileobj(response, f, CHUNK_BYTES)
            partial.replace(path)
        return path

    def read(self, dataset, data_product, usecols=None, chunksize=None):
        """Read a downloaded data product as a DataFrame (or an iterator of chunks)."""
        return pd.read_csv(self.download(dataset, data_product), compression='gzip', usecols=usecols, chunksize=chunksize)


# Cell type table of each Codex dataset: data product and the columns with the root id and the primary cell type.
CODEX_CELL_TYPES = {
    'fafb': {'cell_types_product': 'consolidated_cell_types', 'id_col': 'root_id', 'type_col': 'primary_type'},
    'banc': {'cell_types_product': 'neurons', 'id_col': 'Root ID', 'type_col': 'Primary Cell Type'},
}


def _find_column(columns, base):
    """The column of `columns` equal to `base` or starting with ``base + '_'`` (e.g. 'root_id_720575940')."""
    for col in columns:
        if col == base or col.startswith(f'{base}_'):
            return col
    raise KeyError(f'no column {base!r} among {list(columns)}')


def _full_ids(table, base):
    """The ids of column `base` of `table`, with the dropped root-id prefix restored where the column name says so."""
    col = _find_column(table.columns, base)
    ids = table[col].astype('int64')
    return ids + CODEX_ID_PREFIX if col.endswith('720575940') else ids


def load_codex(name, dataset, client=None, connections_product='connections_princeton', cell_types_product=None,
               type_col=None, id_col=None, pre_col='pre_root_id', post_col='post_root_id', roi_col='neuropil',
               weight_col='syn_count', roi=REGION, type_map=None, chunksize=2_000_000):
    """FAFB or BANC as a :class:`TableDataset` from Codex downloads.

    Reads the table of connections (only the rows in `roi` are kept, chunk by chunk, as the tables are large) and the table
    of cell types (the primary cell type of each neuron).

    Args:
        name: Label of the dataset.
        dataset: Codex dataset, 'fafb' or 'banc'.
        client: :class:`CodexClient` (default: one with the token of the environment).
        connections_product: Data product with pre / post root ids, brain region and number of synapses per connection.
            Codex's ``connections_princeton`` lists connections of at least 5 synapses, whereas the neuPrint datasets are
            analysed with connections of at least 3: use a table without that threshold, if available, to compare weights.
        cell_types_product, type_col, id_col: Data product, type column and root id column of the cell type table; by default
            those of :data:`CODEX_CELL_TYPES` for `dataset`.
        pre_col, post_col, roi_col, weight_col: Columns of the connections table (the root-id columns may carry the
            ``_720575940`` suffix of the files that omit the common prefix of the ids).
        roi: Brain region of the connectivity.
        type_map: Optional function from the dataset's type names (Series) to hemibrain-style names.
        chunksize: Rows of the connections table read at a time.
    """
    client = client or CodexClient()
    defaults = CODEX_CELL_TYPES.get(dataset, {})
    cell_types_product = cell_types_product or defaults['cell_types_product']
    type_col, id_col = type_col or defaults['type_col'], id_col or defaults['id_col']

    chunks = []
    for chunk in client.read(dataset, connections_product, chunksize=chunksize):
        chunk = chunk[chunk[_find_column(chunk.columns, roi_col)] == roi]
        chunks.append(pd.DataFrame({'pre_id': _full_ids(chunk, pre_col), 'post_id': _full_ids(chunk, post_col),
                                    'neuropil': roi, 'weight': chunk[_find_column(chunk.columns, weight_col)].to_numpy()}))
    connections = pd.concat(chunks, ignore_index=True)

    cell_types = client.read(dataset, cell_types_product)
    neurons = pd.DataFrame({'bodyId': _full_ids(cell_types, id_col), 'type': cell_types[type_col]}).dropna(subset=['type'])
    if type_map is not None:
        neurons['type'] = type_map(neurons['type'])
    return TableDataset(name, neurons, connections, roi=roi, weight_col='weight')
