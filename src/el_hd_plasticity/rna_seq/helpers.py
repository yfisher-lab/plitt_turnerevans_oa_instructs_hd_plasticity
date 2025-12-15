"""Helper utilities for RNA-Seq plotting.

This module contains helper functions moved out of the notebook so they can be
reused and tested more easily. Function names use snake_case (PEP8).
"""
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_sub_dat(cells_meta: pd.DataFrame, cells_count: pd.DataFrame, name: str, norm: bool = True) -> pd.DataFrame:
    """Return a subset of count table for samples containing `name`.

    Args:
        cells_meta: DataFrame with sample metadata, must contain column `sample` and `qp_well_fq`.
        cells_count: Count DataFrame with columns for qp_well_fq identifiers and 'gene_id'.
        name: Substring to search for in the `sample` column of `cells_meta`.
        norm: If True, convert counts to CPM (counts per million).

    Returns:
        DataFrame indexed by `gene_id` with selected (and optionally normalized) columns.
    """
    # Find samples whose sample string contains the requested name
    mask = cells_meta['sample'].apply(lambda s: name in s)
    samp_ids = cells_meta.loc[mask].reset_index(drop=True)
    if samp_ids.empty:
        return pd.DataFrame()

    # Select the matching columns from counts using qp_well_fq identifiers
    cols = [c for c in cells_count.columns if c in samp_ids['qp_well_fq'].values]
    get_dat = cells_count[cols].copy()

    # Keep gene_id as index
    if 'gene_id' not in cells_count.columns:
        raise KeyError('cells_count must contain a "gene_id" column')
    get_dat['gene_id'] = cells_count['gene_id']
    get_dat = get_dat.set_index('gene_id')

    # Normalize to CPM if requested
    if norm:
        # Avoid repeated loops by vectorized operation: divide each column by column sum and multiply
        col_sums = get_dat.sum(axis=0).replace(0, np.nan)
        get_dat = get_dat.multiply(1e6 / col_sums, axis=1)

    # Re-label columns to include the sample name and qp id
    new_cols = [f"{sname}-{qp}" for sname, qp in zip(samp_ids['sample'].values, samp_ids['qp_well_fq'].values)]
    # If number of cols mismatch (defensive), keep original names
    if len(new_cols) == len(get_dat.columns):
        get_dat.columns = new_cols

    get_dat = get_dat.sort_index()
    return get_dat


def plt_ht_map(genes: pd.DataFrame, counts: pd.DataFrame, c_label: str, max_cts: float,
               figsize: Optional[tuple] = None, cmap: str = 'viridis', cbar_shrink: float = 0.5):
    """Plot a heatmap of selected genes across provided count table.

    Args:
        genes: DataFrame with at least a `gene_id` column and optionally `name` for labels.
        counts: DataFrame indexed by `gene_id` with sample/count columns.
        c_label: String label for counts (used in colorbar label).
        max_cts: Maximum value (on plotted scale) to use for vmin/vmax for color normalization.
        figsize: Optional matplotlib figsize tuple.
        cmap: Colormap name.
        cbar_shrink: Fraction to shrink colorbar.

    Returns:
        (fig, ax) tuple for the created plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Select rows using gene ids present in the counts index
    gene_ids = genes['gene_id'].values
    look_at_genes = counts.loc[gene_ids].copy()

    # Plot log10(counts + 1)
    sns.heatmap(np.log10(look_at_genes + 1), cmap=cmap,
                cbar_kws={'label': f'log10({c_label}+1)', 'shrink': cbar_shrink},
                vmin=0, vmax=max_cts, ax=ax)

    # Y tick labels: use provided name if present, fall back to gene_id
    if 'name' in genes.columns:
        y_labels = genes['name'].values
    else:
        y_labels = genes['gene_id'].astype(str).values

    ax.set_yticks(np.arange(len(genes)) + 0.5)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xticks(np.arange(len(look_at_genes.columns)) + 0.5)
    ax.set_xticklabels(look_at_genes.columns, rotation=90, fontsize=6)
    return fig, ax


def names_for_genes_pvals(dge_table: pd.DataFrame, genes_of_int: pd.DataFrame) -> pd.DataFrame:
    """Map gene IDs in a DGE table to friendly names and return their p-values.

    Args:
        dge_table: DataFrame produced by DE analysis; must contain a gene id column (often 'Unnamed: 0') and 'PValue'.
        genes_of_int: DataFrame with columns 'gene_id' and 'name'.

    Returns:
        DataFrame with columns ['gene_names', 'PValue'] for overlapping genes.
    """
    # Try common patterns for gene id column
    possible_gene_cols = [c for c in dge_table.columns if 'gene' in c.lower()]
    if not possible_gene_cols:
        raise KeyError('Could not find gene ID column in dge_table')
    gene_col = possible_gene_cols[0]

    # Prepare a mapping from gene_id to name
    mapping = genes_of_int.set_index('gene_id')['name'].to_dict()

    # Filter rows present in mapping
    mask = dge_table[gene_col].isin(mapping.keys())
    sub = dge_table.loc[mask, [gene_col, 'PValue']].copy()
    sub['gene_names'] = sub[gene_col].map(mapping)
    return sub[['gene_names', 'PValue']].reset_index(drop=True)
