"""Figures for hemibrain connectivity."""

import navis.interfaces.neuprint as neu
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

ER_INPUT_TYPES = ["EPG", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)", "EL"]


def plot_er_inputs_with_marginal(conn_df, weight_col='relative_weight', input_types=ER_INPUT_TYPES):
    """Heatmap of connections onto ER neurons with a marginal plot of the mean weight per input type.

    The heatmap shows the weight of connections from each presynaptic type (rows) onto each ER
    neuron (columns); the marginal plot to its right shows the mean weight per presynaptic type.
    Types without connections in `conn_df` (e.g. PEG) are shown as zero.

    Args:
        conn_df: DataFrame of connections (e.g. from ``neuprint.fetch_adjacencies``) with columns
            'bodyId_post', 'type_pre' / 'type_post' and `weight_col`.
        weight_col: Column holding the connection weight.
        input_types: Presynaptic types to show, in order, top to bottom.

    Returns:
        Tuple of (figure, (heatmap_ax, marginal_ax, cbar_ax)).
    """
    fig = plt.figure(figsize=(12, 2))
    gs = GridSpec(1, 6, figure=fig)
    heatmap_ax = fig.add_subplot(gs[0, 0:4])
    marginal_ax = fig.add_subplot(gs[0, 4], sharey=heatmap_ax)
    cbar_ax = fig.add_subplot(gs[0, 5])

    conn_matrix_df = neu.connection_table_to_matrix(conn_df, weight_col=weight_col, sort_by='type', group_cols='type')
    conn_matrix_df.loc['PEG'] = 0
    conn_matrix_df = conn_matrix_df.loc[input_types]

    sns.heatmap(conn_matrix_df, ax=heatmap_ax, cmap='Blues', annot=False, cbar_kws={'label': 'Weight'},
                cbar_ax=cbar_ax)

    mu = conn_matrix_df.mean(axis=1)
    marginal_ax.barh(np.arange(len(mu.index)) + .5, mu, color=plt.cm.Blues(.7))
    heatmap_ax.set_yticks(np.arange(len(mu.index)) + .5, input_types)
    heatmap_ax.set_ylim(len(mu.index))
    marginal_ax.set_ylabel('')
    marginal_ax.spines['right'].set_visible(False)
    marginal_ax.spines['top'].set_visible(False)
    return fig, (heatmap_ax, marginal_ax, cbar_ax)
