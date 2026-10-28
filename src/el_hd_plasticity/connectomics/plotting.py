"""Figures for hemibrain connectivity."""

import navis
import networkx as nx
import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from .connectivity import er_input_matrix

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

    conn_matrix_df = er_input_matrix(conn_df, weight_col, input_types)

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


# ----------------------------------------------------------------------------
# Neurons and synapses in the EB frame
# ----------------------------------------------------------------------------
def clean_axes(ax):
    """Remove ticks, labels, spines and grid of `ax`, as in the anatomy panels."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)


def plot_neurons_2d(items, colors=None, view=('x', 'y'), ax=None, figsize=(5, 5), linewidth=1.2, alpha=1.,
                    radius=True, **kwargs):
    """Project neurons and ROI meshes onto a plane with :func:`navis.plot2d` and clean up the axes.

    Args:
        items: navis neurons / volumes (e.g. a skeleton from :meth:`EBCoordinateSystem.fetch_skeleton`, an ROI mesh from
            :meth:`EBCoordinateSystem.fetch_roi`).
        colors: One color per item, or a single color.
        view: The two axes of the projection, e.g. ``('x', 'z')``.
        ax: Axes to draw on; a new figure of size `figsize` if None.
        linewidth, alpha, radius: Passed to :func:`navis.plot2d` (`radius` draws neurites with their radius).
        **kwargs: Further arguments of :func:`navis.plot2d` (e.g. ``shade_by``, ``smin``, ``smax``).

    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if colors is not None:
        kwargs['color'] = tuple(colors) if isinstance(colors, (list, tuple)) and not np.isscalar(colors[0]) else colors
    navis.plot2d(items, method='2d', connectors=False, linewidth=linewidth, view=view, alpha=alpha, radius=radius, ax=ax,
                 **kwargs)
    clean_axes(ax)
    return ax.figure, ax


def plot_instance_example(skeleton, syns, instances, colors=('black', 'orangered', 'dodgerblue'), view=('x', 'y'), margin=400,
                          bar_um=5, skeleton_color=None, ax=None, figsize=(5, 3), legend=False):
    """An axon segment with the synapses of a few EPG instances, zoomed on those synapses.

    Args:
        skeleton: The ER neuron's skeleton in the EB frame (:meth:`EBCoordinateSystem.fetch_skeleton`).
        syns: Synapses of the neuron with ``type`` (instance) and ``x_eb``, ``y_eb``, ``z_eb`` columns.
        instances: Instance labels to draw, one color each.
        view: The two axes of the projection.
        margin: Voxels around the synapses shown.
        bar_um: Length of the scale bar in microns.

    Returns:
        Tuple of (figure, axes).
    """
    skeleton_color = plt.cm.cool(.7) if skeleton_color is None else skeleton_color
    fig, ax = plot_neurons_2d([skeleton], colors=skeleton_color, view=view, linewidth=1., ax=ax, figsize=figsize)
    selected = syns[syns['type'].isin(instances)]
    xs, ys = (f'{axis}_eb' for axis in view)
    for instance, color in zip(instances, colors[:len(instances)], strict=True):
        syn = selected[selected['type'] == instance]
        ax.scatter(syn[xs], syn[ys], s=40, color=color, zorder=100, label=f'{instance} (n={len(syn)})')
    ax.set_xlim(selected[xs].min() - margin-500, selected[xs].max() + margin)
    # ax.set_ylim(selected[ys].min() - margin, selected[ys].max() + margin)
    ax.set_ylim([-4200,100])
    x0, y0 = ax.get_xlim()[0] + 100, ax.get_ylim()[0] + 100
    ax.plot([x0, x0], [y0, y0 + bar_um * 1000 / 8], linewidth=5, color='black')
    
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    return fig, ax


def phase_colormap(n=18):
    """Function mapping a ring phase in (-pi, pi] to a color of a circular palette."""
    cmap = sns.color_palette('hls', n, as_cmap=True)
    return lambda phase: cmap((phase + np.pi) / 2 / np.pi)


def plot_grouped_synapses(skeleton, syns, colors, group_col='type', view=('x', 'y'), bar_um=10,
                          skeleton_color=None, items=(), item_color=None, ax=None, figsize=(5, 5),
                          s=20, legend=True, xlim=None, ylim=None, linewidth=1, **kwargs):
    """A neuron with every synapse colored by the group it belongs to.

    Args:
        skeleton: The neuron's skeleton in the EB frame (:meth:`EBCoordinateSystem.fetch_skeleton`).
        syns: Its synapses with ``x_eb``, ``y_eb``, ``z_eb`` and `group_col` columns.
        colors: Mapping from group label to color (e.g. the ring phase of each group through
            :func:`phase_colormap`), in the order the groups are drawn and listed; groups missing from
            it are not drawn.
        group_col: Column of `syns` holding the group label, e.g. the EPG column of a synapse.
        view: The two axes of the projection.
        bar_um: Length of the scale bar in microns.
        skeleton_color: Color of the neuron; the color it has in the other anatomy panels by default.
        items: Further navis items drawn under the neuron, e.g. the EB mesh.
        item_color: Color of `items`.
        s: Size of the synapse markers.
        legend: Label each group with its name and synapse count.
        xlim, ylim: Axis limits, to zoom on a section of the neuron; the whole neuron if None. Synapses
            outside them are left out, so the legend counts the synapses actually in the view.
        **kwargs: Further arguments of :func:`plot_neurons_2d`, e.g. ``shade_by``, ``smin``, ``smax``.

    Returns:
        Tuple of (figure, axes).
    """
    skeleton_color = plt.cm.cool(.7) if skeleton_color is None else skeleton_color
    item_color = plt.cm.Greys(.3) if item_color is None else item_color
    # navis colors the neurons of `items` before its volumes, whatever order they are passed in
    fig, ax = plot_neurons_2d([skeleton, *items], colors=[skeleton_color] + [item_color] * len(items),
                              view=view, linewidth=linewidth, ax=ax, figsize=figsize, **kwargs)

    xs, ys = (f'{axis}_eb' for axis in view)
    for axis, limits in ((xs, xlim), (ys, ylim)):      # what falls outside the view is invisible anyway, and
        if limits is not None:                         # dropping it keeps the legend's counts to what is drawn
            syns = syns[syns[axis].between(*sorted(limits))]
    present = set(syns[group_col].dropna())
    for group, color in colors.items():
        if group not in present:
            continue
        syn = syns[syns[group_col] == group]
        ax.scatter(syn[xs], syn[ys], s=s, color=color, zorder=100, label=f'{group} (n={len(syn)})')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    x0, y0 = ax.get_xlim()[0] + 100, ax.get_ylim()[0] + 100
    ax.plot([x0, x0], [y0, y0 + bar_um * 1000 / 8], linewidth=5, color='black')
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    return fig, ax


def xsec_outline(ebcoords):
    """Vertices of the convex hull of the EB cross-section (xsec_x, xsec_y coordinates of its mesh vertices)."""
    r, p = ebcoords.eb_vertices_xsec['radius'], ebcoords.eb_vertices_xsec['phase']
    points = np.column_stack((r * np.cos(p), r * np.sin(p)))
    return points[sp.spatial.ConvexHull(points).vertices]


def _plot_density(datasets, x, y, outlines, figsize, fill, share_axes, equal_aspect=False):
    n_panels = 1 if share_axes else len(datasets)
    fig, axs = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    for i, (table, cmap, alpha) in enumerate(datasets):
        sns.kdeplot(data=table, x=x, y=y, fill=fill, ax=axs[0, 0 if share_axes else i], cmap=cmap, alpha=alpha)
    for ax in axs.flat:
        for outline in outlines:
            ax.plot(*np.vstack([outline, outline[:1]]).T, color='black')
        clean_axes(ax)
        if equal_aspect:
            ax.set_aspect('equal')
    return fig, axs


def plot_xsec_density(datasets, ebcoords, figsize=(5, 5), fill=False, share_axes=True):
    """Kernel density of synapse positions in the EB cross-section, drawn inside the cross-section outline.

    Args:
        datasets: List of (synapse table with xsec_x / xsec_y, colormap, alpha) tuples.
        ebcoords: :class:`~el_hd_plasticity.connectomics.connectomics.EBCoordinateSystem`.
        figsize: Size of the figure (with ``share_axes=False`` one panel per dataset).
        fill: Fill the density contours.
        share_axes: Draw all datasets in one panel; otherwise one panel each.

    Returns:
        Tuple of (figure, array of axes).
    """
    return _plot_density(datasets, 'xsec_x', 'xsec_y', [xsec_outline(ebcoords)], figsize, fill, share_axes)


def plot_eb_density(datasets, ebcoords, figsize=(5, 5), fill=False, share_axes=True):
    """Kernel density of synapse positions in the x, y plane of the EB frame, drawn inside the outline of the ring.

    Args:
        datasets: List of (synapse table with x_eb / y_eb, colormap, alpha) tuples, see
            :func:`~el_hd_plasticity.connectomics.connectomics.add_eb_coords`.
        ebcoords: :class:`~el_hd_plasticity.connectomics.connectomics.EBCoordinateSystem`.
        figsize: Size of the figure (with ``share_axes=False`` one panel per dataset).
        fill: Fill the density contours.
        share_axes: Draw all datasets in one panel; otherwise one panel each.

    Returns:
        Tuple of (figure, array of axes).
    """
    return _plot_density(datasets, 'x_eb', 'y_eb', ebcoords.ring_outline(), figsize, fill, share_axes, equal_aspect=True)


# ----------------------------------------------------------------------------
# Connectivity graph and synapse distances
# ----------------------------------------------------------------------------
def plot_connectivity_graph(graph, colors, pos, edge_attr='weight', arrow_scale=.00015, figsize=(5, 5)):
    """Draw a type-level connectivity graph: node size by label, arrow width by `edge_attr`, edge labels with the weight.

    Args:
        graph: Directed graph from :func:`~el_hd_plasticity.connectomics.connectivity.connectivity_graph`.
        colors: Dict node -> color (edges take the color of their source).
        pos: Dict node -> (x, y) position.
        edge_attr: Edge attribute setting the arrow size and label.
        arrow_scale: Arrow head size per unit of `edge_attr`.
        figsize: Size of the figure.

    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color=[colors[n] for n in graph.nodes()], alpha=1., ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black', ax=ax,
                            labels={n: f"{n} \n N={d['n_neurons']}" for n, d in graph.nodes(data=True)})
    arrows = [f"simple, head_width={d[edge_attr] * arrow_scale:.2f}, head_length={d[edge_attr] * arrow_scale:.2f}, "
              f"tail_width={d[edge_attr] * arrow_scale / 2:.2f}" for _, _, d in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, connectionstyle='arc3,rad=0.12', edge_color=[colors[u] for u, _, _ in graph.edges(data=True)],
                           node_size=[d['n_neurons'] * 30 + 200 for _, d in graph.nodes(data=True)], ax=ax,
                           arrowstyle=arrows, alpha=1.)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): f"{d[edge_attr]:.0f}" for u, v, d in graph.edges(data=True)},
                                 font_size=8, connectionstyle='arc3,rad=0.12', ax=ax)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    return fig, ax


def _distance_axis(ax, xlim=None):
    ax.set_xlabel('approx. cable distance (microns)')
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xlim is not None:
        ax.set_xlim(xlim)


def plot_distance_histogram(df, hue_order, palette, bins=None, ax=None, figsize=(2, 4)):
    """Distribution of cable distances (``distances`` column of a distance table) by ``source``, as filled polygons.

    Args:
        df: Table with ``source`` and ``distances`` (microns), see
            :func:`~el_hd_plasticity.connectomics.synapse_distance.load_distance_table`.
        hue_order: Sources to draw, in order.
        palette: One color per source.
        bins: Histogram bins (microns); seaborn's default if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.histplot(data=df.loc[df['source'].isin(hue_order)], x='distances', hue='source', hue_order=hue_order,
                 palette=palette, fill=True, common_norm=False, cumulative=False, element='poly', stat='probability',
                 bins=bins if bins is not None else 'auto', ax=ax)
    _distance_axis(ax)
    return ax.figure, ax


def plot_family_distance_histogram(df, group, source='el', bins=None, color=None, ax=None, figsize=(2, 4), xlims=None):
    """Distribution of cable distances to `source` of one family of ER neuron types (``er`` in `group`)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(0, 16, 80) if bins is None else bins
    xlims = [0, 16] if xlims is None else xlims
    sns.histplot(data=df.loc[(df['source'] == source) & df['er'].isin(group)], x='distances', bins=bins, ax=ax, fill=True,
                 stat='probability', element='poly', common_norm=False, cumulative=False,
                 color=plt.cm.Greens(.8) if color is None else color)
    _distance_axis(ax, xlim=xlims)
    ax.set_title(str(group))
    ax.set_ylim([0, .15])
    return ax.figure, ax


def plot_cumulative_distances(df, groups, sources=('el', 'exr2'), bins=None, figsize=(4, 4), xlims = None):
    """Cumulative distance distributions of every ER family: one panel per source (solid for the first, dashed otherwise)."""
    bins = np.linspace(0, 16, 80) if bins is None else bins
    xlims = [0, 16] if xlims is None else xlims
    fig, axs = plt.subplots(1, len(sources), figsize=figsize)
    for ax, source, linestyle in zip(np.atleast_1d(axs), sources, ('-', '--', ':'), strict=False):
        for group in groups:
            sns.histplot(data=df.loc[(df['source'] == source) & df['er'].isin(group)], x='distances', bins=bins, ax=ax,
                         fill=False, stat='probability', element='poly', common_norm=False, cumulative=True,
                         linestyle=linestyle, linewidth=1 if linestyle != '-' else None)
        _distance_axis(ax, xlim=xlims)
    fig.legend([str(g) for g in groups], bbox_to_anchor=(1.05, 1), loc='lower left', borderaxespad=0.)
    return fig, axs


def plot_distance_ecdf(df, sources, labels=None, colors=None, xlim=(0, 16), ax=None, figsize=(2.5, 2.5)):
    """Cumulative distance distributions of several sources in one panel, over all the distances (not just `xlim`).

    Unlike :func:`plot_cumulative_distances`, which bins the distances and so renormalises over `xlim`, each curve here
    is the empirical CDF of every distance of that source: a curve that ends below 1 at the right edge has that
    fraction of its synapses beyond it.

    Args:
        df: Table with ``source`` and ``distances`` (microns).
        sources: Sources to draw, in order.
        labels: Legend label of each source (default: the source itself).
        colors: One color per source.
        xlim: Limits of the distance axis.

    Returns:
        Tuple of (figure, axes).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    labels = labels or {}
    colors = colors if colors is not None else [None] * len(sources)
    for source, color in zip(sources, colors, strict=True):
        distances = np.sort(df.loc[df['source'] == source, 'distances'].dropna().to_numpy())
        ax.plot(distances, np.arange(1, len(distances) + 1) / len(distances), color=color,
                label=labels.get(source, source))
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    _distance_axis(ax, xlim=xlim)
    return ax.figure, ax


def plot_distance_stack(df, group, sources, labels=None, colors=None, bins=None, figsize=(3.2, 6), ecdf_height=3.2):
    """One wide histogram per source, stacked above a square ECDF of all of them, for one family of ER types.

    Args:
        df: Table with ``er``, ``source`` and ``distances`` (microns).
        group: ER types to pool in this figure.
        sources: Sources to draw, one histogram each, in order.
        labels: Title of each source's panel and its legend entry (default: the source itself).
        colors: One color per source.
        bins: Histogram bins (microns); the 0-16 default of :func:`plot_family_distance_histogram` if None.
        ecdf_height: Height of the ECDF panel, relative to one histogram.

    Returns:
        Tuple of (figure, list of axes: one per source, then the ECDF).
    """
    labels = labels or {}
    colors = colors if colors is not None else [None] * len(sources)
    sub = df.loc[df['er'].isin(group)]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(sources) + 1, 1, height_ratios=[1] * len(sources) + [ecdf_height], figure=fig, hspace=.9)
    axs = []
    for position, (source, color) in enumerate(zip(sources, colors, strict=True)):
        ax = fig.add_subplot(gs[position])
        plot_family_distance_histogram(sub, group, source=source, bins=bins, color=color, ax=ax)
        ax.set_title(labels.get(source, source), fontsize=6)
        ax.set_xlabel('')
        axs.append(ax)
    axs[-1].set_xlabel('approx. cable distance (microns)', fontsize=7)

    ax = fig.add_subplot(gs[-1])
    plot_distance_ecdf(sub, sources, labels=labels, colors=colors, ax=ax)
    ax.legend(fontsize=5, loc='upper left', bbox_to_anchor=(0, -.42), frameon=False)    # below the panel, clear of the curves and the x label
    axs.append(ax)
    return fig, axs


def plot_partner_counts(table, column, ylabel, groups, ymax=20, figsize=(3, 3)):
    """Number of partners of every neuron (points), colored by ER family; ``table`` from
    :func:`~el_hd_plasticity.connectomics.synapse_distance.partner_count_table`."""
    fig, ax = plt.subplots(figsize=figsize)
    data = table.assign(plot_x=0)
    sns.stripplot(data=data, x='plot_x', y=column, hue='group', hue_order=[str(g) for g in groups], dodge=True, ax=ax, s=8)
    sns.move_legend(ax, 'upper right', bbox_to_anchor=(3, 1))
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_yticks(np.arange(0, ymax, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def plot_connectivity_comparison(table, edge_attr='relative_weight', palette=None, figsize=None):
    """Connection strength between types in several datasets: one group of bars per connection, one bar per dataset.

    Args:
        table: Stacked type connectivity tables (type_pre, type_post, `edge_attr`, ``dataset``), see
            :func:`~el_hd_plasticity.connectomics.connectivity.compare_datasets`.
        edge_attr: Column plotted on the y axis (weight, relative_weight, pcnt_output or nrn_weight).
        palette: Dict dataset -> color (default: seaborn's).
        figsize: Size of the figure (default: wide enough for the connection labels, which are rotated).
    """
    data = table.assign(connection=table['type_pre'] + ' \u2192 ' + table['type_post'])
    n_connections = data['connection'].nunique()
    fig, ax = plt.subplots(figsize=figsize or (max(4., .9 * n_connections + 1.5), 3.5))
    sns.barplot(data=data, x='connection', y=edge_attr, hue='dataset', palette=palette, ax=ax)
    ax.tick_params(axis='x', labelrotation=45)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    ax.set_xlabel('')
    ax.set_ylabel(edge_attr.replace('_', ' '))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    return fig, ax


def plot_input_comparison(marginals, palette=None, ylabel='Mean weight onto ER types', figsize=(5, 3)):
    """Mean input of each presynaptic type onto ER neurons in several datasets (output of
    :func:`~el_hd_plasticity.connectomics.connectivity.er_input_marginals`)."""
    data = marginals.rename_axis('input type').reset_index().melt(id_vars='input type', var_name='dataset', value_name='weight')
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x='input type', y='weight', hue='dataset', palette=palette, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    return fig, ax
