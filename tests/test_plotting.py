"""Every figure function renders; layout facts that the notebooks depend on are checked."""
import pytest
from conftest import FakePP
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as tp

plotting = tp.plotting


def sessions(*keys, n=3000):
    d = {'fly': 'flyX'}
    for i, k in enumerate(keys):
        d[k] = tp.session.GetTS(FakePP(seed=20 + i, n=n))
    return d


def figures(result):
    found = []
    def rec(o):
        if isinstance(o, plt.Figure):
            found.append(o)
        elif isinstance(o, (tuple, list)):
            [rec(x) for x in o]
    rec(result)
    return found


def render(result):
    figs = figures(result)
    assert figs
    for fig in figs:
        fig.canvas.draw()
    return figs


CL_DARK = ('closed_loop', 'dark')
EPOCHS = ('closed_loop 1', 'dark', 'closed_loop 2')
SCANS = ('baseline', 'post_0deg', 'post_180deg')


@pytest.mark.parametrize('module', [tp.tbh_rnai, tp.oa_receptor_ko])
@pytest.mark.parametrize('fn', ['plot_sess_heatmaps', 'plot_sess_histograms', 'plot_sess_heatmaps_w_hist'])
def test_closed_loop_vs_dark_figures(module, fn):
    render(getattr(module, fn)(sessions(*CL_DARK)))


@pytest.mark.parametrize('fn', ['plot_transpose_heatmaps', 'plot_sess_heatmaps', 'plot_sess_histograms',
                                'plot_sess_heatmaps_w_hist'])
def test_epg_across_dark_figures(fn):
    render(getattr(tp.epg_across_dark, fn)(sessions(*EPOCHS)))


@pytest.mark.parametrize('fn', ['plot_sess_heatmaps', 'plot_sess_histograms', 'plot_sess_heatmaps_w_hist'])
def test_remapping_figures(fn):
    render(getattr(tp.remapping, fn)(sessions(*SCANS)))


def test_grab_sensor_figures():
    d = sessions('closed loop')
    for fn in ('plot_sess_heatmaps', 'plot_transpose_heatmaps', 'plot_sess_histograms', 'plot_sess_heatmaps_w_hist'):
        render(getattr(tp.grab_sensors, fn)(d))


def test_shibire_figure():
    render(tp.shibirets.plot_sess_heatmaps_w_hist(sessions('baseline', 'post_heat')))


def test_dual_imaging_figures():
    ts = tp.session.GetTS(FakePP(seed=5), channels=[0, 1])
    for fn in ('plot_sess_heatmaps', 'plot_pva_diff_histograms', 'plot_sess_histograms',
               'plot_transpose_heatmaps', 'plot_sess_heatmaps_w_hist'):
        render(getattr(tp.dual_imaging, fn)(ts, 'flyD', 'sess1'))


def test_heatmap_grid_layout_and_titles():
    fig, ax = tp.epg_across_dark.plot_sess_heatmaps(sessions(*EPOCHS))
    assert ax.shape == (3, 2)
    assert [ax[i, 0].get_title() for i in range(3)] == list(EPOCHS)
    assert fig._suptitle.get_text() == 'flyX'
    fig, ax = tp.epg_across_dark.plot_transpose_heatmaps(sessions(*EPOCHS))
    assert ax.shape == (2, 3)


def test_heatmap_with_hist_rows_and_slots():
    _, (heat, hist, cbar) = tp.remapping.plot_sess_heatmaps_w_hist(sessions(*SCANS), sess_order=(2, 0, 1))
    assert len(heat) == len(hist) == len(cbar) == 3
    # baseline was placed in row 2
    assert heat[2].get_title() == 'baseline' and heat[0].get_title() == 'post_0deg'


def test_hist_title_reports_vector_length_or_variance():
    d = sessions(*CL_DARK)
    _, (_, hist, _) = tp.tbh_rnai.plot_sess_heatmaps_w_hist(d)
    assert hist[0].get_title().startswith('vec length=')
    traces = [plotting.Trace.from_ts('EPGs', tp.session.GetTS(FakePP(seed=5), channels=[0, 1]), channel=0)]
    _, (_, hist, _) = plotting.plot_heatmaps_with_offset_hist(traces, 't', [8, 2], hist_title='variance')
    assert hist[0].get_title().startswith('variance=')


def test_time_window_limits_the_heatmap_extent():
    d = sessions(*CL_DARK)
    _, (heat, _, _) = tp.tbh_rnai.plot_sess_heatmaps_w_hist(d, twindow=(10, 20))
    n_cols = heat[0].images[0].get_array().shape[1]
    t = d['closed_loop'].time
    assert n_cols == int(((t >= 10) & (t <= 20)).sum())


def test_tick_clipping_drops_times_past_end_of_data():
    d = sessions(*CL_DARK, n=1000)                     # 50 s of data
    _, ax = tp.tbh_rnai.plot_sess_heatmaps(d)         # unclipped: legacy behaviour keeps every tick
    assert len(ax[0, 0].get_xticks()) == 6
    _, (heat, _, _) = tp.tbh_rnai.plot_sess_heatmaps_w_hist(d)   # clipped
    assert len(heat[0].get_xticks()) == 1               # only t=0 lies inside the 50 s of data


def test_outline_flag_changes_histogram_edges():
    traces = [plotting.Trace.from_ts('a', tp.session.GetTS(FakePP(seed=1)), color='black')]
    (_, ax_a), _ = plotting.plot_offset_histograms(traces, outline=True)
    (_, ax_b), _ = plotting.plot_offset_histograms(traces, outline=False)
    def edge(ax):
        return ax.collections[0].get_edgecolor()

    assert len(edge(ax_a)) == 1 and len(edge(ax_b)) == 0


def test_hatched_trace_is_drawn_dashed_in_polar_plot():
    ts = tp.session.GetTS(FakePP(seed=1))
    traces = [plotting.Trace.from_ts('a', ts, color='black'), plotting.Trace.from_ts('b', ts, color='black', hatch='/')]
    _, (_, ax_polar) = plotting.plot_offset_histograms(traces)
    assert [line.get_linestyle() for line in ax_polar.lines] == ['-', '--']
