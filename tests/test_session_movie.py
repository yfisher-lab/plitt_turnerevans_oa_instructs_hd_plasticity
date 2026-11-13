"""The session movie draws every moment of a session consistently and writes a playable file."""
import numpy as np
from conftest import FakePP
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as tp

movie_mod = tp.session_movie


def make_movie(**kwargs):
    ts = tp.session.GetTS(FakePP(seed=3, n=600))
    return ts, movie_mod.ClosedLoopMovie(ts, **kwargs)


def test_ellipsoid_body_shows_the_activity_of_the_current_sample():
    ts, movie = make_movie()
    t = float(np.asarray(ts.time)[300])
    movie.draw(t)
    i = movie.index_at(t)
    expected = [movie.cmap(movie.norm(v)) for v in ts.dff[:, i]]
    assert np.allclose([w.get_facecolor() for w in movie.wedges], expected)
    plt.close(movie.fig)


def test_heatmap_fills_for_one_window_then_scrolls_and_hides_the_future():
    ts, movie = make_movie(window=20.)
    time = np.asarray(ts.time)
    movie.draw(time[0] + 5)                               # filling: fixed axis, nothing drawn after now
    i = movie.index_at(time[0] + 5)
    assert np.allclose(movie.heat_ax.get_xlim(), (time[0], time[0] + 20))
    assert movie.heat_images[0].get_array().shape == (ts.dff.shape[0], i + 1)
    assert np.isclose(movie.heat_images[0].get_extent()[1], time[i] + ts.dt / 2)
    assert np.nanmax(movie.heading_line.get_xdata()) == time[i]
    movie.draw(time[0] + 25)                              # scrolling: the last window, now at the right edge
    i = movie.index_at(time[0] + 25)
    assert np.allclose(movie.heat_ax.get_xlim(), (time[i] - 20, time[i]))
    plt.close(movie.fig)


def test_display_angles_are_smoothed_but_statistics_are_not():
    ts, movie = make_movie(angle_sigma=1.)
    assert np.abs(np.diff(np.unwrap(movie.phi))).mean() < np.abs(np.diff(np.unwrap(ts.phi))).mean()
    # the histogram is of the session's own offsets, not the smoothed ones
    expected, _ = tp.utilities.normalized_histogram(ts.offset, movie.bins)
    assert np.allclose(movie.running_hist[-1], expected)
    # no smoothing leaves the angles as they are
    _, raw = make_movie(angle_sigma=0)
    assert np.allclose(raw.phi, ts.phi) and np.allclose(raw.heading, ts.heading)
    plt.close(movie.fig)
    plt.close(raw.fig)


def test_pva_arrow_length_follows_magnitude():
    ts, movie = make_movie()
    lengths = np.array([movie.pva_length(i) for i in range(len(movie.rho))])
    assert lengths.max() <= movie_mod.PVA_ARROW_MAX + 1e-12
    strong, weak = int(np.argmax(movie.rho)), int(np.argmin(movie.rho))
    assert lengths[strong] > lengths[weak]
    assert np.isclose(lengths[weak], movie_mod.PVA_ARROW_MAX * movie.rho[weak] / movie.pva_scale)
    plt.close(movie.fig)


def test_schematic_images_are_used_when_given(tmp_path):
    from PIL import Image
    path = tmp_path / 'icon.png'
    Image.new('RGBA', (40, 30), (0, 0, 0, 255)).save(path)
    _, movie = make_movie(fly_image=path, rig_image=path)
    assert len(movie.arena_ax.images) == 1
    movie.draw(1.)
    movie.fig.canvas.draw()
    plt.close(movie.fig)


def test_running_offset_histogram_and_vector_length():
    ts, movie = make_movie()
    i = 250
    expected, _ = tp.utilities.normalized_histogram(ts.offset[:i + 1], movie.bins)
    assert np.allclose(movie.running_hist[i], expected)
    assert np.isclose(movie.running_veclen[i], np.abs(np.mean(np.exp(1j * ts.offset[:i + 1]))))
    plt.close(movie.fig)


def test_eb_angles_follow_the_heatmap_axis():
    # heatmap angle 0 (its top row, theta = -pi) at the top of the ring, increasing counterclockwise
    assert np.isclose(movie_mod._eb_angle(-np.pi) % (2 * np.pi), np.pi / 2)
    assert np.isclose(movie_mod._eb_angle(-np.pi / 2) % (2 * np.pi), np.pi)


def test_save_writes_an_mp4(tmp_path):
    ts, movie = make_movie(fps=4, t_range=(0, 2), speed=2.)
    assert len(movie.frame_times()) == 4
    out = movie.save(tmp_path / 'clip.mp4', progress=False)
    assert out.exists() and out.stat().st_size > 1000
    plt.close(movie.fig)


def test_visual_center_weights_dark_pixels_and_ignores_transparent_ones():
    image = np.zeros((10, 20, 4))
    image[2:4, 12:16] = [0, 0, 0, 1]                      # a dark, opaque block
    image[6:9, 1:5] = [1, 1, 1, 1]                        # an opaque white block: no visual weight
    cx, cy = movie_mod.visual_center(image)
    assert np.isclose(cx, 13.5) and np.isclose(cy, 2.5)


def test_fly_rotates_about_its_visual_center(tmp_path):
    from PIL import Image
    image = np.zeros((30, 40, 4), dtype=np.uint8)
    image[5:15, 25:35] = [0, 0, 0, 255]                   # all of the fly's weight sits off the image centre
    path = tmp_path / 'fly.png'
    Image.fromarray(image).save(path)
    _, movie = make_movie(fly_image=path)
    left, right, bottom, top = movie.arena_ax.images[0].get_extent()
    cx, cy = movie_mod.visual_center(image)
    # the visual centre of mass is drawn at the origin, which the rotation leaves in place
    x0 = left + (cx + .5) / image.shape[1] * (right - left)
    y0 = top - (cy + .5) / image.shape[0] * (top - bottom)
    assert np.isclose(x0, 0) and np.isclose(y0, 0)
    plt.close(movie.fig)


def test_pva_arrow_scale_is_fixed_not_fitted_to_the_session():
    ts, movie = make_movie()
    _, smoother = make_movie(angle_sigma=2.)              # a different distribution of displayed magnitudes
    # the same scale whatever the session, so the arrow's length is the magnitude itself
    assert movie.pva_scale == smoother.pva_scale == movie_mod.PVA_SCALE
    for i in (int(np.argmin(movie.rho)), len(movie.rho) // 2):
        expected = movie_mod.PVA_ARROW_MAX * min(movie.rho[i] / movie_mod.PVA_SCALE, 1)
        assert np.isclose(movie.pva_length(i), expected)
    plt.close(movie.fig)
    plt.close(smoother.fig)


def _appended_sequence():
    """Two fake sessions with a NaN gap between them, a colormap epoch starting in the second, a screen-off stretch."""
    from types import SimpleNamespace
    ts = tp.session.GetTS(FakePP(seed=4, n=600))
    dff, heading, phi = (np.array(x, dtype=float, copy=True) for x in (ts.dff, ts.heading, ts.phi))
    rho, offset = np.array(ts.rho, dtype=float, copy=True), np.array(ts.offset, dtype=float, copy=True)
    gap = slice(250, 300)
    for x in (heading, phi, rho, offset):
        x[gap] = np.nan
    dff[:, gap] = np.nan
    dff[:, 400:410] = np.nan                                     # masked stimulation frames
    time = np.arange(dff.shape[1]) * ts.dt
    seq = SimpleNamespace(dff=dff, heading=heading, phi=phi, rho=rho, offset=offset, time=time, dt=ts.dt)
    cue_visible = np.ones(len(time), dtype=bool)
    cue_visible[320:340] = False
    accumulate = np.isfinite(offset) & cue_visible & ~((time >= time[380]) & (time < time[450]))
    epochs = [movie_mod.ColorEpoch(0., 'Greys', 'black'), movie_mod.ColorEpoch(time[380], 'GnBu', 'blue')]
    return seq, dict(epochs=epochs, accumulate=accumulate, cue_visible=cue_visible,
                     segments=[(0, time[249], 'first'), (time[250], time[299], 'break', False),
                               (time[300], time[-1], 'second')])


def test_appended_sessions_switch_colormap_and_freeze_the_earlier_histogram():
    seq, kwargs = _appended_sequence()
    movie = movie_mod.ClosedLoopMovie(seq, fps=4, speed=1., **kwargs)
    assert len(movie.heat_images) == 2
    # the first epoch's histogram stops growing once the second epoch begins
    first = movie.epoch_counts[0]
    assert np.allclose(first[379], first[-1])
    assert movie.epoch_counts[1][379].sum() == 0 and movie.epoch_counts[1][-1].sum() > 0
    # nothing accumulates where the cue is off or a pairing is under way
    assert np.allclose(movie.epoch_counts[1][449], 0)
    movie.draw(seq.time[500])
    assert movie.colorbar_image.get_cmap().name.startswith('GnBu')
    assert [image.get_visible() for image in movie.heat_images] == [True, True]
    assert movie.clock.get_text().startswith('second')
    plt.close(movie.fig)


def test_gaps_and_masked_frames_are_blank_and_the_screen_goes_dark():
    seq, kwargs = _appended_sequence()
    movie = movie_mod.ClosedLoopMovie(seq, fps=4, speed=1., **kwargs)
    movie.draw(seq.time[270])                                    # in the gap
    assert all(np.allclose(w.get_facecolor(), (1, 1, 1, 1)) for w in movie.wedges)
    assert not movie.pva_arrow.get_visible() and not movie.eb_cue.get_visible()
    assert movie.clock.get_text() == 'break'                     # no clock: the break's real length is unknown
    movie.draw(seq.time[330])                                    # screen off
    assert not movie.arena_cue.get_visible() and not movie.eb_cue.get_visible()
    assert np.allclose(movie.arena_ring.get_edgecolor()[:3], plt.matplotlib.colors.to_rgb(movie_mod.DARK_SCREEN_COLOR))
    # a video frame reaching into masked frames shows the latest imaged sample within it ...
    per_frame = int(round(movie.frame_dt / seq.dt))
    assert per_frame >= 2
    assert movie.index_at(seq.time[400 + per_frame // 2]) == 399
    # ... and one entirely within them shows that nothing was imaged
    assert not movie.valid[movie.index_at(seq.time[409])]
    plt.close(movie.fig)


def test_unsmoothed_samples_keep_their_jumps():
    seq, kwargs = _appended_sequence()
    raw = np.zeros(len(seq.time), dtype=bool)
    raw[380:450] = True
    movie = movie_mod.ClosedLoopMovie(seq, fps=4, speed=1., unsmoothed=raw, **kwargs)
    assert np.allclose(movie.heading[raw], seq.heading[raw], equal_nan=True)
    assert not np.allclose(movie.heading[:200], seq.heading[:200])
    plt.close(movie.fig)


def test_pva_is_not_drawn_where_hidden():
    seq, kwargs = _appended_sequence()
    hidden = np.zeros(len(seq.time), dtype=bool)
    hidden[380:450] = True                                       # e.g. the pairing, imaging windows included
    movie = movie_mod.ClosedLoopMovie(seq, fps=4, speed=1., pva_hidden=hidden, **kwargs)
    movie.draw(seq.time[420])
    assert not movie.pva_arrow.get_visible()
    movie.draw(seq.time[500])
    assert movie.pva_arrow.get_visible()
    plt.close(movie.fig)


def test_append_sessions_gives_each_session_its_own_epoch_and_histogram():
    sessions = [tp.session.GetTS(FakePP(seed=s, n=300)) for s in (7, 8, 9)]
    styles = [movie_mod.ColorEpoch(0., 'Greys', 'black', .5), movie_mod.ColorEpoch(0., 'Purples', 'purple', .5),
              movie_mod.ColorEpoch(0., 'Greys', 'black', 1., '/')]
    parts = [(label, ts, style, cue) for label, ts, style, cue in
             zip(('closed loop 1', 'dark', 'closed loop 2'), sessions, styles, (True, False, True), strict=True)]
    seq = movie_mod.append_sessions(parts, gap_s=2.)
    n_gap = int(round(2. / seq.dt))
    assert seq.dff.shape[1] == 900 + 2 * n_gap and np.isnan(seq.dff[:, 300:300 + n_gap]).all()
    assert [e.cmap for e in seq.epochs] == ['Greys', 'Purples', 'Greys'] and seq.epochs[2].hist_hatch == '/'
    assert not seq.cue_visible[300 + n_gap:600 + n_gap].any() and seq.cue_visible[:300].all()
    assert [label for *_, label in [s[:3] for s in seq.segments]] == ['closed loop 1', 'break', 'dark', 'break',
                                                                       'closed loop 2']
    movie = movie_mod.ClosedLoopMovie(seq, fps=4, speed=1., epochs=seq.epochs, accumulate=seq.accumulate,
                                      cue_visible=seq.cue_visible, trace_visible=np.ones(len(seq.time), bool),
                                      segments=seq.segments)
    # each session's histogram is that session's own offsets, and the dark one keeps its VR (ball) angle trace
    for e, ts in enumerate(sessions):
        expected, _ = tp.utilities.normalized_histogram(ts.offset, movie.bins)
        assert np.allclose(movie.epoch_counts[e][-1], expected)
    dark = 300 + n_gap + 50
    assert np.isfinite(movie.heading_traced[dark]) and np.isnan(movie.heading_shown[dark])
    movie.draw(seq.time[-1])
    hatched = [fill for fill in movie.hist_fills if fill.get_hatch()]
    assert len(movie.hist_fills) == 3 and len(hatched) == 1
    plt.close(movie.fig)


def test_a_turn_moves_the_fly_and_the_cue_opposite_ways():
    """`heading` is the cue's orientation relative to the fly: as it increases (the fly turning clockwise, the cue slipping
    counterclockwise), the fly turns clockwise on screen and the cue moves counterclockwise around the ring."""
    from types import SimpleNamespace
    n = 200
    heading = np.linspace(-1., 1., n)                        # the fly turning clockwise
    dff = np.random.default_rng(0).standard_normal((16, n))
    seq = SimpleNamespace(dff=dff, heading=heading, phi=heading, rho=np.ones(n), offset=np.zeros(n),
                          time=np.arange(n) * .1, dt=.1)
    movie = movie_mod.ClosedLoopMovie(seq, angle_sigma=0, fps=10, speed=1.)

    def fly_angle(i):
        movie.draw(seq.time[i])
        m = movie.fly_rotation.get_matrix()
        return np.arctan2(m[1, 0], m[0, 0])

    def cue_angle(i):
        movie.draw(seq.time[i])
        x, y = movie.eb_cue.get_data()
        return np.arctan2(np.mean(y), np.mean(x))

    assert fly_angle(150) < fly_angle(50)                    # the fly turns clockwise on screen
    assert cue_angle(150) > cue_angle(50)                    # the cue moves counterclockwise around the ring
    # a bar angle of 0 (heading -pi) still has the cue straight behind the fly
    seq.heading[:] = -np.pi
    still = movie_mod.ClosedLoopMovie(seq, angle_sigma=0, fps=10, speed=1.)
    still.draw(seq.time[10])
    m = still.fly_rotation.get_matrix()
    assert np.isclose(abs(np.arctan2(m[1, 0], m[0, 0])), np.pi)
    plt.close(movie.fig)
    plt.close(still.fig)


def test_ghost_cue_outlines_the_cue_only_in_the_dark():
    """With `ghost_cue`, where the cue is not shown the ring has a dotted outline of the cue's arc at ``heading``;
    where it is shown, the solid cue and no outline."""
    from types import SimpleNamespace
    n = 200
    heading = np.full(n, .5)
    seq = SimpleNamespace(dff=np.random.default_rng(0).standard_normal((16, n)), heading=heading, phi=heading,
                          rho=np.ones(n), offset=np.zeros(n), time=np.arange(n) * .1, dt=.1)
    dark = np.arange(n) >= 100
    movie = movie_mod.ClosedLoopMovie(seq, angle_sigma=0, fps=10, speed=1., cue_visible=~dark, ghost_cue=dark)
    movie.draw(seq.time[50])
    assert movie.eb_cue.get_visible() and not movie.eb_ghost.get_visible()
    x, y = movie.eb_cue.get_data()
    cue_at = np.arctan2(np.mean(y), np.mean(x))
    movie.draw(seq.time[150])
    assert movie.eb_ghost.get_visible() and not movie.eb_cue.get_visible()
    assert movie.eb_ghost.get_linestyle() == ':'
    x, y = movie.eb_ghost.get_data()
    assert np.isclose(np.arctan2(np.mean(y), np.mean(x)), cue_at, atol=.02)
    plt.close(movie.fig)
    no_ghost = movie_mod.ClosedLoopMovie(seq, angle_sigma=0, fps=10, speed=1., cue_visible=~dark)
    no_ghost.draw(seq.time[150])
    assert not no_ghost.eb_ghost.get_visible()
    plt.close(no_ghost.fig)
