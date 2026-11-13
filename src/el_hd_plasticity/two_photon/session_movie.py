"""Explanatory movie of one closed-loop imaging session: the fly, its ellipsoid body and the heatmap together.

The movie plays a session's real data and shows, frame by frame, how the session heatmaps of the paper are built:

* the fly turning in the virtual arena, with the visual cue fixed at the top (an allocentric view), under a
  picture of the rig;
* the ellipsoid body as 16 wedges (the imaging ROIs) shaded by the imaged activity (EPG calcium by default; the text
  naming it is set by :class:`MovieLabels`), with the population vector average
  (PVA) as an arrow (its angle the bump's position, its length the PVA magnitude) and the visual cue's angle
  (VR angle);
* the session heatmap building up from left to right and then scrolling, with the VR angle on the same axis, drawn as
  in the paper's heatmaps (same colormap, limits and heading-to-ROI mapping);
* the bump-cue offset histogram and its vector length, accumulating over the session.

To ease viewing at high playback speeds, the heading and the PVA angle and magnitude are smoothed for display
(``angle_sigma``): the fly, the cue, the PVA arrow and the heatmap's VR-angle trace use the smoothed values. The activity (wedges and heatmap) and the offset histogram and vector length use the unsmoothed data, so the
statistics shown are those of the session.

:class:`ClosedLoopMovie` builds the figure once and updates its artists for each video frame; use
:meth:`ClosedLoopMovie.save` to write an MP4 (ffmpeg from ``imageio[ffmpeg]``) and :meth:`ClosedLoopMovie.snapshot`
to check single frames.

Angle conventions: ROI ``j`` is centered at ``-pi + (j + 1/2) * 2pi / n_rois`` (as in the population vector). ROI
angles increase counterclockwise in the imaging plane as displayed (the preprocessing assigns each pixel
``arctan2(col - col0, row - row0)``), and the ellipsoid body is drawn the same way round, so the bump turns as it
does in the images; the heatmap's angle axis (``theta + pi``, 0 at its top row to 2pi at the bottom) starts at the top
of the ring, and these angles are labeled around it. ``heading`` is the visual cue's orientation relative to the fly
(the preprocessing sign-flips the bar signal): 0 with the cue straight ahead, +-pi with it behind. When the fly turns
clockwise the cue slips counterclockwise across its visual field and ``heading`` increases, so the fly is drawn rotated
by minus the heading, turning opposite to the cue and bump on the ring.
"""

import pathlib
from dataclasses import dataclass
from typing import Any

import imageio_ffmpeg
import numpy as np
import scipy as sp
from matplotlib import animation, transforms
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle, Wedge

from . import plotting, session, utilities
from .utilities import DEFAULT_OFFSET_BINS

CUE_COLOR = '#1ab0e8'
HEADING_COLOR = '#f0962a'
ARENA_COLOR = '#ececec'
SUBTITLE_COLOR = '#777777'
DARK_SCREEN_COLOR = '#1a1a1a'      # the arena when the screen is off
LASER_COLOR = '#e8574f'            # the optogenetic stimulation, as in the remapping figure's protocol panel

FLY_IMAGE_HEIGHT = 1.1      # arena units (ring radius 1) of the fly schematic's height
PVA_ARROW_MAX = .76         # ellipsoid body units (wedge ring from .26 to .82) of the longest PVA arrow
# PVA magnitude drawn as the longest arrow, the same in every movie so arrow lengths compare across sessions and
# sensors: a weak bump draws a short arrow. About the 99th percentile of EPG sessions' (smoothed) magnitude.
PVA_SCALE = 1.
PVA_STYLE = 'simple,head_width=1.6,head_length=1.4'
HEATMAP_POSITION = (.53, .392, .375, .235)    # figure fraction: 720 x 254 px at 1920 x 1080, centered on the EB
# with the VR angle in its own panel under the heatmap: the heatmap keeps its size and the pair is centered on the EB
HEATMAP_ABOVE_PANEL = (.53, .47, .375, .235)
HEADING_PANEL_POSITION = (.53, .37, .375, .085)


def _eb_angle(theta):
    """Drawing angle (radians, counterclockwise from +x) of ring angle `theta` on the ellipsoid body.

    The heatmap's top row (``theta = -pi``, labeled 0) is drawn at the top of the ring.
    """
    return np.asarray(theta) - np.pi / 2


def _polar(r, alpha):
    return r * np.cos(alpha), r * np.sin(alpha)


def _arc(r, alpha_start, alpha_stop, n=60):
    alphas = np.linspace(alpha_start, alpha_stop, n)
    return _polar(r, alphas)


def _break_at_wraps(y, jump):
    """Copy of `y` with NaN where consecutive samples jump by more than `jump`, so a line plot is not drawn across."""
    y = np.asarray(y, dtype=float).copy()
    y[1:][np.abs(np.diff(y)) > jump] = np.nan
    return y


def nan_gaussian(x, sigma, axis=-1):
    """Gaussian smoothing (width `sigma` samples) that ignores NaN: each value is a weighted mean of its finite
    neighbours only, and samples that were NaN stay NaN. Equal to ordinary smoothing where there is no NaN."""
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        return x
    valid = np.isfinite(x)
    num = sp.ndimage.gaussian_filter1d(np.where(valid, x, 0.), sigma, axis=axis, mode='nearest')
    den = sp.ndimage.gaussian_filter1d(valid.astype(float), sigma, axis=axis, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[~valid] = np.nan
    return out


def circular_smooth(angle, sigma):
    """Angles (rad) smoothed with a Gaussian of width `sigma` samples, as unit vectors so wrapping is handled.

    NaN samples are ignored by their neighbours and stay NaN (see :func:`nan_gaussian`).
    """
    angle = np.asarray(angle, dtype=float)
    if sigma <= 0:
        return angle
    return np.arctan2(nan_gaussian(np.sin(angle), sigma), nan_gaussian(np.cos(angle), sigma))


@dataclass(frozen=True)
class ColorEpoch:
    """A stretch of the movie drawn in one colormap, from `start` (s) until the next epoch starts.

    The heatmap columns, the ellipsoid body and the colorbar take the epoch's colormap, and the offset histogram
    accumulates a separate histogram per epoch, filled with `hist_color`.
    """
    start: float
    cmap: str = 'Greys'
    hist_color: Any = 'black'
    hist_alpha: float = .5
    hist_hatch: str | None = None       # draw the histogram as an open outline with this hatch, as the session figures do


@dataclass(frozen=True)
class AppendedSessions:
    """Sessions appended into one timeseries, `gap` samples of NaN between consecutive ones.

    Has the fields :class:`ClosedLoopMovie` reads from a timeseries (``dff`` (n_rois, T), ``heading``, ``phi``, ``rho``,
    ``offset``, ``time``, ``dt``) and the arguments it takes for appended sessions (``epochs``, ``accumulate``,
    ``cue_visible``, ``segments``).

    Attributes:
        sessions: Name of each session, in order.
    """
    dff: np.ndarray
    heading: np.ndarray
    phi: np.ndarray
    rho: np.ndarray
    offset: np.ndarray
    time: np.ndarray
    dt: float
    cue_visible: np.ndarray
    accumulate: np.ndarray
    epochs: tuple
    segments: tuple
    sessions: tuple


def append_sessions(parts, gap_s=15.):
    """Append session timeseries one after another, `gap_s` seconds of NaN apart, each its own colormap epoch.

    Args:
        parts: List of (label, timeseries, :class:`ColorEpoch` style, cue shown) in order; the timeseries need
            ``dff``, ``heading``, ``phi``, ``rho``, ``offset`` and ``dt``, all sampled alike. The epoch's ``start`` is
            replaced by the session's start; its histogram accumulates over the whole session.
        gap_s: Break between consecutive sessions (s): blank, and labeled 'break' without a time, since the real
            time between sessions is not part of the data.

    Returns:
        :class:`AppendedSessions`.
    """
    dt = float(parts[0][1].dt)
    for label, ts, _, _ in parts:
        if abs(float(ts.dt) / dt - 1) > .01:
            raise ValueError(f'{label} is sampled every {float(ts.dt):.4f} s, not {dt:.4f} s like the first session')
    n_gap = int(round(gap_s / dt))
    n_rois = parts[0][1].dff.shape[0]
    pieces = {key: [] for key in ('dff', 'heading', 'phi', 'rho', 'offset', 'cue_visible', 'counted')}
    starts, n = [], 0
    for k, (_, ts, _, cue) in enumerate(parts):
        if k:
            pieces['dff'].append(np.full((n_rois, n_gap), np.nan))
            for key in ('heading', 'phi', 'rho', 'offset'):
                pieces[key].append(np.full(n_gap, np.nan))
            pieces['cue_visible'].append(np.ones(n_gap, dtype=bool))
            pieces['counted'].append(np.zeros(n_gap, dtype=bool))
            n += n_gap
        starts.append(n)
        length = ts.dff.shape[1]
        for key in ('dff', 'heading', 'phi', 'rho', 'offset'):
            pieces[key].append(np.asarray(getattr(ts, key), dtype=float))
        pieces['cue_visible'].append(np.full(length, bool(cue)))
        pieces['counted'].append(np.ones(length, dtype=bool))
        n += length
    data = {key: np.concatenate(values, axis=-1) for key, values in pieces.items()}
    time = np.arange(n) * dt
    epochs, segments = [], []
    for k, ((label, ts, style, _), start) in enumerate(zip(parts, starts, strict=True)):
        epochs.append(ColorEpoch(time[start], style.cmap, style.hist_color, style.hist_alpha, style.hist_hatch))
        if k:
            segments.append((time[start - n_gap], time[start - 1], 'break', False))
        segments.append((time[start], time[start + ts.dff.shape[1] - 1], label))
    return AppendedSessions(dff=data['dff'], heading=data['heading'], phi=data['phi'], rho=data['rho'],
                            offset=data['offset'], time=time, dt=dt, cue_visible=data['cue_visible'],
                            accumulate=data['counted'] & np.isfinite(data['offset']), epochs=tuple(epochs),
                            segments=tuple(segments), sessions=tuple(label for label, *_ in parts))


def _forward_fill(x):
    """`x` with each NaN replaced by the last finite value before it (leading NaN by the first finite value)."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x)
    idx = np.where(valid, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = x[idx]
    filled[:np.argmax(valid)] = x[np.argmax(valid)]
    return filled


def visual_center(image):
    """Visual centre of mass of an RGBA image, as (column, row) in pixels from the top left (pixel centres at integers).

    Each pixel counts by its opacity times its darkness, so outlines and dark fill (a fly's head, thorax and legs) carry
    the weight and near-white fill (its wings) little, as they do to the eye. Transparent background counts for nothing.
    """
    image = np.asarray(image, dtype=float)
    if image.max() > 1:
        image = image / 255
    alpha = image[..., 3] if image.shape[-1] == 4 else np.ones(image.shape[:2])
    weight = alpha * (1 - image[..., :3].mean(axis=-1))
    rows, cols = np.mgrid[0:weight.shape[0], 0:weight.shape[1]]
    return (weight * cols).sum() / weight.sum(), (weight * rows).sum() / weight.sum()


@dataclass(frozen=True)
class MovieLabels:
    """The text naming what was imaged: the ellipsoid body's title, the heatmap's title and angle axis, and the legend
    entry of the wedges."""
    ring_title: str = 'ellipsoid body (EPG activity)'
    heatmap_title: str = 'EPG activity'
    angle_axis: str = 'EPG & VR angle (rad)'
    wedge_activity: str = 'EPG wedge activity'


EPG_LABELS = MovieLabels()


class ClosedLoopMovie:
    """Figure and artists of the movie of one closed-loop session; :meth:`draw` shows one moment of it.

    Args:
        ts: Session timeseries (:class:`session.SessionTimeseries`, single channel) with ``dff`` (n_rois, T),
            ``heading``, ``phi``, ``rho``, ``offset``, ``time`` and ``dt``.
        speed: Session seconds per video second.
        fps: Video frames per second.
        window: Seconds of session across the heatmap: it fills from left to right for the first `window` seconds,
            then scrolls with the current time at its right edge.
        scale_bar: Length (s) of the time scale bar under the heatmap.
        vmin, vmax: Color limits of the activity, as in the session heatmaps.
        cmap: Colormap of the activity.
        angle_sigma: Width (s) of the Gaussian smoothing of the displayed heading and PVA angle and magnitude.
        pva_scale: PVA magnitude drawn as the longest arrow (larger ones are drawn at that length). A fixed value
            (:data:`PVA_SCALE`), not one fitted to the session, so a weak bump is drawn as a short arrow.
        cue_ahead_heading: Heading (rad) at which the visual cue is straight ahead of the fly: 0, since ``heading`` is
            the cue's orientation relative to the fly. It sets only how the fly is drawn in the arena.
        fly_image, rig_image: Paths of a top-down fly schematic (facing up) and of a picture of the imaging rig, as
            RGBA images with transparent backgrounds; simple drawn versions are used if None.
        t_range: (start, stop) seconds of the session to play; the whole session if None.
        labels: :class:`MovieLabels` naming what was imaged.
        heading_panel: Draw the VR angle in its own panel under the heatmap (as in the GRAB sensor session figures)
            instead of over the heatmap (as in the EPG session figures).
        hist_color: Fill of the offset histogram; translucent black if None.
        outline_color: Color of the ellipsoid body's outline (its outer and inner edges), e.g. the color of the cell
            type imaged.
        epochs: :class:`ColorEpoch` sequence in time order; one epoch in `cmap` and `hist_color` if None.
        accumulate: Boolean per sample: whether it adds to its epoch's offset histogram (all samples if None).
        cue_visible: Boolean per sample: whether the visual cue was shown. Where it was not the arena is drawn dark,
            and neither the cue nor the VR angle is drawn (all samples if None).
        segments: (start, stop, label) of the parts of the movie, shown with the time within each by the clock, or
            (start, stop, label, False) for a part shown by its label alone, without a time; the clock counts the whole
            movie if None.
        unsmoothed: Boolean per sample: where to draw the heading and PVA as recorded instead of smoothed, e.g. where
            the cue jumps rather than turns with the fly (smoothing would draw a jump as a ramp).
        stim_theta: Ring angle (as ``heading``) of the wedge being optogenetically stimulated at each sample, NaN when
            none is; drawn as a laser cone at that wedge, as in the remapping figure.
        stim_targets: Ring angles of the protocol's stimulation targets, marked on the ellipsoid body where
            `stim_block` is True (the current one filled while it is stimulated).
        stim_block: Boolean per sample: where to mark the stimulation targets.
        pva_hidden: Boolean per sample: where not to draw the PVA arrow (e.g. during optogenetic pairing).
        trace_visible: Boolean per sample: where to draw the VR angle on the heatmap; where the cue is shown if None.
            (In darkness the VR angle is the ball's angle, which the dark session figures do draw.)
        ghost_cue: Boolean per sample: where the cue is not shown, whether to outline on the ellipsoid body where it
            would be, from ``heading`` (a faint dotted outline), so drift of the bump from the cue in darkness can be
            seen; nowhere if None.

    NaN in ``dff``, ``phi`` or ``heading`` (frames that were masked, or gaps between appended sessions) is drawn blank:
    white heatmap columns and wedges, no PVA arrow, no cue on the ellipsoid body; the fly keeps its last orientation.
    """

    def __init__(self, ts, speed=8., fps=24, window=180., scale_bar=30., vmin=-.5, vmax=2.5, cmap='Greys',
                 angle_sigma=.5, pva_scale=PVA_SCALE, cue_ahead_heading=0., fly_image=None,
                 rig_image=None, t_range=None, labels=EPG_LABELS, heading_panel=False, hist_color=None,
                 outline_color='black', epochs=None, accumulate=None, cue_visible=None, segments=None,
                 unsmoothed=None, stim_theta=None, stim_targets=(), stim_block=None, pva_hidden=None,
                 trace_visible=None, ghost_cue=None):
        self.ts = ts
        self.labels = labels
        self.heading_panel = heading_panel
        self.outline_color = outline_color
        self.heatmap_position = HEATMAP_ABOVE_PANEL if heading_panel else HEATMAP_POSITION
        self.time = np.asarray(ts.time, dtype=float)
        n = len(self.time)
        self.speed, self.fps, self.window = speed, fps, window
        self.frame_dt = speed / fps
        self.cue_ahead_heading = cue_ahead_heading
        self.t_range = (self.time[0], self.time[-1]) if t_range is None else t_range
        self.norm = Normalize(vmin, vmax)
        self.n_rois = ts.dff.shape[0]
        self.roi_theta = -np.pi + (np.arange(self.n_rois) + .5) * 2 * np.pi / self.n_rois
        self.segments = tuple(segments) if segments is not None else None
        self.stim_theta = None if stim_theta is None else np.asarray(stim_theta, dtype=float)
        self.stim_targets = np.asarray(stim_targets, dtype=float)
        self.stim_block = np.zeros(n, dtype=bool) if stim_block is None else np.asarray(stim_block, dtype=bool)

        # colormap epochs; masked values are drawn white
        if epochs is None:
            epochs = [ColorEpoch(self.time[0], cmap, 'black' if hist_color is None else hist_color,
                                 .5 if hist_color is None else 1.)]
        self.epochs = tuple(epochs)
        self.epoch_cmaps = [plt.get_cmap(e.cmap).with_extremes(bad='white') for e in self.epochs]
        self.epoch_of = np.clip(np.searchsorted([e.start for e in self.epochs], self.time, side='right') - 1, 0, None)
        self.epoch_first = [int(np.searchsorted(self.time, e.start)) for e in self.epochs] + [n]
        self.cmap = self.epoch_cmaps[0]

        self.cue_visible = np.ones(n, dtype=bool) if cue_visible is None else np.asarray(cue_visible, dtype=bool)
        self.valid = np.isfinite(ts.dff).any(axis=0)

        # smoothed for display only
        sigma = angle_sigma / ts.dt
        self.heading = circular_smooth(ts.heading, sigma)
        self.phi = circular_smooth(ts.phi, sigma)
        self.rho = nan_gaussian(ts.rho, sigma)
        if unsmoothed is not None:
            raw = np.asarray(unsmoothed, dtype=bool)
            for name in ('heading', 'phi', 'rho'):
                setattr(self, name, np.where(raw, np.asarray(getattr(ts, name), dtype=float), getattr(self, name)))
        self.fly_heading = _forward_fill(self.heading)               # the fly keeps its last orientation in gaps
        self.pva_hidden = np.zeros(n, dtype=bool) if pva_hidden is None else np.asarray(pva_hidden, dtype=bool)
        self.heading_shown = np.where(self.cue_visible, self.heading, np.nan)   # no cue, no cue on the ring
        ghost = np.zeros(n, dtype=bool) if ghost_cue is None else np.asarray(ghost_cue, dtype=bool)
        self.heading_ghost = np.where(ghost & ~self.cue_visible, self.heading, np.nan)
        trace = self.cue_visible if trace_visible is None else np.asarray(trace_visible, dtype=bool)
        self.heading_traced = np.where(trace, self.heading, np.nan)
        self.pva_scale = pva_scale

        # offset histograms (unsmoothed offset), one per epoch: the proportion in each bin of the epoch's
        # accumulated samples up to each sample; `running_hist` / `running_veclen` are those of each sample's epoch
        self.bins = DEFAULT_OFFSET_BINS
        self.centers = (self.bins[1:] + self.bins[:-1]) / 2
        offset = np.asarray(ts.offset, dtype=float)
        counted = np.isfinite(offset) & (np.ones(n, dtype=bool) if accumulate is None else np.asarray(accumulate, bool))
        idx = np.clip(np.digitize(np.nan_to_num(offset), self.bins) - 1, 0, len(self.centers) - 1)
        onehot = np.eye(len(self.centers))[idx]
        unit = np.exp(1j * np.nan_to_num(offset))
        self.epoch_counts, self.epoch_veclen = [], []
        for e in range(len(self.epochs)):
            mask = counted & (self.epoch_of == e)
            counts = np.cumsum(onehot * mask[:, None], axis=0)
            total = counts.sum(axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                self.epoch_counts.append(np.where(total > 0, counts / total, 0.))
                self.epoch_veclen.append(np.abs(np.cumsum(unit * mask)) / np.cumsum(mask))
        self.running_hist = np.stack([self.epoch_counts[e][i] for i, e in enumerate(self.epoch_of)])
        self.running_veclen = np.array([self.epoch_veclen[e][i] for i, e in enumerate(self.epoch_of)])

        self.fig = plt.figure(figsize=[19.2, 10.8], dpi=100, facecolor='white')
        self._build_rig(rig_image)
        self._build_arena(fly_image)
        self._build_colorbar()
        self._build_ellipsoid_body()
        self._build_heatmap(scale_bar)
        self._build_histogram()
        self._build_legend()

    # ------------------------------------------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------------------------------------------
    def _build_rig(self, rig_image):
        ax = self.fig.add_axes([.03, .765, .105, .2])
        ax.axis('off')
        if rig_image is not None:
            ax.imshow(plt.imread(rig_image), interpolation='antialiased')
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1.1, 1.2)
            ax.set_aspect('equal')
            ax.add_patch(Rectangle((-.28, .45), .56, .7, color='#3a3a3a'))                   # objective
            ax.add_patch(Polygon([(-.28, .45), (.28, .45), (.14, .2), (-.14, .2)], color='#3a3a3a'))
            ax.add_patch(Circle((0, -.45), .55, facecolor='#bdbdbd', edgecolor='#8a8a8a'))   # ball
            ax.add_patch(Ellipse((0, .12), .16, .22, color='#555555'))                       # fly on the ball
        self.fig.text(.14, .865, '2P imaging\nin VR', ha='left', va='center', fontsize=13, color=SUBTITLE_COLOR)

    def _build_arena(self, fly_image):
        ax = self.arena_ax = self.fig.add_axes([.035, .4, .17, .3])
        ax.axis('off')
        self.arena_ring = ax.add_patch(Circle((0, 0), 1., facecolor='none', edgecolor=ARENA_COLOR, linewidth=22))
        self.arena_cue, = ax.plot(*_arc(1., np.pi / 2 - .2, np.pi / 2 + .2), color=CUE_COLOR, linewidth=22,
                                  solid_capstyle='butt')
        self.arena_cue_label = ax.text(0, 1.22, 'visual cue', ha='center', va='bottom', color=CUE_COLOR, fontsize=15)

        self.fly_rotation = transforms.Affine2D()          # rotated by the heading each frame
        frame = self.fly_rotation + ax.transData
        if fly_image is not None:
            image = plt.imread(fly_image)
            # placed so its visual centre of mass is at the origin, the point the rotation turns the fly about
            scale = FLY_IMAGE_HEIGHT / image.shape[0]
            cx, cy = visual_center(image)
            extent = ((-.5 - cx) * scale, (image.shape[1] - .5 - cx) * scale,
                      (cy - image.shape[0] + .5) * scale, (cy + .5) * scale)
            ax.imshow(image, extent=extent, transform=frame, interpolation='antialiased', zorder=2)
        else:
            self._draw_fly(ax, transforms.Affine2D().scale(1.45) + frame)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')

    @staticmethod
    def _draw_fly(ax, frame):
        """A simple top-down fly facing +y, for when no fly schematic is given."""
        body = dict(edgecolor='black', linewidth=1.2, transform=frame)
        for (x0, y0), (x1, y1), (x2, y2) in (((.06, .09), (.16, .21), (.19, .31)),          # legs
                                             ((.08, .05), (.21, .03), (.29, -.02)),
                                             ((.06, .01), (.16, -.13), (.2, -.27))):
            for side in (1, -1):
                ax.plot([side * x0, side * x1, side * x2], [y0, y1, y2], color='black', linewidth=1.8,
                        transform=frame)
        ax.add_patch(Ellipse((0, -.17), .17, .31, facecolor='#9a9a9a', **body))                # abdomen
        for side in (1, -1):                                                                 # wings
            ax.add_patch(Ellipse((side * .075, -.15), .13, .37, angle=-side * 12, facecolor='white', alpha=.75,
                                 **body))
        ax.add_patch(Ellipse((0, .05), .19, .2, facecolor='#6e6e6e', **body))                  # thorax
        ax.add_patch(Ellipse((0, .19), .14, .1, facecolor='#4f4f4f', **body))                  # head

    def _build_colorbar(self):
        ax = self.fig.add_axes([.05, .3, .14, .018])
        gradient = np.linspace(self.norm.vmin, self.norm.vmax, 256)[np.newaxis]
        self.colorbar_image = ax.imshow(gradient, aspect='auto', cmap=self.cmap, norm=self.norm,
                                        extent=(self.norm.vmin, self.norm.vmax, 0, 1))
        ax.set_yticks([])
        ax.set_xticks(np.arange(np.ceil(self.norm.vmin), np.floor(self.norm.vmax) + 1))
        ax.tick_params(labelsize=12)
        ax.set_title('z-scored F', fontsize=13, pad=8)
        for spine in ax.spines.values():
            spine.set_linewidth(.8)

    def _build_ellipsoid_body(self):
        ax = self.eb_ax = self.fig.add_axes([.215, .27, .27, .48])
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        self.fig.text(.35, .78, self.labels.ring_title, ha='center', fontsize=16, va='bottom')
        # the heatmap's angle axis (0 at its top row) around the ring, so each wedge can be found in the heatmap
        for heatmap_angle, label in ((0, '0'), (np.pi / 2, r'$\pi$/2'), (np.pi, r'$\pi$'), (3 * np.pi / 2, r'3$\pi$/2')):
            ax.text(*_polar(1.13, _eb_angle(heatmap_angle - np.pi)), label, ha='center', va='center', fontsize=14,
                    color=SUBTITLE_COLOR)

        r_in, r_out = .26, .82
        half = 180 / self.n_rois
        self.wedges = []
        for theta in self.roi_theta:
            center = np.degrees(_eb_angle(theta))
            wedge = Wedge((0, 0), r_out, center - half, center + half, width=r_out - r_in, edgecolor='none')
            ax.add_patch(wedge)
            self.wedges.append(wedge)
        for theta in self.roi_theta - np.pi / self.n_rois:                                     # wedge boundaries
            ax.plot(*zip(_polar(r_in, _eb_angle(theta)), _polar(r_out, _eb_angle(theta)), strict=True),
                    color='#9a9a9a', linestyle='--', linewidth=.9)
        ax.add_patch(Circle((0, 0), r_out, facecolor='none', edgecolor=self.outline_color, linewidth=2))
        ax.add_patch(Circle((0, 0), r_in, facecolor='white', edgecolor=self.outline_color, linewidth=2, zorder=3))

        cue_width = 16
        self.eb_cue, = ax.plot([], [], color=CUE_COLOR, linewidth=cue_width, solid_capstyle='butt', zorder=4)
        # where the cue would be, in darkness: a faint dotted outline of the same arc
        self.eb_ghost, = ax.plot([], [], color=CUE_COLOR, linewidth=2, linestyle=':', alpha=.5, zorder=4,
                                 visible=False)
        data_per_inch = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_position().width * self.fig.get_figwidth())
        self.ghost_half_width = cue_width / 72 * data_per_inch / 2
        self.pva_arrow = FancyArrowPatch((0, 0), (0, .7), arrowstyle=PVA_STYLE, mutation_scale=14,
                                         facecolor='white', edgecolor='black', linewidth=1.8, zorder=5)
        ax.add_patch(self.pva_arrow)
        self.r_cue = .91

        # the optogenetic stimulation: the protocol's targets on the wedge ring, and a laser cone at the current one,
        # stacked translucent triangles narrow and dense at the wedge, wide and faint outside
        r_mid = (r_in + r_out) / 2
        self.stim_markers = [ax.add_patch(Circle(_polar(r_mid, _eb_angle(theta)), .045, facecolor='none',
                                                 edgecolor=LASER_COLOR, linewidth=2, zorder=4, visible=False))
                             for theta in self.stim_targets]
        self.laser_layers = []
        if self.stim_theta is not None:
            for width, alpha in ((.2, .1), (.15, .14), (.1, .2), (.06, .3)):
                layer = Polygon(np.zeros((3, 2)), closed=True, facecolor=LASER_COLOR, edgecolor='none', alpha=alpha,
                                zorder=4.5, visible=False)
                ax.add_patch(layer)
                self.laser_layers.append((width, layer))
        self.r_stim = r_mid

    def _build_heatmap(self, scale_bar):
        ax = self.heat_ax = self.fig.add_axes(self.heatmap_position)
        # one image per colormap epoch; each is cut off at the current sample every frame (see draw), so nothing
        # after "now" is drawn
        self.heat_images = [ax.imshow(self.ts.dff[:, :1], aspect='auto', cmap=cmap, norm=self.norm,
                                      interpolation='none', visible=False) for cmap in self.epoch_cmaps]
        if self.heading_panel:
            # VR angle on the heatmap's angle axis (0 at the top, 2pi at the bottom), in radians, below the heatmap
            time_ax = self.heading_ax = self.fig.add_axes(HEADING_PANEL_POSITION, sharex=ax)
            self.heading_y = _break_at_wraps(self.heading_traced + np.pi, np.pi)
            self.heading_ylim = (2 * np.pi, 0)
            time_ax.set_yticks([0, np.pi, 2 * np.pi], labels=plotting.ROI_TICK_LABELS)
            time_ax.tick_params(labelsize=13)
            time_ax.set_ylabel('VR angle\n(rad)', fontsize=13, labelpad=6)
            for side in ('top', 'right', 'bottom'):
                time_ax.spines[side].set_visible(False)
            time_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        else:
            time_ax = self.heading_ax = ax
            self.heading_y = _break_at_wraps(utilities.heading_to_roi_coords(self.heading_traced, self.n_rois),
                                             self.n_rois / 2)
            self.heading_ylim = (self.n_rois - .5, -.5)
        self.heading_line, = time_ax.plot([], [], color=HEADING_COLOR, linewidth=3)
        time_ax.set_ylim(*self.heading_ylim)
        ax.set_ylim(self.n_rois - .5, -.5)
        ax.set_yticks(plotting.ROI_TICKS, labels=plotting.ROI_TICK_LABELS)
        ax.set_xticks([])
        ax.tick_params(labelsize=15)
        ax.set_ylabel(self.labels.angle_axis, fontsize=14, labelpad=6)
        for side in ('top', 'right', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.set_title(self.labels.heatmap_title, fontsize=16, pad=12)

        # rows under the lowest panel, placed in pixels so they keep their spacing whatever the panel's height
        height_px = time_ax.get_position().height * self.fig.get_figheight() * self.fig.dpi

        def below(px):
            return -px / height_px

        # the current time, under the newest column: x in data, y in axes coordinates
        under = transforms.blended_transform_factory(time_ax.transData, time_ax.transAxes)
        self.now_marker, = time_ax.plot([0], [below(10)], marker='^', markersize=11, color=SUBTITLE_COLOR,
                                        transform=under, clip_on=False)
        # the window keeps its width, so the scale bar can live in axes coordinates
        length, y = scale_bar / self.window, below(56)
        time_ax.plot([0, length], [y, y], color='black', linewidth=3, transform=time_ax.transAxes, clip_on=False)
        time_ax.text(length + .02, y, f'{scale_bar:g} s', transform=time_ax.transAxes, ha='left', va='center',
                     fontsize=14)
        self.clock = time_ax.text(1., y, '', transform=time_ax.transAxes, ha='right', va='center', fontsize=13,
                                  color=SUBTITLE_COLOR)

    def _build_histogram(self):
        ax = self.hist_ax = self.fig.add_axes([.92, self.heatmap_position[1], .04, self.heatmap_position[3]])
        # scaled to the whole session's histogram; the first seconds' narrow histograms run off the axis instead
        self.hist_xmax = max(.15, 1.2 * max(counts[-1].max() for counts in self.epoch_counts))
        ax.set_xlim(0, self.hist_xmax)
        ax.set_ylim(np.pi, -np.pi)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_yticks([-np.pi, 0, np.pi], labels=[r'-$\pi$', '0', r'$\pi$'])
        ax.set_xticks([.2] if self.hist_xmax > .22 else [.1])
        ax.tick_params(labelsize=14)
        ax.set_ylabel('bump-cue offset (rad)', fontsize=14, labelpad=8)
        for side in ('top', 'left'):
            ax.spines[side].set_visible(False)
        self.hist_fills = []
        # one vector length per epoch, stacked above the histogram, the newest on top
        self.veclen_texts = [ax.text(1., 1.04 + .075 * e, '', transform=ax.transAxes, ha='right', va='bottom',
                                     fontsize=14, color=epoch.hist_color)
                             for e, epoch in enumerate(self.epochs)]

    def _build_legend(self):
        x, y, dy = .25, .2, .042
        # the wedge entry in the colormap's own dark end, so it matches the wedges and the heatmap
        items = (('patch', CUE_COLOR, 'visual cue'), ('patch', self.cmap(.85), self.labels.wedge_activity),
                 ('line', HEADING_COLOR, 'VR angle'), ('arrow', 'white', 'PVA'))
        if self.stim_theta is not None:
            items += (('laser', LASER_COLOR, 'optogenetic stimulation'),)
        for i, (kind, color, label) in enumerate(items):
            col, row = divmod(i, 2)
            xi, yi = x + .2 * col, y - dy * row
            if kind in ('patch', 'laser'):
                marker = Rectangle((xi, yi - .011), .009, .022, color=color, transform=self.fig.transFigure,
                                   alpha=.75 if kind == 'laser' else 1)
                if label == self.labels.wedge_activity:
                    self.wedge_swatch = marker
            elif kind == 'line':
                marker = plt.Line2D([xi - .002, xi + .011], [yi, yi], color=color, linewidth=4,
                                    transform=self.fig.transFigure)
            else:
                marker = FancyArrowPatch((xi - .003, yi), (xi + .013, yi), arrowstyle=PVA_STYLE, mutation_scale=10,
                                         facecolor=color, edgecolor='black', linewidth=1.2,
                                         transform=self.fig.transFigure)
            self.fig.add_artist(marker)
            self.fig.text(xi + .018, yi, label, va='center', fontsize=14)

    # ------------------------------------------------------------------------------------------------------
    # One moment of the session
    # ------------------------------------------------------------------------------------------------------
    def index_at(self, t):
        """Index of the sample shown at time `t` (s): the latest imaged (non-NaN) sample within the video frame ending
        at `t`, so brief imaging between masked frames is never skipped, else the latest sample before `t`."""
        last = int(np.clip(np.searchsorted(self.time, t, side='right') - 1, 0, len(self.time) - 1))
        first = int(np.clip(np.searchsorted(self.time, t - self.frame_dt, side='right'), 0, last))
        imaged = np.flatnonzero(self.valid[first:last + 1])
        return first + int(imaged[-1]) if len(imaged) else last

    def heatmap_xlim(self, i):
        """Time limits of the heatmap at sample `i`: fixed while it fills, then the last `window` seconds."""
        start = self.time[0]
        if self.time[i] < start + self.window:
            return start, start + self.window
        return self.time[i] - self.window, self.time[i]

    def pva_length(self, i):
        """Length of the PVA arrow at sample `i`, proportional to the smoothed PVA magnitude."""
        return PVA_ARROW_MAX * float(np.clip(np.nan_to_num(self.rho[i]) / self.pva_scale, 0, 1))

    def draw(self, t):
        """Update every artist to session time `t` (s)."""
        i = self.index_at(t)
        ts = self.ts

        epoch = int(self.epoch_of[i])
        cmap = self.epoch_cmaps[epoch]

        # the fly in the arena; the screen goes dark when the cue is not shown
        # heading is the cue's orientation relative to the fly, so the fly's own orientation is minus it
        self.fly_rotation.clear().rotate(-(self.fly_heading[i] - self.cue_ahead_heading))
        visible = bool(self.cue_visible[i])
        self.arena_ring.set_edgecolor(ARENA_COLOR if visible else DARK_SCREEN_COLOR)
        self.arena_cue.set_visible(visible)
        self.arena_cue_label.set_visible(visible)

        # the ellipsoid body
        for wedge, value in zip(self.wedges, ts.dff[:, i], strict=True):
            wedge.set_facecolor(cmap(self.norm(value)))
        a_cue, a_pva = _eb_angle(self.heading_shown[i]), _eb_angle(self.phi[i])
        self.eb_cue.set_visible(bool(np.isfinite(a_cue)))
        self.eb_cue.set_data(*_arc(self.r_cue, a_cue - np.pi / self.n_rois, a_cue + np.pi / self.n_rois))
        a_ghost = _eb_angle(self.heading_ghost[i])
        self.eb_ghost.set_visible(bool(np.isfinite(a_ghost)))
        if self.eb_ghost.get_visible():
            lo, hi = a_ghost - np.pi / self.n_rois, a_ghost + np.pi / self.n_rois
            outer = _arc(self.r_cue + self.ghost_half_width, lo, hi)
            inner = _arc(self.r_cue - self.ghost_half_width, hi, lo)
            self.eb_ghost.set_data(np.r_[outer[0], inner[0], outer[0][:1]], np.r_[outer[1], inner[1], outer[1][:1]])
        self.pva_arrow.set_visible(bool(np.isfinite(a_pva) and np.isfinite(self.rho[i]) and not self.pva_hidden[i]))
        if self.pva_arrow.get_visible():
            length = max(self.pva_length(i), .03)          # never zero: FancyArrowPatch needs two distinct points
            self.pva_arrow.set_positions((0, 0), _polar(length, a_pva))
            self.pva_arrow.set_mutation_scale(14 * float(np.clip(length / (.5 * PVA_ARROW_MAX), .35, 1)))
        self._draw_stimulation(i)
        self.colorbar_image.set_cmap(cmap)
        if hasattr(self, 'wedge_swatch'):
            self.wedge_swatch.set_color(cmap(.85))

        # the heatmap fills, then scrolls; each epoch's columns in its own colormap; the marker sits under the newest
        half = self.ts.dt / 2
        for e, image in enumerate(self.heat_images):
            a, b = self.epoch_first[e], min(self.epoch_first[e + 1], i + 1)
            image.set_visible(b > a)
            if b > a:
                image.set_data(ts.dff[:, a:b])
                image.set_extent((self.time[a] - half, self.time[b - 1] + half, self.n_rois - .5, -.5))
        self.heading_line.set_data(self.time[:i + 1], self.heading_y[:i + 1])
        self.heat_ax.set_xlim(*self.heatmap_xlim(i))
        self.heat_ax.set_ylim(self.n_rois - .5, -.5)
        self.heading_ax.set_ylim(*self.heading_ylim)
        self.now_marker.set_xdata([self.time[i]])
        self.clock.set_text(self.clock_text(i))

        # the offset histograms accumulate, one per epoch that has begun, older ones frozen
        for fill in self.hist_fills:
            fill.remove()
        self.hist_fills = []
        for e, epoch_style in enumerate(self.epochs[:epoch + 1]):
            hist = self.epoch_counts[e][i]
            if hist.sum() > 0 and epoch_style.hist_hatch:            # an open, hatched outline
                self.hist_fills.append(self.hist_ax.fill_betweenx(
                    self.centers, 0, hist, color='none', edgecolor=epoch_style.hist_color,
                    hatch=epoch_style.hist_hatch, alpha=1))
            elif hist.sum() > 0:
                self.hist_fills.append(self.hist_ax.fill_betweenx(
                    self.centers, 0, hist, linewidth=0, color=epoch_style.hist_color, alpha=epoch_style.hist_alpha))
            veclen = self.epoch_veclen[e][i]
            self.veclen_texts[e].set_text(f'vec. length={veclen:.2f}' if np.isfinite(veclen) else '')
        for e in range(epoch + 1, len(self.epochs)):
            self.veclen_texts[e].set_text('')

    def _draw_stimulation(self, i):
        """The laser cone at the stimulated wedge, and the targets of the pairing protocol, at sample `i`."""
        theta = self.stim_theta[i] if self.stim_theta is not None else np.nan
        on = bool(np.isfinite(theta))
        for marker, target in zip(self.stim_markers, self.stim_targets, strict=True):
            marker.set_visible(bool(self.stim_block[i]))
            current = on and abs(np.angle(np.exp(1j * (target - theta)))) < 1e-6
            marker.set_facecolor(LASER_COLOR if current else 'none')
        for width, layer in self.laser_layers:
            layer.set_visible(on)
            if on:
                a = _eb_angle(theta)
                layer.set_xy([_polar(self.r_stim, a), _polar(1.35, a - width * 1.6), _polar(1.35, a + width * 1.6)])

    def clock_text(self, i):
        """The clock under the heatmap at sample `i`: the time in the whole movie, or within the current segment."""
        t = self.time[i]
        if self.segments is None:
            return f't = {t - self.time[0]:.0f} s of {self.time[-1] - self.time[0]:.0f} s ({self.speed:g}x speed)'
        for start, stop, label, *timed in self.segments:
            if start <= t <= stop:
                if timed and not timed[0]:                     # e.g. a break of no recorded length: its name alone
                    return label
                return f'{label}: t = {t - start:.0f} s of {stop - start:.0f} s ({self.speed:g}x speed)'
        return f'({self.speed:g}x speed)'


    def frame_times(self):
        """Session time (s) of every video frame."""
        start, stop = self.t_range
        return np.arange(start, stop, self.speed / self.fps)

    def snapshot(self, t, path, dpi=100):
        """Save the figure at session time `t` as an image."""
        self.draw(t)
        self.fig.savefig(path, dpi=dpi, facecolor='white')

    def save(self, outfile, dpi=100, bitrate=6000, progress=True):
        """Write the movie to `outfile` (MP4, H.264, 1920 x 1080 at the default dpi)."""
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=self.fps, codec='libx264', bitrate=bitrate,
                                        extra_args=['-pix_fmt', 'yuv420p'])
        times = self.frame_times()
        with writer.saving(self.fig, str(outfile), dpi):
            for n, t in enumerate(times):
                self.draw(t)
                writer.grab_frame(facecolor='white')
                if progress and n % (self.fps * 10) == 0:
                    print(f'frame {n} / {len(times)}', flush=True)
        return outfile


def render(movie, outfile, snapshot_times=None):
    """Write `movie` to `outfile`, or save PNG frames at `snapshot_times` (s) next to it; then close its figure.

    Returns:
        Path of the movie, or the list of snapshot paths.
    """
    outfile = pathlib.Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        if snapshot_times is not None:
            paths = [outfile.with_name(f'{outfile.stem}_t{t:06.1f}.png') for t in snapshot_times]
            for t, path in zip(snapshot_times, paths, strict=True):
                movie.snapshot(t, path)
            return paths
        return movie.save(outfile)
    finally:
        plt.close(movie.fig)


def movie_from_session(session_manager, row, outfile, snapshot_times=None, **kwargs):
    """Movie (or preview frames) of one session of a session table, loaded with the default session settings.

    The session is loaded exactly as for the session heatmaps (default :func:`session.GetTS` settings, including the
    -200 ms neural shift), so the movie shows the same data.

    Args:
        session_manager: :class:`loading.SessionManager` holding the session.
        row: The session's row of ``session_manager.sess_df``.
        outfile: MP4 path; its folder is created if needed.
        snapshot_times: If given, save PNG frames at these times (s from the start of the session) next to `outfile`
            instead of the movie, to check the layout quickly.
        **kwargs: Passed to :class:`ClosedLoopMovie`.

    Returns:
        Path of the movie, or the list of snapshot paths.
    """
    ts = session.GetTS(session_manager.load_row(row))
    movie = ClosedLoopMovie(ts, **kwargs)
    outfile = pathlib.Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        if snapshot_times is not None:
            paths = [outfile.with_name(f'{outfile.stem}_t{t:05.1f}.png') for t in snapshot_times]
            for t, path in zip(snapshot_times, paths, strict=True):
                movie.snapshot(ts.time.iloc[0] + t, path)
            return paths
        return movie.save(outfile)
    finally:
        plt.close(movie.fig)
