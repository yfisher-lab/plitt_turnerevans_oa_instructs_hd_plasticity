"""Movies of the optogenetic remapping experiments: an animated schematic of the protocol, and a fly's real sessions.

:class:`RemappingSchematicMovie` animates the protocol panel of the remapping figure (described below).
:func:`remap_session_sequence` appends a fly's real imaging sessions (baseline, the two pairing scans and the two
post-pairing scans) into one timeseries for :class:`session_movie.ClosedLoopMovie`, with the colormap epochs, offset
histograms and screen-off periods of the experiment.

The schematic:

The movie animates the three stages of the protocol on a schematic ellipsoid body, in the style of the figure panel:

* **baseline** -- closed loop: the fly turns, the visual cue moves around it, and the EPG activity bump follows the
  cue at the fly's own bump-cue offset;
* **optogenetic pairing** -- the cue is flashed at a series of positions while the EPG neurons at the wedge a fixed
  angle away (the opto-paired offset) are driven with the stimulation laser, which moves the bump there; the pairing
  steps around the ellipsoid body at every angle of the protocol, cycle after cycle;
* **post-pairing** -- closed loop again, the bump now following the cue at the opto-paired offset.

The activity is simulated (a von Mises bump and a smooth random turning trajectory): this is a schematic, not data.
The protocol itself -- the number of stimulation angles and cycles, the stimulation and gap durations and the step of
the cue between pairings -- comes from :class:`PairingProtocol`, which :meth:`PairingProtocol.from_session` reads from
a recorded pairing scan, so the movie runs the experiment's actual schedule.

Angles are drawing angles on the ellipsoid body (radians counterclockwise from the positive x axis). The cue's angle on
the ellipsoid body is the heading, which is the cue's orientation relative to the fly: a clockwise turn slips the cue
counterclockwise across the fly's visual field, so in the arena, where the fly is drawn facing up, the cue moves the same
way round as it does on the ellipsoid body.
"""

import pathlib
from dataclasses import dataclass

import imageio_ffmpeg
import numpy as np
import scipy as sp
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

from . import remapping, session
from .session_movie import ClosedLoopMovie, ColorEpoch, nan_gaussian

CUE_COLOR = '#1ab0e8'
OFFSET_COLOR = '#5b2a86'
EB_COLOR = '#a9a9a9'
BUMP_COLOR = '#e4574c'
LASER_COLOR = '#e8574f'
PAIRING_BACKGROUND = '#f7e8ee'
PAIRING_TEXT = '#a31a5c'
ARENA_COLOR = '#ececec'
LABEL_COLOR = '#777777'

N_WEDGES = 16
BUMP_KAPPA = 8.                     # concentration of the schematic von Mises bump: about 2-3 wedges wide
TOP = np.pi / 2                     # drawing angle of the top of the ellipsoid body


@dataclass(frozen=True)
class PairingProtocol:
    """Schedule of an optogenetic pairing scan.

    Attributes:
        n_locations: Stimulation angles around the ellipsoid body (one cycle visits each once).
        n_cycles: Cycles through the angles.
        stim_s: Duration of each stimulation (s).
        gap_s: Time between the end of one stimulation and the start of the next (s).
        step: Change of the cue's angle on the ellipsoid body from one pairing to the next (rad; negative is
            clockwise). The stimulated wedge moves with it.
        paired_offset: The opto-paired offset: angle from the cue to the stimulated wedge (rad).
        first_cue: Drawing angle of the cue at the first pairing.
    """
    n_locations: int = 8
    n_cycles: int = 5
    stim_s: float = 1.8
    gap_s: float = .2
    step: float = -np.pi / 4
    paired_offset: float = np.pi
    first_cue: float = TOP

    @property
    def n_pairings(self):
        return self.n_locations * self.n_cycles

    @property
    def period(self):
        return self.stim_s + self.gap_s

    @property
    def duration(self):
        return self.n_pairings * self.period

    def cue_angle(self, k):
        """Drawing angle of the cue at pairing `k` (0-based)."""
        return self.first_cue + (k % self.n_locations) * self.step

    @classmethod
    def from_session(cls, pp, paired_offset, first_cue=TOP):
        """The schedule of a recorded pairing scan (a preprocessed session with mark-points metadata).

        The stimulation and gap durations are the mark-points ``duration`` and ``inter_point_delay``; the number of
        angles is the number of distinct stimulation points (an initial point that is never repeated is a placeholder
        and is left out) and the number of cycles the number of stimulations per angle. The step is the mean change of
        the cue between consecutive stimulations, from the bar position at each opto trigger, converted to the
        cue's angle on the ellipsoid body (``heading = -bar - pi``, so it changes by the opposite of the bar).

        Args:
            pp: Preprocessed pairing scan.
            paired_offset: The scan's opto-paired offset (rad): 0 for a 0 degree pairing scan, pi for 180 degrees.
            first_cue: Drawing angle of the cue at the first pairing.
        """
        mark_points = pp.metadata['mark_points']
        points = [int(p['Index']) for p in np.atleast_1d(mark_points['points_list'])]
        counts = {index: points.count(index) for index in set(points)}
        stimulated = [index for index, count in counts.items() if count > 1]
        n_cycles = counts[stimulated[0]]
        if any(counts[index] != n_cycles for index in stimulated):
            raise ValueError(f'stimulation points are not repeated equally often: {counts}')

        voltage = pp.voltage_recording_aligned
        onsets = np.flatnonzero(np.diff((voltage[' Opto Trigger'].to_numpy() > 1000).astype(int)) > 0) + 1
        bar = np.unwrap(voltage[' Heading'].to_numpy()[onsets])
        within_cycle = np.diff(bar).reshape(-1)[np.arange(len(onsets) - 1) % len(stimulated) != len(stimulated) - 1]
        return cls(n_locations=len(stimulated), n_cycles=n_cycles,
                   stim_s=float(np.squeeze(mark_points['duration'])) / 1000,
                   gap_s=float(np.squeeze(mark_points['inter_point_delay'])) / 1000,
                   step=-float(np.mean(within_cycle)), paired_offset=paired_offset, first_cue=first_cue)


def _wrap(angle):
    return np.angle(np.exp(1j * np.asarray(angle)))


def _polar(r, alpha):
    return r * np.cos(alpha), r * np.sin(alpha)


def _arc(r, start, stop, n=80):
    return _polar(r, np.linspace(start, stop, n))


@dataclass(frozen=True)
class SchematicTimeline:
    """Simulated cue, bump and stimulation of the whole protocol, one value per video frame.

    Attributes:
        time: Experiment time of each frame (s).
        stage: 0 baseline, 1 pairing, 2 post-pairing.
        cue: Drawing angle of the cue on the ellipsoid body.
        bump: Drawing angle of the EPG bump.
        bump_gain: Height of the bump (0-1).
        laser: Whether the stimulation laser is on.
        stim: Drawing angle of the stimulated wedge (NaN outside pairing).
        pairing: Index of the current pairing (-1 outside pairing).
        offset_drawn: Signed angle of the offset arc from the cue to the bump (rad), drawn the same way round for a
            whole stage so it does not flip sides when the offset is near pi.
    """
    time: np.ndarray
    stage: np.ndarray
    cue: np.ndarray
    bump: np.ndarray
    bump_gain: np.ndarray
    laser: np.ndarray
    stim: np.ndarray
    pairing: np.ndarray
    offset_drawn: np.ndarray

    @classmethod
    def simulate(cls, protocol, baseline_s=20., post_s=20., baseline_offset=np.pi / 2, dt=1 / 24, seed=0,
                 turn_tau=.6, turn_sd=1.2, jitter_sd=.12):
        """Simulate the protocol at time step `dt`.

        Args:
            protocol: :class:`PairingProtocol`.
            baseline_s, post_s: Durations of the closed-loop stages (s).
            baseline_offset: Bump-cue offset before pairing (rad).
            dt: Time step (s); one video frame.
            seed: Seed of the turning trajectory and bump jitter.
            turn_tau, turn_sd: Correlation time (s) and SD (rad/s) of the fly's angular velocity.
            jitter_sd: SD (rad) of the slow wander of the bump around the cue plus offset.
        """
        rng = np.random.default_rng(seed)
        n_base, n_pair, n_post = (int(round(x / dt)) for x in (baseline_s, protocol.duration, post_s))
        n = n_base + n_pair + n_post
        time = np.arange(n) * dt
        stage = np.repeat([0, 1, 2], [n_base, n_pair, n_post])

        # turning: an Ornstein-Uhlenbeck angular velocity, integrated into a heading
        velocity = np.zeros(n)
        noise = rng.standard_normal(n)
        for i in range(1, n):
            velocity[i] = velocity[i - 1] * (1 - dt / turn_tau) + turn_sd * np.sqrt(2 * dt / turn_tau) * noise[i]
        heading = np.cumsum(velocity) * dt
        jitter = sp.ndimage.gaussian_filter1d(rng.standard_normal(n), 1 / dt)
        jitter *= jitter_sd / jitter.std()

        cue = TOP + heading
        offset = np.where(stage == 2, protocol.paired_offset, baseline_offset) + jitter
        bump = cue + offset
        gain = np.ones(n)
        laser = np.zeros(n, dtype=bool)
        stim = np.full(n, np.nan)
        pairing = np.full(n, -1)

        # pairing: the cue at its position and the bump driven to the stimulated wedge
        t_pair = time[stage == 1] - baseline_s
        k = np.minimum((t_pair // protocol.period).astype(int), protocol.n_pairings - 1)
        within = t_pair - k * protocol.period
        on = within < protocol.stim_s
        cue_k = np.array([protocol.cue_angle(j) for j in k])
        stim_k = cue_k + protocol.paired_offset
        cue[stage == 1] = cue_k
        bump[stage == 1] = stim_k
        gain[stage == 1] = np.where(on, 1., .55)
        laser[stage == 1] = on
        stim[stage == 1] = stim_k
        pairing[stage == 1] = k

        # the closed-loop stages continue the cue from where the pairing leaves it, so nothing jumps
        post = stage == 2
        cue[post] = cue[post] - cue[post][0] + cue_k[-1] + protocol.step
        bump[post] = cue[post] + offset[post]

        # the arc goes the way round of the stage's nominal offset, plus the wander
        nominal = np.select([stage == 0, stage == 1, stage == 2],
                            [baseline_offset, protocol.paired_offset, protocol.paired_offset])
        nominal = np.where(np.isclose(_wrap(nominal), np.pi), -np.pi, _wrap(nominal))   # pi is drawn clockwise
        offset_drawn = nominal + _wrap(bump - cue - nominal)
        return cls(time=time, stage=stage, cue=_wrap(cue), bump=_wrap(bump), bump_gain=gain, laser=laser,
                   stim=np.where(np.isnan(stim), np.nan, _wrap(stim)), pairing=pairing, offset_drawn=offset_drawn)


STAGE_TITLES = ('baseline', 'optogenetic pairing', 'post-pairing')


class RemappingSchematicMovie:
    """Figure and artists of the animated remapping protocol; :meth:`draw` shows one frame.

    Args:
        protocol: :class:`PairingProtocol`.
        speed: Experiment seconds per video second.
        fps: Video frames per second.
        fly_image, rig_image: Paths of a top-down fly schematic (facing up) and of a picture of the imaging rig (RGBA,
            transparent background); left out if None.
        **simulate_kwargs: Passed to :meth:`SchematicTimeline.simulate`.
    """

    def __init__(self, protocol, speed=2., fps=24, fly_image=None, rig_image=None, **simulate_kwargs):
        self.protocol, self.speed, self.fps = protocol, speed, fps
        self.timeline = SchematicTimeline.simulate(protocol, dt=speed / fps, **simulate_kwargs)
        self.fig = plt.figure(figsize=[19.2, 10.8], dpi=100, facecolor='white')
        self._build_progress()
        self._build_rig(rig_image)
        self._build_arena(fly_image)
        self._build_ellipsoid_body()
        self._build_legend()

    # ------------------------------------------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------------------------------------------
    def _build_progress(self):
        """The protocol's stages as a bar across the top, with a marker at the current time."""
        tl = self.timeline
        ax = self.progress_ax = self.fig.add_axes([.24, .905, .56, .04])
        ax.set_xlim(0, tl.time[-1])
        ax.set_ylim(0, 1)
        ax.axis('off')
        bounds = [tl.time[tl.stage == s][[0, -1]] for s in range(3)]
        for s, (start, stop) in enumerate(bounds):
            ax.add_patch(Rectangle((start, .25), stop - start, .5, facecolor=PAIRING_BACKGROUND if s == 1 else
                                   ARENA_COLOR, edgecolor='none'))
            ax.text((start + stop) / 2, 1.15, STAGE_TITLES[s], ha='center', va='bottom', fontsize=14,
                    style='italic', color=PAIRING_TEXT if s == 1 else 'black')
        start = bounds[1][0]
        for c in range(1, self.protocol.n_cycles):                       # one tick per cycle of the pairing
            x = start + c * self.protocol.n_locations * self.protocol.period
            ax.plot([x, x], [.25, .75], color='white', linewidth=2)
        self.progress_marker, = ax.plot([0, 0], [0, 1], color='black', linewidth=3)

    def _build_rig(self, rig_image):
        if rig_image is None:
            return
        ax = self.fig.add_axes([.02, .74, .1, .19])
        ax.axis('off')
        ax.imshow(plt.imread(rig_image), interpolation='antialiased')

    def _build_arena(self, fly_image):
        """The fly's view: the fly fixed facing up, the cue where it sees it."""
        ax = self.arena_ax = self.fig.add_axes([.04, .3, .2, .36])
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(Circle((0, 0), 1., facecolor='none', edgecolor=ARENA_COLOR, linewidth=22))
        ax.text(0, 1.25, "the fly's view", ha='center', va='bottom', fontsize=15)
        self.arena_cue, = ax.plot([], [], color=CUE_COLOR, linewidth=22, solid_capstyle='butt')
        if fly_image is not None:
            image = plt.imread(fly_image)
            h = 1.1
            w = h * image.shape[1] / image.shape[0]
            ax.imshow(image, extent=(-w / 2, w / 2, -h / 2, h / 2), interpolation='antialiased', zorder=2)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)

    def _build_ellipsoid_body(self):
        # the pink panel of the pairing stage, behind the ellipsoid body
        self.pairing_panel = FancyBboxPatch((.335, .13), .335, .68, boxstyle='round,pad=0,rounding_size=.02',
                                            transform=self.fig.transFigure, facecolor=PAIRING_BACKGROUND,
                                            edgecolor='none', zorder=-1)
        self.fig.add_artist(self.pairing_panel)
        self.stage_title = self.fig.text(.5025, .835, '', ha='center', va='bottom', fontsize=24, style='italic')

        ax = self.eb_ax = self.fig.add_axes([.3025, .14, .4, .711])
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.75, 1.75)
        ax.set_aspect('equal')
        ax.axis('off')
        self.wedge_angles = np.arange(N_WEDGES) * 2 * np.pi / N_WEDGES + np.pi / N_WEDGES
        half = 180 / N_WEDGES
        r_in, r_out = .3, 1.
        self.wedges = []
        for alpha in self.wedge_angles:
            wedge = Wedge((0, 0), r_out, np.degrees(alpha) - half, np.degrees(alpha) + half, width=r_out - r_in,
                          facecolor=EB_COLOR, edgecolor='none', zorder=1)
            ax.add_patch(wedge)
            self.wedges.append(wedge)
        for alpha in self.wedge_angles - np.pi / N_WEDGES:
            ax.plot(*zip(_polar(r_in, alpha), _polar(r_out, alpha), strict=True), color='black', linestyle='--',
                    linewidth=1.2, zorder=2)
        ax.add_patch(Circle((0, 0), r_out, facecolor='none', edgecolor='black', linewidth=2.5, zorder=3))
        ax.add_patch(Circle((0, 0), r_in, facecolor='white', edgecolor='black', linewidth=2.5, zorder=3))

        # the 8 stimulation targets, fixed in the brain
        targets = [_wrap(self.protocol.cue_angle(k) + self.protocol.paired_offset)
                   for k in range(self.protocol.n_locations)]
        self.targets = [ax.add_patch(Circle(_polar(.65, a), .055, facecolor='none', edgecolor=LASER_COLOR,
                                            linewidth=2, zorder=4)) for a in targets]
        self.target_angles = np.array(targets)

        # the laser: stacked translucent triangles, narrow and dense near the target, wide and faint outside
        self.laser_layers = []
        for width, alpha in ((.2, .1), (.15, .14), (.1, .2), (.06, .3)):
            layer = Polygon(np.zeros((3, 2)), closed=True, facecolor=LASER_COLOR, edgecolor='none', alpha=alpha,
                            zorder=5)
            ax.add_patch(layer)
            self.laser_layers.append((width, layer))

        self.eb_cue, = ax.plot([], [], color=CUE_COLOR, linewidth=26, solid_capstyle='butt', zorder=6)
        self.offset_arc, = ax.plot([], [], color=OFFSET_COLOR, linewidth=4, zorder=6)
        self.offset_head = FancyArrowPatch((0, 0), (0, 0), arrowstyle='-|>', mutation_scale=30, color=OFFSET_COLOR,
                                           linewidth=3, zorder=6)
        ax.add_patch(self.offset_head)
        self.offset_label = ax.text(0, 0, '', ha='center', va='center', fontsize=16, color=OFFSET_COLOR, zorder=7)
        self.counter = self.fig.text(.5025, .785, '', ha='center', va='center', fontsize=15, color=PAIRING_TEXT)

    def _build_legend(self):
        x, y, dy = .76, .62, .065
        items = ((CUE_COLOR, 'visual cue'), (BUMP_COLOR, 'EPG activity bump'), (LASER_COLOR, 'optogenetic stimulation'),
                 (OFFSET_COLOR, 'bump-cue offset'))
        for i, (color, label) in enumerate(items):
            yi = y - dy * i
            self.fig.add_artist(Rectangle((x, yi - .014), .012, .028, color=color, transform=self.fig.transFigure,
                                          alpha=.8 if color == LASER_COLOR else 1))
            self.fig.text(x + .02, yi, label, va='center', fontsize=16)

    # ------------------------------------------------------------------------------------------------------
    # One frame
    # ------------------------------------------------------------------------------------------------------
    def draw(self, i):
        """Update every artist to frame `i`."""
        tl = self.timeline
        stage = int(tl.stage[i])
        self.progress_marker.set_xdata([tl.time[i], tl.time[i]])
        self.stage_title.set_text(STAGE_TITLES[stage])
        self.stage_title.set_color(PAIRING_TEXT if stage == 1 else 'black')
        self.pairing_panel.set_visible(stage == 1)

        # the fly's view: the cue slips across it the same way round as it moves on the ellipsoid body
        seen = tl.cue[i]
        self.arena_cue.set_data(*_arc(1., seen - .2, seen + .2))

        # the bump
        base, red = np.array(to_rgb(EB_COLOR)), np.array(to_rgb(BUMP_COLOR))
        activity = tl.bump_gain[i] * np.exp(BUMP_KAPPA * (np.cos(self.wedge_angles - tl.bump[i]) - 1))
        for wedge, a in zip(self.wedges, activity, strict=True):
            wedge.set_facecolor(base + (red - base) * a)

        # the cue and the offset arc from it to the bump
        self.eb_cue.set_data(*_arc(1.14, tl.cue[i] - np.pi / 16, tl.cue[i] + np.pi / 16))
        offset = tl.offset_drawn[i]
        x, y = _arc(1.36, tl.cue[i], tl.cue[i] + offset)
        self.offset_arc.set_data(x[:-3], y[:-3])
        self.offset_head.set_positions((x[-4], y[-4]), (x[-1], y[-1]))
        self.offset_label.set_text('opto-paired\noffset' if stage == 1 else 'bump-cue\noffset')
        self.offset_label.set_position(_polar(1.6, tl.cue[i] + offset / 2))

        # the stimulation
        pairing = stage == 1
        for target, angle in zip(self.targets, self.target_angles, strict=True):
            target.set_visible(pairing)
            current = pairing and abs(_wrap(angle - tl.stim[i])) < 1e-6
            target.set_facecolor(LASER_COLOR if current and tl.laser[i] else 'none')
        for width, layer in self.laser_layers:
            layer.set_visible(bool(tl.laser[i]))
            if tl.laser[i]:
                a = tl.stim[i]
                layer.set_xy([_polar(.65, a), _polar(1.8, a - width * 1.6), _polar(1.8, a + width * 1.6)])
        self.counter.set_text(f'pairing {tl.pairing[i] + 1} of {self.protocol.n_pairings}   '
                              f'({self.protocol.n_locations} angles x {self.protocol.n_cycles} cycles)'
                              if pairing else '')

    def snapshot(self, i, path, dpi=100):
        """Save frame `i` as an image."""
        self.draw(i)
        self.fig.savefig(path, dpi=dpi, facecolor='white')

    def frame_at(self, t):
        """Index of the frame at experiment time `t` (s)."""
        return int(np.clip(np.searchsorted(self.timeline.time, t), 0, len(self.timeline.time) - 1))

    def save(self, outfile, dpi=100, bitrate=6000):
        """Write the movie to `outfile` (MP4, H.264, 1920 x 1080 at the default dpi)."""
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = animation.FFMpegWriter(fps=self.fps, codec='libx264', bitrate=bitrate,
                                        extra_args=['-pix_fmt', 'yuv420p'])
        outfile = pathlib.Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with writer.saving(self.fig, str(outfile), dpi):
            for i in range(len(self.timeline.time)):
                self.draw(i)
                writer.grab_frame(facecolor='white')
        return outfile


# ----------------------------------------------------------------------------
# A fly's sessions, appended
# ----------------------------------------------------------------------------
SCAN_LABELS = {'baseline': 'baseline', 'remap_180deg': 'pairing (180 deg)', 'post_180deg': 'post-pairing (180 deg)',
               'remap_0deg': 'pairing (0 deg)', 'post_0deg': 'post-pairing (0 deg)'}
SCREEN_OFF_INDEX = 5.               # V on the Y/Index channel above which the arena screen is off (no visual cue)
# the processing of session.GetTS, applied to the pairing scans after their stimulation frames are masked
T_SIGMA, CIRC_SIGMA, NEURAL_SHIFT = .1, .5, -.2


@dataclass(frozen=True)
class SessionSequence:
    """Sessions appended into one timeseries, with gaps between them; NaN where nothing was imaged.

    Has the fields :class:`session_movie.ClosedLoopMovie` reads from a timeseries (``dff`` (n_rois, T), ``heading``,
    ``phi``, ``rho``, ``offset``, ``time``, ``dt``) and the arguments it takes for an appended sequence.

    Attributes:
        cue_visible: Whether the arena showed the visual cue (False when the screen was off, see
            :data:`SCREEN_OFF_INDEX`).
        accumulate: Whether a sample adds to the offset histogram: it belongs to a closed-loop scan (baseline or
            post-pairing; pairing scans are left out entirely) and the cue was shown.
        pairing: Whether the sample lies in a pairing block (one pairing period before the first stimulation to one
            after the last).
        stimulating: Whether the sample lies in the stimulation block: from the first stimulation to the end of the
            imaging window after the last (the PVA is not drawn there).
        stim_theta: Ring angle (as ``heading``) of the wedge being stimulated, NaN when no stimulation is on.
        stim_targets: Ring angles of the stimulation targets of the pairing protocol.
        epochs: :class:`session_movie.ColorEpoch` of the baseline and of each pairing, from its first stimulation.
        segments: (start, stop, label) of every session, and (start, stop, 'break', False) of the breaks between them
            (labeled without a time: their real length was not recorded).
        sessions: (scan type, session name) of the sessions, in order.
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
    pairing: np.ndarray
    stimulating: np.ndarray
    stim_theta: np.ndarray
    stim_targets: tuple
    epochs: tuple
    segments: tuple
    sessions: tuple


def stim_angle(k):
    """Ring angle (as the heading: -pi to pi) of the wedge stimulated at the `k`-th pairing of a pairing scan."""
    return float(np.angle(np.exp(1j * (-np.pi + remapping.stimulated_roi(k) * np.pi / 8))))


def _pairing_scan_arrays(pp):
    """dF/F, heading and population vector of a pairing scan, its stimulation frames masked (NaN), processed as
    :func:`session.GetTS` processes a session (time and ring smoothing, -200 ms neural shift), and its pairing block."""
    onsets, n_frames, dff = remapping.masked_pairing_activity(pp)
    voltage = pp.voltage_recording_aligned
    time = (voltage['Time(ms)'].to_numpy() - voltage['Time(ms)'].iloc[0]) / 1000
    dt = float(np.diff(time).mean())
    dff = nan_gaussian(dff, T_SIGMA / dt, axis=-1)
    dff = sp.ndimage.gaussian_filter1d(dff, CIRC_SIGMA, axis=-2, mode='wrap')      # masked frames are NaN in every ROI
    dff = session._shift_neural(dff, dt, NEURAL_SHIFT)
    heading = np.angle(np.exp(1j * (-voltage[' Heading'].to_numpy() - np.pi)))
    _, _, rho, phi, offset = session._population_vector(dff, heading)
    # the stimulated wedge during each stimulation. The analysis's stimulated ROI (remapping.stimulated_roi) sits half
    # an ROI from the cue in its index convention; the angle below is the cue's for the 0 degree pairing and the
    # opposite one for the 180 degree pairing, as the protocol intends
    stim_theta = np.full(len(time), np.nan)
    for k, onset in enumerate(onsets):
        stim_theta[onset:onset + n_frames] = stim_angle(k)
    # the pairing block, padded by one pairing period on each side: the cue sits at a pairing position for the gap
    # after each stimulation too, so the frames just around the first and last stimulation belong to the protocol
    pairing = np.zeros(len(time), dtype=bool)
    stimulating = np.zeros(len(time), dtype=bool)        # the stimulations and the imaging windows between them
    if len(onsets):
        period = int(np.median(np.diff(onsets))) if len(onsets) > 1 else n_frames
        pairing[max(onsets[0] - period, 0):onsets[-1] + period] = True
        stimulating[onsets[0]:onsets[-1] + period] = True
    return dict(dff=dff, heading=heading, phi=phi, rho=rho, offset=offset, time=time, dt=dt, pairing=pairing,
                stimulating=stimulating, stim_theta=stim_theta, first_stim=int(onsets[0]) if len(onsets) else None)


def _closed_loop_arrays(pp):
    ts = session.GetTS(pp)
    time = np.asarray(ts.time, dtype=float)
    return dict(dff=ts.dff, heading=ts.heading, phi=ts.phi, rho=ts.rho, offset=ts.offset, time=time, dt=float(ts.dt),
                pairing=np.zeros(len(time), dtype=bool), stimulating=np.zeros(len(time), dtype=bool),
                stim_theta=np.full(len(time), np.nan), first_stim=None)


def remap_session_sequence(session_manager, fly_id, gap_s=15., scan_types=tuple(SCAN_LABELS)):
    """A fly's remapping sessions appended in the order they were acquired, `gap_s` seconds apart.

    Closed-loop scans are loaded as for the session heatmaps (:func:`session.GetTS`); pairing scans as in the
    optogenetic-response analysis, with the frames of every stimulation masked, then smoothed and shifted the same way.
    The colormap is the baseline's (Greys) until the first stimulation of the first pairing scan, then that pairing's
    (:data:`remapping.PAIRING_CMAPS`) until the first stimulation of the next; each epoch's offset histogram is filled
    with its colormap at 0.8, at alpha 0.5, as in the session figures, and accumulates over the closed-loop scans
    only (baseline and post-pairing, while the cue is shown), so it is the histogram of that scan's session figure.

    Args:
        session_manager: Session manager of the experiment (``scan_type`` column).
        fly_id: The fly.
        gap_s: Break between consecutive sessions, in experiment seconds (drawn blank).
        scan_types: Scan types to include.

    Returns:
        :class:`SessionSequence`.
    """
    df = session_manager.sess_df
    rows = df.loc[(df['fly_id'] == fly_id) & df['scan_type'].isin(scan_types)]
    if rows.empty:
        raise ValueError(f'no sessions of {fly_id} with scan types {scan_types}')
    rows = rows.iloc[np.argsort([int(name[-3:]) for name in rows['session']])]

    parts = []
    for _, row in rows.iterrows():
        pp = session_manager.load_row(row)
        arrays = (_pairing_scan_arrays if row['scan_type'] in remapping.OPTO_SCANS else _closed_loop_arrays)(pp)
        index = pp.voltage_recording_aligned.get(' Y/Index')
        arrays['cue_visible'] = (np.ones(len(arrays['time']), dtype=bool) if index is None
                                 else index.to_numpy() <= SCREEN_OFF_INDEX)
        parts.append((row['scan_type'], row['session'], arrays))

    dt = parts[0][2]['dt']
    for _, name, arrays in parts:
        if abs(arrays['dt'] / dt - 1) > .01:
            raise ValueError(f'{name} is sampled every {arrays["dt"]:.4f} s, not {dt:.4f} s like the first session')
    n_gap = int(round(gap_s / dt))
    n_rois = parts[0][2]['dff'].shape[0]

    def gap(fill, dtype=float, rows=None):
        return np.full((rows, n_gap) if rows else n_gap, fill, dtype=dtype)

    for scan_type, _, arrays in parts:
        arrays['closed_loop'] = np.full(len(arrays['time']), scan_type not in remapping.OPTO_SCANS)
    keys = ('dff', 'heading', 'phi', 'rho', 'offset', 'cue_visible', 'pairing', 'stimulating', 'stim_theta',
            'closed_loop')
    pieces = {key: [] for key in keys}
    starts, epochs_at = [], []
    n = 0
    for k, (scan_type, _, arrays) in enumerate(parts):
        if k:
            for key in ('heading', 'phi', 'rho', 'offset', 'stim_theta'):
                pieces[key].append(gap(np.nan))
            pieces['dff'].append(gap(np.nan, rows=n_rois))
            pieces['cue_visible'].append(gap(True, bool))
            pieces['pairing'].append(gap(False, bool))
            pieces['stimulating'].append(gap(False, bool))
            pieces['closed_loop'].append(gap(False, bool))
            n += n_gap
        starts.append(n)
        for key in pieces:
            pieces[key].append(np.asarray(arrays[key]))
        if arrays['first_stim'] is not None:
            epochs_at.append((n + arrays['first_stim'], scan_type))
        n += len(arrays['time'])
    data = {key: np.concatenate(values, axis=-1) for key, values in pieces.items()}
    time = np.arange(n) * dt

    epochs = [ColorEpoch(time[0], 'Greys', plt.get_cmap('Greys')(.8), .5)]
    epochs += [ColorEpoch(time[i], remapping.PAIRING_CMAPS[scan_type], plt.get_cmap(remapping.PAIRING_CMAPS[scan_type])(.8),
                          .5) for i, scan_type in epochs_at]
    segments = []
    for k, ((scan_type, _, arrays), start) in enumerate(zip(parts, starts, strict=True)):
        if k:
            # the real time between sessions was not recorded, so the break is labeled without a clock
            segments.append((time[start - n_gap], time[start - 1], 'break', False))
        segments.append((time[start], time[start + len(arrays['time']) - 1], SCAN_LABELS.get(scan_type, scan_type)))
    accumulate = data['closed_loop'] & data['cue_visible'] & np.isfinite(data['offset'])
    targets = tuple(stim_angle(k) for k in range(remapping.N_STIM_POINTS))
    return SessionSequence(dff=data['dff'], heading=data['heading'], phi=data['phi'], rho=data['rho'],
                           offset=data['offset'], time=time, dt=dt, cue_visible=data['cue_visible'],
                           accumulate=accumulate, pairing=data['pairing'], stimulating=data['stimulating'],
                           stim_theta=data['stim_theta'],
                           stim_targets=targets, epochs=tuple(epochs),
                           segments=tuple(segments), sessions=tuple((s, name) for s, name, _ in parts))


def remap_sequence_movie(session_manager, fly_id, outfile, vmax, gap_s=15., speed=5., window=180.,
                         snapshot_times=None, fly_image=None, rig_image=None, **kwargs):
    """Movie of all of a fly's remapping sessions, appended (:func:`remap_session_sequence`), with the stimulation.

    Every stimulation is drawn on the ellipsoid body as a laser cone at the stimulated wedge, with the protocol's
    stimulation targets marked during each pairing block, as in the remapping figure's protocol panel.

    Args:
        vmax: Upper color limit (the session figures' limit of the experiment).
        snapshot_times: If given, save PNG frames at these times (s of the appended sequence) next to `outfile`
            instead of the movie.
        **kwargs: Passed to :class:`session_movie.ClosedLoopMovie`.

    Returns:
        Tuple of (path of the movie, or the list of snapshot paths; the :class:`SessionSequence`).
    """
    seq = remap_session_sequence(session_manager, fly_id, gap_s=gap_s)
    movie = ClosedLoopMovie(seq, speed=speed, window=window, vmin=-.5, vmax=vmax, epochs=seq.epochs,
                            accumulate=seq.accumulate, cue_visible=seq.cue_visible, segments=seq.segments,
                            unsmoothed=seq.pairing, pva_hidden=seq.stimulating, stim_theta=seq.stim_theta,
                            stim_targets=seq.stim_targets,
                            stim_block=seq.pairing, fly_image=fly_image, rig_image=rig_image, **kwargs)
    outfile = pathlib.Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        if snapshot_times is not None:
            paths = [outfile.with_name(f'{outfile.stem}_t{t:06.1f}.png') for t in snapshot_times]
            for t, path in zip(snapshot_times, paths, strict=True):
                movie.snapshot(t, path)
            return paths, seq
        return movie.save(outfile), seq
    finally:
        plt.close(movie.fig)

