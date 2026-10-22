"""Optogenetic responses of ex vivo brain preparations (GFlamp imaging of the outer ring).

In these preparations the EL neurons are stimulated with a series of optogenetic pulses (CsChrimson, mark points of
the two-photon microscope) while a GFlamp sensor in the postsynaptic cells (ER or EPG neurons) is imaged. Each
pulse's response is baseline-normalized and summarized by its peak and area; :class:`OptoResponses` holds
these for one session and :func:`session_responses` collects them for a set of session managers.

Session tables need ``condition`` and ``session`` columns; the preprocessed sessions need ``metadata`` with
``frame_times`` and ``mark_points`` (``initial_delay``, ``duration``, ``inter_point_delay``, ``repetitions`` in
ms) and ``timeseries['outer_ring']``.
"""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

# Window around each pulse (s, relative to the pulse) and the sampling of the common time axis.
WINDOW = (-2, 8)
STEP = .1
# The imaging frames around the pulse are contaminated by the stimulation light and left out (s).
ARTIFACT = (-.2, .22)


@dataclass(frozen=True, eq=False, repr=False)
class OptoResponses:
    """Responses of one session to its optogenetic pulses; immutable.

    The NumPy arrays among the fields are frozen (``resp.dff[0] = 0`` raises ``ValueError``).

    Attributes:
        fly_id, condition, session: Identify the session.
        frame_times: Time of each imaging frame (s).
        stim_times: Time of each pulse (s).
        f: Fluorescence around each pulse, shape (n_pulses, n_times), on the common time axis `plot_time`
            (NaN in the artifact window).
        dff: Baseline-normalized responses ``f / baseline - 1``, same shape; the baseline is the mean of `f`
            before the artifact window.
        aoc: Area under the response curve of each pulse (dF/F x s): the integral of `dff` after the artifact
            window, as a sum over the samples times the sampling step (n_pulses,).
        max_dff: Peak of `dff` after the artifact window (n_pulses,).
        plot_time: Common time axis (s, relative to the pulse).
    """

    fly_id: str
    condition: str
    session: str
    frame_times: np.ndarray
    stim_times: np.ndarray
    f: np.ndarray
    dff: np.ndarray
    aoc: np.ndarray
    max_dff: np.ndarray
    plot_time: np.ndarray

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def __repr__(self):
        return f"{type(self).__name__}(fly_id={self.fly_id}, condition={self.condition}, session={self.session})"

    @classmethod
    def from_session(cls, fly_id, condition, session, pp, window=WINDOW, artifact=ARTIFACT):
        """Responses of a preprocessed session.

        Args:
            fly_id, condition, session: Identify the session.
            pp: Preprocessed session with ``metadata`` and ``timeseries['outer_ring']`` (last channel, first ROI).
            window: (start, stop) of the window around each pulse (s).
            artifact: (start, stop) of the window left out around each pulse (s).
        """
        frame_times = np.array(pp.metadata['frame_times']).mean(axis=1)
        stim_times = cls.stim_times_from_metadata(pp.metadata['mark_points'])
        f, dff, aoc, max_dff, plot_time = cls._responses(pp.timeseries['outer_ring'], frame_times, stim_times,
                                                         window, artifact)
        return cls(fly_id=fly_id, condition=condition, session=session, frame_times=frame_times,
                   stim_times=stim_times, f=f, dff=dff, aoc=aoc, max_dff=max_dff, plot_time=plot_time)

    @staticmethod
    def stim_times_from_metadata(mark_points):
        """Times (s) of the pulses of a mark-points protocol (all durations in ms)."""
        step = (mark_points['duration'] + mark_points['inter_point_delay']) / 1000
        return mark_points['initial_delay'] / 1000 + step * np.arange(int(np.squeeze(mark_points['repetitions'])))

    @staticmethod
    def _responses(f, frame_times, stim_times, window, artifact):
        start_time, stop_time = window
        plot_time = np.arange(start_time, stop_time, STEP)
        responses = []
        for stim_time in stim_times:
            id_stim = np.argmin(np.abs(frame_times - stim_time))
            id_start = np.argmin(np.abs(frame_times - (stim_time + start_time)))
            id_end = np.argmin(np.abs(frame_times - (stim_time + stop_time)))
            sample_times = frame_times[id_start:id_end] - frame_times[id_stim]
            # all pulses on the same clock
            responses.append(np.interp(plot_time, sample_times, f[-1, 0, id_start:id_end]))
        responses = np.vstack(responses)

        responses[:, (plot_time >= artifact[0]) & (plot_time <= artifact[1])] = np.nan
        baseline = np.nanmean(responses[:, plot_time < artifact[0]], axis=1, keepdims=True)
        dff = responses / baseline - 1
        after = plot_time > artifact[1]
        return responses, dff, np.nansum(dff[:, after], axis=-1) * STEP, np.amax(dff[:, after], axis=-1), plot_time


def session_responses(session_managers, **kwargs):
    """Summarize the responses of every session of several preparations.

    Args:
        session_managers: Dict cell type -> session manager (each ``sess_df`` needs ``condition`` and ``session``).
        **kwargs: Passed to :meth:`OptoResponses.from_session`.

    Returns:
        Tuple of (DataFrame, plot_time). The DataFrame has one row per session with cell_type, fly_id, condition,
        session, ``mean_dff`` (the response averaged over the pulses, on `plot_time`), ``aoc`` and ``max_dff``
        (averaged over the pulses).
    """
    rows = []
    for cell_type, session_manager in session_managers.items():
        for _, row in session_manager.iterrows():
            resp = OptoResponses.from_session(row['fly_id'], row['condition'], row['session'],
                                              session_manager.load_row(row), **kwargs)
            rows.append({'cell_type': cell_type, 'fly_id': resp.fly_id, 'condition': resp.condition,
                         'session': resp.session, 'mean_dff': resp.dff.mean(axis=0), 'aoc': resp.aoc.mean(),
                         'max_dff': resp.max_dff.mean()})
            plot_time = resp.plot_time
    return pd.DataFrame(rows), plot_time
