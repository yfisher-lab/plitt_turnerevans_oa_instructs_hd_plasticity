"""The remapping schematic runs the pairing protocol it is given, and reads that protocol from a pairing scan."""
import numpy as np
from matplotlib import pyplot as plt

import el_hd_plasticity.two_photon as tp

rm = tp.remapping_movie


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


def test_pairing_schedule_steps_the_cue_and_keeps_the_paired_offset():
    protocol = rm.PairingProtocol(n_locations=4, n_cycles=2, stim_s=1., gap_s=.5, step=-np.pi / 2,
                                  paired_offset=np.pi)
    tl = rm.SchematicTimeline.simulate(protocol, baseline_s=3, post_s=3, dt=.1)
    pairing = tl.stage == 1
    assert np.isclose(pairing.sum() * .1, protocol.duration)
    assert set(tl.pairing[pairing]) == set(range(protocol.n_pairings))
    # the stimulated wedge is always the paired offset away from the cue, and the bump is driven there
    assert np.allclose(wrap(tl.stim[pairing] - tl.cue[pairing] - np.pi), 0, atol=1e-9)
    assert np.allclose(wrap(tl.bump[pairing] - tl.stim[pairing]), 0, atol=1e-9)
    # the cue steps by `step` from one pairing to the next and comes back after a cycle
    cues = [tl.cue[pairing][tl.pairing[pairing] == k][0] for k in range(protocol.n_pairings)]
    assert np.allclose(wrap(np.diff(cues[:4]) - protocol.step), 0, atol=1e-9)
    assert np.isclose(wrap(cues[4] - cues[0]), 0)
    # the laser is on for the stimulation and off for the gap of every pairing
    assert np.isclose(tl.laser[pairing].mean(), protocol.stim_s / protocol.period, atol=.05)


def test_post_pairing_offset_is_the_paired_offset_and_baseline_is_not():
    protocol = rm.PairingProtocol(n_locations=2, n_cycles=1, paired_offset=np.pi)
    tl = rm.SchematicTimeline.simulate(protocol, baseline_s=20, post_s=20, baseline_offset=np.pi / 2, dt=.1)
    def mean_offset(stage):
        return np.angle(np.mean(np.exp(1j * (tl.bump - tl.cue)[tl.stage == stage])))
    assert abs(wrap(mean_offset(0) - np.pi / 2)) < .2
    assert abs(wrap(mean_offset(2) - np.pi)) < .2
    # the drawn arc keeps to one side of the ring through a stage, rather than flipping around pi
    assert np.all(tl.offset_drawn[tl.stage == 2] < 0)


def test_movie_draws_every_stage():
    protocol = rm.PairingProtocol(n_locations=2, n_cycles=1)
    movie = rm.RemappingSchematicMovie(protocol, speed=4., fps=4, baseline_s=4, post_s=4)
    for stage in range(3):
        i = int(np.flatnonzero(movie.timeline.stage == stage)[0])
        movie.draw(i)
        assert movie.stage_title.get_text() == rm.STAGE_TITLES[stage]
        assert movie.pairing_panel.get_visible() == (stage == 1)
    movie.fig.canvas.draw()
    plt.close(movie.fig)


class _FakePairingScan:
    """A pairing scan with a placeholder point, 3 points x 2 cycles, and a bar stepping +pi/4 per stimulation."""

    def __init__(self):
        points = [{'Index': 9.}] + [{'Index': float(i)} for i in (1, 2, 3, 1, 2, 3)]
        self.metadata = {'mark_points': {'points_list': np.array(points, dtype=object), 'duration': np.array(1800.),
                                         'inter_point_delay': np.array(200.)}}
        n, onsets = 700, np.arange(6) * 100 + 50
        trigger = np.zeros(n)
        bar = np.zeros(n)
        for k, onset in enumerate(onsets):
            trigger[onset:onset + 20] = 3000
            bar[onset:] = k * np.pi / 4
        import pandas as pd
        self.voltage_recording_aligned = pd.DataFrame({' Opto Trigger': trigger, ' Heading': wrap(bar)})


def test_protocol_is_read_from_a_pairing_scan():
    protocol = rm.PairingProtocol.from_session(_FakePairingScan(), paired_offset=0.)
    assert (protocol.n_locations, protocol.n_cycles) == (3, 2)
    assert np.isclose(protocol.stim_s, 1.8) and np.isclose(protocol.gap_s, .2)
    assert np.isclose(protocol.step, -np.pi / 4)           # the cue on the ellipsoid body moves against the bar

