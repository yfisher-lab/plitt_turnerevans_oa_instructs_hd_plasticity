"""GRAB sensor analyses that are not already covered by the shared statistics tests."""
import numpy as np

import el_hd_plasticity.two_photon as tp
from el_hd_plasticity.two_photon import grab_sensors as gs

ROI_RAD = 2 * np.pi / tp.utilities.N_EB_ROIS


def bump_centers(profiles):
    """Bump position of each profile in (fractional) ROI indices, from its first-harmonic phase."""
    return (-np.angle(np.fft.fft(profiles, axis=-1)[..., 1]) / ROI_RAD) % tp.utilities.N_EB_ROIS


def test_split_halves_align_the_first_half_to_the_center_roi_to_sub_roi_precision(manager):
    first, second = gs.align_peaks_split_halves(manager, rot_speed_mask=False)
    np.testing.assert_allclose(bump_centers(first), tp.utilities.HEADING_CENTER_ROI, atol=1e-6)

    by_max, _ = gs.align_peaks_split_halves(manager, rot_speed_mask=False, alignment='max')
    assert np.abs(bump_centers(by_max) - tp.utilities.HEADING_CENTER_ROI).max() > .05   # whole ROIs only


def test_split_halves_shift_both_halves_by_the_same_amount(manager):
    """The drift between the halves is the measurement, so alignment must not change it."""
    first, second = gs.align_peaks_split_halves(manager, rot_speed_mask=False)
    by_max_first, by_max_second = gs.align_peaks_split_halves(manager, rot_speed_mask=False, alignment='max')
    drift = (bump_centers(second) - bump_centers(first) + 8) % 16 - 8
    max_drift = (bump_centers(by_max_second) - bump_centers(by_max_first) + 8) % 16 - 8
    np.testing.assert_allclose(drift, max_drift, atol=1e-6)
