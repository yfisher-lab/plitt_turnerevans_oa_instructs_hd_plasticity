import numpy as np
import pytest

from el_hd_plasticity.two_photon import bump_shape as ra

N = 16
ANGLES = np.linspace(-np.pi, np.pi, N, endpoint=False)


def bump(center, kappa=3.0):
    """Von Mises bump around the ring, shape (len(center), N)."""
    return np.exp(kappa * np.cos(ANGLES[None, :] - np.atleast_1d(center)[:, None]))


def test_fourier_upsample_interpolates_through_original_samples():
    F = bump(np.array([0.3, -1.0]))
    up = ra.fourier_upsample(F, factor=8)
    assert up.shape == (2, N * 8)
    np.testing.assert_allclose(up[:, ::8], F, atol=1e-10)


def test_phase_null_recovers_phase_and_centers_the_peak():
    centers = np.array([-2.0, 0.4, 2.5])
    F_centered, phase, _, rho, harmonic_ratio = ra.phase_null(bump(centers))
    # fft convention: phase = -(bump angle measured from the first sample, which sits at -pi)
    expected = -(centers + np.pi)
    np.testing.assert_allclose(np.cos(phase), np.cos(expected), atol=1e-2)
    np.testing.assert_allclose(np.sin(phase), np.sin(expected), atol=1e-2)
    assert np.all(np.argmax(F_centered, axis=1) == N // 2)
    assert np.all((rho > 0) & (harmonic_ratio > 0))


@pytest.mark.parametrize('kappa', [1.0, 3.0, 6.0])
def test_fwhm_matches_analytic_von_mises_width(kappa):
    # half-max of exp(k cos x) between trough exp(-k) and peak exp(k)
    half = (np.exp(kappa) + np.exp(-kappa)) / 2
    expected = 2 * np.arccos(np.log(half) / kappa)
    width = ra.fwhm(bump(0.7, kappa), factor=64)
    assert width.shape == (1,)
    np.testing.assert_allclose(width[0], expected, rtol=2e-2)


def test_fwhm_nan_for_flat_profile():
    assert np.isnan(ra.fwhm(np.ones((1, N)))[0])


def test_mean_profile_fwhm_counts_rois_above_half_max():
    dff = np.zeros((N, 5))
    dff[:4] = 1.0                      # 4 of 16 ROIs active
    dff = np.stack([np.roll(dff[:, 0], s) for s in range(5)], axis=1)
    assert np.isclose(ra.mean_profile_fwhm(dff), 4 * 2 * np.pi / N)


def test_shift_by_max_puts_reference_peak_at_center_and_mutates_in_place():
    rng = np.random.default_rng(0)
    arr = np.stack([bump(rng.uniform(-3, 3, 20)).T, bump(rng.uniform(-3, 3, 20)).T])   # (2, N, 20)
    out = ra.shift_by_max(arr, 0)
    assert out is arr
    assert np.all(np.argmax(arr[0], axis=0) == N // 2)


def test_shift_by_pva_centers_each_time_point_on_its_pva():
    centers = np.array([-2.0, 0.0, 2.0])
    arr = bump(centers).T[None]                                  # (1, N, 3)
    pva = centers[None, :]
    out = ra.shift_by_pva(arr.copy(), pva, 0)
    # the rolled peak lies within one ROI of the center for every time point
    assert np.all(np.abs(np.argmax(out[0], axis=0) - N // 2) <= 1)
