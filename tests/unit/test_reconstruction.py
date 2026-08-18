import numpy as np

from piv_suite.calibration.reconstruction import reconstruct_stereo


def test_reconstruct_stereo_recovers_known_displacement_scalar_angles():
    alpha1, alpha2 = np.deg2rad(-44.765), np.deg2rad(44.765)
    beta1, beta2 = 0.0, 0.0  # degenerate case, matches the original rig's config
    dX, dY, dZ = 1.5, -2.0, 3.0

    dx1 = dX - dZ * np.tan(alpha1)
    dx2 = dX - dZ * np.tan(alpha2)
    dy1 = dY - dZ * np.tan(beta1)
    dy2 = dY - dZ * np.tan(beta2)

    X, Y, Z = reconstruct_stereo(dx1, dy1, dx2, dy2, alpha1, alpha2, beta1, beta2)
    assert np.isclose(X, dX, atol=1e-8)
    assert np.isclose(Y, dY, atol=1e-8)
    assert np.isclose(Z, dZ, atol=1e-8)


def test_reconstruct_stereo_recovers_known_displacement_field():
    rng = np.random.default_rng(1)
    ny, nx = 6, 7
    alpha1, alpha2 = np.deg2rad(-30.0), np.deg2rad(35.0)
    beta1, beta2 = np.deg2rad(2.0), np.deg2rad(-3.0)  # non-degenerate

    dX = rng.uniform(-5, 5, size=(ny, nx))
    dY = rng.uniform(-5, 5, size=(ny, nx))
    dZ = rng.uniform(-2, 2, size=(ny, nx))

    dx1 = dX - dZ * np.tan(alpha1)
    dx2 = dX - dZ * np.tan(alpha2)
    dy1 = dY - dZ * np.tan(beta1)
    dy2 = dY - dZ * np.tan(beta2)

    X, Y, Z = reconstruct_stereo(dx1, dy1, dx2, dy2, alpha1, alpha2, beta1, beta2)
    np.testing.assert_allclose(X, dX, atol=1e-8)
    np.testing.assert_allclose(Y, dY, atol=1e-8)
    np.testing.assert_allclose(Z, dZ, atol=1e-8)
