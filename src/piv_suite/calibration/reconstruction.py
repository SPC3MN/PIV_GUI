"""Two-camera in-plane displacement -> 3-component reconstruction.

Migrated unchanged from stereo_common.reconstruct_stereo.
"""

import numpy as np


def reconstruct_stereo(dx1, dy1, dx2, dy2, alpha1, alpha2, beta1, beta2):
    """Combine two cameras' in-plane displacement fields (on a common
    dewarped grid) into 3-component displacement (dX, dY, dZ) by solving,
    in a least-squares sense:
        dx1 = dX          - dZ*tan(alpha1)
        dx2 = dX          - dZ*tan(alpha2)
        dy1 =       dY    - dZ*tan(beta1)
        dy2 =       dY    - dZ*tan(beta2)
    This handles degenerate cases automatically -- e.g. beta1 == beta2, as
    in a rig where the two cameras are tilted apart in only one plane (no
    relative vertical tilt): the y-pair becomes redundant rather than a
    divide-by-zero, and dZ is correctly identified from the x-pair alone.
    Angles may be scalars or arrays matching dx1's shape."""
    ta1, ta2 = np.tan(alpha1), np.tan(alpha2)
    tb1, tb2 = np.tan(beta1), np.tan(beta2)
    shape = np.broadcast(dx1, ta1, ta2, tb1, tb2).shape
    ta1b, ta2b, tb1b, tb2b = (np.broadcast_to(np.asarray(a, dtype=float), shape)
                               for a in (ta1, ta2, tb1, tb2))
    zeros, ones = np.zeros(shape), np.ones(shape)
    A = np.stack([
        np.stack([ones,  zeros, -ta1b], axis=-1),
        np.stack([ones,  zeros, -ta2b], axis=-1),
        np.stack([zeros, ones,  -tb1b], axis=-1),
        np.stack([zeros, ones,  -tb2b], axis=-1),
    ], axis=-2)  # (..., 4, 3)
    b = np.stack(np.broadcast_arrays(dx1, dx2, dy1, dy2), axis=-1)  # (..., 4)
    AtA = np.einsum('...ki,...kj->...ij', A, A)   # (..., 3, 3)
    Atb = np.einsum('...ki,...k->...i', A, b)     # (..., 3)
    sol = np.linalg.solve(AtA, Atb[..., None])[..., 0]  # (..., 3)
    return sol[..., 0], sol[..., 1], sol[..., 2]
