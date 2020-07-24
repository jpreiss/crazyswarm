"""Useful functions for both pycrazyswarm internals and user scripts."""

import numpy as np


def pairwise_squared_distance(X):
    """Computes the pairwise squared distances between rows of a data table.

    Implemented because there is something strange about scipy on anaconda.

    Args:
        X (np.ndarray, n x d): Data table.

    Returns:
        D (np.ndarray, n x n): Matrix such that
                               D[i, j] == ||X[i, :] - X[j, :]||_2^2.
    """
    inner = np.dot(X, X.T)
    norm2 = np.diag(inner)
    D = norm2[:, None] + norm2[None, :] - 2 * inner
    return D


def check_ellipsoid_collisions(positions, radii):
    """Checks for collisions between a set of ellipsoids at given positions.

    Args:
        positions (array float[n, 3]): The ellipsoid centers.
        radii (array float[3]): The radii of the axis-aligned ellipsoids.

    Returns:
        colliding (array bool[n]): True at index i if the i'th ellipsoid
            intersects any of the other ellipsoids.
    """
    scaled = positions / radii[None, :]
    dist2s = pairwise_squared_distance(scaled)
    # Do not consider 0 distance to self as a collision!
    n, _ = positions.shape
    dist2s[range(n), range(n)] = np.inf
    colliding = np.any(dist2s < 4.0, axis=1)
    return colliding
