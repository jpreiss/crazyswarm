import itertools as it

import numpy as np

from pycrazyswarm import util


def test_pdist():
    dim_vals = (1, 2, 10)
    npr = np.random.RandomState(0)
    for m, n in it.product(dim_vals, repeat=2):
        X = npr.normal(size=(m, n))
        D = util.pairwise_squared_distance(X)
        assert D.shape == (m, m)
        for i, j in it.product(range(m), repeat=2):
            actual = np.sum((X[i] - X[j])**2)
            assert np.isclose(D[i, j], actual)
