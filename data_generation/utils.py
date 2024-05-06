import numpy as np


def get_local_minima(solutions):
    """a local minimum is defined as a point whose L2 magnitude is less than that of both of its neighbors"""
    dfo = np.linalg.norm(solutions, axis=2)
    minima = []
    mindex = []
    for s in range(dfo.shape[0]):
        # first calculate direction change indicators for each pair of points
        dci = np.convolve(dfo[s], [1, -1], mode="valid")
        # a positive value followed by negative value indicates a local minimum
        dci = np.sign(dci)
        # distinguish (pos, neg) pairs from (neg, pos) pairs
        indicators = np.convolve(dci, [-1, 1], mode="valid")
        # now all negative values indicate local minima
        idx = (np.argwhere(indicators < 0) + 1)[:, 0]
        minima.extend(dfo[s, idx])
        mindex.extend(idx)
    return np.array(minima), np.array(mindex)
