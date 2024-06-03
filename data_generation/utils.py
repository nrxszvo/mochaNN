import numpy as np


def get_local_minima_from_solutions(solutions):
    dfo = np.linalg.norm(solutions, axis=-1)
    minima = []
    mindex = []
    for s in range(dfo.shape[0]):
        s_min, s_idx = get_local_minima(dfo[s])
        minima.extend(s_min)
        mindex.extend([[s, i] for i in s_idx])
    return np.array(minima), np.array(mindex)


def get_local_minima(pts):
    """a local minimum is defined as a point whose L2 magnitude is less than that of both of its neighbors"""
    # first calculate direction change indicators for each pair of points
    dci = np.convolve(pts, [1, -1], mode="valid")
    # a positive value followed by negative value indicates a local minimum
    dci = np.sign(dci)
    # distinguish (pos, neg) pairs from (neg, pos) pairs
    indicators = np.convolve(dci, [-1, 1], mode="valid")
    # now negative values indicate local minima
    idx = (np.argwhere(indicators < 0) + 1)[:, 0]
    return pts[idx], idx


def get_local_maxima(pts):
    neg_maxima, idx = get_local_minima(-pts)
    return -neg_maxima, idx
