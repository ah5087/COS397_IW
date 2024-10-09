import numpy as np
from scipy.signal import convolve2d

def construct_2D_model_flux_rescaled(N, L, pattern=None):
    dxy = L / N
    D = 50  # in um^2/min
    f = 0.3  # in min-1

    center = np.arange(N)
    right = np.roll(center, -1)
    left = np.roll(center, 1)
    down = np.roll(center, -1)
    up = np.roll(center, 1)

    if pattern is None:
        pattern = np.zeros((N, N))

    idf, idt, iuf, iut, ilf, ilt, irf, irt = get_pattern_inds(pattern)

    def ode_system(t, y):
        y = y.reshape((N, N))
        dydt = np.zeros_like(y)

        # Center
        dydt[center[:, None], center] = (y[left[:, None], center] + y[right[:, None], center] +
                                         y[center[:, None], up] + y[center[:, None], down] -
                                         4 * y[center[:, None], center])

        # Edges
        dydt[0, center] = y[1, center] + y[0, right] + y[0, left] - 3 * y[0, center]
        dydt[-1, center] = y[-2, center] + y[-1, right] + y[-1, left] - 3 * y[-1, center]
        dydt[center, 0] = y[up, 0] + y[down, 0] + y[center, 1] - 3 * y[center, 0]
        dydt[center, -1] = y[up, -1] + y[down, -1] + y[center, -2] - 3 * y[center, -1]

        # Corners
        dydt[0, 0] = y[0, 1] + y[1, 0] - 2 * y[0, 0]
        dydt[-1, -1] = y[-1, -2] + y[-2, -1] - 2 * y[-1, -1]
        dydt[0, -1] = y[0, -2] + y[1, -1] - 2 * y[0, -1]
        dydt[-1, 0] = y[-2, 0] + y[-1, 1] - 2 * y[-1, 0]

        # Scale all
        dydt = D / dxy**2 * dydt

        # Fluxes
        dydt.flat[iuf] -= f / dxy * y.flat[iuf]
        dydt.flat[iut] += f / dxy * y.flat[iuf]
        dydt.flat[idf] -= f / dxy * y.flat[idf]
        dydt.flat[idt] += f / dxy * y.flat[idf]
        dydt.flat[ilf] -= f / dxy * y.flat[ilf]
        dydt.flat[ilt] += f / dxy * y.flat[ilf]
        dydt.flat[irf] -= f / dxy * y.flat[irf]
        dydt.flat[irt] += f / dxy * y.flat[irf]

        return dydt.flatten()

    return ode_system

def get_pattern_inds(pattern):
    sz = pattern.shape
    inds_down = np.where(convolve2d(pattern, [[1], [0], [-1]], mode='same') > 0)
    inds_up = np.where(convolve2d(pattern, [[1], [0], [-1]], mode='same') < 0)
    inds_right = np.where(convolve2d(pattern, [[1, 0, -1]], mode='same') > 0)
    inds_left = np.where(convolve2d(pattern, [[1, 0, -1]], mode='same') < 0)

    id, jd = inds_down
    iu, ju = inds_up
    ir, jr = inds_right
    il, jl = inds_left

    idt = np.ravel_multi_index(inds_down, sz)
    idf = np.ravel_multi_index((id - 1, jd), sz)
    iut = np.ravel_multi_index(inds_up, sz)
    iuf = np.ravel_multi_index((iu + 1, ju), sz)
    ilt = np.ravel_multi_index(inds_left, sz)
    ilf = np.ravel_multi_index((il, jl + 1), sz)
    irt = np.ravel_multi_index(inds_right, sz)
    irf = np.ravel_multi_index((ir, jr - 1), sz)

    return idf, idt, iuf, iut, ilf, ilt, irf, irt