import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import scipy.io

def construct_2D_model(N, L, pattern):
    """
    Constructs the ODE system for cell movement simulation with flux.
    N: grid size (number of compartments)
    L: length of the grid (physical size)
    pattern: illumination pattern (binary matrix of 1s and 0s)
    """
    # Parameters
    dxy = L / N  # size of each compartment
    D = 50       # diffusion constant in um^2/min
    f = 0.3      # flux parameter in min-1

    center = np.arange(N)  # compartments
    right = np.roll(center, -1)  # right neighbor
    left = np.roll(center, 1)    # left neighbor
    down = np.roll(center, -1)   # downward neighbor
    up = np.roll(center, 1)      # upward neighbor

    # Pattern indices (boundary flux indices)
    idf, idt, iuf, iut, ilf, ilt, irf, irt = get_pattern_inds(pattern)

    def ode_system(t, y):
        y = y.reshape(N, N)
        dydt = np.zeros_like(y)

        # Center
        dydt[center[:, np.newaxis], center] = (
            y[left[:, np.newaxis], center] + y[right[:, np.newaxis], center] +
            y[center[:, np.newaxis], up] + y[center[:, np.newaxis], down] -
            4 * y[center[:, np.newaxis], center]
        )

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

        # Scaling for diffusion
        dydt *= D / dxy**2

        # Flux terms
        dydt[iuf] -= f / dxy * y[iuf]
        dydt[iut] += f / dxy * y[iuf]
        dydt[idf] -= f / dxy * y[idf]
        dydt[idt] += f / dxy * y[idf]
        dydt[ilf] -= f / dxy * y[ilf]
        dydt[ilt] += f / dxy * y[ilf]
        dydt[irf] -= f / dxy * y[irf]
        dydt[irt] += f / dxy * y[irf]

        return dydt.flatten()

    return ode_system

def get_pattern_inds(pattern):
    """
    Extract boundary indices based on the illumination pattern.
    """
    N = pattern.shape[0]
    
    inds_down  = np.argwhere(np.diff(pattern, axis=0, prepend=0) > 0)
    inds_up    = np.argwhere(np.diff(pattern, axis=0, append=0) < 0)
    inds_right = np.argwhere(np.diff(pattern, axis=1, prepend=0) > 0)
    inds_left  = np.argwhere(np.diff(pattern, axis=1, append=0) < 0)

    idf = (inds_down[:, 0], inds_down[:, 1])
    idt = (np.clip(inds_down[:, 0] + 1, 0, N-1), inds_down[:, 1])
    iuf = (inds_up[:, 0], inds_up[:, 1])
    iut = (np.clip(inds_up[:, 0] - 1, 0, N-1), inds_up[:, 1])
    ilf = (inds_left[:, 0], inds_left[:, 1])
    ilt = (inds_left[:, 0], np.clip(inds_left[:, 1] - 1, 0, N-1))
    irf = (inds_right[:, 0], inds_right[:, 1])
    irt = (inds_right[:, 0], np.clip(inds_right[:, 1] + 1, 0, N-1))

    return idf, idt, iuf, iut, ilf, ilt, irf, irt

def simulate_Fig2G():
    # Make a uniform initial pattern of cells
    ic = np.ones((101, 101))  # a 101 x 101 discretized grid of cells

    plt.imshow(ic)
    plt.title("Initial Condition")
    plt.show()

    # Construct illumination circle centered at the origin
    R = 9.35  # 500 um / 5400 um * 101 units = 9.35 unit diameter

    x = np.arange(-50, 51)
    y = np.arange(-50, 51)
    xx, yy = np.meshgrid(x, y)

    u = np.zeros_like(xx)
    u[(xx**2 + yy**2) < R**2] = 1

    plt.figure()
    plt.imshow(u)
    plt.title("Illumination Pattern")
    plt.axis('equal')
    plt.show()

    pattern = u

    # Simulate with "outgrowth" initial conditions
    N = 101
    L = 5400  # simulation width in um (6 mm = 6,000 um)
    tF = 1920  # simulation time in min (32 h = 1,920 min)

    f = construct_2D_model(N, L, pattern)

    ic = ic.reshape(N**2)
    t_span = [0, tF]

    # Solve the ODE system
    sol = solve_ivp(f, t_span, ic, method='LSODA')

    t = sol.t
    y = sol.y.T

    tv = np.linspace(0, tF, 100)
    yv = interp1d(t, y, axis=0)(tv)

    # Save the results
    np.savez('Fig2G_simulation_results.npz', L=L, N=N, tF=tF, tv=tv, yv=yv)

    # Plot the results
    xv = np.linspace(0, L, N)
    X, Y = np.meshgrid(xv, xv)
    i20 = np.argmax(tv >= 20*60)
    Z20 = yv[i20].reshape(N, N)
    i32 = np.argmax(tv >= 32*60)
    Z32 = yv[i32].reshape(N, N)
    xx = X[50, 50:] - L/2

    plt.figure()
    plt.plot(xx, Z20[50, 50:], label='20 hours')
    plt.plot(xx, Z32[50, 50:], label='32 hours')
    plt.xlabel('distance from center')
    plt.ylabel('relative cell density')
    plt.ylim(0, 2)
    plt.xlim(0, 1000)
    plt.legend()
    plt.title("Fig2G Results")
    plt.show()

def simulate_Fig6():
    # Load illumination pattern
    pattern_data = scipy.io.loadmat('illuminati_pattern.mat')
    pattern = pattern_data['pattern']

    pattern = np.flipud(pattern)
    pattern[:, :2] = 0
    pattern[:, -2:] = 0
    pattern[:2, :] = 0
    pattern[-2:, :] = 0
    pattern = pattern > 0

    # Simulate with "outgrowth" initial conditions
    N = 101
    L = 5400  # simulation width in um (5.4 mm = 5,400 um)
    tF = 2880  # simulation time in min (48 h = 2,880 min)

    f = construct_2D_model(N, L, pattern)

    ic = np.ones(N**2)  # Initial condition
    t_span = [0, tF]

    # Solve the ODE system
    sol = solve_ivp(f, t_span, ic, method='LSODA')

    t = sol.t
    y = sol.y.T

    tv = np.linspace(0, tF, 290)
    yv = interp1d(t, y, axis=0)(tv)

    # Save the results
    np.savez('Fig6_simulation_results.npz', L=L, N=N, tF=tF, tv=tv, yv=yv)

    # Plot the results
    xv = np.linspace(0, L, N)
    X, Y = np.meshgrid(xv, xv)

    fig, ax = plt.subplots()

    for i in range(len(tv)):
        Z = yv[i].reshape(N, N)
        im = ax.pcolormesh(X, Y, Z, edgecolors='none', vmin=0, vmax=1.2)
        ax.set_title(f't = {tv[i]/60:.1f} hours')
        ax.set_aspect('equal')
        plt.pause(0.1)
        if i < len(tv) - 1:
            im.remove()

    plt.colorbar(im)
    plt.title("Fig6 Results")
    plt.show()

if __name__ == "__main__":
    simulate_Fig2G()
    simulate_Fig6()